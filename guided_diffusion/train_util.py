import copy
import functools
import os
import time
import itertools

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from .respace import SpacedDiffusion
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from .zecon_losses import zecon_loss_direct
from .gaussian_diffusion import _extract_into_tensor
from .unet import NLayerDiscriminator, GANLoss, EncoderUNetModel
from .script_util import create_gaussian_diffusion, create_classifier

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion:SpacedDiffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        l_zecon=0.0,
        l_cyc=0.0,
        l_adv=0.0,
        l_tadv=0.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.l_zecon = l_zecon
        self.l_cyc = l_cyc
        self.l_adv = l_adv
        self.l_tadv = l_tadv
        self.mseloss = th.nn.MSELoss()
        if self.l_cyc != 0:
            self.D_modal = NLayerDiscriminator(input_nc=3, n_layers=4)
            self.D_modal.to(dist_util.dev())
            self.opt_modal = AdamW(self.D_modal.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.l_adv != 0:
            self.D_ir  = NLayerDiscriminator(input_nc=3, n_layers=3).to(dist_util.dev())
            self.D_rgb = NLayerDiscriminator(input_nc=3, n_layers=3).to(dist_util.dev())
            self.optimizer_D = AdamW(itertools.chain(self.D_ir.parameters(), self.D_rgb.parameters()), lr=self.lr)
            self.ddp_D_ir   = DDP(self.D_ir,  device_ids=[dist_util.dev()], output_device=dist_util.dev(), 
                                broadcast_buffers=False, bucket_cap_mb=128, find_unused_parameters=False)
            self.ddp_D_rgb  = DDP(self.D_rgb, device_ids=[dist_util.dev()], output_device=dist_util.dev(), 
                                broadcast_buffers=False, bucket_cap_mb=128, find_unused_parameters=False)
            self.sample_diffusion = create_gaussian_diffusion(
                steps=1000, learn_sigma=True, timestep_respacing='ddim25')
            self.ganloss = GANLoss(gan_mode='lsgan').to(dist_util.dev())
        
        if self.l_tadv != 0:
            self.D_ir = create_classifier(
                image_size=128,
                classifier_use_fp16=True,
                classifier_width=128,
                classifier_depth=2,
                classifier_attention_resolutions="32,16,8",
                classifier_use_scale_shift_norm=True,
                classifier_resblock_updown=True,
                classifier_pool="attention",
                dataset='real_fake',
            ).to(dist_util.dev())
            self.D_rgb = create_classifier(
                image_size=128,
                classifier_use_fp16=True,
                classifier_width=128,
                classifier_depth=2,
                classifier_attention_resolutions="32,16,8",
                classifier_use_scale_shift_norm=True,
                classifier_resblock_updown=True,
                classifier_pool="attention",
                dataset='real_fake',
            ).to(dist_util.dev())

            self.D_ir_mp_trainer = MixedPrecisionTrainer(model=self.D_ir, use_fp16=True, initial_lg_loss_scale=16.0)
            self.ddp_D_ir = DDP(
                self.D_ir,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )

            self.D_rgb_mp_trainer = MixedPrecisionTrainer(model=self.D_rgb, use_fp16=True, initial_lg_loss_scale=16.0)
            self.ddp_D_rgb = DDP(
                self.D_rgb,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.opt_D_ir = AdamW(self.D_ir_mp_trainer.master_params, lr=3e-4, weight_decay=0.05)
            self.opt_D_rgb = AdamW(self.D_rgb_mp_trainer.master_params, lr=3e-4, weight_decay=0.05)
            self.ganloss = GANLoss(gan_mode='vanilla').to(dist_util.dev())
            

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            # self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        start_time = time.time()
        prev_time = start_time
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                cur_time = time.time()
                logger.log(f"step:{self.step+self.resume_step}/{self.lr_anneal_steps}, time:{int(cur_time-start_time)}s, eta:{int((cur_time-prev_time)/self.log_interval*(self.lr_anneal_steps-self.step-self.resume_step))}s")
                prev_time = time.time()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        tmp = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        if self.l_tadv:
            self.train_D(tmp)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            # train D
            if self.l_cyc != 0:
                self.opt_modal.zero_grad()
                self.set_requires_grad(self.D_modal, True)
                D_out = self.D_modal(micro)
                y = micro_cond['y'].float()
                y = y.reshape(-1,1,1,1).expand_as(D_out)
                loss_D = self.mseloss(D_out, y)
                loss_D.backward()
                self.opt_modal.step()
                if self.step % self.log_interval == 0 and dist.get_rank() == 0:
                    print('loss_D:', loss_D.item())

            # train diffusion model
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            if self.l_zecon != 0:
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                    l_zecon=self.l_zecon,
                )
            else:
                compute_losses = functools.partial(
                    self.diffusion.training_losses_cycle,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                    l_cyc=self.l_cyc,
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            
            if self.l_zecon != 0:
                x_t = losses['x_t']
                pred_xstart = losses['pred_xstart']
                losses.pop('x_t')
                losses.pop('pred_xstart')

            if self.step >= 50000 and self.l_zecon != 0:
                fac = _extract_into_tensor(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
                x_in = pred_xstart * fac + x_t * (1 - fac)
                y_t = self.diffusion.q_sample(micro, t)
                y_in = micro * fac + y_t * (1 - fac)

                zecon_loss = zecon_loss_direct(self.ddp_model, x_in, y_in, th.zeros_like(t), micro_cond) * self.l_zecon
                losses['zecon'] = zecon_loss
                losses['loss'] = losses['loss'] + zecon_loss
            
            if self.l_cyc != 0:
                pred_ori_xstart = losses['pred_ori_xstart']
                pred_trs_xstart = losses['pred_trs_xstart']
                pred_cyc_xstart = losses['pred_cyc_xstart']
                losses.pop('pred_ori_xstart')
                losses.pop('pred_trs_xstart')
                losses.pop('pred_cyc_xstart')
                cyc_loss = self.mseloss(pred_cyc_xstart, pred_ori_xstart) * self.l_cyc
                losses['cyc'] = cyc_loss
                losses['loss'] = losses['loss'] + cyc_loss

                self.set_requires_grad(self.D_modal, False)
                D_out_ori = self.D_modal(pred_ori_xstart)
                D_out_trs = self.D_modal(pred_trs_xstart)
                modal_loss = self.mseloss(D_out_ori, micro_cond['y'].reshape(-1,1,1,1).expand_as(D_out_ori).float()) + \
                    self.mseloss(D_out_trs, (1-micro_cond['y']).reshape(-1,1,1,1).expand_as(D_out_trs).float())
                losses['modal'] = modal_loss
                losses['loss'] = losses['loss'] + modal_loss

            if self.l_adv > 0:
                if self.step >= 170000:
                    # Diffusion Model
                    self.set_requires_grad([self.ddp_D_ir, self.ddp_D_rgb], requires_grad=False)
                    ir_kwargs = {}
                    ir_kwargs['contour'] = micro_cond['contour']
                    ir_kwargs['y'] = th.ones_like(micro_cond['y'])
                    fake_ir = self.sample_diffusion.ddim_sample_loop(
                        model=self.ddp_model, 
                        shape=micro.shape,
                        model_kwargs=ir_kwargs,
                        keep_grad=True,
                    )
                    # fake_ir = micro
                    fake_ir_D = fake_ir.detach()
                    rgb_kwargs = {}
                    rgb_kwargs['contour'] = micro_cond['contour']
                    rgb_kwargs['y'] = th.zeros_like(micro_cond['y'])
                    fake_rgb = self.sample_diffusion.ddim_sample_loop(
                        model=self.ddp_model, 
                        shape=micro.shape,
                        model_kwargs=rgb_kwargs,
                        keep_grad=True,
                    )
                    # fake_rgb = micro
                    fake_rgb_D = fake_rgb.detach()
                    loss_G_ir  = self.ganloss(self.ddp_D_ir(fake_ir), True)
                    loss_G_rgb = self.ganloss(self.ddp_D_rgb(fake_rgb), True)
                    losses['gan'] = (loss_G_ir + loss_G_rgb) * self.l_adv
                    losses['loss'] = losses['loss'] + losses['gan']

                    # Discriminator
                    self.set_requires_grad([self.ddp_D_ir, self.ddp_D_rgb], requires_grad=True)
                    self.optimizer_D.zero_grad()
                    real_ir = micro[micro_cond['y']==1]
                    # print(real_ir.shape)
                    loss_D_ir = self.ganloss(self.ddp_D_ir(real_ir), True) + \
                                self.ganloss(self.ddp_D_ir(fake_ir_D[:fake_ir_D.shape[0]//2]), False)
                    real_rgb = micro[micro_cond['y']==0]
                    # print(real_rgb.shape, micro_cond['y'])
                    loss_D_rgb = self.ganloss(self.ddp_D_rgb(real_rgb), True) + \
                                 self.ganloss(self.ddp_D_rgb(fake_rgb_D[:fake_rgb_D.shape[0]//2]), False)
                    loss_D_ir.backward()
                    loss_D_rgb.backward()
                    losses['D_ir'] = loss_D_ir
                    losses['D_rgb'] = loss_D_rgb
                    self.optimizer_D.step()

            if self.l_tadv > 0:
                micro_x_t = self.diffusion.q_sample(micro, t, noise=th.randn_like(micro))
                denoised_x_t = self.diffusion.ddim_sample(
                    self.ddp_model,
                    micro_x_t,
                    t,
                    clip_denoised=False,
                    cond_fn=None,
                    model_kwargs=micro_cond,
                )['sample']
                
                micro_x_t_D = micro_x_t.detach().clone()
                denoised_x_t_D = denoised_x_t.detach().clone()

                denoised_x_t_ir   = denoised_x_t[micro_cond['y']==1]
                denoised_x_t_rgb  = denoised_x_t[micro_cond['y']==0]
                
                t_ir = th.clamp(t[micro_cond['y']==1]-1, min=0)
                t_rgb = th.clamp(t[micro_cond['y']==0]-1, min=0)

                # self.set_requires_grad([self.ddp_D_ir, self.ddp_D_rgb], requires_grad=False)
                self.D_ir_mp_trainer.zero_grad()
                self.D_rgb_mp_trainer.zero_grad()
                losses['G_ir']   = self.ganloss(self.ddp_D_ir(denoised_x_t_ir, t_ir), True)
                losses['G_rgb']  = self.ganloss(self.ddp_D_rgb(denoised_x_t_rgb, t_rgb), True)
                losses['loss'] = losses['loss'] + losses['G_ir'] + losses['G_rgb']

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

        if self.l_tadv > 0:
            return {'y':micro_cond['y'], 't':t, 'x_t':micro_x_t_D, 'denoised_x_t': denoised_x_t_D}
        else:
            return None

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def train_D(self, d_data):
        # self.set_requires_grad([self.ddp_D_ir, self.ddp_D_rgb], requires_grad=True)
        self.D_ir_mp_trainer.zero_grad()
        self.D_rgb_mp_trainer.zero_grad()

        y = d_data['y']
        t = d_data['t']
        x_t = d_data['x_t']
        denoised_x_t = d_data['denoised_x_t']

        x_t_ir  = x_t[y==1]
        x_t_rgb = x_t[y==0]
        t_ir  = t[y==1]
        t_rgb = t[y==0]

        d_x_t_ir   = denoised_x_t[y==1]
        d_x_t_rgb  = denoised_x_t[y==0]
        d_t_ir  = th.clamp(t_ir-1, min=0)
        d_t_rgb = th.clamp(t_rgb-1, min=0)
        
        loss_ir = self.ganloss(self.ddp_D_ir(x_t_ir, t_ir), True) + \
                self.ganloss(self.ddp_D_ir(d_x_t_ir, d_t_ir), False)

        self.D_ir_mp_trainer.backward(loss_ir)
        self.D_ir_mp_trainer.optimize(self.opt_D_ir)

        loss_rgb = self.ganloss(self.ddp_D_rgb(x_t_rgb, t_rgb), True) + \
                self.ganloss(self.ddp_D_rgb(d_x_t_rgb, d_t_rgb), False)

        self.D_rgb_mp_trainer.backward(loss_rgb)
        self.D_rgb_mp_trainer.optimize(self.opt_D_rgb)

        if self.step % self.log_interval == 0:
            print(f'loss_D_ir:{loss_ir.item()}, loss_D_rgb:{loss_rgb.item()}')

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
        
        if self.l_cyc != 0 and dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"D_modal{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.D_modal.state_dict(), f)
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt_modal{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt_modal.state_dict(), f)

        if self.l_adv != 0 and dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"D_ir{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.D_ir.state_dict(), f)
            with bf.BlobFile(bf.join(get_blob_logdir(), f"D_rgb{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.D_rgb.state_dict(), f)
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt_D{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.optimizer_D.state_dict(), f)
        
        if self.l_tadv != 0 and dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"D_ir{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.D_ir_mp_trainer.master_params_to_state_dict(self.D_ir_mp_trainer.master_params), f)
            with bf.BlobFile(bf.join(get_blob_logdir(), f"D_rgb{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.D_rgb_mp_trainer.master_params_to_state_dict(self.D_rgb_mp_trainer.master_params), f)
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt_D_ir{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.opt_D_ir.state_dict(), f)
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt_D_rgb{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.opt_D_rgb.state_dict(), f)

        dist.barrier()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
