import os
import argparse
import time
from mpi4py import MPI
import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from PIL import Image as im
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import RegDBFolder, RegDBModFolder
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    SYSU_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    classifier_defaults,
)


def to_numpy(sample):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.cpu().numpy()
    return sample

def pil_save_arr(arr, path, mode='RGB'):
    img = im.fromarray(arr, mode)
    img.save(path)

def save_imgs(ori, gen, file_paths, con=None, noise=None):
    """
    ori: original images
    con: contour images
    gen: generated images
    """
    ori = to_numpy(ori)
    gen = to_numpy(gen)
    if con is not None:
        con = to_numpy(con)
    if noise is not None:
        noise = to_numpy(noise)
    for i, file_path in enumerate(file_paths):
        file_name = os.path.split(file_path)[1]
        pil_save_arr(ori[i], os.path.join(logger.get_dir(), file_name.replace('.', '_real.')))
        pil_save_arr(gen[i], os.path.join(logger.get_dir(), file_name.replace('.', '_fake.')))
        if con is not None:
            pil_save_arr(con[i][:,:,0], os.path.join(logger.get_dir(), file_name.replace('.', '_cont.')), mode='L')
        if noise is not None:
            pil_save_arr(noise[i], os.path.join(logger.get_dir(), file_name.replace('.', '_noise.')))


def unscale_timestep(num_timesteps, t):
    unscaled_timestep = (t * (num_timesteps / 1000)).long()
    return unscaled_timestep

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args=args)

    logger.log("creating model...")
    model, diffusion = SYSU_create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        contour=args.contour,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())


    def cond_fn(x, t, **kwargs):
        pass

    logger.log("creating data loader...")
    dataset = RegDBModFolder(
        data_dir=args.data_dir,
        image_size=args.image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        return_path=True,
        random_crop=False,
        contour=args.contour,
        hist_match=args.hist_match,
        modality='rgb',
        class_cond=args.class_cond,
    )

    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logger.log("creating samples...")
    if args.timestep_respacing.find('ddim') != -1:
        sample_fn = diffusion.ddim_sample_loop
        logger.log('using ddim_sample_loop')
    else:
        sample_fn = diffusion.p_sample_loop
        logger.log('using p_sample_loop')
    # set cond_fn
    used_cond_fn = None
    if not (args.l_cls_id==0 and args.l_cls_mod==0 and args.l_zecon==0):
        used_cond_fn = cond_fn

    start_time = time.time()
    for i, (imgs, out_dict) in enumerate(data):
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in out_dict.items() if not isinstance(v, list)}
        imgs = imgs.to(dist_util.dev())
        bs = imgs.size(dim=0)

        if args.l_zecon != 0:
            model_kwargs['img'] = imgs

        # get latent noise
        noise = None
        if args.use_latent_noise:
            noise = diffusion.ddim_reverse_sample_loop(
                    model, 
                    imgs, 
                    clip_denoised=True,
                    skip_timesteps=args.skip_timesteps,
                    model_kwargs=model_kwargs,
                    progress=args.show_progress,
                )
        # switch labels
        if args.class_cond:
            model_kwargs['y'] = 1 - model_kwargs['y']
        sample = sample_fn(
            model,
            (bs, 3, args.image_size*2, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=used_cond_fn,
            progress=args.show_progress,
            skip_timesteps=args.skip_timesteps,
        )

        # save images
        save_imgs(imgs, sample, out_dict['path'])

        cur_time = time.time()
        logger.log(f"created {i+1}/{len(data)} batch samples, time:{int(cur_time-start_time)}s, eta:{int((cur_time-start_time)/(i+1)*(len(data)))}s")
        # break
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        data_dir="",
        batch_size=16,
        model_path="",
        cls_id_path='',
        cls_mod_path='',
        l_cls_id=0,
        l_cls_mod=0,
        l_zecon=0,
        dataset="",
        contour=True,
        skip_timesteps=0,
        use_latent_noise=False,
        show_progress=False,
        hist_match=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
