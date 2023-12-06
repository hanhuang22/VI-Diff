export PYTHONPATH=.
export OPENAI_LOGDIR="checkpoints/SYSU_128*256"
NUM_GPUS=2
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --lr_anneal_steps 200000"
mpiexec -n $NUM_GPUS python scripts/SYSU_cond_train.py \
    --data_dir ./datasets/SYSU-MM01 --dataset sysu \
    $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS