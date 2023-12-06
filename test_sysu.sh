export PYTHONPATH=.
export OPENAI_LOGDIR="results/SYSU_128*256"
NUM_GPUS=2
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --timestep_respacing ddim25"
MODEL_PATH="--model_path ./checkpoints/SYSU_128*256/ema_0.9999_200000.pt"
DATA_PATH="--data_dir ./datasets/SYSU-MM01/trainA"
mpiexec -n $NUM_GPUS python scripts/SYSU_sample.py $DATA_PATH $MODEL_PATH $CLS_PATH $MODEL_FLAGS $DIFFUSION_FLAGS 