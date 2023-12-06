export PYTHONPATH=.
export OPENAI_LOGDIR="results/regdb"
NUM_GPUS=2
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --timestep_respacing ddim25"
MODEL_PATH="--model_path ./checkpoints/regdb/ema_0.9999_200000.pt"
DATA_PATH="--data_dir ./datasets/RegDB/trainA"
mpiexec -n $NUM_GPUS python scripts/RegDB_sample.py $DATA_PATH $MODEL_PATH $CLS_PATH $MODEL_FLAGS $DIFFUSION_FLAGS 