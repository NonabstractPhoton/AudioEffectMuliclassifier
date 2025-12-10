#!/bin/bash
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q interactive
#SBATCH -t 07:00:00
#SBATCH -N 4
#SBATCH -G 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=atharv.thakur@rutgers.edu
#SBATCH --mail-type=begin

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=16

ml conda
conda activate fx_removal_env
# source /pscratch/sd/w/wtang99/nanoGPT/envs/venv/bin/activate

srun -N $1 --ntasks-per-node=1 --gpus-per-node=4 torchrun \
        --nnodes=$1 \
        --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        /pscratch/sd/a/atharvt/AudioEffectRemoval/audioset_tagging_cnn/pytorch/train.py \
                --working_root=/pscratch/sd/a/atharvt/AudioEffectRemoval \
                --batch_size=108 \
                --gpus_per_node=4 \
                --nodes=$1 \
                --checkpoint="lightning_logs/version_69/checkpoints/epoch=80-step=38277.ckpt"\
		--learning_rate=3e-5 \
                --indexes_hdf5_path=/pscratch/sd/a/atharvt/AudioEffectRemoval/updated_dataset/hdf5s/indexes/full_indexes.h5 \
