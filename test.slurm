#!/bin/bash
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q shared_interactive
#SBATCH -t 07:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=atharv.thakur@rutgers.edu
#SBATCH --mail-type=begin

ml conda
conda activate fx_removal_env


srun python \
        /pscratch/sd/a/atharvt/AudioEffectRemoval/audioset_tagging_cnn/pytorch/test.py \
                --working_root=/pscratch/sd/a/atharvt/AudioEffectRemoval \
                --batch_size=96 \
                --checkpoint="lightning_logs/version_69/checkpoints/epoch=80-step=38277.ckpt"\
                --indexes_hdf5_path=/pscratch/sd/a/atharvt/AudioEffectRemoval/test_egset12_processed/hdf5s/singleclassindexes/full_indexes.h5 \
