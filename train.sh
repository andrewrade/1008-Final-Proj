#!/bin/bash
#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_2572-2024fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --bind /scratch \
    --nv \
    --overlay /scratch/ad3254/overlay-25GB-500K-01.ext3:rw \
    /scratch/ad3254/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate dl
        cd /home/ad3254/1008-Final-Proj/
        python encoder_train.py
    "