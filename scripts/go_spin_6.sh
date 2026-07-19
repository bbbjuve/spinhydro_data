#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -G 1

module load intelpython/2025-3.12

nvidia-smi

source activate pinn-env

srun python SpinU260701_yitp.py \
  --log_file_name training_log_SpinU_6.txt \
  --obs_model_name model_spininit.pth \
  --gamma_factor 2.0 \
  --t_max2 0.42 \
  --N_LAYER 5 \
  --ang_con 1 \
  --SWITCH_LOSS2 1 \
  --SWITCH_INIT 1 \
  --N_obs 150000
