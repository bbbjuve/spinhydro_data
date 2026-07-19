#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -G 1

module load intelpython/2025-3.12

nvidia-smi

source activate pinn-env

srun python SpinU260701_yitp.py \
  --log_file_name training_log_SpinU_1_woLOSS.txt \
  --obs_model_name model_orbinit_short_woLOSS.pth \
  --gamma_factor 2.0 \
  --t_max2 0.2 \
  --N_LAYER 3 \
  --ang_con 0 \
  --SWITCH_LOSS2 0 \
  --SWITCH_INIT 0 \
  --N_obs 100000
