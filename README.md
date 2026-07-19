# Code and Data for Generating the Manuscript Figures

## Paper

**Achieving angular-momentum conservation with physics-informed neural networks in computational relativistic spin hydrodynamics**

arXiv:2512.17971
https://arxiv.org/abs/2512.17971

## Overview

This directory contains the scripts, saved PyTorch models, training logs, and generated data files used to generate the figures in the paper.
The final figure files themselves are not included in this repository; they can be regenerated from the files provided here.

## Contents

- `SpinU260701_yitp.py`: main PyTorch/PINN training script.
- `go_spin_*.sh`: SLURM launch scripts for training runs.
- `Read_*.py`: postprocessing scripts that load saved models and generate heatmaps or data tables.
- `*.plt`: gnuplot scripts for line plots.
- `*.pth`: saved PyTorch model files.
- `*.dat` and `training_log_*.txt`: generated numerical data and training logs.

## Environment and computational resources

The Python scripts use PyTorch, NumPy, Matplotlib, SciPy, and `sobol_seq`. CUDA/GPU execution is expected for both training and postprocessing.
The calculations used one NVIDIA H100 NVL GPU. On this hardware, the typical training time for the main runs is of order 1--3 hours.

The included saved models, logs, and data files are sufficient for regenerating the plotted figures;
full retraining is needed only to reproduce the training runs themselves.

## Figure-generation map

The table below maps manuscript figures to the scripts and main inputs used to generate them.
The manuscript figure numbering follows the numbering in the paper.

| Manuscript figure | Script(s) | Main inputs | Purpose |
| --- | --- | --- | --- |
| Figure 4 | `Resi_Levo_orbinit.plt` | `training_log_SpinU_5.txt`, `Ltot_orbinit_v2.dat` | Training loss, residual measure, and total angular-momentum evolution for the long orbital-initialized run. |
| Figure 5 | `Lcomp_orbinit.plt` | `Ltot_orbinit_ideal_v2.dat`, `Ltot_orbinit_v2.dat` | Orbital/spin angular-momentum components for orbital-initialized runs. |
| Figure 6 | `Read_orbinit_ideal.py`, `Read_orbinit.py` | `model_orbinit_ideal.pth`, `model_orbinit.pth` | Heatmaps for orbital and spin angular-momentum densities. |
| Figure 8 | `Read_orbinit.py` | `model_orbinit.pth` | Heatmaps for spin-potential and vorticity-related quantities. |
| Figure 9 | `Read_orbinit_ideal.py`, `Read_orbinit.py`, `Read_spininit.py` | `model_orbinit_ideal.pth`, `model_orbinit.pth`, `model_spininit.pth` | Hydrodynamic-variable heatmaps and an angular-momentum-density panel. |
| Figure 10 | `Lcomp_spininit.plt` | `Ltot_spininit_v2.dat` | Orbital/spin angular-momentum components for the spin-initialized run. |
| Figure 11 | `Read_spininit.py` | `model_spininit.pth` | Heatmaps for orbital and spin angular-momentum densities in the spin-initialized run. |
| Figure 13 | `Read_spininit.py` | `model_spininit.pth` | Heatmaps for spin-potential and vorticity-related quantities in the spin-initialized run. |
| Figure 14 | `Read_spininit.py` | `model_spininit.pth` | Hydrodynamic-variable heatmaps for the spin-initialized run. |
| Figure 15 | `Loss_comp_orbinit.plt` | `training_log_SpinU_1.txt`, `training_log_SpinU_1_woLOSS.txt` | Comparison of the conservation-law residual with and without the penalty term. |
| Figure 16 | `Read_orbinit_short.py`, `Read_orbinit_short_woLOSS.py`, `int.plt`, `spec.plt` | `model_orbinit_short.pth`, `model_orbinit_short_woLOSS.pth`, `Rd_avg_r_*.dat`, `Rd_spectrum_*.dat` | Residual profile and spectrum comparing runs with and without the penalty term. |

## Typical workflow

1. Use the included `.pth` files unless retraining is necessary.
2. Run the appropriate `Read_*.py` script to regenerate heatmaps or `.dat` files.
3. Run the relevant gnuplot script to regenerate line plots.
4. To retrain, submit the relevant `go_spin_*.sh` script on a SLURM GPU system, for example:

```bash
sbatch go_spin_5.sh
```
The postprocessing scripts and plotting scripts assume they are run from this directory, so that relative paths to models, logs, and data files resolve correctly.

## Update log

- 2026-07-06: Updated the repository to the current code/data set associated with arXiv:2512.17971.
