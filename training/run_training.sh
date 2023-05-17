#!/bin/bash

# Reconstruction.
sbatch training/train_dict.py --objective=recon --train-type=fixed --config-file=configs/A_init_noise_scaling.json
sbatch training/train_dict.py --objective=recon --train-type=fixed --config-file=configs/points_init_noise_scaling.json
# sbatch training/train_dict.py --objective=recon --train-type=fixed --config-file=configs/A_init_fixed_d_low_noise.json
# sbatch training/train_dict.py --objective=recon --train-type=fixed --config-file=configs/points_init_fixed_d_low_noise.json
# sbatch training/train_dict.py --objective=recon --train-type=fixed --config-file=configs/A_init_d_scaling_sqrt_d_noise.json
# sbatch training/train_dict.py --objective=recon --train-type=fixed --config-file=configs/points_init_d_scaling_sqrt_d_noise.json

# Masking.
sbatch training/train_dict.py --objective=masking --train-type=fixed --config-file=configs/A_init_noise_scaling.json
sbatch training/train_dict.py --objective=masking --train-type=fixed --config-file=configs/points_init_noise_scaling.json
# sbatch training/train_dict.py --objective=masking --train-type=fixed --config-file=configs/A_init_fixed_d_low_noise.json
# sbatch training/train_dict.py --objective=masking --train-type=fixed --config-file=configs/points_init_fixed_d_low_noise.json
# sbatch training/train_dict.py --objective=masking --train-type=fixed --config-file=configs/A_init_d_scaling_sqrt_d_noise.json
# sbatch training/train_dict.py --objective=masking --train-type=fixed --config-file=configs/points_init_d_scaling_sqrt_d_noise.json