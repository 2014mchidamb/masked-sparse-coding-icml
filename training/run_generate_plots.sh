#!/bin/bash

python3 training/generate_plots.py --recon-config-file=configs/A_init_noise_scaling.json --masking-config-file=configs/A_init_noise_scaling.json --x-label=noise_scaling
python3 training/generate_plots.py --recon-config-file=configs/points_init_noise_scaling.json --masking-config-file=configs/points_init_noise_scaling.json --x-label=noise_scaling

python3 training/generate_plots.py --recon-config-file=configs/A_init_d_scaling_sqrt_d_noise.json --masking-config-file=configs/A_init_d_scaling_sqrt_d_noise.json --x-label=pp
python3 training/generate_plots.py --recon-config-file=configs/points_init_d_scaling_sqrt_d_noise.json --masking-config-file=configs/points_init_d_scaling_sqrt_d_noise.json --x-label=pp

python3 training/generate_plots.py --recon-config-file=configs/A_init_fixed_d_sqrt_d_noise.json --masking-config-file=configs/A_init_fixed_d_sqrt_d_noise.json --x-label=pp
python3 training/generate_plots.py --recon-config-file=configs/points_init_fixed_d_sqrt_d_noise.json --masking-config-file=configs/points_init_fixed_d_sqrt_d_noise.json --x-label=pp