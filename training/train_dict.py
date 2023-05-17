#SBATCH --job-name=self_supervised
#SBATCH -t 48:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=compsci-gpu
import argparse
import json
import numpy as np
import os
import pickle
import random
import sys
import torch

sys.path.append(os.getcwd())

from pathlib import Path
from utils.analysis_utils import *
from utils.modeling_utils import *

# Set up commandline arguments.
parser = argparse.ArgumentParser(description="Hyperparameters for training.")
parser.add_argument("--config-file", dest="config_file", default="configs/test_config.json", type=str)
parser.add_argument("--objective", dest="objective", default="recon", type=str)
parser.add_argument("--train-type", dest="train_type", default="fixed", type=str)
args = parser.parse_args()

# For reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Extract training hyperparameters from config.
config = json.load(open(args.config_file, "r"))

# Train dictionaries and get results.
# Assumes JSON config is specified correctly.
col_recovery_results_euclidean = []
col_recovery_results_cosine = []
improvement_results_cosine = []
for n_run in range(config["n_runs"]):
    cur_recovery_results_euclidean = []
    cur_recovery_results_cosine = []
    cur_improvement_results_cosine = []
    for i, d in enumerate(config["d"]):
        # Set up hyperparameters.
        k = config["k"][i]
        p = config["p"][i]
        pp = config["pp"][i]
        mask_size = config["mask_size"][i]

        if config["noise_level"] == "0":
            noise_level = 0
        elif config["noise_level"] == "sqrt_d":
            noise_level = 1 / np.sqrt(d) # Reasonably large noise.
        elif config["noise_level"] == "scaling":
            noise_level = config["noise_scaling"][i]
        else:
            noise_level = 1 / d # Low noise setting.

        # Generate ground truth dictionary.
        A = generate_dictionary(d, p)  # Ground truth dictionary.

        # Initialize learned dictionary.
        if config["init_type"] == "points_init":
            B = torch.tensor(
                init_with_points_population(A, k, pp, noise_level=noise_level),
                dtype=torch.double,
                requires_grad=True,
            )
        elif config["init_type"] == "A_init":
            B = torch.tensor(
                init_with_A_points_population(A, k, pp, noise_level=noise_level),
                dtype=torch.double,
                requires_grad=True,
            )
        else:
            B = torch.tensor(init_with_gauss(d, pp), dtype=torch.double, requires_grad=True)

        train_func = train_dict_fixed_dataset
        if args.train_type == "population":
            train_func = train_dict_population

        # Run training.
        orig_B = B.detach().numpy().copy()
        optimizer = torch.optim.Adam([B], lr=config["lr"])
        if args.objective == "recon":
            B = train_func(
                B=B,
                A=A,
                noise_level=noise_level,
                dataset_size=config["dataset_size"],
                batch_size=config["batch_size"],
                optimizer=optimizer,
                n_iter=config["n_iter"],
                mask_size=0,
                sparsity=k,
                debug=False,
            )
        else:
            B = train_func(
                B=B,
                A=A,
                noise_level=noise_level,
                dataset_size=config["dataset_size"],
                batch_size=config["batch_size"],
                optimizer=optimizer,
                n_iter=config["n_iter"],
                mask_size=mask_size,
                sparsity=k,
                debug=False,
            )

        B = B.detach().numpy()
        cur_recovery_results_euclidean.append(get_mean_col_recovery(A, B, dist_type="euclidean"))
        cur_recovery_results_cosine.append(get_mean_col_recovery(A, B, dist_type="cosine"))
        cur_improvement_results_cosine.append(get_mean_col_recovery(A, orig_B, dist_type="cosine") - cur_recovery_results_cosine[-1])

    col_recovery_results_euclidean.append(cur_recovery_results_euclidean)
    col_recovery_results_cosine.append(cur_recovery_results_cosine)
    improvement_results_cosine.append(cur_improvement_results_cosine)

# Set up file details.
results_path = f"results/{args.objective}_{args.config_file.split('/')[-1].split('.')[0]}"
Path(results_path).mkdir(parents=True, exist_ok=True)

# Dump results.
pickle.dump(col_recovery_results_euclidean, open(f"{results_path}/col_recovery_euclidean.p", "wb"))
pickle.dump(col_recovery_results_cosine, open(f"{results_path}/col_recovery_cosine.p", "wb"))
pickle.dump(improvement_results_cosine, open(f"{results_path}/improvement_cosine.p", "wb"))

