import argparse
import json
import numpy as np
import os
import pickle
import sys

sys.path.append(os.getcwd())

from pathlib import Path
from utils.visualization_utils import *

# Set up commandline arguments.
parser = argparse.ArgumentParser(description="Hyperparameters for training.")
parser.add_argument("--recon-config-file", dest="recon_config_file", default="configs/main_plot_1.json", type=str)
parser.add_argument("--masking-config-file", dest="masking_config_file", default="configs/main_plot_1.json", type=str)
parser.add_argument("--x-label", dest="x_label", default="noise_scaling", type=str)
args = parser.parse_args()

config = json.load(open(args.masking_config_file, "r"))

# Paths where our data is stored.
recon_path = f"results/recon_{args.recon_config_file.split('/')[-1].split('.')[0]}"
masking_path = f"results/masking_{args.masking_config_file.split('/')[-1].split('.')[0]}"

# Generate plots.
plots_path = f"plots/{args.masking_config_file.split('/')[-1].split('.')[0]}"
Path(plots_path).mkdir(parents=True, exist_ok=True)

title = "Ground Truth Recovery Error Comparison"
if config["noise_level"] == "0":
    title = "Ground Truth (Noiseless) Recovery Error Comparison"

x_label = args.x_label
if x_label == "d":
    x_label = "Dimension"
elif x_label == "pp":
    x_label = "p'"
elif x_label ==  "noise_scaling":
    x_label = "Noise Standard Deviation"
y_label_euclidean = "Recovery Error (Euclidean)"
y_label_cosine = "Recovery Error (Cosine)"

xs = config[args.x_label]
recon_metrics_euclidean = pickle.load(open(f"{recon_path}/col_recovery_euclidean.p", "rb"))
masking_metrics_euclidean = pickle.load(open(f"{masking_path}/col_recovery_euclidean.p", "rb"))
recon_metrics_cosine = pickle.load(open(f"{recon_path}/col_recovery_cosine.p", "rb"))
masking_metrics_cosine = pickle.load(open(f"{masking_path}/col_recovery_cosine.p", "rb"))

plot_recon_vs_masking(
    title=title,
    x_label=x_label,
    y_label=y_label_euclidean,
    fname=f"{plots_path}/col_recovery_euclidean.png",
    xs=xs,
    recon_metrics=recon_metrics_euclidean,
    masking_metrics=masking_metrics_euclidean,
)

plot_recon_vs_masking(
    title=title,
    x_label=x_label,
    y_label=y_label_cosine,
    fname=f"{plots_path}/col_recovery_cosine.png",
    xs=xs,
    recon_metrics=recon_metrics_cosine,
    masking_metrics=masking_metrics_cosine,
)
