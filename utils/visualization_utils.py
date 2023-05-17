import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_recon_vs_masking(
    title: str,
    x_label: str, 
    y_label: str,
    fname: str,
    xs: np.ndarray,
    recon_metrics: np.ndarray,
    masking_metrics: np.ndarray,
):
    """Plots recon/mask metrics.

    Args:
        title (str): Title for plot.
        x_label (str): Title for x-axis.
        y_label (str): Title for y-axis.
        fname (str): Relative filepath for saving file.
        xs (np.ndarray): X-ticks for plot.
        recon_metrics (np.ndarray): Metrics for reconstruction objective.
        mask_metrics (np.ndarray): Metrics for masking objective.
    """
    # Get means and stds.
    recon_metrics = np.array(recon_metrics)
    masking_metrics = np.array(masking_metrics)
    recon_means = np.mean(recon_metrics, axis=0)
    recon_stds = np.std(recon_metrics, axis=0)
    masking_means = np.mean(masking_metrics, axis=0)
    masking_stds = np.std(masking_metrics, axis=0)

    # Plot parameters.
    plt.figure(figsize=(9, 7))
    plt.rc("axes", titlesize=18, labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.rc("legend", fontsize=18)
    plt.rc("figure", titlesize=18)

    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.plot(xs, recon_means, label="Baseline", color="C0")
    plt.plot(xs, masking_means, label="Masking", color="C1")

    plt.fill_between(
        xs,
        recon_means - recon_stds,
        recon_means + recon_stds,
        facecolor="C0",
        alpha=0.3,
    )
    plt.fill_between(
        xs,
        masking_means - masking_stds,
        masking_means + masking_stds,
        facecolor="C1",
        alpha=0.3,
    )

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fname)
