import numpy as np
import torch

from sklearn.decomposition import SparseCoder
from utils.sparsity_utils import *


def init_with_points_population(A: np.ndarray, k: int, pp: int, noise_level: float = 0):
    """Returns a dictionary initialized with pp random points generated as z @ A + \epsilon.

    Args:
        A (np.ndarray): Ground truth dictionary.
        k (int): Sparsity of z.
        pp (int): Total number of points to generate.
        noise_level (float): Variance of random noise added to samples.
    """
    B = generate_code(n=pp, p=A.shape[0], k=k) @ A
    B = B + noise_level * np.random.randn(*B.shape)
    return B / np.expand_dims(np.sqrt(np.sum(B**2, axis=1)), 1)


def init_with_A_points_population(
    A: np.ndarray, k: int, pp: int, noise_level: float = 0
):
    """Returns a dictionary initialized with A and then pp-A.shape[0] random points generated as z @ A.

    Args:
        A (np.ndarray): Ground truth dictionary.
        k (int): Sparsity of z.
        pp (int): Total number of points to generate.
    """
    B = init_with_points_population(A, k, pp, noise_level)
    B[: A.shape[0]] = A.copy()
    return B


def init_with_gauss(d: int, pp: int):
    """Returns a dictionary initialized with normalized gaussian columns.

    Args:
        d (int): Dimension of data.
        pp (int): Number of dictionary atoms.
    """
    B = np.random.randn(pp, d)
    return B / np.expand_dims(np.sqrt(np.sum(B**2, axis=1)), 1)


def fit_sparse_code(
    fitted_dict: np.ndarray,
    signals: np.ndarray,
    transform_alpha: float,
    exact_sparsity: int = None,
):
    """Fit a regular sparse code (no masking).

    Args:
        fitted_dict (torch.DoubleTensor): Estimated dictionary.
        signals (torch.DoubleTensor): Observed signals.
        transform_alpha (float): For Lasso LARS.
        exact_sparsity (int): If set to something other than None, uses LARS for code-fitting.
    """
    if isinstance(fitted_dict, torch.Tensor):
        fitted_dict = fitted_dict.detach().numpy()
    if exact_sparsity is None:
        coder = SparseCoder(
            dictionary=fitted_dict,
            transform_algorithm="lasso_lars",
            transform_alpha=transform_alpha,
        )
    else:
        coder = SparseCoder(
            dictionary=fitted_dict,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=exact_sparsity,
        )
    return coder.fit_transform(signals)


def recon_loss(
    fitted_dict: torch.DoubleTensor,
    signals: torch.DoubleTensor,
    transform_alpha: float,
    exact_sparsity: int = None,
):
    """Reconstruction objective.

    Args:
        fitted_dict (torch.DoubleTensor): Estimated dictionary.
        signals (torch.DoubleTensor): Observed signals.
        transform_alpha (float): For Lasso LARS.
        exact_sparsity (int): If set to something other than None, uses LARS for code-fitting.
    """
    z_hat = torch.DoubleTensor(
        fit_sparse_code(fitted_dict, signals, transform_alpha, exact_sparsity)
    )
    y_hat = z_hat @ fitted_dict
    return ((signals - y_hat) ** 2).sum(dim=1).mean()


def fit_masked_sparse_code(
    fitted_dict: torch.DoubleTensor,
    signals: torch.DoubleTensor,
    mask_size: int,
    transform_alpha: float,
    exact_sparsity: int = None,
    custom_mask: np.ndarray = None,
):
    """Fits sparse code to first mask_size dimensions of dictionary elements and signal.

    Args:
        fitted_dict (torch.DoubleTensor): Estimated dictionary.
        signals (torch.DoubleTensor): Observed signals.
        mask_size (int): Mask size.
        transform_alpha (float): For Lasso LARS.
        exact_sparsity (int): If set to something other than None, uses LARS for code-fitting.
        custom_mask (np.ndarray): List of indices. If provided, uses this instead of first mask_size dims.
    """
    if custom_mask is None:
        custom_mask = np.arange(mask_size)
    if exact_sparsity is None:
        coder = SparseCoder(
            dictionary=fitted_dict.detach().numpy()[:, custom_mask],
            transform_algorithm="lasso_lars",
            transform_alpha=transform_alpha,
        )
    else:
        coder = SparseCoder(
            dictionary=fitted_dict.detach().numpy()[:, custom_mask],
            transform_algorithm="omp",
            transform_n_nonzero_coefs=exact_sparsity,
        )
    return torch.DoubleTensor(coder.fit_transform(signals[:, custom_mask]))


def masking_loss(
    fitted_dict: torch.DoubleTensor,
    signals: torch.DoubleTensor,
    mask_size: int,
    transform_alpha: float,
    exact_sparsity: int = None,
    custom_mask: np.ndarray = None,
    custom_mask_comp: np.ndarray = None,
):
    """Masking objective.

    Args:
        fitted_dict (torch.DoubleTensor): Estimated dictionary.
        signals (torch.DoubleTensor): Observed signals.
        mask_size (int): Mask size.
        transform_alpha (float): For Lasso LARS.
        exact_sparsity (int): If set to something other than None, uses OMP for code-fitting.
        custom_mask (np.ndarray): List of indices to use for mask.
        custom_mask_comp (np.ndarray): Complement of custom_mask.
    """
    if custom_mask is None or custom_mask_comp is None:
        inds = np.arange(fitted_dict.shape[1])
        np.random.shuffle(inds)
        custom_mask = inds[:mask_size]
        custom_mask_comp = inds[mask_size:]

    z_hat = fit_masked_sparse_code(
        fitted_dict,
        signals,
        mask_size,
        transform_alpha,
        exact_sparsity,
        custom_mask=custom_mask,
    )
    y_hat = z_hat @ fitted_dict[:, custom_mask_comp]
    return ((signals[:, custom_mask_comp] - y_hat) ** 2).sum(dim=1).mean()


def exhaustive_masking_loss(
    fitted_dict: torch.DoubleTensor,
    signals: torch.DoubleTensor,
    mask_size: int,
    transform_alpha: float,
    exact_sparsity: int = None,
):
    """Masking objective, but ensures that loss is computed over all dimensions.

    Args:
        fitted_dict (torch.DoubleTensor): Estimated dictionary.
        signals (torch.DoubleTensor): Observed signals.
        mask_size (int): Mask size. Assumes d - mask_size divides d, otherwise get unequal masks.
        transform_alpha (float): For Lasso LARS.
        exact_sparsity (int): If set to something other than None, uses OMP for code-fitting.
    """
    inds = np.arange(fitted_dict.shape[1])
    mask_comp_size = fitted_dict.shape[1] - mask_size
    loss = 0
    for _ in range(0, fitted_dict.shape[1], mask_comp_size):
        z_hat = fit_masked_sparse_code(
            fitted_dict,
            signals,
            mask_size,
            transform_alpha,
            exact_sparsity,
            custom_mask=inds[:mask_size],
        )
        y_hat = z_hat @ fitted_dict[:, inds[mask_size:]]
        loss += ((signals[:, inds[mask_size:]] - y_hat) ** 2).sum(dim=1).mean()
        np.roll(inds, mask_comp_size)
    return loss


def train_dict_fixed_dataset(
    B: torch.tensor,
    A: np.ndarray,
    noise_level: float,
    dataset_size: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    n_iter: int,
    mask_size: int,
    sparsity: int,
    debug: bool = False,
):
    """Trains B matrix on a fixed dataset.

    Args:
        B (torch.tensor): Dictionary to estimate.
        A (torch.tensor): Ground truth dictionary.
        noise_level (float): Noise added to samples.
        dataset_size (int): Total size of generated dataset.
        batch_size (int): Number of samples to process at each iteration.
        optimizer (torch.optim.Optimizer): Optimizer.
        n_iter (int): Number of training epochs.
        mask_size (int): Mask size. If set to <= 0, does reconstruction.
        sparsity (int): Code sparsity
        debug (bool): Whether to print per-epoch losses or not.
    """
    # Generate dataset first.
    Z = generate_code(n=dataset_size, p=A.shape[0], k=sparsity)
    Y = Z @ A
    Y = torch.DoubleTensor(Y + noise_level * np.random.randn(*Y.shape))

    # Set up data loader.
    dataset = torch.utils.data.TensorDataset(Y, Y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    d = A.shape[1]
    indices = np.arange(d)
    for epoch in range(1, n_iter + 1):
        avg_loss, avg_grad_norm = 0, 0
        for _, target in data_loader:
            optimizer.zero_grad()
            if mask_size > 0:
                np.random.shuffle(indices)
                loss = masking_loss(
                    B,
                    target,
                    mask_size,
                    transform_alpha=0,
                    exact_sparsity=sparsity,
                    custom_mask=indices[:mask_size],
                    custom_mask_comp=indices[mask_size:],
                )
            else:
                loss = recon_loss(B, target, transform_alpha=0, exact_sparsity=sparsity)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(data_loader)
            avg_grad_norm += B.grad.data.norm(2).item() / len(data_loader)

            # Project B matrix to have unit norm rows.
            with torch.no_grad():
                B.data = B.data / torch.linalg.norm(B.data, dim=1).unsqueeze(dim=1)

        if debug:
            print(f"[Epoch {epoch}] \t Loss: {avg_loss} \t Grad Norm: {avg_grad_norm}")

    return B


def train_dict_population(
    B: torch.tensor,
    A: np.ndarray,
    noise_level: float,
    dataset_size: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    n_iter: int,
    mask_size: int,
    sparsity: int,
    debug: bool = False,
):
    """Trains B matrix by generating new samples from ground truth at every iteration.

    Args:
        B (torch.tensor): Dictionary to estimate.
        A (torch.tensor): Ground truth dictionary.
        noise_level (float): Noise added to samples.
        dataset_size (int): Not used in population training, kept to keep signature the same.
        batch_size (int): Number of samples to generate at each iteration.
        optimizer (torch.optim.Optimizer): Optimizer.
        n_iter (int): Number of training iterations.
        mask_size (int): Mask size. If set to <= 0, does reconstruction.
        sparsity (int): Code sparsity
        debug (bool): Whether to print per-epoch losses or not.
    """
    d = A.shape[1]
    indices = np.arange(d)  # To use for mask.
    for epoch in range(1, n_iter + 1):
        # Generate samples for this iteration.
        z = generate_code(n=batch_size, p=A.shape[0], k=sparsity)
        y = z @ A
        y = y + noise_level * np.random.randn(*y.shape)

        # Compute loss and do optimization.
        optimizer.zero_grad()
        if mask_size > 0:
            np.random.shuffle(indices)
            loss = masking_loss(
                B,
                torch.DoubleTensor(y),
                mask_size,
                transform_alpha=0,
                exact_sparsity=sparsity,
                custom_mask=indices[:mask_size],
                custom_mask_comp=indices[mask_size:]
            )
            # loss = exhaustive_masking_loss(
            #     B,
            #     torch.DoubleTensor(y),
            #     mask_size,
            #     transform_alpha=0,
            #     exact_sparsity=sparsity,
            # )
        else:
            loss = recon_loss(
                B, torch.DoubleTensor(y), transform_alpha=0, exact_sparsity=sparsity
            )
        loss.backward()
        optimizer.step()

        # Project B matrix to have unit norm rows.
        with torch.no_grad():
            B.data = B.data / torch.linalg.norm(B.data, dim=1).unsqueeze(dim=1)

        if debug:
            print(
                f"[Epoch {epoch}] \t Loss: {loss.item()} \t Grad Norm: {B.grad.data.norm(2).item()}"
            )

    return B
