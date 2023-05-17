import numpy as np


def generate_dictionary(d: int, p: int):
    """Generates p x d dictionary.

    Args:
        d (int): Dimension of data.
        p (int): Number of atoms.
    """
    A = np.random.randn(p, d)
    return A / np.expand_dims(np.sqrt(np.sum(A**2, axis=1)), 1)


def generate_code(n: int, p: int, k: int):
    """Generates n x p sparse codes.

    Args:
        n (int): Number of code samples.
        p (int): Dimension of individual code.
        k (int): Sparsity.
    """
    z = np.zeros((n, p))
    for i in range(n):
        idx = np.arange(p)
        np.random.shuffle(idx)
        idx = idx[:k]
        z[i, idx] = np.random.randn(k)
    return z / np.expand_dims(np.sqrt(np.sum(z**2, axis=1)), 1)