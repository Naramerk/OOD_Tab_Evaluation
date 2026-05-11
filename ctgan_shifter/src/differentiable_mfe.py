# See full source at: https://github.com/Naramerk/latent-tabular-lens/blob/main/shifter/src/differentiable_mfe.py
"""
Differentiable meta-features for the CTGAN Shifter experiment.

MF list: gravity, w_lambda, p_trace, lh_trace, roy_root, sd_ratio,
         mean, sd, var, max, min, range, h_mean,
         eigenvalues, cor, cov,
         mad, t_mean, sparsity,
         can_cor,
         attr_ent, joint_ent.
"""
from __future__ import annotations

import numpy as np
import torch


# ============== STATISTICAL (differentiable) ==============

def ft_mean(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Mean per feature. (n, d) -> (d,)."""
    return N.mean(dim=dim)


def ft_var(N: torch.Tensor, ddof: int = 1, dim: int = 0) -> torch.Tensor:
    """Variance per feature."""
    return N.var(dim=dim, unbiased=(ddof == 1))


def ft_sd(N: torch.Tensor, ddof: int = 1, dim: int = 0) -> torch.Tensor:
    """Standard deviation."""
    return ft_var(N, ddof=ddof, dim=dim).sqrt()


def ft_max(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.quantile(N, 1.0, dim=dim, interpolation="linear")


def ft_min(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.quantile(N, 0.0, dim=dim, interpolation="linear")


def ft_range(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return ft_max(N, dim=dim) - ft_min(N, dim=dim)


def ft_cov(N: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """Lower triangle of covariance matrix (excluding diagonal)."""
    n, d = N.shape
    c = (N - N.mean(0)).T @ (N - N.mean(0)) / (n - ddof)
    idx = torch.tril_indices(d, d, offset=-1)
    return torch.abs(c[idx[0], idx[1]])


def ft_cor(N: torch.Tensor) -> torch.Tensor:
    """Absolute correlations (lower triangle)."""
    c = torch.corrcoef(N.T).abs()
    d = N.shape[1]
    idx = torch.tril_indices(d, d, offset=-1)
    return c[idx[0], idx[1]]


def ft_eigenvalues(N: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    n, d = N.shape
    cov = (N - N.mean(0)).T @ (N - N.mean(0)) / (n - ddof)
    return torch.linalg.eigvalsh(cov)


def ft_h_mean(N: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Harmonic mean."""
    safe = torch.clamp(N, min=epsilon)
    inv_sum = (1.0 / safe).sum(0)
    n = N.shape[0]
    return torch.tensor(n, dtype=N.dtype, device=N.device) / inv_sum


def ft_mad(N: torch.Tensor, factor: float = 1.4826, dim: int = 0) -> torch.Tensor:
    """Median Absolute Deviation."""
    med = torch.quantile(N, 0.5, dim=dim, interpolation="linear")
    dev = (N - med.unsqueeze(0)).abs()
    return torch.quantile(dev, 0.5, dim=dim, interpolation="linear") * factor


def ft_t_mean(N: torch.Tensor, pcut: float = 0.2, dim: int = 0) -> torch.Tensor:
    """Trimmed mean."""
    n = N.shape[dim]
    k = int(n * pcut)
    if k < 1:
        return N.mean(dim=dim)
    sorted_vals, _ = torch.sort(N, dim=dim)
    trimmed = sorted_vals.narrow(dim, k, n - 2 * k)
    return trimmed.mean(dim=dim)


# ============== INFO-THEORY ==============

def _soft_bin_weights_1d(x, num_bins, sigma_scale=0.5, eps=1e-8, boundaries=None):
    if boundaries is None:
        q = torch.linspace(0.0, 1.0, num_bins + 1, device=x.device, dtype=x.dtype)
        boundaries = torch.quantile(x, q, dim=0, interpolation="linear")
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    width = (boundaries[1:] - boundaries[:-1]).clamp(min=eps)
    sigma = (width * sigma_scale).clamp(min=eps)
    dist = (x.unsqueeze(1) - centers.unsqueeze(0)) ** 2
    log_w = -dist / (2 * sigma.unsqueeze(0).clamp(min=1e-10))
    return torch.softmax(log_w, dim=1)


def _entropy_soft(p, eps=1e-8, base=2.0):
    p_safe = (p + eps).clamp(max=1.0)
    return -(p_safe * torch.log(p_safe)).sum(dim=-1) / torch.log(torch.tensor(base, device=p.device, dtype=p.dtype))


def ft_attr_ent(N, num_bins=None, sigma_scale=0.5, eps=1e-8, boundaries=None):
    n, d = N.shape
    B = num_bins if num_bins is not None else max(2, int(n ** (1 / 3)))
    ent = []
    for j in range(d):
        b_j = boundaries[j] if boundaries is not None else None
        w = _soft_bin_weights_1d(N[:, j], B, sigma_scale=sigma_scale, eps=eps, boundaries=b_j)
        p = w.mean(0)
        ent.append(_entropy_soft(p.unsqueeze(0), eps=eps).squeeze(0))
    return torch.stack(ent)


def ft_joint_ent(N, y_onehot, num_bins=None, sigma_scale=0.5, eps=1e-8, boundaries=None):
    n, d = N.shape
    B = num_bins if num_bins is not None else max(2, int(n ** (1 / 3)))
    ent = []
    for j in range(d):
        b_j = boundaries[j] if boundaries is not None else None
        w = _soft_bin_weights_1d(N[:, j], B, sigma_scale=sigma_scale, eps=eps, boundaries=b_j)
        p_joint = (w.unsqueeze(2) * y_onehot.unsqueeze(1)).sum(0) / n
        ent.append(_entropy_soft(p_joint.reshape(1, -1), eps=eps).squeeze(0))
    return torch.stack(ent)


# ============== Aggregation ==============

def extract_torch_mfe(N, features=None, ddof=1, y=None,
                      cat_cols=None, attr_ent_boundaries=None):
    """Extract meta-features from numeric matrix N (n, d)."""
    if features is None:
        features = ["mean", "sd", "var", "max", "min", "range",
                    "eigenvalues", "cor", "cov", "h_mean", "mad",
                    "t_mean", "attr_ent", "joint_ent"]
    y_onehot = None
    if y is not None:
        y_np = np.asarray(y).ravel()
        n_classes = int(y_np.max()) + 1
        y_onehot = torch.nn.functional.one_hot(
            torch.tensor(y_np, dtype=torch.long, device=N.device), n_classes
        ).to(N.dtype)
    results, names = {}, []
    for f in features:
        if f == "mean":      v = ft_mean(N)
        elif f == "sd":      v = ft_sd(N, ddof=ddof)
        elif f == "var":     v = ft_var(N, ddof=ddof)
        elif f == "max":     v = ft_max(N)
        elif f == "min":     v = ft_min(N)
        elif f == "range":   v = ft_range(N)
        elif f == "cor":     v = ft_cor(N)
        elif f == "cov":     v = ft_cov(N, ddof=ddof)
        elif f == "eigenvalues": v = ft_eigenvalues(N, ddof=ddof)
        elif f == "h_mean":  v = ft_h_mean(N)
        elif f == "mad":     v = ft_mad(N)
        elif f == "t_mean":  v = ft_t_mean(N)
        elif f == "attr_ent":    v = ft_attr_ent(N, boundaries=attr_ent_boundaries)
        elif f == "joint_ent" and y_onehot is not None:
            v = ft_joint_ent(N, y_onehot, boundaries=attr_ent_boundaries)
        else:
            continue
        results[f] = v
        names.append(f)
    return results, names


def flatten_results(results):
    """Flatten results to one vector."""
    values, names = [], []
    for k, v in results.items():
        v_flat = v.flatten()
        for i in range(v_flat.shape[0]):
            values.append(v_flat[i])
            names.append(f"{k}_{i}" if v_flat.shape[0] > 1 else k)
    if values:
        return torch.stack(values), names
    return torch.tensor([], dtype=torch.float32), []
