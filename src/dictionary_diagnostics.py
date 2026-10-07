"""Dictionary-side superposition diagnostics that SAEBench does not provide.

SAEBench's own evals (absorption, core, sparse_probing, scr, tpp) are used as-is.
This module covers the three complements:

  * shrinkage / hedging   -- is the decoder systematically under-scaling?
  * dark matter           -- how much residual error is LINEARLY predictable from x?
  * decoder geometry      -- cosine structure, feature density, dead latents

Conventions
-----------
x       : [N, d_model]   activations (float32, already centred? NO -- raw)
z       : [N, d_sae]     latent activations (post-nonlinearity, non-negative)
x_hat   : [N, d_model]   reconstruction
W_dec   : [d_sae, d_model]

Every function takes tensors already on the target device and returns plain
Python floats / small numpy arrays, so results are cheap to serialise.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
# operating point
# --------------------------------------------------------------------------

def l0(z: torch.Tensor, eps: float = 0.0) -> float:
    """Mean number of active latents per token."""
    return (z > eps).float().sum(dim=-1).mean().item()


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained, variance taken about the dataset mean.

    This is the SAEBench convention: the denominator is the variance of x about
    its own mean, NOT ||x||^2. The two differ a lot for residual-stream
    activations, which have a large mean offset, and reporting the wrong one
    flatters every SAE equally but by different amounts.
    """
    num = (x - x_hat).pow(2).sum()
    den = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    return (num / den).item()


# --------------------------------------------------------------------------
# shrinkage / hedging
# --------------------------------------------------------------------------

def shrinkage(x: torch.Tensor, x_hat: torch.Tensor) -> dict:
    """Optimal global rescale of the reconstruction, and what it buys.

    gamma* = argmin_g ||x - g*x_hat||^2 = <x, x_hat> / ||x_hat||^2

    gamma* > 1 means the SAE is systematically under-scaling (shrinkage, the
    known consequence of an L1 penalty on the latents). gamma* ~= 1 is the
    signature you expect from TopK/JumpReLU, which have no such penalty.

    `fvu_gain` is how much of the reconstruction error was pure scale error --
    error that carries no information about which features fired. A variant
    whose FVU advantage disappears after rescaling did not decode anything
    extra; it merely had a better-calibrated norm.

    `norm_slope` is the per-token version: OLS slope of ||x_hat|| on ||x||.
    It separates a uniform scale error (slope ~= gamma*, intercept ~= 0) from a
    magnitude-dependent one (slope < 1 with positive intercept -- hedging,
    where the SAE pulls every token toward a typical norm).
    """
    dot = (x * x_hat).sum()
    gamma = (dot / x_hat.pow(2).sum()).item()

    nx = x.norm(dim=-1)
    nxh = x_hat.norm(dim=-1)
    nx_c = nx - nx.mean()
    slope = ((nx_c * (nxh - nxh.mean())).sum() / nx_c.pow(2).sum()).item()
    intercept = (nxh.mean() - slope * nx.mean()).item()

    return {
        "gamma_star": gamma,
        "fvu": fvu(x, x_hat),
        "fvu_rescaled": fvu(x, gamma * x_hat),
        "fvu_gain_from_rescale": fvu(x, x_hat) - fvu(x, gamma * x_hat),
        "norm_slope": slope,
        "norm_intercept": intercept,
        "mean_norm_ratio": (nxh / nx).mean().item(),
    }


# --------------------------------------------------------------------------
# dark matter
# --------------------------------------------------------------------------

def dark_matter(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    n_train: int | None = None,
    ridge: float = 1.0,
) -> dict:
    """How much of the residual is a LINEAR function of the input?

    Fits r = x @ B + b by ridge regression on a train split and reports
    out-of-sample R^2 on the held-out split. Held-out is the whole point: an
    unregularised in-sample fit of a [d, d] map on a few thousand tokens will
    report high R^2 for any SAE whatsoever.

    Interpretation. Residual error that a single linear map recovers from the
    input was never "features the dictionary hasn't learned yet" -- it is
    structure the linear-dictionary-plus-sparsity prior is systematically
    unable to express at this operating point. High dark-matter R^2 with low
    FVU is the interesting failure: the SAE reconstructs well and is still
    missing something simple.

    Returns the R^2 and the share of total variance it accounts for, so it can
    be compared across variants at different FVU.
    """
    n, d = x.shape
    n_train = n // 2 if n_train is None else n_train
    assert n_train < n, f"need a held-out split: n={n}, n_train={n_train}"

    r = (x - x_hat).double()
    xd = x.double()

    xtr, xte = xd[:n_train], xd[n_train:]
    rtr, rte = r[:n_train], r[n_train:]

    mu = xtr.mean(dim=0, keepdim=True)
    xtr_c, xte_c = xtr - mu, xte - mu
    rmu = rtr.mean(dim=0, keepdim=True)

    gram = xtr_c.T @ xtr_c + ridge * torch.eye(d, dtype=torch.float64, device=x.device)
    B = torch.linalg.solve(gram, xtr_c.T @ (rtr - rmu))     # [d, d]

    pred = xte_c @ B + rmu
    ss_res = (rte - pred).pow(2).sum()
    ss_tot = (rte - rtr.mean(dim=0, keepdim=True)).pow(2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()

    x_var = (xte - xtr.mean(dim=0, keepdim=True)).pow(2).sum()
    return {
        "dark_matter_r2": r2,
        "residual_share_of_variance": (rte.pow(2).sum() / x_var).item(),
        "linearly_predictable_share_of_variance": (
            ((rte.pow(2).sum() - ss_res) / x_var).item()
        ),
    }


# --------------------------------------------------------------------------
# decoder geometry + feature density
# --------------------------------------------------------------------------

def decoder_geometry(W_dec: torch.Tensor, chunk: int = 2048) -> dict:
    """Cosine structure of the decoder, computed without materialising [d_sae, d_sae].

    max_cos is the headline: a dictionary whose latents are near-duplicates of
    each other (max_cos -> 1) has split one model direction across several
    latents, which is feature splitting, not extra resolution. Reported as a
    distribution, not a mean, because splitting affects a minority of latents.
    """
    d_sae = W_dec.shape[0]
    W = torch.nn.functional.normalize(W_dec.float(), dim=-1)

    max_cos = torch.empty(d_sae, device=W.device)
    for i in range(0, d_sae, chunk):
        blk = W[i : i + chunk] @ W.T                        # [chunk, d_sae]
        idx = torch.arange(blk.shape[0], device=W.device)
        blk[idx, idx + i] = -1.0                            # drop self-similarity
        max_cos[i : i + chunk] = blk.max(dim=-1).values

    q = torch.tensor([0.5, 0.9, 0.99, 1.0], device=W.device)
    return {
        "max_cos_median": max_cos.quantile(q[0]).item(),
        "max_cos_p90": max_cos.quantile(q[1]).item(),
        "max_cos_p99": max_cos.quantile(q[2]).item(),
        "max_cos_max": max_cos.max().item(),
        "frac_max_cos_above_0.9": (max_cos > 0.9).float().mean().item(),
        "decoder_norm_mean": W_dec.float().norm(dim=-1).mean().item(),
        "decoder_norm_cv": (
            W_dec.float().norm(dim=-1).std() / W_dec.float().norm(dim=-1).mean()
        ).item(),
    }


def feature_density(z: torch.Tensor, eps: float = 0.0) -> dict:
    """Firing frequency distribution and dead/ultra-dense tails.

    Dead latents inflate nominal width without contributing capacity, so a
    width comparison that ignores them is comparing the wrong number. Ultra-
    dense latents (firing on >10% of tokens) are usually reconstructing a mean
    offset rather than a feature.
    """
    freq = (z > eps).float().mean(dim=0)                    # [d_sae]
    alive = freq > 0
    logf = torch.log10(freq[alive].clamp_min(1e-12))
    return {
        "d_sae": int(z.shape[1]),
        "frac_dead": (~alive).float().mean().item(),
        "frac_ultra_dense": (freq > 0.1).float().mean().item(),
        "log10_freq_median": logf.median().item(),
        "log10_freq_p10": logf.quantile(0.1).item(),
        "log10_freq_p90": logf.quantile(0.9).item(),
        "log10_freq_hist": np.histogram(
            logf.cpu().numpy(), bins=40, range=(-7.0, 0.0)
        )[0].tolist(),
    }


# --------------------------------------------------------------------------
# baselines -- the table is unreadable without them
# --------------------------------------------------------------------------

def pca_baseline(x: torch.Tensor, k: int, n_fit: int | None = None) -> dict:
    """Rank-k PCA reconstruction, fit on a train split and scored held-out.

    Not a sparse method and not a competitor -- it is the reference for how much
    of the activation variance k linear directions can possibly explain. An SAE
    at L0 = k that loses to this on FVU is not buying anything with its
    dictionary; note that PCA at rank k uses the SAME k directions for every
    token, so it is a genuine floor for a method that gets to choose k
    directions per token.
    """
    n = x.shape[0]
    n_fit = n // 2 if n_fit is None else n_fit
    xf, xe = x[:n_fit].double(), x[n_fit:].double()
    mu = xf.mean(dim=0, keepdim=True)
    _, _, V = torch.linalg.svd(xf - mu, full_matrices=False)
    P = V[:k]                                               # [k, d]
    rec = (xe - mu) @ P.T @ P + mu
    return {"k": k, "fvu": fvu(xe.float(), rec.float())}


def random_dict_topk_baseline(
    x: torch.Tensor, d_sae: int, k: int, seed: int = 0
) -> dict:
    """Random dictionary, TopK encoding, least-squares decode on the chosen support.

    The control for "does a LEARNED dictionary beat an arbitrary one at the same
    width and L0". Sparse coding over a random overcomplete basis is a
    surprisingly strong reconstructor, which is exactly why it belongs in the
    table: a variant that beats other SAEs but not this has learned a good
    sparse code, not interpretable features.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = x.shape[1]
    D = torch.randn(d_sae, d, generator=g).to(x.device, x.dtype)
    D = torch.nn.functional.normalize(D, dim=-1)

    proj = x @ D.T                                          # [N, d_sae]
    idx = proj.abs().topk(k, dim=-1).indices                # [N, k]

    x_hat = torch.empty_like(x)
    for i in range(x.shape[0]):
        A = D[idx[i]].T.double()                            # [d, k]
        coef = torch.linalg.lstsq(A, x[i].double().unsqueeze(-1)).solution
        x_hat[i] = (A @ coef).squeeze(-1).to(x.dtype)
    return {"d_sae": d_sae, "k": k, "fvu": fvu(x, x_hat)}


# --------------------------------------------------------------------------

def run_all(
    x: torch.Tensor,
    z: torch.Tensor,
    x_hat: torch.Tensor,
    W_dec: torch.Tensor,
) -> dict:
    """Every diagnostic in this module for one (SAE, activation buffer) pair."""
    out = {"l0": l0(z)}
    out.update(shrinkage(x, x_hat))
    out.update(dark_matter(x, x_hat))
    out.update(decoder_geometry(W_dec))
    out.update(feature_density(z))
    return out
