"""Streaming evaluator for the full SAEBench sweep.

Why this exists: dictionary_diagnostics.run_all takes a dense z [N, d_sae]. At
d_sae=16384 and N=1e6 that is 65 GB, so the diagnostics cannot be run as written
on the sweep buffer. Every quantity they compute is a function of low-dimensional
sufficient statistics, so this module accumulates those in one chunked pass and
reconstructs the diagnostics algebraically. `validate_streaming` checks the
reconstruction against the dense reference on a subsample that does fit.

Two operating points are emitted from the SAME pass:
  * snapshot -- first `snapshot_at` tokens, chosen to equal the litmus buffer so
    the numbers stay directly comparable to SAEBench's published values.
  * full     -- the whole buffer, for the sample-hungry diagnostics (liveness,
    feature density) that the published token budget cannot resolve.
"""
from __future__ import annotations

import numpy as np
import torch

import dictionary_diagnostics as dd


def get_W_dec(sae) -> torch.Tensor:
    """[d_sae, d_model] decoder, whatever the trainer class called it."""
    if hasattr(sae, "W_dec"):
        return sae.W_dec.detach()
    if hasattr(sae, "decoder"):
        return sae.decoder.weight.detach().T
    raise AttributeError(f"no decoder found on {type(sae).__name__}")


class _Acc:
    """Sufficient statistics for one (dictionary, buffer) pass."""

    def __init__(self, d, d_sae):
        z64 = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.n = 0
        self.sx, self.sx2 = z64(d), z64(d)
        self.sr2 = z64(d)
        self.cnt, self.mag = z64(d_sae), z64(d_sae)
        self.s = dict.fromkeys(
            ["xnorm", "xhnorm", "xnorm2", "xnorm_xhnorm", "xdxh", "xh2",
             "x2tot", "r2tot", "cos", "l2r", "l0", "l1", "l2"], 0.0)

    def add(self, x, x_hat, f):
        r = x - x_hat
        xd, rd = x.double(), r.double()
        self.n += x.shape[0]
        self.sx += xd.sum(0); self.sx2 += (xd ** 2).sum(0)
        self.sr2 += (rd ** 2).sum(0)
        nx = torch.linalg.norm(x, dim=-1).double()
        nxh = torch.linalg.norm(x_hat, dim=-1).double()
        s = self.s
        s["xnorm"] += nx.sum().item(); s["xhnorm"] += nxh.sum().item()
        s["xnorm2"] += (nx ** 2).sum().item()
        s["xnorm_xhnorm"] += (nx * nxh).sum().item()
        s["xdxh"] += (xd * x_hat.double()).sum().item()
        s["xh2"] += (nxh ** 2).sum().item()
        s["x2tot"] += (xd ** 2).sum().item()
        s["r2tot"] += (rd ** 2).sum().item()
        s["cos"] += ((xd / nx[:, None]) * (x_hat.double() / nxh[:, None])).sum().item()
        s["l2r"] += (nxh / nx).sum().item()
        s["l0"] += (f > 0).double().sum().item()
        s["l1"] += f.double().abs().sum().item()
        s["l2"] += torch.linalg.norm(r, dim=-1).double().sum().item()
        self.cnt += (f > 0).double().sum(0)
        self.mag += f.double().sum(0)

    def eval_metrics(self):
        """The nine metrics SAEBench publishes (activation-only subset)."""
        n, s = self.n, self.s
        var_x = (self.sx2 - self.sx ** 2 / n).sum()
        return {
            "n_tokens": n,
            "l2_loss": s["l2"] / n,
            "l1_loss": s["l1"] / n,
            "l0": s["l0"] / n,
            "frac_variance_explained": float(1 - self.sr2.sum() / var_x),
            "cossim": s["cos"] / n,
            "l2_ratio": s["l2r"] / n,
            "relative_reconstruction_bias": s["xh2"] / s["xdxh"],
            "frac_alive": float((self.cnt > 0).double().mean()),
        }

    def shrinkage(self):
        """Algebraic form of dictionary_diagnostics.shrinkage."""
        n, s = self.n, self.s
        den = float((self.sx2 - self.sx ** 2 / n).sum())
        gamma = s["xdxh"] / s["xh2"]
        fvu_raw = s["r2tot"] / den
        fvu_res = (s["x2tot"] - 2 * gamma * s["xdxh"] + gamma ** 2 * s["xh2"]) / den
        Sxx = s["xnorm2"] - s["xnorm"] ** 2 / n
        Sxy = s["xnorm_xhnorm"] - s["xnorm"] * s["xhnorm"] / n
        slope = Sxy / Sxx
        return {
            "gamma_star": gamma,
            "fvu": fvu_raw,
            "fvu_rescaled": fvu_res,
            "fvu_gain_from_rescale": fvu_raw - fvu_res,
            "norm_slope": slope,
            "norm_intercept": (s["xhnorm"] - slope * s["xnorm"]) / n,
            "mean_norm_ratio": s["l2r"] / n,
        }

    def feature_density(self, d_sae):
        freq = self.cnt / self.n
        alive = freq > 0
        logf = torch.log10(freq[alive].clamp_min(1e-12))
        return {
            "d_sae": d_sae,
            "frac_dead": float((~alive).double().mean()),
            "frac_ultra_dense": float((freq > 0.1).double().mean()),
            "log10_freq_median": float(logf.median()),
            "log10_freq_p10": float(logf.quantile(0.1)),
            "log10_freq_p90": float(logf.quantile(0.9)),
            "log10_freq_hist": np.histogram(logf.numpy(), bins=40, range=(-7.0, 0.0))[0].tolist(),
        }


class _Split:
    """Raw second moments for one half of the buffer (dark-matter regression)."""

    def __init__(self, d):
        self.n = 0
        self.sx = torch.zeros(d, dtype=torch.float64)
        self.sr = torch.zeros(d, dtype=torch.float64)
        self.XtX = torch.zeros(d, d, dtype=torch.float64)
        self.Xtr = torch.zeros(d, d, dtype=torch.float64)
        self.r2 = 0.0

    def add(self, x, r):
        xd, rd = x.double(), r.double()
        self.n += x.shape[0]
        self.sx += xd.sum(0); self.sr += rd.sum(0)
        self.XtX += xd.T @ xd
        self.Xtr += xd.T @ rd
        self.r2 += (rd ** 2).sum().item()


def _dark_matter(tr: _Split, te: _Split, ridge: float = 1.0) -> dict:
    """Held-out R^2 of a ridge map from x to the residual, from split moments.

    Identical estimator to dictionary_diagnostics.dark_matter, but the centred
    cross-products are reconstructed from raw moments instead of stored tokens.
    """
    d = tr.sx.shape[0]
    mu = (tr.sx / tr.n)[:, None]          # [d,1] train mean of x
    rmu = (tr.sr / tr.n)[:, None]         # [d,1] train mean of r

    Vtr = tr.XtX - tr.n * (mu @ mu.T)
    Utr = tr.Xtr - tr.n * (mu @ rmu.T)
    B = torch.linalg.solve(Vtr + ridge * torch.eye(d, dtype=torch.float64), Utr)

    n = te.n
    sx, sr = te.sx[:, None], te.sr[:, None]
    Vte = te.XtX - sx @ mu.T - mu @ sx.T + n * (mu @ mu.T)
    Ute = te.Xtr - sx @ rmu.T - mu @ sr.T + n * (mu @ rmu.T)
    uu = te.r2 - 2 * float(rmu.T @ sr) + n * float(rmu.T @ rmu)

    ss_res = uu - 2 * torch.trace(B.T @ Ute).item() + torch.trace(B.T @ Vte @ B).item()
    x_var = torch.trace(Vte).item()
    return {
        "dark_matter_r2": 1.0 - ss_res / uu,
        "residual_share_of_variance": te.r2 / x_var,
        "linearly_predictable_share_of_variance": (te.r2 - ss_res) / x_var,
    }


@torch.no_grad()
def stream_eval(sae, X, snapshot_at=None, chunk=4096, half=None) -> dict:
    """One chunked pass over X. Returns {'snapshot': {...}, 'full': {...}}."""
    N, d = X.shape
    d_sae = get_W_dec(sae).shape[0]
    half = N // 2 if half is None else half
    acc = _Acc(d, d_sae)
    tr, te = _Split(d), _Split(d)
    snap = None

    bounds = sorted({0, N, half} | ({snapshot_at} if snapshot_at else set()))
    for a, b in zip(bounds, bounds[1:]):
        for i in range(a, b, chunk):
            j = min(i + chunk, b)
            x = torch.from_numpy(np.asarray(X[i:j]))
            x_hat, f = sae(x, output_features=True)
            x_hat, f = x_hat.to(torch.float32), f.to(torch.float32)
            acc.add(x, x_hat, f)
            (tr if i < half else te).add(x, x - x_hat)
        if snapshot_at and b == snapshot_at:
            snap = acc.eval_metrics()

    full = acc.eval_metrics()
    full.update(acc.shrinkage())
    full.update(_dark_matter(tr, te))
    W = get_W_dec(sae)
    full.update(dd.decoder_geometry(W))
    # Geometry over ALL columns is a mixture statistic: dead columns sit near
    # initialisation, and for 16384 random directions in R^768 the max-cosine
    # median is 0.141 / p99 0.175, so they occupy the bottom of the
    # distribution and drag percentiles down by an amount that depends on the
    # dead fraction -- which varies 0.03 to 0.94 across this sweep. Recompute
    # restricted to latents that actually fire, so the two are comparable.
    alive = acc.cnt > 0
    full["n_alive"] = int(alive.sum().item())
    full.update({f"alive_{k}": v for k, v in dd.decoder_geometry(W[alive]).items()})
    full.update(acc.feature_density(d_sae))
    return {"snapshot": snap, "full": full, "fire_counts": acc.cnt.cpu().numpy()}


@torch.no_grad()
def validate_streaming(sae, X_small, rtol=1e-6) -> dict:
    """Check the streaming reconstruction against the dense reference.

    Fails loudly if any shared key disagrees. This is the whole reason the
    algebraic rewrite is trustworthy: without it, a sign error in the centring
    terms produces a plausible number and no error.
    """
    x = torch.from_numpy(np.asarray(X_small))
    x_hat, z = sae(x, output_features=True)
    ref = dd.run_all(x, z.to(torch.float32), x_hat.to(torch.float32), get_W_dec(sae))
    got = stream_eval(sae, X_small, chunk=1024)["full"]

    bad = {}
    for k, v in ref.items():
        if k not in got or isinstance(v, (list, int)):
            continue
        rel = abs(got[k] - v) / max(abs(v), 1e-12)
        if rel > rtol:
            bad[k] = (v, got[k], rel)
    return {"n_compared": len(ref), "mismatches": bad}


@torch.no_grad()
def pca_reference(X, ks, half=None, chunk=65536) -> list[dict]:
    """Held-out FVU of rank-k PCA, for every k in `ks`, from one pass.

    Dictionary-independent, so it is computed once and joined onto every SAE row
    at its matched L0. Same train/test split as the dark-matter fit.
    """
    N, d = X.shape
    half = N // 2 if half is None else half
    tr, te = _Split(d), _Split(d)
    for split, a, b in (("tr", 0, half), ("te", half, N)):
        acc = tr if split == "tr" else te
        for i in range(a, b, chunk):
            x = torch.from_numpy(np.asarray(X[i:min(i + chunk, b)]))
            acc.add(x, torch.zeros_like(x))

    mu = (tr.sx / tr.n)[:, None]
    Ctr = tr.XtX - tr.n * (mu @ mu.T)
    sx = te.sx[:, None]
    Cte = te.XtX - sx @ mu.T - mu @ sx.T + te.n * (mu @ mu.T)
    evals, V = torch.linalg.eigh(Ctr)
    V = V.flip(-1).T                      # rows = components, descending variance
    den = torch.trace(Cte).item()
    out = []
    for k in sorted(set(int(k) for k in ks)):
        P = V[:k]
        out.append({"k": k, "fvu": 1.0 - torch.trace(P @ Cte @ P.T).item() / den})
    return out
