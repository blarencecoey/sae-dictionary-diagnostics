"""Reproduce SAEBench's published eval metrics locally.

Metric definitions copied from dictionary_learning/evaluation.py::evaluate so the
comparison is like-for-like. Accumulated in chunks with float64 sufficient
statistics, which is exact for variance and for all per-token means.
"""
import torch, numpy as np


@torch.no_grad()
def collect_activations(model, tok, doc_iter, layer, n_docs, ctx=1024, batch=8, log=None):
    """Residual stream after `layer`, padding masked out. Returns (N, d_model) float32."""
    store = {}
    h = model.gpt_neox.layers[layer].register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o[0] if isinstance(o, tuple) else o))
    chunks, seen = [], 0
    try:
        while seen < n_docs:
            docs = []
            while len(docs) < min(batch, n_docs - seen):
                txt = next(doc_iter)["text"]
                if txt.strip():
                    docs.append(txt)
            b = tok(docs, return_tensors="pt", max_length=ctx, padding=True, truncation=True)
            model(**b)
            mask = b["attention_mask"].bool()
            chunks.append(store["a"][mask].to(torch.float32).clone())
            seen += len(docs)
            if log and seen % (batch * 10) == 0:
                log(f"  {seen}/{n_docs} docs, {sum(c.shape[0] for c in chunks)} tokens")
    finally:
        h.remove()
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def eval_sae(sae, X, chunk=4096):
    """Chunked reimplementation of dictionary_learning.evaluation.evaluate (activation-only
    metrics; loss_recovered needs the LM and is handled separately)."""
    d = X.shape[1]
    n = 0
    sx = torch.zeros(d, dtype=torch.float64); sx2 = torch.zeros(d, dtype=torch.float64)
    sr = torch.zeros(d, dtype=torch.float64); sr2 = torch.zeros(d, dtype=torch.float64)
    acc = dict(l2=0.0, l1=0.0, l0=0.0, cos=0.0, l2r=0.0, xh2=0.0, xdxh=0.0)
    active = torch.zeros(sae.dict_size, dtype=torch.float64)

    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        x_hat, f = sae(x, output_features=True)
        x_hat = x_hat.to(torch.float32); f = f.to(torch.float32)
        r = x - x_hat
        n += x.shape[0]
        sx += x.sum(0).double(); sx2 += (x.double() ** 2).sum(0)
        sr += r.sum(0).double(); sr2 += (r.double() ** 2).sum(0)
        acc["l2"] += torch.linalg.norm(r, dim=-1).double().sum().item()
        acc["l1"] += f.norm(p=1, dim=-1).double().sum().item()
        acc["l0"] += (f != 0).float().sum(-1).double().sum().item()
        xn = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        xhn = x_hat / torch.linalg.norm(x_hat, dim=-1, keepdim=True)
        acc["cos"] += (xn * xhn).sum(-1).double().sum().item()
        acc["l2r"] += (torch.linalg.norm(x_hat, dim=-1) / torch.linalg.norm(x, dim=-1)).double().sum().item()
        acc["xh2"] += (torch.linalg.norm(x_hat, dim=-1) ** 2).double().sum().item()
        acc["xdxh"] += (x * x_hat).sum(-1).double().sum().item()
        active += f.sum(0).double()

    var_x = (sx2 - sx ** 2 / n) / (n - 1)
    var_r = (sr2 - sr ** 2 / n) / (n - 1)
    return {
        "n_tokens": n,
        "l2_loss": acc["l2"] / n,
        "l1_loss": acc["l1"] / n,
        "l0": acc["l0"] / n,
        "frac_variance_explained": float(1 - var_r.sum() / var_x.sum()),
        "cossim": acc["cos"] / n,
        "l2_ratio": acc["l2r"] / n,
        "relative_reconstruction_bias": (acc["xh2"] / n) / (acc["xdxh"] / n),
        "frac_alive": float((active != 0).double().mean()),
    }
