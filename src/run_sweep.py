"""Drive the streaming evaluator across every Pythia-160M SAEBench dictionary.

Two passes on purpose:
  1. verify -- construct and load every checkpoint, discarding each. A class
     mismatch or a missing config field fails here, in minutes, rather than
     three hours into the evaluation. It also warms the HF cache so the
     evaluation pass touches no network.
  2. evaluate -- stream_eval per dictionary, appended to JSONL as each one
     finishes, so an interrupted run resumes instead of restarting.

The snapshot is taken at 204,800 tokens = SAEBench's declared 200 contexts x
1024, keeping that column comparable to the published values.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import hf_hub_download  # noqa: E402

import sweep  # noqa: E402
from dlref.dictionary import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder  # noqa: E402
from dlref.trainers.batch_top_k import BatchTopKSAE  # noqa: E402
from dlref.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE  # noqa: E402
from dlref.trainers.top_k import AutoEncoderTopK  # noqa: E402

CLS = {
    "AutoEncoder": AutoEncoder,
    "GatedAutoEncoder": GatedAutoEncoder,
    "JumpReluAutoEncoder": JumpReluAutoEncoder,
    "AutoEncoderTopK": AutoEncoderTopK,
    "BatchTopKSAE": BatchTopKSAE,
    "MatryoshkaBatchTopKSAE": MatryoshkaBatchTopKSAE,
}

SNAPSHOT_AT = 204_800
# Paths are resolved against the current working directory: run from the repository
# root. The buffer and the append log are regenerable and gitignored; the manifest is
# tracked under results/.
BUFFER = "acts_pythia160m_L8.npy"
OUT = "sweep_pythia160m.jsonl"
MANIFEST = os.path.join("results", "saebench_manifest.csv")


def build(row) -> tuple[torch.nn.Module, dict]:
    """Instantiate the right trainer class from the checkpoint's own config."""
    cfg = json.load(open(hf_hub_download(row["repo"], f"{row['unit']}/config.json")))["trainer"]
    cls = CLS[cfg["dict_class"]]
    a, d = int(cfg["activation_dim"]), int(cfg["dict_size"])
    if cls is MatryoshkaBatchTopKSAE:
        sae = cls(a, d, int(cfg["k"]), [int(g) for g in cfg["group_sizes"]])
    elif cls in (AutoEncoderTopK, BatchTopKSAE):
        sae = cls(a, d, int(cfg["k"]))
    else:
        sae = cls(a, d)
    sd = torch.load(hf_hub_download(row["repo"], f"{row['unit']}/ae.pt"),
                    map_location="cpu", weights_only=True)
    sae.load_state_dict(sd)
    sae.eval()
    return sae, cfg


def main(model: str = "pythia-160m", stop_after: int | str | None = None) -> None:
    """stop_after: halt once this many records exist in OUT, at a record
    boundary. Counts records already present, so it is a target total rather
    than a count of new work -- resuming with the same value is a no-op."""
    target = int(stop_after) if stop_after is not None else None
    man = pd.read_csv(MANIFEST)
    rows = man[man.model == model].reset_index(drop=True)

    failures = []
    for _, row in rows.iterrows():
        try:
            sae, _ = build(row)
            del sae
        except Exception as e:  # noqa: BLE001 - reported, then aborts
            failures.append((row["unit"], type(e).__name__, str(e)[:160]))
    if failures:
        for f in failures:
            print("LOAD FAILED", f, flush=True)
        raise SystemExit(f"{len(failures)}/{len(rows)} checkpoints failed to load")
    print(f"verify: all {len(rows)} checkpoints load", flush=True)

    X = np.load(BUFFER, mmap_mode="r")
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["unit"] for line in open(OUT)}
        print(f"resuming: {len(done)} already done", flush=True)

    t0 = time.time()
    n = len(done)
    with open(OUT, "a") as fh:
        for i, row in rows.iterrows():
            if row["unit"] in done:
                continue
            if target is not None and n >= target:
                print(f"stopping at {n} records as requested", flush=True)
                return
            sae, cfg = build(row)
            res = sweep.stream_eval(sae, X, snapshot_at=SNAPSHOT_AT, chunk=8192)
            os.makedirs("fire_counts", exist_ok=True)
            np.save(os.path.join("fire_counts", row["unit"].replace("/", "__") + ".npy"),
                    res["fire_counts"])
            rec = {
                "model": model,
                "unit": row["unit"],
                "arch_dir": row["arch_dir"],
                "trainer": int(row["trainer"]),
                "dict_class": cfg["dict_class"],
                "trainer_class": cfg["trainer_class"],
                "k": cfg.get("k"),
                "snapshot": res["snapshot"],
                "full": res["full"],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            del sae
            print(f"[{i + 1}/{len(rows)}] {row['unit']}  {time.time() - t0:.0f}s", flush=True)
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
