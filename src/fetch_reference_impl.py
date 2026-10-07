"""Fetch the third-party SAE classes needed to load SAEBench checkpoints.

These classes come from saprmarks/dictionary_learning and are NOT vendored into this
repository, so that this project's MIT licence covers only its own code. The script
writes them into ./dlref/ , which is gitignored.

Why not use a converter instead: SAELens ships a `dictionary_learning_1` loader that
appears to handle these checkpoints, but it maps only AutoEncoderTopK and
GatedAutoEncoder to their true architectures and lets everything else fall through to
plain ReLU. BatchTopK, JumpReLU and Matryoshka checkpoints then load with their learned
thresholds silently ignored: no error, plausible output, wrong sparsity.

Why not install the PyPI package instead: the published `dictionary-learning` release is
the original upstream and does not contain the TopK / BatchTopK / Matryoshka classes,
which live only in the SAEBench fork. It also pulls a heavy dependency that pins an older
transformers API and attempts to open a log file inside site-packages at import time.

REF is pinned to 43421f59 (2025-01-16), the last upstream commit before the files moved
under a dictionary_learning/ package directory (0ff88883, 2025-02-11) and the nearest to
the SAEBench Pythia checkpoints (date-0108). Moving REF past that commit requires
prefixing every path in FILES. Upstream is MIT-licensed; check its terms yourself before
redistributing anything fetched here.
"""
from __future__ import annotations

import os
import sys
import urllib.request

REPO = "saprmarks/dictionary_learning"
REF = "43421f5934a1476cb3f32f0b9e1b5d14b84540a1"  # simplify matryoshka loss, 2025-01-16
BASE = f"https://raw.githubusercontent.com/{REPO}/{REF}"

FILES = [
    "dictionary.py",
    "trainers/trainer.py",
    "trainers/top_k.py",
    "trainers/batch_top_k.py",
    "trainers/matryoshka_batch_top_k.py",
]

# Upstream modules import a package-level config flag. Rather than pull the whole
# training stack in to satisfy one relative import, provide the minimum.
STUB = {"config.py": "DEBUG = False\n"}

# Written next to this script, because run_sweep.py puts src/ on sys.path and imports
# dlref as a top-level package.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlref")


def main() -> None:
    os.makedirs(os.path.join(OUT, "trainers"), exist_ok=True)
    for pkg in (OUT, os.path.join(OUT, "trainers")):
        open(os.path.join(pkg, "__init__.py"), "w").close()
    for name, body in STUB.items():
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(body)

    failures = []
    for rel in FILES:
        url = f"{BASE}/{rel}"
        dst = os.path.join(OUT, rel)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001 - reported together below
            failures.append((rel, type(e).__name__, str(e)[:120]))
            continue
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"  {rel:42s} {len(body):7d} bytes")

    if failures:
        for rel, kind, msg in failures:
            print(f"  FAILED {rel}: {kind}: {msg}", file=sys.stderr)
        raise SystemExit(
            f"{len(failures)}/{len(FILES)} files failed. The upstream layout may have "
            "changed; check the paths in FILES against the repository."
        )
    print(f"\nwrote {OUT}/ — import as `from dlref.dictionary import AutoEncoder` etc.")


if __name__ == "__main__":
    main()
