"""python -m lmto.evaluate SAMPLES.npz [--workers 16]

Scores the designs of a samples file (lmto.sample, lmto.correct) by the protocol of the OpenTO papers: the compliance
error CE against the reference design and the volume fraction error VFE of every design, both computed by FE on the
binary designs. A design fails when CE >= 100 % or when its FE solve fails. The report gives, for the first design
and for the best of k (lowest CE), the failure rate, the median CE of the designs that do not fail, the median CE of
all designs and the mean |VFE|. Written to SAMPLES_scores.json."""
import argparse
import json
from multiprocessing import Pool

import numpy as np

from lmto.data import load


def score(args):
    """CE and VFE (%) of the k designs of one problem; a design whose FE solve fails gets CE = inf."""
    row, ds = args
    from lmto.fe import Evaluator
    errors = []

    def safe(f, *a):
        try:
            return f(*a)
        except Exception as e:                               # CHOLMOD fails, or the reference compliance is zero
            errors.append(f"{type(e).__name__}: {e}")
            return None

    ev = safe(Evaluator, row)
    res = [(ev and safe(ev.evaluate, d.astype(np.float64))) or (np.inf, np.nan) for d in ds]
    return {"ce": [float(c) for c, _ in res], "vfe": [float(v) for _, v in res], "errors": errors}


def report(rows):
    ce, vfe = np.array([r["ce"] for r in rows]), np.array([r["vfe"] for r in rows])
    best = np.argmin(ce, 1)
    out = {}
    for name, j in (("first", np.zeros(len(ce), dtype=int)), ("best", best)):
        c, v = ce[np.arange(len(ce)), j], vfe[np.arange(len(ce)), j]
        ok = c < 100
        out[name] = {"n": len(c), "fail_%": float(100 * (~ok).mean()), "median_CE_%": float(np.median(c[ok])),
                     "median_CE_all_%": float(np.median(c)), "mean_|VFE|_%": float(np.nanmean(np.abs(v)))}
    out["fe_errors"] = int(sum(len(r["errors"]) for r in rows))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    z = np.load(args.path)
    keys = list(zip([str(s) for s in z["split"]], z["idx"].tolist()))
    rows = {}
    for split in sorted({s for s, _ in keys}):
        k, r = load(split, fe=True)
        rows.update(zip(k, r))
    with Pool(args.workers) as pool:
        res = pool.map(score, [(rows[q], z[f"d{i}"]) for i, q in enumerate(keys)], chunksize=1)
    rep = report(res)
    with open(args.path.replace(".npz", "_scores.json"), "w") as f:
        json.dump({"report": rep, "rows": res}, f, indent=0)
    print(json.dumps({k: (round(v["fail_%"], 2), round(v["median_CE_%"], 2), round(v["mean_|VFE|_%"], 2))
                      for k, v in rep.items() if k != "fe_errors"}))
