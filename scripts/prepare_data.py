"""The problems of OpenTO (labeled, NITO, test, NITO_test) and OpenTO-NL (test) in one format: shards
<split>-<first row>.parquet of 5,000 rows in LMTO_DERIVED with

  H, W, vf     domain in elements and volume fraction
  cond         supports and loads as JSON segments [x0, y0, x1, y1, a, b, n]
  design       the optimized design, np.packbits of the (H, W) material mask
  loads, bcs   the original node rows (x, y, fx, fy) and (x, y, fix_x, fix_y), for the FE solves

  python scripts/prepare_data.py [--workers 16]
"""
import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from lmto import config

SOURCES = {"labeled": "OpenTO/data/labeled", "NITO": "OpenTO/data/NITO", "test": "OpenTO/data/test",
           "NITO_test": "OpenTO/data/NITO_test", "NLtest": "OpenTO-NL/data/test"}
ROWS = 5000


def sig(v):
    return float(f"{v:.3g}")


def segments(nodes, load):
    """Runs of evenly spaced collinear nodes (x, y, a, b) with equal fixity, or force direction, as
    [x0, y0, x1, y1, a, b, n]; a load segment carries the total force."""
    nodes = np.asarray(nodes, dtype=np.float64)
    runs, cur = [], [nodes[0]]
    for r in nodes[1:]:
        q = cur[-1]
        if load:
            same = bool(np.all(np.sign(r[2:]) == np.sign(q[2:])) and np.allclose(r[2:], q[2:], rtol=0.5, atol=1e-12))
        else:
            same = bool(np.array_equal(r[2:], q[2:]))
        step = r[:2] - q[:2]
        ds = float(np.hypot(*step))
        if len(cur) == 1:
            ok = same and 0.0 < ds < 0.05
        else:
            d0 = cur[1][:2] - cur[0][:2]
            s0 = float(np.hypot(*d0))
            cross = abs(d0[0] * step[1] - d0[1] * step[0]) / (s0 * ds + 1e-15)
            dot = float(d0 @ step) / (s0 * ds + 1e-15)
            ok = same and cross < 1e-3 and dot > 0.999 and abs(ds - s0) <= 0.5 * s0 + 1e-9
        if ok:
            cur.append(r)
        else:
            runs.append(np.asarray(cur))
            cur = [r]
    runs.append(np.asarray(cur))
    out = []
    for s in runs:
        a, b = s[0], s[-1]
        attr = [sig(s[:, 2].sum()), sig(s[:, 3].sum())] if load else [int(a[2]), int(a[3])]
        out.append([sig(a[0]), sig(a[1]), sig(b[0]), sig(b[1])] + attr + [int(len(s))])
    return out


def segments_best(nodes, load):
    """The shortest of `segments` over the given, x-major and y-major node orders."""
    nodes = np.asarray(nodes, dtype=np.float64)
    cands = [segments(nodes, load)]
    for keys in ((nodes[:, 1], nodes[:, 0]), (nodes[:, 0], nodes[:, 1])):
        cands.append(segments(nodes[np.lexsort(keys)], load))
    return min(cands, key=len)


def record(row):
    design = np.array(Image.open(io.BytesIO(row["topology"]["bytes"])), dtype=bool)
    h, w = design.shape
    cond = {"problem": "min_compliance_2d", "H": h, "W": w, "volfrac": sig(row["volume fraction"]),
            "supports": segments_best(row["boundary conditions"], False), "loads": segments_best(row["loads"], True)}
    return {"H": h, "W": w, "vf": float(row["volume fraction"]), "cond": json.dumps(cond, separators=(",", ":")),
            "design": np.packbits(design.reshape(-1)).tobytes(), "loads": row["loads"],
            "bcs": row["boundary conditions"]}


def shard(args):
    split, f, s, start = args
    recs = [record(r) for r in pq.read_table(f).slice(s, ROWS).to_pylist()]
    cols = {k: [r[k] for r in recs] for k in recs[0]}
    nodes = pa.list_(pa.list_(pa.float32()))
    table = pa.table({"idx": pa.array(range(start, start + len(recs)), pa.int32()),
                      "H": pa.array(cols["H"], pa.int16()), "W": pa.array(cols["W"], pa.int16()),
                      "vf": pa.array(cols["vf"], pa.float32()), "cond": pa.array(cols["cond"], pa.string()),
                      "design": pa.array(cols["design"], pa.binary()),
                      "loads": pa.array(cols["loads"], nodes), "bcs": pa.array(cols["bcs"], nodes)})
    pq.write_table(table, f"{config.DERIVED}/{split}-{start:07d}.parquet", compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="one split of SOURCES")
    args = ap.parse_args()
    os.makedirs(config.DERIVED, exist_ok=True)
    tasks = []
    for split, src in SOURCES.items():
        if args.only not in (None, split):
            continue
        files = sorted(glob.glob(f"{config.DATA}/{src}-*.parquet"))
        assert files, f"no {src}-*.parquet under LMTO_DATA={config.DATA}"
        start = 0
        for f in files:
            n = pq.ParquetFile(f).metadata.num_rows
            for s in range(0, n, ROWS):
                tasks.append((split, f, s, start))
                start += min(ROWS, n - s)
        print(split, start, "problems")
    with Pool(args.workers) as pool:
        pool.map(shard, tasks, chunksize=1)


if __name__ == "__main__":
    main()
