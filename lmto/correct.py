"""python -m lmto.correct --split test [--config corrector] [--k 4] [--n 64]

Corrects each of the k drafts in results/generator-e4/<split>_k<k>.pkl (lmto.sample) in one forward pass,
rho = clip(rho_G + tanh(head), 0, 1) with the elements at the loads set to one, and writes
results/corrector/<split>_k<k>.pkl and .npz as lmto.sample does. No FE solve is involved."""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from lmto import config
from lmto.data import P, Problems, patches
from lmto.model import DEV, block_mask, load, prefix_embed
from lmto.sample import batches, density, save
from lmto.train import length


@torch.no_grad()
def correct(model, plans):
    B = len(plans)
    pres = [prefix_embed(model, p) for p in plans]
    ns = [len(p["rc"]) for p in plans]
    seqs = []
    for b, p in enumerate(plans):
        pe = model.unit_embed(torch.from_numpy(p["rc"]).to(DEV), torch.from_numpy(p["forced"]).to(DEV))
        seqs.append(torch.cat([pres[b], (model.query[None] + pe).to(pres[b].dtype)], 0))
    L = max(len(s) for s in seqs)
    emb = torch.zeros(B, L, model.d, device=DEV, dtype=seqs[0].dtype)
    for b, s in enumerate(seqs):
        emb[b, :len(s)] = s
    h = model.backbone(emb, block_mask([len(x) for x in pres], ns, L, True, emb.dtype, DEV),
                       torch.arange(L, device=DEV)[None].expand(B, -1))
    out = []
    for b, p in enumerate(plans):
        x = h[b, len(pres[b]):len(pres[b]) + ns[b]].float()
        d = torch.tanh(model.head(x)).float()
        out.append((torch.from_numpy(p["base"]).to(DEV) + d).clamp(0.0, 1.0).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="corrector")
    ap.add_argument("--split", required=True, help="a test set: test, NITO_test, NLtest")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=None, help="the first n problems only")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=40000)
    args = ap.parse_args()
    t0 = time.time()
    cfg, model, tok = load(args.config)
    with open(f"{config.RESULTS}/{cfg['init']}/{args.split}_k{args.k}.pkl", "rb") as f:
        drafts = pickle.load(f)
    data = Problems(tok, args.split, args.n, drafts)
    size = {i: length(data.rows[i], True) for i in range(len(data))}
    jobs = sorted(((i, j) for i in size for j in range(args.k)), key=lambda job: size[job[0]])
    out = {q: {} for q in data.keys}
    for c, (a, b) in enumerate(batches(jobs, lambda job: size[job[0]], args.bs, args.tokens)):
        plans = []
        for i, j in jobs[a:b]:
            data.j = j
            plans.append(dict(data.plan(i), base=patches(data.draft(i), P).astype(np.float32)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rhos = correct(model, plans)
        for (i, j), pl, u in zip(jobs[a:b], plans, rhos):
            p = np.maximum(density(pl, u), pl["fixed"])
            out[data.keys[i]][j] = (p.astype(np.float16), (p > 0.5).astype(np.uint8))
        if c % 20 == 0:
            print("{} {}: {}/{} drafts ({:.0f}s)".format(args.config, args.split, b, len(jobs), time.time() - t0),
                  flush=True)
    out = {q: tuple(np.stack(x) for x in zip(*[e[j] for j in sorted(e)])) for q, e in out.items()}
    path = f"{config.RESULTS}/{args.config}/{args.split}_k{args.k}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save(out, path, data.keys)


if __name__ == "__main__":
    main()
