"""python -m lmto.train --config generator-e1 [--n 1000]

One configuration of lmto.config: a generator epoch on the reference designs (binary cross-entropy, inputs flipped
with probability `corrupt`), or the corrector on the compliance and the volume of its own output for one draft of the
fourth generator epoch per training problem, with FE solves on CPU workers. Writes results/<config>/weights.pt; a
stopped run resumes from results/<config>/ckpt.pt."""
import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from lmto import config
from lmto.data import Problems
from lmto.model import DEV, block_mask, build, prefix_embed


class TokenBudget(Sampler):
    """Shuffled batches of problems of similar length under a budget of padded positions."""

    def __init__(self, lengths, budget, seed=0, max_bs=32):
        lengths = np.asarray(lengths)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(lengths))
        self.batches = []
        for s in range(0, len(order), 256):
            chunk = order[s:s + 256]
            chunk = chunk[np.argsort(lengths[chunk])]
            cur, mx = [], 0
            for i in chunk:
                mx2 = max(mx, lengths[i])
                if cur and (mx2 * (len(cur) + 1) > budget or len(cur) >= max_bs):
                    self.batches.append(cur)
                    cur, mx2 = [], lengths[i]
                cur.append(int(i))
                mx = mx2
            if cur:
                self.batches.append(cur)
        rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def length(row, corrector=False):
    """Sequence positions of a problem for the batches: header, image patches, draft blocks, point tokens, blocks."""
    n = -(-row["H"] // 4) * -(-row["W"] // 4)
    cond = json.loads(row["cond"])
    return 30 + n // 4 + (n if corrector else 0) + len(cond["supports"]) + len(cond["loads"]) + n + 8


def assemble(model, batch, corrector, corrupt=0.0):
    """Embeddings, mask and positions of a batch; the generator reads its targets (bits flipped with probability
    `corrupt`), the corrector one query per block."""
    seqs, prefix, n = [], [], []
    for it in batch:
        pre = prefix_embed(model, it)
        u = torch.from_numpy(it["units"]).to(DEV)
        pe = model.unit_embed(torch.from_numpy(it["rc"]).to(DEV), torch.from_numpy(it["forced"]).to(DEV))
        if corrector:
            x = model.query[None].expand(len(u), -1) + pe
        else:
            if corrupt > 0:
                u = torch.where(torch.rand_like(u) < corrupt, 1 - u, u)
            x = torch.cat([model.start[None], model.unit_in(u[:-1])], 0) + pe
        seqs.append(torch.cat([pre, x.to(pre.dtype)], 0))
        prefix.append(len(pre))
        n.append(len(u))
    L = max(len(s) for s in seqs)
    emb = torch.zeros(len(seqs), L, model.d, device=DEV, dtype=seqs[0].dtype)
    for b, s in enumerate(seqs):
        emb[b, :len(s)] = s
    mask = block_mask(prefix, n, L, corrector, emb.dtype, DEV)
    return emb, mask, torch.arange(L, device=DEV)[None].expand(len(seqs), -1), prefix, n


def to_map(x, rc, shape):
    m = np.zeros((rc[:, 0].max() * 4 + 4, rc[:, 1].max() * 4 + 4))
    for (r, c), v in zip(rc, x):
        m[r * 4:r * 4 + 4, c * 4:c * 4 + 4] = v.reshape(4, 4)
    return m[:shape[0], :shape[1]]


def to_units(m, rc):
    full = np.zeros((rc[:, 0].max() * 4 + 4, rc[:, 1].max() * 4 + 4))
    full[:m.shape[0], :m.shape[1]] = m
    return np.stack([full[r * 4:r * 4 + 4, c * 4:c * 4 + 4].reshape(-1) for r, c in rc])


def filtered(rho, it, r=1.5):
    """The density filter of the FE model, weights (r - distance) / r normalized inside the domain, on the (H, W) map
    of a corrector output whose blocks are in raster order."""
    k = int(math.ceil(r))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    w = torch.from_numpy(np.maximum(0.0, (r - np.hypot(x, y)) / r)).to(rho.device, rho.dtype)[None, None]
    rows, cols = int(it["rc"][:, 0].max()) + 1, int(it["rc"][:, 1].max()) + 1
    h, wd = it["shape"]
    m = rho.reshape(rows, cols, 4, 4).permute(0, 2, 1, 3).reshape(1, 1, rows * 4, cols * 4)[..., :h, :wd]
    return (F.conv2d(m, w, padding=k) / F.conv2d(torch.ones_like(m), w, padding=k))[0, 0]


def corrector_loss(batch, rhos, physics, cfg):
    """vol_weight (mean rho - vf)^2 + gray_weight mean f (1 - f) + <rho, g - mean g> per problem, with f the filtered
    density and g = grad log c of the SIMP compliance at rho, held fixed: material moves to where it lowers the
    compliance most at fixed volume, and the scale of g does not depend on c. Also the mean log c and the mean of
    mean rho - vf over the batch."""
    maps = [to_map(r.detach().cpu().numpy(), it["rc"], it["shape"]) for it, r in zip(batch, rhos)]
    res = physics(batch, maps)
    term, dv = 0.0, []
    for it, rho, (c, g) in zip(batch, rhos, res):
        inside = torch.from_numpy(to_units(np.ones(it["shape"]), it["rc"])).to(DEV).float()
        mean = (rho * inside).sum() / inside.sum()
        term = term + cfg["vol_weight"] * (mean - it["vf"]) ** 2
        f = filtered(rho, it)
        term = term + cfg["gray_weight"] * (f * (1 - f)).mean()
        if np.isfinite(c):
            g = torch.from_numpy(to_units(g, it["rc"])).to(DEV).float()
            g = (g - (g * inside).sum() / inside.sum()) * inside
            term = term + (rho * g).sum()
        dv.append(mean.item() - it["vf"])
    logc = [np.log(c) for c, _ in res if np.isfinite(c)]
    return term / len(batch), {"logc": float(np.mean(logc)) if logc else float("nan"), "dv": float(np.mean(dv))}


def losses(model, h, batch, prefix, n, cfg, physics):
    hs = [h[b, prefix[b]:prefix[b] + n[b]].float() for b in range(len(batch))]
    if not cfg.get("corrector"):
        l = [F.binary_cross_entropy_with_logits(model.head(x), torch.from_numpy(it["units"]).to(DEV),
                                                reduction="none").mean(-1) for x, it in zip(hs, batch)]
        return torch.cat(l).mean(), None
    rhos = [(torch.from_numpy(it["base"]).to(DEV) + torch.tanh(model.head(x))).clamp(0.0, 1.0)
            for x, it in zip(hs, batch)]
    rhos = [torch.where(torch.from_numpy(it["fixed_units"]).to(DEV) > 0.5, 1.0, r) for r, it in zip(rhos, batch)]
    return corrector_loss(batch, rhos, physics, cfg)


def train(name, n=None):
    cfg = config.CONFIGS[name]
    corrector = cfg.get("corrector", False)
    out = f"{config.RESULTS}/{name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    model, tok = build(cfg)
    if cfg["init"]:
        model.load_state_dict(torch.load(f"{config.RESULTS}/{cfg['init']}/weights.pt", weights_only=True), strict=False)
    drafts, physics = None, None
    if corrector:
        with torch.no_grad():                                # the correction starts as the identity
            model.unit_out.weight.zero_()
            model.unit_out.bias.zero_()
        with open(f"{config.RESULTS}/{cfg['init']}/train_k1.pkl", "rb") as f:
            drafts = pickle.load(f)
    data = Problems(tok, "train", n, drafts, fe=corrector)
    if corrector:
        from lmto.fe import Physics
        physics = Physics(cfg["workers"])
    sampler = TokenBudget([length(r, corrector) for r in data.rows], cfg["budget"], seed=cfg["seed"])
    total = len(sampler)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=lambda b: b, num_workers=4, persistent_workers=True,
                        prefetch_factor=4)
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base = [p for k, p in model.named_parameters() if p.requires_grad and k.startswith("base.")]
    new = [p for k, p in model.named_parameters() if p.requires_grad and not k.startswith("base.")]
    opt = torch.optim.AdamW([{"params": base, "lr": cfg["lr"]}, {"params": new, "lr": cfg["lr_new"]}],
                            weight_decay=0.0, betas=(0.9, 0.95))
    warm = min(200, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    keep = lambda k: not k.startswith("base.") or "lora_" in k
    ckpt = f"{out}/ckpt.pt"
    log, t0, step = [], time.time(), 0
    if os.path.exists(ckpt):                                 # resume a stopped run
        ck = torch.load(ckpt, weights_only=True, map_location=DEV)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, log, t0 = ck["step"], ck["log"], time.time() - ck["sec"]
        sampler.batches = sampler.batches[step:]
        del ck
    model.train()
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb, mask, pos, prefix, n_units = assemble(model, batch, corrector, cfg.get("corrupt", 0.0))
            h = model.backbone(emb, mask, pos)
        loss, stats = losses(model, h, batch, prefix, n_units, cfg, physics)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base + new, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 250 == 0:
            torch.save({"model": {k: v for k, v in model.state_dict().items() if keep(k)}, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "log": log, "sec": time.time() - t0}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
        if step % 25 == 0 or step == total:
            log.append({"step": step, "loss": loss.item(), **(stats or {}), "sec": time.time() - t0})
            print("{} step {}/{} loss {:.4f}{} ({:.0f}s)".format(
                name, step, total, loss.item(), "" if stats is None else
                " log c {logc:.3f} dv {dv:+.4f}".format(**stats), time.time() - t0), flush=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if keep(k)}, f"{out}/weights.pt")
    with open(f"{out}/train.json", "w") as f:
        json.dump({"cfg": cfg, "n": len(data), "steps": step, "sec": time.time() - t0, "log": log}, f, indent=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=None, help="the first n training problems only (a quick test)")
    args = ap.parse_args()
    train(args.config, args.n)
