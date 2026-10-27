"""python -m lmto.sample --split test [--config generator-e4] [--k 4] [--n 64]

Draws k drafts per problem and writes results/<config>/<split>_k<k>.pkl, {key: (k, H, W) density maps in float16}
for the corrector, and for a test set <split>_k<k>.npz with the designs (density above 1/2) for lmto.evaluate. The
density map holds the probabilities of the drawn blocks with the forced elements at one. The batches depend on --bs
and --tokens; the defaults are those of the paper."""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from lmto import config
from lmto.data import P, Problems, unpatches
from lmto.model import DEV, load, prefix_embed
from lmto.train import length


def lora_parts(m):
    if hasattr(m, "base_layer"):
        dt = m.base_layer.weight.dtype
        return (m.base_layer.weight, m.base_layer.bias,
                [(m.lora_A[a].weight.to(dt), m.lora_B[a].weight.to(dt), m.scaling[a]) for a in m.active_adapters])
    return m.weight, m.bias, []


def linear(x, parts):
    w, b, loras = parts
    y = torch.nn.functional.linear(x, w, b)
    for a, bb, s in loras:
        y = y + torch.nn.functional.linear(torch.nn.functional.linear(x, a), bb) * s
    return y


class StaticKV:
    """Decoding steps of the Qwen2 backbone after an HF prefill, with keys and values in preallocated buffers; the
    query heads that share a key / value head go to SDPA as that head's query rows, so nothing is concatenated or
    repeated per step (HF's DynamicCache does both, which grows with the square of the length). The prefill runs in
    micro-batches of about `tokens` positions, whose activations bound the memory."""

    def __init__(self, lm, emb, att, pos, length, tokens=16384):
        from transformers import DynamicCache
        cfg = lm.config
        self.lm, self.nh, self.nkv = lm, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        B, L = att.shape
        self.k, self.v, hs = [], [], []
        mb = max(1, tokens // L)
        for s in range(0, B, mb):
            cache = DynamicCache()
            out = lm(inputs_embeds=emb[s:s + mb], attention_mask=att[s:s + mb], position_ids=pos[s:s + mb],
                     past_key_values=cache, use_cache=True)
            hs.append(out.last_hidden_state[:, -1])
            for li, layer in enumerate(cache.layers):
                if s == 0:
                    self.k.append(torch.zeros(B, self.nkv, length, self.hd, dtype=layer.keys.dtype, device=att.device))
                    self.v.append(torch.zeros_like(self.k[-1]))
                self.k[li][s:s + mb, :, :L] = layer.keys
                self.v[li][s:s + mb, :, :L] = layer.values
            del cache, out
        self.h = torch.cat(hs, 0)
        self.valid = torch.zeros(B, length, dtype=torch.bool, device=att.device)
        self.valid[:, :L] = att.bool()
        self.t = L
        if not hasattr(lm, "lora_w"):
            lm.lora_w = [{n: lora_parts(m) for n, m in (("q", l.self_attn.q_proj), ("k", l.self_attn.k_proj),
                                                         ("v", l.self_attn.v_proj), ("o", l.self_attn.o_proj),
                                                         ("g", l.mlp.gate_proj), ("u", l.mlp.up_proj),
                                                         ("d", l.mlp.down_proj))} for l in lm.layers]

    def step(self, e, pos):
        """Final-norm hidden state (B, d) of one new token per sequence, embedding e (B, 1, d) at positions pos."""
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        lm, B, t = self.lm, e.shape[0], self.t
        cos, sin = lm.rotary_emb(e, pos[:, None])
        self.valid[:, t] = True
        mask = self.valid[:, None, None, :t + 1]
        x = e
        for li, layer in enumerate(lm.layers):
            w = lm.lora_w[li]
            h = layer.input_layernorm(x)
            q = linear(h, w["q"]).reshape(B, 1, self.nh, self.hd).transpose(1, 2)
            k = linear(h, w["k"]).reshape(B, 1, self.nkv, self.hd).transpose(1, 2)
            v = linear(h, w["v"]).reshape(B, 1, self.nkv, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.k[li][:, :, t] = k[:, :, 0]
            self.v[li][:, :, t] = v[:, :, 0]
            o = torch.nn.functional.scaled_dot_product_attention(
                q.reshape(B, self.nkv, self.nh // self.nkv, self.hd), self.k[li][:, :, :t + 1], self.v[li][:, :, :t + 1],
                attn_mask=mask).reshape(B, 1, self.nh * self.hd)
            x = x + linear(o, w["o"])
            h = layer.post_attention_layernorm(x)
            x = x + linear(torch.nn.functional.silu(linear(h, w["g"])) * linear(h, w["u"]), w["d"])
        self.t += 1
        return lm.norm(x)[:, 0]


@torch.no_grad()
def decode(model, plans, temp):
    """Blocks drawn one after another with the forced elements set, and the probabilities of the elements of every
    block, in the order of each plan."""
    B = len(plans)
    pres = [prefix_embed(model, p) for p in plans]
    ns = [len(p["rc"]) for p in plans]
    N = max(ns)
    rc = torch.zeros(B, N, 2, dtype=torch.long, device=DEV)
    forced = torch.zeros(B, N, 16, device=DEV)
    for b, p in enumerate(plans):
        rc[b, :ns[b]] = torch.from_numpy(p["rc"]).to(DEV)
        forced[b, :ns[b]] = torch.from_numpy(p["forced"]).to(DEV)
    pe = model.unit_embed(rc, forced)
    L = max(len(x) for x in pres)
    emb = torch.zeros(B, L + 1, model.d, device=DEV, dtype=pres[0].dtype)
    att = torch.zeros(B, L + 1, dtype=torch.long, device=DEV)
    for b, x in enumerate(pres):                             # left-padded prompts, then the start vector
        emb[b, L - len(x):L] = x
        att[b, L - len(x):] = 1
    emb[:, L] = (model.start + pe[:, 0]).to(emb.dtype)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    kv = StaticKV(model.base.model, emb, att, pos, L + N)
    h, last = kv.h, pos[:, -1]
    probs = []
    for i in range(N):
        pr = torch.sigmoid(model.head(h.float()) / max(temp, 1e-6))
        u = torch.maximum((torch.rand_like(pr) < pr).float(), forced[:, i])
        probs.append(pr)
        if i == N - 1:
            break
        last = last + 1
        h = kv.step((model.unit_in(u) + pe[:, i + 1]).to(emb.dtype)[:, None], last)
    pr = torch.stack(probs, 1).float().cpu().numpy()
    return [pr[b, :ns[b]] for b in range(B)]


def density(plan, blocks):
    """(H, W) map of per-block values given in the order of the plan."""
    inv = np.empty_like(plan["perm"])
    inv[plan["perm"]] = np.arange(len(inv))
    return unpatches(np.asarray(blocks)[inv], P, plan["shape"])


def batches(jobs, size, bs, tokens):
    """Consecutive jobs (sorted by length) under bs sequences and `tokens` positions."""
    starts = [0]
    while starts[-1] < len(jobs):
        a, m = starts[-1], min(bs, len(jobs) - starts[-1])
        while tokens and m > 1 and m * size(jobs[a + m - 1]) > tokens:
            m -= 1
        starts.append(a + m)
    return list(zip(starts[:-1], starts[1:]))


def save(out, path, keys):
    """The density maps to path.pkl and, for a test set (designs kept), the designs to path.npz."""
    if next(iter(out.values()))[1] is not None:
        np.savez_compressed(path + ".npz", idx=np.array([i for _, i in keys]), split=np.array([s for s, _ in keys]),
                            **{f"d{i}": out[q][1] for i, q in enumerate(keys)})
    with open(path + ".pkl.tmp", "wb") as f:
        pickle.dump({q: p for q, (p, _) in out.items()}, f)
    os.replace(path + ".pkl.tmp", path + ".pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="generator-e4")
    ap.add_argument("--split", required=True, help="train or a test set: test, NITO_test, NLtest")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=None, help="the first n problems only")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=660000, help="positions per batch; lower it on a smaller GPU")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    t0 = time.time()
    cfg, model, tok = load(args.config)
    data = Problems(tok, args.split, args.n)
    path = f"{config.RESULTS}/{args.config}/{args.split}_k{args.k}"
    out, buf = {}, {}
    if os.path.exists(path + ".partial"):                    # resume a stopped run
        with open(path + ".partial", "rb") as f:
            out = pickle.load(f)
    torch.manual_seed(args.seed)
    size = {i: length(data.rows[i]) for i in range(len(data)) if data.keys[i] not in out}
    jobs = sorted(((i, j) for i in size for j in range(args.k)), key=lambda job: size[job[0]])
    saved, test = len(out), args.split != "train"
    for c, (a, b) in enumerate(batches(jobs, lambda job: size[job[0]], args.bs, args.tokens)):
        plans = {i: data.plan(i) for i, _ in jobs[a:b]}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prs = decode(model, [plans[i] for i, _ in jobs[a:b]], args.temp)
        for (i, _), pr in zip(jobs[a:b], prs):
            p = np.maximum(density(plans[i], pr), plans[i]["fixed"])
            e = buf.setdefault(i, ([], []))
            e[0].append(p.astype(np.float16))
            if test:
                e[1].append((p > 0.5).astype(np.uint8))
            if len(e[0]) == args.k:
                e = buf.pop(i)
                out[data.keys[i]] = (np.stack(e[0]), np.stack(e[1]) if test else None)
        if len(out) - saved >= (2000 if test else 100_000):
            with open(path + ".partial.tmp", "wb") as f:
                pickle.dump(out, f)
            os.replace(path + ".partial.tmp", path + ".partial")
            saved = len(out)
        if c % 20 == 0:
            print("{} {}: {}/{} problems ({:.0f}s)".format(args.config, args.split, len(out), len(data),
                                                           time.time() - t0), flush=True)
    save(out, path, data.keys)
    if os.path.exists(path + ".partial"):
        os.remove(path + ".partial")


if __name__ == "__main__":
    main()
