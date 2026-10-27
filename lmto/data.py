import glob
import json

import numpy as np
import pyarrow.parquet as pq

from lmto import config

P = 4                                                    # a block of P x P elements is one output position
DIMS = {"bc": 4 * 8 * 8, "draft": P * P, "pts": 8}       # image patches, draft blocks (corrector), point tokens
COLUMNS = ["idx", "H", "W", "vf", "cond", "design"]
TESTS = ("test", "NITO_test", "NLtest")


def grid(shape, p):
    h, w = shape
    return -(-h // p), -(-w // p)


def patches(x, p):
    """(n, p * p) blocks of an (H, W) array in raster order, zero-padded to a multiple of p."""
    h, w = x.shape
    r, c = grid(x.shape, p)
    pad = np.zeros((r * p, c * p))
    pad[:h, :w] = x
    return pad.reshape(r, p, c, p).transpose(0, 2, 1, 3).reshape(-1, p * p)


def unpatches(blocks, p, shape):
    r, c = grid(shape, p)
    x = np.asarray(blocks, dtype=np.float64).reshape(r, c, p, p).transpose(0, 2, 1, 3)
    return x.reshape(r * p, c * p)[:shape[0], :shape[1]]


def expand(segs, load):
    """The nodes (x, y, a, b) of segments [x0, y0, x1, y1, a, b, n]; a load segment shares its force among its nodes."""
    nodes = []
    for x0, y0, x1, y1, a, b, n in segs:
        for t in (np.linspace(0.0, 1.0, int(n)) if n > 1 else np.zeros(1)):
            nodes.append([x0 + t * (x1 - x0), y0 + t * (y1 - y0)] + ([a / n, b / n] if load else [a, b]))
    return np.asarray(nodes, dtype=np.float64)


def node_elements(x, shape):
    """Elements touching the nodes x (n, 2); x runs along H and y along W, both divided by max(H, W)."""
    h, w = shape
    m = max(h, w)
    i = np.clip(np.rint(x[:, 0] * m).astype(int), 0, h)
    j = np.clip(np.rint(x[:, 1] * m).astype(int), 0, w)
    mask = np.zeros(shape, dtype=bool)
    for di in (-1, 0):
        for dj in (-1, 0):
            mask[np.clip(i + di, 0, h - 1), np.clip(j + dj, 0, w - 1)] = True
    return mask


def nodes(col):
    """A list<list<float>> column of node rows as one (n, 4) float32 array per problem."""
    col = col.combine_chunks()
    return np.split(col.flatten().flatten().to_numpy().reshape(-1, 4), col.offsets.to_numpy()[1:-1])


def read(split, columns, n=None):
    keys, rows = [], []
    for f in sorted(glob.glob(f"{config.DERIVED}/{split}-*.parquet")):
        if n is not None and len(rows) >= n:
            break
        t = pq.read_table(f, columns=columns)
        part = t.select([c for c in columns if c not in ("loads", "bcs")]).to_pylist()
        for c in ("loads", "bcs"):
            if c in columns:
                for r, x in zip(part, nodes(t.column(c))):
                    r[c] = x
        keys += [(split, r["idx"]) for r in part]
        rows += part
    return keys[:n], rows[:n]


def load(split, n=None, fe=False):
    """Keys (source split, row) and rows of a split: train (the labeled and the NITO problems of OpenTO) or one of
    TESTS; with fe also the node rows of the loads and the supports."""
    columns = COLUMNS + (["loads", "bcs"] if fe else [])
    if split == "train":
        k1, r1 = read("labeled", columns, n)
        k2, r2 = read("NITO", columns, None if n is None else n - len(r1))
        return k1 + k2, r1 + r2
    return read(split, columns, n)


def design(row):
    h, w = row["H"], row["W"]
    return np.unpackbits(np.frombuffer(row["design"], dtype=np.uint8))[:h * w].reshape(h, w).astype(np.float64)


def header(row):
    r, c = grid((row["H"], row["W"]), P)
    return f"<topopt> H={row['H']} W={row['W']} rows={r} cols={c} vf={row['vf']:.3f}\n<design>\n"


def bc_image(cond, shape):
    """(4, H, W): x fixed, y fixed, Fx, Fy on the elements touching each node, forces scaled by the largest."""
    img = np.zeros((4,) + tuple(shape))
    sup = expand(cond["supports"], False)
    for k in (0, 1):
        x = sup[sup[:, 2 + k] > 0][:, :2]
        if len(x):
            img[k] = node_elements(x, shape)
    lod = expand(cond["loads"], True)
    fmax = np.abs(lod[:, 2:]).max() + 1e-12
    for x, f in zip(lod[:, :2], lod[:, 2:]):
        m = node_elements(x[None], shape)
        img[2][m] += f[0] / fmax
        img[3][m] += f[1] / fmax
    return img


def point_tokens(cond, shape):
    """One token per support or load segment, (x0, y0, x1, y1, a, b, log10(n) / 3, is_load) with a, b the fixed
    directions of a support or the force of a load over the largest; placed at the block of its midpoint."""
    m = max(shape)
    r, c = grid(shape, P)
    fmax = max([abs(v) for s in cond["loads"] for v in s[4:6]] + [1e-12])
    x, rc = [], []
    for load, segs in ((0.0, cond["supports"]), (1.0, cond["loads"])):
        for x0, y0, x1, y1, a, b, n in segs:
            if load:
                a, b = a / fmax, b / fmax
            x.append([x0, y0, x1, y1, a, b, np.log10(max(n, 1)) / 3, load])
            rc.append([min(max(int((x0 + x1) / 2 * m / P), 0), r - 1), min(max(int((y0 + y1) / 2 * m / P), 0), c - 1)])
    return np.asarray(x, dtype=np.float32), np.asarray(rc, dtype=np.int64)


def image_tokens(chans, p):
    x = np.concatenate([patches(c.astype(np.float64), p) for c in chans], 1)
    r, c = grid(chans.shape[1:], p)
    return x.astype(np.float32), np.stack(np.divmod(np.arange(r * c), c), 1)


class Problems:
    """Prompts, plans and training items of one split. The generator writes the blocks that hold a forced element
    first and then the others, each group in raster order. The corrector reads the density map of draft j of every
    problem from `drafts` ({key: (k, H, W)}) and all blocks in raster order."""

    def __init__(self, tok, split, n=None, drafts=None, fe=False):
        self.tok, self.drafts, self.j = tok, drafts, 0
        self.keys, self.rows = load(split, n, fe)

    def __len__(self):
        return len(self.rows)

    def draft(self, i):
        x = self.drafts[self.keys[i]]
        return np.asarray(x[self.j % len(x)], dtype=np.float64)

    def plan(self, i):
        """Prompt and order of the blocks; "mask" holds the forced elements (supports and loads) and "fixed" those set
        to one in the output: all forced elements for the generator, those of the loads for the corrector."""
        row = self.rows[i]
        shape = (row["H"], row["W"])
        cond = json.loads(row["cond"])
        imgs = [("bc",) + image_tokens(bc_image(cond, shape), 8)]
        if self.drafts is not None:
            imgs.append(("draft",) + image_tokens(self.draft(i)[None], P))
        imgs.append(("pts",) + point_tokens(cond, shape))
        loads = node_elements(expand(cond["loads"], True)[:, :2], shape)
        mask = loads | node_elements(expand(cond["supports"], False)[:, :2], shape)
        r, c = grid(shape, P)
        forced = patches(mask, P)
        if self.drafts is None:
            hit = forced.sum(1) > 0
            perm = np.concatenate([np.flatnonzero(hit), np.flatnonzero(~hit)])
        else:
            perm = np.arange(r * c)
        return {"ids": self.tok(header(row))["input_ids"], "imgs": imgs, "mask": mask, "shape": shape,
                "fixed": mask if self.drafts is None else loads, "vf": row["vf"], "perm": perm,
                "rc": np.stack(np.divmod(np.arange(r * c), c), 1)[perm], "forced": forced[perm].astype(np.float32)}

    def __getitem__(self, i):
        it = self.plan(i)
        d = np.maximum(design(self.rows[i]), it["mask"])
        it.update(key=self.keys[i], units=patches(d, P)[it["perm"]].astype(np.float32))
        if self.drafts is not None:
            it["base"] = patches(self.draft(i), P).astype(np.float32)
            it["fixed_units"] = patches(it["fixed"], P).astype(np.float32)
            it["fe"] = {k: self.rows[i][k] for k in ("loads", "bcs", "vf")}
        return it
