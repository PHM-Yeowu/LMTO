import gc
import multiprocessing as mp
import os
import sys

import numpy as np

from lmto import config
from lmto.data import design

sys.path.insert(0, config.PYEDGE)


def problem(loads, bcs, vf, shape):
    """Minimum compliance on the (H, W) element grid as in OpenTO: SIMP p = 3, density filter of radius 1.5."""
    from pyEDGE.CPU import (CHOLMOD, FiniteElement, MinimumCompliance, StructuredFilter2D, StructuredMesh2D,
                            StructuredStiffnessKernel)
    h, w = shape
    loads, bcs = np.asarray(loads, dtype=np.float64), np.asarray(bcs, dtype=np.float64)
    mesh = StructuredMesh2D(h, w, h / max(h, w), w / max(h, w))
    kernel = StructuredStiffnessKernel(mesh)
    fe = FiniteElement(mesh, kernel, CHOLMOD(kernel))
    fe.reset_dirichlet_boundary_conditions()
    fe.reset_forces()
    fe.add_dirichlet_boundary_condition(positions=bcs[:, 0:2], dofs=bcs[:, 2:])
    fe.add_point_forces(positions=loads[:, 0:2], forces=loads[:, 2:])
    return MinimumCompliance(fe, StructuredFilter2D(mesh, 1.5), void=1e-6, volume_fraction=[float(vf)],
                             heavyside=False)


class Evaluator:
    """Compliance error CE and volume fraction error VFE (signed, %) of a binary design against the reference design,
    as in the RunComplianceTest of OAT."""

    def __init__(self, row):
        self.gt = design(row)
        self.shape, self.vf = self.gt.shape, float(row["vf"])
        self.problem = problem(row["loads"], row["bcs"], self.vf, self.shape)
        self.problem.desvars = self.gt.reshape(-1) > 0
        self.c_gt = float(self.problem.FEA()["compliance"])

    def evaluate(self, d):
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        self.problem.desvars = d
        c = float(self.problem.FEA()["compliance"])
        vf = float((d > 0.5).sum() / self.problem.N())
        return (c - self.c_gt) / self.c_gt * 100.0, (vf - self.vf) / self.vf * 100.0


_CACHE = {}


def solve(args):
    """(c, grad log c) of a density map; (nan, 0) when CHOLMOD fails, as on a few degenerate OpenTO problems."""
    key, fe, rho = args
    try:
        prob = _CACHE.get(key)
        if prob is None:
            if len(_CACHE) >= 8:                             # a 256 x 256 model with its factor is about 50 MB
                del _CACHE[next(iter(_CACHE))]
                gc.collect()
            prob = _CACHE[key] = problem(fe["loads"], fe["bcs"], fe["vf"], rho.shape)
        prob.set_desvars(np.clip(rho, 0.0, 1.0).reshape(-1).astype(np.float64))
        c = float(prob.f())
        return c, np.asarray(prob.nabla_f(), dtype=np.float64).reshape(rho.shape) / max(c, 1e-300)
    except Exception:
        return float("nan"), np.zeros(rho.shape)


class Physics:
    """CPU workers that return the SIMP compliance c of each density map and the gradient of log c, whose scale does
    not depend on c (a disconnected design has c ~ 1e6 times that of a connected one)."""

    def __init__(self, workers):
        env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")}
        os.environ.update(OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        self.pool = mp.get_context("spawn").Pool(workers)
        for k, v in env.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v

    def __call__(self, items, rhos):
        return self.pool.map(solve, [(it["key"], it["fe"], r) for it, r in zip(items, rhos)], chunksize=1)
