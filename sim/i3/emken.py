"""Emken-style scalar error+effort learner (non-RL sanity check) under the same four assistance conditions.

State x = learner's own compensation (skill) against a disturbance d_k.  Raw error e = d_k - x.  The robot
supplies assistance u(e) >= 0, so the residual error the learner experiences is e' = e - u.  Update
    x_{k+1} = a x_k + b e'_k - c u_k            (a<1: effort-driven forgetting; c: offloading to the robot)
A push 'falls' if the residual e' > E_CRIT (a fall is the failure of the combined human+robot system).
Assistance rules (matching the physical conditions qualitatively):
  B0  u = 0
  B1  threshold lock: u = e when e > theta_lock (a catch removes the whole excess), else 0
  B2  always-on spring: u = alpha * e (every push, proportional)
  CP  minimal-intervention: u = max(0, e - E_CRIT + margin)  (acts only when needed to avoid failure)
Retention = fresh pushes with u = 0.
"""
from __future__ import annotations

import numpy as np

A_FORGET = 0.95
B_GAIN = 0.08
C_OFFLOAD = 0.15
E_CRIT = 1.0
D_LO, D_HI = 0.4, 2.0          # disturbance range (normalised); fall-free requires x >= d - E_CRIT
THETA_LOCK = 0.5
ALPHA_SPRING = 0.5
MARGIN = 0.1


def assist(cond, e):
    if cond == "B0":
        return np.zeros_like(e)
    if cond == "B1":
        return np.where(e > THETA_LOCK, e, 0.0)
    if cond == "B2":
        return ALPHA_SPRING * np.maximum(e, 0.0)
    if cond == "CP":
        return np.maximum(0.0, e - E_CRIT + MARGIN)
    raise ValueError(cond)


def run(cond, x0, seeds, n_sess=12, per_sess=20, n_ret=200, c=C_OFFLOAD, a=A_FORGET, b=B_GAIN):
    """x0: array (n_seed,) initial skill (severity-dependent).  Returns dict of per-seed arrays."""
    x0 = np.asarray(x0, float)
    S = len(x0)
    x = x0.copy()
    falls = np.zeros(S)
    curve = [x.copy()]
    for s in range(n_sess):
        for k in range(per_sess):
            d = np.array([np.random.default_rng([sd, s, k]).uniform(D_LO, D_HI) for sd in seeds])
            e = d - x
            u = assist(cond, e)
            er = e - u
            falls += er > E_CRIT
            x = np.maximum(a * x + b * er - c * u, 0.0)   # skill cannot be negative
        curve.append(x.copy())
    rng = np.random.default_rng([int(seeds[0]), 777])
    dret = rng.uniform(D_LO, D_HI, (S, n_ret))
    ret_self = ((dret - x[:, None]) <= E_CRIT).mean(1)
    return dict(x=np.array(curve), falls=falls, retention_self=ret_self)
