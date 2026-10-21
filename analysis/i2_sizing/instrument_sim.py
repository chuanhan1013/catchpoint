"""Arrest load cases extracted from the EXISTING CatchPoint Monte Carlo (CP m=0.03).

Non-invasive: the sim package is imported unchanged.  We wrap two module-level functions of
catchpoint.model (lip_step, dist_to) with pass-through loggers - they return exactly what the originals
return, so the simulation results are bit-identical (checked below against the unwrapped run's outcome
vector and peakCatch).

Logged per tick for all trials: COM x, v, total pelvic force Ftot, and the vector from the COM to the
nearest point of the geometric support polygon (the quantity the SPEC fall criterion uses).
Engagement tick = round(t_engage/dt).  Catch force after engagement is reconstructed from the SPEC spring-damper
Fc = -Kc (x - x_lock) - Dc v  (x_lock = x at engagement tick) and cross-checked against res['peakCatch'].

Usage:  python instrument_sim.py [tau_lock]      (default 0.28 s = SPEC)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from dataclasses import replace  # noqa: E402

from catchpoint import model as M  # noqa: E402
from catchpoint.controllers import CP  # noqa: E402
from catchpoint.montecarlo import make_population  # noqa: E402


def run_logged(tau_lock=0.28, marg=0.03):
    cfg = replace(M.Config(), tau_lock=tau_lock)
    pop = make_population()
    N = len(pop["m"])
    nsteps = int(round(cfg.T / cfg.dt))
    X = np.zeros((nsteps, N, 2), np.float32)
    V = np.zeros_like(X)
    D = np.zeros_like(X)
    state = dict(k=0, first=True)
    orig_lip, orig_dist = M.lip_step, M.dist_to

    def dist_wrap(p, h):
        if state["first"]:
            state["first"] = False
            if p.shape[0] == N:                      # the fall check: dist_to(x, hull_geo)
                D[state["k"]] = p - M.project(p, h)
        return orig_dist(p, h)

    def lip_wrap(x, v, p, F, m, om, dt):
        k = state["k"]
        X[k] = x
        V[k] = v
        state["k"] += 1
        state["first"] = True
        return orig_lip(x, v, p, F, m, om, dt)

    M.lip_step, M.dist_to = lip_wrap, dist_wrap
    try:
        res = M.simulate(pop, CP(marg), cfg)
    finally:
        M.lip_step, M.dist_to = orig_lip, orig_dist
    # reference (unwrapped) run for the bit-identical check
    ref = M.simulate(pop, CP(marg), cfg)
    assert np.array_equal(ref["outcome"], res["outcome"]) and np.array_equal(ref["peakCatch"], res["peakCatch"])
    return cfg, pop, res, X, V, D


def extract(cfg, pop, res, X, V, D):
    dt = cfg.dt
    cau = np.nonzero(res["outcome"] == M.CAUGHT)[0]
    rows = []
    for i in cau:
        ke = int(round(res["t_engage"][i] / dt))
        x0 = X[ke, i].astype(float)
        v0 = V[ke, i].astype(float)
        d = D[ke, i].astype(float)
        s0 = float(np.hypot(*d))
        speed = float(np.hypot(*v0))
        n = d / s0 if s0 > 1e-6 else v0 / max(speed, 1e-9)
        vout = float(v0 @ n)
        x = X[ke:, i].astype(float)
        v = V[ke:, i].astype(float)
        Fc = -cfg.Kc * (x - x0) - cfg.Dc * v
        Fm = np.hypot(Fc[:, 0], Fc[:, 1])
        n300 = min(len(Fm), int(0.3 / dt))
        rows.append(dict(
            trial=int(i), m=float(pop["m"][i]), z0=float(pop["z0"][i]), sev=float(pop["sev"][i]),
            t_decision=float(res["t_decision"][i]), t_engage=float(res["t_engage"][i]),
            t_force=float(res["t_force"][i]),
            speed=speed, vout=vout, s0=s0, vx=float(v0[0]), vy=float(v0[1]),
            ke=0.5 * float(pop["m"][i]) * speed ** 2,
            F_peak=float(Fm.max()), t_peak=float(np.argmax(Fm) * dt),
            F_step=float(Fm[0]), F_hold=float(Fm[-1]),
            imp_300=float(Fm[:n300].sum() * dt), imp_all=float(Fm.sum() * dt),
            excursion=float(np.hypot(*(x - x0).T).max()),
            dir_deg=float(np.degrees(np.arctan2(v0[1], v0[0]))),
        ))
    return rows


def stats(a):
    a = np.asarray(a, float)
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()),
                mean=float(a.mean()))


def main():
    tau_lock = float(sys.argv[1]) if len(sys.argv) > 1 else 0.28
    cfg, pop, res, X, V, D = run_logged(tau_lock)
    rows = extract(cfg, pop, res, X, V, D)
    pk_sim = float(res["peakCatch"][res["outcome"] == M.CAUGHT].max())
    pk_rec = max(r["F_peak"] for r in rows)
    assert abs(pk_sim - pk_rec) < 1.0, (pk_sim, pk_rec)       # reconstruction cross-check
    keys = ["speed", "vout", "s0", "ke", "F_peak", "F_step", "F_hold", "imp_300", "imp_all", "excursion", "t_peak"]
    out = dict(tau_lock=tau_lock, n_trials=int(len(res["outcome"])), n_caught=len(rows),
               pct_caught=100.0 * len(rows) / len(res["outcome"]),
               peakCatch_sim_max=pk_sim, peakCatch_reconstructed_max=pk_rec,
               stats={k: stats([r[k] for r in rows]) for k in keys},
               lead_times=dict(
                   decision_to_engage=stats([r["t_engage"] - r["t_decision"] for r in rows]),
                   push_to_decision=stats([r["t_decision"] - cfg.t_push for r in rows]),
                   push_to_engage=stats([r["t_engage"] - cfg.t_push for r in rows])))
    tag = f"{int(round(tau_lock * 1000)):03d}"
    json.dump(out, open(os.path.join(HERE, "data", f"arrest_loads_{tag}.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(HERE, "data", f"arrest_rows_{tag}.json"), "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
