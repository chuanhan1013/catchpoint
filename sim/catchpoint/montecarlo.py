"""Monte Carlo evaluation, sweeps, AAN sessions, sizing statistics (SPEC 5.2, 5.4-5.6, 6)."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .controllers import B0, B1, B1XCoM, B2, CP
from .model import (ASSISTED, CAUGHT, FALL, SELF, Config, build_pop, sample_noise, simulate)

SEED = 5412
XTHETAS = [0.02, 0.05, 0.08, 0.12, 0.16]
THETAS = [0.03, 0.05, 0.08, 0.11, 0.14, 0.18]
KS = [100, 200, 400, 800, 1600, 3200]
MARGS = [-0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.15]
DELAYS = [0.05, 0.10, 0.15, 0.20, 0.30]


def make_population(seed=SEED, n_pat=50, n_push=40):
    rng = np.random.default_rng(seed)
    sev = rng.uniform(0, 1, n_pat)
    par = rng.integers(0, 2, n_pat)
    m = rng.uniform(60, 85, n_pat)
    z0 = rng.uniform(0.85, 1.00, n_pat)
    mag = rng.uniform(0.2, 0.9, (n_pat, n_push))
    ang = rng.uniform(0, 2 * np.pi, (n_pat, n_push))
    dv = np.stack([mag * np.cos(ang), mag * np.sin(ang)], -1).reshape(-1, 2)
    noise = sample_noise(np.random.default_rng(seed + 1), n_pat)
    rep = lambda a: np.repeat(a, n_push, axis=0)
    pop = build_pop(rep(sev), rep(par), rep(m), rep(z0), dv, noise={k: rep(v) for k, v in noise.items()})
    pop["patient"] = np.repeat(np.arange(n_pat), n_push)
    return pop


def run_setting(pop, ctrl, cfg=None):
    return simulate(pop, ctrl, cfg or Config())


def summarize(label, res, b0_outcome=None, group=None):
    o = res["outcome"]
    n = len(o)
    row = dict(setting=label, n=n)
    for name, code in (("fall", FALL), ("self", SELF), ("assisted", ASSISTED), ("caught", CAUGHT)):
        row[name + "_pct"] = 100.0 * np.mean(o == code)
    if b0_outcome is not None:
        # UNNECESSARY: B0 ended SELF and the robot ever applied qualifying assistance (impulse >= 1 N s) or ever
        # latched a catch decision, regardless of the controller's final outcome (a later FALL still counts).
        un = (b0_outcome == SELF) & (res['ever_assisted'] | res['ever_catch_decided'])
        row["unnecessary_pct"] = 100.0 * np.mean(un)
        row["unnecessary_of_b0self_pct"] = 100.0 * un.sum() / max((b0_outcome == SELF).sum(), 1)
        row["rescued_pct"] = 100.0 * np.mean((b0_outcome == FALL) & (o != FALL))
    row["mean_impulse"] = float(np.mean(res["impulse"]))
    row["mean_steps"] = float(np.mean(res["n_steps"]))
    row["mean_peak_F"] = float(np.mean(res["peakF"]))
    for k in ("t_decision", "t_force", "t_engage"):
        v = res[k]
        row["median_" + k] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
    return row


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def full_grid(pop, cfg=None, log=print):
    """All controller settings of SPEC 5.4 on the same trials. Returns (rows, results dict)."""
    cfg = cfg or Config()
    results = {}
    t0 = time.time()
    settings = [("B0 none", B0())]
    settings += [(f"B1 theta={th:g}", B1(th)) for th in THETAS]
    settings += [(f"B2 K={K:g}", B2(K)) for K in KS]
    settings += [(f"B1-XCoM theta={th:g}", B1XCoM(th)) for th in XTHETAS]
    settings += [(f"CP m={m:g}", CP(m)) for m in MARGS]
    settings += [(f"CP-pop m={m:g}", CP(m, ident="ident_pop")) for m in MARGS]
    settings += [("CP-noDC m=0.03", CP(0.03, dc=False))]
    for label, ctrl in settings:
        results[label] = run_setting(pop, ctrl, cfg)
        log(f"  {label:18s} {time.time() - t0:6.1f}s")
    b0 = results["B0 none"]["outcome"]
    rows = [summarize(lbl, results[lbl], b0) for lbl, _ in settings]
    return rows, results


def delay_sweep(pop, cfg=None, log=print):
    """Fall rate vs total delay Delta = tau_s + tau_a (tau_s fixed 0.03 s, tau_a = Delta - tau_s)."""
    cfg = cfg or Config()
    out = []
    for D in DELAYS:
        c2 = replace(cfg, tau_a=D - cfg.tau_s)
        for dc in (True, False):
            r = run_setting(pop, CP(0.03, dc=dc), c2)
            k = int((r["outcome"] == FALL).sum())
            lo, hi = wilson(k, len(r["outcome"]))
            out.append(dict(delay=D, dc=dc, fall_pct=100.0 * k / len(r["outcome"]), lo=lo, hi=hi,
                            self_pct=100.0 * np.mean(r["outcome"] == SELF)))
        log(f"  delay {D:.2f}")
    return out


def dose_sessions(cfg=None, n_sessions=12, per_session=20, sevs=(0.2, 0.5, 0.8), seed=SEED + 2,
                  gain=0.10, target=0.70, dose0=0.30, dmin=0.15, dmax=1.0, marg=0.03):
    """Adaptive perturbation dose (replaces the SPEC margin-AAN, see QUESTIONS.md / summary).
    CP (fixed margin) is the safety layer.  Per session 20 pushes, magnitude = dose_k * U(0.8, 1.2),
    random direction; dose_{k+1} = clip(dose_k + gain (SELF_k - target), dmin, dmax) with SELF_k the
    CP self-recovery rate.  The identical pushes are replayed with B0 (no robot) for comparison."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    P = len(sevs)
    par = rng.integers(0, 2, P)
    m = rng.uniform(60, 85, P)
    z0 = rng.uniform(0.85, 1.0, P)
    noise = sample_noise(np.random.default_rng(seed + 1), P)
    dose = np.full(P, dose0)
    rep = lambda a: np.repeat(a, per_session, axis=0)
    H = {k: [] for k in ("dose", "self_cp", "self_b0", "fall_cp", "fall_b0")}
    for s in range(n_sessions):
        mag = dose[:, None] * rng.uniform(0.8, 1.2, (P, per_session))
        ang = rng.uniform(0, 2 * np.pi, (P, per_session))
        dv = np.stack([mag * np.cos(ang), mag * np.sin(ang)], -1).reshape(-1, 2)
        pop = build_pop(rep(np.array(sevs)), rep(par), rep(m), rep(z0), dv,
                        noise={k: rep(v) for k, v in noise.items()})
        o_cp = simulate(pop, CP(marg), cfg)["outcome"].reshape(P, per_session)
        o_b0 = simulate(pop, B0(), cfg)["outcome"].reshape(P, per_session)
        H["dose"].append(dose.copy())
        H["self_cp"].append((o_cp == SELF).mean(1)); H["self_b0"].append((o_b0 == SELF).mean(1))
        H["fall_cp"].append((o_cp == FALL).sum(1)); H["fall_b0"].append((o_b0 == FALL).sum(1))
        dose = np.clip(dose + gain * (H["self_cp"][-1] - target), dmin, dmax)
    out = {k: np.array(v) for k, v in H.items()}
    out.update(sevs=list(sevs), target=target, per_session=per_session)
    return out


FAMILIES = [("B1", "B1 theta="), ("B1-XCoM", "B1-XCoM theta="), ("B2", "B2 K="), ("CP", "CP m="), ("CP-pop", "CP-pop m=")]


def matched_table(rows, targets=(4.0, 6.0)):
    """Linear interpolation of SELF% at common fall rates along each family's sweep (only if bracketed)."""
    out = []
    for fam, prefix in FAMILIES:
        pts = sorted([(r["fall_pct"], r["self_pct"], r["setting"]) for r in rows if r["setting"].startswith(prefix)])
        for tgt in targets:
            row = dict(family=fam, fall_target=tgt, self_pct_interp=float("nan"), bracket="not bracketed")
            for (f0, s0, n0), (f1, s1, n1) in zip(pts[:-1], pts[1:]):
                if f0 <= tgt <= f1 and f1 > f0:
                    row["self_pct_interp"] = s0 + (s1 - s0) * (tgt - f0) / (f1 - f0)
                    row["bracket"] = f"{n0} ({f0:.1f}% fall, {s0:.1f}% SELF) / {n1} ({f1:.1f}% fall, {s1:.1f}% SELF)"
                    break
            out.append(row)
    return out


def bootstrap_ci(pop, results, labels, reps=1000, seed=SEED + 3):
    """Paired patient-level bootstrap (resample the 50 patients with replacement, same resample for every
    controller) 95% CIs for fall% and SELF%."""
    rng = np.random.default_rng(seed)
    pid = pop["patient"]; npat = int(pid.max()) + 1
    cnt = {}
    for lbl in labels:
        o = results[lbl]["outcome"]
        cnt[lbl] = (np.bincount(pid, weights=(o == FALL), minlength=npat), np.bincount(pid, weights=(o == SELF), minlength=npat))
    ntr = np.bincount(pid, minlength=npat).astype(float)
    idx = rng.integers(0, npat, (reps, npat))
    out = []
    for lbl in labels:
        for j, name in ((0, "fall_pct"), (1, "self_pct")):
            est = 100 * cnt[lbl][j].sum() / ntr.sum()
            bs = 100 * cnt[lbl][j][idx].sum(1) / ntr[idx].sum(1)
            out.append(dict(setting=lbl, metric=name, estimate=est, ci_lo=float(np.percentile(bs, 2.5)),
                            ci_hi=float(np.percentile(bs, 97.5))))
    return out
