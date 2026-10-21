"""I3 surrogate-learner experiment.  Usage (from repo root):
    .venv/Scripts/python sim/i3/run_i3.py            # simulate (parallel, cached) + analyse
    .venv/Scripts/python sim/i3/run_i3.py --analyze  # re-do tables/figures from cache only
Env: I3_SEEDS (default 8)."""
from __future__ import annotations

import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figs")
CACHE = os.path.join(RES, "_i3_cache")

from catchpoint.controllers import B0, B1, B2, CP          # noqa: E402
from i3 import emken, learner as L                          # noqa: E402

CONDS = {"B0": lambda: B0(), "B1": lambda: B1(0.05), "B2": lambda: B2(200), "CP": lambda: CP(0.03)}
LABEL = {"B0": "B0 no robot", "B1": "B1 lock (0.05)", "B2": "B2 spring (K=200)", "CP": "CP (m=0.03)"}
LAMS = [0.0, 0.5]
N_SEED = int(os.environ.get("I3_SEEDS", 12))


def work(cond):
    t0 = time.time()
    groups = L.make_groups(N_SEED)
    if cond == "BASE":                       # untrained theta0 (and library-default skill) no-robot retention
        gid = np.arange(len(groups))
        z0 = np.tile(L.to_z(L.THETA0), (len(groups), 1))
        zd = np.tile(L.to_z(np.array([0.01, 0.5, 1.0, 0.0])), (len(groups), 1))
        s0, f0 = L.retention(groups, gid, z0)
        sd, fd = L.retention(groups, gid, zd)
        np.savez(os.path.join(CACHE, "BASE.npz"), self0=s0, fall0=f0, selfd=sd, falld=fd)
        return cond, time.time() - t0
    H = L.train(groups, LAMS, CONDS[cond])
    z = H["z"][-1]
    rs, rf = L.retention(groups, H["gid"], z)
    np.savez(os.path.join(CACHE, cond + ".npz"), retention_self=rs, retention_fall=rf, **H)
    return cond, time.time() - t0


def ci(a):
    from scipy import stats
    a = np.asarray(a, float)
    n = len(a)
    m = a.mean()
    h = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return m, h


def analyze():
    groups = L.make_groups(N_SEED)
    sev_of = np.array([g["name"] for g in groups])
    D = {c: np.load(os.path.join(CACHE, c + ".npz")) for c in CONDS}
    B = np.load(os.path.join(CACHE, "BASE.npz"))
    rows = []
    for c, d in D.items():
        gid, lam = d["gid"], d["lam"]
        tfalls = d["falls"].sum(0)                       # (GL,) training falls (injury events) of 240 pushes
        teffort = d["effort"].mean(0)
        for sev in L.SEVS:
            for lm in LAMS:
                sel = (sev_of[gid] == sev) & (lam == lm)
                row = dict(condition=c, severity=sev, lam=lm, n_seeds=int(sel.sum()))
                for key, arr in (("train_falls", tfalls), ("retention_self_pct", 100 * d["retention_self"]),
                                 ("retention_fall_pct", 100 * d["retention_fall"]), ("train_effort", teffort)):
                    m, h = ci(arr[sel])
                    row[key], row[key + "_ci"] = m, h
                for i, n in enumerate(L.NAMES):
                    row["theta_" + n] = float(L.to_theta(d["z"][-1][sel]).mean(0)[i])
                rows.append(row)
    for sev in L.SEVS:
        sel = sev_of == sev
        m, h = ci(100 * B["self0"][sel]); md, hd = ci(100 * B["selfd"][sel])
        rows.append(dict(condition="untrained theta0", severity=sev, lam=float("nan"), n_seeds=int(sel.sum()),
                         train_falls=0.0, train_falls_ci=0.0, retention_self_pct=m, retention_self_pct_ci=h,
                         retention_fall_pct=100 * float(B["fall0"][sel].mean()), retention_fall_pct_ci=float("nan"),
                         train_effort=float("nan"), train_effort_ci=float("nan")))
        rows.append(dict(condition="library-default skill", severity=sev, lam=float("nan"), n_seeds=int(sel.sum()),
                         train_falls=0.0, train_falls_ci=0.0, retention_self_pct=md, retention_self_pct_ci=hd,
                         retention_fall_pct=100 * float(B["falld"][sel].mean()), retention_fall_pct_ci=float("nan"),
                         train_effort=float("nan"), train_effort_ci=float("nan")))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RES, "i3_results.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    # per-seed paired differences vs B0: retention (same patient, same seed, same pushes)
    paired = []
    for lm in LAMS:
        for c in ("B1", "B2", "CP"):
            for sev in L.SEVS:
                sel0 = (sev_of[D["B0"]["gid"]] == sev) & (D["B0"]["lam"] == lm)
                sel1 = (sev_of[D[c]["gid"]] == sev) & (D[c]["lam"] == lm)
                dr = 100 * (D[c]["retention_self"][sel1] - D["B0"]["retention_self"][sel0])
                m, h = ci(dr)
                paired.append(dict(lam=lm, condition=c, severity=sev, d_retention_self_vs_B0_pct=m, ci=h))
    with open(os.path.join(RES, "i3_paired_vs_B0.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, list(paired[0])); w.writeheader(); w.writerows(paired)

    # Emken sanity learner
    em = []
    x0 = {"mild": 0.4, "moderate": 0.2, "severe": 0.0}
    bg = {"mild": 0.09, "moderate": 0.07, "severe": 0.05}      # slower learning with severity
    for sev in L.SEVS:
        seeds = list(range(N_SEED))
        for c in CONDS:
            r = emken.run(c, np.full(N_SEED, x0[sev]) + 0.03 * np.random.default_rng(7).standard_normal(N_SEED), seeds, b=bg[sev])
            mf, hf = ci(r["falls"]); ms, hs = ci(100 * r["retention_self"])
            em.append(dict(condition=c, severity=sev, train_falls=mf, train_falls_ci=hf,
                           retention_self_pct=ms, retention_self_pct_ci=hs, x_final=float(r["x"][-1].mean())))
    with open(os.path.join(RES, "i3_emken.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, list(em[0])); w.writeheader(); w.writerows(em)
    figures(rows, D, sev_of, em, B)
    summary(rows, paired, em)


COL = {"B0": "#555555", "B1": "#d95f02", "B2": "#7570b3", "CP": "#1b9e77"}
MK = {"mild": "o", "moderate": "s", "severe": "^"}


def figures(rows, D, sev_of, em, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    get = lambda c, s, lm: next(r for r in rows if r["condition"] == c and r["severity"] == s and r["lam"] == lm)

    fig = plt.figure(figsize=(14, 5.2))
    ax = fig.add_subplot(1, 3, 1)
    for c in CONDS:
        for s in L.SEVS:
            r = get(c, s, 0.5)
            ax.errorbar(r["train_falls"], r["retention_self_pct"], xerr=r["train_falls_ci"], yerr=r["retention_self_pct_ci"],
                        fmt=MK[s], color=COL[c], ms=8, capsize=2, lw=1, alpha=.9)
    for s in L.SEVS:
        r = next(r for r in rows if r["condition"] == "untrained theta0" and r["severity"] == s)
        ax.axhline(r["retention_self_pct"], color="k", lw=.5, ls=":")
    h1 = [plt.Line2D([], [], color=COL[c], marker="o", ls="", label=LABEL[c]) for c in CONDS]
    h2 = [plt.Line2D([], [], color="k", marker=MK[s], ls="", mfc="none", label=s) for s in L.SEVS]
    ax.legend(handles=h1 + h2, fontsize=7.5, loc="best")
    ax.set_xlabel("training falls (injury events, of 240 pushes)"); ax.set_ylabel("no-robot retention SELF rate (%)")
    ax.set_title("I3: training safety vs retention (lambda=0.5)\nmean +/- 95% CI over seeds; dotted = untrained", fontsize=10)
    ax.grid(alpha=.3)
    for j, (title, s) in enumerate((("learning curve: moderate", "moderate"), ("learning curve: mild", "mild"))):
        ax = fig.add_subplot(1, 3, 2 + j)
        for c in CONDS:
            d = D[c]
            sel = (sev_of[d["gid"]] == s) & (d["lam"] == 0.5)
            P = 100 * d["probe"][:, sel]                       # (n_probe, seeds)
            x = np.arange(P.shape[0]) * 3
            m = P.mean(1); h = np.array([ci(P[i])[1] for i in range(P.shape[0])])
            ax.plot(x, m, "-o", color=COL[c], label=LABEL[c], ms=4); ax.fill_between(x, m - h, m + h, color=COL[c], alpha=.12)
        ax.set_xlabel("training session"); ax.set_ylabel("no-robot probe SELF rate (%)"); ax.set_title(title + " (lambda=0.5)", fontsize=10)
        ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "i3_learning.png"), dpi=140); plt.close(fig)

    # ablation figure
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    w = 0.2
    for k, key, yl in ((0, "retention_self_pct", "no-robot retention SELF (%)"), (1, "train_falls", "training falls (of 240)"),
                       (2, "train_effort", "mean effort per push (m-equiv)")):
        ax = axs[k]
        for si, s in enumerate(L.SEVS):
            for ci_, c in enumerate(CONDS):
                for li, lm in enumerate(LAMS):
                    r = get(c, s, lm)
                    xpos = si * 5 + ci_ * 1.0 + (li - .5) * 0.38
                    ax.bar(xpos, r[key], 0.36, yerr=r[key + "_ci"], color=COL[c], alpha=1.0 if lm > 0 else 0.4,
                           hatch="" if lm > 0 else "//", capsize=1.5, edgecolor="k", lw=.3)
        ax.set_xticks([si * 5 + 1.5 for si in range(3)]); ax.set_xticklabels(list(L.SEVS)); ax.set_ylabel(yl); ax.grid(alpha=.3, axis="y")
    hh = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CONDS] + [plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="//"),
                                                                       plt.Rectangle((0, 0), 1, 1, fc="grey", ec="k")]
    axs[0].legend(hh, [LABEL[c] for c in CONDS] + ["lambda=0 (hatched)", "lambda=0.5 (solid)"], fontsize=7, loc="upper right")
    fig.suptitle("I3 ablation: error-only (lambda=0) vs error+effort (lambda=0.5)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "i3_ablation.png"), dpi=140); plt.close(fig)

    # Emken sanity figure
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for k, key in enumerate(("train_falls", "retention_self_pct")):
        ax = axs[k]
        for si, s in enumerate(L.SEVS):
            for ci_, c in enumerate(CONDS):
                r = next(r for r in em if r["condition"] == c and r["severity"] == s)
                ax.bar(si * 5 + ci_, r[key], 0.9, yerr=r[key + "_ci"], color=COL[c], capsize=2)
        ax.set_xticks([si * 5 + 1.5 for si in range(3)]); ax.set_xticklabels(list(L.SEVS)); ax.grid(alpha=.3, axis="y")
        ax.set_ylabel("training falls (of 240)" if k == 0 else "no-robot retention SELF (%)")
    axs[0].legend([plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CONDS], [LABEL[c] for c in CONDS], fontsize=7)
    fig.suptitle("Emken-style scalar learner (sanity check)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "i3_emken.png"), dpi=140); plt.close(fig)


def summary(rows, paired, em):
    get = lambda c, s, lm: next(r for r in rows if r["condition"] == c and r["severity"] == s and r["lam"] == lm)
    out = ["# I3 surrogate-learner experiment (auto-generated by sim/i3/run_i3.py)", ""]
    out.append(f"Seeds per cell: {N_SEED}. 12 sessions x 20 pushes (240 training pushes, 0.2-0.6 m/s). Retention: 200 fresh pushes, NO robot. "
               "Mean +/- 95% t-CI across seeds. Training falls = injury events during training (of 240).")
    out.append("")
    for lm in (0.5, 0.0):
        out.append(f"## lambda = {lm:g}" + ("  (main)" if lm > 0 else "  (ablation: error-only reward)"))
        out.append("")
        out.append("| condition | severity | training falls | retention SELF % | retention FALL % | effort/push | mean theta (eps, kc, g, rho) |")
        out.append("|---|---|---|---|---|---|---|")
        for s in L.SEVS:
            for r in [next(r for r in rows if r["condition"] == "untrained theta0" and r["severity"] == s)] + [get(c, s, lm) for c in CONDS]:
                th = ", ".join(f"{r['theta_' + n]:.2f}" for n in L.NAMES) if "theta_eps" in r else "(0.08, 0.20, 0.75, 0.00)"
                lab = LABEL.get(r["condition"], r["condition"])
                ef = f"{r['train_effort']:.2f}" if r["train_effort"] == r["train_effort"] else "-"
                out.append(f"| {lab} | {s} | {r['train_falls']:.1f} +/- {r['train_falls_ci']:.1f} | "
                           f"{r['retention_self_pct']:.1f} +/- {r['retention_self_pct_ci']:.1f} | {r['retention_fall_pct']:.1f} | {ef} | {th} |")
        out.append("")
    for lm in (0.5, 0.0):
        out.append(f"## Paired retention difference vs B0 (lambda={lm:g}, percentage points, mean +/- 95% CI)")
        out.append("")
        out.append("| condition | mild | moderate | severe |")
        out.append("|---|---|---|---|")
        for c in ("B1", "B2", "CP"):
            cells = [next(p for p in paired if p["condition"] == c and p["severity"] == s and p["lam"] == lm) for s in L.SEVS]
            out.append(f"| {LABEL[c]} | " + " | ".join(f"{p['d_retention_self_vs_B0_pct']:+.1f} +/- {p['ci']:.1f}" for p in cells) + " |")
        out.append("")
    out.append("## Emken-style scalar learner (sanity check, same four conditions)")
    out.append("")
    out.append("| condition | severity | training falls | retention SELF % |")
    out.append("|---|---|---|---|")
    for r in em:
        out.append(f"| {LABEL[r['condition']]} | {r['severity']} | {r['train_falls']:.1f} +/- {r['train_falls_ci']:.1f} | "
                   f"{r['retention_self_pct']:.1f} +/- {r['retention_self_pct_ci']:.1f} |")
    out.append("")
    ip = os.path.join(HERE, "interpretation.md")
    out.append(open(ip, encoding="utf8").read() if os.path.exists(ip) else "(no interpretation.md)")
    open(os.path.join(RES, "i3_summary.md"), "w", encoding="utf8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    if "--analyze" not in sys.argv:
        t0 = time.time()
        with Pool(5) as p:
            for cond, dt in p.imap_unordered(work, ["B0", "B1", "B2", "CP", "BASE"]):
                print(f"[{time.time() - t0:6.1f}s] {cond} done ({dt:.0f}s)", flush=True)
    analyze()
    print("analysis done", flush=True)
