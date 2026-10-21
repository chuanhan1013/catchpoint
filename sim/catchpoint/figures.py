"""Figures, animations and text outputs (SPEC 5.6, 6, 7).  Titles state computed findings."""
from __future__ import annotations

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle

from .model import HS, OUTCOME_NAMES, SIDE

C = dict(L0="#6B7A8F", L1="#E0A030", L2="#C8443A", teal="#1F6F78",
         B0="#9AA0A6", B1="#7B5EA7", B2="#3C78B4", CP="#1F6F78", ink="#2B2B2B", fall="#2B2B2B")
OUT_COL = [C["teal"], C["L1"], C["L2"], C["fall"]]
LEVEL_COL = [C["L0"], C["L1"], C["L2"]]
LEVEL_NAME = ["L0 transparent", "L1 assist", "L2 catch"]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "legend.frameon": False, "figure.dpi": 100,
        "axes.grid": True, "grid.color": "#E3E6EA", "grid.linewidth": 0.6,
    })


_style()


def save_fig(fig, path_noext):
    fig.savefig(path_noext + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path_noext + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_pareto(rows, matched, path):
    by = {r["setting"]: r for r in rows}
    groups = [
        ([r for r in rows if r["setting"].startswith("B1 theta=")], C["B1"], "B1 threshold-lock (theta)", "s", "-"),
        ([r for r in rows if r["setting"].startswith("B1-XCoM")], "#B58BD6", "B1-XCoM threshold (theta)", "v", "--"),
        ([r for r in rows if r["setting"].startswith("B2 K=")], C["B2"], "B2 impedance + lock (K)", "^", "-"),
        ([r for r in rows if r["setting"].startswith("CP-pop")], "#6FB3BA", "CP-pop, no patient model (m)", "P", "--"),
        ([r for r in rows if r["setting"].startswith("CP m=")], C["CP"], "CP (margin m)", "o", "-"),
    ]
    cpd, b0 = by["CP m=0.03"], by["B0 none"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.22))
    for ax, zoom in zip(axes, (False, True)):
        for grp, col, lab, mk, ls in groups:
            ax.plot([r["fall_pct"] for r in grp], [r["self_pct"] for r in grp], ls, marker=mk, color=col,
                    lw=2.6 if col == C["CP"] else 1.5, ms=6, label=lab if not zoom else None,
                    zorder=3 if col == C["CP"] else 2)
            for ii, r in enumerate(grp):
                tag = r["setting"].split("=")[1]
                keep = (zoom and col in (C["B1"], "#B58BD6", C["B2"])) or (not zoom and col in (C["B1"], C["B2"]))
                if col == C["CP"]:
                    keep = zoom and ii in (0, len(grp) - 1)
                if col == C["B2"] and tag not in ("100", "200", "400"):
                    keep = False
                if col == "#6FB3BA":
                    keep = False
                if not keep:
                    continue
                ax.annotate(tag, (r["fall_pct"], r["self_pct"]), textcoords="offset points",
                            xytext=(5, -10 if col != C["B2"] else 6), fontsize=7, color=col)
        ax.plot(b0["fall_pct"], b0["self_pct"], "D", color=C["B0"], ms=8, label="B0 no robot" if not zoom else None, zorder=4)
        ax.plot(cpd["fall_pct"], cpd["self_pct"], "o", ms=14, mfc="none", mec=C["ink"], mew=1.6, zorder=5,
                label="CP default m = 0.03" if not zoom else None)
        ax.set_xlabel("Fall rate (% of 2000 trials)  <- safer")
        ax.set_ylabel("Self-recovery rate, SELF (%)  -> more training value")
        if zoom:
            ax.set_xlim(-0.5, 12); ax.set_ylim(-2, 42)
            for tgt in (4.0, 6.0):
                ax.axvline(tgt, color="#BBBBBB", lw=0.8, ls=":")
            ax.set_title("zoom: equal-safety region (dotted = 4% and 6% falls)", loc="left", fontsize=9)
        else:
            ax.set_xlim(-2, 64); ax.set_ylim(-3, 50)
    axes[0].legend(loc="center right", fontsize=7.5)
    mt = {(m["family"], m["fall_target"]): m["self_pct_interp"] for m in matched}
    f = lambda v: "n/a" if not np.isfinite(v) else f"{v:.0f}%"
    fig.suptitle(f"Default CP: {cpd['fall_pct']:.1f}% falls, {cpd['self_pct']:.0f}% SELF. At matched 4% falls (interpolated) SELF is "
                 f"CP {f(mt[('CP', 4.0)])}, CP-pop {f(mt[('CP-pop', 4.0)])}, B1-XCoM {f(mt[('B1-XCoM', 4.0)])}, B1 {f(mt[('B1', 4.0)])}, B2 {f(mt[('B2', 4.0)])}",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    save_fig(fig, path)


# --------------------------------------------------------------------------
def _segments_by_level(t, y, lev):
    cols, segs = [], []
    for i in range(len(t) - 1):
        segs.append([(t[i], y[i]), (t[i + 1], y[i + 1])]); cols.append(LEVEL_COL[int(lev[i])])
    return segs, cols


def _step_times(sw, t):
    sw = np.asarray(sw)
    return t[1:][(sw[1:] > 0) & (sw[:-1] == 0)]


def fig_timeline(show, path, cfg_note=""):
    names = [("B1", "B1 threshold-lock theta=0.08"), ("B2", "B2 impedance K=800"), ("CP", "CP m=0.03")]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 7.6), sharex=True)
    tmax = max(show[k]["t"][-1] for k, _ in names)
    for r, (key, label) in enumerate(names):
        d = show[key]; t = d["t"]
        a = axes[r, 0]
        a.plot(t, d["rho"], color=C[key], lw=2, label="predicted landing offset rho")
        a.plot(t, d["rbound"], "--", color=C["ink"], lw=1.2, label="recoverable bound r = Lmax - m")
        a.plot(t, d["dist_xi"], color="#9AA0A6", lw=1.2, label="distance of xi from support")
        a.fill_between(t, d["rbound"], np.maximum(d["rho"], d["rbound"]), color=C["L2"], alpha=0.18, lw=0)
        a.set_ylim(0, 1.2)
        a.set_ylabel(f"{label}\n[m]", fontsize=8.5, color=C[key], fontweight="bold")
        b = axes[r, 1]
        Fm = np.hypot(d["F"][:, 0], d["F"][:, 1])
        b.fill_between(t, 0, Fm, color=C[key], alpha=0.25, lw=0)
        b.plot(t, Fm, color=C[key], lw=1.6)
        b.set_ylabel("|F| [N]")
        c = axes[r, 2]
        segs, cols = _segments_by_level(t, d["level"].astype(float), d["level"])
        c.add_collection(LineCollection(segs, colors=cols, linewidths=4))
        c.set_ylim(-0.5, 2.5); c.set_yticks([0, 1, 2]); c.set_yticklabels(["L0", "L1", "L2"])
        steps = _step_times(d["swing"], t)
        for ax in (a, b, c):
            for ts in steps:
                ax.axvline(ts, color="#333333", lw=0.9, ls=":")
            ax.axvline(0.3, color="#BBBBBB", lw=0.8)
        out = d["outcome"]
        imp = float(np.sum(np.hypot(d["F"][:, 0], d["F"][:, 1])) * (t[1] - t[0]))
        c.text(0.99, 0.93, f"{out}  (impulse {imp:.0f} N s)", transform=c.transAxes, ha="right", va="top",
               fontsize=9, fontweight="bold", color=C["fall"] if out == "FALL" else C["ink"])
        if out == "FALL":
            for ax in (a, b, c):
                ax.axvline(t[-1], color=C["fall"], lw=1.5)
    axes[2, 0].legend(fontsize=7, loc="upper right")
    axes[0, 0].set_title("Is the patient still recoverable?", loc="left")
    axes[0, 1].set_title("Robot force (pelvis + harness)", loc="left")
    axes[0, 2].set_title("Intervention level", loc="left")
    for ax in axes[2]:
        ax.set_xlabel("time [s]  (push at 0.3 s; dotted = patient lifts a foot)")
        ax.set_xlim(0, tmax)
    out = {k: show[k]["outcome"] for k, _ in names}
    cpd = show["CP"]
    t = cpd["t"]; lev = cpd["level"]
    t1 = float(np.sum(lev == 1) * (t[1] - t[0])); t2 = float(np.sum(lev == 2) * (t[1] - t[0]))
    pk = float(np.max(np.hypot(cpd["F"][:, 0], cpd["F"][:, 1])))
    fig.suptitle(f"Same push: threshold-lock -> {out['B1']}, impedance -> {out['B2']}, CP -> {out['CP']} "
                 f"(L1 for {t1:.2f} s, L2 for {t2:.2f} s, peak {pk:.0f} N)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, path)


# --------------------------------------------------------------------------
def fig_map(show, pop1, marg, path):
    d = show["CP"]; b0 = show["B0"]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    par = int(pop1["par"][0]); L = pop1["Lmax"][0]; sc = pop1["scale"][0]
    feet0 = {0: np.array([0.0, 0.10]), 1: np.array([0.0, -0.10])}
    # reachable discs (centre = other foot at rest)
    for j in (0, 1):
        ax.add_patch(Circle((feet0[1 - j][1], feet0[1 - j][0]), L[j] - marg, fc=C["teal"], alpha=0.15, ec=C["teal"],
                            lw=0.8, zorder=1))
        ax.text(feet0[1 - j][1] + (-1 if j == 0 else 1) * (L[j] - marg) * 0.62, feet0[1 - j][0] + (L[j] - marg) * 0.62,
                ("left" if j == 0 else "right") + " leg reach" + (" (paretic)" if j == par else ""),
                fontsize=7.5, color=C["teal"], ha="center")
    for j in (0, 1):
        ax.add_patch(Circle((feet0[1 - j][1], feet0[1 - j][0]), 0.7 * L[j] - marg, fc="none", ec=C["teal"],
                            lw=0.9, ls=":", zorder=1))
    # S_cop at rest
    pts = []
    for j in (0, 1):
        h = HS * sc[j]
        for sx in (1, -1):
            for sy in (1, -1):
                pts.append(feet0[j] + np.array([sx * h[0], sy * h[1]]))
    from scipy.spatial import ConvexHull
    pts = np.array(pts); hv = pts[ConvexHull(pts).vertices]
    ax.add_patch(Polygon(hv[:, ::-1], closed=True, fc="#FFFFFF00", ec=C["L0"], lw=1.4, ls="--", zorder=2))
    for j in (0, 1):
        ax.add_patch(Rectangle((feet0[j][1] - HS[1], feet0[j][0] - HS[0]), 2 * HS[1], 2 * HS[0], fc="#DDE1E6",
                               ec=C["L0"], lw=1, zorder=2))
    # landed feet (final)
    for key, col in (("foot_l", C["ink"]), ("foot_r", C["ink"])):
        f = d[key][-1]
        if np.linalg.norm(f - feet0[0 if key == "foot_l" else 1]) > 1e-6:
            ax.add_patch(Rectangle((f[1] - HS[1], f[0] - HS[0]), 2 * HS[1], 2 * HS[0], fc="none", ec=C["ink"],
                                   lw=1.2, ls="-", zorder=4))
    # B0 path for reference
    ax.plot(b0["xi"][:, 1], b0["xi"][:, 0], color=C["B0"], lw=1.4, ls="--", zorder=3, label="xi path, no robot (B0)")
    ax.plot(b0["xi"][-1, 1], b0["xi"][-1, 0], "x", color=C["B0"], ms=8, mew=2, zorder=3)
    # CP paths
    ax.plot(d["com"][:, 1], d["com"][:, 0], color=C["teal"], lw=1.2, alpha=0.8, zorder=4, label="COM path (CP)")
    segs = [[(d["xi"][i, 1], d["xi"][i, 0]), (d["xi"][i + 1, 1], d["xi"][i + 1, 0])] for i in range(len(d["t"]) - 1)]
    lc = LineCollection(segs, colors=[LEVEL_COL[int(l)] for l in d["level"][:-1]], linewidths=3.2, zorder=5)
    ax.add_collection(lc)
    ax.plot([], [], color=C["L0"], lw=3, label="xi path, L0 transparent")
    ax.plot([], [], color=C["L1"], lw=3, label="xi path, L1 assist")
    ax.plot([], [], color=C["L2"], lw=3, label="xi path, L2 catch")
    ax.plot(0, 0, "o", color=C["teal"], ms=5, zorder=6)
    ax.set_aspect("equal"); ax.invert_xaxis()
    xs = np.concatenate([d["xi"][:, 1], b0["xi"][:, 1]]); ys = np.concatenate([d["xi"][:, 0], b0["xi"][:, 0]])
    ax.set_xlim(max(xs.max() + 0.15, 0.55), min(-0.45, xs.min() - 0.1))
    ax.set_ylim(min(ys.min() - 0.15, -0.35), max(ys.max() + 0.2, 0.45))
    ax.set_xlabel("lateral y [m]  (left = +, toward the paretic side; dotted circles = crossover reach)"); ax.set_ylabel("forward x [m]")
    lev = d["level"]; dt = d["t"][1] - d["t"][0]
    ax.set_title(f"CP lets xi wander until the reach limit: L0 for {np.sum(lev == 0) * dt:.2f} s, "
                 f"L1 {np.sum(lev == 1) * dt:.2f} s, L2 {np.sum(lev == 2) * dt:.2f} s", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left")
    save_fig(fig, path)


# --------------------------------------------------------------------------
def fig_delay(sweep, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for dc, col, lab, mk in ((True, C["CP"], "CP (delay-compensated)", "o"),
                             (False, "#5A5A5A", "CP-noDC (ablation)", "s")):
        rows = [r for r in sweep if r["dc"] == dc]
        x = [r["delay"] for r in rows]; y = np.array([r["fall_pct"] for r in rows])
        lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=("-" if dc else "--") + mk, color=col, lw=2, ms=6, capsize=3, label=lab)
    ax.set_xlabel("total sensing + actuation delay  Delta = tau_s + tau_a  [s]")
    ax.set_ylabel("Fall rate [%]  (95% Wilson CI, trials pooled)")
    a = {r["delay"]: r["fall_pct"] for r in sweep if r["dc"]}
    b = {r["delay"]: r["fall_pct"] for r in sweep if not r["dc"]}
    d0 = 0.10
    dmax = max(a, key=lambda k: b[k] - a[k])
    ax.set_title(f"Delay compensation: falls {a[d0]:.1f}% vs {b[d0]:.1f}% at 0.10 s; "
                 f"largest gap {b[dmax] - a[dmax]:.1f} pp at {dmax:.2f} s", loc="left", fontsize=9.5)
    ax.set_xticks(sorted(a)); ax.legend()
    ax.set_ylim(bottom=0)
    save_fig(fig, path)


# --------------------------------------------------------------------------
def terciles(sev):
    return np.digitize(sev, [1 / 3, 2 / 3])


def fig_severity(pop, results, path):
    ctrls = [("B0 none", "B0", "B0"), ("B1 theta=0.08", "B1\n0.08", "B1"),
             ("B2 K=800", "B2\n800", "B2"), ("CP m=0.03", "CP\n0.03", "CP")]
    tz = terciles(pop["sev"])
    names = ["mild (s<1/3)", "moderate", "severe (s>2/3)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4), sharey=True)
    stats = {}
    for ti, ax in enumerate(axes):
        for ci, (lbl, short, key) in enumerate(ctrls):
            o = results[lbl]["outcome"][tz == ti]
            frac = [100 * np.mean(o == k) for k in range(4)]
            stats[(ti, key)] = frac
            bottom = 0
            for k in range(4):
                ax.bar(ci, frac[k], bottom=bottom, color=OUT_COL[k], width=0.72, edgecolor="white", lw=0.6,
                       hatch="//" if k == 3 else None, label=OUTCOME_NAMES[k] if (ti == 0 and ci == 0) else None)
                if frac[k] > 6:
                    ax.text(ci, bottom + frac[k] / 2, f"{frac[k]:.0f}", ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
                bottom += frac[k]
        ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in ctrls], fontsize=8.5)
        ax.set_title(names[ti] + f"  (n={np.sum(tz == ti)})", loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of trials [%]")
    axes[0].legend(ncol=2, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.12))
    sv = {k: stats[(2, k)] for k in ("B0", "B1", "B2", "CP")}
    fig.suptitle(f"Severe tercile: CP falls {sv['CP'][3]:.0f}% and keeps {sv['CP'][0]:.0f}% SELF "
                 f"(B0 {sv['B0'][3]:.0f}% falls / {sv['B0'][0]:.0f}% SELF; B1 {sv['B1'][3]:.0f}% / {sv['B1'][0]:.0f}%)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, path)
    return stats


# --------------------------------------------------------------------------
def fig_aan(d, path):
    cols = [C["L0"], C["B2"], C["L2"]]
    names = {0.2: "mild (s=0.2)", 0.5: "moderate (s=0.5)", 0.8: "severe (s=0.8)"}
    S = np.arange(1, d["dose"].shape[0] + 1)
    fig = plt.figure(figsize=(10.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.28)
    a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])
    for i, s in enumerate(d["sevs"]):
        lab = names.get(s, f"s={s}")
        a0.plot(S, d["dose"][:, i], "-o", color=cols[i], lw=1.8, ms=4, label=lab)
        a1.plot(S, np.cumsum(d["fall_b0"][:, i]), "--", color=cols[i], lw=1.6)
        a1.plot(S, np.cumsum(d["fall_cp"][:, i]), "-o", color=cols[i], lw=2, ms=3.5)
        a2.plot(S, 100 * d["self_cp"][:, i], "-o", color=cols[i], lw=1.8, ms=4)
    a2.axhline(100 * d["target"], color=C["ink"], ls="--", lw=1)
    a2.text(S[-1], 100 * d["target"] + 2, "70% target", ha="right", fontsize=8)
    a0.set_ylabel("push dose [m/s]"); a0.set_xlabel("session (20 pushes each, magnitude = dose x U(0.8, 1.2))")
    a0.legend(fontsize=8, ncol=3, loc="upper left")
    a0.set_title("Dose titration: dose <- clip(dose + 0.10 (SELF - 0.70), 0.15, 1.0)", loc="left", fontsize=9.5)
    a1.set_ylabel("cumulative falls"); a1.set_xlabel("session")
    a1.plot([], [], "--", color=C["ink"], label="B0 (no robot), same pushes")
    a1.plot([], [], "-o", color=C["ink"], label="CP safety layer")
    a1.legend(fontsize=8, loc="upper left")
    a2.set_ylabel("CP SELF rate [%]"); a2.set_xlabel("session"); a2.set_ylim(0, 100)
    for ax in (a0, a1, a2):
        ax.set_xticks(S)
    fb0, fcp = int(d["fall_b0"].sum()), int(d["fall_cp"].sum())
    n = d["fall_b0"].shape[0] * d["fall_b0"].shape[1] * d["per_session"]
    fig.suptitle(f"Titrated dose ends at {', '.join(f'{x:.2f}' for x in d['dose'][-1])} m/s (mild, moderate, severe);\n"
                 f"the CP layer allows {fcp} falls in {n} pushes where B0 would fall {fb0} times; "
                 f"CP SELF mean {100 * d['self_cp'].mean():.0f}% vs target 70%",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(top=0.90)
    save_fig(fig, path)


# --------------------------------------------------------------------------
def make_anim(d, label, path, step=20):
    import imageio.v2 as imageio
    t = d["t"]; n = len(t)
    idx = list(range(0, n, step))
    frames = []
    fig = plt.figure(figsize=(7.2, 3.7), dpi=80)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.07, right=0.98, bottom=0.14, top=0.86, wspace=0.28)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1])
    par = d["_par"]; sc = d["_scale"]; z0 = d["_z0"]
    for fi in idx + [n - 1] * 12:
        for ax in (a1, a2):
            ax.clear(); ax.grid(False)
        a1.set_aspect("equal"); a1.set_xlim(0.55, -0.55); a1.set_ylim(-0.45, 0.55)
        a1.set_xlabel("y [m] (left +)", fontsize=7); a1.set_ylabel("x forward [m]", fontsize=7)
        a1.tick_params(labelsize=6)
        lev = int(d["level"][fi])
        # support
        for key, j in (("foot_l", 0), ("foot_r", 1)):
            f = d[key][fi]
            h = HS * sc[j]
            lifted = d["swing_h"][fi] > 1e-6 and d["swing"][fi] == j + 1
            a1.add_patch(Rectangle((f[1] - HS[1], f[0] - HS[0]), 2 * HS[1], 2 * HS[0],
                                   fc="#FFFFFF" if lifted else "#DDE1E6", ec=C["L0"], lw=1, ls=":" if lifted else "-"))
        a1.plot(d["xi"][:fi + 1, 1], d["xi"][:fi + 1, 0], color=C["L0"], lw=1)
        a1.plot(d["com"][:fi + 1, 1], d["com"][:fi + 1, 0], color=C["teal"], lw=1.4)
        a1.plot(d["com"][fi, 1], d["com"][fi, 0], "o", color=C["teal"], ms=6)
        a1.plot(d["xi"][fi, 1], d["xi"][fi, 0], "D", color=LEVEL_COL[lev], ms=6, mec=C["ink"])
        F = d["F"][fi]
        if np.hypot(*F) > 1:
            a1.annotate("", xy=(d["com"][fi, 1] + F[1] * 0.0012, d["com"][fi, 0] + F[0] * 0.0012),
                        xytext=(d["com"][fi, 1], d["com"][fi, 0]),
                        arrowprops=dict(arrowstyle="-|>", color=LEVEL_COL[max(lev, 1)], lw=2))
        a1.set_title("top view", fontsize=8, loc="left")
        # frontal plane
        a2.set_xlim(0.7, -0.7); a2.set_ylim(-0.05, 1.85); a2.set_aspect("equal")
        a2.axhline(0, color="#999999", lw=1)
        a2.tick_params(labelsize=6); a2.set_xlabel("y [m] (left +)", fontsize=7)
        yc = d["com"][fi, 1]
        pel = np.array([yc, z0])
        for key, j in (("foot_l", 0), ("foot_r", 1)):
            f = d[key][fi]
            foot = np.array([f[1], d["swing_h"][fi] if d["swing"][fi] == j + 1 else 0.0])
            a2.plot([foot[0], pel[0]], [foot[1], pel[1]], color=C["ink"], lw=3, solid_capstyle="round")
            a2.plot([foot[0] - HS[1], foot[0] + HS[1]], [foot[1], foot[1]], color=C["ink"], lw=4)
        sh = pel + np.array([0, 0.55]); a2.plot([pel[0], sh[0]], [pel[1], sh[1]], color=C["teal"], lw=5, solid_capstyle="round")
        a2.add_patch(Circle(sh + np.array([0, 0.12]), 0.09, fc="#F2F4F6", ec=C["teal"], lw=2))
        a2.plot(d["xi"][fi, 1], 0.0, "D", color=LEVEL_COL[lev], ms=7, mec=C["ink"])
        if np.hypot(*F) > 1:
            a2.annotate("", xy=(pel[0] + F[1] * 0.0012, pel[1]), xytext=(pel[0], pel[1]),
                        arrowprops=dict(arrowstyle="-|>", color=LEVEL_COL[max(lev, 1)], lw=2.5))
        a2.set_title("frontal plane (xi on the floor)", fontsize=8, loc="left")
        for sp in ("top", "right", "left"):
            a2.spines[sp].set_visible(False)
        a2.set_yticks([])
        fig.suptitle(f"{label}   t = {t[fi]:.2f} s   {LEVEL_NAME[lev]}   |F| = {np.hypot(*F):.0f} N"
                     + (f"   -> {d['outcome']}" if fi >= n - 1 else ""),
                     x=0.02, ha="left", fontsize=9, color=LEVEL_COL[lev] if lev else C["ink"], fontweight="bold")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(path, frames, duration=0.04, loop=0)
