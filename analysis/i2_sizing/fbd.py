"""Free-body-diagram schematics (matplotlib) for nudge and arrest, AP (side view) and ML (front view), recommended design D3.
Arrows are the actual loads from sizing.load_case.  Run: python fbd.py"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import publish as PB  # noqa: E402  (shared rounding policy: loads UP, capacities / SF DOWN)
import sizing as S  # noqa: E402
from run_analysis import CAP_ARREST, CAP_NUDGE, REC, FIG  # noqa: E402

C_FORCE, C_FRAME, C_CABLE, C_N = "#1F6F78", "#C8443A", "#7B5EA7", "#3C78B4"


def arrow(ax, p0, vec, scale, color, label=None, lab_off=(0.02, 0.02), fs=8, lw=2.0):
    v = np.asarray(vec, float) * scale
    if np.hypot(*v) < 0.02:
        return
    ax.annotate("", xy=(p0[0] + v[0], p0[1] + v[1]), xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12))
    if label:
        ax.text(p0[0] + v[0] + lab_off[0], p0[1] + v[1] + lab_off[1], label, color=color, fontsize=fs)


def panel(ax, view, F, phi, cap, title, tmax=1500.0):
    w, lay = REC["walker"], REC["layout"]
    r = S.load_case(w, lay, F, phi, Fz_cap=cap, Tmax=tmax)
    i, j = (0, 2) if view == "AP" else (1, 2)       # horizontal axis index, vertical z
    sc = 0.0010 if F > 200 else 0.0016              # m per N for force arrows
    # ground + base
    xs = w.supports()
    lo, hi = xs[:, i].min() - 0.05, xs[:, i].max() + 0.05
    ax.plot([lo - 0.15, hi + 0.15], [0, 0], color="k", lw=1.5)
    ax.fill_between([lo - 0.15, hi + 0.15], -0.04, 0, color="#ddd")
    ax.add_patch(plt.Rectangle((xs[:, i].min() + (0.0 if view == "AP" else 0.0), 0.08), np.ptp(xs[:, i]), 0.08,
                               fc="#bbb", ec="k", lw=0.8))
    wh = S.Walker.supports(replace_wheels(w))
    for p in wh:
        ax.add_patch(plt.Circle((p[i], 0.075), 0.075, fc="#555", ec="k"))
    for p in xs[len(wh):]:
        ax.plot([p[i], p[i]], [0, 0.08], color="#E0A030", lw=4)
    # columns / top beam
    cols = sorted(set(round(a[i], 3) for a in lay.anchors if a[2] > 1.1))
    ax.plot([min(cols), max(cols)], [1.2, 1.2], color="k", lw=3)
    for c in (min(cols), max(cols)):
        ax.plot([c, c], [0.16, 1.2], color="k", lw=3)
    # patient
    P = S.PELVIS
    ax.plot([P[i], P[i]], [0, P[2]], color="#999", lw=7, alpha=0.6, solid_capstyle="round")
    ax.plot([P[i], P[i]], [P[2], 1.45], color="#999", lw=11, alpha=0.6, solid_capstyle="round")
    ax.add_patch(plt.Circle((P[i], 1.62), 0.09, fc="#bbb", ec="none", alpha=0.7))
    ax.text(P[i], 0.5, "patient\n(on floor)", fontsize=7, ha="center", color="#333")
    # cables + frame reaction forces
    for a, k, T, att in zip(lay.anchors, lay.kinds, r["T"], r["att"]):
        if T < 5:
            ax.plot([att[i], a[i]], [att[2], a[2]], color=C_CABLE, lw=0.6, ls=":", alpha=0.6)
            continue
        ax.plot([att[i], a[i]], [att[2], a[2]], color=C_CABLE, lw=1.0 + T / 70.0)
        u = (att - a) / np.linalg.norm(att - a)
        arrow(ax, (a[i], a[j]), [u[i] * T, u[2] * T], sc, C_FRAME)       # force of cable on frame at the anchor
    ax.plot([P[i]], [P[2]], "o", color=C_FORCE, ms=8)
    Fp = r["Fp"]
    arrow(ax, (P[i], P[2]), [Fp[i], Fp[2]], sc, C_FORCE, None, (0, 0), 8, 2.6)
    tl = ", ".join(f"{k} {PB.req_s(T, 0)}" for k, T in zip(lay.kinds, r["T"]) if T >= 5)
    ax.text(0.98, 0.86, f"pelvis force: {PB.req_s(F, 0)} N horizontal + {PB.req_s(r['Fz_pelvis'], 0)} N lift\ncable tensions [N]\n(others at 2 N pretension):\n{tl}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top", color=C_CABLE,
            bbox=dict(fc="white", ec=C_CABLE, lw=0.5, alpha=0.9))
    # CG weight
    ax.plot([w.cg[i]], [w.cg[2]], "k+", ms=12, mew=2)
    arrow(ax, (w.cg[i], w.cg[2]), [0, -w.m * S.G], 0.00028, "k", f"m_w g = {w.m * S.G:.0f} N", (0.04, 0.14))
    # wheel normals grouped in two halves
    N = r["wheel_N"]
    mid = 0.5 * (xs[:, i].min() + xs[:, i].max())
    for grp, nm in ((xs[:, i] < mid, "N_a"), (xs[:, i] >= mid, "N_b")):
        Ng = np.nansum(N[grp])
        xg = np.mean(xs[grp, i])
        arrow(ax, (xg, -0.0), [0, Ng], 0.00035, C_N, f"{PB.req_s(Ng, 0)} N", (-0.04, 0.10))
    Hv = r["H"]
    ax.text(0.02, 0.98, f"friction needed at floor {PB.req_s(r['H_mag'], 0)} N (cap {PB.cap_s(w.mu * r['N_tot'], 0)} N)\ntip SF {PB.cap_s(r['sf_tip'], 2)} about {r['tip_edge']} edge",
            transform=ax.transAxes, fontsize=7.5, ha="left", va="top", color="#333",
            bbox=dict(fc="white", ec="#999", lw=0.5, alpha=0.9))
    if view == "ML":
        ax.invert_xaxis()
        ax.set_xlabel("y [m]  (left on the left)")
    else:
        ax.set_xlabel("x [m]  (forward to the right)")
    ax.set_ylabel("z [m]")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(lo - 0.15, hi + 0.15) if view == "AP" else ax.set_xlim(hi + 0.15, lo - 0.15)
    ax.set_ylim(-0.1, 2.35)
    ax.set_aspect("equal")


def replace_wheels(w):
    from dataclasses import replace
    return replace(w, ex=0.0, ey=0.0)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    panel(axs[0, 0], "AP", 150, 180, CAP_NUDGE, "NUDGE, AP: 150 N rearward pull on the pelvis (patient drifting forward)", tmax=250)
    panel(axs[0, 1], "AP", 300, 180, CAP_ARREST, "ARREST, AP: 300 N, forward fall (cables pull the pelvis back)")
    panel(axs[1, 0], "ML", 150, 270, CAP_NUDGE, "NUDGE, ML: 150 N rightward pull (patient drifting left)", tmax=250)
    panel(axs[1, 1], "ML", 300, 270, CAP_ARREST, "ARREST, ML: 300 N, leftward fall")
    fig.text(0.5, 0.005, "teal = force on pelvis; red = cable force on frame at anchors; blue = ground normal (two halves); black = walker weight; "
             "orange = deployed outriggers; purple = cables (width ~ tension)", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(FIG, "fig6_fbd.png"), dpi=150)


if __name__ == "__main__":
    main()
