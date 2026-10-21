"""Per-spool slip thresholds for D3: resultant pelvic force when each brake slips at its own tension.
Writes data/slip.json and figures/fig7_slip_resultant.png.  Run after run_analysis.py (imports its design definitions)."""
from __future__ import annotations

import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sizing as S  # noqa: E402
from run_analysis import CAP_ARREST, DESIGNS, FIG, DATA, SF, C_TEAL, C_L2, C_L1, C_BLUE, C_GREY, envelope  # noqa: E402

F_TARGET = float(S.D3C["F_lim"])
R_MAX = float(S.D3C["envelope_SF1p5"])     # SF-1.5 static envelope of D3 (resultant bound), from the contract
R_SF = SF
DIRS = np.arange(0, 360, 1)          # 360 patient-motion directions (1 deg)

d3 = DESIGNS["D3 recommended"]
w, lay = d3["walker"], d3["layout"]
att, U, Lc = S.cable_dirs(lay)
kinds = lay.kinds
EXC = np.array([k == "bws" for k in kinds])
GROUPS = {"corner": [0, 1, 2, 3], "lateral": [4, 5], "centre": [6, 7]}


def metrics(Tfun, label, derr=0.0, statics=True):
    """Tfun(phi_est_deg, d_true_xy) -> per-cable slip tension vector (bws forced slack).  d = patient motion direction."""
    Fpar, Fh, perp, lift, sf, ang, ssl = [], [], [], [], [], [], []
    for dd in DIRS:
        dv = np.array([np.cos(np.radians(dd)), np.sin(np.radians(dd))])
        phi_true = (dd + 180.0) % 360.0
        Ts = Tfun((phi_true + derr) % 360.0, dv)
        F, T = S.slip_resultant(U, Ts, dv, EXC)
        e = -dv
        Fpar.append(F[:2] @ e)
        Fh.append(np.hypot(*F[:2]))
        perp.append(abs(F[0] * e[1] - F[1] * e[0]))
        lift.append(F[2])
        st = S.walker_statics(w, lay, U, T) if statics else dict(sf_tip=np.inf, sf_slide=np.inf)
        sf.append(st["sf_tip"])
        ssl.append(st["sf_slide"])
        ang.append(np.degrees(np.arctan2(perp[-1], max(Fpar[-1], 1e-9))))
    return dict(label=label, Fpar=np.array(Fpar), Fh=np.array(Fh), perp=np.array(perp), lift=np.array(lift),
                sf=np.array(sf), ang=np.array(ang), sf_slide_min=float(min(ssl)),
                Fpar_min=float(min(Fpar)), Fh_max=float(max(Fh)), perp_max=float(max(perp)), lift_max=float(max(lift)),
                sf_min=float(min(sf)), ang_max=float(max(ang)))


_DV = np.stack([np.cos(np.radians(DIRS)), np.sin(np.radians(DIRS))], 1)      # (n_dir, 2) motion directions
_TAUT = ((_DV @ U[:, :2].T) * -1.0 > 0.05) & ~EXC[None, :]                      # same rule as S.taut_mask (u . d < -0.05)


def fast_metrics(Tvec):
    """Vectorised equivalent of metrics() without walker statics, for the threshold grid screen."""
    F = (_TAUT * np.asarray(Tvec)[None, :]) @ U                                  # (n_dir, 3)
    par = -(F[:, :2] * _DV).sum(1)
    h = np.hypot(F[:, 0], F[:, 1])
    perp = np.abs(F[:, 0] * -_DV[:, 1] - F[:, 1] * -_DV[:, 0])
    ang = np.degrees(np.arctan2(perp, np.maximum(par, 1e-9)))
    return dict(Fpar_min=float(par.min()), Fh_max=float(h.max()), lift_max=float(F[:, 2].max()), ang_max=float(ang.max()))


def ok(m):
    return m["Fpar_min"] >= F_TARGET - 1e-6 and m["Fh_max"] <= R_MAX and m["sf_min"] >= R_SF and m["lift_max"] <= CAP_ARREST


def passive(Tvec):
    Tvec = np.array(Tvec, float)
    Tvec[EXC] = 0.0
    return lambda phi, dv: Tvec


def group_vec(tc, tl, tm):
    v = np.zeros(len(U))
    v[GROUPS["corner"]] = tc
    v[GROUPS["lateral"]] = tl
    v[GROUPS["centre"]] = tm
    return v


def qp_table():
    TQ = {}
    for phi in range(0, 360, 5):
        r = S.load_case(w, lay, F_TARGET, phi, Fz_cap=CAP_ARREST)
        TQ[phi] = r["T"].copy()
    return TQ


def main():
    out = {}
    # (a) upper bound: every cable at its slip tension, geometry ignored
    # uniform thresholds
    uni = {}
    for Tu in range(100, 201, 10):
        m = metrics(passive(np.full(len(U), float(Tu))), f"uniform {Tu}")
        uni[Tu] = dict(Fpar_min=m["Fpar_min"], Fh_max=m["Fh_max"], perp_max=m["perp_max"], lift_max=m["lift_max"],
                       sf_min=m["sf_min"], ang_max=m["ang_max"], upper_bound_all_slip=S.slip_upper_bound(U, np.full(len(U), float(Tu)), EXC),
                       ok=ok(m))
    out["uniform"] = uni
    # (b) group-tuned passive thresholds: grid search, objective = smallest resultant spread meeting all constraints
    best = None
    grid = range(40, 221, 5)
    rows = []
    for tc, tl, tm in itertools.product(grid, grid, grid):   # cheap screen first (no walker statics), then the tip SF
        m = fast_metrics(group_vec(tc, tl, tm))
        if m["Fpar_min"] >= F_TARGET and m["Fh_max"] <= R_MAX and m["lift_max"] <= CAP_ARREST:
            rows.append((m["Fh_max"] / m["Fpar_min"] + 0.002 * m["ang_max"], tc, tl, tm))
    rows.sort(key=lambda r: r[0])
    n_screen = len(rows)
    for sc, tc, tl, tm in rows:
        mg = metrics(passive(group_vec(tc, tl, tm)), "g")
        if mg["sf_min"] >= R_SF:
            break
    Tg = group_vec(tc, tl, tm)
    # contract thresholds (what run_analysis' passive-slip coverage uses) evaluated on the same 72 directions
    Tc = S.D3C["T_slip"]
    mc_ = metrics(passive(group_vec(Tc["corner"], Tc["lateral"], Tc["centre"])), "contract")
    out["contract_thresholds"] = dict(T_slip=Tc, Fpar_min=mc_["Fpar_min"], Fh_max=mc_["Fh_max"], perp_max=mc_["perp_max"],
                                      lift_max=mc_["lift_max"], sf_min=mc_["sf_min"], sf_slide_min=mc_["sf_slide_min"], ang_max=mc_["ang_max"], ok=ok(mc_),
                                      equals_grid_search=(dict(corner=tc, lateral=tl, centre=tm) == {k: int(v) for k, v in Tc.items()}))
    out["group_tuned"] = dict(T_slip=dict(corner=tc, lateral=tl, centre=tm), n_feasible=n_screen, n_directions=len(DIRS), Fpar_min=mg["Fpar_min"], Fh_max=mg["Fh_max"],
                              perp_max=mg["perp_max"], lift_max=mg["lift_max"], sf_min=mg["sf_min"], sf_slide_min=mg["sf_slide_min"], ang_max=mg["ang_max"],
                              upper_bound_all_slip=S.slip_upper_bound(U, Tg, EXC))
    # (c) controller-set thresholds from the QP at 300 N for the ESTIMATED direction, with direction error
    TQ = qp_table()
    def ctrl(phi, dv):
        T = TQ[int(round(phi / 5.0)) * 5 % 360].copy()
        T[EXC] = 0.0
        return T
    ctl = {}
    for derr in (0, 15, 30, 45):
        m_pos = metrics(ctrl, "ctl", derr)
        m_neg = metrics(ctrl, "ctl", -derr)
        ctl[derr] = dict(Fpar_min=min(m_pos["Fpar_min"], m_neg["Fpar_min"]), Fh_max=max(m_pos["Fh_max"], m_neg["Fh_max"]),
                         perp_max=max(m_pos["perp_max"], m_neg["perp_max"]), sf_min=min(m_pos["sf_min"], m_neg["sf_min"]),
                         lift_max=max(m_pos["lift_max"], m_neg["lift_max"]))
    out["controller_qp"] = ctl
    # (d) hybrid: spring-applied passive floor = group-tuned thresholds, electromagnet may only LOWER them per direction
    def hyb(phi, dv):
        return np.minimum(Tg, ctrl(phi, dv))
    hy = {}
    for derr in (0, 15, 30, 45):
        a = metrics(hyb, "h", derr)
        b = metrics(hyb, "h", -derr)
        hy[derr] = dict(Fpar_min=min(a["Fpar_min"], b["Fpar_min"]), Fh_max=max(a["Fh_max"], b["Fh_max"]),
                        perp_max=max(a["perp_max"], b["perp_max"]), sf_min=min(a["sf_min"], b["sf_min"]),
                        lift_max=max(a["lift_max"], b["lift_max"]))
    out["hybrid"] = hy
    # per-spool brake table
    rs = 0.025
    tab = {}
    for g, idx in GROUPS.items():
        Ts = float(Tg[idx[0]])
        tab[g] = dict(cables=[kinds[i] for i in idx], T_slip=Ts, slip_torque=Ts * rs, rated_hold_torque=1.5 * Ts * rs)
    Tqmax = float(max(np.max(TQ[p][~EXC]) for p in TQ))
    tab["controller (current-set) brake rating"] = dict(T_max_qp=Tqmax, rated_hold_torque=1.5 * Tqmax * rs)
    hang = 1.5 * 85 * S.G
    tab["bws (hang only, kept slack in horizontal arrest)"] = dict(T_hold=hang, rated_hold_torque=1.5 * hang * rs)
    out["brake_table_rs0.025"] = tab
    json.dump(out, open(os.path.join(DATA, "slip.json"), "w"), indent=1, default=float)
    # figure
    mu = metrics(passive(np.full(len(U), 150.0)), "uniform 150")
    mc = metrics(ctrl, "controller, no direction error")
    mce = metrics(ctrl, "controller, 30 deg error", 30.0)
    mgp = metrics(passive(Tg), f"group-tuned passive")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for m, col, ls in ((mu, C_L1, "-"), (mgp, C_TEAL, "-"), (mc, C_BLUE, "--"), (mce, C_L2, ":")):
        ax[0].plot(DIRS, m["Fpar"], color=col, ls=ls, lw=2, label=m["label"])
        ax[1].plot(DIRS, m["Fh"], color=col, ls=ls, lw=2, label=m["label"])
    ax[0].axhline(F_TARGET, color="k", lw=1)
    ax[1].axhline(R_MAX, color="k", lw=1, ls="--")
    ax[0].set_title("Force along the arrest direction (need >= 300 N)", fontsize=10)
    ax[1].set_title(f"Horizontal resultant magnitude (must stay <= {R_MAX:.0f} N)", fontsize=10)
    for a_ in ax:
        a_.set_xlabel("patient motion direction [deg from +x, toward +y]")
        a_.set_ylabel("force on pelvis [N]")
        a_.grid(alpha=0.25)
        a_.legend(fontsize=8)
        a_.set_xlim(0, 360)
    ax[0].set_ylim(100, 480)
    ax[1].set_ylim(100, 480)
    ax[0].text(5, 305, "300 N target", fontsize=8)
    ax[1].text(5, R_MAX + 5, f"{R_MAX:.0f} N = SF 1.5 envelope", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_slip_resultant.png"), dpi=160)
    plt.close(fig)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
