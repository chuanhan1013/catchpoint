"""Run the I2 sizing study: load cases from the sim, design sweeps, figures, tables -> data/results.json.

Run:  .venv/Scripts/python analysis/i2_sizing/run_analysis.py      (needs data/arrest_rows_*.json from instrument_sim.py)
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sizing as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D3C = S.D3C
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)

# project palette (SPEC section 7)
C_L0, C_L1, C_L2 = "#6B7A8F", "#E0A030", "#C8443A"
C_TEAL, C_VIOLET, C_BLUE, C_GREY = "#1F6F78", "#7B5EA7", "#3C78B4", "#9AA0A6"

SF = 1.5                          # static tip safety factor used for the allowable envelope
SLIDE_TOL = 0.01                  # m: walker slide counted as 'zero' for the coverage metric (state this bound)
XI_ARREST, XI_NUDGE = 0.5, 0.15   # max pelvis lift (fraction of 60 kg body weight) in arrest / nudge
MP_MIN, MP_MAX = 60.0, 85.0
CAP_ARREST = XI_ARREST * MP_MIN * S.G
CAP_NUDGE = XI_NUDGE * MP_MIN * S.G
F_NUDGE = 150.0

# ------------------------------------------------------------------------------------------------ designs
DESIGNS = {
    "D0 as proposed": dict(
        walker=S.Walker(m=60, cg=(0.40, 0, 0.45)), layout=S.make_layout(6, z_hi=0.95),
        note="60 kg, 6 cables all at pelvis height (horizontal pull), no outriggers"),
    "D1 high anchors": dict(
        walker=S.Walker(m=60, cg=(0.40, 0, 0.45)), layout=S.make_layout(8, z_hi=1.20, z_lat=0.95),
        note="60 kg, 4 corner anchors on the 1.20 m columns/top beam, 4 mid anchors at 0.95 m"),
    "D2 + ballast": dict(
        walker=S.Walker(m=100, cg=(0.45, 0, 0.35)), layout=S.make_layout(8, z_hi=1.20, z_lat=0.95),
        note="D1 layout, 100 kg walker (40 kg low ballast), no outriggers"),
    "D3 recommended": dict(
        walker=S.d3_walker(), layout=S.d3_layout(),     # geometry from data/d3_contract.json (single source of truth)
        note="80 kg (20 kg low ballast), outrigger feet deployed on catch (+0.25 m all round), D1 layout + overhead BWS cable"),
}
DCOL = {"D0 as proposed": C_L2, "D1 high anchors": C_L1, "D2 + ballast": C_BLUE, "D3 recommended": C_TEAL}
REC = DESIGNS["D3 recommended"]

# fall direction (patient COM moving) -> pelvis force direction phi (cables pull the pelvis AGAINST the motion)
PHI_FALL = {"forward (+x)": 180, "backward (-x)": 0, "left (+y)": 270, "right (-y)": 90}


def load_rows(tag):
    rows = json.load(open(os.path.join(DATA, f"arrest_rows_{tag}.json")))
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    a["omega"] = np.sqrt(S.G / a["z0"])
    a["Freq"] = S.required_force(a["m"], a["omega"], a["s0"], a["vout"])
    a["phi"] = (a["dir_deg"] + 180.0) % 360.0
    return a


def stats3(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))


# ------------------------------------------------------------------------------------------------ A. loads
def fig_loads(rows):
    a = rows["280"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax = ax.ravel()
    def h(axx, x, xl, col, ref=None, rl=None, bins=30, clip=None):
        xx = np.clip(x, None, clip) if clip else x
        axx.hist(xx, bins=bins, color=col, alpha=0.85)
        axx.set_xlabel(xl)
        axx.set_ylabel("caught trials")
        for q, ls in ((50, "-"), (95, "--")):
            axx.axvline(np.percentile(x, q), color="k", ls=ls, lw=1)
        axx.text(0.97, 0.95, f"median {np.median(x):.3g}\np95 {np.percentile(x, 95):.3g}\nmax {np.max(x):.3g}",
                 transform=axx.transAxes, ha="right", va="top", fontsize=9)
        if ref:
            axx.axvline(ref, color=C_L2, lw=2)
            axx.text(ref, axx.get_ylim()[1] * 0.55, rl, color=C_L2, rotation=90, va="center", ha="right", fontsize=9)
    h(ax[0], a["vout"], "outward COM speed at engagement [m/s]", C_TEAL)
    h(ax[1], a["s0"] * 100, "COM distance from support at engagement [cm]", C_TEAL)
    h(ax[2], a["ke"], "COM kinetic energy at engagement [J]", C_TEAL)
    h(ax[3], a["F_peak"], "sim catch-force peak (K,D spring-damper) [N]", C_L1, 491, "491 N (sim max)")
    h(ax[4], np.clip(a["Freq"], 0, 600), "required constant arrest force [N] (stop within 0.25 m)", C_BLUE)
    ax[4].text(0.5, 0.5, f"{(a['Freq'] > 600).sum()} trials > 600 N\n(s0 close to 0.25 m)", transform=ax[4].transAxes, fontsize=8)
    h(ax[5], a["F_hold"], "sustained hold force at t = 3 s [N]", C_L0)
    fig.suptitle("Arrest load cases from the CatchPoint Monte Carlo (CP m=0.03, 814 caught of 2000 trials, lock delay 0.28 s)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_arrest_loads.png"), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ B. tip margin
def sweep_tip(design, Fs, step=15, Fz_cap=CAP_ARREST):
    w, lay = design["walker"], design["layout"]
    sf_min, feas = [], []
    for F in Fs:
        best, ok = np.inf, True
        for phi in range(0, 360, step):
            r = S.load_case(w, lay, F, phi, Fz_cap=Fz_cap)
            if not r["feasible"]:
                ok = False
                break
            best = min(best, r["sf_tip"])
        sf_min.append(best if ok else np.nan)
        feas.append(ok)
    return np.array(sf_min), np.array(feas)


def fig_tip(res):
    Fs = np.arange(25, 626, 25.0)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name, d in DESIGNS.items():
        sf, ok = sweep_tip(d, Fs)
        res["tip_curve"][name] = dict(F=Fs.tolist(), sf=[None if np.isnan(v) else float(v) for v in sf])
        sfp = np.minimum(sf, 5.0)
        ax.plot(Fs, sfp, "-o", ms=3.5, color=DCOL[name], lw=2.2 if "D3" in name else 1.6, label=name)
        if (~ok).any():
            f0 = Fs[~ok][0]
            ax.plot([f0], [0.12], marker="X", ms=9, color=DCOL[name], ls="none")
            ax.annotate("cable set cannot\nhold lift cap", (f0, 0.12), (f0 + 10, 0.5), fontsize=8, color=DCOL[name],
                        arrowprops=dict(arrowstyle="-", color=DCOL[name]))
    ax.axhline(1.0, color="k", lw=1)
    ax.axhline(SF, color="k", lw=1, ls="--")
    ax.text(628, 0.9, "SF = 1 (wheel lifts)", ha="right", fontsize=8)
    ax.text(615, SF + 0.03, f"SF = {SF} (design margin)", ha="right", fontsize=8)
    for x, t in ((F_NUDGE, "nudge\n150 N"), (300, "300 N"), (491, "491 N\n(sim max)")):
        ax.axvline(x, color=C_GREY, ls=":", lw=1.2)
        ax.text(x + 4, 4.55, t, fontsize=8, color="#555")
    ax.set_xlim(0, 630)
    ax.set_ylim(0, 5)
    ax.set_xlabel("horizontal arrest force on the pelvis [N] (static, worst of 24 directions)")
    ax.set_ylabel("static tip safety factor (load multiple that lifts a wheel)")
    ax.set_title("60 kg + horizontal cables tips at ~200 N; D3 holds >400 N at SF 1.5",
                 fontsize=9.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_tip_margin.png"), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ C. mass vs width
def scaled_walker(base, hw):
    s = hw / 0.325
    xc = 0.475
    return replace(base, wy=hw, wx=(xc - 0.4 * s, xc + 0.4 * s))


def fig_mass(res):
    hws = np.arange(0.30, 0.801, 0.05)
    cases = [("horizontal cables (z = 0.95 m)", DESIGNS["D0 as proposed"]["layout"]),
             ("high corner anchors (1.20 m) + mid anchors", DESIGNS["D1 high anchors"]["layout"])]
    base = S.Walker(m=60, cg=(0.45, 0, 0.40))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    out = {}
    sty = {(491, 1.0): ("-", 2.0), (491, SF): ("--", 1.4), (300, SF): (":", 2.0)}
    cols = [C_L2, C_TEAL]
    for (cn, lay), col in zip(cases, cols):
        for (F, sf), (ls, lw) in sty.items():
            ms = []
            for hw in hws:
                ms.append(S.required_mass(lay, F, scaled_walker(base, hw), sf=sf, step=30, Fz_cap=CAP_ARREST))
            ms = np.array(ms)
            out[f"{cn}|F={F}|SF={sf}"] = dict(hw=hws.tolist(), mass=[None if not np.isfinite(m) else float(m) for m in ms])
            okm = np.isfinite(ms)
            ax.plot(hws[okm], ms[okm], ls=ls, lw=lw, color=col, label=f"{cn}: {F} N, SF {sf:g}")
    ax.axhline(60, color=C_GREY, lw=1)
    ax.text(0.31, 62, "60 kg (review's assumed walker)", fontsize=8, color="#555")
    ax.plot([0.4], [119], marker="*", ms=13, color="k", ls="none")
    ax.annotate("review's hand estimate:\n119 kg at 0.40 m, 491 N,\nhorizontal, SF 1", (0.4, 119), (0.5, 175), fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    ax.axvline(0.325, color=C_L0, ls="--", lw=1)
    ax.text(0.328, 395, "SPEC wheel\nhalf-width 0.325", fontsize=8, color=C_L0)
    ax.set_xlim(0.3, 0.8)
    ax.set_ylim(0, 420)
    ax.set_xlabel("base half-width between wheel (or outrigger) contacts [m]  (AP wheelbase scaled in proportion)")
    ax.set_ylabel("walker mass for a static tip SF, any direction [kg]")
    ax.set_title("Required walker mass to hold the arrest force statically, vs base width", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_mass_vs_width.png"), dpi=160)
    plt.close(fig)
    res["mass_curves"] = out


# ------------------------------------------------------------------------------------------------ D. tables
def envelope(design, sf=SF, step=15, Fz_cap=CAP_ARREST, Fcap=800.0):
    al = S.max_arrest_force(design["walker"], design["layout"], sf=sf, Fz_cap=Fz_cap, step=step, Fcap=Fcap)
    v = np.array(list(al.values()))
    imin = int(np.argmin(v))
    return dict(min=float(v.min()), argmin_phi=list(al.keys())[imin], mean=float(v.mean()), per_dir={int(k): float(x) for k, x in al.items()})


def slide_limit(design, F=300.0, mu=None):
    """Horizontal force at which the braked walker starts to slide: F = mu (m g + kappa F)."""
    w = design["walker"] if mu is None else replace(design["walker"], mu=mu)
    kap = []
    for phi in range(0, 360, 30):
        r = S.load_case(w, design["layout"], F, phi, Fz_cap=CAP_ARREST)
        if r["feasible"]:
            kap.append(r["Fz_down_on_frame"] / F)
    k = float(np.min(kap))
    den = 1 - w.mu * k
    return (w.mu * w.m * S.G / den if den > 0 else np.inf), k


def cable_table(design, F_arr, res):
    w, lay = design["walker"], design["layout"]
    names = lay.kinds
    rows = []
    for label, phi in PHI_FALL.items():
        r = S.load_case(w, lay, F_arr, phi, Fz_cap=CAP_ARREST)
        rows.append(dict(case=f"arrest {F_arr:.0f} N, fall {label}", phi=phi, T=dict(zip(names, np.round(r["T"], 1).tolist())),
                         Tmax=float(r["Tmax_cable"]), lift=float(r["Fz_pelvis"]), sf_tip=float(r["sf_tip"]), edge=r["tip_edge"],
                         Fz_down=float(r["Fz_down_on_frame"]), H=float(r["H_mag"]), sf_slide=float(r["sf_slide"]),
                         N_tot=float(r["N_tot"]), margin=float(r["margin_min"]),
                         wheel_N=np.round(r["wheel_N"], 0).tolist()))
    # worst direction over 24
    worst = None
    for phi in range(0, 360, 15):
        r = S.load_case(w, lay, F_arr, phi, Fz_cap=CAP_ARREST)
        if worst is None or r["Tmax_cable"] > worst["Tmax_cable"]:
            worst = r
            worst["phi"] = phi
    res["Tmax_arrest_worst"] = float(worst["Tmax_cable"])
    res["Tmax_arrest_worst_phi"] = int(worst["phi"])
    rows_n = []
    Tn_worst = 0.0
    for phi in range(0, 360, 15):
        r = S.load_case(w, lay, F_NUDGE, phi, Fz_cap=CAP_NUDGE, Tmax=250.0)
        if not r["feasible"]:
            rows_n.append(dict(phi=phi, feasible=False))
            continue
        Tn_worst = max(Tn_worst, r["Tmax_cable"])
        rows_n.append(dict(phi=phi, feasible=True, Tmax=float(r["Tmax_cable"]), lift=float(r["Fz_pelvis"]),
                           sf_tip=float(r["sf_tip"]), Fz_down=float(r["Fz_down_on_frame"]), H=float(r["H_mag"]),
                           sf_slide=float(r["sf_slide"]), T=dict(zip(names, np.round(r["T"], 1).tolist()))))
    res["nudge_rows"] = rows_n
    res["Tmax_nudge_worst"] = float(Tn_worst)
    res["arrest_rows"] = rows
    res["spool_brake"] = {f"r_s={rs}": S.spool_brake(worst["Tmax_cable"], Tn_worst, r_s=rs) for rs in (0.02, 0.025, 0.03)}


def hang_case(res):
    """Patient hanging in the harness (BWS cable only): 1.5 x 85 kg dynamic load, anchor above the pelvis vs at the
    SPEC rear beam."""
    out = {}
    w = REC["walker"]
    Fv = 1.5 * MP_MAX * S.G
    for label, xb in (("overhead x=0.42 (cantilevered top beam)", 0.42), ("SPEC rear beam x=0.03", 0.03)):
        for wname, ww in (("D3 recommended", w), ("D1 60 kg", DESIGNS["D1 high anchors"]["walker"])):
            lay = S.make_layout(4, z_hi=1.2, bws=True, x_bws=xb, z_bws=(1.80 if xb > 0.3 else 1.20))  # D3 cantilever 1.80; SPEC rear beam 1.20
            lay = S.Layout("bws-only", lay.anchors[-1:], ["bws"])
            att, U, L = S.cable_dirs(lay)
            T = np.array([Fv / U[0, 2]])
            st = S.walker_statics(ww, lay, U, T)
            out[f"{label} | {wname}"] = dict(T=float(T[0]), H=float(st["H_mag"]), Fz_down=float(st["Fz_down_on_frame"]),
                                              sf_tip=float(st["sf_tip"]), edge=st["tip_edge"], sf_slide=float(st["sf_slide"]),
                                              margin=float(st["margin_min"]))
    res["hang"] = out


# ------------------------------------------------------------------------------------------------ E. arrest dynamics
def trial_dyn(rows, design, F_lim, sf_env, k_c=15000.0, tag="280", mu=None, dyn_tip=True):
    a = rows[tag]
    w = design["walker"] if mu is None else replace(design["walker"], mu=mu)
    lay = design["layout"]
    n = len(a["m"])
    # per-trial friction coupling (kappa) and tipping reference moment, cached by direction (15 deg grid)
    phi_q = (np.round(a["phi"] / 15.0) * 15.0) % 360
    cache = {}
    Fref = max(float(F_lim), 50.0)
    for ph in np.unique(phi_q):
        r = S.load_case(w, lay, Fref, ph, Fz_cap=CAP_ARREST)
        if not r["feasible"]:
            cache[ph] = dict(kappa=0.0, M=0.0, a=0.4, sf_ref=0.0, feas=False)
            continue
        edge = r["tip_edge"]
        M, ae = S.edge_moment(w, lay, r["U"], r["T"], edge)
        cache[ph] = dict(kappa=r["Fz_down_on_frame"] / Fref, M=M, a=ae, sf_ref=r["sf_tip"], feas=True)
    kappa = np.array([cache[p]["kappa"] for p in phi_q])
    out = S.arrest_1d(a["m"], a["omega"], a["s0"], a["vout"], F_lim, w.m, w.mu, kappa, k_c=k_c)
    tipped = np.zeros(n, bool)
    if dyn_tip:
        for i in range(n):
            c = cache[phi_q[i]]
            hist = out["F_hist"][:, i]
            if hist.max() < 1.0 or not c['feas']:
                continue
            tp, _, _ = S.dynamic_tip(w, c["M"], Fref, hist, out["dt_hist"], c["a"])
            tipped[i] = tp
    # static envelope check at peak force (SF)
    stat_ok = np.array([out["F_peak"][i] * sf_env <= Fref * cache[phi_q[i]]["sf_ref"] for i in range(n)])
    feas = np.array([cache[p]['feas'] for p in phi_q])
    return dict(ok=out["ok"], tipped=tipped, stat_ok=stat_ok & feas, F_peak=out["F_peak"], s_max=out["s_max"], t_stop=out["t_stop"],
                slide=out["walker_slide"], feasible=feas,
                good_dynamic=out["ok"] & ~tipped & feas,
                good=out["ok"] & ~tipped & feas & stat_ok & feas & (out["walker_slide"] <= SLIDE_TOL))


# ------------------------------------------------------------------------------------------------ E2. passive-slip coverage
SLIP_DIR_RES = 0.5       # deg: fall directions are binned to this resolution for the (expensive) statics


def passive_slip_table(design, T_slip, dirs_deg):
    """For each fall direction (patient COM motion direction, deg from +x toward +y) the force the group-tuned
    passive slip brakes produce on the pelvis: every cable whose length grows along the motion sits at its own slip
    tension (BWS cable slack).  Returns per-direction dicts: Fpar (along the arrest direction), Fh (horizontal
    resultant magnitude), perp, lift, sf_tip / edge (statics of the actual cable loads), kappa (downward frame load
    per N of Fpar), hscale (= Fh / Fpar), M / a (overturning moment about the tip edge and lever arm, for the
    dynamic tip check)."""
    w, lay = design["walker"], design["layout"]
    _, U, _ = S.cable_dirs(lay)
    exc = np.array([k == "bws" for k in lay.kinds])
    Tvec = S.group_slip_vector(lay, T_slip)
    tab = {}
    for dd in np.unique(np.round(np.asarray(dirs_deg) / SLIP_DIR_RES) * SLIP_DIR_RES % 360):
        dv = np.array([np.cos(np.radians(dd)), np.sin(np.radians(dd))])
        F, T = S.slip_resultant(U, Tvec, dv, exc)
        e = -dv
        fpar = float(F[:2] @ e)
        fh = float(np.hypot(*F[:2]))
        st = S.walker_statics(w, lay, U, T)
        M, a = S.edge_moment(w, lay, U, T, st["tip_edge"])
        tab[float(dd)] = dict(Fpar=fpar, Fh=fh, perp=float(abs(F[0] * e[1] - F[1] * e[0])), lift=float(F[2]),
                              sf_tip=float(st["sf_tip"]), edge=st["tip_edge"], kappa=float(st["Fz_down_on_frame"]) / fpar,
                              hscale=fh / fpar, M=M, a=a)
    return tab


def trial_dyn_passive(rows, design, T_slip, sf_env=SF, k_c=15000.0, tag="280"):
    """Coverage with the force the passive slip brakes actually produce for each trial's fall direction.
    1-D stopping model: the brake slips at the ALONG-arrest component Fpar(direction).  Static tip / slide / dynamic
    tip use the FULL resultant (incl. the perpendicular component): friction load = |H| (hscale), tip statics of the
    actual cable tensions, scaled by F_peak / Fpar when the brake did not reach its slip level."""
    a = rows[tag]
    w = design["walker"]
    n = len(a["m"])
    tab = passive_slip_table(design, T_slip, a["dir_deg"])
    key = np.round(np.asarray(a["dir_deg"]) / SLIP_DIR_RES) * SLIP_DIR_RES % 360

    def g(f):
        return np.array([tab[float(k)][f] for k in key])

    Fpar, Fh, perp, kappa, hsc, sf_tip = g("Fpar"), g("Fh"), g("perp"), g("kappa"), g("hscale"), g("sf_tip")
    out = S.arrest_1d(a["m"], a["omega"], a["s0"], a["vout"], Fpar, w.m, w.mu, kappa, k_c=k_c, hscale=hsc)
    tipped = np.zeros(n, bool)
    for i in range(n):
        hist = out["F_hist"][:, i]
        if hist.max() < 1.0:
            continue
        c = tab[float(key[i])]
        tipped[i], _, _ = S.dynamic_tip(w, c["M"], Fpar[i], hist, out["dt_hist"], c["a"])
    stat_ok = out["F_peak"] * sf_env <= Fpar * sf_tip
    slid_ok = out["walker_slide"] <= SLIDE_TOL
    good = out["ok"] & ~tipped & stat_ok & slid_ok
    return dict(good=good, ok=out["ok"], tipped=tipped, stat_ok=stat_ok, slid_ok=slid_ok, Fpar=Fpar, Fh=Fh, perp=perp,
                F_peak=out["F_peak"], slide=out["walker_slide"], s_max=out["s_max"], t_stop=out["t_stop"], sf_tip=sf_tip,
                F_peak_res=out["F_peak"] * hsc, energy=Fpar * out["pay"])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def run_dynamics(rows, res):
    res["dyn"] = {}
    sweepF = [100, 150, 200, 250, 300, 350, 400, 491]
    curves = {}
    for tag in ("280", "150", "080"):
        n = len(rows[tag]["m"])
        for dname in ("D0 as proposed", "D3 recommended"):
            d = DESIGNS[dname]
            gl = []
            for F in sweepF:
                r = trial_dyn(rows, d, F, SF, tag=tag, dyn_tip=True)
                kk = int(r["good"].sum()); nn = len(r["good"]); lo_, hi_ = wilson(kk, nn)
                gl.append(dict(F_lim=F, k=kk, n=nn, ci_lo=lo_, ci_hi=hi_, good_dynamic=float(r["good_dynamic"].mean()), ok=float(r["ok"].mean()), good=float(r["good"].mean()), tipped=float(r["tipped"].mean()),
                               static_sf1=float(r["stat_ok"].mean()), slide_p95=float(np.percentile(r["slide"], 95)),
                               slide_max=float(r["slide"].max()), smax_p95=float(np.percentile(r["s_max"][r["ok"]], 95))))
            curves[f"{tag}|{dname}"] = gl
    res["dyn"]["curves"] = curves
    # detail for recommended design at F_lim = 300, lock 0.28
    det = {}
    for F in (300,):
        r = trial_dyn(rows, REC, F, SF, tag="280")
        kk = int(r["good"].sum()); nn = len(r["good"])
        det[F] = dict(k=kk, n=nn, ci=wilson(kk, nn), k_dynamic_only=int(r["good_dynamic"].sum()), ok=float(r["ok"].mean()), good=float(r["good"].mean()), tipped=float(r["tipped"].mean()),
                      F_peak=stats3(r["F_peak"]), s_max=stats3(r["s_max"][r["ok"]]), slide=stats3(r["slide"]),
                      t_stop=stats3(r["t_stop"][np.isfinite(r["t_stop"])]), n_fail=int((~r["good"]).sum()))
    res["dyn"]["rec_detail"] = det
    # HEADLINE: passive-slip coverage (force = what the group-tuned passive brakes give for each trial's fall direction)
    pas = {}
    for tag in ("280", "150", "080"):
        r = trial_dyn_passive(rows, REC, D3C["T_slip"], tag=tag)
        kk, nn = int(r["good"].sum()), len(r["good"])
        lo_, hi_ = wilson(kk, nn)
        old = curves[f"{tag}|D3 recommended"][sweepF.index(300)]
        pas[tag] = dict(k=kk, n=nn, pct=100 * kk / nn, ci_lo=lo_, ci_hi=hi_,
                        fixed300_k=old["k"], fixed300_n=old["n"], fixed300_pct=100 * old["k"] / old["n"],
                        fixed300_ci_lo=old["ci_lo"], fixed300_ci_hi=old["ci_hi"],
                        n_not_arrested=int((~r["ok"]).sum()), n_dyn_tip=int(r["tipped"].sum()), n_static_fail=int((~r["stat_ok"]).sum()),
                        n_slide_fail=int((~r["slid_ok"]).sum()), n_fail=int((~r["good"]).sum()),
                        Fpar=stats3(r["Fpar"]), Fpar_min=float(r["Fpar"].min()), Fh=stats3(r["Fh"]), perp=stats3(r["perp"]),
                        F_peak=stats3(r["F_peak"]), t_stop=stats3(r["t_stop"][np.isfinite(r["t_stop"]) & r["ok"]]), F_peak_resultant=stats3(r["F_peak_res"]), slide=stats3(r["slide"]),
                        sf_tip_min=float(r["sf_tip"].min()), s_max=stats3(r["s_max"][r["ok"]]),
                        brake_energy_max_J=float(r["energy"][r["good"]].max()))
    res["dyn"]["passive"] = pas
    res["dyn"]["passive_params"] = dict(T_slip=D3C["T_slip"], dir_res_deg=SLIP_DIR_RES, SF=SF, slide_tol=SLIDE_TOL)
    # D0 at the sim's own 491 N, rigid lock
    r0 = trial_dyn(rows, DESIGNS["D0 as proposed"], 491, SF, tag="280")
    res["dyn"]["D0_491"] = dict(k=int(r0["good"].sum()), n=len(r0["good"]), tipped=float(r0["tipped"].mean()), good=float(r0["good"].mean()), slide=stats3(r0["slide"]))
    # figure: fraction of caught trials arrested (no fall, no tip) vs F_lim
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for tag, ls, lw in (("280", "-", 2.2), ("150", "--", 1.5), ("080", ":", 1.5)):
        for dname, col in (("D0 as proposed", C_L2), ("D3 recommended", C_TEAL)):
            g = curves[f"{tag}|{dname}"]
            ax[0].plot([x["F_lim"] for x in g], [100 * x["good"] for x in g], ls=ls, lw=lw, color=col, marker="o", ms=3.5,
                       label=f"{dname}, lock {int(tag) / 1000:.2f} s")
    for tag, mk in (("280", "D"), ("150", "s"), ("080", "^")):
        p_ = pas[tag]
        ax[0].plot([p_["Fpar"]["median"]], [p_["pct"]], marker=mk, ms=8, color=C_TEAL, mec="k", ls="none",
                   label=f"D3 passive slip brakes (lock {int(tag) / 1000:.2f} s)")
    ax[0].set_xlabel("brake slip (force-limit) setting F_lim [N]  (markers: passive slip, at its median along-arrest force)")
    ax[0].set_ylabel("caught trials arrested, SF>=1.5, slide<=1 cm [%]")
    ax[0].set_title("Share of simulated catches the device can arrest", fontsize=10)
    ax[0].set_ylim(0, 101)
    ax[0].legend(fontsize=7.5, loc="lower right")
    ax[0].grid(alpha=0.25)
    # right: required force CDF
    for tag, ls in (("280", "-"), ("150", "--"), ("080", ":")):
        F = np.sort(rows[tag]["Freq"])
        ax[1].plot(np.minimum(F, 700), 100 * np.arange(1, len(F) + 1) / len(F), ls=ls, color=C_BLUE, lw=2 if tag == "280" else 1.4,
                   label=f"lock delay {int(tag) / 1000:.2f} s")
    sf_env = envelope(DESIGNS["D3 recommended"])["min"]
    ax[1].axvline(sf_env, color=C_TEAL, lw=1.5)
    ax[1].text(sf_env - 5, 55, f"D3 static envelope\n(SF {SF}) {sf_env:.0f} N", color=C_TEAL, ha="right", fontsize=8)
    ax[1].axvline(491, color=C_L2, lw=1.5)
    ax[1].text(486, 55, "491 N", color=C_L2, ha="right", fontsize=8)
    ax[1].axvline(300, color=C_GREY, ls=":")
    ax[1].set_xlim(0, 700)
    ax[1].set_ylim(40, 101)
    ax[1].set_xlabel("force needed to stop the COM within 0.25 m of support [N]")
    ax[1].set_ylabel("cumulative share of caught trials [%]")
    ax[1].set_title("Arrest force actually required (closed form, rigid cable)", fontsize=10)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_arrest_coverage.png"), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ F. sensitivity
def sensitivity(res):
    base = REC
    out = {}
    def env_of(walker=None, layout=None, P=S.PELVIS, cap=CAP_ARREST, sf=SF):
        w = walker or base["walker"]
        l = layout or base["layout"]
        al = S.max_arrest_force(w, l, sf=sf, P=P, Fz_cap=cap, step=30, Fcap=800)
        return float(min(al.values()))
    out["nominal (SF 1.5)"] = env_of()
    out["SF 1.0"] = env_of(sf=1.0)
    out["pelvis 0.10 m further rearward"] = env_of(P=S.PELVIS + np.array([-0.10, 0, 0]))
    out["pelvis 0.10 m further forward"] = env_of(P=S.PELVIS + np.array([0.10, 0, 0]))
    out["pelvis 0.10 m to the left"] = env_of(P=S.PELVIS + np.array([0, 0.10, 0]))
    out["pelvis 0.15 m lower (0.80)"] = env_of(P=S.PELVIS + np.array([0, 0, -0.15]))
    cgx, _, cgz = base["walker"].cg
    out["walker CG 0.10 m higher"] = env_of(walker=replace(base["walker"], cg=(cgx, 0, cgz + 0.10)))
    out["walker CG 0.10 m rearward"] = env_of(walker=replace(base["walker"], cg=(cgx - 0.10, 0, cgz)))
    out["lift cap 0.3 BW (60 kg)"] = env_of(cap=0.3 * MP_MIN * S.G)
    out["lift cap 0.8 BW (60 kg)"] = env_of(cap=0.8 * MP_MIN * S.G)
    out["outriggers not deployed"] = env_of(walker=replace(base["walker"], ex=0.0, ey=0.0))
    out["outriggers +0.15 m"] = env_of(walker=replace(base["walker"], ex=0.15, ey=0.15))
    out["walker 60 kg (CG 0.45 m)"] = env_of(walker=replace(base["walker"], m=60, cg=(cgx, 0, 0.45)))
    out["walker 100 kg"] = env_of(walker=replace(base["walker"], m=100, cg=(cgx, 0, 0.38)))
    out["6 cables (no centre anchors)"] = env_of(layout=S.d3_layout(6))
    out["4 cables (corners only)"] = env_of(layout=S.d3_layout(4))
    out["all 8 anchors at 1.20 m"] = env_of(layout=S.d3_layout(8, z_lat=1.2, z_c=1.2))
    out["corners 1.50 m, mid 0.95 m"] = env_of(layout=S.d3_layout(8, z_hi=1.5))
    res["sensitivity"] = out
    # slide limits
    sl = {}
    for mu in (0.3, 0.5, 0.7):
        f, k = slide_limit(base, mu=mu)
        sl[f"mu={mu}"] = dict(F_slide=float(f), kappa=k)
    res["slide_limit"] = sl


# ------------------------------------------------------------------------------------------------ G. timing
def fig_timing(rows, res):
    a = rows["280"]
    d = trial_dyn_passive(rows, REC, D3C["T_slip"], tag="280")      # passive slip brakes, lock 0.28 s
    ts = d["t_stop"][np.isfinite(d["t_stop"]) & d["ok"]]
    t_stop_med, t_stop_p95 = float(np.median(ts)), float(np.percentile(ts, 95))
    res["t_stop"] = dict(median=t_stop_med, p95=t_stop_p95)
    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.axvspan(0.3, 0.5, color=C_L2, alpha=0.12)
    ax.text(0.4, 5.55, "0.3-0.5 s fall-development window", ha="center", fontsize=9, color=C_L2)
    def bar(y, t0, dt, col, label, txtcol="white", lw=0):
        ax.barh(y, dt, left=t0, height=0.5, color=col, edgecolor="k", lw=0.6)
        if label:
            ax.text(t0 + dt / 2, y, label, ha="center", va="center", fontsize=7.5, color=txtcol)
    # row 1: SPEC catch chain
    y = 4.4
    bar(y, 0, 0.03, C_BLUE, "")
    bar(y, 0.03, 0.09, C_L1, "")
    bar(y, 0.12, 0.16, C_L0, "lock delay")
    bar(y, 0.28, t_stop_med, C_L2, "stop")
    ax.text(0.0, y + 0.42, "decision", fontsize=8)
    ax.text(0.0, y - 0.55, "SENSE 30 ms | ACT 70 ms + lag 20 ms", fontsize=7.5)
    ax.text(0.28 + t_stop_med + 0.01, y, f"  engaged at 0.28 s; COM stopped {0.28 + t_stop_med:.2f} s (p95 {0.28 + t_stop_p95:.2f} s)",
            va="center", fontsize=8)
    ax.text(-0.015, y, "SPEC chain\n(catch)", ha="right", va="center", fontsize=8.5)
    # row 2: nudge
    y = 3.2
    bar(y, 0, 0.03, C_BLUE, "")
    bar(y, 0.03, 0.09, C_L1, "")
    ax.text(0.125, y, "  nudge force arrives 0.12 s after decision (sense 0.03 + delay 0.07 + lag 0.02)", va="center", fontsize=8)
    ax.text(-0.015, y, "SPEC chain\n(nudge)", ha="right", va="center", fontsize=8.5)
    # row 3: hardware budget (assumed)
    y = 2.0
    seq = [("sense+compute", 0.04, C_BLUE), ("slack take-up", 0.05, C_L1), ("brake apply", 0.03, C_L0), ("cable stretch", 0.02, C_VIOLET)]
    t = 0
    for lbl, dt, col in seq:
        bar(y, t, dt, col, "")
        ax.text(t + dt / 2, y - 0.38 - 0.28 * (len(seq) and (seq.index((lbl, dt, col)) % 2)), lbl, ha="center", fontsize=7)
        t += dt
    bar(y, t, t_stop_med, C_L2, "")
    ax.text(t + t_stop_med + 0.01, y, f"  ~{t:.2f} s to hold (assumed budget, NOT measured)", va="center", fontsize=8)
    ax.text(-0.015, y, "Hardware\nbudget (assumed)", ha="right", va="center", fontsize=8.5)
    # row 4: mode states
    y = 0.8
    ax.barh(y, 0.12, left=0, height=0.45, color=C_L0, edgecolor="k", lw=0.6)
    ax.barh(y, 0.16, left=0.12, height=0.45, color=C_L1, edgecolor="k", lw=0.6)
    ax.barh(y, 0.62, left=0.28, height=0.45, color=C_L2, edgecolor="k", lw=0.6)
    ax.text(0.06, y, "slack", ha="center", va="center", fontsize=8, color="white")
    ax.text(0.20, y, "winding", ha="center", va="center", fontsize=7.5)
    ax.text(0.59, y, "LOCKED (spring brake, slip at F_lim)", ha="center", va="center", fontsize=8, color="white")
    ax.text(-0.015, y, "Mode of the\nsame cable set", ha="right", va="center", fontsize=8.5)
    ax.set_xlim(-0.2, 0.92)
    ax.set_ylim(0.2, 6.0)
    ax.set_yticks([])
    ax.set_xlabel("time from catch decision [s]")
    ax.set_title("With the SPEC 0.28 s lock delay the arrest lands inside the 0.3-0.5 s fall window; an assumed 0.14 s hardware budget would finish before it opens", fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_timing.png"), dpi=160)
    plt.close(fig)


def main():
    res = dict(tip_curve={}, params=dict(SF=SF, XI_ARREST=XI_ARREST, XI_NUDGE=XI_NUDGE, CAP_ARREST=CAP_ARREST, CAP_NUDGE=CAP_NUDGE))
    rows = {t: load_rows(t) for t in ("280", "150", "080")}
    res["loads"] = {}
    for t, a in rows.items():
        res["loads"][t] = {k: stats3(a[k]) for k in ("vout", "s0", "ke", "F_peak", "Freq", "F_hold")}
        res["loads"][t]["n"] = len(a["m"])
        res["loads"][t]["Freq_frac_le"] = {str(F): float((a["Freq"] <= F).mean()) for F in (150, 200, 250, 300, 400, 491)}
        res["loads"][t]["n_s0_gt_0.2"] = int((a["s0"] > 0.2).sum())
    fig_loads(rows)
    fig_tip(res)
    fig_mass(res)
    res["envelope"] = {n: {k: v for k, v in envelope(d).items() if k != "per_dir"} for n, d in DESIGNS.items()}
    res["envelope_sf1"] = {n: {k: v for k, v in envelope(d, sf=1.0).items() if k != "per_dir"} for n, d in DESIGNS.items()}
    res["envelope_rec_per_dir"] = envelope(REC)["per_dir"]
    res["slide"] = {n: dict(zip(("F_slide", "kappa"), map(float, slide_limit(d)))) for n, d in DESIGNS.items()}
    cable_table(REC, 300.0, res)
    hang_case(res)
    sensitivity(res)
    run_dynamics(rows, res)
    fig_timing(rows, res)
    json.dump(res, open(os.path.join(DATA, "results.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("envelope", "envelope_sf1", "slide", "slide_limit", "sensitivity", "hang", "t_stop")}, indent=1, default=float))


if __name__ == "__main__":
    main()
