import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sizing as S  # noqa: E402

G = S.G


def d0():
    return S.Walker(m=60, cg=(0.40, 0, 0.45)), S.make_layout(6, z_hi=0.95)


def d3():
    """D3 walker + layout straight from data/d3_contract.json (single source of truth)."""
    return S.d3_walker(), S.d3_layout()


@pytest.mark.parametrize("phi", [0, 37, 90, 145, 180, 270, 333])
@pytest.mark.parametrize("F", [10.0, 150.0, 300.0])
def test_tension_distribution_reproduces_force(phi, F):
    _, lay = d3()
    _, U, _ = S.cable_dirs(lay)
    ph = np.radians(phi)
    Fh = F * np.array([np.cos(ph), np.sin(ph)])
    T, ok = S.tension_distribution(U, Fh, None, Tmin=2.0, Tmax=1500.0)
    assert ok
    assert np.abs((U[:, :2].T @ T) - Fh).max() < 1e-6
    assert T.min() >= 2.0 - 1e-9 and T.max() <= 1500.0 + 1e-9


def test_tension_distribution_with_vertical_pin():
    _, lay = d3()
    _, U, _ = S.cable_dirs(lay)
    T, ok = S.tension_distribution(U, np.array([-300.0, 0.0]), 120.0, 2.0, 1500.0)
    assert ok
    assert np.abs(U.T @ T - np.array([-300.0, 0.0, 120.0])).max() < 1e-6


def test_tension_is_minimum_norm():
    _, lay = d3()
    _, U, _ = S.cable_dirs(lay)
    from scipy.optimize import minimize
    Fh = np.array([-200.0, 50.0])
    T, _ = S.tension_distribution(U, Fh, None, 2.0, 1500.0)
    # independent reference: SLSQP (analytic gradients) from several starts; runs count only if they satisfy the
    # equality (< 1e-6 N) and the bounds
    funs = []
    for x0 in (np.full(len(U), 50.0), np.full(len(U), 10.0), T * 1.05 + 1.0):
        r = minimize(lambda t: (t ** 2).sum(), x0, jac=lambda t: 2 * t, method="SLSQP", bounds=[(2, 1500)] * len(U),
                     constraints=[{"type": "eq", "fun": lambda t: U[:, :2].T @ t - Fh, "jac": lambda t: U[:, :2].T}],
                     options=dict(ftol=1e-14, maxiter=1000))
        if np.abs(U[:, :2].T @ r.x - Fh).max() < 1e-6 and r.x.min() >= 2.0 - 1e-6:
            funs.append(r.fun)
    assert funs, "reference optimiser produced no feasible point"
    assert (T ** 2).sum() <= min(funs) + 1e-3


@pytest.mark.parametrize("phi", [0, 90, 180, 270, 45])
def test_static_equilibrium_vertical_and_horizontal(phi):
    w, lay = d3()
    r = S.load_case(w, lay, 300.0, phi, Fz_cap=0.5 * 60 * G)
    assert r["feasible"]
    # sum of wheel normals = walker weight + downward cable reactions
    assert abs(np.nansum(r["wheel_N"]) - (w.m * G + r["Fz_down_on_frame"])) < 1e-6
    # vertical reaction = cable lift on the pelvis (action-reaction); friction needed = horizontal pull
    assert abs(r["Fz_down_on_frame"] - r["Fz_pelvis"]) < 1e-6
    assert abs(r["H_mag"] - 300.0) < 1e-6
    # moment balance: wheel normals reproduce the CoP
    S_ = r["supports"]
    N = r["wheel_N"]
    assert abs((N * S_[:, 0]).sum() / N.sum() - r["cop"][0]) < 1e-6
    assert abs((N * S_[:, 1]).sum() / N.sum() - r["cop"][1]) < 1e-6
    assert (N >= -1e-9).all()


def test_only_vertical_cables_transfer_patient_weight():
    """Horizontal cables at pelvis height put NO extra load on the wheels: patient weight is not shared."""
    w, lay = d0()
    r = S.load_case(w, lay, 100.0, 0.0)
    assert abs(r["Fz_down_on_frame"]) < 1e-9
    assert abs(r["N_tot"] - w.m * G) < 1e-9


def test_known_tipping_case_491N_at_0p95m():
    """60 kg walker, horizontal 491 N pull at z = 0.95 m: rear-edge tip.  Hand calc: SF = m g a / (F z)."""
    w, lay = d0()
    r = S.load_case(w, lay, 491.0, 0.0)
    a = w.cg[0] - w.wx[0]
    sf_hand = w.m * G * a / (491.0 * 0.95)
    assert r["tipped"] and r["tip_edge"] == "rear"
    assert abs(r["sf_tip"] - sf_hand) < 1e-6
    assert sf_hand < 0.5
    # and the reviewer's 119 kg (0.40 m lever) zero-margin figure is reproduced by required_mass
    base = S.Walker(m=60, cg=(0.475, 0, 0.45), wx=(0.075, 0.875), wy=0.4)
    base = S.Walker(m=60, cg=(0.40, 0, 0.45), wx=(0.0, 0.8), wy=0.4)      # 0.40 m lever in AP and ML
    m_req = S.required_mass(lay, 491.0, base, sf=1.0, step=90)
    assert 115 < m_req < 125


def test_no_tip_at_nudge():
    w, lay = d0()
    for phi in range(0, 360, 30):
        r = S.load_case(w, lay, 150.0, phi)
        assert not r["tipped"] and r["sf_tip"] > 1.0


def test_recommended_design_holds_300N_all_directions():
    w, lay = d3()
    for phi in range(0, 360, 15):
        r = S.load_case(w, lay, 300.0, phi, Fz_cap=0.5 * 60 * G)
        assert r["feasible"] and r["sf_tip"] >= 1.5


def test_required_force_closed_form_matches_dynamics():
    kw = dict(m_w=500.0, mu=0.9, kappa=0.0, k_c=1e7, zeta=0.2)
    m, om = 70.0, np.sqrt(G / 0.925)
    # (a) capture-point branch: s0 + v/w <= 0.25 -> F_min = m w (w s0 + v); 10 % above stops, 10 % below falls
    s0, v = 0.0, 0.5
    F = float(S.required_force(m, om, s0, v))
    assert abs(F - m * om * (om * s0 + v)) < 1e-9
    assert S.arrest_1d([m], [om], [s0], [v], [F * 1.1], **kw)["ok"][0]
    assert not S.arrest_1d([m], [om], [s0], [v], [F * 0.9], **kw)["ok"][0]
    # (b) distance branch: s0 + v/w > 0.25 -> F from s_m = 0.25; 3 % above stops within 0.25, 3 % below does not
    s0, v = 0.10, 0.9
    F = float(S.required_force(m, om, s0, v))
    assert s0 + v / om > 0.25
    hi = S.arrest_1d([m], [om], [s0], [v], [F * 1.03], **kw)
    lo = S.arrest_1d([m], [om], [s0], [v], [F * 0.97], **kw)
    assert hi["ok"][0] and not lo["ok"][0]
    assert abs(hi["s_max"][0] - 0.25) < 0.03


def test_walker_slides_when_friction_exceeded():
    out = S.arrest_1d([70.0], [3.2], [0.0], [0.6], [400.0], m_w=40.0, mu=0.3, kappa=0.0)
    assert out["walker_slide"][0] > 0.01 and out["F_peak"][0] <= 400.0 + 1e-9


def test_dynamic_tip_short_pulse_survives_long_pulse_tips():
    w, lay = d0()
    r = S.load_case(w, lay, 300.0, 0.0)
    M, a = S.edge_moment(w, lay, r["U"], r["T"], r["tip_edge"])
    assert M > w.m * G * a            # statically over the limit
    short, *_ = S.dynamic_tip(w, M, 300.0, np.full(3, 300.0), 0.01, a)
    long_, *_ = S.dynamic_tip(w, M, 300.0, np.full(100, 300.0), 0.01, a)
    assert (not short) and long_


def test_spool_brake_torque():
    sb = S.spool_brake(200.0, 100.0, r_s=0.025, sf_dyn=1.5)
    assert abs(sb["tau_brake"] - 1.5 * 200 * 0.025) < 1e-12


def _slip_metrics(Tslip, step=15):
    w, lay = d3()
    _, U, _ = S.cable_dirs(lay)
    exc = np.array([k == "bws" for k in lay.kinds])
    Fpar, Fh, sf = [], [], []
    for dd in range(0, 360, step):
        dv = np.array([np.cos(np.radians(dd)), np.sin(np.radians(dd))])
        F, T = S.slip_resultant(U, Tslip, dv, exc)
        Fpar.append(F[:2] @ (-dv))
        Fh.append(np.hypot(*F[:2]))
        sf.append(S.walker_statics(w, lay, U, T)["sf_tip"])
    return U, exc, min(Fpar), max(Fh), min(sf)


def test_slip_resultant_bound_group_tuned():
    """Passive per-spool slip (contract thresholds, BWS slack): the resultant on the pelvis never exceeds the contract
    bound (and the SF-1.5 envelope), even if EVERY cable slips at once, and still delivers >= F_lim (300 N) along the
    arrest direction in every direction (1 deg grid) with tip SF >= 1.5."""
    w, lay = d3()
    Ts = S.group_slip_vector(lay, S.D3C["T_slip"])
    U, exc, Fpar_min, Fh_max, sf_min = _slip_metrics(Ts, step=1)
    ub = S.slip_upper_bound(U, Ts, exc)
    assert Fpar_min >= S.D3C["F_lim"] - 1e-6
    assert Fh_max <= ub + 0.5 and ub <= S.D3C["resultant_bound_all_slip"] + 1.0
    assert S.D3C["resultant_bound_all_slip"] <= S.D3C["envelope_SF1p5"]
    assert sf_min >= 1.5


def test_uniform_slip_too_high_breaks_envelope():
    w, lay = d3()
    Ts = np.full(len(lay.kinds), 190.0)
    Ts[-1] = 0.0
    U, exc, _, Fh_max, _ = _slip_metrics(Ts)
    assert Fh_max > S.D3C["envelope_SF1p5"]


def test_arrest_1d_hscale_loads_friction():
    """A perpendicular component (hscale > 1) raises the horizontal load on the walker friction: more slide."""
    kw = dict(m_w=40.0, mu=0.5, kappa=0.0)
    a = S.arrest_1d([70.0], [3.2], [0.0], [0.6], [150.0], **kw)
    b = S.arrest_1d([70.0], [3.2], [0.0], [0.6], [150.0], hscale=1.5, **kw)
    assert b["walker_slide"][0] > a["walker_slide"][0]


def test_passive_slip_table_matches_slip_resultant():
    """Per-direction passive force table used by the coverage metric reproduces slip_resultant()."""
    import run_analysis as R
    tab = R.passive_slip_table(R.REC, S.D3C["T_slip"], [0.0, 90.0, 213.5])
    w, lay = d3()
    _, U, _ = S.cable_dirs(lay)
    exc = np.array([k == "bws" for k in lay.kinds])
    for dd, row in tab.items():
        dv = np.array([np.cos(np.radians(dd)), np.sin(np.radians(dd))])
        F, T = S.slip_resultant(U, S.group_slip_vector(lay, S.D3C["T_slip"]), dv, exc)
        assert abs(row["Fpar"] - F[:2] @ (-dv)) < 1e-9 and abs(row["Fh"] - np.hypot(*F[:2])) < 1e-9
        assert row["Fpar"] >= S.D3C["F_lim"] - 1e-6 and row["Fh"] <= S.D3C["resultant_bound_all_slip"] + 1.0
        assert row["hscale"] >= 1.0 and row["sf_tip"] >= 1.5
