"""One source of truth for D3 geometry: data/d3_contract.json drives the analysis (sizing.py) AND the CAD
(cad/build_cad_d3.py).  These tests load the contract, the generated CAD parts (cad/parts_d3.json + the part STLs) and the
analysis objects, and assert that anchors (= cable exit points), wheel contacts, belt dimensions / attachment points,
BWS anchor, outrigger feet and walker data agree within 1 mm.  Regenerate the CAD first:  python cad/build_cad_d3.py"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import publish as PB  # noqa: E402
import sizing as S  # noqa: E402

TOL = 1e-3          # 1 mm
C = S.load_contract()
PARTS_JSON = os.path.join(ROOT, "cad", "parts_d3.json")
PARTS_DIR = os.path.join(ROOT, "cad", "parts_d3")


@pytest.fixture(scope="module")
def cad():
    with open(PARTS_JSON, encoding="utf8") as f:
        return json.load(f)


def _anchors():
    out = {}
    for grp, key in (("corner", "corner_anchors"), ("lateral", "lateral_anchors"), ("centre", "centre_anchors")):
        for n, a in zip(C["anchor_names"][grp], C[key]):
            out[n] = np.array(a, float)
    return out


def _mesh(name):
    import trimesh
    return trimesh.load(os.path.join(PARTS_DIR, name + ".stl"))


def _bounds(cad, name):
    return np.array(next(p["bounds"] for p in cad["parts"] if p["name"] == name))


def test_cad_json_is_current(cad):
    """The CAD parts file is generated from this contract (same belt, wheels, anchors)."""
    assert abs(cad["belt"]["hd"] - C["belt"]["hd"]) < TOL and abs(cad["belt"]["hw"] - C["belt"]["hw"]) < TOL
    assert set(cad["anchors"]) == set(_anchors())


def test_anchors_are_cable_exit_points_in_cad_json(cad):
    for n, a in _anchors().items():
        assert np.abs(np.array(cad["anchors"][n]["exit"]) - a).max() < TOL, n
        assert np.abs(np.array(cad["anchors"][n]["contract"]) - a).max() < TOL, n
    assert np.abs(np.array(cad["bws"]["exit"]) - np.array(C["bws_anchor"])).max() < TOL
    assert np.abs(np.array(cad["bws"]["tip"]) - np.array(C["bws_anchor"])).max() < TOL


def test_fairlead_mesh_tip_sits_on_anchor(cad):
    """Geometric check on the actual spool fairlead meshes: the end-face centre (cable exit) is on the anchor."""
    for n, a in _anchors().items():
        m = _mesh(f"spool_{n}_fairlead")
        d = np.array(cad["anchors"][n]["n"])
        proj = m.vertices @ d
        tip = m.vertices[proj > proj.max() - 1e-6].mean(0)
        assert np.abs(tip - a).max() < TOL, (n, tip, a)
    m = _mesh("bws_fairlead")
    top = m.vertices[m.vertices[:, 2] > m.vertices[:, 2].max() - 1e-6].mean(0)
    assert np.abs(top - np.array(C["bws_anchor"])).max() < TOL


def test_wheel_contacts(cad):
    wx, wy, R = C["wheels_x"], C["wheels_y"], C["wheel_radius"]
    want = {"RL": (wx[0], wy), "RR": (wx[0], -wy), "FL": (wx[1], wy), "FR": (wx[1], -wy)}
    assert set(cad["wheels"]) == set(want)
    w = S.d3_walker()
    sup = w.supports()[:4]
    for k, (x, y) in want.items():
        assert np.abs(np.array(cad["wheels"][k]["contact"]) - [x, y]).max() < TOL, k
        assert abs(cad["wheels"][k]["radius"] - R) < TOL
        b = _bounds(cad, f"wheel_{k}_hub")                     # hub is centred on the axle
        assert np.abs((b[0] + b[1]) / 2 - [x, y, R]).max() < TOL, (k, b)
        assert min(np.abs(sup - [x, y]).max(1)) < TOL           # analysis support point
    # footprint 0.95 x 0.80 inset by the wheel radius
    assert abs(wx[0] - R) < TOL and abs(wx[1] - (C["footprint_xy"][0] - R)) < TOL
    assert abs(wy - (C["footprint_xy"][1] / 2 - R)) < TOL


def test_belt_dimensions_and_attachment(cad):
    hd, hw, t = C["belt"]["hd"], C["belt"]["hw"], C["belt"]["t"]
    P = np.array(C["pelvis"])
    b = _bounds(cad, "pelvic_belt")                           # centre-line ellipse + half the belt thickness
    assert abs((b[1][0] - b[0][0]) / 2 - (hd + t / 2)) < TOL
    assert abs((b[1][1] - b[0][1]) / 2 - (hw + t / 2)) < TOL
    assert np.abs((b[0][:2] + b[1][:2]) / 2 - P[:2]).max() < TOL
    for n, bp in C["belt_attach"].items():
        bp = np.array(bp)
        assert np.abs(np.array(cad["anchors"][n]["belt"]) - bp).max() < TOL, n
        assert np.abs(np.array(cad["belt"]["attach"][n]) - bp).max() < TOL, n
        assert abs(((bp[0] - P[0]) / hd) ** 2 + ((bp[1] - P[1]) / hw) ** 2 - 1.0) < 1e-4, n    # on the belt ellipse


def test_analysis_uses_the_same_geometry():
    lay = S.d3_layout()
    names = [n for g in ("corner", "lateral", "centre") for n in C["anchor_names"][g]] + ["BWS"]
    contract_anchors = np.array([_anchors()[n] for n in names[:-1]] + [C["bws_anchor"]])
    assert np.abs(lay.anchors - contract_anchors).max() < TOL
    assert tuple(lay.belt) == (C["belt"]["hd"], C["belt"]["hw"])
    att, U, _ = S.cable_dirs(lay)
    for i, n in enumerate(names[:-1]):
        assert np.abs(att[i] - np.array(C["belt_attach"][n])).max() < TOL, n
    assert np.abs(att[-1] - np.array(C["pelvis"])).max() < TOL          # BWS cable attaches above the pelvis
    assert np.abs(S.PELVIS - np.array(C["pelvis"])).max() < TOL


def test_walker_and_outriggers(cad):
    w = S.d3_walker()
    assert w.m == C["mass"] and tuple(w.cg) == tuple(C["cg"]) and w.mu == C["mu_pad"]
    assert tuple(w.wx) == tuple(C["wheels_x"]) and w.wy == C["wheels_y"]
    assert w.ex == C["outrigger_ex"] and w.ey == C["outrigger_ey"]
    feet_an = {tuple(np.round(p, 6)) for p in w.supports()[4:]}
    feet_cad = {tuple(np.round(o["foot"], 6)) for o in cad["outriggers"].values()}
    assert feet_an == feet_cad
    for fx in C["outrigger_feet_x"]:
        for s in (1, -1):
            assert (round(fx, 6), round(s * C["outrigger_feet_y"], 6)) in feet_cad


def test_contract_results_are_current():
    """Contract numbers that are analysis OUTPUTS (slip thresholds, bounds, envelope) match the data files."""
    slip = json.load(open(os.path.join(HERE, "data", "slip.json")))
    res = json.load(open(os.path.join(HERE, "data", "results.json")))
    g = slip["group_tuned"]
    assert g["T_slip"] == C["T_slip"]
    assert slip["contract_thresholds"]["ok"]
    assert g["Fpar_min"] >= C["F_lim"] - 1e-6
    assert round(g["upper_bound_all_slip"]) == C["resultant_bound_all_slip"] or abs(g["upper_bound_all_slip"] - C["resultant_bound_all_slip"]) < 1.0
    assert g["Fh_max"] <= C["resultant_bound_all_slip"] + 1.0 and C["resultant_bound_all_slip"] <= C["envelope_SF1p5"]
    assert abs(res["envelope"]["D3 recommended"]["min"] - C["envelope_SF1p5"]) < 1e-6
    for grp, T in C["T_slip"].items():
        # rated hold torque = 1.5 x slip torque, a required rating -> published rounded UP (publish.req, 2 decimals)
        assert abs(C["rated_hold_torque_Nm"][grp] - PB.req(1.5 * T * C["spool_radius"], 2)) < 1e-9
    assert res["dyn"]["passive_params"]["T_slip"] == C["T_slip"]
