"""Regenerate the published I2 documents from the JSON data, with every number routed through publish.py.

    report.md                      <- templates/report.md.tmpl
    results/sizing_D3_summary.md   <- templates/sizing_D3_summary.md.tmpl
    data/d3_contract.json          <- derived/wording fields re-synchronised (bounds, hold torques, lift caps, wording)

Usage:  python analysis/i2_sizing/make_report.py            (write the committed files)
        python analysis/i2_sizing/make_report.py --out DIR  (write report.md + sizing_D3_summary.md + d3_contract.json to DIR)

Template placeholders  {kind:path[:arg]}  (JSON paths use '/', list items are picked by index or by field=value):
    req/cap/desc  {req:S/group_tuned/Fh_max:1}   round UP / DOWN / NEAREST(descriptive only) to d decimals
    geo           nominal geometry or model input, nearest, fixed d decimals
    txt           a text field copied verbatim (e.g. a material name)
    int           an exact integer (count or integer-valued parameter), asserted integral
    ci            {ci:path_lo:path_hi}            Wilson CI in percent: lo DOWN, hi UP, 1 decimal
    pcap/preq     {pcap:path_k:path_n}            k/n in percent, rounded DOWN (coverage) / UP (failures), 1 decimal
    covp/covps/pctp  passive-slip coverage for lock 280|150|080 (with CI / without CI / percentage only)
    covf/covfs/pctf  the same for the old fixed-300 N metric
    covc/covcs/pctc  {covc:280:D3 recommended:300} dyn.curves entry (lock, design, F_lim)
    plain {name}  a named value computed in code below (V), e.g. interpolations.
A path may end in '*factor' (unit conversion applied before rounding).  Roots: R results.json, S slip.json,
K d3_contract.json, L28/L15/L08 arrest_loads_{280,150,080}.json.
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import gate_check as GC  # noqa: E402
import publish as PB  # noqa: E402
import sizing as S  # noqa: E402

DATA = os.path.join(HERE, "data")
TEMPLATES = os.path.join(HERE, "templates")
OUTPUTS = {  # template -> committed output (relative to ROOT)
    "report.md.tmpl": os.path.join("analysis", "i2_sizing", "report.md"),
    "sizing_D3_summary.md.tmpl": os.path.join("results", "sizing_D3_summary.md"),
}
CONTRACT_PATH = os.path.join(DATA, "d3_contract.json")


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------------------------
# contract synchronisation (derived fields of d3_contract.json come from the policy, not from typing)
# --------------------------------------------------------------------------------------------------------------
def sync_contract(K, R, SL):
    """Return a copy of the contract with its derived fields regenerated from the data via the rounding policy."""
    K = json.loads(json.dumps(K))
    g = SL["group_tuned"]
    K["resultant_bound_all_slip"] = int(PB.req(g["upper_bound_all_slip"], 0))          # a load -> UP
    K["envelope_SF1p5"] = int(PB.cap(R["envelope"]["D3 recommended"]["min"], 0))         # an envelope -> DOWN
    br = SL["brake_table_rs0.025"]
    K["rated_hold_torque_Nm"] = {
        "corner": PB.req(br["corner"]["rated_hold_torque"], 2),                           # required rating -> UP
        "lateral": PB.req(br["lateral"]["rated_hold_torque"], 2),
        "centre": PB.req(br["centre"]["rated_hold_torque"], 2),
        "bws": PB.req(br["bws (hang only, kept slack in horizontal arrest)"]["rated_hold_torque"], 2),
    }
    K["lift_cap_arrest_N"] = PB.cap(R["params"]["CAP_ARREST"], 1)                         # a limit -> DOWN
    K["lift_cap_nudge_N"] = PB.cap(R["params"]["CAP_NUDGE"], 1)
    dl = GC.design_loads(K, R)                                                            # front-gate loads -> UP
    K["front_gate"]["design_loads_N"] = {
        "T_slip_centre": PB.req(dl["T_slip_centre"], 1),
        "T_qp_frontC_worst": PB.req(dl["T_qp_frontC_worst"], 1),
        "design_cable_force": PB.req(dl["design_cable_force"], 1),
        "latch_rating_min": PB.req(dl["latch_rating"], 1),
    }
    K["F_lim_wording"] = (
        f"passive slip brakes give >= F_lim ({K['F_lim']} N) along the fall direction and "
        f"<= resultant_bound_all_slip ({K['resultant_bound_all_slip']} N) horizontal resultant; "
        f"the SF-{PB.desc_s(R['params']['SF'], 1)} static envelope is envelope_SF1p5 ({K['envelope_SF1p5']} N)")
    return K


def dump_contract(K):
    return json.dumps(K, indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------------------
# data + named values
# --------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, contract_override=None):
        self.R = _load("results.json")
        self.S = _load("slip.json")
        self.G = _load("gate.json")
        self.K = contract_override if contract_override is not None else _load("d3_contract.json")
        self.L28, self.L15, self.L08 = (_load(f"arrest_loads_{t}.json") for t in ("280", "150", "080"))
        self.roots = {"R": self.R, "S": self.S, "K": self.K, "G": self.G, "L28": self.L28, "L15": self.L15, "L08": self.L08}
        with open(os.path.join(ROOT, "results", "summary.csv"), encoding="utf8") as f:
            self.summary = {r["setting"]: r for r in csv.DictReader(f)}
        self.V = {}
        self._named()

    # ---- path resolution ------------------------------------------------------------------------------------
    def get(self, path):
        scale = 1.0
        if "*" in path:
            path, fac = path.rsplit("*", 1)
            scale = float(fac)
        el = path.split("/")
        node = self.roots[el[0]]
        for e in el[1:]:
            if isinstance(node, list):
                if "=" in e:
                    fld, val = e.split("=", 1)
                    node = next(x for x in node if str(x[fld]) == val)
                else:
                    node = node[int(e)]
            else:
                node = node[e]
        return node * scale if scale != 1.0 else node

    # ---- named (computed) values ----------------------------------------------------------------------------
    @staticmethod
    def interp(xs, ys, x):
        """Linear interpolation between the two ADJACENT tabulated points (chord of a convex curve = upper bound)."""
        xs = [float(v) for v in xs]
        for i in range(len(xs) - 1):
            lo, hi = sorted((xs[i], xs[i + 1]))
            if lo - 1e-12 <= x <= hi + 1e-12:
                t = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        raise ValueError(x)

    def _named(self):
        R, SL, K, V = self.R, self.S, self.K, self.V
        P28 = R["dyn"]["passive"]["280"]
        g = SL["group_tuned"]
        cur = R["dyn"]["curves"]
        # trials / counts
        ntot = self.L28["n_trials"]
        assert self.L15["n_trials"] == self.L08["n_trials"] == ntot
        fall_pct = float(self.summary["CP m=0.03"]["fall_pct"])
        falls0 = round(fall_pct * ntot / 100)
        assert abs(falls0 * 100 / ntot - fall_pct) < 1e-9
        nfail = P28["n_fail"]
        V["ntot"] = str(ntot)
        V["ncaught"] = str(P28["n"])
        V["never"] = str(ntot - P28["n"])
        V["falls0"] = str(falls0)
        V["falls0_pct"] = PB.req_s(falls0 * 100 / ntot, 2)
        V["worst_falls"] = str(falls0 + nfail)
        V["worst_pct"] = PB.req_s((falls0 + nfail) * 100 / ntot, 1)
        V["nfail"] = str(nfail)
        V["nfail_fixed"] = str(R["dyn"]["rec_detail"]["300"]["n_fail"])
        # the 491 N case label (a simulation parameter, F_lim of the "allow 491 N" curves, not a rounded result)
        f491 = max(c["F_lim"] for c in cur["280|D3 recommended"])
        assert f491 == 491 and abs(self.L28["peakCatch_sim_max"] - f491) < 1.0
        V["f491"] = str(f491)
        V["peak491_2"] = PB.req_s(self.L28["peakCatch_sim_max"], 2)
        # SF of the 60 kg design at 491 N: SF*F is constant for it (statics linear in F), so SF(491) = SF*F / 491
        tc = R["tip_curve"]["D0 as proposed"]
        prod = [f * s for f, s in zip(tc["F"], tc["sf"])]
        assert max(prod) - min(prod) < 1e-6 * max(prod)
        V["d0_sf_491"] = PB.cap_s(prod[0] / f491, 2)
        # required-force statistics (trials above 600 N computed from the per-trial rows)
        rows = _load("arrest_rows_280.json")
        m = np.array([r["m"] for r in rows]); z0 = np.array([r["z0"] for r in rows])
        s0 = np.array([r["s0"] for r in rows]); vo = np.array([r["vout"] for r in rows])
        freq = S.required_force(m, np.sqrt(S.G / z0), s0, vo)
        assert abs(float(np.median(freq)) - R["loads"]["280"]["Freq"]["median"]) < 1e-9
        V["n_gt600"] = str(int((freq > 600).sum()))
        # interpolations from the mass curves (state: linear between adjacent JSON points, requirement -> UP)
        hw = K["wheels_y"]
        mc = R["mass_curves"]
        def mass_at(key, x):
            c = mc[key]
            return self.interp(c["hw"], c["mass"], x)
        kh = f"horizontal cables (z = 0.95 m)|F={f491}|SF=1.0"
        km = "high corner anchors (1.20 m) + mid anchors|F=300|SF=1.5"
        V["hw_a"] = PB.desc_s(hw, 3)                       # 0.325 (wheel-line half-width = contract wheels_y)
        V["hw_b"] = PB.desc_s(0.40, 2)                     # second tabulated half-width quoted (0.40 m, a table point)
        assert any(abs(h - 0.40) < 1e-9 for h in mc[kh]["hw"])
        V["m_h_a"] = PB.req_s(mass_at(kh, hw), 0)
        V["m_h_b"] = PB.req_s(mass_at(kh, 0.40), 0)
        V["m_m_a"] = PB.req_s(mass_at(km, hw), 1)
        V["m_m_b"] = PB.req_s(mass_at(km, 0.40), 1)
        # half-width at which a 60 kg walker suffices (mixed-height, 300 N, SF 1.5): invert the tabulated curve
        c = mc[km]
        hw60 = self.interp(c["mass"], c["hw"], 60.0)
        V["hw60"] = PB.req_s(hw60, 2)
        # arrest rows
        AR = R["arrest_rows"]
        mu = K["mu_pad"]
        for i, a in enumerate(AR):
            V[f"fcap{i}"] = PB.cap_s(mu * a["N_tot"], 0)
        V["fcap_min"] = PB.cap_s(min(mu * a["N_tot"] for a in AR), 0)
        V["lift_ar_lo"] = PB.cap_s(min(a["lift"] for a in AR), 1)
        V["lift_ar_hi"] = PB.req_s(max(a["lift"] for a in AR), 1)
        V["sf_ar_min"] = PB.cap_s(min(a["sf_tip"] for a in AR), 2)
        assert abs(AR[2]["Tmax"] - AR[3]["Tmax"]) < 1e-6 and abs(AR[2]["lift"] - AR[3]["lift"]) < 1e-6
        assert abs(AR[2]["sf_tip"] - AR[3]["sf_tip"]) < 1e-9 and abs(AR[2]["N_tot"] - AR[3]["N_tot"]) < 1e-6
        assert abs(AR[2]["T"]["latR"] - AR[3]["T"]["latL"]) < 1e-9
        # nudge
        NR = R["nudge_rows"]
        assert all(r["feasible"] for r in NR)
        V["nudge_n"] = PB.desc_s(NR[0]["H"], 0)
        assert all(abs(r["H"] - NR[0]["H"]) < 1e-9 for r in NR)
        V["nudge_tmax"] = PB.req_s(R["Tmax_nudge_worst"], 1)
        V["nudge_lift"] = PB.req_s(max(r["lift"] for r in NR), 1)
        V["nudge_sf"] = PB.cap_s(min(r["sf_tip"] for r in NR), 2)
        V["nudge_sfs"] = PB.cap_s(min(r["sf_slide"] for r in NR), 2)
        V["ndir_nudge"] = str(len(NR))
        V["ndir_env"] = str(len(R["envelope_rec_per_dir"]))
        qp = SL["controller_qp"]
        V["qp_err_sf"] = PB.cap_s(min(qp[e]["sf_min"] for e in ("15", "30", "45")), 2)
        V["qp_err_lift"] = PB.req_s(max(qp[e]["lift_max"] for e in ("15", "30", "45")), 0)
        # brake torques
        sb = R["spool_brake"]["r_s=0.025"]
        assert abs(R["spool_brake"]["r_s=0.025"]["tau_brake"] - 1.5 * R["Tmax_arrest_worst"] * K["spool_radius"]) < 1e-9
        V["tau_brake"] = PB.req_s(sb["tau_brake"], 2)
        V["tau_brake_env"] = PB.req_s(sb["tau_brake"] * (K["envelope_SF1p5"] / K["F_lim"]), 2)
        # hang
        hg = R["hang"]
        spec = hg["SPEC rear beam x=0.03 | D3 recommended"]; spec60 = hg["SPEC rear beam x=0.03 | D1 60 kg"]
        V["hang_h"] = PB.req_s(spec["H"], 0)
        V["hang_sf_d3"] = PB.cap_s(spec["sf_tip"], 2)
        V["hang_sf_60"] = PB.cap_s(spec60["sf_tip"], 2)
        V["hang_t"] = PB.req_s(hg["overhead x=0.42 (cantilevered top beam) | D3 recommended"]["T"], 0)
        # sensitivities (cap, envelope-type values), looked up by their JSON key
        for key, name in (("outriggers not deployed", "s_noout"), ("outriggers +0.15 m", "s_out15"), ("walker 60 kg (CG 0.45 m)", "s_w60"),
                          ("walker 100 kg", "s_w100"), ("pelvis 0.15 m lower (0.80)", "s_low"), ("pelvis 0.10 m further rearward", "s_rear"),
                          ("pelvis 0.10 m further forward", "s_fwd"), ("pelvis 0.10 m to the left", "s_left"),
                          ("walker CG 0.10 m higher", "s_cgup"), ("walker CG 0.10 m rearward", "s_cgrear"),
                          ("6 cables (no centre anchors)", "s_6"), ("4 cables (corners only)", "s_4"),
                          ("all 8 anchors at 1.20 m", "s_all12"), ("corners 1.50 m, mid 0.95 m", "s_c15")):
            V[name] = PB.cap_s(R["sensitivity"][key], 0)
        # timing: lock delay + COM stop time after engagement
        lock = self.L28["tau_lock"]
        ts = R["dyn"]["passive"]["280"]["t_stop"]
        V["lock28"] = PB.desc_s(lock, 2)
        V["t_stop_med_abs"] = PB.desc_s(lock + ts["median"], 2)       # descriptive (median)
        V["t_stop_p95_abs"] = PB.req_s(lock + ts["p95"], 2)           # 95th percentile of a duration, published UP
        # energies
        V["energy_max"] = PB.req_s(max(R["dyn"]["passive"][k]["brake_energy_max_J"] for k in R["dyn"]["passive"]), 1)
        V["energy_28"] = PB.req_s(P28["brake_energy_max_J"], 1)
        # counts of the passive failure classes
        for lk in ("280", "150", "080"):
            p = R["dyn"]["passive"][lk]
            assert p["n_dyn_tip"] == p["n_static_fail"] == p["n_slide_fail"] == 0 and p["n_not_arrested"] == p["n_fail"]
        # front gate (data/gate.json from gate_check.py): must be current w.r.t. the contract + results
        Gt = self.G
        assert Gt["pass"] and abs(Gt["loads"]["design_cable_force"] - GC.design_loads(K, R)["design_cable_force"]) < 1e-9
        assert abs(Gt["span_m"] - K["front_gate"]["arm_length"]) < 1e-9
        V["gate_pass"] = "PASS" if Gt["pass"] else "FAIL"
        # margin between the resultant bound and the envelope
        V["margin_420"] = PB.cap_s(K["envelope_SF1p5"] - g["Fh_max"], 0)
        V["n_geo_dirs"] = str(g["n_directions"])
        V["ballast"] = PB.desc_s(K["mass"] - 60, 0)
        V["slide_tol_cm"] = PB.desc_s(R["dyn"]["passive_params"]["slide_tol"] * 100, 0)


# --------------------------------------------------------------------------------------------------------------
# placeholder rendering
# --------------------------------------------------------------------------------------------------------------
PH = re.compile(r"\{([a-z][a-z0-9_]*)(?::([^{}]*))?\}")


def render(tmpl, D):
    R = D.R

    def cov(k, n, lo, hi, with_ci=True):
        s = f"{k}/{n} = {PB.pct_cap(k, n)} %"
        return s + (f" (CI {PB.pct_ci(lo, hi, sep='-')} %)" if with_ci else "")

    def sub(mo):
        name, arg = mo.group(1), mo.group(2)
        a = arg.split(":") if arg is not None else []
        if name in ("req", "cap", "desc", "geo"):
            v = D.get(a[0])
            d = int(a[1]) if len(a) > 1 else 0
            return {"req": PB.req_s, "cap": PB.cap_s, "desc": PB.desc_s, "geo": PB.desc_s}[name](v, d)
        if name == "txt":
            return str(D.get(a[0]))
        if name == "int":
            v = D.get(a[0])
            assert float(v) == int(v), a[0]
            return str(int(v))
        if name == "ci":
            return PB.pct_ci(D.get(a[0]), D.get(a[1]), sep="-")
        if name == "pcap":
            return PB.cap_s(100.0 * D.get(a[0]) / D.get(a[1]), 1)
        if name == "preq":
            return PB.req_s(100.0 * D.get(a[0]) / D.get(a[1]), 1)
        if name in ("covp", "covps", "pctp", "covf", "covfs", "pctf"):
            p = R["dyn"]["passive"][a[0]]
            pre = "fixed300_" if name.endswith("f") or name.endswith("fs") else ""
            k, n = p[pre + "k"], p[pre + "n"]
            if name.startswith("pct"):
                return PB.cap_s(100.0 * k / n, 1)
            return cov(k, n, p[pre + "ci_lo"], p[pre + "ci_hi"], with_ci=name in ("covp", "covf"))
        if name in ("covc", "covcs", "pctc"):
            c = next(c for c in R["dyn"]["curves"][f"{a[0]}|{a[1]}"] if c["F_lim"] == int(a[2]))
            if name == "pctc":
                return PB.cap_s(100.0 * c["k"] / c["n"], 1)
            return cov(c["k"], c["n"], c["ci_lo"], c["ci_hi"], with_ci=(name == "covc"))
        if arg is None and name in D.V:
            return D.V[name]
        raise KeyError(mo.group(0))

    out = PH.sub(sub, tmpl)
    left = re.findall(r"\{[^{}\n]*\}", out)
    assert not left, f"unresolved placeholders: {left[:5]}"
    return out


def build(out_dir=None):
    """Render every template; return {relative output path: text}. Contract synchronised first (derived fields)."""
    K0 = _load("d3_contract.json")
    Kn = sync_contract(K0, _load("results.json"), _load("slip.json"))
    D = Data(contract_override=Kn)
    outs = {}
    for tname, rel in OUTPUTS.items():
        with open(os.path.join(TEMPLATES, tname), encoding="utf8", newline="") as f:
            outs[rel] = render(f.read(), D)
    outs[os.path.join("analysis", "i2_sizing", "data", "d3_contract.json")] = dump_contract(Kn)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write to this directory (flat file names) instead of the repo")
    args = ap.parse_args()
    outs = build()
    for rel, text in outs.items():
        path = os.path.join(args.out, os.path.basename(rel)) if args.out else os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf8", newline="\n") as f:
            f.write(text)
        print("wrote", path)


if __name__ == "__main__":
    main()
