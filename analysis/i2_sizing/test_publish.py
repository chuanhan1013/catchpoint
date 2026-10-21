"""Tests for the I2 rounding policy and the generated documents.

(a) policy unit tests for publish.py;
(b) no drift: report.md, results/sizing_D3_summary.md and the contract regenerate byte-identically (line endings
    normalised: a Windows checkout may hold CRLF);
(c) audit: for every quantity named in review/codex_round5.md plus the headline numbers, the value PUBLISHED in
    report.md / sizing_D3_summary.md / the deck text dump is conservative with respect to the JSON source
    (kind 'req': published >= source; kind 'cap': published <= source) and within one unit of its last printed digit;
(d) no hand-typed results: the only numbers-with-units left outside placeholders in the templates are model inputs and labels.
"""
import csv
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import make_report as MR  # noqa: E402
import publish as PB  # noqa: E402

TOL = 1e-9


def _text(rel):
    with open(os.path.join(ROOT, rel), encoding="utf8", newline="") as f:
        return f.read().replace("\r\n", "\n")


# ------------------------------------------------------------------------------------------------ (a) policy
def test_req_rounds_up_cap_rounds_down():
    assert PB.req(5.625, 1) == 5.7 and PB.cap(5.625, 1) == 5.6
    assert PB.req(101.084, 0) == 102.0 and PB.cap(101.084, 0) == 101.0
    assert PB.req(101.084, 1) == 101.1 and PB.cap(4.36996, 2) == 4.36
    assert PB.req(512.458, 0) == 513.0 and PB.cap(1.8754, 2) == 1.87
    assert PB.cap(259.901, 0) == 259.0 and PB.cap(447.621, 0) == 447.0
    assert PB.req(-1.21, 1) == -1.2 and PB.cap(-1.21, 1) == -1.3          # direction is numeric, not magnitude


def test_exact_values_are_not_moved_and_float_noise_is_snapped():
    for v in (0.0, 2.0, 305.0, 420.0):
        assert PB.req(v, 0) == v and PB.cap(v, 0) == v
    assert PB.req(0.30000000000000004, 1) == 0.3 and PB.cap(0.29999999999999993, 1) == 0.3
    assert PB.req(5.2, 1) == 5.2 and PB.cap(1.15 * 2, 1) == 2.3
    assert PB.req_s(5.0, 2) == "5.00" and PB.cap_s(3.0, 1) == "3.0" and PB.req_s(0, 0) == "0"


def test_req_never_below_cap_never_above_source():
    for v in (97.753, 99.331, 0.0049, 12345.6789, 1e-7, 2.0000001):
        for d in (0, 1, 2, 3):
            assert PB.req(v, d) >= v - TOL and PB.cap(v, d) <= v + TOL
            assert PB.req(v, d) - PB.cap(v, d) <= 10 ** -d + TOL


def test_pct_ci_lo_down_hi_up():
    assert PB.pct_ci(97.753, 99.331) == "97.7–99.4"
    assert PB.pct_ci(97.753, 99.331, sep="-") == "97.7-99.4"
    assert PB.pct_ci(42.914, 49.749, sep="-") == "42.9-49.8"
    assert PB.pct_ci(97.7, 99.4, sep="-") == "97.7-99.4"       # already on the grid: unchanged


def test_descriptive_is_nearest_and_coverage_rounds_down():
    assert PB.desc(214.4, 0) == 214.0 and PB.desc(214.5, 0) == 215.0
    assert PB.pct_cap(805, 814) == 98.8 and PB.pct_req(9, 814) == 1.2
    assert PB.pct_cap(804, 814) == 98.7


# ------------------------------------------------------------------------------------------------ (b) no drift
def test_generated_files_have_no_drift():
    outs = MR.build()
    assert len(outs) == 3
    for rel, text in outs.items():
        assert _text(rel) == text.replace("\r\n", "\n"), f"{rel} differs from its regeneration: run make_report.py"


def test_cli_writes_identical_files(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, os.path.join(HERE, "make_report.py"), "--out", str(tmp_path)], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    for rel in MR.build():
        got = (tmp_path / os.path.basename(rel)).read_bytes().replace(b"\r\n", b"\n")
        assert got == _text(rel).encode("utf8"), rel


def test_contract_derived_fields_follow_policy():
    D = MR.Data()
    K, R, SL = D.K, D.R, D.S
    g = SL["group_tuned"]
    assert K["resultant_bound_all_slip"] >= g["Fh_max"] and K["resultant_bound_all_slip"] >= g["upper_bound_all_slip"]
    assert K["resultant_bound_all_slip"] - g["Fh_max"] < 1.0
    assert K["envelope_SF1p5"] <= R["envelope"]["D3 recommended"]["min"]
    assert K["lift_cap_nudge_N"] <= R["params"]["CAP_NUDGE"] and K["lift_cap_arrest_N"] <= R["params"]["CAP_ARREST"]
    br = SL["brake_table_rs0.025"]
    for grp in ("corner", "lateral", "centre"):
        assert K["rated_hold_torque_Nm"][grp] >= br[grp]["rated_hold_torque"] - TOL
    assert K["rated_hold_torque_Nm"]["bws"] >= br["bws (hang only, kept slack in horizontal arrest)"]["rated_hold_torque"] - TOL
    w = K["F_lim_wording"]
    assert f"({K['resultant_bound_all_slip']} N)" in w and f"({K['F_lim']} N)" in w and f"({K['envelope_SF1p5']} N)" in w
    for note in K["notes"]:                                    # free text carries no force/torque numbers
        assert not re.search(r"\d+(\.\d+)?\s?(N|N m|N·m)\b", note), note


# ------------------------------------------------------------------------------------------------ (c) audit
D_ = MR.Data()
FILES = {"report": "analysis/i2_sizing/report.md", "summary": "results/sizing_D3_summary.md", "deck": "deck/_deck_text.md"}


def J(path):
    return D_.get(path)


def _csv_ci(setting, metric, col):
    with open(os.path.join(ROOT, "results", "ci.csv"), encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["setting"] == setting and r["metric"] == metric:
                return float(r[col])
    raise KeyError((setting, metric))


def _mass_interp(key, x):
    c = D_.R["mass_curves"][key]
    return D_.interp(c["hw"], c["mass"], x)


def _minimum_friction():
    return min(D_.K["mu_pad"] * a["N_tot"] for a in D_.R["arrest_rows"])


PAS = "R/dyn/passive/"
CUR = "R/dyn/curves/"
BR = "S/brake_table_rs0.025/"
BWS = BR + "bws (hang only, kept slack in horizontal arrest)/rated_hold_torque"
# (label, file, regex with one group per checked value, [(source, kind), ...] in group order); source = JSON path or callable
AUDIT = [
    # ---- Round-5 item 1: coverage CIs (incl. D0 rows, deck table, CSV-derived CIs)
    ("r5.1 passive CI 0.28 (report)", "report", r"arrest 805/814 = [\d.]+ % \(CI ([\d.]+)-([\d.]+) %\)", [(PAS + "280/ci_lo", "cap"), (PAS + "280/ci_hi", "req")]),
    ("r5.1 fixed CI 0.28 (report)", "report", r"804/814 = [\d.]+ % \(CI ([\d.]+)-([\d.]+) %\)\. The other", [(PAS + "280/fixed300_ci_lo", "cap"), (PAS + "280/fixed300_ci_hi", "req")]),
    ("r5.1 passive CI 0.15 (report)", "report", r"lock 0\.15 s: passive 821/844 = [\d.]+ % \(CI ([\d.]+)-([\d.]+) %\)", [(PAS + "150/ci_lo", "cap"), (PAS + "150/ci_hi", "req")]),
    ("r5.1 passive CI 0.08 (report)", "report", r"lock 0\.08 s: passive 817/866 = [\d.]+ % \(CI ([\d.]+)-([\d.]+) %\)", [(PAS + "080/ci_lo", "cap"), (PAS + "080/ci_hi", "req")]),
    ("r5.1 D0 CI 0.28 (report)", "report", r"D0 \(as proposed\), same metric: 377/814 = [\d.]+ % \(CI ([\d.]+)-([\d.]+) %\)", [(CUR + "280|D0 as proposed/F_lim=300/ci_lo", "cap"), (CUR + "280|D0 as proposed/F_lim=300/ci_hi", "req")]),
    ("r5.1 passive CI (summary)", "summary", r"= [\d.]+ % \(Wilson 95 % CI ([\d.]+)-([\d.]+) %\)", [(PAS + "280/ci_lo", "cap"), (PAS + "280/ci_hi", "req")]),
    ("r5.1 D0 CI (summary)", "summary", r"377/814 = [\d.]+ % \(CI ([\d.]+)-([\d.]+) %\)", [(CUR + "280|D0 as proposed/F_lim=300/ci_lo", "cap"), (CUR + "280|D0 as proposed/F_lim=300/ci_hi", "req")]),
    ("r5.1 passive CI 0.28 (deck)", "deck", r"805/814 = [\d.]+%  \(CI ([\d.]+)–([\d.]+)\)", [(PAS + "280/ci_lo", "cap"), (PAS + "280/ci_hi", "req")]),
    ("r5.1 D0 CI 0.28 (deck)", "deck", r"377/814 = [\d.]+%  \(CI ([\d.]+)–([\d.]+)\)", [(CUR + "280|D0 as proposed/F_lim=300/ci_lo", "cap"), (CUR + "280|D0 as proposed/F_lim=300/ci_hi", "req")]),
    ("r5.1 deck headline slide CI", "deck", r"\(lock 0\.28 s\): [\d.]+% \(Wilson CI ([\d.]+)–([\d.]+)%\)", [(PAS + "280/ci_lo", "cap"), (PAS + "280/ci_hi", "req")]),
    ("r5.1 deck slide 18 CI (ci.csv) B1 falls", "deck", r"Threshold lock 0\.05 \| [\d.]+% \[([\d.]+)–([\d.]+)\] \|",
     [(lambda: _csv_ci("B1 theta=0.05", "fall_pct", "ci_lo"), "cap"), (lambda: _csv_ci("B1 theta=0.05", "fall_pct", "ci_hi"), "req")]),
    ("r5.1 deck slide 18 CI (ci.csv) B2 falls", "deck", r"Spring 200 N/m \| [\d.]+% \[([\d.]+)–([\d.]+)\] \|",
     [(lambda: _csv_ci("B2 K=200", "fall_pct", "ci_lo"), "cap"), (lambda: _csv_ci("B2 K=200", "fall_pct", "ci_hi"), "req")]),
    ("r5.1 deck slide 18 CI (ci.csv) CP 0.15 falls", "deck", r"CatchPoint 0\.15 \| [\d.]+% \[([\d.]+)–([\d.]+)\] \|",
     [(lambda: _csv_ci("CP m=0.15", "fall_pct", "ci_lo"), "cap"), (lambda: _csv_ci("CP m=0.15", "fall_pct", "ci_hi"), "req")]),
    ("r5.1 deck slide 18 CI (ci.csv) B1 solo", "deck", r"Threshold lock 0\.05 \| [\d.]+% \[[\d.]+–[\d.]+\] \| [\d.]+% \[([\d.]+)–([\d.]+)\]",
     [(lambda: _csv_ci("B1 theta=0.05", "self_pct", "ci_lo"), "cap"), (lambda: _csv_ci("B1 theta=0.05", "self_pct", "ci_hi"), "req")]),
    # ---- Round-5 item 2: section 4b table
    ("r5.2 uniform150 along/res/SF/lift", "report", r"Uniform T_slip = 150 N, passive \| (\d+) N \| (\d+) N \| ([\d.]+) \| (\d+) N",
     [("S/uniform/150/Fpar_min", "cap"), ("S/uniform/150/Fh_max", "req"), ("S/uniform/150/sf_min", "cap"), ("S/uniform/150/lift_max", "req")]),
    ("r5.2 uniform190 along/res/SF/lift", "report", r"Uniform T_slip = 190 N, passive \| (\d+) N \| (\d+) N \(> \d+\) \| ([\d.]+) \| (\d+) N",
     [("S/uniform/190/Fpar_min", "cap"), ("S/uniform/190/Fh_max", "req"), ("S/uniform/190/sf_min", "cap"), ("S/uniform/190/lift_max", "req")]),
    ("r5.2 group-tuned along/res/SF/lift", "report", r"\*\*Group-tuned passive[^|]*\*\* \| \*\*(\d+) N\*\* \| \*\*(\d+) N\*\* \| ([\d.]+) \| (\d+) N",
     [("S/group_tuned/Fpar_min", "cap"), ("S/group_tuned/Fh_max", "req"), ("S/group_tuned/sf_min", "cap"), ("S/group_tuned/lift_max", "req")]),
    ("r5.2 controller QP exact", "report", r"exact direction \| (\d+) N \| (\d+) N \| ([\d.]+) \| (\d+) N",
     [("S/controller_qp/0/Fpar_min", "cap"), ("S/controller_qp/0/Fh_max", "req"), ("S/controller_qp/0/sf_min", "cap"), ("S/controller_qp/0/lift_max", "req")]),
    ("r5.2 hybrid", "report", r"Hybrid [^|]*\| (\d+) N \(0 deg error\) \| (\d+) N \| ([\d.]+) \| (\d+) N",
     [("S/hybrid/0/Fpar_min", "cap"), ("S/hybrid/0/Fh_max", "req"), ("S/hybrid/0/sf_min", "cap"), ("S/hybrid/0/lift_max", "req")]),
    # ---- Round-5 item 3: nudge
    ("r5.3 nudge Tmax/lift/SF (report)", "report", r"T_max ([\d.]+) N, lift ≤ ([\d.]+) N, tip SF ≥ ([\d.]+), sliding SF ≥ ([\d.]+)\.",
     [("R/Tmax_nudge_worst", "req"), (lambda: max(r["lift"] for r in D_.R["nudge_rows"]), "req"),
      (lambda: min(r["sf_tip"] for r in D_.R["nudge_rows"]), "cap"), (lambda: min(r["sf_slide"] for r in D_.R["nudge_rows"]), "cap")]),
    ("r5.3 nudge contract row (report)", "report", r"\| Nudge \| 150 N resultant, T_max ([\d.]+) N per cable, tip SF >= ([\d.]+) \|",
     [("R/Tmax_nudge_worst", "req"), (lambda: min(r["sf_tip"] for r in D_.R["nudge_rows"]), "cap")]),
    ("r5.3 nudge (deck brake table)", "deck", r"T_max ([\d.]+) N per cable \| Servo torque [\d.]+ N·m at the spool; tip SF ≥ ([\d.]+)",
     [("R/Tmax_nudge_worst", "req"), (lambda: min(r["sf_tip"] for r in D_.R["nudge_rows"]), "cap")]),
    ("r5.3 nudge lift/SF (deck FBD slide)", "deck", r"tip safety factor [\d.]+ \(AP\) and ([\d.]+) \(ML\); lift on the pelvis ≤ ([\d.]+) N",
     [(lambda: next(r for r in D_.R["nudge_rows"] if r["phi"] == 90)["sf_tip"], "cap"), (lambda: max(r["lift"] for r in D_.R["nudge_rows"]), "req")]),
    # ---- Round-5 item 4: hardware ratings
    ("r5.4 hold torques corner/lateral (report table)", "report", r"\| corner \| rearL, rearR, frontL, frontR \| \d+ N \| ([\d.]+) N m \| ([\d.]+) N m \|",
     [(BR + "corner/slip_torque", "req"), (BR + "corner/rated_hold_torque", "req")]),
    ("r5.4 hold torque lateral (report table)", "report", r"\| lateral \| latL, latR \| \d+ N \| ([\d.]+) N m \| ([\d.]+) N m \|",
     [(BR + "lateral/slip_torque", "req"), (BR + "lateral/rated_hold_torque", "req")]),
    ("r5.4 hold torque centre (report table)", "report", r"\| centre \| rearC, frontC \| \d+ N \| ([\d.]+) N m \| ([\d.]+) N m \|",
     [(BR + "centre/slip_torque", "req"), (BR + "centre/rated_hold_torque", "req")]),
    ("r5.4 BWS hold torque (report)", "report", r"\| bws \| not a slip spool \| - \| ([\d.]+) N m \|", [(BWS, "req")]),
    ("r5.4 worst QP tension / current-set torque (report)", "report", r"max QP tension ([\d.]+) N \| - \| ([\d.]+) N m",
     [("R/Tmax_arrest_worst", "req"), (BR + "controller (current-set) brake rating/rated_hold_torque", "req")]),
    ("r5.4 worst QP tension (report brake line)", "report", r"\(1\.5 × the ([\d.]+) N worst QP tension", [("R/Tmax_arrest_worst", "req")]),
    ("r5.4 contract hold torques (report)", "report", r"rated hold ([\d.]+) / ([\d.]+) / ([\d.]+) N m \|",
     [(BR + "corner/rated_hold_torque", "req"), (BR + "lateral/rated_hold_torque", "req"), (BR + "centre/rated_hold_torque", "req")]),
    ("r5.4 hold torques (summary)", "summary", r"rated hold torque ([\d.]+) / ([\d.]+) / ([\d.]+) N m",
     [(BR + "corner/rated_hold_torque", "req"), (BR + "lateral/rated_hold_torque", "req"), (BR + "centre/rated_hold_torque", "req")]),
    ("r5.4 hold torques (deck)", "deck", r"Rated hold torque \| ([\d.]+) / ([\d.]+) / ([\d.]+) N·m",
     [(BR + "corner/rated_hold_torque", "req"), (BR + "lateral/rated_hold_torque", "req"), (BR + "centre/rated_hold_torque", "req")]),
    ("r5.4 slip torques (deck)", "deck", r"Slip torque \(at the spool\) \| ([\d.]+) / ([\d.]+) / ([\d.]+) N·m",
     [(BR + "corner/slip_torque", "req"), (BR + "lateral/slip_torque", "req"), (BR + "centre/slip_torque", "req")]),
    ("r5.4 worst QP tension (deck)", "deck", r"Worst cable tension, 300 N arrest \| ([\d.]+) N", [("R/Tmax_arrest_worst", "req")]),
    ("r5.4 BWS hold torque (deck)", "deck", r"BWS spool brake \| ([\d.]+) N·m", [(BWS, "req")]),
    ("brake torque 7.49 / servo (report)", "report", r"r_s = 25 mm: ([\d.]+) N·m .*Nudge servo: ([\d.]+) N·m at\nspool, ([\d.]+) N·m motor, ([\d.]+) kg reflected",
     [("R/spool_brake/r_s=0.025/tau_brake", "req"), ("R/spool_brake/r_s=0.025/tau_spool_nudge", "req"),
      ("R/spool_brake/r_s=0.025/tau_motor", "req"), ("R/spool_brake/r_s=0.025/m_reflected", "req")]),
    ("slip-brake energy (report)", "report", r"≤ ([\d.]+) J over the three\nlock delays \(([\d.]+) J at",
     [(lambda: max(D_.R["dyn"]["passive"][k]["brake_energy_max_J"] for k in D_.R["dyn"]["passive"]), "req"), (PAS + "280/brake_energy_max_J", "req")]),
    ("slip-brake energy (deck)", "deck", r"Slip-brake energy \| ≤ ([\d.]+) J \(([\d.]+) J at 0\.28 s\)",
     [(lambda: max(D_.R["dyn"]["passive"][k]["brake_energy_max_J"] for k in D_.R["dyn"]["passive"]), "req"), (PAS + "280/brake_energy_max_J", "req")]),
    ("hang load (deck)", "deck", r"hang load (\d+) N", [("R/hang/overhead x=0.42 (cantilevered top beam) | D3 recommended/T", "req")]),
    # ---- Round-5 item 5: force limits
    ("r5.5 trial along-force minimum (report)", "report", r"they deliver ([\d.]+)-[\d.]+ N along the fall", [(PAS + "280/Fpar_min", "cap")]),
    ("r5.5 trial along-force minimum (summary)", "summary", r"\(([\d.]+) N minimum over the 814 trials", [(PAS + "280/Fpar_min", "cap")]),
    ("r5.5 trial along-force maximum (report)", "report", r"they deliver [\d.]+-([\d.]+) N along the fall", [(PAS + "280/Fpar/max", "req")]),
    ("r5.5 mu=0.3 slide limit (report 1.6)", "report", r"D3 slides at (\d+) N if the pads give mu = 0\.3", [("R/slide_limit/mu=0.3/F_slide", "cap")]),
    ("r5.5 slide limits (report sec 5)", "report", r"walker slides at (\d+) N \((\d+) N at μ 0\.3\)", [("R/slide_limit/mu=0.5/F_slide", "cap"), ("R/slide_limit/mu=0.3/F_slide", "cap")]),
    ("r5.5 slide limits (summary)", "summary", r"walker slides at (\d+) N \((\d+) N if mu = 0\.3\)", [("R/slide_limit/mu=0.5/F_slide", "cap"), ("R/slide_limit/mu=0.3/F_slide", "cap")]),
    ("r5.5 slide limits (deck)", "deck", r"the walker slides at (\d+) N \((\d+) N if mu = 0\.3\)", [("R/slide_limit/mu=0.5/F_slide", "cap"), ("R/slide_limit/mu=0.3/F_slide", "cap")]),
    ("r5.5 friction capacity minimum (report 1.4)", "report", r"capacity ≥ (\d+) N at μ = 0\.5", [(_minimum_friction, "cap")]),
    ("r5.5 backward friction cap (report table)", "report", r"\| backward \|[^\n]*\| [\d.]+ \(rear\) \| \d+ / (\d+) N \|", [(lambda: D_.K["mu_pad"] * D_.R["arrest_rows"][1]["N_tot"], "cap")]),
    ("r5.5 forward friction cap (report table)", "report", r"\| forward \|[^\n]*\| [\d.]+ \(front\) \| \d+ / (\d+) N \|", [(lambda: D_.K["mu_pad"] * D_.R["arrest_rows"][0]["N_tot"], "cap")]),
    ("r5.5 left friction cap (report table)", "report", r"\| left \|[^\n]*\| [\d.]+ \(left\) \| \d+ / (\d+) N \|", [(lambda: D_.K["mu_pad"] * D_.R["arrest_rows"][2]["N_tot"], "cap")]),
    ("r5.5 friction cap (deck FBD)", "deck", r"friction needed 300 N against a cap of (\d+) N \(AP\)", [(lambda: D_.K["mu_pad"] * D_.R["arrest_rows"][0]["N_tot"], "cap")]),
    ("r5.5 resultant wording (report)", "report", r"and ≤ (\d+) N \(([\d.]+) N\) horizontal resultant", [("S/group_tuned/Fh_max", "req"), ("S/group_tuned/Fh_max", "req")]),
    ("r5.5 resultant wording (summary)", "summary", r"and <= (\d+) N horizontal resultant \(([\d.]+) N\)", [("S/group_tuned/Fh_max", "req"), ("S/group_tuned/Fh_max", "req")]),
    ("r5.5 resultant (deck)", "deck", r"worst-case horizontal resultant (\d+) N", [("S/group_tuned/Fh_max", "req")]),
    ("r5.5 all-slip bound (report)", "report", r"is ([\d.]+) N \(≤ \d+ N\) for the", [("S/group_tuned/upper_bound_all_slip", "req")]),
    ("r5.5 contract JSON bound", "contract", r"\"resultant_bound_all_slip\": (\d+)", [("S/group_tuned/upper_bound_all_slip", "req")]),
    ("r5.5 contract JSON wording", "contract", r"<= resultant_bound_all_slip \((\d+) N\)", [("S/group_tuned/Fh_max", "req")]),
    ("r5.5 contract hold torque corner", "contract", r"\"corner\": ([\d.]+),\n  \"lateral\": [\d.]+,\n  \"centre\": [\d.]+,\n  \"bws\"", [(BR + "corner/rated_hold_torque", "req")]),
    ("r5.5 contract BWS hold torque", "contract", r"\"bws\": ([\d.]+)\n", [(BWS, "req")]),
    ("r5.5 contract lift cap nudge", "contract", r"\"lift_cap_nudge_N\": ([\d.]+)", [("R/params/CAP_NUDGE", "cap")]),
    # ---- Round-5 item 6: mass / width
    ("r5.6 mixed-height mass at 0.325 m", "report", r"Mixed-height cables, 300 N, SF 1\.5 → ≈([\d.]+) kg at 0\.325 m",
     [(lambda: _mass_interp("high corner anchors (1.20 m) + mid anchors|F=300|SF=1.5", 0.325), "req")]),
    ("r5.6 mixed-height mass at 0.40 m", "report", r"at 0\.325 m, ≈([\d.]+) kg at\n0\.40 m", [(lambda: _mass_interp("high corner anchors (1.20 m) + mid anchors|F=300|SF=1.5", 0.40), "req")]),
    ("r5.6 horizontal-cable mass at 0.325 m", "report", r"SF 1 → ≈(\d+) kg at 0\.325 m",
     [(lambda: _mass_interp("horizontal cables (z = 0.95 m)|F=491|SF=1.0", 0.325), "req")]),
    # ---- headline numbers
    ("headline coverage % (report 1.5)", "report", r"arrest 805/814 = ([\d.]+) % \(CI", [(lambda: 100 * 805 / 814, "cap")]),
    ("headline coverage % (summary)", "summary", r"slide <= 1 cm, no dynamic tip\) = ([\d.]+) %", [(lambda: 100 * 805 / 814, "cap")]),
    ("headline coverage % (deck)", "deck", r"\(lock 0\.28 s\): ([\d.]+)% \(Wilson", [(lambda: 100 * 805 / 814, "cap")]),
    ("fixed coverage % (report)", "report", r"fixed 300 N along the fall, QP-distributed tensions\): 804/814 = ([\d.]+) %", [(PAS + "280/fixed300_pct", "cap")]),
    ("coverage 0.15 passive % (summary)", "summary", r"0\.15 s 821/844 = ([\d.]+) %", [(PAS + "150/pct", "cap")]),
    ("coverage 0.08 passive % (summary)", "summary", r"0\.08 s 817/866 = ([\d.]+) %", [(PAS + "080/pct", "cap")]),
    ("D0 coverage % (summary)", "summary", r"377/814 = ([\d.]+) %", [(CUR + "280|D0 as proposed/F_lim=300/good*100", "cap")]),
    ("resultant, along-fall min, perp, ang (report 1.4)", "report",
     r"\(([\d]+) N minimum over 360 directions\) and ≤ \d+ N \([\d.]+ N\) horizontal resultant \(the perpendicular component is up to (\d+) N;\n   the resultant deviates up to (\d+) deg",
     [("S/group_tuned/Fpar_min", "cap"), ("S/group_tuned/perp_max", "req"), ("S/group_tuned/ang_max", "req")]),
    ("tip SF / sliding SF / lift (report 1.4)", "report", r"Tip SF ≥ ([\d.]+) and sliding SF ≥ ([\d.]+) at the full slip resultant", [("S/group_tuned/sf_min", "cap"), ("S/group_tuned/sf_slide_min", "cap")]),
    ("pelvis lift under slip (report 1.4)", "report", r"Pelvis lift ≤ (\d+) N under passive slip", [("S/group_tuned/lift_max", "req")]),
    ("tip SF / sliding SF (summary)", "summary", r"tip SF >= ([\d.]+) and sliding SF >= ([\d.]+) at the full", [("S/group_tuned/sf_min", "cap"), ("S/group_tuned/sf_slide_min", "cap")]),
    ("tip SF / sliding SF (deck)", "deck", r"Tip SF ≥ ([\d.]+), sliding SF ≥ ([\d.]+) at the full resultant", [("S/group_tuned/sf_min", "cap"), ("S/group_tuned/sf_slide_min", "cap")]),
    ("D3 envelope row (report 4)", "report", r"\*\*D3 recommended\*\* \|[^|]*\|[^|]*\| \*\*(\d+) N\*\* \| \*\*(\d+) N\*\* \| (\d+) N \|",
     [("R/envelope/D3 recommended/min", "cap"), ("R/envelope_sf1/D3 recommended/min", "cap"), ("R/slide/D3 recommended/F_slide", "cap")]),
    ("D1 envelope row (report 4)", "report", r"\| D1 high anchors \|[^|]*\|[^|]*\| (\d+) N \| (\d+) N \| (\d+) N \|",
     [("R/envelope/D1 high anchors/min", "cap"), ("R/envelope_sf1/D1 high anchors/min", "cap"), ("R/slide/D1 high anchors/F_slide", "cap")]),
    ("D2 envelope row (report 4)", "report", r"\| D2 \+ ballast \|[^|]*\|[^|]*\| (\d+) N \| (\d+) N \| (\d+) N \|",
     [("R/envelope/D2 + ballast/min", "cap"), ("R/envelope_sf1/D2 + ballast/min", "cap"), ("R/slide/D2 + ballast/F_slide", "cap")]),
    ("D0 envelope row (report 4)", "report", r"\| D0 as proposed \|[^|]*\|[^|]*\| (\d+) N \| (\d+) N \| (\d+) N \|",
     [("R/envelope/D0 as proposed/min", "cap"), ("R/envelope_sf1/D0 as proposed/min", "cap"), ("R/slide/D0 as proposed/F_slide", "cap")]),
    ("envelope (summary 7)", "summary", r"envelope (\d+) N \(SF 1\.0: (\d+) N\)", [("R/envelope/D3 recommended/min", "cap"), ("R/envelope_sf1/D3 recommended/min", "cap")]),
    ("lift cap (report 4 header)", "report", r"lift cap (\d+) N; SF =", [("R/params/CAP_ARREST", "cap")]),
    ("arrest rows tension/Tmax/lift (report): forward", "report", r"\| forward \| rearL (\d+), rearR \d+, rearC (\d+) \| (\d+) \| (\d+) N \|",
     [("R/arrest_rows/0/T/rearL", "req"), ("R/arrest_rows/0/T/rearC", "req"), ("R/arrest_rows/0/Tmax", "req"), ("R/arrest_rows/0/lift", "req")]),
    ("arrest rows Tmax/lift/SF (report): left", "report", r"\| left \| rearR \d+, frontR \d+, latR (\d+), rearC \d+ \| (\d+) \| (\d+) N \| ([\d.]+) \(left\)",
     [("R/arrest_rows/2/T/latR", "req"), ("R/arrest_rows/2/Tmax", "req"), ("R/arrest_rows/2/lift", "req"), ("R/arrest_rows/2/sf_tip", "cap")]),
    ("arrest rows SF (report): forward/backward", "report", r"\| ([\d.]+) \(front\) \|.*\n\| backward \|[^\n]*\| ([\d.]+) \(rear\)",
     [("R/arrest_rows/0/sf_tip", "cap"), ("R/arrest_rows/1/sf_tip", "cap")]),
    ("arrest SF/lift (deck FBD)", "deck", r"Arrest, 300 N, controller-set tensions: tip safety factor ([\d.]+) \(AP\) and ([\d.]+) \(ML\)",
     [("R/arrest_rows/0/sf_tip", "cap"), ("R/arrest_rows/2/sf_tip", "cap")]),
    ("pelvis lift under slip (deck)", "deck", r"\(≤ (\d+) N under passive slip\)", [("S/group_tuned/lift_max", "req")]),
    ("sensitivity: outriggers not deployed (report)", "report", r"outriggers not deployed (\d+);", [("R/sensitivity/outriggers not deployed", "cap")]),
    ("required force share <=300 N (report 3)", "report", r"\(rounded down\): [\d.]+ / [\d.]+ / ([\d.]+) / ([\d.]+) %", [("R/loads/280/Freq_frac_le/300*100", "cap"), ("R/loads/280/Freq_frac_le/400*100", "cap")]),
    ("max catch force / impulse (report 3)", "report", r"\| Sim catch-force peak \[N\] \| \d+ \| \d+ \| ([\d.]+) \|", [("R/loads/280/F_peak/max", "req")]),
    ("max COM distance / KE (report 3)", "report", r"\| COM distance from support \[cm\] \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", [("R/loads/280/s0/max*100", "req")]),
    ("max hold force (report 3)", "report", r"\| Hold force at t = 3 s \[N\] \| \d+ \| \d+ \| (\d+) \|", [("R/loads/280/F_hold/max", "req")]),
    ("max impulse (report 3)", "report", r"first 0\.3 s \[N·s\] \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", [("L28/stats/imp_300/max", "req")]),
    ("worst-case falls pct (report 1.5)", "report", r"= 104/2000 = ([\d.]+) % at worst", [(lambda: 100 * 104 / 2000, "req")]),
    ("p95 stop time (report 1.7)", "report", r"p95 ([\d.]+) s, rounded up", [(lambda: D_.L28["tau_lock"] + D_.R["dyn"]["passive"]["280"]["t_stop"]["p95"], "req")]),
]


def _decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def _resolve(src):
    return src() if callable(src) else J(src)


@pytest.fixture(scope="module")
def files():
    d = {k: _text(v) for k, v in FILES.items()}
    d["contract"] = _text("analysis/i2_sizing/data/d3_contract.json")
    return d


@pytest.mark.parametrize("label,fkey,regex,checks", AUDIT, ids=[a[0] for a in AUDIT])
def test_published_value_is_conservative(files, label, fkey, regex, checks):
    mo = re.search(regex, files[fkey])
    assert mo, f"{label}: pattern not found in {fkey}"
    assert len(mo.groups()) == len(checks), label
    for txt, (src, kind) in zip(mo.groups(), checks):
        pub, val = float(txt), _resolve(src)
        d = _decimals(txt)
        if kind == "req":
            assert pub >= val - TOL, f"{label}: published {txt} < source {val} (non-conservative for a requirement)"
        else:
            assert pub <= val + TOL, f"{label}: published {txt} > source {val} (non-conservative for a capacity)"
        assert abs(pub - val) < 10 ** -d + TOL, f"{label}: published {txt} is not within one unit of the source {val}"


def test_every_round5_blocker_has_an_audit_entry():
    labels = " ".join(a[0] for a in AUDIT)
    for tag in ("r5.1", "r5.2", "r5.3", "r5.4", "r5.5", "r5.6"):
        assert tag in labels, tag


def test_deck_dump_matches_json_for_the_headline_numbers(files):
    """The deck text dump is regenerated from the same JSON (deck/build_deck.py -> dump_text.py)."""
    deck = files["deck"]
    G = D_.S["group_tuned"]
    assert f"worst-case horizontal resultant {int(PB.req(G['Fh_max'], 0))} N" in deck
    assert f"{D_.R['dyn']['passive']['280']['k']}/{D_.R['dyn']['passive']['280']['n']}" in deck


# ------------------------------------------------------------------------------------------------ (d) no typed results
ALLOWED_TYPED = {  # model inputs, labels of simulated cases and literature values: stated, not computed
    "0 deg", "0.08 s", "0.10 m", "0.12 s", "0.14 s", "0.15 m", "0.15 s", "0.25 m", "0.3 s", "0.5 s", "1 deg", "1 mm", "1.20 m",
    "1.50 m", "100 kg", "110 N", "15 kN/m", "150 N", "1500 N", "190 N", "2 N", "250 N", "3 s", "30 deg", "300 N", "35 N", "45 deg",
    "5 N", "6 kg·m²", "60 kg", "800 N·s/m", "85 kg", "9 N", "400 N",
}
UNITS = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?:\s?(?:N·m|N m|N·s/m|N·s|N|J|%|kg·m²|kg|m/s|mm|cm|deg|kN/m|m|s|BW)\b)")


@pytest.mark.parametrize("tmpl", sorted(MR.OUTPUTS))
def test_templates_carry_no_hand_typed_results(tmpl):
    with open(os.path.join(MR.TEMPLATES, tmpl), encoding="utf8") as f:
        t = re.sub(r"\{[^{}]*\}", "", f.read())
    stray = {m.group(0) for m in UNITS.finditer(t)} - ALLOWED_TYPED
    assert not stray, f"hand-typed numbers in {tmpl}: {sorted(stray)} (use a placeholder or add to ALLOWED_TYPED if it is a model input)"
