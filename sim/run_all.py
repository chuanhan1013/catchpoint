"""Regenerate every SPEC 5.6 / 6 result from scratch (seed 5412).  Usage: python sim/run_all.py"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figs")

from catchpoint import figures as fg                                   # noqa: E402
from catchpoint.controllers import B0, B1, B2, CP                      # noqa: E402
from catchpoint.model import (CAUGHT, FALL, SELF, ASSISTED, Config, build_pop, params_json,  # noqa: E402
                              sample_noise, simulate, G)
from catchpoint.montecarlo import (SEED, bootstrap_ci, dose_sessions, delay_sweep, full_grid,   # noqa: E402
                                   make_population, matched_table)

T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)


def showcase():
    """Moderate-to-severe patient (s=0.6, left leg paretic), 0.55 m/s push at 60 deg from forward, i.e. mostly
    lateral toward the paretic (left) side.  Chosen (see summary.md) so that B0 falls."""
    ang = np.radians(60.0)
    dv = [[0.55 * np.cos(ang), 0.55 * np.sin(ang)]]
    noise = sample_noise(np.random.default_rng(SEED), 1)
    pop = build_pop([0.6], 0, 72.0, 0.92, dv, noise=noise)
    ctrls = {"B0": B0(), "B1": B1(0.08), "B2": B2(800), "CP": CP(0.03)}
    out = {}
    for k, c in ctrls.items():
        r = simulate(pop, c, Config(), record=True)
        d = r["rec"]
        d["outcome"] = ["SELF", "ASSISTED", "CAUGHT", "FALL"][int(r["outcome"][0])]
        d["_par"] = 0; d["_scale"] = pop["scale"][0]; d["_z0"] = float(pop["z0"][0])
        d["params"] = params_json(pop)
        d["impulse"] = float(r["impulse"][0])
        out[k] = d
        log(f"showcase {k}: {d['outcome']} impulse {d['impulse']:.1f}")
    return pop, out


def save_npz(show):
    keys = ["t", "com", "comvel", "xi", "foot_l", "foot_r", "swing", "F", "level", "swing_h"]
    for k, d in show.items():
        np.savez(os.path.join(RES, f"showcase_{k}.npz"),
                 **{kk: (d[kk].astype(int) if kk in ("swing", "level") else d[kk]) for kk in keys},
                 outcome=np.array(d["outcome"]), params=np.array(d["params"]))


def write_csv(rows, path):
    cols = ["setting", "n", "fall_pct", "self_pct", "assisted_pct", "caught_pct", "unnecessary_pct",
            "unnecessary_of_b0self_pct", "rescued_pct", "mean_impulse", "mean_steps", "mean_peak_F",
            "median_t_decision", "median_t_force", "median_t_engage"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["setting"]] + [f"{r[c]:.3f}" if isinstance(r[c], float) or hasattr(r[c], "dtype") else r[c]
                                         for c in cols[1:]])


def sizing(res, path, cfg):
    r = res["CP m=0.03"]
    pk, dF, v1, pc = r["peakF"], r["peakdF"], r["peakv1"], r["peakCatch"]
    act = pk > 0.5
    p = lambda a, q=99: float(np.percentile(a, q))
    lead, eta, Fmax = 0.005, 0.9, 150.0
    tq = lambda F: F * lead / (2 * np.pi * eta)
    F99, F99a = p(pk), p(pk[act])
    dF99, dF99a = p(dF), p(dF[act])
    v99a = p(v1[act])
    k_sea = Fmax / 0.010
    rpm = lambda v: 60 * v / lead
    winch = 1.5 * 85 * G
    cau = res["CP m=0.03"]["engaged"]
    pc99 = p(pc[cau]) if cau.any() else float("nan")
    md = f"""# Actuator sizing (from CP Monte Carlo, default m = 0.03, 2000 trials, seed {SEED})

Per-trial peaks are taken over each trial (while the patient has not fallen); percentiles are across trials.
"All" = all 2000 trials; "L1-active" = the {int(act.sum())} trials in which the pelvic force exceeded 0.5 N.
The pelvic SEA force excludes the harness catch force.

| Quantity | All trials | L1-active trials |
|---|---|---|
| 99th-pct peak pelvic force |F| [N] | {F99:.0f} | {F99a:.0f} |
| 99th-pct peak force rate |dF/dt| [N/s] | {dF99:.0f} | {dF99a:.0f} |
| 99th-pct peak pelvis velocity during L1 [m/s] | - | {v99a:.2f} |
| max over trials of peak |F| [N] | {pk.max():.0f} | |

Note: the commanded force is saturated at F_max = {Fmax:.0f} N by the arbiter (above that the harness catch is requested),
so the 99th percentile sits at or just below the limit by construction; it is a design check (the SEA must deliver
F_max), not a free estimate of demand.

## Ball-screw SEA (per side, AP or ML axis), lead {lead * 1000:.0f} mm, eta = {eta}
- Motor torque tau = F * lead / (2 pi eta): at F_max = {Fmax:.0f} N -> **{tq(Fmax):.3f} N m**; at the 99th-pct force
  ({F99a:.0f} N, L1-active) -> {tq(F99a):.3f} N m.
- Screw/motor speed for the 99th-pct pelvis velocity during L1 ({v99a:.2f} m/s) if the carriage must follow the pelvis
  rigidly: {rpm(v99a):.0f} rpm (n = v/lead = {v99a / lead:.0f} rev/s). That is high for a 5 mm lead; the series spring
  decouples the motor from the pelvis (the motor only has to move the spring end), so this is an upper bound.
- Peak mechanical power at F_max and v = {v99a:.2f} m/s: {Fmax * v99a:.0f} W per axis (upper bound, same assumption).
- Spring stiffness for 10 mm deflection at F_max: k = {Fmax:.0f} N / 0.010 m = **{k_sea:.0f} N/m**.
- Force-rate demand: 99th-pct |dF/dt| = {dF99a:.0f} N/s (L1-active) -> spring-end speed {dF99a / k_sea * 1000:.0f} mm/s
  ({rpm(dF99a / k_sea):.0f} rpm on the screw), which the motor must supply in addition to nothing else (SEA deflection rate).
  The sim does not resolve force-control bandwidth; the SPEC value (>= 5 Hz) is taken as given.

## Catch winch
- Static holding requirement 1.5 m g at m = 85 kg: **{winch:.0f} N** ({winch / G:.0f} kgf).
- Simulated harness force (spring-damper, K_c = {cfg.Kc:.0f} N/m, D_c = {cfg.Dc:.0f} N s/m): 99th-pct peak over the
  {int(cau.sum())} CAUGHT trials = {pc99:.0f} N; max = {pc[cau].max() if cau.any() else float('nan'):.0f} N
  ({'above' if cau.any() and p(pc[cau]) > winch else 'below'} the static 1.5 m g requirement, i.e. the dynamic
  catch peak {'drives' if cau.any() and p(pc[cau]) > winch else 'does not drive'} the winch/brake rating).
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return dict(F99=F99, F99a=F99a, v99a=v99a, pc99=pc99, winch=winch)


def fam(rows, prefix):
    return [r for r in rows if r["setting"].startswith(prefix)]


def interpretation(rows, stats, sweep, aan, matched, ci):
    by = {r["setting"]: r for r in rows}
    b0, cp, nodc, pop_ = by["B0 none"], by["CP m=0.03"], by["CP-noDC m=0.03"], by["CP-pop m=0.03"]
    cp15 = by["CP m=0.15"]
    b1s, xs, b2s = fam(rows, "B1 theta="), fam(rows, "B1-XCoM theta="), fam(rows, "B2 K=")
    b2_800 = by["B2 K=800"]
    b2_low = min(b2s, key=lambda r: r["fall_pct"])
    cid = {(c["setting"], c["metric"]): c for c in ci}
    ciText = lambda s, m: "%.1f%% [%.1f, %.1f]" % (cid[(s, m)]["estimate"], cid[(s, m)]["ci_lo"], cid[(s, m)]["ci_hi"])
    mt = {(m["family"], m["fall_target"]): m["self_pct_interp"] for m in matched}
    fmt = lambda v: "n/a (not bracketed)" if not np.isfinite(v) else "%.1f%%" % v
    lines = []
    lines.append(
        f"1. DEFAULT comparison (CP m=0.03, B1 theta=0.05, B2 K=200 are single sweep points; CP default was fixed in SPEC): CP m=0.03 falls "
        f"{ciText('CP m=0.03', 'fall_pct')} and keeps SELF {ciText('CP m=0.03', 'self_pct')} (B0: falls {ciText('B0 none', 'fall_pct')}, SELF "
        f"{ciText('B0 none', 'self_pct')}; paired patient bootstrap 95% CI). B1 theta=0.05 falls {ciText('B1 theta=0.05', 'fall_pct')} with SELF "
        f"{ciText('B1 theta=0.05', 'self_pct')}; B2 K=200 falls {ciText('B2 K=200', 'fall_pct')} with SELF {ciText('B2 K=200', 'self_pct')}.")
    r4 = mt[("CP", 4.0)] / mt[("B1", 4.0)] if np.isfinite(mt[("CP", 4.0)]) and np.isfinite(mt[("B1", 4.0)]) and mt[("B1", 4.0)] > 0 else float("nan")
    lines.append(
        f"2. MATCHED-FALL comparison (SELF interpolated along each sweep at a common fall rate; matched.csv): at 4% falls SELF = B1 {fmt(mt[('B1', 4.0)])}, "
        f"B1-XCoM {fmt(mt[('B1-XCoM', 4.0)])}, B2 {fmt(mt[('B2', 4.0)])}, CP {fmt(mt[('CP', 4.0)])}, CP-pop {fmt(mt[('CP-pop', 4.0)])}; at 6% falls "
        f"B1 {fmt(mt[('B1', 6.0)])}, B1-XCoM {fmt(mt[('B1-XCoM', 6.0)])}, B2 {fmt(mt[('B2', 6.0)])}, CP {fmt(mt[('CP', 6.0)])}, CP-pop {fmt(mt[('CP-pop', 6.0)])}. "
        f"CP/B1 at 4% = {r4:.1f}x. CP m=0.15 (the sweep point nearest 4% falls) has falls {cp15['fall_pct']:.1f}% / SELF {cp15['self_pct']:.1f}%; "
        f"this is a post-selection operating point, not the default.")
    best_x = max(xs, key=lambda r: r["self_pct"] - 2 * r["fall_pct"])
    lines.append(
        f"3. Ablations: CP-pop (population-mean model, no identification) at m=0.03 falls {pop_['fall_pct']:.1f}% with SELF {pop_['self_pct']:.1f}% vs CP "
        f"{cp['fall_pct']:.1f}% / {cp['self_pct']:.1f}%, so the patient model is worth {pop_['fall_pct'] - cp['fall_pct']:+.1f} pp falls and "
        f"{cp['self_pct'] - pop_['self_pct']:+.1f} pp SELF. B1-XCoM (capture-point threshold, no reach model or graded force) spans falls "
        f"{min(r['fall_pct'] for r in xs):.1f}-{max(r['fall_pct'] for r in xs):.1f}% with SELF {min(r['self_pct'] for r in xs):.1f}-{max(r['self_pct'] for r in xs):.1f}% "
        f"(e.g. {best_x['setting']}: {best_x['fall_pct']:.1f}% / {best_x['self_pct']:.1f}%): the capture-point idea alone explains most of the SELF gain over COM-distance thresholds at equal falls (matched.csv: B1-XCoM {fmt(mt[('B1-XCoM', 4.0)])} vs CP {fmt(mt[('CP', 4.0)])} at 4% falls), "
        f"so CP's remaining advantage over B1-XCoM is mainly less harness use and impulse (at similar falls B1-XCoM theta=0.08: CAUGHT {by['B1-XCoM theta=0.08']['caught_pct']:.0f}%, impulse {by['B1-XCoM theta=0.08']['mean_impulse']:.0f} N s, unnecessary {by['B1-XCoM theta=0.08']['unnecessary_pct']:.1f}% vs CP m=0.03: {cp['caught_pct']:.0f}%, {cp['mean_impulse']:.0f} N s, {cp['unnecessary_pct']:.1f}%). The delay-compensation ablation CP-noDC falls {nodc['fall_pct']:.1f}% (vs {cp['fall_pct']:.1f}%) but has fewer unnecessary "
        f"interventions ({nodc['unnecessary_pct']:.1f}% vs {cp['unnecessary_pct']:.1f}%).")
    hi_b1 = [r for r in b1s if r["setting"] in ("B1 theta=0.14", "B1 theta=0.18")]
    worse = [f"CP falls more than B2 K>=400 ({cp['fall_pct']:.1f}% vs {by['B2 K=400']['fall_pct']:.1f}%, K=800 {b2_800['fall_pct']:.1f}%)",
             f"CP ends {cp['caught_pct']:.1f}% of trials in a harness catch (L2), more than B1 theta>=0.14 ({', '.join('%.0f%%' % r['caught_pct'] for r in hi_b1)}), which however fall far more often",
             f"its mean impulse ({cp['mean_impulse']:.0f} N s) is {'above' if cp['mean_impulse'] > b2_800['mean_impulse'] else 'below'} B2 K=800 ({b2_800['mean_impulse']:.0f} N s) and above B1 theta>=0.14 ({by['B1 theta=0.14']['mean_impulse']:.0f} N s)",
             f"it intervenes unnecessarily in {cp['unnecessary_pct']:.1f}% of all trials ({cp['unnecessary_of_b0self_pct']:.0f}% of B0-SELF trials) vs B1 theta=0.18 {by['B1 theta=0.18']['unnecessary_pct']:.1f}%",
             f"in the severe tercile CP falls {stats[(2, 'CP')][3]:.0f}% and keeps {stats[(2, 'CP')][0]:.0f}% SELF (B0 SELF there: {stats[(2, 'B0')][0]:.0f}%)"]
    lines.append("4. Where CP is NOT better: " + "; ".join(worse) + ". In these simulations every non-caught B2 trial exceeded the 1 N s impulse threshold (SELF 0.0% is "
                 "an empirical result, not a definition). The harness remains the main safety mechanism: the push distribution is harsh (B0 falls in most trials), "
                 "so most rescues need more than the 150 N pelvic limit.")
    a = {r["delay"]: r["fall_pct"] for r in sweep if r["dc"]}
    b = {r["delay"]: r["fall_pct"] for r in sweep if not r["dc"]}
    lines.append(
        f"5. Caveats: simplified LIP model, assumed population ranges (SPEC 5.2), 10% identification error (one draw per patient, all model-using controllers share it), no sensor noise; "
        f"delay compensation (exact lag-aware prediction) lowers falls {b[0.1]:.1f}% -> {a[0.1]:.1f}% at 0.10 s and {b[0.3]:.1f}% -> {a[0.3]:.1f}% at 0.30 s; the adaptive-dose loop (CP safety layer) settles at "
        f"{', '.join('%.2f' % x for x in aan['dose'][-1])} m/s for mild/moderate/severe patients with {int(aan['fall_cp'].sum())} falls vs {int(aan['fall_b0'].sum())} for B0 on the same pushes "
        f"(3 patients, noisy). CP has privileged patient-model information the baselines lack (see ablation CP-pop). Not a clinical result.")
    return lines


def matched_text(matched):
    t = "| Family | Fall target % | SELF % (interpolated) | Bracketing settings |\n|---|---|---|---|\n"
    for m in matched:
        v = "n/a" if not np.isfinite(m["self_pct_interp"]) else "%.1f" % m["self_pct_interp"]
        t += f"| {m['family']} | {m['fall_target']:.0f} | {v} | {m['bracket']} |\n"
    return t


def ci_text(ci):
    t = "| Setting | Metric | Estimate % | 95% CI (patient bootstrap, 1000 reps) |\n|---|---|---|---|\n"
    for c in ci:
        t += f"| {c['setting']} | {c['metric']} | {c['estimate']:.1f} | [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}] |\n"
    return t


CAUSE_NAMES = ["(a) no crossover-valid step candidate at all (catch requested directly)",
               "(b) valid candidates exist but even the best needs F* > F_max, time-to-landing short (ongoing swing or T_land < 0.25 s)",
               "(c) valid candidates exist but even the best needs F* > F_max, long horizon (required force simply exceeds 150 N)"]


def cause_rows(res):
    out = []
    for lbl, r in res.items():
        if "cause" not in r:
            continue
        cz, eng = r["cause"], r["engaged"]
        row = dict(setting=lbl, n_decisions=int((cz >= 0).sum()), n_engaged=int(eng.sum()))
        for k, nm in enumerate("abc"):
            row[f"cause_{nm}_decided"] = int((cz == k).sum())
            row[f"cause_{nm}_engaged"] = int(((cz == k) & eng).sum())
        out.append(row)
    return out


def cause_text(res):
    r = res["CP m=0.03"]
    cz, eng = r["cause"], r["engaged"]
    n_dec, n_eng = int((cz >= 0).sum()), int(eng.sum())
    cnt = [int(((cz == k) & eng).sum()) for k in range(3)]
    cnt_all = [int((cz == k).sum()) for k in range(3)]
    t = (f"Cause recorded at the first tick the arbiter latched the catch decision (CP m=0.03). {n_dec} trials latched a decision, {n_eng} of them "
         f"engaged the catch (CAUGHT); counts are for the CAUGHT trials (all latched decisions in brackets). Whenever a catch is requested for causes (b)/(c) the best "
         f"candidate has h(0) < 0, i.e. no valid candidate is reach-feasible at F = 0; the (b)/(c) split by a 0.25 s horizon is a descriptive convention. "
         f"All settings are in catch_causes.csv.\n\n")
    for k in range(3):
        t += f"- {CAUSE_NAMES[k]}: {cnt[k]} ({100 * cnt[k] / max(n_eng, 1):.1f}%) [{cnt_all[k]}]\n"
    t += ("\nCP-sat (apply F_max, catch only if the predicted barrier with F_max is still < 0) is mathematically identical to the implemented arbiter "
          "(h(F_max u) < 0 exactly when |F*| > F_max; asserted in tests), so it was not added.")
    return t


def write_summary(rows, interp, devs, path, show_note, cause_md, matched, ci):
    hdr = "| Controller setting | Fall % | SELF % | ASSISTED % | CAUGHT % | Unnecessary % | Mean impulse [N s] |\n|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += (f"| {r['setting']} | {r['fall_pct']:.1f} | {r['self_pct']:.1f} | {r['assisted_pct']:.1f} | "
                 f"{r['caught_pct']:.1f} | {r['unnecessary_pct']:.1f} | {r['mean_impulse']:.1f} |\n")
    md = ("# CatchPoint simulation summary\n\n"
          "2000 trials per controller setting (50 patients x 40 pushes, seed 5412), identical patients and pushes across settings. Outcomes per SPEC 5.5. "
          "UNNECESSARY = B0 ended SELF for this patient and push AND the robot ever applied qualifying assistance (impulse >= 1 N s) or ever latched a catch decision, "
          "regardless of the controller's final outcome (a later FALL still counts); the % is of ALL trials, conditional rates are in summary.csv. "
          "Mean impulse includes pelvic and harness force. Threshold references: B1 and the B2 lock use the area-weighted centroid of the same scaled CoP region (S_cop) the plant uses; "
          "B1-XCoM uses the distance of the delayed identified-omega XCoM outside the support polygon; B2 damping uses the identified (noisy) mass. "
          "Intervention times (decision, applied force, catch engagement) are logged separately in summary.csv (medians).\n\n"
          + hdr + body + "\n## Interpretation\n\n" + "\n".join(interp)
          + "\n\n## Matched-fall comparison (interpolated SELF at common fall rate)\n\n" + matched_text(matched)
          + "\n## Paired bootstrap CIs\n\n" + ci_text(ci)
          + "\n## CP catch causes\n\n" + cause_md + "\n\n## Showcase push\n\n" + show_note
          + "\n\n## Deviations from SPEC\n\n" + "\n".join(f"- {d}" for d in devs) + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


def write_dicts(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([("%.4f" % r[c]) if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])


DEVIATIONS = [
    "dt = 0.002 s for every run (including Monte Carlo); the LIP is integrated with the exact (matrix-exponential) step for "
    "piecewise-constant CoP and force, which SPEC allows. Whole run_all takes several minutes, so dt was not coarsened.",
    "Support-region geometry: the convex hull of two axis-aligned rectangles and the projection onto it are exact; the ankle-strategy centroid c_S is the "
    "area-weighted mean of the two (scaled) foot-rectangle centres rather than the exact hull centroid (millimetres). B1 and the B2 lock reference this same c_S "
    "(review item 6; the earlier midpoint-of-feet reference was replaced).",
    "Delay sweep (F4): the total delay Delta = tau_s + tau_a is varied by keeping tau_s = 0.03 s fixed and setting tau_a = Delta - 0.03 s "
    "(SPEC does not say how Delta is split). The 0.02 s force lag is unchanged.",
    "Plant ordering: commanded force -> pure delay tau_a -> first-order lag 0.02 s. Once the catch is engaged all pelvic commands are "
    "set to zero (the harness holds); the already-queued delayed force still decays through the plant. The catch spring-damper force "
    "is added on top and is unbounded. Impulse = integral of |pelvic + harness force| until the fall (or 3 s).",
    "CatchPoint internals not fixed by SPEC (documented, not tuned against baselines): (i) delay-compensation forward prediction uses the projection of the measured "
    "xi onto the identified S_cop as constant CoP and propagates the applied force EXACTLY (logged applied force plus the pure-delay + first-order-lag recursion of the "
    "queued commands; verified against the plant in tests); (ii) the controller keeps its own violation timer on the predicted xi to estimate the patient's remaining reaction time "
    "(shifted by -tau_a with compensation, +tau_s without); (iii) during an ongoing swing the remaining swing time is floored at 0.05 s so beta does not vanish; "
    "(iv) candidates (2 legs x side-step/crossover) that violate the crossover rule are excluded (-inf); if no candidate is valid and xi-hat is outside S_cop the arbiter "
    "requests the catch directly; (v) the identified model multiplies m, omega, tau_r, both L_max, both T_s and the paretic ankle factor by (1 + N(0,0.1^2)), "
    "drawn once per patient (the same draw feeds B2 damping and B1-XCoM's omega); (vi) the arbiter latches the catch decision the first tick |F*| > 150 N, and the engine engages it 0.28 s later.",
    "Ablations added after review: CP-pop (population-mean model: tau_r 0.35 s, L_max 0.45 m and T_s 0.4625 s and ankle factor 0.825 for both legs, mass 72.5 kg, z0 0.925 m) and "
    "B1-XCoM (theta sweep 0.02-0.16 m). CP has privileged patient-model information that B1/B2 do not; these two controllers bound how much that matters.",
    "Patient model details: foot lift-off happens when the reaction timer expires (0.6 tau_r for later steps) and the timer restarts "
    "after each landing; the target of an infeasible step is clipped onto that candidate's reachable set. Fall check uses the support polygon at the start of each tick. "
    "Patients may keep stepping (<= 3 steps) after the catch engages.",
    "Crossover steps (coordinator correction): candidate with T_s x 1.25, L_max x 0.7, swing centre at least 0.09 m across the stance foot (side-steps keep 0.08 m). Patient prefers a "
    "feasible side-step, else a feasible crossover, else the least-infeasible candidate. CP evaluates the same 4 candidates with identified parameters.",
    "F6 replaces the SPEC margin-AAN (raising m only makes the robot help earlier, so SELF can only fall) by an adaptive perturbation DOSE: CP at fixed m = 0.03, 12 sessions x 20 pushes, "
    "magnitude = dose_k x U(0.8,1.2), dose_{k+1} = clip(dose_k + 0.10 (SELF_k - 0.70), 0.15, 1.0), dose_0 = 0.3 m/s; 3 patients (s = 0.2/0.5/0.8), the same pushes replayed under B0.",
    "Per-trial 'level' for B1/B2: 2 once the catch is decided (not when it engages), 1 while the commanded pelvic force exceeds 0.5 N, else 0. The physical first-intervention time "
    "(t_first) is the first applied pelvic force > 0.5 N or the catch engagement; decision, force-arrival and engagement times are logged separately. F2 plots rho = |xi_land(0) - anchor| of the "
    "best swing-leg candidate, evaluated with TRUE patient parameters and the true reaction timer, against r = L_max - 0.03 (a diagnostic).",
    "Showcase push is an illustrative single case, selected by scanning so that B0 falls while CP resolves it with assist only; it is not representative of the population.",
    "Sizing percentiles are over per-trial peak values; pelvis velocity 'relative to the base' assumes a stationary base.",
]


def main():
    os.makedirs(FIGS, exist_ok=True)
    cfg = Config()
    log("population")
    pop = make_population()
    log("controller grid")
    rows, res = full_grid(pop, cfg, log)
    log("delay sweep")
    sweep = delay_sweep(pop, cfg, log)
    log("dose sessions")
    aan = dose_sessions(cfg)
    log("showcase")
    pop1, show = showcase()
    save_npz(show)

    matched = matched_table(rows)
    ci = bootstrap_ci(pop, res, ["B0 none", "B1 theta=0.05", "B2 K=200", "CP m=0.03", "CP m=0.15"])
    write_csv(rows, os.path.join(RES, "summary.csv"))
    write_dicts(os.path.join(RES, "matched.csv"), matched)
    write_dicts(os.path.join(RES, "ci.csv"), ci)
    write_dicts(os.path.join(RES, "delay.csv"), sweep)
    write_dicts(os.path.join(RES, "catch_causes.csv"), cause_rows(res))
    ds = []
    for s in range(aan["dose"].shape[0]):
        for i, sv in enumerate(aan["sevs"]):
            ds.append(dict(session=s + 1, severity=sv, dose=aan["dose"][s, i], self_rate_cp=aan["self_cp"][s, i],
                           self_rate_b0=aan["self_b0"][s, i], falls_cp=int(aan["fall_cp"][s, i]), falls_b0=int(aan["fall_b0"][s, i])))
    write_dicts(os.path.join(RES, "dose_sessions.csv"), ds)
    sizing(res, os.path.join(RES, "sizing.md"), cfg)
    log("figures")
    fg.fig_pareto(rows, matched, os.path.join(FIGS, "pareto"))
    fg.fig_timeline({k: show[k] for k in ("B1", "B2", "CP")}, os.path.join(FIGS, "timeline"))
    fg.fig_map(show, pop1, 0.03, os.path.join(FIGS, "recoverability_map"))
    fg.fig_delay(sweep, os.path.join(FIGS, "delay"))
    stats = fg.fig_severity(pop, res, os.path.join(FIGS, "severity"))
    sv_rows = []
    for (ti, key), fr in sorted(stats.items()):
        sv_rows.append(dict(tercile=["mild", "moderate", "severe"][ti], controller=key, self_pct=fr[0], assisted_pct=fr[1],
                            caught_pct=fr[2], fall_pct=fr[3]))
    write_dicts(os.path.join(RES, "severity.csv"), sv_rows)
    fg.fig_aan(aan, os.path.join(FIGS, "aan_sessions"))
    interp = interpretation(rows, stats, sweep, aan, matched, ci)
    note = (f"Moderate-to-severe patient (severity 0.6, left leg paretic, m = 72 kg, z0 = 0.92 m), push 0.55 m/s at 60 deg from forward "
            f"(mostly lateral, toward the paretic side) at t = 0.3 s. Outcomes: "
            + ", ".join(f"{k} {show[k]['outcome']} ({show[k]['impulse']:.0f} N s)" for k in ("B0", "B1", "B2", "CP"))
            + ". Chosen illustratively (scan over severity, angle and magnitude) so that B0 falls while CP resolves it with assist only; it is not representative.")
    write_summary(rows, interp, DEVIATIONS, os.path.join(RES, "summary.md"), note, cause_text(res), matched, ci)
    log("animations")
    for k in ("B0", "B1", "B2", "CP"):
        fg.make_anim(show[k], {"B0": "B0 no robot", "B1": "B1 threshold-lock", "B2": "B2 impedance",
                               "CP": "CP"}[k], os.path.join(RES, f"anim_{k}.gif"))
        log(f"anim_{k}.gif {os.path.getsize(os.path.join(RES, f'anim_{k}.gif')) / 1e6:.2f} MB")
    log("done")


if __name__ == "__main__":
    main()
