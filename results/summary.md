# CatchPoint simulation summary

2000 trials per controller setting (50 patients x 40 pushes, seed 5412), identical patients and pushes across settings. Outcomes per SPEC 5.5. UNNECESSARY = B0 ended SELF for this patient and push AND the robot ever applied qualifying assistance (impulse >= 1 N s) or ever latched a catch decision, regardless of the controller's final outcome (a later FALL still counts); the % is of ALL trials, conditional rates are in summary.csv. Mean impulse includes pelvic and harness force. Threshold references: B1 and the B2 lock use the area-weighted centroid of the same scaled CoP region (S_cop) the plant uses; B1-XCoM uses the distance of the delayed identified-omega XCoM outside the support polygon; B2 damping uses the identified (noisy) mass. Intervention times (decision, applied force, catch engagement) are logged separately in summary.csv (medians).

| Controller setting | Fall % | SELF % | ASSISTED % | CAUGHT % | Unnecessary % | Mean impulse [N s] |
|---|---|---|---|---|---|---|
| B0 none | 53.6 | 46.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| B1 theta=0.03 | 2.6 | 0.7 | 0.0 | 96.7 | 45.8 | 90.5 |
| B1 theta=0.05 | 5.0 | 9.6 | 0.0 | 85.5 | 36.9 | 98.9 |
| B1 theta=0.08 | 13.4 | 25.9 | 0.0 | 60.8 | 20.5 | 115.7 |
| B1 theta=0.11 | 20.7 | 31.3 | 0.0 | 48.0 | 15.1 | 92.6 |
| B1 theta=0.14 | 30.2 | 33.1 | 0.0 | 36.7 | 13.4 | 64.3 |
| B1 theta=0.18 | 40.1 | 36.1 | 0.0 | 23.8 | 10.2 | 41.3 |
| B2 K=100 | 8.2 | 0.0 | 49.9 | 42.0 | 46.4 | 97.2 |
| B2 K=200 | 3.9 | 0.0 | 57.5 | 38.6 | 46.4 | 97.9 |
| B2 K=400 | 0.2 | 0.0 | 63.6 | 36.1 | 46.4 | 90.7 |
| B2 K=800 | 0.0 | 0.0 | 75.0 | 25.1 | 46.4 | 90.5 |
| B2 K=1600 | 0.0 | 0.0 | 79.3 | 20.6 | 46.4 | 113.2 |
| B2 K=3200 | 0.0 | 0.0 | 78.5 | 21.6 | 46.4 | 179.1 |
| B1-XCoM theta=0.02 | 1.2 | 31.1 | 0.0 | 67.7 | 15.2 | 130.7 |
| B1-XCoM theta=0.05 | 1.7 | 31.2 | 0.0 | 67.0 | 15.2 | 136.0 |
| B1-XCoM theta=0.08 | 4.5 | 31.8 | 0.0 | 63.7 | 14.6 | 134.2 |
| B1-XCoM theta=0.12 | 9.0 | 32.8 | 0.0 | 58.2 | 13.7 | 125.8 |
| B1-XCoM theta=0.16 | 15.3 | 34.2 | 0.0 | 50.4 | 12.2 | 104.9 |
| CP m=-0.06 | 7.3 | 35.6 | 18.1 | 38.9 | 10.8 | 70.3 |
| CP m=-0.03 | 6.2 | 35.4 | 18.3 | 40.2 | 11.1 | 71.8 |
| CP m=0 | 5.3 | 34.7 | 19.4 | 40.6 | 11.8 | 73.3 |
| CP m=0.03 | 4.8 | 34.2 | 20.4 | 40.7 | 12.3 | 74.2 |
| CP m=0.06 | 4.4 | 33.3 | 22.8 | 39.5 | 13.1 | 74.7 |
| CP m=0.1 | 4.2 | 32.8 | 24.8 | 38.2 | 13.7 | 74.6 |
| CP m=0.15 | 3.9 | 32.5 | 25.7 | 38.0 | 13.9 | 74.3 |
| CP-pop m=-0.06 | 8.4 | 31.6 | 27.4 | 32.6 | 14.8 | 80.7 |
| CP-pop m=-0.03 | 7.8 | 31.0 | 27.7 | 33.5 | 15.4 | 82.4 |
| CP-pop m=0 | 7.3 | 30.8 | 28.0 | 34.0 | 15.7 | 84.6 |
| CP-pop m=0.03 | 6.8 | 30.4 | 28.4 | 34.3 | 16.0 | 84.3 |
| CP-pop m=0.06 | 6.7 | 30.3 | 29.0 | 34.0 | 16.1 | 83.6 |
| CP-pop m=0.1 | 6.2 | 30.0 | 33.5 | 30.2 | 16.4 | 81.6 |
| CP-pop m=0.15 | 5.9 | 29.9 | 33.8 | 30.4 | 16.5 | 80.8 |
| CP-noDC m=0.03 | 9.4 | 36.9 | 17.8 | 35.9 | 9.6 | 71.0 |

## Interpretation

1. DEFAULT comparison (CP m=0.03, B1 theta=0.05, B2 K=200 are single sweep points; CP default was fixed in SPEC): CP m=0.03 falls 4.8% [3.4, 6.2] and keeps SELF 34.1% [31.3, 37.0] (B0: falls 53.6% [49.6, 57.5], SELF 46.4% [42.5, 50.4]; paired patient bootstrap 95% CI). B1 theta=0.05 falls 5.0% [3.4, 6.5] with SELF 9.6% [7.3, 11.8]; B2 K=200 falls 3.9% [3.0, 4.8] with SELF 0.0% [0.0, 0.0].
2. MATCHED-FALL comparison (SELF interpolated along each sweep at a common fall rate; matched.csv): at 4% falls SELF = B1 5.9%, B1-XCoM 31.7%, B2 0.0%, CP 32.6%, CP-pop n/a (not bracketed); at 6% falls B1 11.6%, B1-XCoM 32.1%, B2 0.0%, CP 35.2%, CP-pop 29.9%. CP/B1 at 4% = 5.5x. CP m=0.15 (the sweep point nearest 4% falls) has falls 3.9% / SELF 32.5%; this is a post-selection operating point, not the default.
3. Ablations: CP-pop (population-mean model, no identification) at m=0.03 falls 6.8% with SELF 30.4% vs CP 4.8% / 34.2%, so the patient model is worth +2.1 pp falls and +3.7 pp SELF. B1-XCoM (capture-point threshold, no reach model or graded force) spans falls 1.2-15.3% with SELF 31.1-34.2% (e.g. B1-XCoM theta=0.02: 1.2% / 31.1%): the capture-point idea alone explains most of the SELF gain over COM-distance thresholds at equal falls (matched.csv: B1-XCoM 31.7% vs CP 32.6% at 4% falls), so CP's remaining advantage over B1-XCoM is mainly less harness use and impulse (at similar falls B1-XCoM theta=0.08: CAUGHT 64%, impulse 134 N s, unnecessary 14.6% vs CP m=0.03: 41%, 74 N s, 12.3%). The delay-compensation ablation CP-noDC falls 9.4% (vs 4.8%) but has fewer unnecessary interventions (9.6% vs 12.3%).
4. Where CP is NOT better: CP falls more than B2 K>=400 (4.8% vs 0.2%, K=800 0.0%); CP ends 40.7% of trials in a harness catch (L2), more than B1 theta>=0.14 (37%, 24%), which however fall far more often; its mean impulse (74 N s) is below B2 K=800 (91 N s) and above B1 theta>=0.14 (64 N s); it intervenes unnecessarily in 12.3% of all trials (27% of B0-SELF trials) vs B1 theta=0.18 10.2%; in the severe tercile CP falls 6% and keeps 25% SELF (B0 SELF there: 29%). In these simulations every non-caught B2 trial exceeded the 1 N s impulse threshold (SELF 0.0% is an empirical result, not a definition). The harness remains the main safety mechanism: the push distribution is harsh (B0 falls in most trials), so most rescues need more than the 150 N pelvic limit.
5. Caveats: simplified LIP model, assumed population ranges (SPEC 5.2), 10% identification error (one draw per patient, all model-using controllers share it), no sensor noise; delay compensation (exact lag-aware prediction) lowers falls 9.4% -> 4.8% at 0.10 s and 19.9% -> 3.5% at 0.30 s; the adaptive-dose loop (CP safety layer) settles at 0.43, 0.39, 0.34 m/s for mild/moderate/severe patients with 0 falls vs 28 for B0 on the same pushes (3 patients, noisy). CP has privileged patient-model information the baselines lack (see ablation CP-pop). Not a clinical result.

## Matched-fall comparison (interpolated SELF at common fall rate)

| Family | Fall target % | SELF % (interpolated) | Bracketing settings |
|---|---|---|---|
| B1 | 4 | 5.9 | B1 theta=0.03 (2.6% fall, 0.7% SELF) / B1 theta=0.05 (5.0% fall, 9.6% SELF) |
| B1 | 6 | 11.6 | B1 theta=0.05 (5.0% fall, 9.6% SELF) / B1 theta=0.08 (13.4% fall, 25.9% SELF) |
| B1-XCoM | 4 | 31.7 | B1-XCoM theta=0.05 (1.7% fall, 31.2% SELF) / B1-XCoM theta=0.08 (4.5% fall, 31.8% SELF) |
| B1-XCoM | 6 | 32.1 | B1-XCoM theta=0.08 (4.5% fall, 31.8% SELF) / B1-XCoM theta=0.12 (9.0% fall, 32.8% SELF) |
| B2 | 4 | 0.0 | B2 K=200 (3.9% fall, 0.0% SELF) / B2 K=100 (8.2% fall, 0.0% SELF) |
| B2 | 6 | 0.0 | B2 K=200 (3.9% fall, 0.0% SELF) / B2 K=100 (8.2% fall, 0.0% SELF) |
| CP | 4 | 32.6 | CP m=0.15 (3.9% fall, 32.5% SELF) / CP m=0.1 (4.2% fall, 32.8% SELF) |
| CP | 6 | 35.2 | CP m=0 (5.3% fall, 34.7% SELF) / CP m=-0.03 (6.2% fall, 35.4% SELF) |
| CP-pop | 4 | n/a | not bracketed |
| CP-pop | 6 | 29.9 | CP-pop m=0.15 (5.9% fall, 29.9% SELF) / CP-pop m=0.1 (6.2% fall, 30.0% SELF) |

## Paired bootstrap CIs

| Setting | Metric | Estimate % | 95% CI (patient bootstrap, 1000 reps) |
|---|---|---|---|
| B0 none | fall_pct | 53.6 | [49.6, 57.5] |
| B0 none | self_pct | 46.4 | [42.5, 50.4] |
| B1 theta=0.05 | fall_pct | 5.0 | [3.4, 6.5] |
| B1 theta=0.05 | self_pct | 9.6 | [7.3, 11.8] |
| B2 K=200 | fall_pct | 3.9 | [3.0, 4.8] |
| B2 K=200 | self_pct | 0.0 | [0.0, 0.0] |
| CP m=0.03 | fall_pct | 4.8 | [3.4, 6.2] |
| CP m=0.03 | self_pct | 34.1 | [31.3, 37.0] |
| CP m=0.15 | fall_pct | 3.9 | [2.7, 5.2] |
| CP m=0.15 | self_pct | 32.5 | [29.8, 35.0] |

## CP catch causes

Cause recorded at the first tick the arbiter latched the catch decision (CP m=0.03). 897 trials latched a decision, 814 of them engaged the catch (CAUGHT); counts are for the CAUGHT trials (all latched decisions in brackets). Whenever a catch is requested for causes (b)/(c) the best candidate has h(0) < 0, i.e. no valid candidate is reach-feasible at F = 0; the (b)/(c) split by a 0.25 s horizon is a descriptive convention. All settings are in catch_causes.csv.

- (a) no crossover-valid step candidate at all (catch requested directly): 0 (0.0%) [0]
- (b) valid candidates exist but even the best needs F* > F_max, time-to-landing short (ongoing swing or T_land < 0.25 s): 432 (53.1%) [489]
- (c) valid candidates exist but even the best needs F* > F_max, long horizon (required force simply exceeds 150 N): 382 (46.9%) [408]

CP-sat (apply F_max, catch only if the predicted barrier with F_max is still < 0) is mathematically identical to the implemented arbiter (h(F_max u) < 0 exactly when |F*| > F_max; asserted in tests), so it was not added.

## Showcase push

Moderate-to-severe patient (severity 0.6, left leg paretic, m = 72 kg, z0 = 0.92 m), push 0.55 m/s at 60 deg from forward (mostly lateral, toward the paretic side) at t = 0.3 s. Outcomes: B0 FALL (0 N s), B1 CAUGHT (140 N s), B2 ASSISTED (72 N s), CP ASSISTED (17 N s). Chosen illustratively (scan over severity, angle and magnitude) so that B0 falls while CP resolves it with assist only; it is not representative.

## Deviations from SPEC

- dt = 0.002 s for every run (including Monte Carlo); the LIP is integrated with the exact (matrix-exponential) step for piecewise-constant CoP and force, which SPEC allows. Whole run_all takes several minutes, so dt was not coarsened.
- Support-region geometry: the convex hull of two axis-aligned rectangles and the projection onto it are exact; the ankle-strategy centroid c_S is the area-weighted mean of the two (scaled) foot-rectangle centres rather than the exact hull centroid (millimetres). B1 and the B2 lock reference this same c_S (review item 6; the earlier midpoint-of-feet reference was replaced).
- Delay sweep (F4): the total delay Delta = tau_s + tau_a is varied by keeping tau_s = 0.03 s fixed and setting tau_a = Delta - 0.03 s (SPEC does not say how Delta is split). The 0.02 s force lag is unchanged.
- Plant ordering: commanded force -> pure delay tau_a -> first-order lag 0.02 s. Once the catch is engaged all pelvic commands are set to zero (the harness holds); the already-queued delayed force still decays through the plant. The catch spring-damper force is added on top and is unbounded. Impulse = integral of |pelvic + harness force| until the fall (or 3 s).
- CatchPoint internals not fixed by SPEC (documented, not tuned against baselines): (i) delay-compensation forward prediction uses the projection of the measured xi onto the identified S_cop as constant CoP and propagates the applied force EXACTLY (logged applied force plus the pure-delay + first-order-lag recursion of the queued commands; verified against the plant in tests); (ii) the controller keeps its own violation timer on the predicted xi to estimate the patient's remaining reaction time (shifted by -tau_a with compensation, +tau_s without); (iii) during an ongoing swing the remaining swing time is floored at 0.05 s so beta does not vanish; (iv) candidates (2 legs x side-step/crossover) that violate the crossover rule are excluded (-inf); if no candidate is valid and xi-hat is outside S_cop the arbiter requests the catch directly; (v) the identified model multiplies m, omega, tau_r, both L_max, both T_s and the paretic ankle factor by (1 + N(0,0.1^2)), drawn once per patient (the same draw feeds B2 damping and B1-XCoM's omega); (vi) the arbiter latches the catch decision the first tick |F*| > 150 N, and the engine engages it 0.28 s later.
- Ablations added after review: CP-pop (population-mean model: tau_r 0.35 s, L_max 0.45 m and T_s 0.4625 s and ankle factor 0.825 for both legs, mass 72.5 kg, z0 0.925 m) and B1-XCoM (theta sweep 0.02-0.16 m). CP has privileged patient-model information that B1/B2 do not; these two controllers bound how much that matters.
- Patient model details: foot lift-off happens when the reaction timer expires (0.6 tau_r for later steps) and the timer restarts after each landing; the target of an infeasible step is clipped onto that candidate's reachable set. Fall check uses the support polygon at the start of each tick. Patients may keep stepping (<= 3 steps) after the catch engages.
- Crossover steps (coordinator correction): candidate with T_s x 1.25, L_max x 0.7, swing centre at least 0.09 m across the stance foot (side-steps keep 0.08 m). Patient prefers a feasible side-step, else a feasible crossover, else the least-infeasible candidate. CP evaluates the same 4 candidates with identified parameters.
- F6 replaces the SPEC margin-AAN (raising m only makes the robot help earlier, so SELF can only fall) by an adaptive perturbation DOSE: CP at fixed m = 0.03, 12 sessions x 20 pushes, magnitude = dose_k x U(0.8,1.2), dose_{k+1} = clip(dose_k + 0.10 (SELF_k - 0.70), 0.15, 1.0), dose_0 = 0.3 m/s; 3 patients (s = 0.2/0.5/0.8), the same pushes replayed under B0.
- Per-trial 'level' for B1/B2: 2 once the catch is decided (not when it engages), 1 while the commanded pelvic force exceeds 0.5 N, else 0. The physical first-intervention time (t_first) is the first applied pelvic force > 0.5 N or the catch engagement; decision, force-arrival and engagement times are logged separately. F2 plots rho = |xi_land(0) - anchor| of the best swing-leg candidate, evaluated with TRUE patient parameters and the true reaction timer, against r = L_max - 0.03 (a diagnostic).
- Showcase push is an illustrative single case, selected by scanning so that B0 falls while CP resolves it with assist only; it is not representative of the population.
- Sizing percentiles are over per-trial peak values; pelvis velocity 'relative to the base' assumes a stationary base.
