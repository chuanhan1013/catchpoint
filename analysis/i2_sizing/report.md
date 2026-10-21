# I2 sizing — walker-mounted cable pelvic interface (slack / nudge / arrest)

Status: preliminary statics + simulation feasibility analysis, 22 Oct 2026. Not human-tested, not
hardware-tested, not safety-certified; it supports no fall-arrest claim.

Rounding policy (generated file: do not edit by hand; see `make_report.py`, `publish.py`, `templates/`). Every number below is
filled from the JSON in `data/` by `publish.py`, so a published bound is never better than its source value: loads, tensions,
torques, required ratings, resultants, maxima, upper CI bounds and required mass are rounded UP; capacities, safety factors,
friction capacity, slide limits, envelopes, minima, coverage percentages and lower CI bounds are rounded DOWN; a confidence
interval has its lower bound DOWN and its upper bound UP at 1 decimal. Only descriptive statistics of the simulated sample (the
"median" and "p95" figures) and nominal geometry are rounded to nearest, and they are always labelled as such. Model inputs
and assumptions (for example the 60 kg baseline walker, 15 kN/m cable stiffness, 85 kg patient) are stated, not computed.
Interpolated quantities are linear between the two adjacent tabulated JSON points.

D3 geometry has ONE source of truth: `data/d3_contract.json`. `sizing.py`, `run_analysis.py`, `slip_analysis.py` and
`cad/build_cad_d3.py` all read it; `test_d3_consistency.py` checks the generated CAD parts against it (1 mm).

Reproduce: `.venv/Scripts/python analysis/i2_sizing/instrument_sim.py [tau_lock]` (0.28, 0.15, 0.08),
then `run_analysis.py`, `slip_analysis.py`, `fbd.py`, `make_report.py` (regenerates this file, `results/sizing_D3_summary.md` and the
derived fields of the contract), and `cad/build_cad_d3.py` for the CAD.
Tests: `pytest analysis/i2_sizing` (`test_sizing.py`, `test_d3_consistency.py`, `test_publish.py`).

## 1. Bottom line
1. The 491 N in the concept review is `Dc·v` of the SPEC catch spring-damper (`F = -Kc(x-xlock) - Dc v`,
   Dc = 800 N·s/m), not a physical arrest requirement. The force needed to stop the simulated patients within
   0.25 m is much lower (§3): median 55 N, p95 214 N (descriptive statistics of the sample, nearest).
2. As proposed (60 kg walker, horizontal cables at pelvis height, no outriggers) I2 tips statically at
   200 N (SF 1) and has SF 0.41 at 491 N.
3. Feasible only with five changes together: a brake that slips at a set force (force limiter);
   mixed-height anchors (corners high on the 1.20 m columns, the rest at pelvis height); an 80 kg walker
   with ~20 kg low ballast (CG z 0.40 m); outrigger feet deployed on catch (+0.25 m all round); the BWS
   anchor above the pelvis (x = 0.42 m), not on the rear beam.
4. Force wording (used everywhere): the group-tuned passive slip brakes give ≥ 300 N along the fall direction
   (305 N minimum over 360 directions) and ≤ 380 N (379.3 N) horizontal resultant (the perpendicular component is up to 178 N;
   the resultant deviates up to 28 deg from the fall direction). The resultant stays below the 420 N static envelope
   of D3 (SF 1.5; SF 1.0 to 640 N). Tip SF ≥ 2.09 and sliding SF ≥ 1.28 at the full slip resultant (friction
   capacity ≥ 447 N at μ = 0.5 in the 300 N cases). Pelvis lift ≤ 171 N under passive slip (cap 0.5 BW = 294 N for 60 kg).
5. Denominator, stated explicitly (HEADLINE, passive-slip metric): among the 814 trials that the CP m = 0.03
   simulation (2000 trials, lock 0.28 s) already ended as CAUGHT, the group-tuned passive slip brakes on D3 (corner 150,
   lateral 155, centre 120 N; for each trial's fall direction they deliver 305.5-379.3 N along the fall, median 330 N (descriptive), plus
   up to 178 N perpendicular) arrest 805/814 = 98.8 % (CI 97.9-99.5 %) within 0.25 m (1-D stop with the
   along-fall force), with static tip SF >= 1.5 and no sliding > 1 cm under the FULL resultant, and no dynamic tip.
   Comparison, the earlier metric (fixed 300 N along the fall, QP-distributed tensions): 804/814 = 98.7 % (CI 97.7-99.4 %). The other 1186/2000 trials never engage the catch, so I2 does not change them. This is NOT 99 % of
   all trials, not a population guarantee, and not a fall reduction: the 95 baseline falls (4.75 %) are unchanged
   and the 9 non-arrested caught trials are potential additional falls, 95 + 9 = 104/2000 = 5.2 % at worst
   (counts, under the simplifications of section 2.6). An empirical, conditional result of one seeded simulation.
6. Why a 300 N along-fall target and not the 420 N static (SF 1.5) envelope: coverage plateaus above ~300 N
   (fixed-force metric: 804/814 at 300 N, 806/814 at 400 N); the ~40 N between the ≤ 380 N passive resultant and 420 N,
   and the headroom of the envelope itself, are held as margin for (a) the spread of the per-spool slip resultant
   (section 4b), (b) friction uncertainty (D3 slides at 259 N if the pads give mu = 0.3), (c) dynamic overshoot and
   incomplete outrigger deployment (without outriggers the envelope is 210 N). Lock-delay sensitivity, D3 caught
   trials, passive-slip metric (fixed 300 N metric in brackets): 0.28 s 805/814 = 98.8 % (804/814 = 98.7 %);
   0.15 s 821/844 = 97.2 % (820/844 = 97.1 %); 0.08 s 817/866 = 94.3 % (807/866 = 93.1 %) (a faster lock means a
   faster COM at engagement and a higher load).
7. Timing: with the SPEC 0.28 s lock the catch engages at 0.28 s after decision and the COM stops at
   ≈0.38 s (median, descriptive; p95 0.47 s, rounded up; passive-slip model), inside the 0.3–0.5 s window. A faster lock (assumed ~0.14 s) is
   plausible but undemonstrated, and it raises the load (§3).

## 2. Model and assumptions
Frame per SPEC §4: x forward, y left, z up. D3 geometry is read from `data/d3_contract.json` (the CAD reads the
same file): footprint 0.95 × 0.80 m; wheel contacts inset by the wheel radius 0.075 (x = 0.075/0.875, y = ±0.325).
Columns 1.20 m at x = 0.03/0.90, y = ±0.33. Nominal pelvis P = (0.42, 0, 0.95); belt (D3) half-depth 0.13,
half-width 0.205 (the CAD value, needed for the mannequin to fit; the SPEC value 0.11/0.19 is kept only for D0-D2).
Each anchor is the cable exit point: the CAD places every spool housing so that its fairlead tip lies exactly on the
contract anchor (checked at 1 mm).

2.1 Cables: u_i from belt attachment (ray from the pelvis centre toward the anchor, intersected with the belt ellipse)
to anchor; pelvis force F_p = Σ T_i u_i; frame force at anchor −T_i u_i.
QP: min Σ T² s.t. Σ T_i u_i,xy = F_des, T_min = 2 N ≤ T ≤ T_max (1500 N arrest, 250 N nudge); solved by
bounded least squares + null-space polish (residual < 1e-9 N). Pelvis lift capped (0.15 BW nudge,
0.5 BW arrest, lightest patient 60 kg). Layouts: 4 corner anchors; 6 adds lateral (0.45, ±0.40);
8 adds rear-centre and front-centre; optional overhead BWS cable. Cables only pull → forces toward the
open front of the U need front anchors (open routing question).

2.2 Walker statics: weight at CG, cable reactions at anchors, ground reactions at 4 wheels (+4 outrigger
feet when deployed). Tipping when the ground CoP leaves the support polygon; tip SF = multiple of cable
loads that puts the CoP on an edge; sliding SF = μ(N0 + λN1)/(λ|H|). D0/D1 walker 60 kg, CG
(0.40, 0, 0.45); μ = 0.5 assumed. Patient weight reaches the walker only through vertical cable components
(horizontal cables at pelvis height add zero wheel load — tested).

2.3 Arrest dynamics (1-D along the force): m s'' = m ω² s − F (ankle CoP at support edge, optimistic);
cable lumped stiffness 15 kN/m, damping ratio 0.2; brake slips at F_lim; walker Coulomb friction.
Passive-slip coverage (headline): for each caught trial, F_lim = the along-fall component of the force the group-tuned
passive brakes produce for that trial's fall direction (cables whose length grows along the motion slip at their own
tension, BWS slack; directions binned to 0.5 deg); the walker's friction load uses the full horizontal resultant
(along + perpendicular) and the downward cable component; the static tip check uses the statics of the actual slip
tensions (SF >= 1.5 at the peak force); the dynamic tip check uses their overturning moment about the tip edge.
The fixed-force metric (300 N along the fall, QP tensions) is kept for comparison.
Success = peak s ≤ 0.25 m (SPEC fall criterion). Minimum constant force to stop:
F ≥ m ω(ω s0 + v); if s0 + v/ω > 0.25 m, F = m(ω²(0.25² − s0²) + v²) / (2(0.25 − s0)).

2.4 Dynamic tip check: rigid body rotating about the tipping edge under the arrest force history (indicative).

2.5 Spool/brake: brake torque = 1.5 × T × r_s, brake on the spool side of the gearbox. Nudge servo 12:1,
η = 0.9, rotor inertia 5e-6 kg·m² (assumed, not a part selection).

2.6 Simplifications that flatter the device: ankle CoP at support edge, fixed force direction, rigid frame,
nominal pelvis position, LIP patient; the perpendicular component of the passive slip force is applied to the walker
but not fed back into the 1-D patient dynamics. Stepping after engagement ignored (works against the device).

## 3. Arrest load cases from the simulation
`instrument_sim.py` wraps `lip_step` and `dist_to` with pass-through loggers (sim unchanged; asserts identical
outcomes and peak catch force 491.17 N). CP m = 0.03, 2000 trials, 814 caught, lock 0.28 s (fig1).
The median and p95 columns are descriptive statistics (nearest); the max column is rounded up.

| At engagement (814 trials) | median | p95 | max |
|---|---|---|---|
| COM speed away from support [m/s] | 0.18 | 0.48 | 0.61 |
| COM distance from support [cm] | 1.2 | 16.6 | 24.9 |
| COM kinetic energy [J] | 1.44 | 9.08 | 14.3 |
| Sim catch-force peak [N] | 175 | 399 | 491.2 |
| Hold force at t = 3 s [N] | 33 | 131 | 289 |
| Sim catch impulse, first 0.3 s [N·s] | 20.7 | 68.3 | 100.3 |
| Minimum constant force to stop within 0.25 m [N] | 55 | 214 | 3 trials > 600 |

Share of the 814 caught trials needing ≤ 150 / 200 / 300 / 400 N (rounded down): 78.8 / 92.6 / 99.0 / 99.3 %.
Lock-delay sensitivity (τ_lock 0.15 / 0.08 s): median COM speed at engagement 0.32 / 0.45 m/s (descriptive); p95 required
force 196 / 339 N (descriptive); share ≤ 300 N 97.3 / 94.2 % (rounded down).

## 4. Designs and results (lift cap 294 N; SF = static tip safety factor, worst of 24 directions)

| Design | Walker | Cables | F allowed (SF 1.5) | SF 1.0 | Slide limit (μ 0.5) |
|---|---|---|---|---|---|
| D0 as proposed | 60 kg, CG z 0.45 | 6 at z = 0.95 | 130 N | 200 N | 294 N |
| D1 high anchors | 60 kg | 8: corners 1.20, rest 0.95 | 150 N | 230 N | 345 N |
| D2 + ballast | 100 kg, CG z 0.35 | as D1 | 260 N | 390 N | 575 N |
| **D3 recommended** | 80 kg, CG (0.45, 0, 0.40), outriggers +0.25 m | D1 + BWS | **420 N** | **640 N** | 465 N |

Walker mass vs base half-width (fig3; mass = required mass, rounded up; values between tabulated half-widths are linear
interpolations between the two adjacent JSON points, an upper bound for this convex curve): horizontal cables, 491 N, SF 1 → ≈148 kg at 0.325 m, 119 kg at
0.40 m (reproduces the review). Mixed-height cables, 300 N, SF 1.5 → ≈114.5 kg at 0.325 m, ≈89.3 kg at
0.40 m; a 60 kg walker needs ≈0.56 m half-width (inverse interpolation, rounded up) → outriggers are the practical lever.

D3 arrest at 300 N (QP tension distribution, controller-set; the passive-slip case is section 4b; tensions, lifts and T_max rounded up, SF and friction capacity rounded down):

| Fall | Tensions [N] | T_max | Lift | Tip SF (edge) | Friction needed / cap |
|---|---|---|---|---|---|
| forward | rearL 108, rearR 108, rearC 171 | 171 | 127 N | 2.90 (front) | 300 / 455 N |
| backward | frontL 108, frontR 108, frontC 150, latL/R 12 | 150 | 111 N | 2.37 (rear) | 300 / 447 N |
| left | rearR 114, frontR 91, latR 200, rearC 11 | 200 | 114 N | 2.15 (left) | 300 / 449 N |
| right | mirror of left | 200 | 114 N | 2.15 | 300 / 449 N |

Nudge 150 N, 24 directions: all feasible; T_max 101.1 N, lift ≤ 66.1 N, tip SF ≥ 4.36, sliding SF ≥ 3.05.
Brake at r_s = 25 mm: 7.49 N·m (1.5 × the 199.6 N worst QP tension; ≈10.48 N·m scaled to 420 N). Nudge servo: 2.53 N·m at
spool, 0.24 N·m motor, 1.16 kg reflected mass. Slip-brake energy (passive slip, arrested trials) ≤ 22.2 J over the three
lock delays (16.7 J at 0.28 s). Timing (fig5, passive slip): nudge force at 0.12 s; lock at 0.28 s; COM stopped
0.104 s after engagement (median; p95 0.19 s; descriptive). FBDs: fig6.

### 4b. Per-spool slip versus resultant force (data/slip.json, figures/fig7_slip_resultant.png)
Brake torque must be sized from the resultant the spools can deliver, not from one cable. With independent slip,
cable i carries up to T_slip,i whenever the pelvis moves away from its anchor (cable pays out). For D3 at the
nominal pelvis position (BWS cable kept slack in horizontal arrest), over 360 patient-motion directions (1 deg); resultants and lifts are maxima (rounded up),
along-arrest forces and tip SF are minima (rounded down):

| Option | Along-arrest force (min) | Horizontal resultant (max) | Tip SF (min) | Pelvis lift (max) |
|---|---|---|---|---|
| Uniform T_slip = 150 N, passive | 301 N | 405 N | 1.87 | 171 N |
| Uniform T_slip = 190 N, passive | 382 N | 513 N (> 420) | 1.48 | 217 N |
| **Group-tuned passive: corner 150, lateral 155, centre 120 N** | **305 N** | **380 N** | 2.09 | 171 N |
| Controller-set thresholds (QP at 300 N), exact direction | 302 N | 306 N | 2.09 | 123 N |
| same, direction-estimate error 15 / 30 / 45 deg | 287 / 253 / 205 N | 306 / 306 / 306 N | 2.09 | 123 N |
| Hybrid (passive floor, controller may only lower) | 254 N (0 deg error) | 305 N | 2.59 | 123 N |

Passive slip can deviate from the arrest direction by up to 28 deg (perpendicular component up to 178 N).
The worst-case resultant with EVERY cable at its slip tension at once (geometry ignored) is 379.3 N (≤ 380 N) for the
group-tuned set, below the 420 N SF-1.5 envelope, so the bound holds without any direction logic (tested). Sliding SF
at the full slip resultant is >= 1.28. The thresholds are the grid-search optimum (5 N grid, minimum resultant/along
ratio) over the 2298 triples that meet the three constraints (along >= 300 N, resultant <= 420 N, lift <= 294 N); the
earlier 150/150/110 N set (tuned for the narrower 0.11/0.19 belt) gives less than the 300 N target along the fall with the CAD belt
and was replaced.
Recommendation: passive, group-tuned, spring-applied power-off brakes (guaranteed bound, fail-safe, no controller
in the load path). An electromagnet may trim thresholds per direction, but a current-controlled slip brake alone
is not recommended: it depends on the direction estimate (30 deg error -> 253 N, below the 300 N target) and a
loss of power must still leave a safe bound. Per-spool brake table (r_s = 25 mm, rated holding torque = 1.5 x
slip torque, rounded up):

| Spool group | Cables | T_slip | Slip torque | Rated hold torque |
|---|---|---|---|---|
| corner | rearL, rearR, frontL, frontR | 150 N | 3.75 N m | 5.63 N m |
| lateral | latL, latR | 155 N | 3.88 N m | 5.82 N m |
| centre | rearC, frontC | 120 N | 3.00 N m | 4.50 N m |
| BWS (hang only, 1.5 x 85 kg x g = 1251 N) | bws | not a slip spool | - | 46.91 N m |
| (alternative) current-set brake | any | max QP tension 199.6 N | - | 7.49 N m |

This supersedes the earlier single-cable sizing (one rating for all spools). Not modelled: load sharing through
cable stretch between spools, slip hysteresis, pelvis displacement away from nominal.

## 5. Envelope and sensitivities
- D3 envelope: 300 N along the fall (≥ 305 N achieved) with a ≤ 380 N resultant bound (brake slip torques 3.75-3.88 N m),
  SF >= 2.09 worst direction; walker slides at 465 N (259 N at μ 0.3). Beyond this: conceptual only.
- HEADLINE coverage of caught trials, passive slip (force = what the group-tuned passive brakes give for each trial's
  fall direction; 1-D stop with the along-fall component, static tip / slide / dynamic tip with the full resultant;
  data/results.json `dyn.passive`), next to the old fixed-300 N metric (300 N along the fall, QP tensions); coverage percentages rounded down, CI lower bound down / upper bound up:
  lock 0.28 s: passive 805/814 = 98.8 % (CI 97.9-99.5 %), fixed 804/814 = 98.7 % (CI 97.7-99.4 %).
  lock 0.15 s: passive 821/844 = 97.2 % (CI 95.9-98.2 %), fixed 820/844 = 97.1 % (CI 95.8-98.1 %).
  lock 0.08 s: passive 817/866 = 94.3 % (CI 92.5-95.7 %), fixed 807/866 = 93.1 % (CI 91.3-94.7 %).
  Passive failures are all 1-D stopping failures (9 / 23 / 49 trials not arrested within 0.25 m); none fails the
  static tip (min SF 2.09), slide (0 mm) or dynamic tip check. Along-fall force over the trials: median 330 N (descriptive), p95 376 N (descriptive),
  max 379.3 N (peak resultant 379.3 N).
- Coverage of caught trials with a fixed force limit F_lim = 200/250/300/400 N (old metric). Corrected metric:
  arrested within 0.25 m AND no dynamic tip AND static tip SF >= 1.5 at the peak force AND walker slide <= 1 cm.
  Counts, Wilson 95 % CI:
  lock 0.28 s (n = 814): D3 752/797/804/806 (92.3/97.9/98.7/99.0 %; CI at 300 N 97.7-99.4 %).
  lock 0.15 s (n = 844): D3 806/816/820/822 (95.4/96.6/97.1/97.3 %; CI at 300 N 95.8-98.1 %).
  lock 0.08 s (n = 866): D3 755/791/807/825 (87.1/91.3/93.1/95.2 %; CI at 300 N 91.3-94.7 %).
  D0 (as proposed), same metric: 377/814 = 46.3 % (CI 42.9-49.8 %) at 300 N at lock 0.28 s,
  170/844 = 20.1 % at 0.15 s and 56/866 = 6.4 % at 0.08 s. Its earlier ~96 % came from a dynamic-only check
  that ignores static margin and sliding.
- Allowing 491 N is worse, not better: D3 coverage at F_lim = 491 N is 775/814 = 95.2 % (vs 804/814 at 300 N), because the static SF-1.5 check fails above 420 N.
- D3 SF-1.5 envelope sensitivity (nominal 420 N; each value is an envelope, rounded down): outriggers not deployed 210; outriggers +0.15 m 330;
  60 kg walker 320; 100 kg 530; pelvis 0.15 m lower 280; pelvis 0.10 m rearward/forward 430/430; pelvis 0.10 m to the
  left 410; walker CG 0.10 m higher 420; CG 0.10 m rearward 390; 6 cables 310; 4 cables 270; all anchors at 1.20 m 270;
  corners at 1.50 m 470 (data/results.json `sensitivity`; D3 variants use the contract geometry, BWS anchor
  (0.42, 0, 1.80)). Lateral tipping and outriggers govern.
- Hanging patient (1.5 × 85 kg): with the BWS anchor overhead → 1251 N down, no horizontal load, no tip.
  With the SPEC rear-beam winch → 1952 N horizontal, tip SF 0.54 (D3) / 0.21 (60 kg): unusable.

## 6. NOT validated
No hardware or human data; no stress analysis (columns, base, anchors, belt, terminations, brake shock);
no spool inertia/backlash. "Required force" uses a 1-D LIP stopping model. μ, cable stiffness, walker CG/mass,
servo data are assumptions. Braked omni wheels can slip along roller axes → rubber outrigger pads credited,
not wheel brakes. Outrigger deployment time and front-anchor routing across the open U are not designed.
Independent slipping of several brakes will not preserve force direction (modelled for the walker, not fed back into
the patient dynamics). 0.28 s lock delay is the SPEC
value; 0.14 s is a guess. 0.5 BW lift tolerance and belt loads unvalidated. No mobile cable system rated for
200–300 N arrest was found (mTPAD ≈35 N per cable).

## 7. Suggested wording
"We propose a walker-mounted cable interface with transparent slack, graded nudging (≤ 150 N) and a
power-off spool brake that slips at a set force (passive slip brakes give ≥ 300 N along the fall direction and
≤ 380 N horizontal resultant). A load analysis shows arrest on a walker is
feasible only with an ~80 kg low-CG walker, deployable outriggers and mixed-height anchors; in the
simulated catches the passive slip brakes sufficed in 805 of the 814 trials that the CP simulation ended as CAUGHT
(lock 0.28 s, one seeded 2000-trial run; the other trials are unaffected). The concept is unbuilt and
unvalidated."

## 8. D3 interface contract (what the CAD must build; SPEC section 4 coordinates: floor origin, x forward, y left, z up)

| Item | Value |
|---|---|
| Arrest force (horizontal, any direction) | passive slip brakes give >= 300 N along the fall direction (305 N minimum, 360 directions) and <= 380 N resultant (bound with all cables slipping at once) |
| Static tip safety factor | >= 1.5 required; D3 envelope 420 N (SF 1.5), 640 N (SF 1.0); SF >= 2.09 at the full slip resultant, 2.15 at 300 N (QP) |
| Walker mass / CG | 80 kg total incl. ~20 kg low ballast; CG (0.45, 0, 0.40) m |
| Wheel contact points | x = 0.075 / 0.875, y = +-0.325 (0.95 x 0.80 footprint inset by the 0.075 m wheel radius; the CAD wheels are centred there); braked omni wheels NOT credited for friction |
| Outrigger feet, deployed on catch | ex = ey = 0.25 m beyond the wheel line: feet at x = -0.175 and 1.125, y = +-0.575; rubber pads, mu >= 0.5; fully down before lock engagement (<= decision + 0.28 s) |
| Corner anchors (4), z = 1.20; anchor = cable exit point (fairlead tip) | rearL (0.03, +0.33), rearR (0.03, -0.33), frontL (0.90, +0.33), frontR (0.90, -0.33) |
| Lateral anchors (2), z = 0.95 | (0.45, +0.40), (0.45, -0.40) |
| Centre anchors (2), z = 0.95 | rear (0.03, 0), front (0.90, 0) |
| Belt attachment points (z = 0.95; ray from the pelvis to each anchor on the belt ellipse) | rearL/R (0.305, +-0.097), frontL/R (0.539, +-0.082), latL/R (0.435, +-0.204), rearC (0.29, 0), frontC (0.55, 0) |
| BWS anchor (cantilevered top beam; cable exit = underside of the beam) | (0.42, 0, 1.80), directly above the nominal pelvis (raised in CAD so the beam clears the patient's chest; the cable is vertical, so statics are unchanged); hang load 1251 N; spool brake rated 46.91 N m; slack in horizontal arrest |
| Nominal pelvis / belt | (0.42, 0, 0.95); belt centre-line half-depth 0.13, half-width 0.205 (band 0.10 high, 0.04 thick) |
| Pelvis lift cap | <= 0.5 x 60 kg x g = 294 N in arrest (D3: 110.4-126.1 N at 300 N QP, <= 171 N under passive slip); 0.15 BW = 88 N in nudge (D3: <= 66.1 N) |
| Passive slip thresholds | corner 150, lateral 155, centre 120 N; spool radius 25 mm; rated hold 5.63 / 5.82 / 4.50 N m |
| Nudge | 150 N resultant, T_max 101.1 N per cable, tip SF >= 4.36 |
| Front-centre anchor | (0.90, 0, 0.95) is carried by the hinged front gate arm (section 8b): horizontal tube at z = 0.95, hinged on the front-right column (0.90, -0.33), latched to the front-left column (0.90, +0.33); the CF spool sits at its mid-span so the cable exit is exactly this anchor when latched. No post stands in the entry opening. Interlock: the controller refuses to arm unless the latch sensor reports latched |
| CAD | D3 CAD: cad/build_cad_d3.py → cad/catchpoint_d3.glb; renders in render/blender/d3/. The older cad/build_cad.py is the baseline SEA design, kept for reference |

Machine-readable and authoritative: data/d3_contract.json (read by sizing.py, run_analysis.py, slip_analysis.py and
cad/build_cad_d3.py; `test_d3_consistency.py` asserts contract, generated CAD parts and analysis agree within 1 mm). Its derived fields
(resultant bound, hold torques, lift caps, force wording) are regenerated by `make_report.py` through `publish.py`.

## 8b. Front gate (hinged arm carrying the front-centre anchor; data/gate.json from gate_check.py)

**Design.** A horizontal arm at pelvis height (z = 0.95 m) is hinged on the front-right column at (0.90, -0.33) and swings 90 deg open toward the outside (+x) for entry and exit; closed, it spans the opening and latches onto the front-left column at (0.90, +0.33); hinge-to-latch distance 0.66 m. The front-centre spool is mounted at the arm mid-span, so with the gate latched its cable exit is exactly the contract anchor (0.90, 0, 0.95) and the analysis geometry (sections 3-8) is unchanged. The cable line of action meets the arm axis, so the arm sees no torsion. With the gate open nothing stands in the entry opening below pelvis height.

**Interlock.** A latch sensor reports gate latched / not latched. The controller refuses to arm (no nudge, no catch) unless the gate is latched, and drops to safe-stop if the latch opens while armed. The mechanical latch is rated for at least 224.4 N (1.5 x the design cable force); a failure of the latch while armed (cantilever case) is therefore out of scope.

**Design loads on the gate** (the front-centre cable force, horizontal along x): (a) front-centre passive slip tension 120.0 N; (b) worst controller (QP) tension on frontC over the arrest and nudge rows of data/results.json: 149.6 N (arrest 149.6 N, nudge 75.9 N). The gate is sized for the larger, 149.6 N.

| Item (latched, simply supported span 0.66 m, load at mid-span) | Value |
|---|---|
| Hinge reaction (horizontal / resultant incl. weight) | 74.8 N / 75.4 N |
| Latch reaction (horizontal / resultant incl. weight) | 74.8 N / 75.4 N |
| Bending moment at the hinge / at mid-span | 0.0 N m (pin) / 24.69 N m (cable pull), 2.76 N m (weight) |
| Unlatched, information only (cantilever, load at mid-span): hinge reaction / hinge moment | 149.6 N / 49.37 N m |
| Tube | 40 x 20 x 2 mm 6061-T6 aluminium, wide side horizontal; I = 44458 mm^4, net of the 10 mm fairlead hole 30018 mm^4 (1500 mm^3) |
| Peak stress at the fairlead hole (Kt = 3.0, plus weight bending; spool 1.5 kg assumed) | 51.3 MPa against sigma_allow = 120 MPa (yield 240 MPa / 2) |
| Safety factor on yield | 4.68 (sigma_allow margin 2.34) |
| Mid-span deflection (cable exit shift) under the design load | 0.30 mm (anchor tolerance 1 mm) |
| Check | PASS |

Not validated: static hand calculation only; hinge and latch clearance, pin and latch hardware, fatigue and impact loading of the gate are not analysed. The 1.5 kg spool mass is an assumption.
