# CatchPoint — design spec (single source of truth)

ME5412 Robotics for Healthcare, group project proposal (one member's version).
Deadline: slides submitted as report by 3 Nov 2026; 10–15 min talk + 5 min Q&A.
All workers (Claude subagents, Codex) build from THIS file. If something here is
wrong or underspecified, write the question into `QUESTIONS.md` and pick the most
conservative reading; do not silently invent a different model.

## 1. Problem (healthcare first)

Stroke survivors and frail older adults fall often, and the skill that prevents
falls — a fast, well-placed *reactive recovery step* — has to be practised under
real loss of balance (perturbation-based balance training, PBT). Practising it is
dangerous, so in practice therapists and robotic gait trainers *over-support*:

- Current robotic walkers / body-weight-support (BWS) trainers (NTU MRBA, DRBA,
  IMU-brake walkers, Andago, KineAssist) detect a fall with a **reactive threshold**
  (force change, trunk tilt) and then **lock/brake/catch (binary)**.
  - Reported response: MRBA 0.28 s mechanical response; IMU walker 0.8 s actuation.
  - A low threshold catches early but takes away the patient's own recovery step
    (training value lost, many unnecessary catches); a high threshold is late → falls.
- Fixed-dose perturbation/BWS (e.g. ZeroG perturbation study, n=99 stroke) showed
  no benefit over BWS alone → dose and intervention need to adapt to the patient.

Evidence notes (full refs in research/novelty_and_sources.md):
- PBT meta-analysis (Mansfield 2015, 8 RCTs, n=404): fall rate ratio 0.54 (0.34–0.85);
  but chronic-stroke RCT (Mansfield 2018, n=88) not significant (RR 0.85, p=0.63) and
  Brown 2023 review: insufficient stroke-specific evidence → state honestly; our
  claim is about making higher-challenge practice *safe and adaptive*, not proven fall reduction.
- 71 % of stroke inpatients had impaired reactive stepping at discharge (Inness 2014);
  14.7 % fell during inpatient rehab (Mansfield 2013).
- Closest prior work: RobUST pelvic force field (Omofuma 2023) — restoring force only
  outside a subject-specific static pelvis-position region (assist-as-needed, healthy
  adults). CatchPoint differs: predictive XCoM with delay compensation, step-
  reachability region from the patient's own step capability, minimum-norm graded force,
  escalation to catch. Novelty phrasing: "to our knowledge", never "first".

## 2. Innovation (one sentence)

**CatchPoint is a predictive, delay-compensated safety layer for robotic gait
trainers that lets the patient attempt their own recovery step and applies the
minimum graded pelvic force only when the patient's predicted capture point
leaves their own recoverable region — escalating to a harness catch only when
that force cannot restore recoverability.**

Two-sentence version for slides: "Existing trainers ask *has the patient started
to fall?* and then lock. CatchPoint asks *can this patient still save themself?*
and helps only by the amount and at the moment that the answer turns to no."

It is a **layer**, not a new walker: sensing kit + 2-DOF pelvic series-elastic
interface + software that retrofits onto existing omnidirectional robotic walkers
(e.g. the NUS robotic walker) and can wrap other assistive devices (see §9).

## 3. System architecture (sense → estimate → decide → act)

| Layer | Components |
|---|---|
| Sense | Pelvic brace pose via SEA linkage encoders (AP/ML deflection + base odometry) → pelvis/COM position & velocity; 2D LiDAR at shin height (0.30 m) on the rear crossbar → both legs/feet tracked, swing/stance; 6-axis load cells in the two brace attachments → interaction force; IMU on brace → trunk tilt, redundancy; harness winch encoder + load cell. |
| Estimate | COM state (x, ẋ) with complementary/Kalman filter; extrapolated centre of mass ξ = x + ẋ/ω (Hof 2005) = instantaneous capture point (Pratt 2006); base of support from tracked feet; **delay compensation**: forward-predict the LIP state over Δ = τ_s + τ_a using the buffered commands; patient step-capability model (reaction time, max step length per leg, swing time) identified in a 5-min calibration session. |
| Decide | Recoverability check (§5.3) → predictive barrier / min-norm force (§5.4) → graded arbiter: **L0 Transparent** (F=0, walker follows) → **L1 Assist** (graded pelvic force ≤ 150 N) → **L2 Catch** (harness winch lock + base brake). Session-level assist-as-needed: margin m adapts so the patient succeeds on ~70 % of perturbations (challenge point). |
| Act | Two pelvic SEA modules (left/right), each 2-DOF (AP + ML) ball-screw SEA, ≤150 N per side continuous, force bandwidth ≥ 5 Hz; low-level admittance/force control (course L6); BWS winch with electromagnetic brake (holds ≥ 1.5 × body weight); omni base (4 omni wheels) with wheel brakes. |

## 4. Mechanical concept (for CAD)

Units metres. Origin at floor, x forward (walking direction), y left, z up.
- Base: U-shaped frame open at the FRONT, outer footprint 0.95 (x) × 0.80 (y),
  side rails 0.10 wide, rear crossbar 0.12 deep, height 0.10–0.18; four omni
  wheels Ø0.15 at the corners; battery/electronics box 0.35×0.25×0.15 on the rear crossbar.
- Two columns at the rear corners, 0.06×0.06 section, height 1.20 (telescopic look:
  lower 0.07 sleeve to 0.70, upper 0.06 tube to 1.20).
- Top rear beam between columns at z = 1.20 with BWS winch housing (0.20×0.15×0.12) at centre.
- Two side arms from the columns forward at pelvis height z = 0.95 (adjustable
  0.85–1.05), length 0.45, each ending in an SEA module: cylinder Ø0.07 × 0.20
  along x (AP axis) + small cylinder Ø0.05 × 0.12 along y (ML axis) with a visible
  coil spring section.
- Pelvic brace: padded U-shaped belt (open at front) around the pelvis, inner
  half-width 0.19, depth 0.22, belt height 0.10, attached to both SEA modules.
- Harness straps: two thin cylinders from the winch housing down to the brace.
- LiDAR: puck Ø0.07 × 0.05 on the rear crossbar at z = 0.30 (on a small mast).
- E-stop: red mushroom on the right column top; IMU box on brace rear.
- Human for renders: simple capsule humanoid, height 1.70, pelvis at z ≈ 0.95, standing
  inside the U with feet at (0, ±0.10).

## 5. Simulation model (the evaluation)

### 5.1 Human balance model — 3D linear inverted pendulum (LIP), horizontal plane
State COM x ∈ R² (horizontal), ẋ; constant height z0; ω = sqrt(g/z0); CoP p ∈ R².
External pelvic force F ∈ R² (robot).
  ẍ = ω² (x − p) + F / m
  ξ = x + ẋ/ω,   ξ̇ = ω (ξ − p) + F / (m ω)
Integrate with dt = 0.002 s (semi-implicit Euler or exact LIP step; either fine).

Feet: rectangles centred at foot centre, half-length 0.11 (x), half-width 0.045 (y),
axis-aligned. Start: quiet double stance, foot centres (0, +0.10) left, (0, −0.10) right,
COM at (0,0) at rest. Support polygon S = convex hull of feet in stance.
Paretic foot CoP region scaled by ankle factor a_p (rectangle half-sizes × a_p)
→ S_cop = hull(non-paretic foot rect, scaled paretic foot rect).

Ankle strategy (always on while in stance): p_des = ξ + k_c (ξ − c_S), k_c = 0.5,
c_S = centroid of S_cop; p = projection of p_des onto S_cop (single stance: onto
stance foot rect, scaled if paretic).

Stepping strategy:
- Trigger: ξ outside S_cop by more than ε = 0.01 m. The patient reacts after
  reaction delay τ_r (measured from first violation; if ξ returns inside before
  τ_r elapses, the trigger resets). Later steps in the same trial use 0.6 τ_r.
- Swing leg choice at lift-off: evaluate both legs; for each candidate, stance =
  other foot, predicted landing capture point
  ξ_land = p_st + (ξ − p_st) e^{ω T_s(leg)}, p_st = projection of ξ onto stance foot;
  desired foot centre = ξ_land (place foot under the capture point).
  Feasible if |desired − stance centre| ≤ L_max(leg) AND no crossover:
  (y_desired − y_stance)·side(leg) ≥ 0.08 (side = +1 left leg, −1 right leg).
  Pick the feasible leg with the smallest |desired − stance| / L_max; if none is
  feasible, pick the leg with the smallest violation and clip the target onto its
  reachable set (the patient will likely fall — that is the point).
- Swing duration T_s(leg); during swing S = stance foot only. At landing the foot is
  placed at the (lift-off-time) target; double stance resumes. Max 3 steps per trial.

Fall: horizontal distance from x to S (current support polygon) > d_fall = 0.25 m,
unless the robot catch is already engaged → FALL ends the trial.
Trial length 3.0 s; perturbation at t = 0.3 s as an instantaneous Δv on ẋ.

### 5.2 Patient population (sample per patient)
severity s ~ U(0,1); paretic side ~ {L,R} equiprobable; m ~ U(60,85) kg; z0 ~ U(0.85,1.00) m.
- τ_r = 0.20 + 0.30 s   [s]
- L_max(non-paretic) = 0.60 − 0.10 s ; L_max(paretic) = 0.50 − 0.30 s   [m]
- T_s(non-paretic) = 0.35 + 0.10 s ; T_s(paretic) = 0.40 + 0.25 s   [s]
- ankle factor a_p = 1 − 0.7 s (paretic foot CoP region scale)
Perturbation: |Δv| ~ U(0.2, 0.9) m/s, direction ~ U(0, 2π).
Monte Carlo: 50 patients × 40 perturbations = 2000 trials per controller setting;
fixed RNG seed 5412; identical patient/perturbation set across controllers.
These parameter ranges are modelling assumptions (cite as such); literature anchors
(reaction ~342±124 ms in older fallers, paretic step shorter) go in the slides.

### 5.3 Robot plant
- Pelvic force F (2D) = commanded force through pure delay τ_a = 0.07 s and first-
  order lag 0.02 s; sensing delay τ_s = 0.03 s on the measured state.
- Assist limit F_max = 150 N (L1). Catch (L2): after a lock delay τ_lock = 0.28 s
  from the decision, a spring–damper engages: F = −K_c (x − x_lock) − D_c ẋ,
  K_c = 4000 N/m, D_c = 800 N·s/m, x_lock = COM position at engagement; catch
  force unbounded (harness). Once engaged the trial outcome is CAUGHT.

### 5.4 Controllers
- **B0 None**: F = 0, no catch.
- **B1 Threshold-lock** (MRBA/DRBA style): trigger catch when |x − c_S| > θ
  (COM displacement from base-of-support centre, measured with sensing delay),
  θ swept {0.03, 0.05, 0.08, 0.11, 0.14, 0.18}. No assist force before the lock.
- **B2 Impedance + lock**: always-on pelvic spring-damper to the initial COM:
  F = −K (x − x0) − D ẋ, D = 2·0.7·sqrt(K m), |F| ≤ 300 N, K swept
  {100, 200, 400, 800, 1600, 3200} N/m; plus B1 lock fallback with θ = 0.14.
- **CP CatchPoint** (ours), every control tick (500 Hz):
  1. Delay compensation: take the measured state (delayed τ_s), forward-predict to
     t + τ_a with the LIP using the estimated CoP (projection of ξ onto current
     S_cop) and the buffered commanded forces → x̂, ẋ̂, ξ̂.
  2. Recoverable check with the IDENTIFIED patient model (true params × (1 + N(0, 0.10²)),
     drawn once per patient): if ξ̂ inside S_cop dilated by ε → F = 0 (L0).
     Else, for each candidate swing leg compute the time to landing
     T_land = remaining reaction time + T_s(leg) (during an ongoing swing:
     remaining swing time, and only that leg is a candidate), anchor a = stance
     foot centre, effective reach r = L_max(leg) − m (margin m),
     ξ_land(F) = p_a + (ξ̂ − p_a) e^{ω T_land} + β F,
     β = (e^{ω T_land} − 1)/(m_body ω²), p_a = projection of ξ̂ on the stance foot.
     Barrier h(F) = r − |ξ_land(F) − a| (respecting the no-crossover rule: a candidate
     whose target would cross over is infeasible). Best candidate = max h(0).
  3. Min-norm force (closed-form QP solution of min |F|² s.t. h(F) ≥ 0):
     if h(0) ≥ 0 → F = 0 (L0, patient can recover alone);
     else F* = −(|ξ_land(0) − a| − r)/β · u, u = (ξ_land(0) − a)/|ξ_land(0) − a|.
  4. Arbiter: |F*| ≤ F_max → L1 Assist, command F*. |F*| > F_max → command F_max·(F*/|F*|)
     AND trigger L2 Catch (lock delay τ_lock applies).
  5. Session-level AAN (optional figure): m_{k+1} = clip(m_k + 0.05 (0.70 − success_k), −0.05, 0.15)
     where success_k = SELF rate in session k (20 perturbations per session).
  Margin sweep for the Pareto curve: m ∈ {−0.06, −0.03, 0, 0.03, 0.06, 0.10, 0.15}; default m = 0.03.
- **CP-noDC** (ablation): CatchPoint without step-1 prediction (uses delayed state).

### 5.5 Outcomes per trial
- FALL: fall criterion met with no catch engaged.
- CAUGHT: catch engaged (no fall).
- SELF: no fall, no catch, robot impulse ∫|F|dt < 1.0 N·s.
- ASSISTED: no fall, no catch, impulse ≥ 1.0 N·s.
Also log: n_steps, peak |F|, impulse, time of first intervention, and
UNNECESSARY = robot caught or assisted in a trial where B0 (same patient, same
push) ended SELF.
Headline metrics: fall rate; self-recovery (training) rate = SELF %; unnecessary-
intervention rate; mean impulse.

### 5.6 Figures (results/figs/, PNG 200 dpi + SVG, consistent palette; colour-blind safe)
- F1 `pareto.png`: x = fall rate (%), y = self-recovery rate (%) — B1 θ-sweep line,
  B2 K-sweep line, CP m-sweep line, B0 single point, CP default highlighted.
  Title states the finding.
- F2 `timeline.png`: one showcase push (moderate patient, lateral push toward the
  paretic side): rows = B1 / B2 / CP; panels: distance of ξ from support vs time
  with recoverable bound, robot force |F|, level (L0/L1/L2), patient steps marked.
- F3 `recoverability_map.png`: top view for the showcase: feet, S_cop, reachable
  discs per leg (radius L_max − m around the other foot), COM and ξ paths for CP,
  path coloured by level (L0 grey, L1 amber, L2 red).
- F4 `delay.png`: fall rate vs total delay Δ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} s for CP vs CP-noDC.
- F5 `severity.png`: outcome stacked bars (SELF / ASSISTED / CAUGHT / FALL) per severity tercile, for B0, B1(θ=0.08), B2(K=800), CP(m=0.03).
- F6 `aan_sessions.png`: 10 sessions, margin m and SELF rate per session for 3 example patients (mild/moderate/severe).
- `results/summary.csv` (one row per controller setting) and `results/summary.md`.
- `results/showcase_<ctrl>.npz` for B0, B1, B2, CP on the showcase push, with keys:
  t (N,), com (N,2), comvel (N,2), xi (N,2), foot_l (N,2), foot_r (N,2),
  swing (N,) int {0 none, 1 left swinging, 2 right swinging}, F (N,2), level (N,) int {0,1,2},
  outcome (str), params (json str). Feet positions = current foot centre (swing foot
  interpolated linearly from lift-off to target during swing, with a lift height
  profile stored in key `swing_h` (N,) metres for animation).
- `results/anim_<ctrl>.gif`: top view + side (frontal-plane) stick figure, ≤ 6 MB each.

## 6. Actuator sizing (derive from sim, report in results/sizing.md)
From CP Monte Carlo (default m): 99th-percentile |F| and |dF/dt|, peak pelvis
velocity relative to the base during L1; then ball-screw SEA: lead 5 mm, η = 0.9 →
motor torque τ = F·lead/(2πη); speed from velocity; spring stiffness for 10 mm
deflection at F_max. Catch winch holding ≥ 1.5 m g at m = 85 kg.

## 7. Visual identity (all deliverables)
Palette (semantic): L0 Transparent = slate grey #6B7A8F; L1 Assist = amber #E0A030;
L2 Catch = red #C8443A; patient/COM path = deep teal #1F6F78; reachable region fill =
teal 15 % alpha; baselines: B0 #9AA0A6, B1 #7B5EA7 (violet), B2 #3C78B4 (blue), CP #1F6F78 (teal, bold).
Fonts in plots: DejaVu Sans is fine. Never put the concept name in a figure title;
titles state the finding.

## 8. Five-person division (each member owns one module = one talk segment)
R1 Clinical & user needs — problem stats, stroke balance/fall literature, existing
   solutions & gap, HAAT/ICF framing, clinical protocol & outcome measures (Berg
   Balance Scale, fall count, reactive-step metrics), ethics/regulation.
R2 Mechanical & CAD — walker retrofit, pelvic 2-DOF SEA module, BWS winch & brake,
   actuator sizing (§6), free-body diagrams, renders.
R3 Sensing & estimation — sensor suite, COM/ξ estimation, foot tracking, delay
   compensation, patient-model calibration protocol.
R4 Control & safety — recoverability barrier, min-norm force, graded arbiter,
   low-level admittance, AAN margin adaptation, stability/safety argument.
R5 Simulation, evaluation & integration — Monte Carlo, metrics, figures, demo
   video & interactive demo, slide integration and rehearsal.
Timeline (2026): W1 6–12 Oct agree topic (merge the 5 proposals); W2 13–19 Oct
design freeze; W3 20–26 Oct results & renders; W4 27 Oct–2 Nov slides, rehearsal;
3 Nov submit.

## 9. Mergeability (other members propose their own versions)
CatchPoint is a reusable pattern: "estimate the human's own recoverability →
stay transparent → minimum graded assistance → escalate". It drops into:
- Sit-to-stand / transfer assist (P1): barrier on COM-over-feet during rise.
- Upper-limb rehab (P2): barrier on trunk compensation instead of balance.
- Dressing / feeding arms (P3/P4): barrier on contact force with a moving limb/head.
- Exoskeletons: XCoM-based hip assist instead of a walker pelvic interface.
Every deliverable keeps modules separable (one slide section per module, figures
as standalone files, editable native shapes for diagrams).

## 10. Repository layout
```
SPEC.md  QUESTIONS.md
research/          literature notes and references
sim/               catchpoint/ (model.py, controllers.py, montecarlo.py), run_all.py, tests/
cad/               build_cad.py → catchpoint_walker.glb/.stl, parts
render/            mujoco scene + render scripts → renders/*.png, *.mp4
results/           figs/, summary.csv, showcase_*.npz, anim_*.gif
web/               index.html (interactive demo, single file) 
deck/              build_deck.py → CatchPoint_proposal.pptx
```
Python: `.venv/Scripts/python` (3.9, numpy 2.0, scipy, matplotlib 3.9, mujoco 3.2.7,
trimesh, python-pptx, imageio). Windows; use forward-slash paths; no GUI windows
(matplotlib Agg backend; MuJoCo offscreen `mujoco.Renderer` works).
