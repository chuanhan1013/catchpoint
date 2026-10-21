# physics_demo — a pretrained humanoid policy as a physics "patient" for CatchPoint

Feasibility spike, 22 Oct 2026. **Status: works end-to-end. Priority 1 (pretrained policy + impairment + pushes +
B0/B1/CP videos) done; priority 2 (PPO push-recovery fine-tune, before/after) done, see the last section.** A pretrained Unitree G1 locomotion
policy runs in MuJoCo with a crude stroke-like impairment, is pushed laterally toward its weak side, and a walker
applies the controller's pelvic force through 8 cables. B0 / B1 / CP closed-loop videos are in `videos/`.

> A robot policy is not a stroke patient. Everything here is a *demonstration* of the control idea on a physically
> simulated, self-balancing body. It is not evidence about patients.

## What it is

| Piece | Implementation (`g1cp.py`) |
|---|---|
| "Patient" | Unitree G1 12-dof, pretrained `motion.pt` from unitree_rl_gym (stepping-in-place gait, 0.8 s clock, zero velocity command), MuJoCo 3.2.7, dt 2 ms, policy 50 Hz, PD as in the official deploy script. 32.1 kg, pelvis 0.78 m. |
| Impairment | Right (paretic) leg actuator torque limits × 0.5; policy observation delay 40 ms + action delay 20 ms (60 ms total). |
| Push | Horizontal force pulse on the pelvis (`xfrc_applied`), 0.1 s, toward −y (weak side), at t = 2.4 s. |
| Walker | D3-contract geometry (`analysis/i2_sizing/data/d3_contract.json`) scaled by 0.82 so the belt is at the G1 pelvis; follows the pelvis kinematically (lag 0.25 s) in L0/L1, brakes on catch / fall. Visual only (no collision). |
| Cables | 8 cables (4 corner @1.2 m, 2 lateral, 2 centre; heights scaled). Pelvic force = cable resultant: min-norm non-negative tension distribution (`min_norm_eq_box` from `sizing.py`); during the catch each cable is capped at its slip tension. The 3-D resultant and its moment are applied to the pelvis via `xfrc_applied`. Cables are drawn coloured by tension (grey slack → amber → red at slip). |
| Force scaling | All SPEC force levels are scaled by body mass, λ = 32.1/75 = 0.43 (150 N assist → 64 N; 300 N catch cap → 128 N; K_c, D_c, slip tensions likewise), so the robot gets the accelerations a 75 kg person would get. Overlays show the robot force and the 75 kg-equivalent. |
| Sensing / actuation | COM = `subtree_com` of the pelvis, velocity from `mj_subtreeVel`, feet = ankle-roll bodies + foot offset, stance from foot–floor contacts. τ_s = 30 ms sensing delay, τ_a = 70 ms pure delay + 20 ms lag on the commanded force, τ_lock = 0.28 s (SPEC §5.3). |
| B0 | No assistance (walker follows, no cables drawn). |
| B1 | Threshold lock: \|COM − feet centre\| > θ = 0.08 m (delayed) → after 0.28 s lock: spring-damper to the COM at engagement (K_c, D_c mass-scaled), ≤ 300 N-equivalent, slip-limited, plus BWS harness. |
| CP | Delay-compensated XCoM (exact LIP propagation over τ_s+τ_a with logged/queued forces, as in `sim/catchpoint`), recoverability barrier, closed-form min-norm force ≤ 150 N-equivalent, escalation to the slip-limited catch (+ harness). |
| Catch (both) | Lock: horizontal spring-damper (slip-capped) + BWS harness (one-sided vertical spring, ≤ 1.5 m g) + harness tilt restoring torque (vest straps). Outcome = CAUGHT. |

## Outcomes (push × controller), impaired G1, push at t = 2.4 s, 0.1 s, toward the weak side

| Push | B0 none | B1 threshold lock (θ 8 cm) | CP CatchPoint |
|---|---|---|---|
| (a) 100 N (Δv 0.31 m/s) | recovered alone | recovered alone | recovered alone (0 N) |
| (b) 180 N (Δv 0.56 m/s) | **fall** (t = 3.36 s) | **caught** (trigger 2.65 s; peak 300 N-eq, impulse 494 N·s-eq) | **recovered with a nudge** (peak 150 N-eq, impulse 81 N·s-eq), never caught |
| (c) 300 N (Δv 0.93 m/s) | **fall** (t = 3.07 s) | **caught** (trigger 2.55 s) | **caught** (assist saturated → escalated at 2.56 s) |

Peak = peak horizontal cable resultant; impulse = ∫|F_xy|dt; both in 75 kg-equivalent units (÷λ).
Fall = pelvis < 0.55 m or torso tilt > 45°. "Nudged" = robot impulse ≥ 1 N·s-eq without a catch (SPEC §5.5). Steps
(≥ 5 cm foot moves after the push) are logged in `outcomes.json` but say little: the policy steps in place continuously.

### Robustness: 56 pushes (7 magnitudes 60–300 N × 8 push times 2.0–2.7 s, i.e. different gait phases)
`sweep_all.py`, `sweep_pareto.py` (counts out of 56):

| Controller | fall | caught | nudged | hands-off | not caught & no fall |
|---|---|---|---|---|---|
| B0 | 32 | – | – | 24 | 24 |
| B1 θ = 0.08 / 0.11 / 0.14 / 0.18 m | 0 / 0 / 0 / 1 | 46 / 41 / 37 / 31 | – | 10 / 15 / 19 / 24 | 10 / 15 / 19 / 24 |
| CP r0 = 0.10 / **0.14 (default)** / 0.20 / 0.28 m | 0 / 0 / 0 / 0 | 24 / 26 / 33 / 33 | 27 / 23 / 14 / 9 | 5 / 7 / 9 / 14 | **32 / 30** / 23 / 23 |

Reading (default CP): of the 32 pushes B0 falls from, CP catches 24 and rescues 8 with a nudge alone; B1 catches all
of them. CP keeps more trials going without a catch than any B1 threshold, with zero falls. But it nudges often:
15 of the 24 pushes the robot survives alone get a nudge (median impulse of those nudges ≈ 33 N·s-eq; 46 N·s-eq
over all 23 nudges), and 2 get caught. B1
with a high threshold is more "hands-off", but it catches everything else. Outcomes depend strongly on the gait phase
at push onset.

## Videos (`videos/`)
- `{a,b,c}_{B0,B1,CP}.mp4`: 1280×720, 30 fps, 6 s (sim 1.0–7.0 s). Overlay: controller, level (Hands off / Nudge /
  Catch), pelvic force (robot N and ≈75 kg-equivalent), push label, outcome, footer disclaimer.
- `compare_b.mp4` (1920×820) and `compare_b.gif` (960 px, 12 fps, 5.8 MB): push (b), B0 | B1 | CP side by side.
- `grid_3x3.mp4` (2080×1124): all pushes × controllers.
- `raw/`: renders without text + per-frame metadata JSON (used by `compose.py`).

## Important deviations from SPEC (read before presenting)
1. **The SPEC step-reachability model did not predict this policy's falls.** I implemented the LIP landing-XCoM
   barrier with a phase-clock patient model (`landing_xi`, `calib4.py`). Over 56 B0 calibration pushes it separated
   falls from recoveries hardly better than chance (AUC 0.51–0.61 for any foot dilation / ω scaling tried). The policy
   recovers with several fast side-shuffles and partly with hip/arm motion, which a single-step LIP model with
   L_max does not capture; falls are often a paretic-knee collapse a step later.
2. So CP's **recoverable region is calibrated empirically**: the delay-compensated XCoM, propagated T_H = 0.4 s with the
   LIP, may leave the both-feet support hull by at most R = r0·e^{ωT_H} − m (r0 = 0.14 m toward the weak side,
   0.25 m otherwise, margin m = 0.03). The min-norm force is the SPEC closed form with β = (e^{ωT_H}−1)/(mω²).
3. **Escalation rule changed**: SPEC escalates as soon as F* > F_max. Here that caught every push ≥ 140 N, including
   ones that a saturated 150 N-eq nudge alone rescues (`sweep_noesc.py`). CP escalates when the barrier residual left
   after the saturated assist exceeds 0.45 m for ≥ 3 ticks (calibrated with `calib_esc.py`).
4. Forces are mass-scaled (λ = 0.43), see above. B1's θ is measured from the midpoint of both feet, because in a
   stepping-in-place gait the single-stance centroid jumps ±0.1 m every step.

## Limitations
- A robot locomotion policy ≠ a stroke patient: different mass (32 kg), height (1.3 m), feet, control (RL policy on a
  0.8 s gait clock), no fear/bracing/arm grabbing. Paretic weakness is crude (torque caps + delay only; no spasticity,
  no sensory loss, no altered timing). The unimpaired policy survives 300 N pushes at most phases; the impaired one
  falls from 100–140 N at unlucky phases.
- Calibration (r0, escalation threshold, θ) was done on the same kind of pushes that are reported → optimistic; there
  is no held-out set. Deterministic simulation; results are sensitive to push timing.
- The walker is kinematic and visual (no collision with the robot, no wheel dynamics, no walker tipping); cable
  tensions are a static distribution, not cable/spool dynamics. The harness tilt torque is an assumption.
- The policy keeps running while caught (it was never trained to hang in a harness).
- Results are a demonstration, not evidence.

## Reproduce
```
# from the repo root (Windows, Git Bash); venv: Python 3.9, mujoco 3.2.7, numpy, scipy, pillow, imageio(-ffmpeg)
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # inference only
# third_party: sparse checkout of unitree_rl_gym (see CREDITS.md)
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_gym.git physics_demo/third_party/unitree_rl_gym
git -C physics_demo/third_party/unitree_rl_gym sparse-checkout set deploy/deploy_mujoco deploy/pre_train/g1 resources/robots/g1_description
cd physics_demo
../.venv/Scripts/python check_imp.py      # unperturbed standing, impaired variants, 10 s
../.venv/Scripts/python trials.py         # 3 pushes x 3 controllers -> outcomes.json
../.venv/Scripts/python render.py         # raw renders (9 parallel processes, ~1 min)
../.venv/Scripts/python compose.py        # overlays, compare_b, grid, GIF
../.venv/Scripts/python sweep_all.py      # 56-push robustness table;  sweep_pareto.py for the settings sweep
```
Calibration scripts (kept for traceability): `sweep_push.py` (B0 nominal vs impaired), `calib2.py` + `calib2_an.py`
(XCoM excursion data, writes `../_tmp/calib2.pkl`), `calib4.py` (LIP step-reach model test), `sweep_noesc.py`,
`calib_esc.py` (escalation), `sweep_final.py`.

Checked on this machine: the nominal policy stands/steps in place for 12 s (drift ≈ 2 cm/s); the impaired one
(×0.5 torque, 60 ms delay) for 10 s without falling (drift ≈ 4 cm/s, max torso tilt 6.4°). With 80 ms delay the
impaired policy falls unperturbed after ~9 s, so 60 ms was used.

## Priority 2: PPO push-recovery fine-tuning of the impaired policy (done, 08:15)
`ppo/`: the pretrained actor is an LSTM(47→64) + MLP(64→32→12, ELU); `ppo/net.py` re-implements it (max deviation
from `motion.pt` 3e-6) so it can be trained and exports back to the same stateful TorchScript interface. `ppo/env.py`:
the same impaired G1 (×0.5 right-leg torque, 60 ms delay), 10 s episodes, random 0.1 s pelvic pushes 60–300 N every
2–3.5 s (75 % toward the weak side ±20°, else any direction). Reward: alive +1, fall −50, upright, pelvis height,
velocity, action-rate and torque penalties. `ppo/train.py`: recurrent PPO (truncated BPTT over 64-step chunks,
hidden-state resets at episode ends), asymmetric critic (privileged 53-d state), behaviour-cloning penalty to the
frozen pretrained actor (weight 0.1), 52–60 parallel envs on CPU (GPU not needed for this network size).
Runs: ft1 1200 iterations (4.6 M steps, 31 min) + ft2 resumed for 1500 iterations (5.0 M steps, 38 min). Stochastic
training-episode fall rate (10 s, repeated pushes) 0.97 → 0.34. Final policy: `ppo/policy_ft.pt`
(= `ppo/runs/ft2/policy_1500.pt`; logs in `ppo/runs/*/log.csv`).

Test on the same 56 pushes as above (deterministic; `ppo/compare.py`, `ppo/recal_cp.py`):

| | B0 fall | B0 recovered alone | CP caught | CP nudged | CP hands-off | CP fall |
|---|---|---|---|---|---|---|
| pretrained (CP r0 0.14) | 32 | 24 | 26 | 23 | 7 | 0 |
| fine-tuned, CP not re-identified (r0 0.14) | 3 | 53 | 11 | 26 | 19 | 0 |
| fine-tuned, CP re-identified (r0 0.20 / 0.28 / **0.36**) | 3 | 53 | 9 / 6 / **3** | 23 / 20 / **16** | 24 / 30 / **37** | 0 |

- Fine-tuning cut B0 falls from 32 to 3 of 56. Unperturbed it still steps in place stably (12 s, max tilt 4.4°), with
  a wider stance (mean foot separation 0.27 vs 0.16 m) and more drift (0.88 m in 12 s, ≈ 7 cm/s).
- CP has to be re-identified for the stronger "patient": with the old region it over-assists (26 nudges, 8 catches
  on pushes the robot survives alone). With r0 = 0.36 it catches 3 times (2 of the 3 B0-falls + 1 unnecessary), nudges
  16 times and stays hands-off in 37 of 56. This mirrors the SPEC's per-patient identification and AAN margin idea.
- Videos: `videos/beforeafter_{a,b,c}.mp4` (1920×1180, 2×2: B0 / CP × pretrained / fine-tuned) and
  `videos/beforeafter_c.gif` (7.2 MB), `videos/beforeafter_b.gif` (7.1 MB). Push (c), 300 N: the pretrained robot falls
  (B0) or is caught (CP); the fine-tuned robot recovers alone (B0) or with an 8 N·s-eq nudge (CP, peak 68 N-eq).
  Reproduce (cwd physics_demo, ~70 min on 16 cores):
  `../.venv/Scripts/python ppo/train.py --iters 1200 --workers 15 --envs 4 --bc 0.1 --out ppo/runs/ft1`, then
  `ppo/train.py --iters 1500 --workers 13 --envs 4 --bc 0.1 --warmup 5 --lr 1.5e-4 --resume ppo/runs/ft1/ckpt_1200.pt --out ppo/runs/ft2`,
  copy `ppo/runs/ft2/policy_1500.pt` to `ppo/policy_ft.pt`, then `ppo/compare.py ppo/policy_ft.pt`,
  `ppo/recal_cp.py ppo/policy_ft.pt`, `ppo/render_ft.py`, `compose.py beforeafter`. Seeds are fixed, but multi-process
  timing makes a re-run not bit-identical. `ppo/eval_env.py` evaluates a checkpoint in the training environment.
- Caveats: test pushes come from the training distribution (same magnitudes and directions), so this is not a
  held-out test; r0 was re-identified on the same 56 pushes. A robot that learned to resist pushes is a better robot,
  not a better model of a stroke patient. Label these videos as a robot-policy demonstration.

## v2 video suite (`videos_v2/`, copies in `CatchPoint_proposal/videos/physics_v2/`)
Longer multi-push sessions, built for clarity. v1 (`videos/`) is unchanged.

| File | Content |
|---|---|
| `open_ground_healthy_vs_patient.mp4` (1920×1080, 25 s) | No walker. Unimpaired G1 policy vs the impaired "patient" (right-leg torque ×0.5, 60 ms delay), same pushes. |
| `walker_4ways.mp4` (1920×1080, 25 s) | Impaired patient in the D3 walker, 2×2: B0 none / B1 threshold lock θ 0.08 m / B2 always-on spring / CP CatchPoint (default). |
| `cp_story.mp4` (1280×720, 25 s) | CatchPoint only, with a caption for each level. |
| `*.gif` | Full-length GIFs (720 px / 680 px wide, 8–10 fps, 8.1–9.7 MB) plus `*_push3-4_960.gif` cuts (960 px, 10–11 s, ≤ 8.8 MB). |
| `*_key.png`, `*_endcard.png` | One key still per video and the session-summary card. |

**Push sequence (same for all):** four 0.1 s lateral pelvis pushes toward the weak (right) side, 60 / 120 / 180 /
260 N at t = 2.4 / 6.4 / 10.4 / 14.4 s. Pushes are 4.0 s apart, a multiple of the 0.8 s gait clock, so every push
lands at the same gait phase. The sequence was picked from a small search (`v2/seq_search.py`: 2 magnitude sets ×
2 phases, with the outcome printed per push) so that the healthy policy survives everything while the impaired one
falls at a medium push. A robot stops being pushed once it has fallen.

**Session rules (new in `g1cp.Sim`):** `push` accepts a list. `release_after=2.0` releases a catch once it has been
held ≥ 2 s **and** the COM is within 6 cm of the feet centre at < 0.2 m/s (a therapist re-centres the patient). Then
the session continues.

Each push gets a per-push outcome (`Sim.per_push`), judged over the 2.2 s after the push:
- **fell**: a fall without a lock.
- **caught**: a lock engaged within 2.2 s, or a lock already held when the push arrived (marked "held").
- **nudged**: device impulse ≥ 1 N·s (75 kg-equivalent).
- **alone**: otherwise.

A lock that engages outside every push window counts as a false alarm (shown on the end card).

**B2 (new, SPEC §5.4):** pelvic spring-damper toward the COM recorded at t = 1.0 s. K = 200 N/m × λ (mass-scaled,
86 N/m on the 32 kg robot), D = 2·0.7·√(K m). |F| ≤ 300 N-equivalent. Plus a B1-style lock fallback when
|COM − feet centre| > 0.14 m (0.28 s lock delay as B1). The force goes through the same delay/lag pipeline and the
8-cable tension distribution as CP.

**Walker visual:** the floor-standing front-centre post is replaced by the D3 hinged front gate: a horizontal bar at
belt height between the front columns, with the hinge at −y and a yellow latch at +y (`render.draw_walker(gate=True)`).
The anchor positions are unchanged, so the cable math is unchanged. v1 keeps the old post.

**Outcomes (from the JSON; also on the end cards):**

| | Push 1 (60 N) | Push 2 (120 N) | Push 3 (180 N) | Push 4 (260 N) |
|---|---|---|---|---|
| healthy, no walker | alone | alone | alone | alone |
| patient, no walker | alone | alone | **fell** | – |
| B0 walker, no device | alone | alone | **fell** | – |
| B1 threshold lock | alone | caught (113 N·s) | caught (held) | caught (held) |
| B2 always-on spring | nudged (41 N·s) | nudged (65) | nudged (108) | caught (232) |
| CP CatchPoint | alone | nudged (2) | nudged (26) | caught (194) |

Impulses are 75 kg-equivalent, over the 2.2 s after each push. B1 also has 2 locks with no push: after its first
release the COM drifts > 8 cm from the feet centre and it re-locks, so it is still locked when pushes 3 and 4 arrive.

**Visual elements:**
- Camera: closer, lower front-3/4 view (azimuth 205°, elevation −7°). The robot fills about 63–68 % of the panel
  height while standing.
- 0.5× slow motion from 0.25 s before to 1.25 s after each push, with a SLOW-MO tag.
- Red push arrow at the pelvis.
- COM trail on the floor: teal dots, last 3.5 s.
- Outcome banners that hold 1.2 s. FELL stays up for the rest of the clip.
- Running tally chips, a level badge with the force readout, and a timeline strip per panel (green L0, amber L1,
  red L2, dark red after a fall) with push ticks.
- End card (3 s) with the per-push table, impulses and a one-line takeaway.

**Reproduce:** `python v2/render_v2.py` (7 raw panels in parallel, ≈ 35 s), then `python v2/compose_v2.py`
(overlays, end cards, GIFs, key stills). GIF cuts: see the ffmpeg palettegen commands in `compose_v2.make_gif`
(`-ss 10 -t 10` for walker_4ways, `-ss 9 -t 11` for open_ground, 960 px, 10 fps).

**Honest notes:**
- One deterministic session per controller, at one gait phase. In the 56-push sweep above the outcomes depend
  strongly on push phase, so treat these as illustrations, not statistics.
- The CP nudge at push 2 is tiny (2 N·s-eq): it counts as "nudged" only because the SPEC threshold is 1 N·s.
- The healthy-vs-patient contrast comes from our crude impairment (torque caps + delay) on a robot policy, not from
  patient data.
- After a catch release the robot policy is in an unusual state. B1's re-locks partly reflect this simulator's
  release rule (re-centre then release).
- The walker is still kinematic and visual only (no collision; the fallen B0 robot passes through the frame).
