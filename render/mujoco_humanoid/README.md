# MuJoCo kinematic replay: G1 humanoid inside the CatchPoint D3 walker

Pitch videos/GIFs of the simulated trajectories (SPEC section 5). **Visualization only**: the humanoid is posed
kinematically (`mj_forward`, no dynamics, no balance controller); every frame is driven by the LIP simulation data.

    PY=.venv/Scripts/python                     # from the repo root
    $PY render/mujoco_humanoid/gen_cp_scenarios.py   # data/cp_{small,bigger,big}.npz (uses sim/catchpoint, unchanged)
    $PY render/mujoco_humanoid/replay.py all         # all videos, GIFs and the 3-panel still
    $PY render/mujoco_humanoid/replay.py compare|scen|still
    $PY render/mujoco_humanoid/check_clearance.py    # geometry sanity numbers (IK residual, floor, belt/walker overlap)

| File | Content |
|---|---|
| `build_scene.py` | scene builder: walker (cad/parts_d3 STLs), pelvic belt, 8 cable + 2 BWS-strap tendons, 4 outriggers, G1, arrow, capture-point ring; class `Rig` |
| `replay.py` | data loading, walker/cable/outrigger state, trunk lean + damped-least-squares IK, HUD overlay, mp4/gif/still writers |
| `gen_cp_scenarios.py` | three CP scenarios (same patient as `results/showcase_*.npz`, only the push magnitude changes) |
| `check_clearance.py` | numeric geometry checks |
| `compare_4controllers.mp4/.gif` | 2x2 "same push, four controllers" (B0, B1, B2, CP) from `results/showcase_*.npz`, 1920x1080 |
| `cp_small / cp_bigger / cp_big .mp4/.gif` | CP on three pushes (1280x720, 30 fps, 6.3 s; GIF 960 px, 15 fps) |
| `cp_three_states.png` | three-panel still (SELF / ASSISTED / CAUGHT) |
| `g1/`, `CREDITS.md` | Unitree G1 from MuJoCo Menagerie (BSD-3-Clause) and its license |

## Scenarios (`data/*.npz`, same keys as `results/showcase_*.npz`, SPEC 5.6)
Patient: severity 0.6, left leg paretic, 72 kg, z0 = 0.92 m, identified-model noise draw as in `sim/run_all.py:showcase()`.
Push 60 deg from forward (mostly lateral, toward the weak left side), CP margin 0.03. Chosen by sweeping the magnitude:

| File | push | outcome | peak force | impulse |
|---|---|---|---|---|
| `cp_small.npz` | 0.45 m/s | SELF (ankle strategy only, no intervention) | 0 N | 0 N s |
| `cp_bigger.npz` | 0.55 m/s | ASSISTED (L0 -> L1, patient steps with the right leg) | 119 N | 16.8 N s |
| `cp_big.npz` | 0.60 m/s | CAUGHT (L0 -> L1 -> L2 at 1.09 s, then lock) | 150 N | 59.8 N s |

(Sweep: at this angle CP gives SELF up to 0.45 m/s, ASSISTED 0.475-0.55, CAUGHT from 0.575 m/s.)

## How a frame is built
- Coordinates: sim (x forward, y left) + (0.42, 0) = contract pelvis position; units m.
- **Humanoid**: Unitree G1 scaled x1.30 (1.65 m). Free joint = pelvis at `com`, pelvis-frame height `z0 + 0.075` (slight knee bend),
  lowered automatically if a foot target is out of reach (hip-to-sole distance > 0.96 x straight-leg reach). Trunk lean towards (com - support centre),
  angle = 1.3 |com - support| / z0 (clipped 0.55 rad): 40 % at the pelvis, 60 % in the waist pitch/roll joints.
  Legs: damped-least-squares IK (`mj_jacSite`, 6-DOF per leg, flat foot, Levenberg-Marquardt accept/reject, step cap, joint limits;
  hip adduction and ankle roll limits widened for visual-only crossover steps; residual <= 3 mm on all 7 trajectories) so the foot-sole centres reach `foot_l/foot_r`;
  the swing foot is lifted by `swing_h` x 1.5. Arms hang relaxed (shoulder roll 0.28, elbow 1.25 rad).
- **Walker** from `cad/parts_d3` (contract coordinates, 4 outriggers rebuilt from `parts_d3.json`). The base tracks the
  pelvis with a 0.15 s lag (pelvis at most 8 cm off the base centre) in L0/L1 ("L0 transparent, walker follows", SPEC 3),
  and is braked from the first L2 sample (dragged only beyond 5 cm), the same logic as `render/animate.py`.
- **Belt** follows the pelvis (translation; centre 1.3 cm below the G1 pelvis-frame origin so it wraps pelvis/waist and the hips clear it). **Cables**: 8 spatial tendons anchor(exit point of the spool) -> belt attach point,
  each through a sag marker: L0 slack, grey, thin, with a visible sag (10 % of the chord); L1 the 2-3 cables whose pull
  direction is closest to the robot force F (cos > 0.35, at least 2) become taut, amber, thicker, the others stay slack;
  L2 all taut and red. Slack <-> taut transition: 50 ms low-pass. BWS straps (spreader -> belt) are slack/navy until L2, taut/red on catch.
- **Outriggers** retracted (folded against the rails, pads raised) until the first L2 sample, then deploy in 0.28 s.
- **Force arrow** at the pelvis when |F| > 5 N (level colour, length 0.0028 m/N, 0.14-0.42 m), **capture-point ring** on the floor
  at xi (green L0 / amber L1 / red L2).
- **HUD** (PIL): controller, level badge, time (and "0.5x slow"), |F|, push info, outcome stamp in the hold; footer disclaimer.
- Timing: 1x until 0.2 s, **0.5x from 0.2 s to 2.0 s** (push at 0.3 s), 1x afterwards, then 1.5 s freeze with the outcome.

## Deviations from the CAD / SPEC (visualization choices)
- BWS spreader bar and its straps are raised by 6.5 cm (1.68 -> 1.745 m) so the 1.70 m humanoid's head clears it; the cantilever
  and the 1.80 m anchor of the contract are untouched.
- Walker base follows the pelvis (see above) instead of staying fixed; with a fixed base a 0.3 m lateral displacement drives the
  belt through the lateral spools/posts (the LIP simulation has no walker geometry).
- G1 is a 29-DOF robot standing in for a patient (the "UNITREE" torso logo is visible); proportions differ from the patient
  model (leg length 0.69 m x 1.3).
- The LIP model has no leg/foot collision with the walker; feet may pass close to rails/outriggers, and a "fall" (B0) just stops at
  the fall criterion (no ground impact is animated).
- The logged `level` is the command level; while the delayed/lagged force is still > 5 N after the level has dropped to 0, the HUD shows L1 (badge, arrow, cables, ring).
- Swing-foot lift is exaggerated x1.5 (data: max 6 cm).
- `d3_contract.json` anchor coordinates are used for the walker; the cable end at the walker is the spool fairlead
  (`exit` in `parts_d3.json`), not the abstract anchor point.
