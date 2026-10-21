# CatchPoint

**Let them step. Catch only when they can't.**

A recoverability-aware safety layer for robotic balance trainers (NUS ME5412, Robotics for Healthcare course project). Live site: https://chuanhan1013.github.io/catchpoint/

Everything here is simulation and statics. It is not clinical evidence; the physics demo is a robot-policy demonstration; the hardware design (D3) is unbuilt and unvalidated.

## Contents
| Path | What |
|---|---|
| `index.html`, `assets/`, `site/` | Showcase website (static, no build). Add videos by editing `site/videos.json`. |
| `sim/` | `catchpoint` Monte Carlo package, tests, `run_all.py` |
| `analysis/i2_sizing/` | Walker/cable sizing analysis, data, figures, `report.md` |
| `physics_demo/` | MuJoCo physics demo (Unitree G1 policy with simulated weak leg), PPO fine-tune code |
| `render/` | Render and replay scripts |
| `results/`, `SPEC.md`, `docs/` | Result tables and figures, specification, team plan, sources |

Not included: third-party assets (product photos, Mixamo models, vendored libraries), course materials.

## License and credits
All rights reserved unless stated; third-party credits in CREDITS (`physics_demo/CREDITS.md`, `render/mujoco_humanoid/CREDITS.md`). Unitree G1 (BSD-3 via MuJoCo Menagerie / unitree_rl_gym), Mixamo via three.js examples, Poly Haven (CC0), Tabler icons (MIT).
