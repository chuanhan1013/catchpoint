# Credits and licenses

## Humanoid model: Unitree G1 (MuJoCo Menagerie)

- Source: https://github.com/google-deepmind/mujoco_menagerie, directory `unitree_g1/`
  (files used: `g1.xml` and the 35 meshes `assets/*.STL`; downloaded 2026-10-22 from branch `main`,
  last commit touching that directory: `cda5236f98383f6fbaa4820c6b18dba892cd087f`, 2026-10-20).
- Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO., LTD. ("Unitree Robotics"). All rights reserved.
- License: **BSD-3-Clause** (full text in `g1/LICENSE`, copied unchanged from the Menagerie repository; the MuJoCo
  Menagerie README lists the G1 under this license). Redistribution with the copyright notice and the disclaimer is
  satisfied by keeping `g1/LICENSE` next to the meshes.
- Modifications (done in memory by `build_scene.py`, the files in `g1/` are unmodified): uniform scale x1.30
  (1.27 m -> 1.65 m), collision geometry/actuators/sensors removed, materials recoloured (teal body, black joints),
  two sites added on the ankle links. The robot is used here only as a stand-in for a (stroke) patient; the
  "UNITREE" logo mesh on the torso is part of the original model.
- Unitree and the G1 are trademarks of their owners; this project is not affiliated with or endorsed by Unitree.

## Other
- MuJoCo (Apache-2.0), numpy, Pillow, imageio / imageio-ffmpeg, ffmpeg (palette GIF encoding): used as installed in
  the project venv.
- Walker geometry: `cad/build_cad_d3.py` / `cad/parts_d3/*.stl` of this repository (our own concept CAD).
- The simulated trajectories come from this repository's LIP simulator (`sim/catchpoint`, SPEC section 5).
