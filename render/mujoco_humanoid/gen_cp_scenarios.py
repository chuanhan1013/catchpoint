"""Generate the three CatchPoint scenarios (small / bigger / big push) with the UNCHANGED simulator.

Same patient as results/showcase_*.npz (severity 0.6, left leg paretic, m = 72 kg, z0 = 0.92 m, same identified-model
noise draw), push at 60 deg from forward (mostly lateral, toward the paretic = weak left side), CP margin 0.03.
Only the push magnitude changes:  0.45 m/s -> SELF,  0.55 m/s -> ASSISTED,  0.60 m/s -> CAUGHT.
Saved in the SPEC 5.6 / sim/run_all.py:save_npz format.   usage: python gen_cp_scenarios.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from catchpoint.controllers import CP                                          # noqa: E402
from catchpoint.model import Config, build_pop, params_json, sample_noise, simulate   # noqa: E402
from catchpoint.montecarlo import SEED                                         # noqa: E402

SCEN = {"small": ("SELF", 0.45), "bigger": ("ASSISTED", 0.55), "big": ("CAUGHT", 0.60)}
KEYS = ["t", "com", "comvel", "xi", "foot_l", "foot_r", "swing", "F", "level", "swing_h"]


def main():
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    ang = np.radians(60.0)
    for name, (want, mag) in SCEN.items():
        noise = sample_noise(np.random.default_rng(SEED), 1)       # identical draw to run_all.showcase()
        pop = build_pop([0.6], 0, 72.0, 0.92, [[mag * np.cos(ang), mag * np.sin(ang)]], noise=noise)
        r = simulate(pop, CP(0.03), Config(), record=True)
        d = r["rec"]
        outcome = ["SELF", "ASSISTED", "CAUGHT", "FALL"][int(r["outcome"][0])]
        assert outcome == want, (name, outcome, want)
        np.savez(os.path.join(out, f"cp_{name}.npz"),
                 **{k: (d[k].astype(int) if k in ("swing", "level") else d[k]) for k in KEYS},
                 outcome=np.array(outcome), params=np.array(params_json(pop)))
        print(f"cp_{name}.npz  push {mag:.2f} m/s  outcome {outcome}  impulse {float(r['impulse'][0]):.1f} N s  "
              f"peak|F| {np.linalg.norm(d['F'], axis=1).max():.0f} N  levels {np.unique(d['level']).tolist()}")


if __name__ == "__main__":
    main()
