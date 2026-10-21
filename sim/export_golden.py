"""Export deterministic browser golden cases from the Python reference model.

This script deliberately imports the simulation package but does not modify it.
The JSON is a compact, fully specified contract for the browser port.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sim.catchpoint.controllers as controllers
from sim.catchpoint.controllers import B0, B1, B1XCoM, B2, CP
from sim.catchpoint.model import Config, build_pop, simulate


OUT = ROOT / "web" / "golden.json"


# The checked-in controller diagnostic names the valid-candidate count nfeas,
# while the current reference model exposes the same count as nvalid.  Keep the
# package files read-only and bridge that diagnostic-only naming mismatch here.
_candidate_barrier = controllers.candidate_barrier


def _candidate_barrier_compat(*args, **kwargs):
    result = _candidate_barrier(*args, **kwargs)
    if "nfeas" not in result:
        result["nfeas"] = result["nvalid"]
    return result


controllers.candidate_barrier = _candidate_barrier_compat


def _patient(sev: float, par: int, mass: float, z0: float):
    # Keep the explicit patient parameters in the file so this is not dependent
    # on the browser's slider defaults.
    pop = build_pop([sev], [par], [mass], [z0], [[0.0, 0.0]])
    return {
        "severity": float(sev),
        "paretic": "L" if par == 0 else "R",
        "mass": float(mass),
        "z0": float(z0),
        "tau_r": float(pop["tau_r"][0]),
        "Lmax": [float(x) for x in pop["Lmax"][0]],
        "Ts": [float(x) for x in pop["Ts"][0]],
        "scale": [float(x) for x in pop["scale"][0]],
    }


def main():
    # Five deterministic controller families, deliberately covering mild/severe patients,
    # both paretic sides, forward/lateral/diagonal pushes, and the controller
    # thresholds used by the page.  CP identification is exact by construction.
    specs = [
        ("B0", {"id": "B0"}),
        ("B1", {"id": "B1", "threshold": 0.08}),
        ("B1-XCoM", {"id": "B1-XCoM", "threshold": 0.08}),
        ("B2", {"id": "B2", "k": 800, "threshold": 0.14}),
        ("CP", {"id": "CP", "margin": 0.03, "dc": True}),
    ]
    trials = [
        (0.00, 0, 72.0, 0.92, 0.20, 0.0),
        (0.08, 1, 66.0, 0.87, 0.28, np.pi / 2),
        (0.16, 0, 80.0, 0.99, 0.36, np.pi),
        (0.24, 1, 64.0, 0.90, 0.45, 3 * np.pi / 4),
        (0.32, 0, 74.0, 0.95, 0.55, 5 * np.pi / 3),
        (0.40, 1, 82.0, 0.86, 0.64, np.pi / 6),
        (0.48, 0, 69.0, 0.93, 0.74, 7 * np.pi / 8),
        (0.56, 1, 77.0, 0.98, 0.83, 11 * np.pi / 6),
        (0.64, 0, 61.0, 0.89, 0.90, np.pi / 2),
        (0.72, 1, 71.0, 0.91, 0.49, 2 * np.pi / 3),
        (0.80, 0, 83.0, 0.97, 0.31, 5 * np.pi / 6),
        (0.88, 1, 68.0, 0.85, 0.69, 4 * np.pi / 3),
        (0.93, 0, 79.0, 0.94, 0.77, np.pi / 4),
        (0.97, 1, 73.0, 0.88, 0.58, 3 * np.pi / 2),
        (1.00, 0, 85.0, 1.00, 0.20, 0.0),
    ]

    cfg = Config()
    selected = {
        "B0": list(range(15)),
        # These deterministic cases span the threshold-lock sweep; repeat a few
        # directions where needed so every controller block has exactly 15 cases.
        "B1": [0, 1, 2, 5, 6, 7, 0, 1, 2, 5, 6, 7, 0, 1, 2],
        "B1-XCoM": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "B2": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 0, 1],
        "CP": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 0, 1],
    }
    cases = []
    for ctrl_name, ctrl_settings in specs:
        for j, trial_index in enumerate(selected[ctrl_name]):
            sev, par, mass, z0, mag, ang = trials[trial_index]
            dv = [float(mag * np.cos(ang)), float(mag * np.sin(ang))]
            patient = _patient(sev, par, mass, z0)
            pop = build_pop([sev], [par], [mass], [z0], [dv])
            if ctrl_name == "B0":
                ctrl = B0()
            elif ctrl_name == "B1":
                ctrl = B1(ctrl_settings["threshold"])
            elif ctrl_name == "B1-XCoM":
                ctrl = B1XCoM(ctrl_settings["threshold"])
            elif ctrl_name == "B2":
                ctrl = B2(ctrl_settings["k"], ctrl_settings["threshold"])
            else:
                ctrl = CP(ctrl_settings["margin"])
            result = simulate(pop, ctrl, cfg)
            expected = {
                "outcome": ["SELF", "ASSISTED", "CAUGHT", "FALL"][int(result["outcome"][0])],
                "n_steps": int(result["n_steps"][0]),
                "impulse": float(result["impulse"][0]),
                "peak_force": float(result["peakF"][0]),
            }
            cases.append({
                "id": f"{ctrl_name}-{j + 1:02d}",
                "controller": ctrl_settings,
                "patient": patient,
                "push": dv,
                "identified": {
                    "mass": patient["mass"], "omega": float(pop["omega"][0]),
                    "tau_r": patient["tau_r"], "Lmax": patient["Lmax"],
                    "Ts": patient["Ts"], "scale": patient["scale"],
                },
                "expected": expected,
            })

    OUT.write_text(json.dumps({"version": 1, "dt": cfg.dt, "trial_length": cfg.T,
                               "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
