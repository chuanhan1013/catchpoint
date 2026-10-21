"""Run pushes (a, b, c) x controllers (B0, B1, CP) without rendering; print the outcome table."""
import json, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from g1cp import Sim
PUSHES = {"a": 100, "b": 180, "c": 300}


def count_steps(L, m, off_min=0.05, move_min=0.05):
    """Steps after the push: touchdowns after >= off_min s of no contact where the foot moved > move_min."""
    n = 0
    for c_key, f_key in (("cl", "fl"), ("cr", "fr")):
        c = L[c_key][m].astype(bool); f = L[f_key][m][:, :2]
        off_since, lift_pos = None, None
        for i in range(1, len(c)):
            if c[i - 1] and not c[i]:
                off_since, lift_pos = i, f[i]
            if (not c[i - 1]) and c[i] and off_since is not None:
                if (i - off_since) * 0.002 >= off_min and np.hypot(*(f[i] - lift_pos)) > move_min:
                    n += 1
                off_since = None
    return n
T0, DUR = 2.4, 0.1
def run(a):
    pk, ctrl = a
    s = Sim(ctrl=ctrl, T=7.0, push=dict(t0=T0, dur=DUR, F=PUSHES[pk], dir=(0, -1)))
    r = s.run(); L = r["log"]
    t = L["t"]; m = t > T0
    td = count_steps(L, m)
    return dict(push=pk, F=PUSHES[pk], dv=PUSHES[pk]*DUR/s.mass, ctrl=ctrl, outcome=r["outcome"], fall_t=r["fall_t"],
                catch_t=r["catch_t"], peakF=round(r["peakF"], 1), peakF_h=round(r["peakF_h"], 1),
                impulse=round(r["impulse"], 2), impulse_h=round(r["impulse"] / s.lam, 1), peakFxy=round(r["peakFxy"], 1), peakFxy_h=round(r["peakFxy"] / s.lam, 1), steps=td, lam=s.lam, mass=s.mass)
if __name__ == "__main__":
    jobs = [(p, c) for p in PUSHES for c in ("B0", "B1", "CP")]
    with ProcessPoolExecutor(9) as ex: rows = list(ex.map(run, jobs))
    for r in rows: print(r)
    json.dump(rows, open("outcomes.json", "w"), indent=1)
