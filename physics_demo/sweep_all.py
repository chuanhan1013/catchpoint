"""Robustness sweep: B0/B1/CP over push magnitude x push timing (gait phase), impaired G1, lateral push to the weak side.
Writes sweep_all.json and prints outcome counts per controller and per magnitude."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from g1cp import Sim
FS = [60, 100, 140, 180, 220, 260, 300]
T0S = [round(2.0 + 0.1 * i, 1) for i in range(8)]
def run(a):
    F, t0, ctrl = a
    s = Sim(ctrl=ctrl, T=6.0, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)))
    r = s.run()
    return dict(F=F, t0=t0, ctrl=ctrl, outcome=r["outcome"], peakFxy_h=round(r["peakFxy"] / s.lam, 1),
                impulse_h=round(r["impulse"] / s.lam, 1))
if __name__ == "__main__":
    jobs = [(F, t0, c) for F in FS for t0 in T0S for c in ("B0", "B1", "CP")]
    with ProcessPoolExecutor(15) as ex: rows = list(ex.map(run, jobs, chunksize=2))
    json.dump(rows, open("sweep_all.json", "w"), indent=0)
    b0 = {(r["F"], r["t0"]): r["outcome"] for r in rows if r["ctrl"] == "B0"}
    for c in ("B0", "B1", "CP"):
        rr = [r for r in rows if r["ctrl"] == c]
        cnt = Counter(r["outcome"] for r in rr)
        unn = sum(1 for r in rr if r["outcome"] in ("caught", "nudged") and b0[(r["F"], r["t0"])] == "recovered alone")
        unc = sum(1 for r in rr if r["outcome"] == "caught" and b0[(r["F"], r["t0"])] == "recovered alone")
        print(c, dict(cnt), "| intervened although B0 recovered alone:", unn, "(caught:", unc, ")")
        for F in FS:
            print("   F=%3d " % F + " ".join("%-8s" % r["outcome"][:8] for r in rr if r["F"] == F))
