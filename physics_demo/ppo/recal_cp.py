"""Re-identify CP's recoverable radius r0 for the fine-tuned 'patient' (same 56 pushes).
usage: python ppo/recal_cp.py <policy.pt> [procs] -> ppo/recal_cp.json"""
import json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from g1cp import Sim, G
from sweep_all import FS, T0S
OM = np.sqrt(G / 0.685)
R0S = [0.14, 0.20, 0.28, 0.36]
def run(a):
    F, t0, r0, pol = a
    kw = {} if r0 is None else dict(R_cap=r0 * np.exp(OM * 0.4), R_cap_n=max(0.25, r0 + 0.11) * np.exp(OM * 0.4))
    s = Sim(ctrl="B0" if r0 is None else "CP", T=6.0, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)), policy=pol, **kw)
    r = s.run()
    return dict(F=F, t0=t0, r0=r0, outcome=r["outcome"], impulse_h=round(r["impulse"] / s.lam, 1))
if __name__ == "__main__":
    pol = os.path.abspath(sys.argv[1])
    jobs = [(F, t0, r0, pol) for F in FS for t0 in T0S for r0 in [None] + R0S]
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 15) as ex: rows = list(ex.map(run, jobs, chunksize=2))
    json.dump(dict(policy=pol, rows=rows), open(os.path.join(HERE, "recal_cp.json"), "w"), indent=0)
    b0 = {(r["F"], r["t0"]): r["outcome"] for r in rows if r["r0"] is None}
    print("B0", dict(Counter(b0.values())))
    for r0 in R0S:
        rr = [r for r in rows if r["r0"] == r0]
        un = Counter(r["outcome"] for r in rr if b0[(r["F"], r["t0"])] == "recovered alone")
        fa = Counter(r["outcome"] for r in rr if b0[(r["F"], r["t0"])] == "fall")
        print("CP r0=%.2f" % r0, dict(Counter(r["outcome"] for r in rr)), "| on B0-alone:", dict(un), "| on B0-falls:", dict(fa))
