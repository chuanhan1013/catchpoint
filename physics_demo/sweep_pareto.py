"""Setting sweep (like SPEC F1): CP recoverable radius r0 and B1 threshold theta over the same 56 pushes as sweep_all.py.
Writes sweep_pareto.json; prints fall / hands-off / nudged / caught counts per setting."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from g1cp import Sim, G
from sweep_all import FS, T0S
OM = np.sqrt(G / 0.685)
SETTINGS = [("B1", 0.11), ("B1", 0.14), ("B1", 0.18), ("CP", 0.10), ("CP", 0.20), ("CP", 0.28)]
def run(a):
    F, t0, ctrl, par = a
    kw = dict(theta=par) if ctrl == "B1" else dict(R_cap=par * np.exp(OM * 0.4))
    s = Sim(ctrl=ctrl, T=6.0, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)), **kw)
    r = s.run()
    return dict(F=F, t0=t0, ctrl=ctrl, par=par, outcome=r["outcome"], impulse_h=round(r["impulse"] / s.lam, 1))
if __name__ == "__main__":
    jobs = [(F, t0, c, p) for F in FS for t0 in T0S for c, p in SETTINGS]
    with ProcessPoolExecutor(15) as ex: rows = list(ex.map(run, jobs, chunksize=2))
    json.dump(rows, open("sweep_pareto.json", "w"), indent=0)
    for c, p in SETTINGS:
        cnt = Counter(r["outcome"] for r in rows if r["ctrl"] == c and r["par"] == p)
        print(c, p, dict(cnt))
