import numpy as np, pickle
from concurrent.futures import ProcessPoolExecutor
from g1cp import Sim, HS_G1
import g1cp
from sim.catchpoint import model as cpm
def run(a):
    F, t0 = a
    s = Sim(ctrl="CPmon", T=6.0, weak=0.5, delay=0.06, push=dict(t0=t0, dur=0.1, F=F, dir=(0,-1)))
    r = s.run(); L = r["log"]
    return F, t0, r["outcome"], r["fall_t"], {k: L[k] for k in ("t","xi","fl","fr","cl","cr","com","comv")}
if __name__ == "__main__":
    jobs = [(F, t0) for F in [0, 60, 100, 140, 180, 220, 260, 300] for t0 in [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]]
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(run, jobs))
    pickle.dump(res, open("../_tmp/calib2.pkl", "wb"))
