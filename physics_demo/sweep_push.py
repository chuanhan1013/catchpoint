import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from g1cp import Sim
def run(args):
    weak, delay, F, t0 = args
    s = Sim(ctrl="B0", T=6.0, weak=weak, delay=delay, push=dict(t0=t0, dur=0.1, F=F, dir=(0,-1)))
    r = s.run(); L = r["log"]; t = L["t"]; m = t > t0
    dl = L["fr"][m, :2] - L["fl"][m, :2]
    return (weak, delay, F, t0, r["outcome"], r["fall_t"], np.hypot(*dl.T).max(), (-dl[:,1]).max())
if __name__ == "__main__":
    jobs = [(w, dly, F, t0) for (w, dly) in [(1.0, 0.0), (0.5, 0.06)] for F in [60, 100, 140, 180, 220, 260, 300] for t0 in [2.0, 2.2, 2.4, 2.6]]
    with ProcessPoolExecutor(14) as ex:
        for r in ex.map(run, jobs):
            print("w=%.1f d=%.2f F=%d t0=%.1f %-16s fall=%s maxsep=%.3f maxwidth=%.3f" % r)
