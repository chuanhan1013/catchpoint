import numpy as np
from concurrent.futures import ProcessPoolExecutor
from g1cp import Sim
OM = np.sqrt(9.81/0.685)
def run(a):
    ctrl, r0, F, t0 = a
    s = Sim(ctrl=ctrl, T=6.0, weak=0.5, delay=0.06, theta=0.08, R_cap=r0*np.exp(OM*0.4), T_H=0.4, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)))
    r = s.run()
    return ctrl, r0, F, t0, r["outcome"], r["fall_t"], r["catch_t"], r["peakF_h"], r["impulse"]/s.lam
if __name__ == "__main__":
    jobs = [("B0", 0, F, t0) for F in (100, 180, 300) for t0 in (2.3, 2.4)] + [("B1", 0, F, t0) for F in (100, 180, 300) for t0 in (2.3, 2.4)] + \
           [("CP", r0, F, t0) for r0 in (0.08, 0.14, 0.2) for F in (100, 180, 300) for t0 in (2.3, 2.4)]
    with ProcessPoolExecutor(15) as ex:
        for o in ex.map(run, jobs): print("%s r0=%.2f F=%d t0=%.1f %-16s fall=%s catch=%s peak_h=%.0f imp_h=%.1f" % o)
