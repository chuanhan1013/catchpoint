import numpy as np, pickle
from concurrent.futures import ProcessPoolExecutor
from g1cp import Sim
OM = np.sqrt(9.81/0.685)
def run(a):
    F, t0 = a
    s = Sim(ctrl="CP", T=5.0, weak=0.5, delay=0.06, R_cap=0.08*np.exp(OM*0.4), T_H=0.4, escalate=False, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)))
    r = s.run(); L = r["log"]
    return F, t0, r["outcome"], r["fall_t"], L["t"], L["resid"]
if __name__ == "__main__":
    jobs = [(F, t0) for F in [100, 140, 180, 220, 260, 300] for t0 in [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]]
    with ProcessPoolExecutor(15) as ex: out = list(ex.map(run, jobs))
    pickle.dump(out, open("../_tmp/calib_esc.pkl", "wb"))
    for F, t0, o, ft, t, rs in out:
        m = (t > t0) & (t < (ft - 0.4 if ft else 9))
        # time when resid first exceeds thresholds
        def first(th):
            k = np.nonzero(m & (rs > th))[0]
            return t[k[0]] if len(k) else np.nan
        print("F=%d t0=%.1f %-16s fall=%s maxres=%.3f  t(res>0.05)=%.2f t(res>0.1)=%.2f t(res>0.2)=%.2f" % (F, t0, o, ft and round(ft,2), rs[m].max() if m.any() else np.nan, first(0.05), first(0.1), first(0.2)))
