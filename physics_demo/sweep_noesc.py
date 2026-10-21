import numpy as np
from concurrent.futures import ProcessPoolExecutor
from g1cp import Sim
OM = np.sqrt(9.81/0.685)
def run(a):
    r0, TH, fs, F, t0 = a
    s = Sim(ctrl="CP", T=5.0, weak=0.5, delay=0.06, R_cap=r0*np.exp(OM*TH), T_H=TH, escalate=False, f_assist_scale=fs, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)))
    r = s.run()
    return r0, TH, fs, F, t0, r["outcome"][:6], r["peakF_h"]
if __name__ == "__main__":
    cases = [(140, 2.2), (140, 2.0), (140, 2.6), (180, 2.1), (180, 2.4), (220, 2.2), (300, 2.2)]
    jobs = [(r0, 0.4, fs, F, t0) for r0 in [0.08, 0.2] for fs in [1.0, 2.0] for F, t0 in cases]
    with ProcessPoolExecutor(15) as ex: out = list(ex.map(run, jobs))
    for o in out: print("r0=%.2f TH=%.1f fs=%.0f F=%d t0=%.1f %s peak_h=%.0f" % o)
