import numpy as np, pickle, sys, itertools
sys.path.insert(0, "..")
import g1cp
from sim.catchpoint import model as cpm
from concurrent.futures import ProcessPoolExecutor
res = pickle.load(open("../_tmp/calib2.pkl", "rb"))
def req_trial(args):
    delta, wf, ri = args
    g1cp.HS_G1 = np.array([0.09 + delta, 0.035 + delta])
    om = wf * np.sqrt(9.81/0.685)
    F, t0, out, ft, L = res[ri]
    t = L["t"]; idx = np.nonzero((t > t0) & (t < min(((ft - 0.3) if ft else 9), t0 + 1.2)))[0][::5]
    best = 0
    for i in idx:
        fc = np.stack([L["fl"][i], L["fr"][i]])
        ds = []
        for j in (0, 1):
            xl, T = g1cp.landing_xi(L["xi"][i], fc, t[i] + 0.1, j, om)
            dv = xl - fc[1 - j, :2]
            ds.append(np.inf if cpm.SIDE[j] * dv[1] < 0.06 else np.hypot(*dv))
        best = max(best, min(ds))
    return delta, wf, ri, best, out
if __name__ == "__main__":
    grid = list(itertools.product([0.0, 0.04, 0.08, 0.12], [1.0, 0.75, 0.6, 0.5]))
    jobs = [(d, w, i) for d, w in grid for i in range(len(res)) if res[i][0] > 0]
    with ProcessPoolExecutor(15) as ex: out = list(ex.map(req_trial, jobs, chunksize=4))
    pickle.dump(out, open("../_tmp/calib4.pkl", "wb"))
    for d, w in grid:
        r = [(b, o) for (dd, ww, i, b, o) in out if dd == d and ww == w]
        rec = np.array([b for b, o in r if o != "fall"]); fal = np.array([b for b, o in r if o == "fall"])
        auc = np.mean([[1.0 if f > q else 0.5 if f == q else 0 for q in rec] for f in fal])
        # best threshold accuracy
        ths = np.unique(np.concatenate([rec, fal])); ths = ths[np.isfinite(ths)]
        acc = max(((rec <= th).sum() + (fal > th).sum()) / (len(rec)+len(fal)) for th in ths)
        thb = max(ths, key=lambda th: ((rec <= th).sum() + (fal > th).sum()))
        print("delta=%.2f wf=%.2f AUC=%.2f acc=%.2f thr=%.3f rec_med=%.2f fall_med=%.2f" % (d, w, auc, acc, thb, np.median(rec), np.median(fal)))
