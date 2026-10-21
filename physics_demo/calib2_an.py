import numpy as np, pickle, sys
sys.path.insert(0, "..")
from sim.catchpoint import model as cpm
from g1cp import HS_G1
res = pickle.load(open("../_tmp/calib2.pkl", "rb"))
rows = []
for F, t0, out, ft, L in res:
    h = HS_G1[None].repeat(len(L["t"]), 0)
    hull = cpm.make_hull(L["fl"][:, :2], h, L["fr"][:, :2], h)
    d0 = cpm.dist_to(L["xi"], hull)
    t = L["t"]
    end = (ft - 0.3) if ft else t0 + 2.0
    m = (t > 1.0) & (t < end)
    q = (t > 1.0) & (t < t0)
    rows.append((F, t0, out, ft, d0[q].max(), d0[m].max(), d0, t))
    print("F=%3d t0=%.1f %-16s fall=%5s quiet=%.3f post=%.3f" % (F, t0, out, ft and round(ft,2), d0[q].max(), d0[m].max()))
for thr in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
    rec = [r for r in rows if r[2] != "fall"]; fal = [r for r in rows if r[2] == "fall"]
    fp = np.mean([r[5] > thr for r in rec]); tp = np.mean([r[5] > thr for r in fal])
    lead = [r[3] - r[7][np.argmax((r[6] > thr) & (r[7] > 1.0))] for r in fal if (r[6][r[7]>1.0] > thr).any()]
    print("thr %.2f: recovered flagged %.2f  falls flagged %.2f  lead time median %.2f min %.2f" % (thr, fp, tp, np.median(lead), np.min(lead)))
