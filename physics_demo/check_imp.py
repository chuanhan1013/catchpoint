import sys, numpy as np
from g1cp import Sim
for weak in [1.0, 0.5, 0.4]:
    for delay in [0.06]:
        s = Sim(ctrl="B0", T=10.0, weak=weak, delay=delay); r = s.run(); L=r["log"]
        c = (L["fl"][:, :2]+L["fr"][:, :2])/2; dev = np.hypot(*(L["com"][:, :2]-c).T)
        print(weak, delay, r["outcome"], r["fall_t"], "tilt %.1f"%L["tilt"].max(), "dev %.3f"%dev.max(), "drift", L["com"][-1,:2].round(2), "yaw?")
