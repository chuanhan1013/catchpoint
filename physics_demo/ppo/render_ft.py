"""Render the fine-tuned-policy trials (B0, and CP with the re-identified r0 = 0.36) for pushes a, b, c -> videos/raw/*_ft."""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
from render import render_trial
from g1cp import G
POL = os.path.join(HERE, "policy_ft.pt")
OM = np.sqrt(G / 0.685)
R0 = 0.36
KW = {"CP": dict(R_cap=R0 * np.exp(OM * 0.4), R_cap_n=(R0 + 0.11) * np.exp(OM * 0.4)), "B0": {}}
def job(a):
    pk, c = a
    return render_trial(pk, c, policy=POL, tag="_ft", simkw=KW[c])
if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(job, [(p, c) for p in "abc" for c in ("B0", "CP")]):
            print(r)
