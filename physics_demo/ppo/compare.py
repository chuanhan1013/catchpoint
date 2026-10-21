"""Before/after fine-tuning: 56-push grid (same as sweep_all.py) for B0 and CP with the pretrained vs fine-tuned policy.
usage: python ppo/compare.py <policy_ft.pt>  -> ppo/compare.json + printed table"""
import json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from g1cp import Sim
from sweep_all import FS, T0S
def run(a):
    F, t0, ctrl, pol = a
    s = Sim(ctrl=ctrl, T=6.0, push=dict(t0=t0, dur=0.1, F=F, dir=(0, -1)), policy=pol)
    r = s.run()
    return dict(F=F, t0=t0, ctrl=ctrl, policy="ft" if pol else "pre", outcome=r["outcome"],
                impulse_h=round(r["impulse"] / s.lam, 1))
if __name__ == "__main__":
    ft = os.path.abspath(sys.argv[1])
    jobs = [(F, t0, c, p) for F in FS for t0 in T0S for c in ("B0", "CP") for p in (None, ft)]
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 15) as ex: rows = list(ex.map(run, jobs, chunksize=2))
    json.dump(dict(ft=ft, rows=rows), open(os.path.join(HERE, "compare.json"), "w"), indent=0)
    for c in ("B0", "CP"):
        for p in ("pre", "ft"):
            rr = [r for r in rows if r["ctrl"] == c and r["policy"] == p]
            print(c, p, dict(Counter(r["outcome"] for r in rr)))
            for F in FS:
                print("   F=%3d " % F + " ".join("%-8s" % r["outcome"][:8] for r in rr if r["F"] == F))
