"""Evaluate a policy (TorchScript-equivalent Actor) in PushEnv: mean episode length and fall rate."""
import sys, os, numpy as np, torch
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def run(a):
    ckpt, std, seed, kw = a
    torch.set_num_threads(1)
    from env import PushEnv
    from net import load_pretrained
    act, _, _ = load_pretrained()
    if ckpt:
        act.load_state_dict(torch.load(ckpt)["actor"])
    e = PushEnv(seed=seed, **kw); o, _ = e.reset(); hc = act.zero_state(1); R = 0; g = np.random.default_rng(seed)
    with torch.no_grad():
        while True:
            mu, hc = act(torch.as_tensor(o)[None, None], hc)
            a = mu[0, 0].numpy() + std * g.standard_normal(12)
            o, _, r, d, info = e.step(a); R += r
            if d: return e.k, info["fell"], R
if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
    for std in [0.0, 0.1, 0.25]:
        with ProcessPoolExecutor(15) as ex: out = list(ex.map(run, [(ckpt, std, s, {}) for s in range(30)]))
        L = np.array([o[0] for o in out]); F = np.array([o[1] for o in out])
        print("std", std, "mean len %.0f/500  fall rate %.2f  ret %.1f" % (L.mean(), F.mean(), np.mean([o[2] for o in out])))
