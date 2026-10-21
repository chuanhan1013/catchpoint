"""Recurrent PPO fine-tuning of the pretrained G1 LSTM policy for push recovery under the simulated impairment.

usage: python ppo/train.py --iters 600 --workers 14 --envs 4 --out ppo/runs/ft1
Writes <out>/log.csv, <out>/actor_<it>.pt (state_dict) and <out>/policy_<it>.pt (TorchScript, motion.pt interface)."""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from net import Actor, export_script, load_pretrained  # noqa: E402


def worker(conn, seeds):
    torch.set_num_threads(1)
    from env import PushEnv
    envs = [PushEnv(seed=s) for s in seeds]
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            out = [e.reset() for e in envs]
            conn.send((np.stack([o[0] for o in out]), np.stack([o[1] for o in out])))
        elif cmd == "step":
            A, C, R, D, F, TO, TC = [], [], [], [], [], [], []
            for e, a in zip(envs, data):
                ao, co, r, d, info = e.step(a)
                tc = co
                if d:
                    ao, co = e.reset()
                A.append(ao); C.append(co); R.append(r); D.append(d); F.append(info["fell"])
                TO.append(info["timeout"]); TC.append(tc)
            conn.send((np.stack(A), np.stack(C), np.array(R, np.float32), np.array(D), np.array(F), np.array(TO),
                       np.stack(TC)))
        elif cmd == "close":
            break


class VecEnv:
    def __init__(self, W, K, seed=0):
        self.ps, self.conns = [], []
        for w in range(W):
            a, b = mp.Pipe()
            p = mp.Process(target=worker, args=(b, [seed + 1000 * w + k for k in range(K)]), daemon=True)
            p.start()
            self.ps.append(p); self.conns.append(a)
        self.W, self.K = W, K

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        out = [c.recv() for c in self.conns]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def step(self, act):
        for i, c in enumerate(self.conns):
            c.send(("step", act[i * self.K:(i + 1) * self.K]))
        out = [c.recv() for c in self.conns]
        return [np.concatenate([o[j] for o in out]) for j in range(7)]

    def close(self):
        for c in self.conns:
            c.send(("close", None))


class Critic(nn.Module):
    def __init__(self, n=53):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_lstm(actor, obs, h0, dones_prev):
    """obs (T,B,47), h0 tuple, dones_prev (T,B) = done flag of the previous step (reset hidden before step t)."""
    h, c = h0
    mus = []
    for t in range(obs.shape[0]):
        m = (1.0 - dones_prev[t]).view(1, -1, 1)
        h, c = h * m, c * m
        mu, (h, c) = actor(obs[t:t + 1], (h, c))
        mus.append(mu[0])
    return torch.stack(mus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bc", type=float, default=0.5)
    ap.add_argument("--std0", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "ft1"))
    ap.add_argument("--resume", default=None, help="ckpt_*.pt to continue from (actor, log_std, critic)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)

    actor, _, _ = load_pretrained()
    orig, _, _ = load_pretrained()
    for p in orig.parameters():
        p.requires_grad_(False)
    critic = Critic()
    log_std = nn.Parameter(torch.full((12,), float(np.log(args.std0))))
    if args.resume:
        ck = torch.load(args.resume)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        log_std.data.copy_(ck["log_std"])
    opt_a = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=5e-4)

    venv = VecEnv(args.workers, args.envs)
    N, T = args.workers * args.envs, args.T
    aobs, cobs = venv.reset()
    hc = actor.zero_state(N)
    hco = orig.zero_state(N)
    done_prev = np.zeros(N, bool)
    gamma, lam, clip = 0.99, 0.95, 0.2
    ep_ret = np.zeros(N); ep_len = np.zeros(N)
    fin_ret, fin_len, fin_fell = [], [], []
    f = open(os.path.join(args.out, "log.csv"), "w", newline="")
    wcsv = csv.writer(f)
    wcsv.writerow(["it", "steps", "ret", "len", "fall_rate", "std", "v_loss", "bc", "kl", "sec"])
    t_start = time.time()
    for it in range(args.iters + 1):
        # -------------------------------------------------------------- rollout
        OB = torch.zeros(T, N, 47); CO = torch.zeros(T + 1, N, 53); AC = torch.zeros(T, N, 12)
        LP = torch.zeros(T, N); RW = torch.zeros(T, N); DN = torch.zeros(T, N); DP = torch.zeros(T, N)
        MO = torch.zeros(T, N, 12); TB = torch.zeros(T, N)
        h0 = (hc[0].clone(), hc[1].clone())
        with torch.no_grad():
            for t in range(T):
                m = torch.as_tensor(1.0 - done_prev, dtype=torch.float32).view(1, -1, 1)
                hc = (hc[0] * m, hc[1] * m)
                hco = (hco[0] * m, hco[1] * m)
                o = torch.as_tensor(aobs)
                mu, hc = actor(o[None], hc)
                muo, hco = orig(o[None], hco)
                mu, muo = mu[0], muo[0]
                std = log_std.exp()
                a = mu + std * torch.randn_like(mu)
                lp = (-0.5 * ((a - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                OB[t] = o; CO[t] = torch.as_tensor(cobs); AC[t] = a; LP[t] = lp; MO[t] = muo
                DP[t] = torch.as_tensor(done_prev, dtype=torch.float32)
                aobs, cobs, r, d, fell, to, tc = venv.step(a.numpy())
                if to.any():   # bootstrap value of the truncated state
                    TB[t] = torch.as_tensor(to, dtype=torch.float32) * critic(torch.as_tensor(tc))
                RW[t] = torch.as_tensor(r); DN[t] = torch.as_tensor(d, dtype=torch.float32)
                ep_ret += r; ep_len += 1
                for i in np.nonzero(d)[0]:
                    fin_ret.append(ep_ret[i]); fin_len.append(ep_len[i]); fin_fell.append(bool(fell[i]))
                    ep_ret[i] = 0; ep_len[i] = 0
                done_prev = d
            CO[T] = torch.as_tensor(cobs)
            V = critic(CO.view(-1, 53)).view(T + 1, N)
            adv = torch.zeros(T, N); last = torch.zeros(N)
            for t in reversed(range(T)):
                nd = 1.0 - DN[t]
                delta = RW[t] + gamma * TB[t] + gamma * V[t + 1] * nd - V[t]
                last = delta + gamma * lam * nd * last
                adv[t] = last
            ret = adv + V[:T]
        # -------------------------------------------------------------- update
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(N)
        vls, bcs, kls = [], [], []
        for ep in range(4):
            np.random.shuffle(idx)
            for mb in np.array_split(idx, 4):
                mbt = torch.as_tensor(mb)
                vpred = critic(CO[:T, mbt].reshape(-1, 53)).view(T, -1)
                v_loss = ((vpred - ret[:, mbt]) ** 2).mean()
                opt_c.zero_grad(); v_loss.backward(); opt_c.step()
                vls.append(v_loss.item())
                if it < args.warmup:
                    continue
                mu = run_lstm(actor, OB[:, mbt], (h0[0][:, mbt], h0[1][:, mbt]), DP[:, mbt])
                std = log_std.exp()
                lp = (-0.5 * ((AC[:, mbt] - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                ratio = torch.exp(lp - LP[:, mbt])
                A_ = advn[:, mbt]
                pg = -torch.min(ratio * A_, torch.clamp(ratio, 1 - clip, 1 + clip) * A_).mean()
                bc = ((mu - MO[:, mbt]) ** 2).mean()
                loss = pg + args.bc * bc
                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], 1.0)
                opt_a.step()
                with torch.no_grad():
                    log_std.clamp_(np.log(0.03), np.log(0.5))
                kls.append((LP[:, mbt] - lp).mean().item()); bcs.append(bc.item())
        # -------------------------------------------------------------- log
        if fin_ret:
            rr, ll, ff = np.mean(fin_ret[-200:]), np.mean(fin_len[-200:]), np.mean(fin_fell[-200:])
        else:
            rr = ll = ff = float("nan")
        row = [it, (it + 1) * T * N, round(rr, 2), round(ll, 1), round(ff, 3), round(log_std.exp().mean().item(), 3),
               round(np.mean(vls), 3), round(np.mean(bcs), 4) if bcs else 0, round(np.mean(kls), 4) if kls else 0,
               round(time.time() - t_start, 1)]
        wcsv.writerow(row); f.flush()
        if it % 10 == 0:
            print(row, flush=True)
        if it % 50 == 0 or it == args.iters:
            torch.save(dict(actor=actor.state_dict(), log_std=log_std.data, critic=critic.state_dict()),
                        os.path.join(args.out, f"ckpt_{it:04d}.pt"))
            export_script(actor, os.path.join(args.out, f"policy_{it:04d}.pt"))
    venv.close()


if __name__ == "__main__":
    main()
