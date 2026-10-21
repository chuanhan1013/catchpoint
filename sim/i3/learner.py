"""I3 surrogate learner: low-dimensional skill policy trained by a limited-step (1+candidates) natural-ES-like
black-box search, evaluated with the skill-parameterised engine (i3.skill_sim).

Policy theta = (eps, kc, g, rho) is stored normalised z in [0,1]^4 and mapped linearly to LO..HI.
Everything is vectorised over G independent learners ("groups": severity x seed x lambda).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from catchpoint.model import FALL, SELF, build_pop, sample_noise
from i3.skill_sim import simulate_skill

NAMES = ["eps", "kc", "g", "rho"]
LO = np.array([0.0, 0.0, 0.5, 0.0])
HI = np.array([0.20, 1.0, 1.5, 1.0])
THETA0 = np.array([0.08, 0.20, 0.75, 0.0])        # poor/unskilled: late trigger, weak ankle, undershoot placement
SEVS = {"mild": 0.2, "moderate": 0.5, "severe": 0.8}
PUSH_LO, PUSH_HI = 0.2, 0.6

# search hyper-parameters (motor learning is slow: limited step per session)
N_CAND = 5            # centre + 2 antithetic pairs
PUSH_PER_CAND = 4     # 5 x 4 = 20 pushes per session, shared across candidates (paired within session)
N_SESS = 12
SIGMA = 0.08          # exploration std in normalised units
LR = 0.30
STEP_MAX = 0.10       # max normalised step per session
W_ANKLE = 2.0         # effort = sum(step length [m]) + W_ANKLE * ankle CoP-excursion integral [m s]


def to_theta(z):
    return LO + np.clip(z, 0, 1) * (HI - LO)


def to_z(theta):
    return (np.asarray(theta) - LO) / (HI - LO)


def skill_dict(z):
    th = to_theta(z)
    return {n: th[:, i].copy() for i, n in enumerate(NAMES)}


def make_groups(n_seed, seed0=0, sevs=SEVS):
    """One patient per (severity, seed): par, m, z0, identification noise; push stream seed per group."""
    G = []
    for name, s in sevs.items():
        for k in range(n_seed):
            rng = np.random.default_rng([5412, 3, seed0 + k, int(s * 100)])
            G.append(dict(sev=s, name=name, seed=k, par=int(rng.integers(0, 2)), m=rng.uniform(60, 85),
                          z0=rng.uniform(0.85, 1.0), noise=sample_noise(rng, 1), rng_seed=int(rng.integers(1 << 30))))
    return G


def pushes(rng, shape):
    mag = rng.uniform(PUSH_LO, PUSH_HI, shape)
    ang = rng.uniform(0, 2 * np.pi, shape)
    return np.stack([mag * np.cos(ang), mag * np.sin(ang)], -1)


def build(groups, gidx, dv, skill):
    """Trial arrays: gidx (T,) group index per trial, dv (T,2), skill dict of (T,) arrays."""
    gidx = np.asarray(gidx)
    pick = lambda key: np.array([g[key] for g in groups])[gidx]
    noise = {k: np.concatenate([g["noise"][k] for g in groups])[gidx] for k in groups[0]["noise"]}
    pop = build_pop(pick("sev"), pick("par"), pick("m"), pick("z0"), dv, noise=noise)
    pop["skill"] = skill
    return pop


def effort_of(res):
    return res["step_len"] + W_ANKLE * res["ankle_int"]


def rewards(res, lam):
    """+1 no fall (CAUGHT counts as not fallen), -1 fall, minus lam * effort."""
    return np.where(res["outcome"] == FALL, -1.0, 1.0) - lam * effort_of(res)


def centered_rank(f):
    r = np.apply_along_axis(rankdata, 1, f)          # average ranks for ties
    return (r - (f.shape[1] + 1) / 2) / f.shape[1]


def train(groups, lams, ctrl_factory, n_sess=N_SESS, theta0=THETA0, seed=0, probe_n=16, probe_every=3, cfg=None, log=None):
    """Train one learner per (group, lambda) under one assistance condition.
    Returns dict with z history, per-session train falls/reward/effort, B0 probe SELF curve."""
    from catchpoint.controllers import B0
    G, L = len(groups), len(lams)
    GL = G * L
    gid = np.repeat(np.arange(G), L)                  # learner -> group
    lam = np.tile(np.asarray(lams, float), G)         # learner -> lambda
    z = np.tile(to_z(theta0), (GL, 1))
    # identical push streams per GROUP (paired across conditions and lambdas)
    rngs = [np.random.default_rng([groups[g]["rng_seed"], 1]) for g in range(G)]
    prng = [np.random.default_rng([groups[g]["rng_seed"], 2]) for g in range(G)]
    probe_dv = np.stack([pushes(prng[g], (probe_n,)) for g in range(G)])          # (G,probe_n,2)
    H = dict(z=[z.copy()], falls=[], reward=[], effort=[], probe=[])
    H["probe"].append(probe_self(groups, gid, z, probe_dv, cfg))
    for s in range(n_sess):
        dv_s = np.stack([pushes(rngs[g], (PUSH_PER_CAND,)) for g in range(G)])    # (G,4,2) shared by candidates
        rng_e = np.random.default_rng([seed, s, 17])
        eps = rng_e.standard_normal((G, 2, 4))        # per GROUP: same exploration noise across lambda/conditions
        E = np.concatenate([np.zeros((G, 1, 4)), eps, -eps], 1)                    # (G,5,4)
        E = E[gid]                                    # (GL,5,4)
        zc = np.clip(z[:, None, :] + SIGMA * E, 0, 1)
        sk = {n: to_theta(zc.reshape(-1, 4))[:, i] for i, n in enumerate(NAMES)}
        gt = np.repeat(gid, N_CAND * PUSH_PER_CAND)
        dvt = np.repeat(dv_s[gid][:, None], N_CAND, 1).reshape(-1, 2) if False else \
            np.broadcast_to(dv_s[gid][:, None], (GL, N_CAND, PUSH_PER_CAND, 2)).reshape(-1, 2)
        skt = {n: np.repeat(v, PUSH_PER_CAND) for n, v in sk.items()}
        pop = build(groups, gt, dvt, skt)
        res = simulate_skill(pop, ctrl_factory(), cfg)
        r = rewards(res, np.repeat(lam, N_CAND * PUSH_PER_CAND))
        f = r.reshape(GL, N_CAND, PUSH_PER_CAND).mean(2)
        w = centered_rank(f)
        d = LR * (w[:, :, None] * E).sum(1)
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        d = d * np.minimum(1.0, STEP_MAX / np.maximum(nrm, 1e-12))
        z = np.clip(z + d, 0, 1)
        o = res["outcome"].reshape(GL, -1)
        H["falls"].append((o == FALL).sum(1))
        H["reward"].append(r.reshape(GL, -1).mean(1))
        H["effort"].append(effort_of(res).reshape(GL, -1).mean(1))
        H["z"].append(z.copy())
        if (s + 1) % probe_every == 0:
            H["probe"].append(probe_self(groups, gid, z, probe_dv, cfg))
    out = {k: np.array(v) for k, v in H.items()}
    out.update(gid=gid, lam=lam)
    return out


def probe_self(groups, gid, z, probe_dv, cfg=None):
    from catchpoint.controllers import B0
    GL, n = len(gid), probe_dv.shape[1]
    sk = {k: np.repeat(v, n) for k, v in skill_dict(z).items()}
    pop = build(groups, np.repeat(gid, n), probe_dv[gid].reshape(-1, 2), sk)
    return (simulate_skill(pop, B0(), cfg)["outcome"].reshape(GL, n) == SELF).mean(1)


def retention(groups, gid, z, n_push=200, cfg=None, chunk=2000):
    """No-robot retention: n_push fresh pushes (same stream per group), deterministic policy z."""
    from catchpoint.controllers import B0
    GL = len(gid)
    dv = np.stack([pushes(np.random.default_rng([groups[g]["rng_seed"], 99]), (n_push,)) for g in range(len(groups))])
    outs = []
    sk_all = skill_dict(z)
    T = GL * n_push
    gt = np.repeat(gid, n_push)
    dvt = dv[gid].reshape(-1, 2)
    skt = {k: np.repeat(v, n_push) for k, v in sk_all.items()}
    out = np.empty(T, int)
    for a in range(0, T, chunk):
        sl = slice(a, min(a + chunk, T))
        pop = build(groups, gt[sl], dvt[sl], {k: v[sl] for k, v in skt.items()})
        out[sl] = simulate_skill(pop, B0(), cfg)["outcome"]
    out = out.reshape(GL, n_push)
    return (out == SELF).mean(1), (out == FALL).mean(1)
