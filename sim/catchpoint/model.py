"""3D-LIP human balance model, patient stepping, robot plant and the batch
simulation engine (SPEC sections 5.1-5.3, 5.5).

The engine is vectorised over trials: every state variable is an array with a
leading trial axis N.  A single trial is just N = 1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

G = 9.81
HS = np.array([0.11, 0.045])      # foot half-length (x), half-width (y)
EPS = 0.01                        # step trigger / dilation [m]
KC_ANKLE = 0.5
D_FALL = 0.25
SIDE = np.array([1.0, -1.0])      # leg 0 = left (+y), leg 1 = right (-y)
XOVER = 0.08                      # no-crossover clearance [m]
MAX_STEPS = 3
SELF, ASSISTED, CAUGHT, FALL = 0, 1, 2, 3
OUTCOME_NAMES = ["SELF", "ASSISTED", "CAUGHT", "FALL"]


@dataclass
class Config:
    dt: float = 0.002
    T: float = 3.0
    t_push: float = 0.3
    tau_s: float = 0.03       # sensing delay
    tau_a: float = 0.07       # actuation (pure) delay
    tau_lag: float = 0.02     # first-order force lag
    tau_lock: float = 0.28    # catch lock delay
    Kc: float = 4000.0
    Dc: float = 800.0
    impulse_thr: float = 1.0  # N s, SELF vs ASSISTED
    f_on: float = 0.5         # N, command magnitude counted as "intervening"


# --------------------------------------------------------------------------
# geometry: convex hull of two axis-aligned rectangles
# --------------------------------------------------------------------------
_S = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]], float)


class Hull:
    """Hull of two axis-aligned boxes = bounding box cut at the four corners
    by one line each (through the two extreme corners of the two boxes)."""
    __slots__ = ("lo", "hi", "n", "b", "nn")


def make_hull(c1, h1, c2, h2):
    h = Hull()
    h.lo = np.minimum(c1 - h1, c2 - h2)
    h.hi = np.maximum(c1 + h1, c2 + h2)
    A = c1[:, None, :] + _S * h1[:, None, :]
    B = c2[:, None, :] + _S * h2[:, None, :]
    d = np.abs(A - B)
    n = np.stack([_S[:, 0] * d[..., 1], _S[:, 1] * d[..., 0]], -1)
    h.n = n
    h.b = np.maximum((n * A).sum(-1), (n * B).sum(-1))
    h.nn = (n * n).sum(-1)
    return h


def _lines(lo, hi, n, b, nn):
    """All half-planes n.z <= b (unit normals): 4 bounding-box sides + 4 corner cuts."""
    M = len(lo)
    nb = np.zeros((M, 4, 2)); nb[:, 0, 0] = 1; nb[:, 1, 0] = -1; nb[:, 2, 1] = 1; nb[:, 3, 1] = -1
    bb = np.stack([hi[:, 0], -lo[:, 0], hi[:, 1], -lo[:, 1]], 1)
    sq = np.sqrt(nn)
    ok = sq > 1e-12
    nc = np.where(ok[..., None], n / np.where(ok, sq, 1.0)[..., None], 0.0)
    bc = np.where(ok, b / np.where(ok, sq, 1.0), 0.0)
    return np.concatenate([nb, nc], 1), np.concatenate([bb, bc], 1)


def _project_exact(p, lo, hi, n, b, nn):
    """Exact projection onto {bbox and 4 cuts}: best feasible candidate among
    line projections and line-pair intersections."""
    N_, B_ = _lines(lo, hi, n, b, nn)                       # (M,8,2), (M,8)
    M = len(p)
    pr = p[:, None, :] - ((N_ * p[:, None, :]).sum(-1) - B_)[..., None] * N_      # (M,8,2)
    # intersections of every cut (index 4..7) with every bbox line (0..3)
    n1 = N_[:, 4:, None, :]; n2 = N_[:, None, :4, :]
    b1 = B_[:, 4:, None]; b2 = B_[:, None, :4]
    det = n1[..., 0] * n2[..., 1] - n1[..., 1] * n2[..., 0]
    okd = np.abs(det) > 1e-12
    dd = np.where(okd, det, 1.0)
    ix = (b1 * n2[..., 1] - b2 * n1[..., 1]) / dd
    iy = (n1[..., 0] * b2 - n2[..., 0] * b1) / dd
    inter = np.stack([ix, iy], -1).reshape(M, 16, 2)
    inter = np.where(np.repeat(okd.reshape(M, 16), 2).reshape(M, 16, 2), inter, np.nan)
    corners = np.stack([np.stack([hi[:, 0], hi[:, 1]], -1), np.stack([lo[:, 0], hi[:, 1]], -1),
                        np.stack([hi[:, 0], lo[:, 1]], -1), np.stack([lo[:, 0], lo[:, 1]], -1)], 1)
    cand = np.concatenate([pr, inter, corners], 1)                                # (M,28,2)
    viol = np.einsum("mkd,mjd->mkj", cand, N_) - B_[:, None, :]                 # (M,28,8)
    feas = (viol.max(-1) <= 1e-9) & np.isfinite(cand).all(-1)
    d2 = ((cand - p[:, None, :]) ** 2).sum(-1)
    d2 = np.where(feas, d2, np.inf)
    j = np.argmin(d2, 1)
    best = cand[np.arange(M), j]
    return best, np.isfinite(d2.min(1))


def project(p, h):
    """Euclidean projection of points p (N,2) onto the hull (exact).
    Fast path: clip to the bounding box; exact solve only where a corner cut is active."""
    z = np.clip(p, h.lo, h.hi)
    viol = np.einsum("nkd,nd->nk", h.n, z) - h.b
    act = (viol > 1e-12).any(1)
    if act.any():
        ii = np.nonzero(act)[0]
        best, ok = _project_exact(p[ii], h.lo[ii], h.hi[ii], h.n[ii], h.b[ii], h.nn[ii])
        z[ii[ok]] = best[ok]
    return z


def dist_to(p, h):
    d = p - project(p, h)
    return np.hypot(d[:, 0], d[:, 1])


# --------------------------------------------------------------------------
# LIP
# --------------------------------------------------------------------------
def lip_step(x, v, p, F, m, om, dt):
    """Exact LIP step with p, F constant over dt.
    xddot = om^2 (x - p) + F/m ; xi = x + v/om."""
    om_ = om[:, None]
    q = p - F / (m * om ** 2)[:, None]
    xi = x + v / om_
    e, ei = np.exp(om_ * dt), np.exp(-om_ * dt)
    xi_n = q + (xi - q) * e
    x_n = x * ei + q * (1 - ei) + (xi - q) * (e - ei) / 2.0
    return x_n, om_ * (xi_n - x_n)


# --------------------------------------------------------------------------
# population helpers
# --------------------------------------------------------------------------
def build_pop(sev, par, m, z0, dv, noise=None):
    """Arrays of patients/trials.  par: 0 = left paretic, 1 = right paretic.
    noise: optional dict of multiplicative (1+eps) factors giving the
    IDENTIFIED model used by CatchPoint (true params x (1+eps))."""
    sev = np.atleast_1d(np.asarray(sev, float))
    N = len(sev)
    par = np.broadcast_to(np.asarray(par, int), (N,)).copy()
    m = np.broadcast_to(np.asarray(m, float), (N,)).copy()
    z0 = np.broadcast_to(np.asarray(z0, float), (N,)).copy()
    dv = np.broadcast_to(np.asarray(dv, float), (N, 2)).copy()
    isp = (np.arange(2)[None, :] == par[:, None])
    s = sev[:, None]
    pop = dict(
        sev=sev, par=par, m=m, z0=z0, omega=np.sqrt(G / z0), dv=dv,
        tau_r=0.20 + 0.30 * sev,
        Lmax=np.where(isp, 0.50 - 0.30 * s, 0.60 - 0.10 * s),
        Ts=np.where(isp, 0.40 + 0.25 * s, 0.35 + 0.10 * s),
        scale=np.where(isp, 1 - 0.7 * s, 1.0),
    )
    nz = noise or {}
    one = np.ones(N)
    pop["ident"] = dict(
        m=m * nz.get("m", one), omega=pop["omega"] * nz.get("omega", one),
        tau_r=pop["tau_r"] * nz.get("tau_r", one),
        Lmax=pop["Lmax"] * nz.get("Lmax", np.ones((N, 2))),
        Ts=pop["Ts"] * nz.get("Ts", np.ones((N, 2))),
        scale=np.where(isp, pop["scale"] * nz.get("scale", one)[:, None], 1.0),
    )
    ones = np.ones(N)
    pop["ident_pop"] = dict(           # population-mean model, no per-patient identification (CP-pop ablation)
        m=72.5 * ones, omega=np.sqrt(G / 0.925) * ones, tau_r=0.35 * ones,
        Lmax=np.full((N, 2), 0.45), Ts=np.full((N, 2), 0.4625), scale=np.full((N, 2), 0.825),
    )
    return pop


def sample_noise(rng, N):
    f = lambda *s: 1.0 + 0.10 * rng.standard_normal(s)
    return dict(m=f(N), omega=f(N), tau_r=f(N), Lmax=f(N, 2), Ts=f(N, 2), scale=f(N))


# --------------------------------------------------------------------------
# recoverability barrier (shared by CatchPoint and the diagnostics)
# --------------------------------------------------------------------------
def min_norm_force(land0, a, r, beta):
    """Closed-form QP: min |F|^2 s.t. r - |land0 + beta F - a| >= 0."""
    dv = land0 - a
    d = np.hypot(dv[..., 0], dv[..., 1])
    u = dv / np.maximum(d, 1e-9)[..., None]
    mag = np.maximum(d - r, 0.0) / beta
    return -mag[..., None] * u


def barrier_h(F, land0, a, r, beta):
    dv = land0 + np.asarray(beta)[..., None] * F - a
    return r - np.hypot(dv[..., 0], dv[..., 1])


XOVER_C = 0.09     # crossover: swing centre must end >= 0.09 m on the opposite side of the stance foot
CROSS_T = 1.25     # crossover swing time multiplier
CROSS_L = 0.70     # crossover reach multiplier


def candidate_barrier(xi, fc, sw_leg, sw_mode, sw_el, Tr, om, mass, L, Ts, sc, marg, floor=0.05):
    """Best step candidate (4 = 2 legs x {side-step, crossover}) per trial.
    Returns h0 (barrier at F=0), d, r, a (anchor), land0, beta, Tl, nvalid (crossover/geometry-valid),
    nreach (valid and within reach at F=0), anyvalid, F_star (min-norm).  If no candidate is valid the
    returned entries are those of candidate 0 and anyvalid is False (callers must handle it)."""
    N = len(xi)
    swinging = sw_leg >= 0
    H = np.empty((N, 4)); LAND = np.empty((N, 4, 2)); A = np.empty((N, 4, 2))
    R = np.empty((N, 4)); B = np.empty((N, 4)); SC = np.empty((N, 4)); TL = np.empty((N, 4))
    FE = np.zeros((N, 4), bool); RE = np.zeros((N, 4), bool)
    for mode in (0, 1):
        for j in (0, 1):
            c = 2 * mode + j
            cs = fc[:, 1 - j]
            hh = HS * sc[:, 1 - j, None]
            pa = np.clip(xi, cs - hh, cs + hh)
            Tsc = Ts[:, j] * (CROSS_T if mode else 1.0)
            Lc = L[:, j] * (CROSS_L if mode else 1.0)
            Tl = np.where(swinging, np.maximum(Tsc - sw_el, floor), Tr + Tsc)
            e = np.exp(om * Tl)
            land0 = pa + (xi - pa) * e[:, None]
            dv = land0 - cs
            d = np.hypot(dv[:, 0], dv[:, 1])
            r = Lc - marg
            sd = SIDE[j] * dv[:, 1]
            crossok = (sd >= XOVER) if mode == 0 else (sd <= -XOVER_C)
            valid = (~swinging) | ((sw_leg == j) & (sw_mode == mode))
            H[:, c] = r - d
            LAND[:, c] = land0; A[:, c] = cs; R[:, c] = r; TL[:, c] = Tl
            B[:, c] = (e - 1.0) / (mass * om ** 2)
            geom = valid & (swinging | crossok)          # geometrically valid (crossover rule)
            SC[:, c] = np.where(geom, H[:, c], -np.inf)    # invalid candidates are excluded, never penalised
            FE[:, c] = geom
            RE[:, c] = geom & (d <= r)                      # within reach at F = 0
    jb = np.argmax(SC, axis=1)
    ii = np.arange(N)
    land0, a, r, beta, h0, Tl = LAND[ii, jb], A[ii, jb], R[ii, jb], B[ii, jb], H[ii, jb], TL[ii, jb]
    dv = land0 - a
    return dict(h0=h0, d=np.hypot(dv[:, 0], dv[:, 1]), r=r, a=a, land0=land0, beta=beta, jb=jb, Tl=Tl,
                nvalid=FE.sum(1), nreach=RE.sum(1), anyvalid=FE.any(1), F_star=min_norm_force(land0, a, r, beta))


# --------------------------------------------------------------------------
# patient step selection at lift-off
# --------------------------------------------------------------------------
def _choose_step(xi, fc, om, L, Ts, sc):
    """SPEC 5.1 + crossover option.  Preference: feasible side-step (smallest d/L), else feasible
    crossover (T_s x 1.25, L_max x 0.7, >= 0.09 m across the stance foot), else least-infeasible
    (target clipped onto that candidate's reachable set).  Returns leg, mode, target, T_s."""
    keys, tgts, Tss = [], [], []
    for mode in (0, 1):
        for j in (0, 1):
            cs = fc[:, 1 - j]
            hh = HS * sc[:, 1 - j, None]
            Tsc = Ts[:, j] * (CROSS_T if mode else 1.0)
            Lc = L[:, j] * (CROSS_L if mode else 1.0)
            pst = np.clip(xi, cs - hh, cs + hh)
            land = pst + (xi - pst) * np.exp(om * Tsc)[:, None]
            dd = land - cs
            d = np.hypot(dd[:, 0], dd[:, 1])
            sd = SIDE[j] * dd[:, 1]
            if mode == 0:
                crossok = sd >= XOVER
                cviol = np.maximum(XOVER - sd, 0)
            else:
                crossok = sd <= -XOVER_C
                cviol = np.maximum(sd + XOVER_C, 0)
            feas = (d <= Lc) & crossok
            viol = np.maximum(d - Lc, 0) + cviol
            keys.append(np.where(feas, 10.0 * mode + d / Lc, 1e3 + viol))
            if mode == 0:
                sclip = np.clip(sd, XOVER, Lc)
            else:
                sclip = -np.clip(-sd, XOVER_C, Lc)
            yy = SIDE[j] * sclip
            w = np.sqrt(np.maximum(Lc ** 2 - yy ** 2, 0))
            clipped = cs + np.stack([np.clip(dd[:, 0], -w, w), yy], -1)
            tgts.append(np.where(feas[:, None], land, clipped)); Tss.append(Tsc)
    keys = np.stack(keys, 1); cbest = np.argmin(keys, 1)
    ii = np.arange(len(xi))
    tgt = np.stack(tgts, 1)[ii, cbest]
    return cbest % 2, cbest // 2, tgt, np.stack(Tss, 1)[ii, cbest]


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class _Ctx:
    pass


def simulate(pop, ctrl, cfg=None, record=False, diag_marg=0.03):
    cfg = cfg or Config()
    dt = cfg.dt
    N = len(pop["m"])
    nsteps = int(round(cfg.T / dt))
    kpush = int(round(cfg.t_push / dt))
    Ds = int(round(cfg.tau_s / dt))
    Da = int(round(cfg.tau_a / dt))
    om, mass = pop["omega"], pop["m"]
    L, Ts, sc, taur = pop["Lmax"], pop["Ts"], pop["scale"], pop["tau_r"]
    off = Ds + Da + int(round(cfg.tau_lag / dt)) + 5
    idx = np.arange(N)
    Fcmd_log = np.zeros((nsteps + off, N, 2))
    x_log = np.zeros((nsteps + 1, N, 2))
    v_log = np.zeros((nsteps + 1, N, 2))

    x = np.zeros((N, 2)); v = np.zeros((N, 2))
    fc = np.zeros((N, 2, 2)); fc[:, 0, 1] = 0.10; fc[:, 1, 1] = -0.10
    sw_leg = -np.ones(N, int); sw_mode = np.zeros(N, int); sw_el = np.zeros(N); sw_T = np.ones(N)
    sw_start = np.zeros((N, 2)); sw_tgt = np.zeros((N, 2))
    steps = np.zeros(N, int); viol_t = -np.ones(N)
    alive = np.ones(N, bool); fallen = np.zeros(N, bool); t_fall = np.full(N, np.nan)
    t_eng = np.full(N, np.nan); t_force = np.full(N, np.nan)
    Fact_log = np.zeros((nsteps + off, N, 2))
    dec_t = np.full(N, np.inf); engaged = np.zeros(N, bool); xlock = np.zeros((N, 2))
    Fact = np.zeros((N, 2)); Fprev = np.zeros((N, 2))
    impulse = np.zeros(N); peakF = np.zeros(N); peakdF = np.zeros(N); peakv1 = np.zeros(N)
    peakCatch = np.zeros(N); t_first = np.full(N, np.nan)
    alpha = 1 - np.exp(-dt / cfg.tau_lag)
    ctrl.init(pop, cfg, N)
    c = _Ctx()
    c.dt = dt; c.off = off; c.Fcmd_log = Fcmd_log; c.N = N
    keys = ("t", "com", "comvel", "xi", "foot_l", "foot_r", "swing", "swing_h",
            "F", "level", "dist_xi", "rho", "rbound", "Fcmd")
    rec = {k: [] for k in keys} if record else None
    hs_full = np.broadcast_to(HS, (N, 2))

    for k in range(nsteps):
        t = k * dt
        if k == kpush:
            v = v + pop["dv"]
        x_log[k] = x; v_log[k] = v
        xi = x + v / om[:, None]
        single = sw_leg >= 0
        st = np.where(single, 1 - sw_leg, 0)
        lA = np.where(single, st, 0); lB = np.where(single, st, 1)
        cA, cB = fc[idx, lA], fc[idx, lB]
        hA, hB = HS * sc[idx, lA][:, None], HS * sc[idx, lB][:, None]
        hull_cop = make_hull(cA, hA, cB, hB)
        hull_geo = make_hull(cA, hs_full, cB, hs_full)
        # fall check
        newfall = alive & ~engaged & (dist_to(x, hull_geo) > D_FALL)
        t_fall[newfall] = t; fallen |= newfall; alive &= ~newfall
        # ankle strategy
        wA, wB = hA[:, 0] * hA[:, 1], hB[:, 0] * hB[:, 1]
        cS = (wA[:, None] * cA + wB[:, None] * cB) / (wA + wB)[:, None]
        p = project(xi + KC_ANKLE * (xi - cS), hull_cop)
        dist_xi = dist_to(xi, hull_cop)
        # controller
        c.k, c.t = k, t
        c.x_m, c.v_m = x_log[max(k - Ds, 0)], v_log[max(k - Ds, 0)]
        c.fc, c.sw_leg, c.sw_el, c.steps = fc, sw_leg, sw_el, steps
        c.sw_mode, c.alive = sw_mode, alive
        c.single, c.lA, c.lB, c.cA, c.cB = single, lA, lB, cA, cB
        c.cg = 0.5 * (cA + cB)
        c.cS, c.hull_geo, c.hull_cop = cS, hull_geo, hull_cop
        c.Fact_log = Fact_log
        Fcmd, trig = ctrl.step(c)
        newdec = trig & alive & np.isinf(dec_t)
        dec_t[newdec] = t
        neweng = alive & ~engaged & (t >= dec_t + cfg.tau_lock - 1e-9)
        xlock[neweng] = x[neweng]; engaged |= neweng; t_eng[neweng] = t
        Fcmd = np.where(engaged[:, None], 0.0, Fcmd)
        Fcmd_log[k + off] = Fcmd
        Fact = Fact + alpha * (Fcmd_log[k + off - Da] - Fact)
        Fact_log[k + off] = Fact
        Fc = np.where(engaged[:, None], -cfg.Kc * (x - xlock) - cfg.Dc * v, 0.0)
        Ftot = Fact + Fc
        mag = np.hypot(Ftot[:, 0], Ftot[:, 1])
        magA = np.hypot(Fact[:, 0], Fact[:, 1])
        dF = np.hypot(Fact[:, 0] - Fprev[:, 0], Fact[:, 1] - Fprev[:, 1]) / dt
        Fprev = Fact
        impulse += alive * mag * dt
        peakF = np.maximum(peakF, alive * magA)
        peakdF = np.maximum(peakdF, alive * dF)
        peakCatch = np.maximum(peakCatch, alive * np.hypot(Fc[:, 0], Fc[:, 1]))
        on = alive & (magA > 0.5)
        peakv1 = np.maximum(peakv1, on * np.hypot(v[:, 0], v[:, 1]))
        cmag = np.hypot(Fcmd[:, 0], Fcmd[:, 1])
        level = np.where(np.isfinite(dec_t), 2, np.where(cmag > cfg.f_on, 1, 0))
        fi = alive & np.isnan(t_force) & (magA > cfg.f_on)
        t_force[fi] = t

        if record:
            fl, fr = fc[:, 0].copy(), fc[:, 1].copy()
            frac = np.clip(sw_el / sw_T, 0, 1)
            pos = sw_start + (sw_tgt - sw_start) * frac[:, None]
            h = 0.06 * np.sin(np.pi * frac) * single
            fl = np.where((sw_leg == 0)[:, None], pos, fl)
            fr = np.where((sw_leg == 1)[:, None], pos, fr)
            Tr = np.maximum(taur * np.where(steps > 0, 0.6, 1.0) - np.maximum(viol_t, 0.0), 0.0)
            cb = candidate_barrier(xi, fc, sw_leg, sw_mode, sw_el, Tr, om, mass, L, Ts, sc, diag_marg)
            for key, val in (("t", t), ("com", x), ("comvel", v), ("xi", xi), ("foot_l", fl),
                             ("foot_r", fr), ("swing", np.where(sw_leg < 0, 0, sw_leg + 1)),
                             ("swing_h", h), ("F", Ftot), ("level", level), ("dist_xi", dist_xi),
                             ("rho", cb["d"]), ("rbound", cb["r"]), ("Fcmd", Fcmd)):
                rec[key].append(np.array(val, copy=True))

        # integrate
        x, v = lip_step(x, v, p, Ftot, mass, om, dt)
        # patient stepping logic (uses start-of-tick dist)
        act = sw_leg >= 0
        sw_el = np.where(act, sw_el + dt, sw_el)
        landed = act & (sw_el >= sw_T - 1e-9)
        if landed.any():
            li = np.nonzero(landed)[0]
            fc[li, sw_leg[li]] = sw_tgt[li]
            sw_leg[li] = -1
            viol_t[li] = -1.0
        ns = ~act
        viol = dist_xi > EPS
        viol_t = np.where(ns, np.where(viol, np.where(viol_t < 0, 0.0, viol_t + dt), -1.0), viol_t)
        teff = taur * np.where(steps == 0, 1.0, 0.6)
        lift = ns & (viol_t >= teff - 1e-9) & (steps < MAX_STEPS) & alive
        if lift.any():
            li = np.nonzero(lift)[0]
            xin = x[li] + v[li] / om[li][:, None]
            leg, mode, tgt, Tsel = _choose_step(xin, fc[li], om[li], L[li], Ts[li], sc[li])
            sw_leg[li] = leg; sw_mode[li] = mode; sw_el[li] = 0.0; sw_T[li] = Tsel
            sw_start[li] = fc[li, leg]; sw_tgt[li] = tgt
            steps[li] += 1; viol_t[li] = -1.0

    t_first = np.fmin(t_force, t_eng)   # physical first intervention (applied pelvic force or catch engagement)
    outcome = np.where(fallen, FALL, np.where(engaged, CAUGHT,
                       np.where(impulse >= cfg.impulse_thr, ASSISTED, SELF)))
    res = dict(outcome=outcome, impulse=impulse, peakF=peakF, peakdF=peakdF, peakv1=peakv1,
               peakCatch=peakCatch, n_steps=steps, t_first=t_first, t_fall=t_fall,
               t_decision=np.where(np.isfinite(dec_t), dec_t, np.nan), t_force=t_force, t_engage=t_eng,
               ever_assisted=impulse >= cfg.impulse_thr, ever_catch_decided=np.isfinite(dec_t),
               engaged=engaged, fallen=fallen)
    if hasattr(ctrl, "cause"):
        res["cause"] = ctrl.cause.copy()
    if record:
        n_end = nsteps
        if fallen[0]:
            n_end = min(nsteps, int(round(t_fall[0] / dt)) + 1)
        out = {}
        for kk, vv in rec.items():
            arr = np.stack(vv, 0)
            out[kk] = arr[:n_end] if kk == "t" else arr[:n_end, 0]
        res["rec"] = out
    return res


def params_json(pop, i=0):
    d = {k: np.asarray(pop[k][i]).tolist() for k in
         ("sev", "par", "m", "z0", "omega", "tau_r", "Lmax", "Ts", "scale", "dv")}
    return json.dumps(d)
