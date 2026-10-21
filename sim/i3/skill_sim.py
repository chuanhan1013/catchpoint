"""Skill-parameterised copy of the CatchPoint engine (model.simulate) for the I3 surrogate learner.

The ONLY differences from catchpoint.model.simulate (which stays untouched) are four per-trial skill
parameters read from pop["skill"] (defaults reproduce the original engine exactly, see test_i3.py):
  eps  step-trigger threshold [m]: step is initiated once the XCoM has been > eps outside the CoP region
       for the reaction time (default model.EPS = 0.01)
  kc   ankle-strategy gain (default model.KC_ANKLE = 0.5)
  g    foot-placement gain: target = stance + g * (predicted capture point - stance) (default 1), clipped
       to the physical reach Lmax
  rho  crossover preference in [0,1]: 0 = side-step always preferred (original), 1 = no preference
Physical limits (tau_r, Lmax, Ts, scale) are untouched.  Extra outputs: step_len [m], ankle_int [m s].
"""
from __future__ import annotations

import numpy as np

from catchpoint.model import (CROSS_L, CROSS_T, D_FALL, EPS, HS, KC_ANKLE, MAX_STEPS, XOVER, XOVER_C, SIDE,
                              SELF, ASSISTED, CAUGHT, FALL, Config, _Ctx, candidate_barrier, dist_to,
                              lip_step, make_hull, project)

DEFAULT_SKILL = dict(eps=EPS, kc=KC_ANKLE, g=1.0, rho=0.0)


def default_skill(N):
    return {k: np.full(N, v, float) for k, v in DEFAULT_SKILL.items()}


def _choose_step_skill(xi, fc, om, L, Ts, sc, rho):
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
            keys.append(np.where(feas, 10.0 * (1.0 - rho) * mode + d / Lc, 1e3 + viol))
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




def simulate_skill(pop, ctrl, cfg=None):
    cfg = cfg or Config()
    record = False
    sk = pop["skill"]
    eps_t, kc_t, g_t, rho_t = sk["eps"], sk["kc"], sk["g"], sk["rho"]
    step_len = np.zeros(N_ := len(pop["m"])); ankle_int = np.zeros(N_)
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
        p = project(xi + kc_t[:, None] * (xi - cS), hull_cop)
        ankle_int += alive * np.hypot(*(p - cS).T) * dt
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
        viol = dist_xi > eps_t
        viol_t = np.where(ns, np.where(viol, np.where(viol_t < 0, 0.0, viol_t + dt), -1.0), viol_t)
        teff = taur * np.where(steps == 0, 1.0, 0.6)
        lift = ns & (viol_t >= teff - 1e-9) & (steps < MAX_STEPS) & alive
        if lift.any():
            li = np.nonzero(lift)[0]
            xin = x[li] + v[li] / om[li][:, None]
            leg, mode, tgt, Tsel = _choose_step_skill(xin, fc[li], om[li], L[li], Ts[li], sc[li], rho_t[li])
            csl = fc[li, 1 - leg]
            Lc = L[li, leg] * np.where(mode == 1, CROSS_L, 1.0)
            d_ = g_t[li, None] * (tgt - csl)
            dn = np.hypot(d_[:, 0], d_[:, 1])
            d_ = d_ * np.minimum(1.0, Lc / np.maximum(dn, 1e-9))[:, None]      # physical reach limit
            tgt = np.where((g_t[li] == 1.0)[:, None], tgt, csl + d_)
            step_len[li] += np.hypot(*(tgt - fc[li, leg]).T)
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
               engaged=engaged, fallen=fallen, step_len=step_len, ankle_int=ankle_int)
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


