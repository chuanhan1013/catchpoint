"""Controllers (SPEC 5.4): B0, B1 threshold-lock, B1-XCoM, B2 impedance+lock, CatchPoint (+ ablations).

Each controller exposes init(pop, cfg, N) and step(ctx) -> (F_cmd (N,2), trigger_catch (N,) bool).
The engine latches the catch decision and applies the lock delay.
"""
from __future__ import annotations

import numpy as np

from .model import EPS, HS, candidate_barrier, dist_to, make_hull, project

F_MAX_ASSIST = 150.0
F_MAX_IMP = 300.0


class B0:
    name = "B0"

    def init(self, pop, cfg, N):
        self.N = N

    def step(self, c):
        return np.zeros((self.N, 2)), np.zeros(self.N, bool)


class B1:
    """Threshold-lock: catch when |x - c_S| > theta (measured, delayed).  c_S is the area-weighted
    centroid of the same scaled CoP region (S_cop) the plant's ankle strategy uses (stance foot only
    during single stance)."""

    def __init__(self, theta):
        self.theta = theta
        self.name = f"B1 theta={theta:g}"

    def init(self, pop, cfg, N):
        self.N = N

    def step(self, c):
        d = c.x_m - c.cS
        return np.zeros((self.N, 2)), np.hypot(d[:, 0], d[:, 1]) > self.theta


class B1XCoM:
    """Model-aware-ish threshold lock: catch when the delayed extrapolated COM (identified omega) is more than
    theta outside the support polygon.  Same capture-point idea as CP but no reach/step model, no graded force."""

    def __init__(self, theta):
        self.theta = theta
        self.name = f"B1-XCoM theta={theta:g}"

    def init(self, pop, cfg, N):
        self.N = N
        self.om = pop["ident"]["omega"]

    def step(self, c):
        xi = c.x_m + c.v_m / self.om[:, None]
        return np.zeros((self.N, 2)), dist_to(xi, c.hull_geo) > self.theta


class B2:
    """Always-on pelvic spring-damper to the initial COM + B1 lock fallback (theta = 0.14).
    Damping uses the IDENTIFIED (noisy) mass, same draw as CP."""

    def __init__(self, K, theta=0.14):
        self.K, self.theta = K, theta
        self.name = f"B2 K={K:g}"

    def init(self, pop, cfg, N):
        self.N = N
        self.D = 2 * 0.7 * np.sqrt(self.K * pop["ident"]["m"])

    def step(self, c):
        F = -self.K * c.x_m - self.D[:, None] * c.v_m
        mag = np.hypot(F[:, 0], F[:, 1])
        F = F * np.minimum(1.0, F_MAX_IMP / np.maximum(mag, 1e-9))[:, None]
        d = c.x_m - c.cS
        return F, np.hypot(d[:, 0], d[:, 1]) > self.theta


class CP:
    """CatchPoint: delay-compensated recoverability barrier + min-norm force + arbiter.

    marg  : margin m (scalar or per-trial array).
    dc    : delay compensation on/off (CP-noDC ablation = False).
    ident : which patient model CP uses: 'ident' (true x (1+N(0,0.1^2)) per patient) or
            'ident_pop' (population means, no identification: CP-pop ablation).
    """
    T_SHORT = 0.25   # [s] time-to-landing below this counts as 'short' for the catch-cause split
    T_FLOOR = 0.05   # [s] lower bound on time-to-landing during an ongoing swing

    def __init__(self, marg=0.03, dc=True, ident="ident"):
        self.marg0, self.dc, self.ident = marg, dc, ident
        self.name = f"CP m={marg:g}" if np.isscalar(marg) else "CP"
        if ident == "ident_pop":
            self.name = f"CP-pop m={marg:g}"
        if not dc:
            self.name = "CP-noDC"

    def init(self, pop, cfg, N):
        self.N = N
        self.id = pop[self.ident]
        self.marg = np.broadcast_to(np.asarray(self.marg0, float), (N,)).copy()
        self.vt = -np.ones(N)
        self.cause = -np.ones(N, int)   # -1 none; 0 no valid candidate; 1 F*>Fmax short horizon; 2 F*>Fmax long horizon
        self.tau_s, self.tau_a = cfg.tau_s, cfg.tau_a
        dt = cfg.dt
        self.Ds, self.Da = int(round(cfg.tau_s / dt)), int(round(cfg.tau_a / dt))
        alpha = 1 - np.exp(-dt / cfg.tau_lag)
        om, mh = self.id["omega"], self.id["m"]
        H = self.Ds + self.Da
        e = np.exp(om * dt)
        c0 = (e - 1.0) / (mh * om ** 2)
        W = c0[:, None] * np.exp(om[:, None] * dt * (H - 1 - np.arange(H))[None, :])   # weight of force of tick i
        self.eH = np.exp(om * H * dt)
        self.Wpast = W[:, :self.Ds]          # ticks k-Ds .. k-1 : applied force known from the log
        Wn = W[:, self.Ds:]                  # ticks k .. k+Da-1 : applied force follows the lag recursion
        R = np.zeros_like(Wn)
        acc = np.zeros(N)
        for m in range(self.Da - 1, -1, -1):  # R_m = W_m + (1-alpha) R_{m+1}
            acc = Wn[:, m] + (1 - alpha) * acc
            R[:, m] = acc
        self.Wcmd = alpha * R                # multiplies cmd[k-Da .. k-1]
        self.Wf = (1 - alpha) * R[:, 0]      # multiplies the applied force of tick k-1
        self.idx = np.arange(N)

    def predict_xi(self, xi_m, p, Fact_log, Fcmd_log, off, k):
        """XCoM at the start of tick k+Da from the measured XCoM at the start of tick k-Ds, with constant CoP p,
        the logged applied force of ticks k-Ds..k-1 and the EXACT first-order-lag + pure-delay propagation of the
        commands already issued (cmd[k-Da..k-1])."""
        past = Fact_log[off + k - self.Ds: off + k]
        cmds = Fcmd_log[off + k - self.Da: off + k]
        return (p + (xi_m - p) * self.eH[:, None]
                + np.einsum("nm,mnd->nd", self.Wpast, past)
                + self.Wf[:, None] * Fact_log[off + k - 1]
                + np.einsum("nm,mnd->nd", self.Wcmd, cmds))

    def step(self, c):
        i, N, k = self.id, self.N, c.k
        om, mh = i["omega"], i["m"]
        idx = self.idx
        xi_m = c.x_m + c.v_m / om[:, None]
        hA = HS * i["scale"][idx, c.lA][:, None]
        hB = HS * i["scale"][idx, c.lB][:, None]
        hull = make_hull(c.cA, hA, c.cB, hB)
        if self.dc:
            p = project(xi_m, hull)
            xi_h = self.predict_xi(xi_m, p, c.Fact_log, c.Fcmd_log, c.off, k)
            shift = -self.tau_a
        else:
            xi_h = xi_m
            shift = self.tau_s
        out = dist_to(xi_h, hull) > EPS
        self.vt = np.where(out, np.where(self.vt < 0, 0.0, self.vt + c.dt), -1.0)
        el = np.maximum(np.maximum(self.vt, 0.0) + shift, 0.0) * (self.vt >= 0)
        tau_eff = i["tau_r"] * np.where(c.steps == 0, 1.0, 0.6)
        Tr = np.maximum(tau_eff - el, 0.0)
        cb = candidate_barrier(xi_h, c.fc, c.sw_leg, c.sw_mode, c.sw_el, Tr, om, mh, i["Lmax"], i["Ts"],
                               i["scale"], self.marg, floor=self.T_FLOOR)
        valid = cb["anyvalid"]
        need = out & (~valid | (cb["h0"] < 0))
        Fs = np.where(valid[:, None], cb["F_star"], 0.0)
        mag = np.hypot(Fs[:, 0], Fs[:, 1])
        over = need & (~valid | (mag > F_MAX_ASSIST))      # no valid step candidate -> catch directly
        scale = np.where(over, F_MAX_ASSIST / np.maximum(mag, 1e-9), 1.0)
        F = np.where(need[:, None], Fs * scale[:, None], 0.0)
        first = over & c.alive & (self.cause < 0)
        if first.any():
            short = c.single | (cb["Tl"] < self.T_SHORT)
            cz = np.where(~valid, 0, np.where(short, 1, 2))
            self.cause[first] = cz[first]
        return F, over
