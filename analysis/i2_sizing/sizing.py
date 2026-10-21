"""I2 sizing model: walker-mounted cable pelvic interface (slack / nudge / arrest).

Coordinates (SPEC section 4): origin on the floor, x forward (walking direction), y left, z up.  Metres, N, kg.
Sign convention: T_i >= 0 is the tension of cable i; u_i is the unit vector FROM the belt attachment TOWARD the
anchor, so the force on the pelvis is +T_i u_i and the force on the walker frame at the anchor is -T_i u_i.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import nnls

G = 9.81

# ----------------------------------------------------------------------------------------------------------
# D3 contract: the single source of truth for D3 geometry (also read by cad/build_cad_d3.py)
# ----------------------------------------------------------------------------------------------------------
CONTRACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "d3_contract.json")


def load_contract(path=CONTRACT_PATH):
    with open(path, encoding="utf8") as f:
        return json.load(f)


D3C = load_contract()

# ----------------------------------------------------------------------------------------------------------
# generic least-distance solver (Lawson-Hanson LDP):  min ||x||  s.t.  Gm x >= h
# ----------------------------------------------------------------------------------------------------------


def ldp(Gm, h):
    Gm = np.atleast_2d(np.asarray(Gm, float))
    h = np.asarray(h, float)
    n = Gm.shape[1]
    E = np.vstack([Gm.T, h[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    try:
        u, _ = nnls(E, f, maxiter=50 * (n + len(h)) + 500)
    except RuntimeError:
        return None
    r = E @ u - f
    if abs(r[-1]) < 1e-12:
        return None
    return -r[:n] / r[-1]


def min_norm_eq_box(A, b, lo, hi, tol=1e-6):
    """min ||x||^2  s.t.  A x = b, lo <= x <= hi  (hi may be None).  Bounded least squares with a heavily weighted
    equality block (exact BVLS solver), then a null-space polish on the free variables so that the equality holds
    to ~1e-10.  Returns x, or None if the equality cannot be met within `tol` (infeasible)."""
    from scipy.optimize import lsq_linear
    A = np.atleast_2d(np.asarray(A, float))
    b = np.asarray(b, float)
    n = A.shape[1]
    sc = max(1.0, float(np.abs(b).max()))
    Wt = 1e5
    M = np.vstack([Wt * A, np.eye(n)])
    y = np.concatenate([Wt * b, np.zeros(n)])
    lo_ = np.broadcast_to(np.asarray(lo, float), n).astype(float)
    hi_ = np.full(n, np.inf) if hi is None else np.broadcast_to(np.asarray(hi, float), n).astype(float)
    r = lsq_linear(M, y, bounds=(lo_, hi_), method="bvls", tol=1e-14)
    x = r.x
    for _ in range(3):                      # polish: remove the residual using variables strictly inside bounds
        res = b - A @ x
        free = (x > lo_ + 1e-9) & (x < hi_ - 1e-9)
        if np.abs(res).max() < 1e-11 or not free.any():
            break
        x = x.copy()
        x[free] += np.linalg.pinv(A[:, free]) @ res
        x = np.clip(x, lo_, hi_)
    if np.abs(b - A @ x).max() > tol * max(1.0, sc * 1e-3):
        return None
    return x


# ----------------------------------------------------------------------------------------------------------
# geometry containers
# ----------------------------------------------------------------------------------------------------------


@dataclass
class Walker:
    """Rigid walker on braked wheels (+ optional deployed outrigger feet)."""
    m: float = 60.0                       # total mass incl. ballast [kg]
    cg: tuple = (0.40, 0.0, 0.45)         # centre of gravity [m]
    wx: tuple = (0.075, 0.875)            # wheel contact x (rear, front): footprint 0.95 minus wheel radius
    wy: float = 0.325                     # wheel contact |y|: footprint 0.80 minus wheel radius
    ex: float = 0.0                       # outrigger extension beyond wheel line, fore/aft [m]
    ey: float = 0.0                       # outrigger extension beyond wheel line, lateral [m]
    mu: float = 0.5                       # braked wheel / pad to floor friction (assumption)
    kgyr: float = 0.35                    # radius of gyration about CG [m] (dynamic tip check)

    def supports(self):
        xr, xf = self.wx
        pts = [(xr, self.wy), (xr, -self.wy), (xf, self.wy), (xf, -self.wy)]
        if self.ex > 0 or self.ey > 0:
            yy = self.wy + self.ey
            pts += [(xr - self.ex, yy), (xr - self.ex, -yy), (xf + self.ex, yy), (xf + self.ex, -yy)]
        return np.array(pts)

    def bounds(self):
        p = self.supports()
        return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()


BELT_DEFAULT = (0.11, 0.19)     # belt half-depth (AP) / half-width (ML) [m] for D0-D2 (SPEC: inner half-width 0.19, depth 0.22)


@dataclass
class Layout:
    name: str
    anchors: np.ndarray                    # (n,3) anchor points on the frame (= cable exit points)
    kinds: list = field(default_factory=list)
    belt: tuple = BELT_DEFAULT             # (half-depth, half-width) of the belt ellipse; D3 takes it from the contract


def make_layout(n=6, z_hi=1.20, z_lat=None, bws=False, x_rear=0.03, x_front=0.90, y_c=0.33,
                x_lat=0.45, y_lat=0.40, x_bws=0.45, z_bws=None, z_c=None, belt=BELT_DEFAULT):
    """n = 4 : four corner anchors (rear-L/R, front-L/R) at height z_hi
       n = 6 : + two lateral anchors (x_lat, +-y_lat) at z_lat (default z_hi)
       n = 8 : + rear-centre and front-centre anchors at z_c (default z_lat)
       bws   : + one overhead cable (body-weight-support / harness) above (x_bws, 0) at z_bws (default z_hi)."""
    z_lat = z_hi if z_lat is None else z_lat
    z_c = z_lat if z_c is None else z_c
    z_bws = z_hi if z_bws is None else z_bws
    A = [(x_rear, y_c, z_hi), (x_rear, -y_c, z_hi), (x_front, y_c, z_hi), (x_front, -y_c, z_hi)]
    K = ["rearL", "rearR", "frontL", "frontR"]
    if n >= 6:
        A += [(x_lat, y_lat, z_lat), (x_lat, -y_lat, z_lat)]
        K += ["latL", "latR"]
    if n >= 8:
        A += [(x_rear, 0.0, z_c), (x_front, 0.0, z_c)]
        K += ["rearC", "frontC"]
    if bws:
        A += [(x_bws, 0.0, z_bws)]
        K += ["bws"]
    return Layout(f"C{n}{'+B' if bws else ''}_z{z_hi:.2f}_{z_lat:.2f}", np.array(A, float), K, tuple(belt))


PELVIS = np.array(D3C["pelvis"], float)


def d3_walker(**kw):
    """D3 walker from the contract (mass, CG, wheel contacts, outriggers, pad friction)."""
    c = D3C
    base = dict(m=c["mass"], cg=tuple(c["cg"]), wx=tuple(c["wheels_x"]), wy=c["wheels_y"],
                ex=c["outrigger_ex"], ey=c["outrigger_ey"], mu=c["mu_pad"])
    base.update(kw)
    return Walker(**base)


def d3_layout(n=8, bws=True, **kw):
    """D3 cable layout from the contract anchors (exit points).  n = 8, bws = True reproduces the contract anchors
    exactly; other n / overrides (z_hi, z_lat, ...) give the D3 variants used in the sensitivity study."""
    c = D3C
    ca, la, ce = c["corner_anchors"], c["lateral_anchors"], c["centre_anchors"]
    args = dict(z_hi=ca[0][2], z_lat=la[0][2], z_c=ce[0][2], x_rear=ca[0][0], x_front=ca[2][0], y_c=ca[0][1],
                x_lat=la[0][0], y_lat=la[0][1], x_bws=c["bws_anchor"][0], z_bws=c["bws_anchor"][2],
                belt=(c["belt"]["hd"], c["belt"]["hw"]))
    args.update(kw)
    return make_layout(n, bws=bws, **args)


def cable_dirs(layout, P=PELVIS):
    """Belt attachment points (ray from the pelvis centre toward the anchor, intersected with the belt ellipse
    layout.belt) and unit vectors (belt -> anchor)."""
    P = np.asarray(P, float)
    att, U, L = [], [], []
    for a, k in zip(layout.anchors, layout.kinds):
        d = a[:2] - P[:2]
        if k == "bws" or np.hypot(*d) < 1e-9:
            p = P.copy()
        else:
            d = d / np.hypot(*d)
            s = 1.0 / np.sqrt((d[0] / layout.belt[0]) ** 2 + (d[1] / layout.belt[1]) ** 2)
            p = P + np.array([d[0] * s, d[1] * s, 0.0])
        v = a - p
        att.append(p)
        L.append(np.linalg.norm(v))
        U.append(v / np.linalg.norm(v))
    return np.array(att), np.array(U), np.array(L)


# ----------------------------------------------------------------------------------------------------------
# tension distribution (QP): min sum T^2  s.t.  sum T_i u_i = F_des, Tmin <= T <= Tmax
# ----------------------------------------------------------------------------------------------------------


def tension_distribution(U, Fh, Fz=None, Tmin=2.0, Tmax=1500.0):
    """Fh: desired horizontal force on the pelvis (2,), Fz: desired vertical force (None = free).
    Returns (T, feasible).  The equality is satisfied to solver precision (see tests)."""
    U = np.asarray(U, float)
    rows = [0, 1] + ([2] if Fz is not None else [])
    b = np.array([Fh[0], Fh[1]] + ([Fz] if Fz is not None else []), float)
    T = min_norm_eq_box(U[:, rows].T, b, Tmin, Tmax)
    if T is None:
        return np.full(len(U), np.nan), False
    return T, True


# ----------------------------------------------------------------------------------------------------------
# static equilibrium of the walker
# ----------------------------------------------------------------------------------------------------------


def frame_forces(layout, U, T):
    """Forces on the walker frame at the anchors (= -T u) and their points."""
    return -(np.asarray(T)[:, None] * U), layout.anchors


def _moments(F, R):
    """Return (N, ax, ay): the sums used for CoP:  N = -sum Fz ;  x_c N = sum (z Fx - x Fz) ; y_c N = -sum (y Fz - z Fy)."""
    N = -F[:, 2].sum()
    ax = (R[:, 2] * F[:, 0] - R[:, 0] * F[:, 2]).sum()
    ay = -(R[:, 1] * F[:, 2] - R[:, 2] * F[:, 1]).sum()
    return N, ax, ay


def walker_statics(w: Walker, layout, U, T, extra=None):
    """Static equilibrium of the walker under gravity + cable reactions (+ optional extra [(F(3), r(3))] loads).
    Returns dict with total normal load, CoP, tip margins, safety factors (load-scale to tip / slide), wheel normals."""
    Ff, Rf = frame_forces(layout, U, T)
    if extra:
        Ff = np.vstack([Ff] + [np.asarray(e[0], float)[None] for e in extra])
        Rf = np.vstack([Rf] + [np.asarray(e[1], float)[None] for e in extra])
    g_F = np.array([[0.0, 0.0, -w.m * G]])
    g_R = np.array([w.cg], float)
    N0, ax0, ay0 = _moments(g_F, g_R)
    N1, ax1, ay1 = _moments(Ff, Rf)
    N, ax, ay = N0 + N1, ax0 + ax1, ay0 + ay1
    xc, yc = ax / N, ay / N
    xmin, xmax, ymin, ymax = w.bounds()
    margins = dict(rear=xc - xmin, front=xmax - xc, right=yc - ymin, left=ymax - yc)
    # load-scale lambda at which each edge is reached (cable load scaled by lambda, gravity fixed)
    g = {"front": (xmax * N0 - ax0, ax1 - xmax * N1), "rear": (ax0 - xmin * N0, xmin * N1 - ax1),
         "left": (ymax * N0 - ay0, ay1 - ymax * N1), "right": (ay0 - ymin * N0, ymin * N1 - ay1)}
    lam = {}
    for k, (c0, c1) in g.items():
        lam[k] = c0 / c1 if c1 > 1e-12 else np.inf
    sf_tip = min(lam.values())
    edge = min(lam, key=lam.get)
    H = Ff[:, :2].sum(0)
    Hn = float(np.hypot(*H))
    den = Hn - w.mu * N1
    sf_slide = w.mu * N0 / den if den > 1e-12 else np.inf
    # wheel normals: min-norm, >=0, equilibrium (only meaningful when CoP is inside the polygon)
    S = w.supports()
    Aeq = np.vstack([np.ones(len(S)), S[:, 0], S[:, 1]])
    beq = np.array([N, xc * N, yc * N])
    Nw = None if min(margins.values()) < 0 else min_norm_eq_box(Aeq, beq, 0.0, None)
    if Nw is None:
        Nw = np.full(len(S), np.nan)
    return dict(N_tot=N, cop=(xc, yc), margins=margins, margin_min=min(margins.values()), sf_tip=sf_tip,
                tip_edge=edge, H=H, H_mag=Hn, Fz_down_on_frame=float(-Ff[:, 2].sum()), sf_slide=sf_slide,
                wheel_N=Nw, N_cable=N1, supports=S, tipped=min(margins.values()) < 0)


# ----------------------------------------------------------------------------------------------------------
# one load case
# ----------------------------------------------------------------------------------------------------------


def load_case(w: Walker, layout, F_mag, phi_deg, P=PELVIS, Fz=None, Tmin=2.0, Tmax=1500.0, extra=None, Fz_cap=None):
    """Horizontal force F_mag on the pelvis in direction phi (deg from +x toward +y).
    Fz     : prescribed vertical force on the pelvis (None = whatever the cable geometry gives).
    Fz_cap : if the natural (min-tension) solution lifts the pelvis by more than Fz_cap [N], re-solve with the
             vertical force pinned at Fz_cap (infeasible if the layout cannot do that).
    Returns the statics dict + T, U, feasibility."""
    att, U, Lc = cable_dirs(layout, P)
    ph = np.radians(phi_deg)
    Fh = F_mag * np.array([np.cos(ph), np.sin(ph)])
    T, ok = tension_distribution(U, Fh, Fz, Tmin, Tmax)
    if ok and Fz is None and Fz_cap is not None and (U[:, 2] * T).sum() > Fz_cap + 1e-6:
        T, ok = tension_distribution(U, Fh, Fz_cap, Tmin, Tmax)
    if not ok:
        return dict(feasible=False, T=T, U=U, att=att, Lc=Lc)
    st = walker_statics(w, layout, U, T, extra)
    st.update(feasible=True, T=T, U=U, att=att, Lc=Lc, Fp=U.T @ T, Tmax_cable=float(T.max()),
              Fz_pelvis=float((U[:, 2] * T).sum()), F_mag=F_mag, phi=phi_deg)
    return st


def worst_direction(w, layout, F_mag, P=PELVIS, Fz=None, step=15, Tmin=2.0, Tmax=1500.0, key="margin_min", Fz_cap=None):
    best = None
    for phi in range(0, 360, step):
        r = load_case(w, layout, F_mag, phi, P, Fz, Tmin, Tmax, Fz_cap=Fz_cap)
        if not r["feasible"]:
            return phi, r
        if best is None or r[key] < best[1][key]:
            best = (phi, r)
    return best


def max_arrest_force(w, layout, sf=1.0, P=PELVIS, Fz_cap=None, step=15, Fcap=1000.0, dF=10.0, slide=False,
                     Tmax=1500.0):
    """Per direction, the largest horizontal pelvic force F such that for every F' <= F the cable layout is feasible
    (including the vertical-lift cap) and the static tip safety factor is >= sf (cable loads scaled by sf do not lift a
    wheel); optionally the sliding SF >= sf too.  Returns {phi: F_allow}."""
    out = {}
    for phi in range(0, 360, step):
        Fok = 0.0
        for F in np.arange(dF, Fcap + 1e-9, dF):
            r = load_case(w, layout, F, phi, P, None, 2.0, Tmax, Fz_cap=Fz_cap)
            if not r["feasible"] or r["sf_tip"] < sf or (slide and r["sf_slide"] < sf):
                break
            Fok = F
        out[phi] = Fok
    return out


def required_mass(layout, F_mag, base, sf=1.0, P=PELVIS, step=15, mmax=600.0, Fz=None, Fz_cap=None):
    """Smallest walker mass (CG position fixed) with static tip SF >= sf for horizontal force F_mag in ALL directions."""
    lo, hi = 10.0, mmax

    def ok(m):
        w = replace(base, m=m)
        for phi in range(0, 360, step):
            r = load_case(w, layout, F_mag, phi, P, Fz, Fz_cap=Fz_cap)
            if not r["feasible"] or r["sf_tip"] < sf:
                return False
        return True

    if not ok(hi):
        return np.inf
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ----------------------------------------------------------------------------------------------------------
# required arrest force from the simulated catch states (rigid-cable closed form)
# ----------------------------------------------------------------------------------------------------------


def required_force(m, omega, s0, vout, s_allow=0.25):
    """Minimum constant opposing force that stops the LIP COM (ankle CoP at the nearest support point) with peak
    distance-from-support <= s_allow.  With u = s - F/(m w^2): u'' = w^2 u, energy  u'^2 - w^2 u^2 = const, so the
    COM turns around at  s_m = s* - sqrt((s* - s0)^2 - (v/w)^2),  s* = F/(m w^2).
      * stopping at all needs s* >= s0 + v/w  (capture-point condition; at equality the COM only creeps toward s*,
        s_m = s*).  If s0 + v/w <= s_allow this limit is the answer: F_min = m w^2 (s0 + v/w) = m w (w s0 + v).
      * otherwise s_m = s_allow gives  s* = (s_a^2 - s0^2 + (v/w)^2) / (2 (s_a - s0)).
    v is clipped at 0 (inward motion needs only the lean load).  Returns +inf if s0 >= s_allow."""
    m, omega, s0, vout = map(np.asarray, (m, omega, s0, vout))
    v = np.maximum(vout, 0.0)
    cap = s0 + v / omega
    den = 2 * (s_allow - s0)
    s_form = (s_allow ** 2 - s0 ** 2 + (v / omega) ** 2) / np.where(den > 0, den, np.nan)
    s_req = np.where(cap <= s_allow, cap, s_form)
    F = m * omega ** 2 * s_req
    return np.where(s0 < s_allow, F, np.inf)


# ----------------------------------------------------------------------------------------------------------
# 1-D arrest dynamics (vectorised over trials): patient LIP + compliant cable + slip brake + sliding walker
# ----------------------------------------------------------------------------------------------------------


def arrest_1d(m_p, omega, s0, v0, F_lim, m_w, mu, kappa, k_c=15000.0, zeta=0.2, slack=0.0,
              t_end=1.2, dt=5e-4, s_allow=0.25, hscale=1.0):
    """Along the force direction (n).  s = distance of the patient's COM from the support edge (outward +).
    Patient: m s'' = m w^2 s - F   (ankle CoP at the support edge: optimistic).
    Cable: stiffness k_c [N/m] (cable + belt + tissue + frame, lumped), damping 2 zeta sqrt(k m_p); brake slips at F_lim
    (spool brake as torque limiter), cable taut after `slack` metres of travel.
    Walker: Coulomb friction mu (m_w g + kappa F) (kappa = downward frame load per unit F; cable downward
    component), mass m_w; moves only along +n.  hscale = |horizontal resultant on the frame| / F (1 when the cable
    force is purely along n; > 1 when passive brakes add a perpendicular component, which also loads the friction).
    Returns dict(arrays): s_max, walker_slide, F_peak, t_stop, F_hist (n_steps,Ntrials, decimated), stopped (bool), ok."""
    m_p, omega, s0, v0, F_lim = [np.broadcast_to(np.asarray(a, float), np.shape(m_p)).copy()
                                 for a in (m_p, omega, s0, v0, F_lim)]
    N = len(m_p)
    s, v = s0.copy(), v0.copy()
    w = np.zeros(N)
    u = np.zeros(N)
    pay = np.zeros(N)
    c = 2 * zeta * np.sqrt(k_c * m_p)
    smax = s.copy()
    Fpk = np.zeros(N)
    F = np.zeros(N)
    t_stop = np.full(N, np.nan)
    nst = int(round(t_end / dt))
    dec = 20
    hist = np.zeros((nst // dec + 1, N))
    for k in range(nst):
        e = (s - s0) - w - pay - slack
        ed = v - u
        taut = e > 0
        F0 = np.where(taut, k_c * e + c * ed, 0.0)
        F0 = np.maximum(F0, 0.0)
        slip = F0 > F_lim
        payd = np.where(slip, np.maximum(0.0, ed + (k_c * e - F_lim) / c), 0.0)
        F = np.where(slip, F_lim, F0)
        # walker
        fcap = mu * (m_w * G + kappa * F)
        moving = u > 1e-9
        Fh = hscale * F
        a_w = np.where(moving, (Fh - fcap) / m_w, np.where(Fh > fcap, (Fh - fcap) / m_w, 0.0))
        u = np.maximum(u + a_w * dt, 0.0)
        w = w + u * dt
        pay = pay + payd * dt
        a_p = omega ** 2 * s - F / m_p
        v = v + a_p * dt
        s = s + v * dt
        smax = np.maximum(smax, s)
        t_stop = np.where(np.isnan(t_stop) & (v <= 0) & (k > 0), (k + 1) * dt, t_stop)
        Fpk = np.maximum(Fpk, F)
        if k % dec == 0:
            hist[k // dec] = F
    ok = smax <= s_allow          # unstable (F_lim < lean load) cases diverge exponentially within t_end and fail this
    return dict(s_max=smax, t_stop=t_stop, walker_slide=w, F_peak=Fpk, pay=pay, F_hist=hist, dt_hist=dt * dec, ok=ok,
                s_end=s, v_end=v)


# ----------------------------------------------------------------------------------------------------------
# dynamic tip check: rigid body rotating about the tipping edge under a cable-moment history
# ----------------------------------------------------------------------------------------------------------


def dynamic_tip(w: Walker, M_over_ref, F_ref, F_hist, dt_hist, a_edge, edge_dir_z=None):
    """Planar rotation of the walker about the edge it would tip over.
    M_over_ref : overturning moment of the cable loads about the edge at cable load F_ref  [N m] (from statics:
                 moment of the frame reaction forces about the edge line; positive = tends to lift the far side).
    F_hist     : (K,) horizontal cable force history [N]; M_over(t) = M_over_ref * F_hist/F_ref.
    a_edge     : horizontal distance from the edge to the walker CG [m] ; h = CG height.
    Returns (tipped bool, max rotation angle [rad], critical angle)."""
    h = w.cg[2]
    r = np.hypot(a_edge, h)
    beta = np.arctan2(h, a_edge)
    th_crit = np.pi / 2 - beta
    I = w.m * (r ** 2 + w.kgyr ** 2)
    th = 0.0
    om = 0.0
    th_max = 0.0
    sub = 10
    d = dt_hist / sub
    for Fh in F_hist:
        Mo = M_over_ref * Fh / F_ref
        for _ in range(sub):
            Mr = w.m * G * r * np.cos(beta + th)
            if th <= 0 and Mo <= Mr:
                continue
            al = (Mo - Mr) / I
            om += al * d
            th += om * d
            if th < 0:
                th, om = 0.0, 0.0
            th_max = max(th_max, th)
            if th_max >= th_crit:
                return True, th_max, th_crit
    # free flight after the pulse (no more overturning moment): can it still pass the balance point?
    while om > 0 and th < th_crit:
        Mr = w.m * G * r * np.cos(beta + th)
        om += (-Mr / I) * d
        th += om * d
        th_max = max(th_max, th)
        if th_max >= th_crit:
            return True, th_max, th_crit
    return False, th_max, th_crit


def edge_moment(w, layout, U, T, edge):
    """Overturning moment of the cable (frame-reaction) loads about the named tipping edge line, and the
    horizontal distance a from edge to CG.  Positive = tends to tip over that edge."""
    Ff, Rf = frame_forces(layout, U, T)
    xmin, xmax, ymin, ymax = w.bounds()
    # moment about a floor line: use the component along the line; overturning toward the edge side positive
    if edge == "front":      # line x = xmax, axis y ; tipping rotation about +y... moment_y = z Fx - (x-xmax) Fz
        M = sum(r[2] * f[0] - (r[0] - xmax) * f[2] for f, r in zip(Ff, Rf)); a = xmax - w.cg[0]
    elif edge == "rear":
        M = sum(-(r[2] * f[0]) + (r[0] - xmin) * f[2] for f, r in zip(Ff, Rf)); a = w.cg[0] - xmin
    elif edge == "left":     # line y = ymax
        M = sum(r[2] * f[1] - (r[1] - ymax) * f[2] for f, r in zip(Ff, Rf)); a = ymax - w.cg[1]
    else:                    # right, line y = ymin
        M = sum(-(r[2] * f[1]) + (r[1] - ymin) * f[2] for f, r in zip(Ff, Rf)); a = w.cg[1] - ymin
    return float(M), float(a)


# ----------------------------------------------------------------------------------------------------------
# spool / brake / servo sizing
# ----------------------------------------------------------------------------------------------------------


def spool_brake(T_arrest, T_nudge, r_s=0.025, sf_dyn=1.5, eta=0.9, N_gear=12.0, J_m=5e-6):
    """Brake holding torque, servo torque at the spool, motor-side torque and reflected inertia/mass."""
    tau_brake = sf_dyn * T_arrest * r_s
    tau_spool_nudge = T_nudge * r_s
    tau_motor = tau_spool_nudge / (N_gear * eta)
    J_ref = J_m * N_gear ** 2
    return dict(tau_brake=tau_brake, tau_spool_nudge=tau_spool_nudge, tau_motor=tau_motor, J_ref=J_ref,
                m_reflected=J_ref / r_s ** 2)


# ----------------------------------------------------------------------------------------------------------
# per-spool slip: resultant force when each spool brake slips at its own tension
# ----------------------------------------------------------------------------------------------------------


def taut_mask(U, d_xy, thr=0.05, exclude=None):
    """Cables whose length grows when the pelvis moves along unit vector d_xy (u points belt -> anchor, so moving away
    from the anchor, u . d < 0, pays cable out and loads it).  `exclude`: boolean mask of cables never loaded in the
    horizontal arrest (the overhead BWS cable is kept slack)."""
    m = (np.asarray(U)[:, :2] @ np.asarray(d_xy)) < -thr
    if exclude is not None:
        m = m & ~np.asarray(exclude)
    return m


def slip_resultant(U, Tslip, d_xy, exclude=None):
    """3-D force on the pelvis when every loaded cable sits at its slip tension (upper bound of passive slip)."""
    m = taut_mask(U, d_xy, exclude=exclude)
    T = np.where(m, Tslip, 0.0)
    return U.T @ T, T


SLIP_GROUP = {"rearL": "corner", "rearR": "corner", "frontL": "corner", "frontR": "corner",
              "latL": "lateral", "latR": "lateral", "rearC": "centre", "frontC": "centre"}


def group_slip_vector(layout, T_slip):
    """Per-cable slip tension from group thresholds {'corner','lateral','centre'}; the BWS cable gets 0 (kept slack)."""
    return np.array([float(T_slip[SLIP_GROUP[k]]) if k in SLIP_GROUP else 0.0 for k in layout.kinds])


def slip_upper_bound(U, Tslip, exclude=None, n=720):
    """Largest horizontal resultant any subset of cables at slip tension can produce (all cables slipping, geometry
    ignored): max over directions e of sum_i max(0, T_i u_i . e)."""
    keep = np.ones(len(U), bool) if exclude is None else ~np.asarray(exclude)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    E = np.stack([np.cos(ang), np.sin(ang)], 1)
    P = (E @ U[keep, :2].T) * np.asarray(Tslip)[keep][None, :]
    return float(np.maximum(P, 0).sum(1).max())
