"""Physics 'patient' for CatchPoint: pretrained Unitree G1 12-dof locomotion policy (unitree_rl_gym, motion.pt)
in MuJoCo, with a crude stroke-like impairment, pelvic pushes, and a walker that applies the controller's pelvic
force through 8 cables (min-norm nonnegative tension distribution, analysis/i2_sizing/sizing.py).

Controllers (all on the REAL simulated humanoid state, with sensing delay):
  B0  none
  B1  threshold lock: |COM - feet centre| > theta  -> lock after 0.28 s -> stiff spring-damper (slip-limited)
  CP  CatchPoint: delay-compensated XCoM + step-reachability barrier + min-norm force <= F_assist,
      escalation to the slip-limited catch when more is needed.

Forces are scaled by body mass (LAM = m_G1 / 75 kg) so that the robot sees the same accelerations a 75 kg person
would see from the SPEC force levels (150 N assist, 300 N catch cap, slip tensions).  Overlays show both.
"""
from __future__ import annotations

import os
import sys
from collections import deque

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "i2_sizing"))
from sim.catchpoint import model as cpm  # noqa: E402
from sizing import min_norm_eq_box  # noqa: E402

torch.set_num_threads(1)
UG = os.path.join(HERE, "third_party", "unitree_rl_gym")
G1DIR = os.path.join(UG, "resources", "robots", "g1_description")
POLICY = os.path.join(UG, "deploy", "pre_train", "g1", "motion.pt")

DT, DEC = 0.002, 10
G = 9.81
KPS = np.array([100, 100, 100, 150, 40, 40] * 2, float)
KDS = np.array([2, 2, 2, 4, 2, 2] * 2, float)
Q0 = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2)
TAU_NOM = np.array([88, 139, 88, 139, 50, 50] * 2, float)
PERIOD = 0.8

# --- G1 foot (ankle_roll_link frame): contact spheres at x in [-0.05, 0.12], y +-0.03 (r 0.005)
FOOT_OFF = np.array([0.035, 0.0, -0.03])
HS_G1 = np.array([0.09, 0.035])

# --- SPEC / D3 contract numbers (human scale) ---------------------------------------------------------------
M_HUMAN = 75.0
F_ASSIST_H, F_CATCH_H = 150.0, 300.0
K_C_H, D_C_H = 4000.0, 800.0
TAU_S, TAU_A, TAU_LAG, TAU_LOCK = 0.03, 0.07, 0.02, 0.28
T_SLIP_H = {"corner": 150.0, "lateral": 150.0, "centre": 110.0}
CONTRACT = dict(
    corner=[[0.03, 0.33, 1.2], [0.03, -0.33, 1.2], [0.9, 0.33, 1.2], [0.9, -0.33, 1.2]],
    lateral=[[0.45, 0.4, 0.95], [0.45, -0.4, 0.95]],
    centre=[[0.03, 0.0, 0.95], [0.9, 0.0, 0.95]],
    bws=[0.42, 0.0, 1.8], pelvis=[0.42, 0.0, 0.95])
BELT_A_H, BELT_B_H = 0.11, 0.19


def gravity_orientation(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


def scene_xml():
    """Wrapper scene (written next to the G1 xml so the mesh paths resolve)."""
    xml = f"""<mujoco model="g1 catchpoint">
  <include file="g1_12dof.xml"/>
  <statistic center="0 0 0.7" extent="1.6"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="200" elevation="-12"/>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
    <map znear="0.02" zfar="40"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.93 0.95 0.97" rgb2="0.78 0.83 0.88" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.88 0.88 0.87" rgb2="0.84 0.84 0.83"
             markrgb="0.76 0.77 0.78" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="2 2" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.5 4" dir="-0.3 0.3 -1" directional="true" castshadow="true" diffuse="0.55 0.55 0.55"/>
    <light pos="-2 2 3" dir="0.4 -0.4 -1" directional="true" castshadow="false" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
  </worldbody>
</mujoco>"""
    p = os.path.join(G1DIR, "catchpoint_scene.xml")
    try:
        with open(p) as f:
            if f.read() == xml:
                return p
    except OSError:
        pass
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(xml)
    os.replace(tmp, p)
    return p


class Walker:
    """Static-geometry walker (scaled D3 contract) that follows the pelvis in L0/L1 and brakes on catch."""

    def __init__(self, s, lam):
        self.s = s
        pel = np.array(CONTRACT["pelvis"])
        self.anch_rel, self.kind, tsl = [], [], []
        for k in ("corner", "lateral", "centre"):
            for a in CONTRACT[k]:
                self.anch_rel.append(s * (np.array(a) - np.array([pel[0], pel[1], 0.0])))
                self.kind.append(k)
                tsl.append(T_SLIP_H[k] * lam)
        self.anch_rel = np.array(self.anch_rel)      # relative to (pelvis_x, pelvis_y, floor)
        self.T_slip = np.array(tsl)
        self.bws_rel = s * (np.array(CONTRACT["bws"]) - np.array([pel[0], pel[1], 0.0]))
        self.belt_a, self.belt_b = s * BELT_A_H, s * BELT_B_H
        self.pos = None          # walker reference point = nominal pelvis xy
        self.braked = False

    def follow(self, pel_xy, dt, tau=0.25, vmax=1.5):
        if self.pos is None:
            self.pos = pel_xy.copy()
        if self.braked:
            return
        dv = (pel_xy - self.pos) / tau
        n = np.hypot(*dv)
        if n > vmax:
            dv *= vmax / n
        self.pos = self.pos + dv * dt

    def anchors(self):
        return self.anch_rel + np.array([self.pos[0], self.pos[1], 0.0])

    def attachments(self, pel_pos, yaw):
        """Belt attachment points (ellipse around the pelvis, rotated by pelvis yaw) and unit vectors."""
        A = self.anchors()
        c, s_ = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s_], [s_, c]])
        att, U = [], []
        for a in A:
            d = Rz.T @ (a[:2] - pel_pos[:2])
            d = d / max(np.hypot(*d), 1e-9)
            k = 1.0 / np.sqrt((d[0] / self.belt_a) ** 2 + (d[1] / self.belt_b) ** 2)
            p = np.array([*(pel_pos[:2] + Rz @ (d * k)), pel_pos[2]])
            v = a - p
            att.append(p)
            U.append(v / np.linalg.norm(v))
        return A, np.array(att), np.array(U)


def distribute(U, Fh, Tmax, Tmin=1.0):
    """Min-norm nonnegative tensions with sum T u_xy = Fh, Tmin <= T <= Tmax (slip).  If infeasible (beyond the
    slip-limited envelope) the requested force is scaled down until feasible.  Returns T, scale."""
    sc = 1.0
    for _ in range(12):
        T = min_norm_eq_box(U[:, :2].T, Fh * sc, Tmin, Tmax)
        if T is not None:
            return T, sc
        sc *= 0.8
    return np.full(len(U), Tmin), 0.0


STANCE_FRAC = 0.55          # legged_gym G1 gait: a leg is in stance while its phase < 0.55
T_REDIRECT = 0.12           # a swing with less than this left cannot be redirected -> use the next one


def leg_phase(t, j):
    return ((t / PERIOD) + (0.0 if j == 0 else 0.5)) % 1.0


def landing_xi(xi, fc, t, j, om):
    """Patient model for the clocked G1 gait: propagate the LIP XCoM from time t to the next (redirectable)
    touchdown of leg j, with the CoP at the projection of xi onto the stance foot/feet of each gait phase
    (feet assumed to stay at their current positions).  Returns xi at landing and the horizon T."""
    tt, x, T = t, xi.copy(), 0.0
    ph = leg_phase(t, j)
    t_td = (1.0 - ph) * PERIOD                                 # time to the next touchdown of leg j
    if t_td < T_REDIRECT:
        t_td += PERIOD
    while T < t_td - 1e-9:
        # contact state at time tt: which legs are in stance
        st = [leg_phase(tt, k) < STANCE_FRAC for k in (0, 1)]
        # time to the next contact event (any leg crosses phase 0 or STANCE_FRAC)
        nxt = []
        for k in (0, 1):
            pk = leg_phase(tt, k)
            nxt.append(((STANCE_FRAC - pk) % 1.0 or 1.0) * PERIOD)
            nxt.append(((1.0 - pk) % 1.0 or 1.0) * PERIOD)
        dt = min(min(nxt), t_td - T)
        dt = max(dt, 1e-4)
        feet = [fc[k, :2] for k in (0, 1) if st[k]] or [fc[0, :2], fc[1, :2]]
        lo = np.min([f - HS_G1 for f in feet], 0)
        hi = np.max([f + HS_G1 for f in feet], 0)
        h = cpm.make_hull(feet[0][None], HS_G1[None], feet[-1][None], HS_G1[None])
        p = cpm.project(x[None], h)[0]
        x = p + (x - p) * np.exp(om * dt)
        T += dt
        tt += dt
    return x, T


# ======================================================================================================
class Sim:
    def __init__(self, ctrl="B0", paretic="R", weak=0.5, delay=0.06, theta=0.08, marg=0.03,
                 Lmax=(0.30, 0.20), z_com=None, record=False, seed=0, push=None, T=7.0,
                 R_cap=0.14 * np.exp(np.sqrt(G / 0.685) * 0.4), R_cap_n=0.25 * np.exp(np.sqrt(G / 0.685) * 0.4), T_H=0.40, dc=True, escalate=True, f_assist_scale=1.0,
                 esc_res=0.45, policy=None, release_after=None, b2_K=200.0, b2_theta=0.14, b2_t0=1.0):
        self.xml = scene_xml()
        self.m = mujoco.MjModel.from_xml_path(self.xml)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = DT
        self.pol = torch.jit.load(policy or POLICY)
        self.mass = float(self.m.body_mass.sum())
        self.lam = self.mass / M_HUMAN
        self.ctrl, self.paretic, self.weak = ctrl, paretic, weak
        self.theta, self.marg = theta, marg
        self.R_cap, self.R_cap_n, self.T_H, self.dc = R_cap, R_cap_n, T_H, dc
        self.Lmax = np.array(Lmax, float)          # (left, right) max step distance from stance foot [m]
        self.T = T
        self.push = push                           # dict(t0, dur, F (N), dir (unit 2D)) or a list of such dicts
        self.pushes = [push] if isinstance(push, dict) else list(push or [])
        self.release_after = release_after         # s: release the catch after this hold time (multi-push sessions)
        self.catch_on_t = None
        self.catch_events = []                     # (t_request, t_engaged, t_released)
        self.b2_x0 = None
        self.b2_t0, self.b2_theta = b2_t0, b2_theta
        # impairment: torque limit scale per joint (paretic leg joints 6..11 for R, 0..5 for L)
        self.tau_lim = TAU_NOM.copy()
        sl = slice(6, 12) if paretic == "R" else slice(0, 6)
        self.tau_lim[sl] *= weak
        ntot = int(round(delay / (DT * DEC)))           # total policy ticks of delay (20 ms each)
        self.nd_obs = (ntot + 1) // 2                    # observation delay ticks
        self.nd_act = ntot // 2                          # action delay ticks
        self.obs_q = deque()
        self.act_q = deque()
        self.act = np.zeros(12, np.float32)
        self.tgt = Q0.copy()
        self.pid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        mujoco.mj_forward(self.m, self.d)
        self.s = 0.78 / CONTRACT["pelvis"][2]
        self.walker = Walker(self.s, self.lam)
        self.F_assist = F_ASSIST_H * self.lam * f_assist_scale
        self.escalate, self.esc_res, self.esc_n = escalate, esc_res, 0
        self.F_catch = F_CATCH_H * self.lam
        self.b2_K = b2_K * self.lam
        self.b2_D = 2 * 0.7 * np.sqrt(self.b2_K * self.mass)
        self.Kc, self.Dc = K_C_H * self.lam, D_C_H * self.lam
        self.z_com = z_com if z_com is not None else 0.685
        self.om = np.sqrt(G / self.z_com)
        # delays (sim steps)
        self.Ds, self.Da = int(round(TAU_S / DT)), int(round(TAU_A / DT))
        self.alpha = 1 - np.exp(-DT / TAU_LAG)
        self.meas = deque(maxlen=self.Ds + 1)
        self.cmd_q = deque([np.zeros(2)] * self.Da, maxlen=self.Da)
        self.Fapp = np.zeros(2)
        self.Fapp_log = deque([np.zeros(2)] * self.Ds, maxlen=self.Ds)
        self.catch_req_t = None
        self.catch_on = False
        self.x_lock = None
        self.zlock = None
        self.b1_t = None
        self.k = 0
        self.fall_t = None
        self.log = {k: [] for k in ("t", "com", "comv", "xi", "fl", "fr", "cl", "cr", "F", "Fcmd", "level",
                                    "pel", "tilt", "T", "push", "dj", "resid", "catch")}
        self.record = record
        self.frames_state = []
        self.impulse = 0.0
        self.peakF = 0.0
        self.peakFxy = 0.0
        HS_save = cpm.HS
        cpm.HS = HS_G1
        self._hs_restore = HS_save

    # ------------------------------------------------------------------ helpers
    def t(self):
        return self.k * DT

    def feet(self):
        d = self.d
        c = []
        for b in self.fid:
            c.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ FOOT_OFF)
        cont = [False, False]
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = con.geom1, con.geom2
            if self.floor in (g1, g2):
                other = g2 if g1 == self.floor else g1
                bb = self.m.geom_bodyid[other]
                if bb == self.fid[0]:
                    cont[0] = True
                elif bb == self.fid[1]:
                    cont[1] = True
        return np.array(c), np.array(cont)

    def tilt(self):
        g = gravity_orientation(self.d.qpos[3:7])
        return np.degrees(np.arccos(np.clip(-g[2], -1, 1)))

    def yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # ------------------------------------------------------------------ policy
    def policy_tick(self):
        d = self.d
        ph = (self.k * DT % PERIOD) / PERIOD
        obs = np.zeros(47, np.float32)
        obs[:3] = d.qvel[3:6] * 0.25
        obs[3:6] = gravity_orientation(d.qpos[3:7])
        obs[9:21] = d.qpos[7:] - Q0
        obs[21:33] = d.qvel[6:] * 0.05
        obs[45:47] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
        self.obs_q.append(obs)
        if len(self.obs_q) <= self.nd_obs:
            o = self.obs_q[0].copy()
        else:
            o = self.obs_q.popleft()
        o[33:45] = self.act          # the policy knows its own last action
        with torch.no_grad():
            self.act = self.pol(torch.from_numpy(o)[None]).numpy()[0]
        self.act_q.append(self.act.copy())
        if len(self.act_q) > self.nd_act:
            a = self.act_q.popleft()
            self.tgt = a * 0.25 + Q0

    # ------------------------------------------------------------------ controllers
    def measure(self):
        d = self.d
        mujoco.mj_subtreeVel(self.m, d)
        com = d.subtree_com[self.pid].copy()
        v = d.subtree_linvel[self.pid].copy()
        fc, cont = self.feet()
        st = dict(com=com, v=v, fc=fc, cont=cont, t=self.t())
        self.meas.append(st)
        return st

    def hull(self, fc, which):
        """Support region: hull of the feet given by mask `which` (both -> double stance hull)."""
        c = fc[:, :2]
        if which[0] and which[1]:
            c1, c2 = c[0], c[1]
        elif which[0]:
            c1 = c2 = c[0]
        else:
            c1 = c2 = c[1]
        h = HS_G1[None, :]
        return cpm.make_hull(c1[None], h, c2[None], h)

    def predict_xi(self, xi_m, p):
        """Delay compensation: exact LIP propagation of the measured XCoM over tau_s + tau_a with constant CoP p,
        the logged applied force (past tau_s) and the lag-filtered commands already in the actuation pipeline."""
        e1 = np.exp(self.om * DT)
        c0 = (e1 - 1.0) / (self.mass * self.om ** 2)
        xi = xi_m.copy()
        for F in self.Fapp_log:
            xi = p + (xi - p) * e1 + c0 * F
        Fa = self.Fapp.copy()
        for Fc in self.cmd_q:
            Fa = Fa + self.alpha * (Fc - Fa)
            xi = p + (xi - p) * e1 + c0 * Fa
        return xi

    def cp_step(self, st):
        """CatchPoint on the delayed measurement st.  Returns (F_cmd, trigger_catch, info).

        Recoverable region (calibrated for this robot 'patient', see README): the delay-compensated XCoM, propagated
        T_H ahead with the LIP (CoP at the support boundary), may leave the both-feet support hull by at most R_CAP
        (toward the paretic side; R_CAP_N otherwise) minus margin m.  The SPEC's step-reachability model (landing
        XCoM vs. L_max) was implemented (landing_xi) but did not predict this policy's falls, see calib notes."""
        om, mass = self.om, self.mass
        x, v, fc = st["com"][:2], st["v"][:2], st["fc"]
        xi_m = x + v / om
        hull_both = self.hull(fc, [True, True])
        hull_st = self.hull(fc, st["cont"]) if st["cont"].any() else hull_both
        p = cpm.project(xi_m[None], hull_st)[0]
        xi = self.predict_xi(xi_m, p) if self.dc else xi_m
        info = dict(xi=xi, h0=np.nan, dj=np.full(2, np.nan))
        pb = cpm.project(xi[None], hull_both)[0]
        dv = xi - pb
        d0 = np.hypot(*dv)
        if d0 <= cpm.EPS:
            info["h0"] = self.R_cap - self.marg
            return np.zeros(2), False, info
        n = dv / d0
        towards_paretic = (n[1] < 0) if self.paretic == "R" else (n[1] > 0)
        R = (self.R_cap if towards_paretic else self.R_cap_n) - self.marg
        e = np.exp(om * self.T_H)
        dh = d0 * e
        beta = (e - 1.0) / (mass * om ** 2)
        h0 = R - dh
        info["h0"] = h0
        if h0 >= 0:
            return np.zeros(2), False, info
        F = -(dh - R) / beta * n
        info["resid"] = (dh - R) - beta * self.F_assist
        mag = np.hypot(*F)
        # escalation: the saturated assist leaves a residual barrier violation > esc_res (calibrated; the SPEC rule
        # 'escalate as soon as F* > F_max' over-catches this robot, whose own stepping is not in the LIP model)
        self.esc_n = self.esc_n + 1 if info["resid"] > self.esc_res else 0
        trig = self.escalate and self.esc_n >= 3
        if mag > self.F_assist:
            return F * self.F_assist / mag, trig, info
        return F, trig, info

    def b2_step(self, st):
        """SPEC 5.4 B2: always-on pelvic spring-damper to the initial COM (K mass-scaled from 200 N/m,
        D = 2*0.7*sqrt(K m), |F| <= 300 N-eq) + B1-style lock fallback at theta = 0.14 m."""
        if self.b2_x0 is None:
            if st["t"] < self.b2_t0:
                return np.zeros(2), False
            self.b2_x0 = st["com"][:2].copy()
        F = -self.b2_K * (st["com"][:2] - self.b2_x0) - self.b2_D * st["v"][:2]
        n = np.hypot(*F)
        if n > self.F_catch:
            F *= self.F_catch / n
        c = st["fc"][:, :2].mean(0)
        return F, np.hypot(*(st["com"][:2] - c)) > self.b2_theta

    def release_catch(self, t):
        self.catch_events[-1][2] = t
        self.catch_on = False
        self.catch_req_t = None
        self.catch_on_t = None
        self.walker.braked = False
        self.esc_n = 0
        self.cmd_q = deque([np.zeros(2)] * self.Da, maxlen=self.Da)
        self.Fapp = np.zeros(2)

    def b1_step(self, st):
        c = st["fc"][:, :2].mean(0)
        return np.zeros(2), np.hypot(*(st["com"][:2] - c)) > self.theta

    # ------------------------------------------------------------------ main loop
    def step(self):
        d, m = self.d, self.m
        t = self.t()
        if self.k % DEC == 0:
            self.policy_tick()
        tau = (self.tgt - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
        d.ctrl[:] = np.clip(tau, -self.tau_lim, self.tau_lim)

        st = self.measure()
        stm = self.meas[0]                         # delayed by tau_s
        level = 0
        Fcmd = np.zeros(2)
        trig = False
        info = {}
        if self.ctrl in ("CP", "CPmon"):
            Fcmd, trig, info = self.cp_step(stm)
            self.log["dj"].append(info["dj"])
            self.log["resid"].append(info.get("resid", -1.0))
            if self.ctrl == "CPmon":
                Fcmd, trig = np.zeros(2), False
        elif self.ctrl == "B1":
            Fcmd, trig = self.b1_step(stm)
        elif self.ctrl == "B2":
            Fcmd, trig = self.b2_step(stm)
        if self.fall_t is not None:
            Fcmd, trig = np.zeros(2), False
        if trig and self.catch_req_t is None:
            self.catch_req_t = t
            self.catch_events.append([t, None, None])
        if (self.release_after is not None and self.catch_on and t - self.catch_on_t >= self.release_after
                and self.fall_t is None):
            # release only once the patient is re-centred over the feet and nearly still (therapist resets)
            cf = st["fc"][:, :2].mean(0)
            if np.hypot(*(st["com"][:2] - cf)) < 0.06 and np.hypot(*st["v"][:2]) < 0.2:
                self.release_catch(t)
        if self.catch_req_t is not None and not self.catch_on and t - self.catch_req_t >= TAU_LOCK:
            self.catch_on = True
            self.catch_on_t = t
            self.catch_events[-1][1] = t
            self.walker.braked = True
            self.x_lock = st["com"][:2].copy()
            self.zlock = d.xpos[self.pid][2]
        # actuation pipeline (pure delay + first-order lag)
        self.cmd_q.append(Fcmd)
        Fc_del = self.cmd_q[0]
        self.Fapp = self.Fapp + self.alpha * (Fc_del - self.Fapp)
        Fh = self.Fapp.copy()
        if self.catch_on:   # mechanical lock: spring-damper on the true state, slip/cap limited
            Fh = -self.Kc * (st["com"][:2] - self.x_lock) - self.Dc * st["v"][:2]
            n = np.hypot(*Fh)
            if n > self.F_catch:
                Fh *= self.F_catch / n
        self.Fapp_log.append(Fh.copy())
        # walker kinematics
        pel = d.xpos[self.pid].copy()
        if pel[2] < 0.6:
            self.walker.braked = True
        self.walker.follow(pel[:2], DT)
        A, att, U = self.walker.attachments(pel, self.yaw())
        nF = np.hypot(*Fh)
        Tmax = self.walker.T_slip if self.catch_on else np.full(len(U), 1e4)
        if nF > 1e-6:
            T, sc = distribute(U, Fh, Tmax, Tmin=0.0)
        else:
            T = np.zeros(len(U))
        Fcab = U.T @ T                             # 3-D resultant on the pelvis
        # BWS harness (catch only): one-sided vertical spring below the lock height
        Fbws = 0.0
        if self.catch_on:
            dz = (self.zlock - 0.03) - pel[2]
            vz = d.qvel[2]
            if dz > 0:
                Fbws = float(np.clip(10000 * self.lam * dz - 1000 * self.lam * vz, 0, 1.5 * self.mass * G))
        Ftot = Fcab + np.array([0, 0, Fbws])
        trq = np.zeros(3)
        xip = d.xipos[self.pid]
        for i in range(len(T)):
            trq += np.cross(att[i] - xip, T[i] * U[i])
        trq += np.cross(pel - xip, np.array([0, 0, Fbws]))
        if self.catch_on:   # harness vest straps resist trunk tilt (catch only)
            R = d.xmat[self.pid].reshape(3, 3)
            w_world = R @ d.qvel[3:6]
            tr = 400.0 * self.lam * np.cross(R[:, 2], [0, 0, 1.0]) - 40.0 * self.lam * np.array([w_world[0], w_world[1], 0])
            n_ = np.linalg.norm(tr)
            if n_ > 150.0 * self.lam:
                tr *= 150.0 * self.lam / n_
            trq += tr
        # push
        Fpush = np.zeros(3)
        for pu in self.pushes:
            if pu["t0"] <= t < pu["t0"] + pu["dur"] and self.fall_t is None:
                Fpush[:2] = pu["F"] * np.asarray(pu["dir"], float)
                trq += np.cross(pel - xip, Fpush)
        d.xfrc_applied[self.pid, :3] = Ftot + Fpush
        d.xfrc_applied[self.pid, 3:] = trq
        mujoco.mj_step(m, d)
        self.k += 1
        # bookkeeping
        if self.ctrl != "B0":
            if self.catch_on or self.catch_req_t is not None:
                level = 2
            elif nF > 1.0:
                level = 1
        fH = np.hypot(*Fcab[:2])
        self.impulse += fH * DT
        self.peakF = max(self.peakF, float(np.linalg.norm(Fcab)))
        self.peakFxy = max(self.peakFxy, float(fH))
        if self.fall_t is None and (pel[2] < 0.55 or self.tilt() > 45):
            self.fall_t = t
        L = self.log
        L["t"].append(t); L["com"].append(st["com"]); L["comv"].append(st["v"])
        L["xi"].append(info.get("xi", st["com"][:2] + st["v"][:2] / self.om))
        L["fl"].append(st["fc"][0]); L["fr"].append(st["fc"][1])
        L["cl"].append(st["cont"][0]); L["cr"].append(st["cont"][1])
        L["F"].append(Fcab.copy()); L["Fcmd"].append(Fcmd.copy()); L["level"].append(level)
        L["pel"].append(pel); L["tilt"].append(self.tilt()); L["T"].append(T.copy())
        L["push"].append(np.hypot(*Fpush[:2]))
        L["catch"].append(self.catch_on)
        if self.record:
            self.cur = dict(A=A, att=att, T=T, walker=self.walker.pos.copy(), Tslip=self.walker.T_slip,
                            bws=(self.walker.bws_rel + np.array([*self.walker.pos, 0])), Fbws=Fbws,
                            level=level, F=Fcab.copy(), Fpush=Fpush.copy(), pel=pel, catch_on=self.catch_on)

    def run(self, cb=None, every=None):
        n = int(round(self.T / DT))
        for i in range(n):
            self.step()
            if cb is not None and i % every == 0:
                cb(self)
        return self.result()

    def outcome(self):
        if self.catch_on:
            return "caught"
        if self.fall_t is not None:
            return "fall"
        if self.impulse * 1.0 >= 1.0 * self.lam:
            return "nudged"
        return "recovered alone"

    def result(self):
        L = {k: np.array(v) for k, v in self.log.items()}
        return dict(outcome=self.outcome(), fall_t=self.fall_t, peakF=self.peakF, peakFxy=self.peakFxy,
                    peakF_h=self.peakF / self.lam, impulse=self.impulse, log=L,
                    catch_t=self.catch_req_t, per_push=self.per_push(L), catch_events=self.catch_events)

    def per_push(self, L=None, settle=2.2):
        """Outcome per push window [t0, next t0): fell / caught / nudged / alone, with the time the outcome is
        known (fall time, catch engagement, or t0 + settle).  Pushes after a fall are 'skipped'."""
        L = L or {k: np.array(v) for k, v in self.log.items()}
        t = L["t"]
        Fxy = np.hypot(L["F"][:, 0], L["F"][:, 1]) if len(L["F"]) else np.zeros(0)
        out = []
        ps = sorted(self.pushes, key=lambda p: p["t0"])
        for i, pu in enumerate(ps):
            t0 = pu["t0"]
            t1 = ps[i + 1]["t0"] if i + 1 < len(ps) else t[-1] + 1e-9
            m = (t >= t0) & (t < t1)
            if self.fall_t is not None and self.fall_t < t0:
                out.append(dict(t0=t0, F=pu["F"], outcome="skipped", t_known=None, impulse_h=0.0, peak_h=0.0))
                continue
            imp = float(Fxy[m].sum() * DT) if m.any() else 0.0
            pk = float(Fxy[m].max()) if m.any() else 0.0
            eng = [e[1] for e in self.catch_events if e[1] is not None and t0 <= e[1] < t1]
            if not eng and m.any() and L["catch"][m].any():      # catch still held from before / re-engaged
                eng = [float(t[m][np.argmax(L["catch"][m])])]
            if self.fall_t is not None and t0 <= self.fall_t < t1 and not (eng and eng[0] <= self.fall_t):
                o, tk = "fell", self.fall_t
            elif eng:
                o, tk = "caught", eng[0]
            elif imp >= 1.0 * self.lam:
                o, tk = "nudged", t0 + settle
            else:
                o, tk = "alone", t0 + settle
            out.append(dict(t0=t0, F=pu["F"], outcome=o, t_known=tk, impulse_h=round(imp / self.lam, 1),
                            peak_h=round(pk / self.lam, 1)))
        return out
