"""Kinematic replay of simulated CatchPoint trajectories: Unitree G1 inside the D3 walker (MuJoCo, offscreen).

    python replay.py compare          -> compare_4controllers.mp4/.gif   (results/showcase_{B0,B1,B2,CP}.npz, 2x2, 1920x1080)
    python replay.py scen             -> cp_small / cp_bigger / cp_big .mp4/.gif   (data/cp_*.npz, 1280x720)
    python replay.py still            -> cp_three_states.png
    python replay.py all              -> everything
    python replay.py frame <npz> <t> <out.png> [w h]   (debug: one frame)

The humanoid is NEVER simulated.  Each frame: set the free joint (pelvis) from `com`, lean the trunk along (com - support
centre), solve damped-least-squares IK (mj_jacSite) so that the two foot sites reach foot_l / foot_r (+ swing_h lift) with a
flat foot, then mj_forward.  The walker is static; belt and cable/strap attach points follow the pelvis.
"""
import json
import os
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_scene as bs                                         # noqa: E402
from build_scene import Rig, hex2rgb, BELT_BELOW_PELVIS, PELVIS_XY0, LEVEL_HEX, RING_HEX, CABLES   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
FONT_R = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
FPS = 30
SLOW, T_SLOW0, T_SLOW1 = 0.5, 0.2, 2.0      # 0.5x from t = 0.2 s (just before the push at 0.3 s) to 2.0 s
HOLD = 1.5
SWING_H_GAIN = 1.5
LEVEL_NAME = {0: "L0  Transparent", 1: "L1  Assist", 2: "L2  Catch"}
CTRL_NAME = {"B0": "B0  No assist", "B1": "B1  Threshold lock", "B2": "B2  Impedance + lock",
             "CP": "CP  Predictive graded assist"}
CTRL_HEX = {"B0": "#9AA0A6", "B1": "#7B5EA7", "B2": "#3C78B4", "CP": "#1F6F78"}
OUT_HEX = {"SELF": "#3FAE5C", "ASSISTED": "#E0A030", "CAUGHT": "#C8443A", "FALL": "#7A1F1F"}
DISCLAIMER = "Kinematic replay of simulated trajectories (LIP model) — visualization only"
CAM_POS, CAM_LOOK = [3.69, -2.70, 2.72], [0.475, 0.0, 0.85]


# ------------------------------------------------------------------ data
def find_key(obj, key, default):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key, None)
            if r is not None:
                return r
    return default


class Traj:
    """One showcase npz plus the derived per-sample quantities the renderer needs."""

    def __init__(self, path, name=None):
        d = np.load(path)
        self.__dict__.update({k: d[k] for k in d.files if k not in ("outcome", "params")})
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.outcome = str(d["outcome"])
        prm = json.loads(str(d["params"]))
        self.z0 = float(find_key(prm, "z0", 0.95))
        self.dv = np.array(find_key(prm, "dv", [0, 0]), float)
        # the logged level is the COMMAND level; the delayed/lagged force can still act (> 5 N) for ~0.1 s after it drops
        # to L0 -> show it as L1 so badge, arrow and cables stay consistent
        self.level = np.where((self.level == 0) & (np.linalg.norm(self.F, axis=1) > 5.0), 1, self.level)
        self.T = float(self.t[-1])
        self.dt = float(self.t[1] - self.t[0])
        self._derive()

    def _derive(self):
        n, dt = len(self.t), self.dt
        # cable pull directions (horizontal, belt -> anchor) at rest
        belt0 = {c: np.array(bs.json.load(open(os.path.join(bs.CAD, "parts_d3.json")))["anchors"][c]["belt"]) for c in CABLES}
        ex0 = {c: np.array(bs.json.load(open(os.path.join(bs.CAD, "parts_d3.json")))["anchors"][c]["exit"]) for c in CABLES}
        pull = np.array([(ex0[c] - belt0[c])[:2] / np.linalg.norm((ex0[c] - belt0[c])[:2]) for c in CABLES])
        names = CABLES + ["strapL", "strapR"]
        target = np.zeros((n, len(names)))            # 1 = taut
        colour = np.zeros((n, len(names)), int)       # level code used to colour a taut cable
        for k in range(n):
            lv = int(self.level[k])
            F = self.F[k]
            nf = np.linalg.norm(F)
            if lv >= 2:
                target[k, :] = 1
                colour[k, :] = 2
            elif lv == 1 and nf > 1.0:
                cs = pull @ (F / nf)
                order = np.argsort(-cs)
                pick = [i for i in order[:3] if cs[i] > 0.35] or list(order[:2])
                if len(pick) < 2:
                    pick = list(order[:2])
                for i in pick:
                    target[k, i] = 1
                    colour[k, i] = 1
        a = 1 - np.exp(-dt / 0.05)                    # 50 ms low-pass: slack <-> taut transition
        taut = np.zeros_like(target)
        cur = np.zeros(target.shape[1])
        for k in range(n):
            cur = cur + a * (target[k] - cur)
            taut[k] = cur
        self.taut, self.colour, self.cable_names = taut, colour, names
        self.base, self.rel = base_track(self)
        l2 = np.nonzero(self.level >= 2)[0]
        self.t_l2 = float(self.t[l2[0]]) if len(l2) else None

    def k_at(self, ts):
        return int(np.clip(np.searchsorted(self.t, ts), 0, len(self.t) - 1))


def base_track(tr, tau=0.15, cap=0.08):
    """Walker base (sim coordinates).  L0/L1: the base tracks the pelvis with a first-order lag `tau` ('L0 transparent:
    walker follows', SPEC 3), the pelvis may sit at most `cap` m off the base centre (belt/SEA travel).  From the first
    L2 sample the base is braked and only dragged beyond 5 cm (same logic as render/animate.py).
    Returns base (N,2) and pelvis-relative-to-base (N,2)."""
    n = len(tr.t)
    base, rel = np.zeros((n, 2)), np.zeros((n, 2))
    prev = tr.com[0].copy()
    a = 1 - np.exp(-tr.dt / tau)
    for k in range(n):
        c, lv = tr.com[k], tr.level[k]
        if lv >= 2:
            off = c - prev
            nrm = np.linalg.norm(off)
            b = prev if nrm <= 0.05 else c - off * (0.05 / nrm)
        else:
            b = prev + a * (c - prev)
            off = c - b
            nrm = np.linalg.norm(off)
            if nrm > cap:
                b = c - off * (cap / nrm)
        base[k], prev = b, b
        rel[k] = c - b
    return base, rel


def play_times(T, slow=SLOW):
    """Sim-time of each output frame: 1x until T_SLOW0, `slow` until T_SLOW1, 1x afterwards, then HOLD s freeze."""
    ts, t, dt = [], 0.0, 1.0 / FPS
    while t < T:
        ts.append(t)
        t += dt * (slow if T_SLOW0 <= t < T_SLOW1 else 1.0)
    n_play = len(ts)
    ts += [T] * int(HOLD * FPS)
    return np.array(ts), n_play


# ------------------------------------------------------------------ humanoid driver
LEGS = ("left", "right")
LEG_J = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
Q_NEUTRAL = np.array([-0.12, 0.0, 0.0, 0.25, -0.13, 0.0])


class Driver:
    def __init__(self, rig):
        self.r = rig
        m = rig.model
        self.m, self.d = m, rig.data
        self.leg_q = {s: np.array([rig.qadr[f"{s}_{j}_joint"] for j in LEG_J]) for s in LEGS}
        self.leg_v = {s: np.array([rig.dadr[f"{s}_{j}_joint"] for j in LEG_J]) for s in LEGS}
        self.lo = {s: m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{j}_joint") for j in LEG_J]] for s in LEGS}
        self.site = {s: rig.site[f"foot_{s}"] for s in LEGS}
        self.hip_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_pitch_link") for s in LEGS}
        self.q = {s: Q_NEUTRAL.copy() for s in LEGS}
        for s_ in LEGS:                          # visual-only: allow 60 deg of hip adduction (crossover steps)
            self.lo[s_][1] = [-1.05, 2.97] if s_ == "left" else [-2.97, 1.05]
            self.lo[s_][5] = [-0.8, 0.8]         # ankle roll too (flat foot under a strongly adducted leg)
        # straight-leg reach (hip joint -> sole-centre site)
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        self.reach = float(np.linalg.norm(d.site_xpos[self.site["left"]] - d.xpos[self.hip_body["left"]]))
        self.first = True

    def _base(self, pos, quat):
        q = self.d.qpos
        q[0:3] = pos
        q[3:7] = quat

    def pose(self, com_xy, z0, foot_l, foot_r, h_l, h_r, lean_dir, lean_ang, fall_extra=0.0):
        """Place the humanoid.  All positions are WORLD (walker) coordinates.  Returns (belt_centre, pelvis_pos)."""
        r, d, m = self.r, self.d, self.m
        pel_z = z0 + 0.075                   # G1 pelvis-frame height (slight knee bend); belt sits BELT_BELOW_PELVIS lower
        ang = float(lean_ang)
        # whole-body tilt about the horizontal axis z x lean_dir : 0.4 at the pelvis, 0.6 in the waist joints
        ax = np.array([-lean_dir[1], lean_dir[0], 0.0])
        a_p = 0.4 * ang
        quat = np.array([np.cos(a_p / 2), *(np.sin(a_p / 2) * ax)])
        # waist joints (pelvis yaw = 0): pitch about y, roll about x
        pitch = 0.6 * ang * lean_dir[0]
        roll = -0.6 * ang * lean_dir[1]
        self._arms_waist(pitch, roll)
        tgt = {"left": np.array([foot_l[0], foot_l[1], h_l]), "right": np.array([foot_r[0], foot_r[1], h_r])}
        # lower the pelvis if a foot is out of reach (lunge / wide step)
        z = pel_z
        for _ in range(8):
            self._base([com_xy[0], com_xy[1], z], quat)
            mujoco.mj_kinematics(m, d)
            worst = 0.0
            for s in LEGS:
                dist = np.linalg.norm(tgt[s] - d.xpos[self.hip_body[s]])
                worst = max(worst, dist - 0.96 * self.reach)
            if worst <= 0:
                break
            z -= worst * 1.1
        self._base([com_xy[0], com_xy[1], z], quat)
        self._ik(tgt)
        return np.array([com_xy[0], com_xy[1], z - BELT_BELOW_PELVIS]), z

    def _arms_waist(self, pitch, roll):
        r, d = self.r, self.d
        q = d.qpos
        q[r.qadr["waist_yaw_joint"]] = 0
        q[r.qadr["waist_pitch_joint"]] = np.clip(pitch, -0.5, 0.5)
        q[r.qadr["waist_roll_joint"]] = np.clip(roll, -0.5, 0.5)
        for s, sg in (("left", 1), ("right", -1)):
            q[r.qadr[f"{s}_shoulder_pitch_joint"]] = 0.12
            q[r.qadr[f"{s}_shoulder_roll_joint"]] = sg * 0.28
            q[r.qadr[f"{s}_shoulder_yaw_joint"]] = 0.0
            q[r.qadr[f"{s}_elbow_joint"]] = 1.25
            q[r.qadr[f"{s}_wrist_roll_joint"]] = 0.0
            q[r.qadr[f"{s}_wrist_pitch_joint"]] = 0.0
            q[r.qadr[f"{s}_wrist_yaw_joint"]] = 0.0

    def _err(self, tgt):
        """Task error vectors (pos 3 + rot 3) per leg at the current qpos; also refreshes kinematics."""
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        out = {}
        for s in LEGS:
            sid = self.site[s]
            R = d.site_xmat[sid].reshape(3, 3)       # flat foot: R -> I
            er = 0.5 * (np.cross(R[:, 0], [1, 0, 0]) + np.cross(R[:, 1], [0, 1, 0]) + np.cross(R[:, 2], [0, 0, 1]))
            out[s] = np.concatenate([tgt[s] - d.site_xpos[sid], er])
        return out

    def _ik(self, tgt, iters=40):
        """Damped-least-squares IK (Levenberg-Marquardt damping, step cap, backtracking line search, joint limits)."""
        m, d = self.m, self.d
        if self.first:
            self.q = {s: Q_NEUTRAL.copy() for s in LEGS}
            iters = 120
            self.first = False
        W = np.diag([1, 1, 1, 0.35, 0.35, 0.05])
        lam, mu = 1e-4, 2e-5
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

        def cost(e):
            return sum(float(e[s] @ W @ e[s]) for s in LEGS)

        for s in LEGS:
            d.qpos[self.leg_q[s]] = self.q[s]
        e = self._err(tgt)
        lm = 0.0
        for _ in range(iters):
            q0 = {s: self.q[s].copy() for s in LEGS}
            for s in LEGS:
                mujoco.mj_jacSite(m, d, jp, jr, self.site[s])
                J = np.vstack([jp[:, self.leg_v[s]], jr[:, self.leg_v[s]]])
                A = J.T @ W @ J + (lam + mu + lm) * np.eye(6)
                b = J.T @ W @ e[s] - mu * (q0[s] - Q_NEUTRAL)
                dq = np.clip(np.linalg.solve(A, b), -0.25, 0.25)
                self.q[s] = np.clip(q0[s] + dq, self.lo[s][:, 0], self.lo[s][:, 1])
                d.qpos[self.leg_q[s]] = self.q[s]
            e_new = self._err(tgt)
            if cost(e_new) < cost(e):                      # Levenberg-Marquardt accept / reject
                e = e_new
                lm *= 0.3
            else:
                for s in LEGS:
                    self.q[s] = q0[s]
                    d.qpos[self.leg_q[s]] = q0[s]
                e = self._err(tgt)
                lm = max(lm * 10, 1e-3)
            if max(np.linalg.norm(e[s][:3]) for s in LEGS) < 3e-4:
                break
        self.last_err = max(float(np.linalg.norm(e[s][:3])) for s in LEGS)


# ------------------------------------------------------------------ one frame
def lean_of(tr, k, z0):
    fl, fr, com = tr.foot_l[k], tr.foot_r[k], tr.com[k]
    sw = int(tr.swing[k])
    sup = fr if sw == 1 else fl if sw == 2 else 0.5 * (fl + fr)
    dvec = com - sup
    nd = float(np.linalg.norm(dvec))
    ang = float(np.clip(1.3 * nd / z0, 0.0, 0.55))
    return (dvec / nd if nd > 1e-6 else np.array([1.0, 0.0])), ang


def draw_state(rig, drv, tr, k, ts):
    """Pose the whole scene for sample k of trajectory tr (sim time ts)."""
    off = PELVIS_XY0
    z0 = tr.z0
    com = tr.com[k] + off
    fl, fr = tr.foot_l[k] + off, tr.foot_r[k] + off
    sw = int(tr.swing[k])
    h = float(tr.swing_h[k]) * SWING_H_GAIN
    ldir, lang = lean_of(tr, k, z0)
    belt_c, pel_z = drv.pose(com, z0, fl, fr, h if sw == 1 else 0.0, h if sw == 2 else 0.0, ldir, lang)
    rig.set_walker(tr.base[k])
    rig.set_belt(belt_c)
    shift = belt_c - np.array(rig.meta["belt"]["centre"])
    lv = int(tr.level[k])
    grey = np.array([*hex2rgb(LEVEL_HEX[0]), 1.0])
    for j, name in enumerate(tr.cable_names):
        w = float(tr.taut[k, j])
        tgt_col = np.array([*hex2rgb(LEVEL_HEX[max(int(tr.colour[k, j]), 1)]), 1.0])
        if name.startswith("strap"):
            base = np.array([0.17, 0.23, 0.33, 1.0])
            thin, thick = 0.0060, 0.0110
        else:
            base, thin, thick = grey, 0.0045, 0.0105
        col = base + (tgt_col - base) * w
        rig.set_cable(name, 1.0 - w, col, thin + (thick - thin) * w, shift)
    # outriggers
    u = 0.0
    if tr.t_l2 is not None:
        x = np.clip((ts - tr.t_l2) / 0.28, 0, 1)
        u = x * x * (3 - 2 * x)
    for i in range(4):
        rig.set_outrigger(i, u)
    # force arrow (robot force on the pelvis)
    F = tr.F[k]
    nf = float(np.linalg.norm(F))
    if nf > 5.0:
        uF = F / nf
        L = float(np.clip(0.0028 * nf, 0.14, 0.42))
        tip = com - uF * 0.19
        tail = np.array([tip[0] - uF[0] * L, tip[1] - uF[1] * L, belt_c[2] + 0.14])
        rig.set_arrow(tail, np.array([uF[0] * L, uF[1] * L, 0.0]), (*hex2rgb(LEVEL_HEX[max(lv, 1)]), 1.0))
    else:
        rig.set_arrow(None)
    # capture point ring
    rig.set_ring(tr.xi[k] + off, (*hex2rgb(RING_HEX[lv]), 0.95))


# ------------------------------------------------------------------ overlay
def _f(size, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_R, size)


def overlay(img, tr, ctrl, k, ts, slow, ended, scale=1.0, panel=None, info=None, disclaimer=True, outcome_bl=False):
    """Draw the HUD on a scene image.  With disclaimer=True a footer band (46*scale px) is appended below the image."""
    W, H = img.shape[1], img.shape[0]
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    s = scale
    ink = (30, 37, 48, 255)
    lv = int(tr.level[k])
    f1, f2, f3 = _f(int(30 * s)), _f(int(25 * s)), _f(int(22 * s), False)
    title = panel or CTRL_NAME.get(ctrl, ctrl)
    x0, y0 = int(18 * s), int(16 * s)
    bw = int(240 * s)
    tag = f"t = {ts:4.2f} s" + (f"  ({slow:g}x slow)" if slow != 1 else "")
    cw = max(dr.textlength(title, font=f1) + 44 * s, 24 * s + bw + 14 * s + dr.textlength(tag, font=f3) + 16 * s)
    dr.rounded_rectangle([x0, y0, x0 + cw, y0 + int(112 * s)], int(12 * s), fill=(255, 255, 255, 228))
    if ctrl in CTRL_HEX:
        c = tuple(int(v * 255) for v in hex2rgb(CTRL_HEX[ctrl]))
        dr.rounded_rectangle([x0, y0, x0 + int(10 * s), y0 + int(112 * s)], int(5 * s), fill=c + (255,))
    dr.text((x0 + int(24 * s), y0 + int(8 * s)), title, font=f1, fill=ink)
    col = tuple(int(v * 255) for v in hex2rgb(LEVEL_HEX[lv]))
    dr.rounded_rectangle([x0 + int(24 * s), y0 + int(54 * s), x0 + int(24 * s) + bw, y0 + int(98 * s)], int(10 * s), fill=col + (255,))
    dr.text((x0 + int(36 * s), y0 + int(58 * s)), LEVEL_NAME[lv], font=f2, fill=(255, 255, 255, 255))
    dr.text((x0 + int(24 * s) + bw + int(14 * s), y0 + int(62 * s)), tag, font=f3, fill=ink)
    F = float(np.linalg.norm(tr.F[k]))
    if F > 5:
        dr.rounded_rectangle([x0, y0 + int(122 * s), x0 + int(190 * s), y0 + int(164 * s)], int(10 * s), fill=(255, 255, 255, 228))
        dr.text((x0 + int(14 * s), y0 + int(127 * s)), f"|F| {F:3.0f} N", font=f2, fill=ink)
    if info:
        ft = _f(int(21 * s), False)
        tw = dr.textlength(info, font=ft)
        dr.rounded_rectangle([W - tw - int(40 * s), y0, W - int(16 * s), y0 + int(44 * s)], int(10 * s), fill=(255, 255, 255, 228))
        dr.text((W - tw - int(28 * s), y0 + int(9 * s)), info, font=ft, fill=ink)
    if ended:
        oc = tr.outcome
        col = tuple(int(v * 255) for v in hex2rgb(OUT_HEX[oc]))
        fo = _f(int(32 * s))
        txt = f"Outcome: {oc}"
        tw = dr.textlength(txt, font=fo)
        oy = (H - int(76 * s)) if outcome_bl else (y0 + int(54 * s))
        ox = int(16 * s) if outcome_bl else (W - tw - int(52 * s))
        dr.rounded_rectangle([ox, oy, ox + tw + int(36 * s), oy + int(58 * s)], int(12 * s), fill=col + (255,))
        dr.text((ox + int(18 * s), oy + int(8 * s)), txt, font=fo, fill=(255, 255, 255, 255))
    out = np.asarray(im)
    if disclaimer:
        out = np.concatenate([out, footer(W, int(46 * s), DISCLAIMER, s)], axis=0)
    return out


def footer(W, h, text, s=1.0):
    band = Image.new("RGB", (W, h), (244, 246, 248))
    dr = ImageDraw.Draw(band)
    dr.line([(0, 0), (W, 0)], fill=(205, 210, 217), width=2)
    size = int(22 * s)
    ft = _f(size, False)
    while dr.textlength(text, font=ft) > W - 40 and size > 10:
        size -= 1
        ft = _f(size, False)
    tw = dr.textlength(text, font=ft)
    dr.text(((W - tw) / 2, (h - size) / 2 - 2), text, font=ft, fill=(60, 68, 80))
    return np.asarray(band)


def push_info(tr):
    return f"push {np.linalg.norm(tr.dv):.2f} m/s toward weak side"


# ------------------------------------------------------------------ video / gif
def write_mp4(frames, path):
    with imageio.get_writer(path, fps=FPS, codec="libx264", pixelformat="yuv420p", quality=8, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)


def make_gif(mp4, gif, width=960, fps=15):
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, "-loop", "0", gif], check=True)
    print(gif, f"{os.path.getsize(gif) / 1e6:.2f} MB")


def new_rig(w, h):
    rig = Rig(w, h)
    cam = os.environ.get("CAM")          # debug: CAM="x,y,z,lx,ly,lz" overrides the camera
    if cam:
        v = [float(x) for x in cam.split(",")]
        return rig, Driver(rig), rig.cam_from(v[:3], v[3:])
    return rig, Driver(rig), rig.cam_from(CAM_POS, CAM_LOOK)


def render_state(rig, drv, cam, tr, ts, **ov):
    k = tr.k_at(ts)
    draw_state(rig, drv, tr, k, min(ts, tr.T))
    img = rig.render(cam)
    ended = ts >= tr.t[-1] - 1e-9
    slow = SLOW if T_SLOW0 <= ts < T_SLOW1 else 1
    return overlay(img, tr, ov.pop("ctrl", tr.name), k, ts, slow, ended, **ov)


def run_scenario(npz, out_mp4, title):
    tr = Traj(npz)
    rig, drv, cam = new_rig(1280, 674)                  # + 46 px footer band = 1280x720
    ts, n_play = play_times(tr.T)
    frames = []
    for i, t in enumerate(ts):
        frames.append(render_state(rig, drv, cam, tr, float(t), ctrl="CP", panel=title, info=push_info(tr)))
    write_mp4(frames, out_mp4)
    print(out_mp4, f"{len(frames) / FPS:.1f} s, {len(frames)} frames, outcome {tr.outcome}")
    make_gif(out_mp4, out_mp4[:-4] + ".gif")


SCEN = [("cp_small", "CP  small push  (self-recovery)"), ("cp_bigger", "CP  bigger push  (graded assist)"),
        ("cp_big", "CP  too big a push  (catch)")]


def run_compare(out_mp4):
    names = ["B0", "B1", "B2", "CP"]
    trs = [Traj(os.path.join(ROOT, "results", f"showcase_{n}.npz"), n) for n in names]
    pw, ph = 960, 510                                   # 2 x 2 panels = 1920 x 1020, + 60 px footer = 1080
    rig, drv0, cam = new_rig(pw, ph)
    drvs = [drv0] + [Driver(rig) for _ in range(3)]       # one warm-started IK state per panel
    T = max(t.T for t in trs)
    ts, n_play = play_times(T)
    txt = DISCLAIMER + f"   |   same patient (severity 0.6, left leg paretic), same push {np.linalg.norm(trs[0].dv):.2f} m/s toward the weak side"
    foot = footer(1920, 60, txt, 1.15)
    frames = []
    for t in ts:
        t = float(t)
        panels = [render_state(rig, drv, cam, tr, t, ctrl=tr.name, scale=0.75, info=None, disclaimer=False)
                  for tr, drv in zip(trs, drvs)]
        full = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0).copy()
        im = Image.fromarray(full)
        dr = ImageDraw.Draw(im)
        dr.line([(960, 0), (960, ph * 2)], fill=(255, 255, 255), width=6)
        dr.line([(0, ph), (1920, ph)], fill=(255, 255, 255), width=6)
        frames.append(np.concatenate([np.asarray(im), foot], axis=0))
    write_mp4(frames, out_mp4)
    print(out_mp4, f"{len(frames) / FPS:.1f} s, {len(frames)} frames")
    make_gif(out_mp4, out_mp4[:-4] + ".gif")


def run_still(out_png):
    rig, drv, cam = new_rig(1280, 674)
    pics = []
    for name, title in SCEN:
        tr = Traj(os.path.join(HERE, "data", name + ".npz"))
        if name == "cp_small":
            ts = float(tr.t[int(np.argmax(np.linalg.norm(tr.com, axis=1)))])
        elif name == "cp_bigger":
            ts = float(tr.t[int(np.argmax(np.linalg.norm(tr.F, axis=1)))])
        else:
            ts = (tr.t_l2 or 1.0) + 0.55
        drv.first = True
        k = tr.k_at(ts)
        draw_state(rig, drv, tr, k, ts)
        scene = rig.render(cam)[:, 280:1000]             # crop to the walker (the 16:9 frame is mostly empty floor)
        img = overlay(scene, tr, "CP", k, ts, 1, True, scale=0.82, panel=title, info=None, disclaimer=False, outcome_bl=True)
        pics.append(img)
    sep = np.full((pics[0].shape[0], 8, 3), 255, np.uint8)
    strip = np.concatenate([pics[0], sep, pics[1], sep, pics[2]], axis=1)
    strip = np.concatenate([strip, footer(strip.shape[1], 46, DISCLAIMER, 1.1)], axis=0)
    Image.fromarray(strip).save(out_png)
    print(out_png, strip.shape)


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "all"
    if cmd == "frame":
        path, t, out = a[1], float(a[2]), a[3]
        w, h = (int(a[4]), int(a[5])) if len(a) > 5 else (1280, 674)
        tr = Traj(path, os.path.splitext(os.path.basename(path))[0].replace("showcase_", ""))
        rig, drv, cam = new_rig(w, h)
        # warm up the IK from rest, then jump
        img = render_state(rig, drv, cam, tr, t, ctrl=tr.name if tr.name in CTRL_NAME else "CP", info=push_info(tr))
        imageio.imwrite(out, img)
        print(out, "ik residual", drv.last_err)
    else:
        if cmd in ("scen", "all"):
            for name, title in SCEN:
                run_scenario(os.path.join(HERE, "data", name + ".npz"), os.path.join(HERE, name.replace("cp_", "cp_") + ".mp4"), title)
        if cmd in ("compare", "all"):
            run_compare(os.path.join(HERE, "compare_4controllers.mp4"))
        if cmd in ("still", "all"):
            run_still(os.path.join(HERE, "cp_three_states.png"))
