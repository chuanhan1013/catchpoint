"""Render the push x controller trials: raw frames (no text) -> videos/raw/{push}_{ctrl}.mp4 + per-frame metadata
json.  Text overlays and composites are added by compose.py.

usage: python render.py [push ...]   (default: a b c, all controllers)"""
import json
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

from g1cp import CONTRACT, Sim
from trials import DUR, PUSHES, T0, count_steps

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "videos", "raw")
W, H, FPS = 1280, 720, 30
T_START, T_END = 1.0, 7.0

GREY = np.array([0.42, 0.48, 0.56, 1.0])
AMBER = np.array([0.88, 0.63, 0.19, 1.0])
RED = np.array([0.78, 0.27, 0.23, 1.0])
TEAL = np.array([0.12, 0.44, 0.47, 1.0])
FRAME = np.array([0.55, 0.58, 0.62, 1.0])
FRAME_D = np.array([0.33, 0.36, 0.40, 1.0])


def add(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else mat).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def seg(scn, a, b, r, rgba, gtype=mujoco.mjtGeom.mjGEOM_CAPSULE):
    g = add(scn, gtype, [r, r, r], [0, 0, 0], rgba)
    if g is not None:
        mujoco.mjv_connector(g, gtype, r, np.asarray(a, float), np.asarray(b, float))
    return g


def tension_rgba(T, Tslip):
    if T < 0.5:
        return np.array([0.62, 0.65, 0.70, 0.85])
    f = min(T / Tslip, 1.0)
    c = AMBER * (1 - f) + RED * f
    c[3] = 1.0
    return c


def draw_walker(scn, sim, st):
    """Walker frame: scaled D3 contract geometry (s = 0.82) placed so the belt is at the G1 pelvis."""
    s = sim.s
    pel = np.array(CONTRACT["pelvis"])
    o = np.array([st["walker"][0], st["walker"][1], 0.0])

    def P(x, y, z):
        return o + s * (np.array([x, y, z]) - np.array([pel[0], pel[1], 0.0]))
    xr, xf, yc, yl = 0.03, 0.90, 0.33, 0.40
    zb = 0.14
    r = 0.022 * s
    # base U (open at the front): side rails + rear crossbar
    for y in (yl, -yl):
        seg(scn, P(-0.0, y, zb), P(0.95, y, zb), r * 1.3, FRAME_D)
    seg(scn, P(0.0, yl, zb), P(0.0, -yl, zb), r * 1.3, FRAME_D)
    # wheels
    for x in (0.075, 0.875):
        for y in (0.325, -0.325):
            add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.075 * s, 0.02, 0], P(x, y, 0.075),
                np.array([0.15, 0.15, 0.17, 1]), mat=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float))
    # corner posts to z = 1.2 and top side rails
    for x in (xr, xf):
        for y in (yc, -yc):
            seg(scn, P(x, y if abs(y) < yl else y, zb), P(x, y, 1.2), r, FRAME)
            seg(scn, P(x, y, zb), P(x, yl * np.sign(y), zb), r, FRAME)
    for y in (yc, -yc):
        seg(scn, P(xr, y, 1.2), P(xf, y, 1.2), r, FRAME)
    seg(scn, P(xr, yc, 1.2), P(xr, -yc, 1.2), r, FRAME)
    # lateral anchor posts (x 0.45, y +-0.40, z 0.95)
    for y in (yl, -yl):
        seg(scn, P(0.45, y, zb), P(0.45, y, 0.95), r * 0.8, FRAME)
    # rear-centre crossbar at 0.95 (front-centre swing-away post: drawn as a short post)
    seg(scn, P(xr, yc, 0.95), P(xr, -yc, 0.95), r * 0.8, FRAME)
    ghost = np.array([0.55, 0.58, 0.62, 0.35])      # front-centre swing-away post: translucent so legs stay visible
    seg(scn, P(xf, 0.0, zb), P(xf, 0.0, 0.95), r * 0.7, ghost)
    seg(scn, P(xf, 0.0, zb), P(xf, 0.12, zb), r * 0.7, ghost)
    # overhead BWS gantry to z = 1.8
    for y in (yc, -yc):
        seg(scn, P(xr, y, 1.2), P(xr, y * 0.5, 1.8), r, FRAME)
    seg(scn, P(xr, yc * 0.5, 1.8), P(xr, -yc * 0.5, 1.8), r, FRAME)
    seg(scn, P(xr, 0.0, 1.8), P(0.42, 0.0, 1.8), r, FRAME)
    add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.06 * s, 0.05 * s, 0.04 * s], P(0.42, 0, 1.8), FRAME_D)
    # anchors (spool housings)
    for a in st["A"]:
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], a, FRAME_D)


def draw_cables(scn, sim, st):
    att, A, T = st["att"], st["A"], st["T"]
    for i in range(len(A) if sim.ctrl != "B0" else 0):     # B0: no CatchPoint layer -> no cables
        seg(scn, att[i], A[i], 0.0045 if T[i] < 0.5 else 0.0065, tension_rgba(T[i], st["Tslip"][i]))
    # belt: ellipse around the pelvis through the attachment points
    pel = st["pel"]
    yaw = sim.yaw()
    c, s_ = np.cos(yaw), np.sin(yaw)
    pts = []
    for th in np.linspace(0, 2 * np.pi, 25):
        lx, ly = sim.walker.belt_a * np.cos(th), sim.walker.belt_b * np.sin(th)
        pts.append([pel[0] + c * lx - s_ * ly, pel[1] + s_ * lx + c * ly, pel[2]])
    for i in range(24):
        seg(scn, pts[i], pts[i + 1], 0.014, TEAL)
    # BWS harness line
    bws = st["bws"]
    if sim.ctrl == "B0":
        return
    rg = RED if st["Fbws"] > 1 else np.array([0.62, 0.65, 0.70, 0.7])
    seg(scn, [pel[0], pel[1], pel[2] + 0.02], bws, 0.004 if st["Fbws"] <= 1 else 0.007, rg)


def draw_arrows(scn, sim, st, push_vis):
    pel = st["pel"]
    if push_vis is not None:
        d = np.array([*push_vis, 0.0])
        a = pel + np.array([0, 0, 0.12]) - d * 0.55
        b = pel + np.array([0, 0, 0.12]) - d * 0.12
        seg(scn, a, b, 0.018, np.array([0.85, 0.15, 0.15, 1]))
        # head
        g = add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.04, 0.04, 0.1], [0, 0, 0], np.array([0.85, 0.15, 0.15, 1]))
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.05, b - d * 0.06, b + d * 0.02)
    F = st["F"][:2]
    n = np.hypot(*F)
    if n > 2.0 and st["level"] > 0:
        L = min(0.0035 * n / sim.lam, 0.9)
        u = np.array([*(F / n), 0.0])
        col = AMBER if st["level"] == 1 else RED
        g = add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.03, 0.03, L], [0, 0, 0], col)
        if g is not None:
            p0 = pel + np.array([0, 0, -0.08])
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.035, p0, p0 + u * L)


def render_trial(pk, ctrl, out_dir=RAW, policy=None, tag="", simkw=None):
    os.makedirs(out_dir, exist_ok=True)
    push = dict(t0=T0, dur=DUR, F=PUSHES[pk], dir=(0, -1))
    sim = Sim(ctrl=ctrl, T=T_END, push=push, record=True, policy=policy, **(simkw or {}))
    rnd = mujoco.Renderer(sim.m, H, W)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 152.0, -9.0, 2.55
    look = None
    path = os.path.join(out_dir, f"{pk}_{ctrl}{tag}.mp4")
    wr = imageio.get_writer(path, fps=FPS, codec="libx264", quality=None,
                            ffmpeg_params=["-crf", "12", "-preset", "medium"])
    meta = []
    n = int(round(T_END / 0.002))
    next_t = T_START
    push_dir = np.array(push["dir"], float)
    for i in range(n):
        sim.step()
        t = sim.t()
        if t + 1e-9 < next_t:
            continue
        next_t += 1.0 / FPS
        st = sim.cur
        tgt = np.array([st["pel"][0] * 0.5 + st["walker"][0] * 0.5, st["pel"][1] * 0.5 + st["walker"][1] * 0.5, 0.72])
        look = tgt if look is None else look + 0.15 * (tgt - look)
        cam.lookat[:] = look
        rnd.update_scene(sim.d, camera=cam, scene_option=opt)
        scn = rnd.scene
        draw_walker(scn, sim, st)
        draw_cables(scn, sim, st)
        pv = push_dir if (T0 - 0.15 <= t < T0 + DUR + 0.25) else None
        draw_arrows(scn, sim, st, pv)
        img = rnd.render()
        wr.append_data(img)
        Fxy = float(np.hypot(*st["F"][:2]))
        meta.append(dict(t=round(t, 4), level=int(st["level"]), F=round(Fxy, 1), F_h=round(Fxy / sim.lam, 1),
                         Tmax=round(float(np.max(st["T"])), 1), catch_on=bool(st["catch_on"]),
                         fallen=bool(sim.fall_t is not None and t >= sim.fall_t), push=pv is not None))
    wr.close()
    r = sim.result()
    L = r["log"]
    info = dict(push=pk, ctrl=ctrl, F_push=PUSHES[pk], dur=DUR, t0=T0, dv=PUSHES[pk] * DUR / sim.mass,
                outcome=r["outcome"], fall_t=r["fall_t"], catch_t=r["catch_t"], peakFxy=round(r["peakFxy"], 1),
                peakFxy_h=round(r["peakFxy"] / sim.lam, 1), impulse=round(r["impulse"], 2),
                impulse_h=round(r["impulse"] / sim.lam, 2), steps=count_steps(L, L["t"] > T0), lam=sim.lam,
                mass=sim.mass, frames=meta)
    info["policy"] = policy or "pretrained motion.pt"
    info["simkw"] = {k: float(v) for k, v in (simkw or {}).items()}
    with open(os.path.join(out_dir, f"{pk}_{ctrl}{tag}.json"), "w") as f:
        json.dump(info, f)
    return {k: v for k, v in info.items() if k != "frames"}


def _job(a):
    return render_trial(*a)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    pushes = sys.argv[1:] or list(PUSHES)
    ctrls = ["B0", "B1", "CP"]
    jobs = [(p, c) for p in pushes for c in ctrls]
    with ProcessPoolExecutor(min(9, len(jobs))) as ex:
        for r in ex.map(_job, jobs):
            print(r)
