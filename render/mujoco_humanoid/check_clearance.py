"""Geometry sanity checks for the kinematic replay (humanoid vs walker / belt / floor, IK residual, foot height).

usage: python check_clearance.py [npz ...]   (default: the 4 showcase files + the 3 CP scenarios)
Prints, per trajectory (every 10th sample): max IK residual, lowest humanoid vertex z, worst penetration depth of humanoid
vertices into walker part bounding boxes (box-like parts only) and into the pelvic belt tube.
"""
import glob
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay as R                                                    # noqa: E402
import build_scene as bs                                              # noqa: E402

BOXY = ("base_rail", "base_crossbar", "column_", "top_beam", "bws_mast", "bws_cantilever", "post_", "ballast_block",
        "battery_box", "spool_", "winch_housing", "bws_beam_cap", "bws_spreader")


def human_vertices(rig):
    m, d = rig.model, rig.data
    out = []
    root = rig.pelvis_body
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        b = m.geom_bodyid[g]
        # human bodies are descendants of the pelvis
        bb = b
        while bb > 0 and bb != root:
            bb = m.body_parentid[bb]
        if bb != root:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n][::3]
        out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(out)


def main(files):
    rig, drv, cam = R.new_rig(320, 180)
    boxes = []
    for p in rig.meta["parts"]:
        if p["name"].startswith(BOXY):
            lo, hi = np.array(p["bounds"])
            boxes.append((p["name"], lo, hi))
    hb = rig.meta["belt"]
    for f in files:
        tr = R.Traj(f)
        drv.first = True
        worst_ik, minz, pen_w, pen_b, who, depth_b = 0.0, 9.0, 0.0, 0.0, "", 0.0
        for k in range(0, len(tr.t), 15):
            R.draw_state(rig, drv, tr, k, float(tr.t[k]))
            mujoco.mj_forward(rig.model, rig.data)
            worst_ik = max(worst_ik, drv.last_err)
            V = human_vertices(rig)
            minz = min(minz, float(V[:, 2].min()))
            sh = rig.data.mocap_pos[rig.mocap["walker"]]
            for nm, lo0, hi0 in boxes:
                if nm == "bws_spreader":
                    lo0, hi0 = lo0 + [0, 0, bs.SPREADER_DZ], hi0 + [0, 0, bs.SPREADER_DZ]
                lo, hi = lo0 + sh, hi0 + sh
                inside = np.all((V > lo + 0.004) & (V < hi - 0.004), axis=1)
                if inside.any():
                    depth = float(np.min(np.minimum(V[inside] - lo, hi - V[inside])))
                    # depth of the deepest vertex (min over axes of distance to the nearest face)
                    dd = np.max(np.min(np.minimum(V[inside] - lo, hi - V[inside]), axis=1))
                    if dd > pen_w:
                        pen_w, who = float(dd), f"{nm}@t={tr.t[k]:.2f}"
            # belt tube: superellipse ring, centre = belt mocap
            c = rig.data.mocap_pos[rig.mocap["belt"]] + np.array(hb["centre"])
            rel = V - c
            band = np.abs(rel[:, 2]) < hb["h"] / 2
            e = 2.6
            fin = (np.abs(rel[band, 0]) / (hb["hd"] - hb["t"] / 2)) ** e + (np.abs(rel[band, 1]) / (hb["hw"] - hb["t"] / 2)) ** e
            fout = (np.abs(rel[band, 0]) / (hb["hd"] + hb["t"] / 2)) ** e + (np.abs(rel[band, 1]) / (hb["hw"] + hb["t"] / 2)) ** e
            bad = (fin > 1.0) & (fout < 1.0)
            if bad.any():
                pen_b = max(pen_b, float(bad.sum()))
                r_xy = np.hypot(rel[band, 0], rel[band, 1])[bad]
                depth_b = max(depth_b, float(np.max(r_xy * (1 - 1 / fin[bad] ** (1 / e)))))
        print(f"{tr.name:10s} ik_resid_max {worst_ik * 100:5.2f} cm | lowest human vertex z {minz * 100:5.1f} cm | "
              f"deepest walker-box penetration {pen_w * 100:4.1f} cm {who} | belt-tube vertices inside (max/frame) {pen_b:.0f}, deepest {depth_b * 100:.1f} cm")


if __name__ == "__main__":
    fs = sys.argv[1:] or sorted(glob.glob(os.path.join(R.ROOT, "results", "showcase_*.npz"))) + \
        sorted(glob.glob(os.path.join(HERE, "data", "cp_*.npz")))
    main(fs)
