"""MuJoCo scene for the CatchPoint D3 walker + a Unitree G1 humanoid (kinematic only, visualization).

    Rig(width, height)  -> MjModel / MjData / Renderer with
        static walker meshes (cad/parts_d3, SPEC 4 / d3_contract.json coordinates),
        pelvic belt (mocap, follows the pelvis),
        8 cable tendons + 2 BWS strap tendons (site -> sag-site -> belt site; sag site is a mocap marker),
        4 outriggers (mocap capsules, retracted <-> deployed),
        G1 humanoid scaled by G1_SCALE (free joint + 29 hinge joints, posed by qpos, never stepped),
        force arrow + capture-point ring (mocap).
    python build_scene.py        -> builds the model once and prints a summary (smoke test)

Frame: metres, x forward, y left, z up (same as the CAD).  Simulation (LIP) coordinates are shifted by SIM_ORIGIN so
that the quiet-stance pelvis sits at the contract pelvis (0.42, 0).
"""
import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CAD = os.path.join(ROOT, "cad")
G1_DIR = os.path.join(HERE, "g1")

G1_SCALE = 1.30                     # G1 is 1.27 m; x1.30 -> 1.65 m, straight-leg pelvis-frame height 1.02 m
PELVIS_XY0 = np.array([0.42, 0.0])  # contract pelvis (x, y); sim origin (0, 0) maps here
BELT_BELOW_PELVIS = 0.010 * G1_SCALE  # belt centre sits this far below the G1 pelvis frame origin
LEVEL_HEX = {0: "#6B7A8F", 1: "#E0A030", 2: "#C8443A"}
RING_HEX = {0: "#3FAE5C", 1: "#E0A030", 2: "#C8443A"}      # capture-point ring: green / amber / red
NO_STATIC = ("cable_", "pelvic_belt", "belt_buckle", "bws_strap_", "outrigger_", "bws_cable", "bws_spreader")
SPREADER_DZ = 0.065                 # spreader bar raised from 1.68 to 1.745 m so the 1.70 m humanoid clears it
CABLES = ["RL", "RR", "FL", "FR", "LL", "LR", "CR", "CF"]


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def quat_z_to(d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    c = d[2]
    if c > 1 - 1e-9:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9:
        return np.array([0.0, 1, 0, 0])
    ax = np.cross([0, 0, 1.0], d)
    s = np.sqrt((1 + c) * 2)
    q = np.array([s / 2, ax[0] / s, ax[1] / s, ax[2] / s])
    return q / np.linalg.norm(q)


def _scl(txt, s):
    return " ".join(f"{float(v) * s:.7g}" for v in txt.split())


# ------------------------------------------------------------------ G1 (scaled, visual only)
def g1_xml(scale):
    """Return (asset_xml, body_xml, mesh_files{name: bytes}) from the Menagerie g1.xml, uniformly scaled."""
    root = ET.parse(os.path.join(G1_DIR, "g1.xml")).getroot()
    files, assets = {}, []
    for m in root.find("asset").findall("mesh"):
        f = m.get("file")
        name = m.get("name") or os.path.splitext(f)[0]
        key = "g1_" + f
        files[key] = open(os.path.join(G1_DIR, "assets", f), "rb").read()
        assets.append(f'<mesh name="{name}" file="{key}" scale="{scale} {scale} {scale}"/>')
    pelvis = root.find("worldbody").find("body")

    def walk(b):
        if b.get("pos"):
            b.set("pos", _scl(b.get("pos"), scale))
        for ch in list(b):
            if ch.tag == "geom":
                if ch.get("class") in ("collision", "foot") or ch.get("mesh") is None:
                    b.remove(ch)
                    continue
                mat = ch.get("material") or "metal"
                if ch.get("pos"):
                    ch.set("pos", _scl(ch.get("pos"), scale))
                for k in list(ch.attrib):
                    if k not in ("pos", "quat", "mesh"):
                        del ch.attrib[k]
                ch.set("type", "mesh")
                ch.set("material", "g1_" + mat)
                ch.set("contype", "0")
                ch.set("conaffinity", "0")
                ch.set("density", "0")
            elif ch.tag == "site":
                b.remove(ch)
            elif ch.tag == "body":
                walk(ch)
            elif ch.tag in ("joint", "freejoint", "inertial"):
                for k in ("armature", "frictionloss", "actuatorfrcrange", "class"):
                    ch.attrib.pop(k, None)
    walk(pelvis)
    for b in pelvis.iter("body"):
        if b.get("name") in ("left_ankle_roll_link", "right_ankle_roll_link"):
            side = b.get("name").split("_")[0]
            ET.SubElement(b, "site", name=f"foot_{side}", pos=_scl("0.035 0 -0.035", scale), size="0.01", group="4")
    pelvis.attrib.pop("childclass", None)
    pelvis.set("pos", "0 0 1.0")
    return "".join(assets), ET.tostring(pelvis, encoding="unicode"), files


# ------------------------------------------------------------------ rig
class Rig:
    def __init__(self, width=1280, height=720, fovy=34, outrigger_len=None):
        self.W, self.H, self.fovy = width, height, fovy
        self.meta = json.load(open(os.path.join(CAD, "parts_d3.json")))
        self.contract = json.load(open(os.path.join(ROOT, "analysis", "i2_sizing", "data", "d3_contract.json")))
        xml, assets = self._xml()
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        m = self.model
        nm = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, G, T, S = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_TENDON, mujoco.mjtObj.mjOBJ_SITE
        self.mocap = {}
        for i in range(m.nbody):
            if m.body_mocapid[i] >= 0:
                self.mocap[mujoco.mj_id2name(m, B, i)] = m.body_mocapid[i]
        self.pelvis_body = nm(B, "pelvis")
        self.qadr = {}
        self.dadr = {}
        for j in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.qadr[n] = m.jnt_qposadr[j]
            self.dadr[n] = m.jnt_dofadr[j]
        self.ten = {n: nm(T, n) for n in CABLES + ["strapL", "strapR"]}
        self.site = {n: nm(S, n) for n in ("foot_left", "foot_right")}
        self.g_arrow = (nm(G, "arrow_shaft"), nm(G, "arrow_head"))
        self.g_ring = nm(G, "ring")
        self.g_arm = [nm(G, f"orig_arm{i}") for i in range(4)]
        self.g_pad = [nm(G, f"orig_pad{i}") for i in range(4)]
        self.g_belt = nm(G, "belt_geom")
        self.og = self.meta["outriggers"]
        self.base_shift = np.zeros(3)
        self.set_arrow(None)
        self.set_ring(None)
        mujoco.mj_forward(m, self.data)

    # ------------------------------------------------------------- xml
    def _xml(self):
        meta = self.meta
        a, files, bodies = [], {}, []
        for k, (col, met, rough) in meta["materials"].items():
            r, g, b = hex2rgb(col)
            a.append(f'<material name="m_{k}" rgba="{r:.3f} {g:.3f} {b:.3f} 1" specular="{0.25 + 0.55 * met:.2f}" '
                     f'shininess="{0.25 + 0.6 * (1 - rough):.2f}"/>')
        tr, tg, tb = hex2rgb("#2E95A1")
        a.append(f'<material name="g1_metal" rgba="{tr:.3f} {tg:.3f} {tb:.3f} 1" specular="0.35" shininess="0.45"/>')
        a.append('<material name="g1_black" rgba="0.13 0.14 0.16 1" specular="0.2" shininess="0.3"/>')
        a.append('<material name="m_arrow" rgba="0.9 0.6 0.2 1" specular="0.2" shininess="0.3"/>')
        fr, fg, fb = 0.875, 0.88, 0.885
        a.append(f'<texture name="grid" type="2d" builtin="checker" rgb1="{fr} {fg} {fb}" '
                 f'rgb2="{fr - 0.04:.3f} {fg - 0.04:.3f} {fb - 0.035:.3f}" width="512" height="512"/>')
        a.append('<material name="m_floor" texture="grid" texrepeat="6 6" texuniform="true" reflectance="0" specular="0" shininess="0.1"/>')
        a.append(f'<texture name="sky" type="skybox" builtin="flat" rgb1="{fr} {fg} {fb}" rgb2="{fr} {fg} {fb}" width="16" height="16"/>')
        # static walker parts
        static = [p for p in meta["parts"] if not p["name"].startswith(NO_STATIC)]
        geoms = []
        for p in static:
            fn = "w_" + p["name"] + ".stl"
            files[fn] = open(os.path.join(CAD, "parts_d3", p["name"] + ".stl"), "rb").read()
            a.append(f'<mesh name="w_{p["name"]}" file="{fn}"/>')
            geoms.append(f'<geom type="mesh" mesh="w_{p["name"]}" material="m_{p["mat"]}" contype="0" conaffinity="0" group="1"/>')
        for nm_ in ("bws_spreader",):
            fn = "w_" + nm_ + ".stl"
            files[fn] = open(os.path.join(CAD, "parts_d3", nm_ + ".stl"), "rb").read()
            a.append(f'<mesh name="w_{nm_}" file="{fn}"/>')
            geoms.append(f'<geom type="mesh" mesh="w_{nm_}" pos="0 0 {SPREADER_DZ}" material="m_teal" contype="0" conaffinity="0" group="1"/>')
        geoms.append(f'<geom type="capsule" fromto="0.42 0 1.765 0.42 0 {1.68 + SPREADER_DZ}" size="0.005" material="m_cable" '
                     f'contype="0" conaffinity="0" group="1"/>')
        # belt (mocap) -- mesh coordinates are world coordinates, mocap_pos holds the shift
        for p in meta["parts"]:
            if p["name"] in ("pelvic_belt", "belt_buckle"):
                fn = "w_" + p["name"] + ".stl"
                files[fn] = open(os.path.join(CAD, "parts_d3", p["name"] + ".stl"), "rb").read()
                a.append(f'<mesh name="w_{p["name"]}" file="{fn}"/>')
        belt_c = np.array(meta["belt"]["centre"])
        sites_belt, sites_fix, sag_bodies, ten = [], [], [], []
        hw, bh = meta["belt"]["hw"], meta["belt"]["h"]
        for n in CABLES:
            an = meta["anchors"][n]
            ex, bp = an["exit"], an["belt"]
            sites_fix.append(f'<site name="a_{n}" pos="{ex[0]:.5f} {ex[1]:.5f} {ex[2]:.5f}" size="0.006" group="4"/>')
            sites_belt.append(f'<site name="b_{n}" pos="{bp[0]:.5f} {bp[1]:.5f} {bp[2]:.5f}" size="0.006" group="4"/>')
            sag_bodies.append(f'<body name="sag_{n}" mocap="true"><site name="s_{n}" size="0.004" group="4"/></body>')
            ten.append(f'<spatial name="{n}" width="0.004" rgba="0.42 0.48 0.56 1"><site site="a_{n}"/><site site="s_{n}"/>'
                       f'<site site="b_{n}"/></spatial>')
        sp_z = meta["bws"]["spreader"][2] + SPREADER_DZ
        sx = meta["bws"]["spreader"][0]
        for sn, sg in (("L", 1), ("R", -1)):
            sites_fix.append(f'<site name="a_strap{sn}" pos="{sx} {sg * 0.35} {sp_z}" size="0.006" group="4"/>')
            bp = belt_c + np.array([0, sg * hw, bh / 2])
            sites_belt.append(f'<site name="b_strap{sn}" pos="{bp[0]:.5f} {bp[1]:.5f} {bp[2]:.5f}" size="0.006" group="4"/>')
            sag_bodies.append(f'<body name="sag_strap{sn}" mocap="true"><site name="s_strap{sn}" size="0.004" group="4"/></body>')
            ten.append(f'<spatial name="strap{sn}" width="0.004" rgba="0.17 0.23 0.33 1"><site site="a_strap{sn}"/>'
                       f'<site site="s_strap{sn}"/><site site="b_strap{sn}"/></spatial>')
        belt_geoms = ('<geom name="belt_geom" type="mesh" mesh="w_pelvic_belt" material="m_brace" contype="0" conaffinity="0"/>'
                      '<geom type="mesh" mesh="w_belt_buckle" material="m_steel" contype="0" conaffinity="0"/>')
        bodies.append(f'<body name="belt" mocap="true">{belt_geoms}{"".join(sites_belt)}</body>')
        # outriggers (mocap capsules rebuilt from the CAD parameters)
        for i in range(4):
            o = self.meta["outriggers"][str(i)]
            L = float(np.hypot(o["foot"][0] - o["pivot"][0], o["foot"][1] - o["pivot"][1]))
            bodies.append(f'<body name="orig_arm{i}" mocap="true"><geom name="orig_arm{i}" type="capsule" size="0.022 {L / 2:.4f}" '
                          f'material="m_teal" contype="0" conaffinity="0"/></body>')
            bodies.append(f'<body name="orig_strut{i}" mocap="true"><geom type="capsule" size="0.016 0.055" material="m_frame_dark" '
                          f'contype="0" conaffinity="0"/></body>')
            bodies.append(f'<body name="orig_pad{i}" mocap="true"><geom name="orig_pad{i}" type="cylinder" size="0.06 0.015" '
                          f'material="m_rubber" contype="0" conaffinity="0"/></body>')
        # arrow + capture-point ring
        cone = trimesh.creation.cone(radius=0.05, height=0.11, sections=24)
        cone.apply_translation([0, 0, -0.055])
        files["arrow_cone.stl"] = cone.export(file_type="stl")
        a.append('<mesh name="arrow_cone" file="arrow_cone.stl"/>')
        ring = trimesh.creation.annulus(r_min=0.065, r_max=0.095, height=0.006, sections=48)
        files["ring.stl"] = ring.export(file_type="stl")
        a.append('<mesh name="ring_mesh" file="ring.stl"/>')
        bodies.append('<body name="arrow_shaft" mocap="true"><geom name="arrow_shaft" type="cylinder" size="0.02 0.1" '
                      'material="m_arrow" contype="0" conaffinity="0"/></body>')
        bodies.append('<body name="arrow_head" mocap="true"><geom name="arrow_head" type="mesh" mesh="arrow_cone" '
                      'material="m_arrow" contype="0" conaffinity="0"/></body>')
        bodies.append('<body name="ring" mocap="true"><geom name="ring" type="mesh" mesh="ring_mesh" material="m_arrow" '
                      'contype="0" conaffinity="0"/></body>')
        # humanoid
        g_assets, g_body, g_files = g1_xml(G1_SCALE)
        a.append(g_assets)
        files.update(g_files)
        lights = """
 <light name="key" directional="true" castshadow="true" pos="2 -2 5" dir="-0.45 0.5 -1" diffuse="0.40 0.40 0.39" specular="0.35 0.35 0.35"/>
 <light name="key2" directional="true" castshadow="false" pos="2 -2 5" dir="-0.30 0.62 -1" diffuse="0.16 0.16 0.16" specular="0.1 0.1 0.1"/>
 <light name="fill" directional="true" castshadow="false" pos="-3 2 3" dir="0.6 -0.5 -0.35" diffuse="0.30 0.31 0.33" specular="0.1 0.1 0.1"/>
 <light name="front" directional="true" castshadow="false" pos="4 0 2" dir="-1 0.1 -0.35" diffuse="0.22 0.22 0.22" specular="0.1 0.1 0.1"/>"""
        xml = f"""<mujoco model="catchpoint_g1">
 <compiler angle="radian"/>
 <option gravity="0 0 0"/>
 <statistic extent="1.0"/>
 <visual>
  <global offwidth="{self.W}" offheight="{self.H}" fovy="{self.fovy}"/>
  <quality shadowsize="8192" offsamples="8"/>
  <headlight ambient="0.40 0.40 0.42" diffuse="0.12 0.12 0.12" specular="0 0 0"/>
  <rgba haze="{fr} {fg} {fb} 1" fog="{fr} {fg} {fb} 1"/>
  <map znear="0.02" zfar="60" fogstart="7" fogend="30"/>
 </visual>
 <asset>{''.join(a)}</asset>
 <worldbody>{lights}
  <geom name="floor" type="plane" size="30 30 0.1" material="m_floor" contype="0" conaffinity="0"/>
  <body name="walker" mocap="true">{''.join(geoms)}{''.join(sites_fix)}</body>
  {''.join(bodies)}{''.join(sag_bodies)}
  {g_body}
 </worldbody>
 <tendon>{''.join(ten)}</tendon>
</mujoco>"""
        return xml, files

    # ------------------------------------------------------------- setters
    def _set(self, name, pos, quat=None):
        i = self.mocap[name]
        self.data.mocap_pos[i] = pos
        if quat is not None:
            self.data.mocap_quat[i] = quat

    def set_belt(self, centre_xyz):
        c0 = np.array(self.meta["belt"]["centre"])
        self._set("belt", np.asarray(centre_xyz, float) - c0, [1, 0, 0, 0])

    def set_walker(self, shift_xy):
        self._set("walker", [shift_xy[0], shift_xy[1], 0.0], [1, 0, 0, 0])
        self.base_shift = np.array([shift_xy[0], shift_xy[1], 0.0])

    def cable_ends(self, name, belt_shift):
        """World positions of (anchor, belt attach) for cable `name` for a given belt shift (3,); the anchors move
        with the walker (set_walker)."""
        if name.startswith("strap"):
            sg = 1 if name.endswith("L") else -1
            c = np.array(self.meta["belt"]["centre"])
            a = np.array([self.meta["bws"]["spreader"][0], sg * 0.35, self.meta["bws"]["spreader"][2] + SPREADER_DZ])
            b = c + np.array([0, sg * self.meta["belt"]["hw"], self.meta["belt"]["h"] / 2]) + belt_shift
            return a + self.base_shift, b
        else:
            an = self.meta["anchors"][name]
            a, b = np.array(an["exit"]), np.array(an["belt"]) + belt_shift
        return a + self.base_shift, b

    def set_cable(self, name, sag, rgba, width, belt_shift):
        """sag in [0, 1]: 0 = straight (taut), 1 = max slack drop (0.10 x chord)."""
        a, b = self.cable_ends(name, belt_shift)
        mid = 0.5 * (a + b)
        mid[2] -= 0.10 * sag * np.linalg.norm(b - a)
        self._set("sag_" + name, mid)
        t = self.ten[name]
        self.model.tendon_rgba[t] = rgba
        self.model.tendon_width[t] = width

    def set_outrigger(self, i, u):
        """u = 0 retracted (folded against the rail, pad raised), 1 deployed (pad on the floor)."""
        o = self.og[str(i)]
        piv = np.array(o["pivot"]) + self.base_shift
        foot = np.array([o["foot"][0], o["foot"][1], piv[2]]) + self.base_shift
        L = float(np.linalg.norm(foot[:2] - piv[:2]))
        sg = 1 if piv[1] > 0 else -1
        d_dep = (foot - piv) / L
        d_ret = np.array([1.0 if o["pivot"][0] < 0.4 else -1.0, 0.0, 0.0])
        ang0 = np.arctan2(d_ret[1], d_ret[0])
        ang1 = np.arctan2(d_dep[1], d_dep[0])
        ang = ang0 + (ang1 - ang0) * u
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        p0 = piv + np.array([0, sg * 0.05 * (1 - u), 0.0])
        p1 = p0 + d * L
        self._set(f"orig_arm{i}", 0.5 * (p0 + p1), quat_z_to(p1 - p0))
        lift = 0.075 * (1 - u)
        s0 = p1.copy()
        s1 = np.array([p1[0], p1[1], 0.03 + lift])
        self._set(f"orig_strut{i}", 0.5 * (s0 + s1), quat_z_to(s0 - s1))
        self._set(f"orig_pad{i}", [p1[0], p1[1], 0.015 + lift], [1, 0, 0, 0])

    def set_arrow(self, tail, vec=None, rgba=None):
        sid, hid = self.g_arrow
        m = self.model
        L = 0.0 if tail is None else float(np.linalg.norm(vec))
        if L < 0.06:
            m.geom_rgba[sid] = (0, 0, 0, 0)
            m.geom_rgba[hid] = (0, 0, 0, 0)
            return
        d = np.asarray(vec, float) / L
        shaft = max(L - 0.11, 0.01)
        m.geom_size[sid] = (0.02, shaft / 2, 0)
        for g in (sid, hid):
            m.geom_matid[g] = -1
            m.geom_rgba[g] = rgba
        tail = np.asarray(tail, float)
        self._set("arrow_shaft", tail + d * shaft / 2, quat_z_to(d))
        self._set("arrow_head", tail + d * (shaft + 0.055), quat_z_to(d))

    def set_ring(self, xy, rgba=None):
        g = self.g_ring
        if xy is None:
            self.model.geom_rgba[g] = (0, 0, 0, 0)
            return
        self.model.geom_matid[g] = -1
        self.model.geom_rgba[g] = rgba
        self._set("ring", [xy[0], xy[1], 0.004], [1, 0, 0, 0])

    def set_belt_colour(self, rgba):
        self.model.geom_matid[self.g_belt] = -1
        self.model.geom_rgba[self.g_belt] = rgba

    # ------------------------------------------------------------- camera / render
    @staticmethod
    def cam_from(pos, lookat):
        pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
        f = lookat - pos
        dist = np.linalg.norm(f)
        f /= dist
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
        cam.elevation = float(np.degrees(np.arcsin(f[2])))
        return cam

    def render(self, cam):
        mujoco.mj_forward(self.model, self.data)       # kinematics only: nothing is ever integrated
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()


if __name__ == "__main__":
    r = Rig(640, 360)
    m = r.model
    print("bodies", m.nbody, "geoms", m.ngeom, "tendons", m.ntendon, "nq", m.nq, "mocap", m.nmocap)
    z = r.render(r.cam_from([3.3, -2.4, 1.7], [0.45, 0.0, 0.9]))
    print("rendered", z.shape, z.dtype, float(z.mean()))
