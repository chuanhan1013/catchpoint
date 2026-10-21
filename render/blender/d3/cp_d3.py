"""D3 scene builder: walker with 8-cable pelvic interface (parts from cad/parts_d3). Blender 4.2."""
import sys, os, json, math
from math import radians
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bpy, bmesh
from mathutils import Vector, Matrix
import cp_common as C
import shots as S

CAD = C.CAD
CX = 0.45       # walker centre x (scene is lit/framed around it)
SPEC = json.load(open(os.path.join(CAD, "parts_d3.json")))
REBUILT = ("cable_", "pelvic_belt", "belt_buckle", "bws_cable", "bws_strap_", "bws_spreader", "outrigger_")


def circle(r, k=12):
    return [(r * math.cos(2 * math.pi * i / k), r * math.sin(2 * math.pi * i / k)) for i in range(k)]


def side_for(t):
    ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
    return t.cross(ref).normalized()


def cable(name, p0, p1, r, mat, sag=0.0, n=28):
    p0, p1 = Vector(p0), Vector(p1)
    pts = []
    for i in range(n + 1):
        u = i / n
        p = p0.lerp(p1, u)
        p.z -= sag * 4 * u * (1 - u)
        pts.append(p)
    ob = C.sweep(name, pts, circle(r), side_fn=side_for, up=None)
    ob.data.materials.append(mat)
    return ob


def mat_cable(name, color, emit=0.0):
    m, b, nt = C._principled(name)
    b.inputs["Base Color"].default_value = C.hexc(color)
    b.inputs["Roughness"].default_value = 0.4
    b.inputs["Specular IOR Level"].default_value = 0.5
    if emit:
        b.inputs["Emission Color"].default_value = C.hexc(color)
        b.inputs["Emission Strength"].default_value = emit
    return m


def build_materials(caliper=None):
    M = C.build_materials(caliper_color=caliper)
    M["ballast"] = C.mat_metal("ballast", C.hexc("#454A52"), 0.55, metallic=0.6, bump=0.0)
    M["cable"] = mat_cable("cable_dark", "#2B2F34")
    return M


MAT = dict(C.MAT_MAP)
MAT.update({"ballast": "ballast", "cable": "cable"})


def build_walker(M, state, dy=0.0, glow=None, glow_strength=3.2, outriggers=False):
    """state in green/amber/red/hero: cable colour + tautness."""
    W = C.Walker()
    W.parts = {}
    for p in SPEC["parts"]:
        n = p["name"]
        if n.startswith(REBUILT):
            continue
        bpy.ops.wm.stl_import(filepath=os.path.join(CAD, "parts_d3", n + ".stl"))
        ob = bpy.context.selected_objects[0]
        ob.name = ob.data.name = n
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(ob.data)
        bm.free()
        ob.data.materials.append(M[MAT[p["mat"]]])
        mn = min(x for x in ob.dimensions if x > 1e-4)
        if not any(k in n for k in ("roller", "screw")):
            C.bevel_apply(ob, max(0.0006, min(0.0035, mn * 0.09)))
        C.smooth_ob(ob, 38)
        W.parts[n] = ob

    # --- belt: closed rounded ring, shifted with the pelvis
    bl = SPEC["belt"]
    pc = Vector(bl["centre"]) + Vector((0, dy, 0))
    ring = []
    for i in range(72):
        a = 2 * math.pi * i / 72
        ca, sa = math.cos(a), math.sin(a)
        e = 2.0
        ring.append(pc + Vector((bl["hd"] * math.copysign(abs(ca) ** (2 / e), ca),
                                 bl["hw"] * math.copysign(abs(sa) ** (2 / e), sa), 0)))
    ring.append(ring[0])
    prof = C.pill_profile(bl["t"], bl["h"], n=4.5)
    side = lambda t: Vector((t.y, -t.x, 0)).normalized()
    belt = C.sweep("pelvic_belt", ring, prof, side_fn=side, caps=False)
    belt.data.materials.append(M["fabric"])
    W.parts["pelvic_belt"] = belt
    gm = C.mat_emit("belt_glow", C.hexc(glow or C.TEAL), glow_strength)
    g1 = C.sweep("belt_glow", ring, C.pill_profile(0.006, 0.016, n=3), side_fn=side, offset=(bl["t"] / 2 + 0.0005, 0), caps=False)
    g1.data.materials.append(gm)
    g2 = C.sweep("belt_glow_top", ring, C.pill_profile(0.02, 0.008, n=3), side_fn=side, offset=(0, bl["h"] / 2 + 0.0005), caps=False)
    g2.data.materials.append(gm)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(pc.x + bl["hd"] + 0.02, pc.y, pc.z))
    bk = bpy.context.active_object
    bk.scale = (0.012, 0.06, 0.06)
    bk.name = "belt_buckle"
    bk.data.materials.append(M["steel"])

    # --- cables
    C_GREEN, C_AMBER, C_RED, C_SLACK = "#3A9D5D", "#E0A030", "#C8443A", "#5B6770"
    taut_set = {"green": set(), "amber": {"LR", "FR"}, "red": set(SPEC["anchors"]), "hero": set()}[state]
    colmap = {"green": C_GREEN, "amber": C_AMBER, "red": C_RED, "hero": "#A7B4BC"}
    mats = {
        "slack": mat_cable("cab_slack", C_SLACK if state in ("amber",) else colmap[state], 0.0 if state in ("hero", "amber") else 0.25),
        "taut": mat_cable("cab_taut", colmap[state], 1.4),
    }
    W.cables = {}
    for name, a in SPEC["anchors"].items():
        bp = Vector(a["belt"]) + Vector((0, dy, 0))
        ex = Vector(a["exit"])
        taut = name in taut_set
        r = 0.0085 if taut else 0.0058
        sag = 0.0 if taut else 0.07
        W.cables[name] = cable(f"cable_{name}", ex, bp, r, mats["taut" if taut else "slack"], sag=sag)
        # small ferrule on the belt
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.014, location=bp)
        fe = bpy.context.active_object
        fe.name = f"ferrule_{name}"
        fe.data.materials.append(M["steel"])
        C.smooth_ob(fe, 30)

    # --- BWS cable, spreader, straps
    b = SPEC["bws"]
    sp = Vector(b["spreader"]) + Vector((0, dy * 0.0, 0))
    ex = Vector(b["exit"])
    taut_b = state == "red"
    bm_ = mat_cable("bws_mat", C_RED if taut_b else "#2B2F34", 1.2 if taut_b else 0.0)
    W.bws = cable("bws_cable", ex, sp, 0.008 if taut_b else 0.005, bm_, sag=0.0)
    spr = C.sweep("bws_spreader", [Vector((sp.x, -0.36, sp.z)), Vector((sp.x, 0.36, sp.z))],
                  C.pill_profile(0.035, 0.028, n=3), side_fn=lambda t: Vector((1, 0, 0)), up=Vector((0, 0, 1)))
    spr.data.materials.append(M["teal"])
    strap_m = mat_cable("bws_strap", C_RED if taut_b else "#26344F", 0.8 if taut_b else 0.0)
    for s in (1, -1):
        a = sp + Vector((0, s * 0.35, 0))
        e = pc + Vector((0, s * bl["hw"], bl["h"] / 2))
        cable(f"bws_strap_{s}", a, e, 0.012 if not taut_b else 0.016, strap_m, sag=0.0 if taut_b else 0.03)

    # --- outriggers
    ob_m = M["teal"]
    for k, o in SPEC["outriggers"].items():
        piv = Vector(o["pivot"])
        fx, fy = o["foot"]
        s = 1 if fy > 0 else -1
        if outriggers:
            top = Vector((fx, fy, 0.14))
            pad_z = 0.0
        else:
            top = piv + Vector((0.0, s * 0.07, 0.0))
            fx, fy = top.x, top.y
            pad_z = 0.075                      # lifted pad tucked next to the frame
        arm = cable(f"outrigger_arm_{k}", piv, top, 0.024, ob_m, sag=0.0)
        strut = cable(f"outrigger_strut_{k}", Vector((fx, fy, top.z)), Vector((fx, fy, pad_z + 0.03)), 0.016, M["alu_dark"], sag=0.0)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.062, depth=0.03, location=(fx, fy, pad_z + 0.015), vertices=48)
        pad = bpy.context.active_object
        pad.name = f"outrigger_pad_{k}"
        pad.data.materials.append(M["rubber"])
        C.bevel_apply(pad, 0.006)
        C.smooth_ob(pad, 35)
    return W


# ------------------------------------------------------------------ human
def place(H, hips_xy, dy=0.0):
    C.human_root(H, 0, 0, 0)
    for _ in range(2):
        hp = C.bone_world(H, "Hips")
        r = H.arm.matrix_world.translation
        C.human_root(H, r.x + (hips_xy[0] - hp.x), r.y + (hips_xy[1] + dy - hp.y), r.z)
    vs = C.human_verts(H)
    zmin = min(v.z for v in vs)
    r = H.arm.matrix_world.translation
    C.human_root(H, r.x, r.y, r.z - zmin)


def studio(cam_loc, **kw):
    az = S.studio((cam_loc[0] - CX, cam_loc[1], cam_loc[2]), (0, 0, 0.9), **kw)
    for o in bpy.data.objects:
        if o.name in ("key", "fill", "rim", "top", "cyc"):
            o.location.x += CX
    return az


def new_shot(w, h, smp, transparent=False, exposure=-0.35):
    S.new_scene(w, h, smp, transparent=transparent, exposure=exposure, hdri_strength=0.55, hdri_rot=200)
