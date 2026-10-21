"""story_d3_{green,amber,red}.png (1200x1200). blender -b -P story_d3.py -- [--state X] [--preview]"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import cp_d3 as D, cp_common as C, shots as S

preview = "--preview" in sys.argv
state_arg = S.arg_after("--state", "all")
out = os.path.dirname(os.path.abspath(__file__))
RES, SMP = (700, 48) if preview else (1200, 224)
LEG = 0.74

ST = {
    "green": dict(col=C.GREEN, dy=0.03, lean=3, leg=dict(abd=14, flex=-34, knee=26), arm=(40, 25), out=False),
    "amber": dict(col=C.AMBER, dy=0.07, lean=14, leg=dict(abd=8, flex=-18, knee=14), arm=(55, 20), out=False),
    "red":   dict(col=C.RED, dy=0.10, lean=26, leg=dict(abd=3, flex=-5, knee=5), arm=(70, 12), out=True),
}


def pose(H, P):
    abd, el = P["arm"]
    C.rot_bone(H, "LeftArm", "X", -(90 - abd)); C.rot_bone(H, "RightArm", "X", (90 - abd))
    C.rot_bone(H, "LeftForeArm", "Y", -el); C.rot_bone(H, "RightForeArm", "Y", -el)
    th = math.degrees(math.asin(P["dy"] / LEG))
    C.rot_bone(H, "RightUpLeg", "X", -th); C.rot_bone(H, "RightFoot", "X", th)
    L = P["leg"]
    C.rot_bone(H, "LeftUpLeg", "X", L["abd"]); C.rot_bone(H, "LeftUpLeg", "Y", L["flex"])
    C.rot_bone(H, "LeftLeg", "Y", L["knee"])
    C.rot_bone(H, "LeftFoot", "Y", -(L["flex"] + L["knee"]) * 0.8)
    C.rot_bone(H, "LeftFoot", "X", -L["abd"] * 0.9)
    S.lean(H, P["lean"])
    C.rot_bone(H, "Neck", "X", P["lean"] * 0.35); C.rot_bone(H, "Head", "X", P["lean"] * 0.3)


def build(st):
    P = ST[st]
    D.new_shot(RES, RES, SMP)
    cam = S.polar(30, 3.5, 1.6, cx=D.CX)
    D.studio(cam, key_power=1300, floor=(0.30, 0.31, 0.33))
    M = D.build_materials(caliper=C.RED if st == "red" else None)
    Wk = D.build_walker(M, st, dy=P["dy"], glow=P["col"], glow_strength=3.2, outriggers=P["out"])
    H = C.load_human("Xbot", scale=0.85)
    C.style_human(H)
    pose(H, P)
    D.place(H, (0.425, 0.0), dy=P["dy"])
    pm = C.mat_translucent("push", C.hexc("#7F93AA"), alpha=0.72, emit=0.35)
    C.make_arrow("push", (0.25, -1.45, 0.95), (0.25, -0.78, 0.95), pm, 0.04, 0.095, 0.24)
    if st == "amber":
        am = C.mat_translucent("nudge", C.hexc("#E0A030"), alpha=0.95, emit=0.0)
        C.make_arrow("nudge", (0.25, 1.35, 0.8), (0.25, 0.78, 0.8), am, 0.04, 0.095, 0.22)
    pl = bpy.data.lights.new("glow_light", "POINT"); pl.energy = 9; pl.color = C.hexc(P["col"])[:3]
    po = bpy.data.objects.new("glow_light", pl); bpy.context.scene.collection.objects.link(po)
    po.location = (0.55, P["dy"], 1.06)
    C.make_camera(cam, (D.CX - 0.02, 0.0, 0.84), lens=36, fstop=9.0, focus=(0.42, 0, 0.95))


for st in (["green", "amber", "red"] if state_arg == "all" else [state_arg]):
    build(st)
    C.render_to(os.path.join(out, f"story_d3_{st}{'_prev' if preview else ''}.png"))
