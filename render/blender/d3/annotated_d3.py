"""annotated_d3_base.png + anchors json. blender -b -P annotated_d3.py -- [--preview]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import cp_d3 as D, cp_common as C, shots as S
preview = "--preview" in sys.argv
out = os.path.dirname(os.path.abspath(__file__))
W, H, SMP = (960, 540, 48) if preview else (1920, 1080, 224)
D.new_shot(W, H, SMP)
cam_loc = S.polar(215, 4.6, 1.6, cx=D.CX)           # rear-right 3/4 so the rear spools, ballast and LiDAR show
D.studio(cam_loc, key_power=1300, floor=(0.30, 0.31, 0.33))
M = D.build_materials()
Wk = D.build_walker(M, "hero", glow=C.TEAL, glow_strength=2.0, outriggers=True)
Hm = C.load_human("Xbot", scale=0.85); C.style_human(Hm)
S.pose_stand(Hm, arm_abd=14, elbow=55, yaw_in=8); D.place(Hm, (0.425, 0.0))
cam = C.make_camera(cam_loc, (D.CX - 0.02, 0, 0.98), lens=40, fstop=7.0, focus=(0.3, 0, 0.9))
bpy.context.view_layer.update()
ANCH = {"spools": tuple(__import__("json").load(open(os.path.join(C.CAD,"parts_d3.json")))["anchors"]["RR"]["housing"]), "belt": (0.30, 0.0, 0.95), "bws": (0.30, 0.0, 1.80),
        "outrigger": (1.125, -0.575, 0.04), "ballast": (-0.05, -0.2, 0.17), "lidar": (0.03, 0.17, 0.30),
        "wheel": (0.04, -0.372, 0.075)}
res = {k: [v.x * W, (1 - v.y) * H, v.z] for k, p in ANCH.items() for v in [world_to_camera_view(bpy.context.scene, cam, Vector(p))]}
json.dump({"size": [W, H], "anchors": res}, open(os.path.join(out, "annotated_d3_anchors.json"), "w"), indent=1)
C.render_to(os.path.join(out, "annotated_d3_base_prev.png" if preview else "annotated_d3_base.png"))
