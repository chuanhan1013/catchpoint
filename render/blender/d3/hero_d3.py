"""hero_d3.png + hero_d3_transparent.png.  blender -b -P hero_d3.py -- [--preview]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp_d3 as D, cp_common as C, shots as S
preview = "--preview" in sys.argv
out = os.path.dirname(os.path.abspath(__file__))
W, H, SMP = (960, 540, 48) if preview else (1920, 1080, 224)

def build(transparent):
    D.new_shot(W, H, SMP, transparent)
    cam = S.polar(52, 4.6, 1.7, cx=D.CX, cy=0)
    D.studio(cam, key_power=1300, shadow_catcher=transparent, floor=(0.30, 0.31, 0.33))
    M = D.build_materials()
    Wk = D.build_walker(M, "hero", glow=C.TEAL, glow_strength=2.5)
    Hm = C.load_human("Xbot", scale=0.85)
    C.style_human(Hm)
    S.pose_stand(Hm, arm_abd=14, elbow=55, yaw_in=8)
    D.place(Hm, (0.425, 0.0))
    C.make_camera(cam, (D.CX - 0.05, 0, 0.92), lens=40, fstop=5.0, focus=(0.42, 0, 0.95))
build(False)
C.render_to(os.path.join(out, "hero_d3_prev.png" if preview else "hero_d3.png"))
if not preview:
    build(True)
    C.render_to(os.path.join(out, "hero_d3_transparent.png"))
