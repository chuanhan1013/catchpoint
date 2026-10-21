"""Draw callouts on annotated_base.png -> annotated.png (system python + Pillow).
python callouts.py [--preview]"""
import json, os, sys
import os as _os
def _font_path(name):
    """System font if present (Windows: %WINDIR%/Fonts), else matplotlib's bundled DejaVu Sans."""
    win = _os.path.join(_os.environ.get("WINDIR", ""), "Fonts", name)
    if _os.path.exists(win):
        return win
    import matplotlib
    bold = "b" in name.replace(".ttf", "")[-2:]
    return _os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
from PIL import Image, ImageDraw, ImageFont, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__))
prev = "--preview" in sys.argv
base = Image.open(os.path.join(HERE, "annotated_d3_base_prev.png" if prev else "annotated_d3_base.png")).convert("RGBA")
meta = json.load(open(os.path.join(HERE, "annotated_d3_anchors.json")))
W, H = base.size
S = W / 1920.0
A = {k: (v[0] * W / meta["size"][0], v[1] * H / meta["size"][1]) for k, v in meta["anchors"].items()}

TEAL, INK = (31, 111, 120), (30, 42, 51)
font = ImageFont.truetype(_font_path("segoeui.ttf"), int(30 * S))
fontb = ImageFont.truetype(_font_path("segoeuib.ttf"), int(30 * S))

# (key, label lines, side, label centre y as fraction of H)
L = [
    ("bws",      ["Overhead BWS cable", "on cantilever beam"], "R", 0.10),
    ("spools",   ["Dual-mode cable spools (x8)", "(servo + slip brake)"], "R", 0.30),
    ("belt",     ["Padded pelvic belt"], "R", 0.50),
    ("outrigger", ["Deploying outrigger feet"], "R", 0.70),
    ("ballast",  ["Low ballast"], "L", 0.76),
    ("lidar",    ["Foot-tracking LiDAR"], "L", 0.52),
    ("wheel",    ["Omni-wheel base"], "R", 0.93),
]
# layout: label x positions
pad = int(18 * S)
lab_w = {}
d0 = ImageDraw.Draw(base)
boxes = {}
for key, lines, side, fy in L:
    tw = max(d0.textlength(t, font=font) for t in lines)
    lh = int(40 * S)
    w, h = int(tw + 2 * pad), int(lh * len(lines) + 2 * pad - 6 * S)
    x = int(0.045 * W) if side == "L" else int(W - 0.045 * W - w)
    y = int(fy * H - h / 2)
    boxes[key] = (x, y, w, h, side, lines)

shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
for key, (x, y, w, h, side, lines) in boxes.items():
    sd.rounded_rectangle((x, y + 4 * S, x + w, y + h + 4 * S), radius=int(14 * S), fill=(20, 30, 40, 70))
shadow = shadow.filter(ImageFilter.GaussianBlur(8 * S))
base = Image.alpha_composite(base, shadow)
# supersample overlay for crisp lines
SS = 3
ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
od = ImageDraw.Draw(ov)
for key, (x, y, w, h, side, lines) in boxes.items():
    ax, ay = A[key]
    ex = (x + w) if side == "L" else x
    ey = y + h / 2
    # elbow leader: horizontal from label, then straight to anchor
    od.line([(ex * SS, ey * SS), (ax * SS, ay * SS)], fill=INK + (235,), width=int(2.2 * S * SS))
    r = 7 * S * SS
    od.ellipse((ax * SS - r - 3 * SS, ay * SS - r - 3 * SS, ax * SS + r + 3 * SS, ay * SS + r + 3 * SS), fill=(255, 255, 255, 255))
    od.ellipse((ax * SS - r, ay * SS - r, ax * SS + r, ay * SS + r), fill=TEAL + (255,))
ov = ov.resize((W, H), Image.LANCZOS)
base = Image.alpha_composite(base, ov)
dr = ImageDraw.Draw(base)
for key, (x, y, w, h, side, lines) in boxes.items():
    dr.rounded_rectangle((x, y, x + w, y + h), radius=int(14 * S), fill=(255, 255, 255, 245), outline=(205, 212, 218, 255), width=max(1, int(S)))
    ty = y + pad - 6 * S
    for i, t in enumerate(lines):
        dr.text((x + pad, ty + i * 40 * S), t, font=font, fill=INK)
base.convert("RGB").save(os.path.join(HERE, "annotated_d3_prev.png" if prev else "annotated_d3.png"))
print("ok")
