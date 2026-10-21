"""Compose story_strip.png (3600x1200) from the three stills + labels. python story_strip.py"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os as _os
def _font_path(name):
    """System font if present (Windows: %WINDIR%/Fonts), else matplotlib's bundled DejaVu Sans."""
    win = _os.path.join(_os.environ.get("WINDIR", ""), "Fonts", name)
    if _os.path.exists(win):
        return win
    import matplotlib
    bold = "b" in name.replace(".ttf", "")[-2:]
    return _os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
HERE = os.path.dirname(os.path.abspath(__file__))
font = ImageFont.truetype(_font_path("segoeui.ttf"), 40)
INK = (30, 42, 51)
items = [("green", "#3A9D5D", "Cables slack — patient steps alone"),
         ("amber", "#E0A030", "Two cables pull the pelvis back to centre"),
         ("red", "#C8443A", "All cables lock — outriggers deploy, BWS catches")]
W = Image.new("RGB", (3600, 1200), "white")
for i, (s, col, txt) in enumerate(items):
    W.paste(Image.open(os.path.join(HERE, f"story_d3_{s}.png")).convert("RGB"), (1200 * i, 0))
base = W.convert("RGBA")
sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(sh)
dr0 = ImageDraw.Draw(base)
boxes = []
for i, (s, col, txt) in enumerate(items):
    tw = dr0.textlength(txt, font=font)
    w, h = int(tw + 110), 84
    x = 1200 * i + (1200 - w) // 2
    y = 1200 - 44 - h
    boxes.append((x, y, w, h))
    sd.rounded_rectangle((x, y + 5, x + w, y + h + 5), radius=42, fill=(20, 30, 40, 80))
sh = sh.filter(ImageFilter.GaussianBlur(9))
base = Image.alpha_composite(base, sh)
dr = ImageDraw.Draw(base)
for (x, y, w, h), (s, col, txt) in zip(boxes, items):
    dr.rounded_rectangle((x, y, x + w, y + h), radius=42, fill=(255, 255, 255, 240))
    r = 14
    dr.ellipse((x + 30, y + h // 2 - r, x + 30 + 2 * r, y + h // 2 + r), fill=col)
    dr.text((x + 76, y + h // 2 - 1), txt, font=font, fill=INK, anchor="lm")
base.convert("RGB").save(os.path.join(HERE, "story_d3_strip.png"))
print("ok")
