"""Build the public deck PDF from the private deck's slide PNGs, covering non-CC product photos.
Usage: python tools/make_public_pdf.py <private_repo>/deck/png assets/deck/CatchPoint_pitch_public.pdf"""
import sys, glob, re
from PIL import Image, ImageDraw, ImageFont
import os as _os
def _font_path(name):
    """System font if present (Windows: %WINDIR%/Fonts), else matplotlib's bundled DejaVu Sans."""
    win = _os.path.join(_os.environ.get("WINDIR", ""), "Fonts", name)
    if _os.path.exists(win):
        return win
    import matplotlib
    bold = "b" in name.replace(".ttf", "")[-2:]
    return _os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(src + "/s*.png"), key=lambda f: int(re.findall(r"s(\d+)\.png", f)[0]))
# Slide 3: the Hocoma Andago photo (© Hocoma/DIH) is the middle photo; box in 1280x720 slide pixels.
COVER = {3: [(450, 139, 829, 307)]}
font = ImageFont.truetype(_font_path("segoeui.ttf"), 16)
pages = []
for f in files:
    n = int(re.findall(r"s(\d+)\.png", f)[0]); im = Image.open(f).convert("RGB")
    sx, sy = im.width / 1280, im.height / 720
    for (x0, y0, x1, y1) in COVER.get(n, []):
        box = (int(x0*sx), int(y0*sy), int(x1*sx), int(y1*sy)); d = ImageDraw.Draw(im)
        d.rectangle(box, fill=(233, 238, 241), outline=(155, 165, 172))
        d.multiline_text(((box[0]+box[2])//2, (box[1]+box[3])//2), "Photo omitted in public version\n(Hocoma Andago — see source link)",
                         fill=(60, 72, 82), font=font, anchor="mm", align="center")
    pages.append(im)
pages[0].save(out, save_all=True, append_images=pages[1:], resolution=96)
print(len(pages), "pages ->", out)
