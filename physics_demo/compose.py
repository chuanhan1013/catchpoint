"""Add text overlays to the raw renders and build the comparison videos + GIF.

videos/{push}_{ctrl}.mp4      1280x720, 30 fps
videos/compare_b.mp4          1x3 (B0 | B1 | CP) for push (b), 1920x820
videos/grid_3x3.mp4           pushes (rows) x controllers (cols), 1920x1124
videos/compare_b.gif          <= 8 MB
"""
import json
import os
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
VID = os.path.join(HERE, "videos")
RAW = os.path.join(VID, "raw")
import os as _os
FONT_DIR = _os.path.join(_os.environ.get("WINDIR", "C:/Windows"), "Fonts")  # Windows system fonts; set WINDIR or edit for other OSes
FOOTER = "Physics simulation: pretrained Unitree G1 policy with simulated paretic weakness \u2014 not a patient model"
CTRL_NAME = {"B0": "B0 \u00b7 No assistance",
             "B1": "B1 \u00b7 Threshold lock (COM > 8 cm \u2192 lock)",
             "CP": "CP \u00b7 CatchPoint (graded)"}
CTRL_SHORT = {"B0": "B0 No assistance", "B1": "B1 Threshold lock", "CP": "CP CatchPoint"}
PUSH_TXT = {"a": "small", "b": "medium", "c": "large"}
LEVEL = [("HANDS OFF", (107, 122, 143)), ("NUDGE", (224, 160, 48)), ("CATCH", (200, 68, 58))]
TEXT = (28, 33, 40)


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "arialbd.ttf" if bold else "arial.ttf"), size)


def load(pk, ctrl):
    with open(os.path.join(RAW, f"{pk}_{ctrl}.json")) as f:
        info = json.load(f)
    rd = imageio.get_reader(os.path.join(RAW, f"{pk}_{ctrl}.mp4"))
    return info, rd


def outcome_text(info, fm):
    t = fm["t"]
    if info["outcome"] == "fall" and fm["fallen"]:
        return "FELL", LEVEL[2][1]
    if info["outcome"] == "caught" and fm["catch_on"] and t > info["catch_t"] + 0.6:
        return "CAUGHT BY HARNESS", LEVEL[2][1]
    if t > info["t0"] + 2.2:
        if info["outcome"] == "recovered alone":
            return "RECOVERED ON ITS OWN", (31, 111, 120)
        if info["outcome"] == "nudged":
            return "RECOVERED WITH A NUDGE", (31, 111, 120)
    return None, None


def badge(dr, x, y, lvl, fm, f_big, f_small, ctrl, scale=1.0):
    name, col = LEVEL[lvl]
    if lvl == 2 and not fm["catch_on"]:
        name = "CATCH (locking)"
    if ctrl == "B0":
        name, col = "NO DEVICE", (154, 160, 166)
    w = dr.textlength(name, font=f_big) + 28 * scale
    h = 40 * scale
    dr.rounded_rectangle([x, y, x + w, y + h], radius=int(10 * scale), fill=col + (255,))
    dr.text((x + 14 * scale, y + h / 2), name, font=f_big, fill=(255, 255, 255, 255), anchor="lm")
    if ctrl != "B0":
        ft = f"pelvic force {fm['F']:.0f} N  (\u2248{fm['F_h']:.0f} N at 75 kg)"
        dr.text((x + w + 12 * scale, y + h / 2), ft, font=f_small, fill=TEXT + (255,), anchor="lm")


def overlay_single(img, info, fm, ctrl):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    W, H = im.size
    f1, f2, f3, f4 = font(30, True), font(21), font(22, True), font(19)
    dr.rounded_rectangle([18, 16, 900, 150], radius=14, fill=(255, 255, 255, 215))
    dr.text((36, 30), CTRL_NAME[ctrl], font=f1, fill=TEXT + (255,))
    ptxt = (f"Push ({info['push']}, {PUSH_TXT[info['push']]}): {info['F_push']} N \u00d7 {info['dur']:.1f} s at the pelvis "
            f"\u2192 \u0394v {info['dv']:.2f} m/s toward the weak (right) side")
    dr.text((36, 70), ptxt, font=f4, fill=(70, 76, 86, 255))
    badge(dr, 36, 98, fm["level"], fm, f3, f2, ctrl)
    if fm["push"]:
        dr.text((W / 2, 190), "PUSH", font=font(34, True), fill=(217, 38, 38, 255), anchor="mm")
    txt, col = outcome_text(info, fm)
    if txt:
        tw = dr.textlength(txt, font=font(40, True))
        dr.rounded_rectangle([W / 2 - tw / 2 - 24, 560, W / 2 + tw / 2 + 24, 622], radius=14, fill=col + (235,))
        dr.text((W / 2, 591), txt, font=font(40, True), fill=(255, 255, 255, 255), anchor="mm")
    dr.text((W - 24, 40), f"t = {fm['t']:.2f} s", font=f2, fill=(70, 76, 86, 255), anchor="rm")
    dr.rectangle([0, H - 42, W, H], fill=(25, 30, 38, 200))
    dr.text((W / 2, H - 21), FOOTER, font=f4, fill=(235, 238, 242, 255), anchor="mm")
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))


def panel_overlay(im, info, fm, ctrl, scale, title=None):
    """Compact overlay for composite panels (PIL image, modified in place via alpha composite)."""
    im = im.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    W, H = im.size
    s = scale
    f1, f2, f3 = font(int(26 * s), True), font(int(17 * s)), font(int(19 * s), True)
    dr.rounded_rectangle([10 * s, 10 * s, W - 10 * s, 104 * s], radius=int(12 * s), fill=(255, 255, 255, 215))
    dr.text((22 * s, 20 * s), title or CTRL_SHORT[ctrl], font=f1, fill=TEXT + (255,))
    name, col = LEVEL[fm["level"]]
    if fm["level"] == 2 and not fm["catch_on"]:
        name = "CATCH (locking)"
    if ctrl == "B0":
        name, col = "NO DEVICE", (154, 160, 166)
    w = dr.textlength(name, font=f3) + 22 * s
    dr.rounded_rectangle([22 * s, 58 * s, 22 * s + w, 92 * s], radius=int(9 * s), fill=col + (255,))
    dr.text((33 * s, 75 * s), name, font=f3, fill=(255, 255, 255, 255), anchor="lm")
    if ctrl != "B0":
        dr.text((34 * s + w, 75 * s), f"{fm['F']:.0f} N (\u2248{fm['F_h']:.0f} N @75 kg)", font=f2,
                fill=TEXT + (255,), anchor="lm")
    if fm["push"]:
        dr.text((W / 2, 140 * s), "PUSH", font=font(int(30 * s), True), fill=(217, 38, 38, 255), anchor="mm")
    txt, colo = outcome_text(info, fm)
    if txt:
        fo = font(int(28 * s), True)
        tw = dr.textlength(txt, font=fo)
        y = H - 80 * s
        dr.rounded_rectangle([W / 2 - tw / 2 - 16 * s, y - 24 * s, W / 2 + tw / 2 + 16 * s, y + 24 * s],
                             radius=int(10 * s), fill=colo + (235,))
        dr.text((W / 2, y), txt, font=fo, fill=(255, 255, 255, 255), anchor="mm")
    return Image.alpha_composite(im, ov).convert("RGB")


def writer(path, fps=30, crf=20):
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=2,
                              ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p", "-preset", "medium"])


def make_single(pk, ctrl):
    info, rd = load(pk, ctrl)
    wr = writer(os.path.join(VID, f"{pk}_{ctrl}.mp4"))
    for img, fm in zip(rd, info["frames"]):
        wr.append_data(overlay_single(img, info, fm, ctrl))
    wr.close()
    rd.close()


def make_compare_b():
    ctrls = ["B0", "B1", "CP"]
    srcs = [load("b", c) for c in ctrls]
    info0 = srcs[0][0]
    wr = writer(os.path.join(VID, "compare_b.mp4"))
    title = (f"Same push, three controllers \u2014 push (b): {info0['F_push']} N \u00d7 {info0['dur']:.1f} s at the pelvis "
             f"(\u0394v {info0['dv']:.2f} m/s) toward the weak (right) side")
    for i, frames in enumerate(zip(*[s[1] for s in srcs])):
        canvas = Image.new("RGB", (1920, 820), (245, 246, 248))
        dr = ImageDraw.Draw(canvas)
        dr.text((960, 28), title, font=font(24, True), fill=TEXT, anchor="mm")
        for j, (img, (info, _)) in enumerate(zip(frames, srcs)):
            fm = info["frames"][min(i, len(info["frames"]) - 1)]
            crop = Image.fromarray(img).crop((320, 0, 960, 720))
            crop = panel_overlay(crop, info, fm, ctrls[j], 1.0)
            canvas.paste(crop, (640 * j, 56))
            if j:
                dr.line([(640 * j, 56), (640 * j, 776)], fill=(255, 255, 255), width=4)
        dr.rectangle([0, 776, 1920, 820], fill=(25, 30, 38))
        dr.text((960, 798), FOOTER, font=font(19), fill=(235, 238, 242), anchor="mm")
        wr.append_data(np.array(canvas))
    wr.close()


def make_grid():
    ctrls = ["B0", "B1", "CP"]
    pushes = ["a", "b", "c"]
    srcs = {(p, c): load(p, c) for p in pushes for c in ctrls}
    wr = writer(os.path.join(VID, "grid_3x3.mp4"))
    readers = {k: iter(v[1]) for k, v in srcs.items()}
    n = min(len(v[0]["frames"]) for v in srcs.values())
    for i in range(n):
        canvas = Image.new("RGB", (1920 + 160, 1080 + 44), (245, 246, 248))
        dr = ImageDraw.Draw(canvas)
        for r, p in enumerate(pushes):
            info = srcs[(p, "B0")][0]
            y = 360 * r
            dr.text((80, y + 150), f"Push ({p})", font=font(24, True), fill=TEXT, anchor="mm")
            dr.text((80, y + 185), f"{info['F_push']} N \u00d7 0.1 s", font=font(18), fill=TEXT, anchor="mm")
            dr.text((80, y + 212), f"\u0394v {info['dv']:.2f} m/s", font=font(18), fill=TEXT, anchor="mm")
            for c_i, c in enumerate(ctrls):
                inf = srcs[(p, c)][0]
                img = next(readers[(p, c)])
                pim = Image.fromarray(img).resize((640, 360), Image.LANCZOS)
                pim = panel_overlay(pim, inf, inf["frames"][i], c, 0.78)
                canvas.paste(pim, (160 + 640 * c_i, y))
        for k in range(1, 3):
            dr.line([(160 + 640 * k, 0), (160 + 640 * k, 1080)], fill=(255, 255, 255), width=3)
            dr.line([(160, 360 * k), (2080, 360 * k)], fill=(255, 255, 255), width=3)
        dr.rectangle([0, 1080, 2080, 1124], fill=(25, 30, 38))
        dr.text((1040, 1102), FOOTER, font=font(19), fill=(235, 238, 242), anchor="mm")
        wr.append_data(np.array(canvas))
    wr.close()


FOOTER_FT = ("Physics simulation: pretrained Unitree G1 policy (and our PPO push-recovery fine-tune) with simulated "
             "paretic weakness — robot-policy demonstration, not a patient model")


def make_beforeafter(pk="b", tag="_ft"):
    """2x2: rows B0 / CP, columns pretrained / fine-tuned policy."""
    cells = [("B0", "", "B0 No assistance · pretrained policy"), ("B0", tag, "B0 No assistance · fine-tuned policy"),
             ("CP", "", "CP CatchPoint · pretrained policy"), ("CP", tag, "CP CatchPoint · fine-tuned (CP re-identified)")]
    srcs = []
    for c, tg, title in cells:
        with open(os.path.join(RAW, f"{pk}_{c}{tg}.json")) as f:
            info = json.load(f)
        srcs.append((c, title, info, imageio.get_reader(os.path.join(RAW, f"{pk}_{c}{tg}.mp4"))))
    info0 = srcs[0][2]
    title = (f"Before / after push-recovery fine-tuning — push ({pk}): {info0['F_push']} N × {info0['dur']:.1f} s "
             f"(Δv {info0['dv']:.2f} m/s) toward the weak (right) side")
    wr = writer(os.path.join(VID, f"beforeafter_{pk}.mp4"))
    n = min(len(s_[2]["frames"]) for s_ in srcs)
    its = [iter(s_[3]) for s_ in srcs]
    for i in range(n):
        canvas = Image.new("RGB", (1920, 56 + 1080 + 44), (245, 246, 248))
        dr = ImageDraw.Draw(canvas)
        dr.text((960, 28), title, font=font(24, True), fill=TEXT, anchor="mm")
        for j, (c, ttl, info, _) in enumerate(srcs):
            img = next(its[j])
            pim = Image.fromarray(img).resize((960, 540), Image.LANCZOS)
            pim = panel_overlay(pim, info, info["frames"][i], c, 0.95, title=ttl)
            canvas.paste(pim, (960 * (j % 2), 56 + 540 * (j // 2)))
        dr.line([(960, 56), (960, 1136)], fill=(255, 255, 255), width=4)
        dr.line([(0, 596), (1920, 596)], fill=(255, 255, 255), width=4)
        dr.rectangle([0, 1136, 1920, 1180], fill=(25, 30, 38))
        dr.text((960, 1158), FOOTER_FT, font=font(18), fill=(235, 238, 242), anchor="mm")
        wr.append_data(np.array(canvas))
    wr.close()


def make_gif(src="compare_b.mp4", dst="compare_b.gif", width=960, fps=12):
    ff = "ffmpeg"
    pal = os.path.join(VID, "_palette.png")
    s = os.path.join(VID, src)
    d = os.path.join(VID, dst)
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", s, "-vf",
                    f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff", pal], check=True)
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", s, "-i", pal, "-lavfi",
                    f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                    d], check=True)
    os.remove(pal)
    return os.path.getsize(d) / 1e6


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor
    what = sys.argv[1:] or ["single", "compare", "grid", "gif"]
    if "beforeafter" in what:
        for pk in "abc":
            if os.path.exists(os.path.join(RAW, f"{pk}_CP_ft.mp4")):
                make_beforeafter(pk)
        mb = make_gif("beforeafter_b.mp4", "beforeafter_b.gif", width=960, fps=12)
        print("beforeafter gif MB", round(mb, 2))
    if "single" in what:
        jobs = [(p, c) for p in "abc" for c in ("B0", "B1", "CP")]
        with ProcessPoolExecutor(9) as ex:
            list(ex.map(make_single, *zip(*jobs)))
    if "compare" in what:
        make_compare_b()
    if "grid" in what:
        make_grid()
    if "gif" in what:
        mb = make_gif()
        if mb > 8.0:
            mb = make_gif(width=800, fps=10)
        print("gif MB", round(mb, 2))
