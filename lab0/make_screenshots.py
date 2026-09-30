"""Render the required hand-in PNGs from the executed notebook."""
import json, os, re, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

SID = os.environ.get("STUDENT_ID", "314551011")
NB = sys.argv[1] if len(sys.argv) > 1 else f"{SID}.ipynb"

MONO = "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"

BG = (30, 30, 30)
BAR = (55, 56, 58)
FG = (212, 212, 212)
ACCENT = (86, 182, 194)
SCALE = 2  # render at 2x for a crisp "screenshot"


def cell_text(nb, idx, only_stdout=True):
    """Text output of code cell `idx`.

    tqdm writes its progress bars to stderr with bare carriage returns, which
    otherwise get glued onto the front of the stdout lines we want to show.
    """
    out = []
    for o in nb["cells"][idx].get("outputs", []):
        if o.get("output_type") == "stream":
            if only_stdout and o.get("name") != "stdout":
                continue
            out.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/plain" in d:
                out.append("".join(d["text/plain"]))
        elif o.get("output_type") == "error":
            out.append("\n".join(o.get("traceback", [])))
    # each output entry is a separate block; a tqdm widget's text/plain has no
    # trailing newline, so without this the next print() is glued onto it
    text = "".join(c if c.endswith("\n") else c + "\n" for c in out if c)
    # keep only the final segment of each carriage-return-overwritten line
    lines = []
    for raw in text.splitlines():
        seg = raw.split("\r")[-1]
        if re.search(r"\d+%\|", seg) or "it/s]" in seg or "B/s]" in seg:
            continue
        lines.append(seg.rstrip())
    return "\n".join(lines)


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


def render(lines, path, title, fontsize=15, pad=18):
    fs = fontsize * SCALE
    pad *= SCALE
    font = ImageFont.truetype(MONO, fs)
    font_b = ImageFont.truetype(MONO_B, fs)
    lh = int(fs * 1.45)
    barh = int(fs * 2.0)

    width = max([font.getlength(l) for l in lines] + [font_b.getlength(title) + 90 * SCALE])
    W = int(width) + 2 * pad
    H = barh + len(lines) * lh + 2 * pad

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # title bar with traffic lights
    d.rectangle([0, 0, W, barh], fill=BAR)
    r = int(fs * 0.32)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = pad + i * int(fs * 1.2)
        cy = barh // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    d.text((pad + int(fs * 4.2), (barh - fs) // 2 - 2), title, font=font_b, fill=(190, 190, 190))

    y = barh + pad
    for line in lines:
        col = FG
        if line.startswith(("final accuracy", "accuracy:", "Speedup")):
            col = ACCENT
        d.text((pad, y), line, font=font, fill=col)
        y += lh

    img.save(path)
    print("wrote", path, img.size)


nb = json.load(open(NB))

# ---- locate cells by a marker in their source (indices are stable but check) --
def find(marker, start=0):
    for i in range(start, len(nb["cells"])):
        c = nb["cells"][i]
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return i
    raise SystemExit(f"cell containing {marker!r} not found")


i_train = find("final accuracy")
i_eval2 = find('print(f"accuracy: {acc}")')
i_orig = find("Time taken without torch.compile")
i_comp = find("Time taken with torch.compile")
i_speed = find("Speedup:")

# ---------------------------------------------------------------- acc_1
t = strip_ansi(cell_text(nb, i_train)).strip().splitlines()
t = [l for l in t if l.strip()]
render(t, f"{SID}_acc_1.png", "Part 1 — accuracy after training")

# ---------------------------------------------------------------- acc_2
t = strip_ansi(cell_text(nb, i_eval2)).strip().splitlines()
t = [l for l in t if l.strip()]
render(t, f"{SID}_acc_2.png", "Part 1 — accuracy after loading model.pt")

# ---------------------------------------------------------------- speedup
def timings(idx, tag):
    rows = []
    for l in strip_ansi(cell_text(nb, idx)).splitlines():
        if l.startswith("Time taken"):
            rows.append(l)
    return rows

lines = []
lines += timings(i_orig, "without")
lines.append("")
lines += timings(i_comp, "with")
lines.append("")
lines += [l for l in strip_ansi(cell_text(nb, i_speed)).strip().splitlines() if l.strip()]
render(lines, f"{SID}_speedup.png", "Part 2 — torch.compile speedup", fontsize=14)
