#!/usr/bin/env python3
"""Print any photo on the Cashino CSN-A1 from a laptop - NO ESP32 needed.

The laptop does all the processing (face framing, even tone, Atkinson dither) and sends the
image straight to the printer's own USB port (PL2303, /dev/ttyUSB0) with Cashino DC2 *
commands, which print without white lines.

usage:  python print_image.py photo.jpg --auto            (put photos in ../images/)
        python print_image.py photo.jpg --auto --full     (whole photo, no face crop)
        python print_image.py photo.jpg --auto --preview out.png   (no paper)
"""
import argparse
import os
import sys
import time

import serial
import math

from PIL import Image, ImageDraw, ImageFilter, ImageOps

WIDTH = 384          # printer dots per line
ROW_BYTES = WIDTH // 8
BAND = 24            # must match the firmware


def prepare(path, max_height, rotate, style="atkinson", cell=8, gamma=1.0, crop=False, bold=False, lift=0.2, sharpen=200, clean=0, local=0, fine=False, scan=3):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L")
    if rotate:
        img = img.rotate(90, expand=True)
    h = round(img.height * WIDTH / img.width)
    if h <= max_height:
        img = img.resize((WIDTH, h), Image.LANCZOS)
    elif crop:                               # full width, cut top and bottom
        img = img.resize((WIDTH, h), Image.LANCZOS)
        top = (h - max_height) // 2
        img = img.crop((0, top, WIDTH, top + max_height))
    else:                                    # fit: whole image visible, centred, white sides
        w = round(img.width * max_height / img.height)
        small = img.resize((w, max_height), Image.LANCZOS)
        img = Image.new("L", (WIDTH, max_height), 255)
        img.paste(small, ((WIDTH - w) // 2, 0))
    img = ImageOps.autocontrast(img, cutoff=5 if bold else 1)
    if local:
        # Local contrast: a wide unsharp mask lifts each area against its surroundings,
        # so eyes/mouths stay visible in shadowed or flat-lit faces.
        img = img.filter(ImageFilter.UnsharpMask(radius=24, percent=local, threshold=0))
    if clean:
        # Busy photos: sharpen the subject, then send light background to pure white
        # (levels: <= black_pt prints solid, >= white_pt prints nothing).
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3))
        black_pt, white_pt = 30, clean
        img = img.point(lambda v: 0 if v <= black_pt else 255 if v >= white_pt
                        else round(255 * (v - black_pt) / (white_pt - black_pt)))
    if gamma != 1.0:
        img = img.point(lambda v: round(255 * (v / 255) ** gamma))
    if style == "halftone":
        return halftone(img, cell, bold, fine)
    if style == "atkinson":
        return atkinson(img, scan=scan)
    if sharpen:
        # Local contrast: keeps eyes/mouth from washing out when the tones are lifted.
        img = img.filter(ImageFilter.UnsharpMask(radius=4, percent=sharpen, threshold=2))
    if lift > 0:
        img = tone_for_dot_gain(img, lift)
    return img.convert("1")                  # Floyd-Steinberg dither


def atkinson(gray, lift=0.05, scan=0):
    """Atkinson error diffusion: passes on only 6/8 of the error, so prints come out lighter
    and crisper - measured most accurate on this printer (tools/evaluate.py: about 2x the
    SSIM of any halftone, even counting the printer's pause lines)."""
    w, h = gray.size
    g = [lift + (1 - lift) * v / 255 for v in gray.getdata()]
    out = bytearray(w * h)                      # 0 = white, 255 = black (inverted later)
    for y in range(h):
        row = y * w
        for x in range(w):
            i = row + x
            old = g[i]
            if scan and y % scan == scan - 1:
                new = 1.0                      # forced-white scanline; its ink moves to neighbours
            else:
                new = 1.0 if old >= 0.5 else 0.0
            if new == 0.0:
                out[i] = 255
            err = (old - new) / 8
            if x + 1 < w: g[i + 1] += err
            if x + 2 < w: g[i + 2] += err
            if y + 1 < h:
                j = i + w
                if x > 0: g[j - 1] += err
                g[j] += err
                if x + 1 < w: g[j + 1] += err
                if y + 2 < h: g[j + w] += err
    img = Image.frombytes("L", (w, h), bytes(out))
    return ImageOps.invert(img).convert("1", dither=Image.NONE)


def tone_for_dot_gain(gray, lift):
    """Thermal dots bleed into their neighbours, so a dithered print comes out darker than
    on screen. Following gauravmanek.com/blog/2014/dithering-with-a-dot-matrix-printer:
    an S-shaped cubic curve restores contrast, then [0..1] is lifted to [lift..1] so fewer
    dots are printed overall."""
    def f(v):
        x = v / 255
        x = 3 * x * x - 2 * x * x * x        # cubic S-curve (contrast)
        x = lift + (1 - lift) * x            # brighten: black becomes `lift` grey
        return round(255 * x)
    return gray.point(f)


STALL_ROWS = 24      # the CSN-A1 at 9600 baud pauses every 24 rows and leaves a thin white line


def halftone(gray, cell, bold=False, fine=False):
    """Round dots on a square grid; dot area follows darkness.

    The grid is aligned to the printer's 24-row pauses and dots never cross a 24-row
    boundary, so the white pause lines fall in the gaps between dot rows and disappear.
    """
    if STALL_ROWS % cell:
        cell = max(c for c in (4, 6, 8, 12, 24) if c <= cell) if cell >= 4 else 4
        print(f"cell snapped to {cell} (must divide {STALL_ROWS} to hide the white lines)")
    w, h = gray.size
    out = Image.new("1", (w, h), 1)          # 1 = white
    draw = ImageDraw.Draw(out)
    small = gray.resize((math.ceil(w / cell), math.ceil(h / cell)), Image.BOX)
    ry_max = cell / 2 - 1.5                  # vertical radius stays inside its own row (last row of each cell stays white)
    for gy in range(small.height):
        for gx in range(small.width):
            dark = 1 - small.getpixel((gx, gy)) / 255
            if dark < (0.30 if bold else 0.04):
                continue
            if bold:
                # Dark areas merge sideways into solid bars (the "filled" look); rows stay
                # separated so the printer's block gaps still hide between them.
                r = cell * 0.18 + cell * 0.45 * dark ** 1.6
                if dark > 0.72:
                    r = cell * 0.62           # touching neighbours -> continuous bar
            else:
                r = min(cell * math.sqrt(dark / math.pi), cell * 0.72)   # area ~ darkness
            ry = min(r, cell / 2 - 0.5 if fine else ry_max)
            cx, cy = gx * cell + cell / 2, gy * cell + cell / 2
            draw.ellipse((cx - r, cy - ry, cx + r, cy + ry), fill=0)
    if fine:
        # Full dots, but the last row of EVERY cell is kept white: a fine, even scanline
        # texture. Because cell divides 24, the printer's pause gap lands on one of these
        # rows and disappears among them (a single white row in a solid area would show).
        for y in range(cell - 1, h, cell):
            draw.line((0, y, w - 1, y), fill=1)
    return out


def to_printer_bytes(img1):
    # PIL mode '1' packs MSB-first with 1 = white; the printer wants 1 = black.
    return bytes(b ^ 0xFF for b in img1.tobytes())


IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")  # project images/
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def list_images():
    names = sorted(n for n in os.listdir(IMAGES_DIR) if n.lower().endswith(IMAGE_EXT)) \
        if os.path.isdir(IMAGES_DIR) else []
    if not names:
        return f"No images yet. Put pictures in: {IMAGES_DIR}"
    return ("Images you can print (folder: images/):\n" + "\n".join("  " + n for n in names)
            + "\n\nPrint one with:  python3 tools/print_image.py <name>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="file name in the images/ folder, or any path")
    ap.add_argument("--auto", action="store_true",
                    help="OpenCV: find faces, crop/rotate/expose automatically (needs .venv)")
    ap.add_argument("--full", action="store_true",
                    help="--auto: keep the whole photo (no face crop); faces still set the brightness")
    ap.add_argument("--face-tone", type=float, default=0.52,
                    help="--auto: target brightness of faces, 0=black 1=white (higher = lighter print)")
    ap.add_argument("--max-height", type=int, default=None, help="dots; 8 dots = 1 mm")
    ap.add_argument("--rotate", action="store_true", help="rotate 90 deg (landscape fills the width better)")
    ap.add_argument("--style", choices=["atkinson", "dither", "halftone"], default="atkinson",
                    help="halftone hides the printer's white pause lines; dither shows them")
    ap.add_argument("--cell", type=int, default=8, help="halftone grid size in dots (bigger = bolder dots)")
    ap.add_argument("--gamma", type=float, default=1.0, help=">1 darker, <1 lighter")
    ap.add_argument("--lift", type=float, default=0.2,
                    help="dither only: dot-gain compensation, black is printed as this grey (0 = off)")
    ap.add_argument("--sharpen", type=int, default=200, help="dither only: edge/local contrast boost in %% (0 = off)")
    ap.add_argument("--clean", type=int, nargs="?", const=170, default=0,
                    help="busy photos: whiten light background (optional white point, default 170)")
    ap.add_argument("--local", type=int, nargs="?", const=80, default=0,
                    help="local contrast boost in %% (default 80) - brings out eyes/mouths")
    ap.add_argument("--fine", action="store_true",
                    help="full round dots (more shades); only the printer-gap row is kept white")
    ap.add_argument("--bold", action="store_true", help="bigger dots that merge into bars in dark areas")
    ap.add_argument("--crop", action="store_true", help="fill the full width and cut top/bottom instead of fitting")
    ap.add_argument("--scan", type=int, default=None, choices=[0, 2, 3, 4, 6, 8, 12],
                    help="atkinson: every Nth row white so the printer's pause line hides (0 = off)")
    ap.add_argument("--printer", default="/dev/ttyUSB0", help="printer's USB serial port (default /dev/ttyUSB0)")
    ap.add_argument("--preview", help="save dithered PNG here and do not print")
    a = ap.parse_args()
    if a.scan is None:
        a.scan = 0   # printing over USB with DC2 * has no pause lines; --scan 3 only for the old TTL path
    a.max_height_set = a.max_height is not None
    if a.max_height is None:
        a.max_height = 288
    if not a.image:
        sys.exit(list_images())

    if not os.path.isfile(a.image):
        in_folder = os.path.join(IMAGES_DIR, a.image)
        if os.path.isfile(in_folder):
            a.image = in_folder
        else:
            sys.exit(f"Image not found: {a.image}\n" + list_images())
    if a.auto:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from auto_enhance import auto_prepare
        except ImportError:
            sys.exit("--auto needs OpenCV: run with  .venv/bin/python tools/print_image.py ... --auto")
        if a.style == "atkinson":
            from auto_enhance import auto_frame
            import cv2
            if a.full:
                from auto_enhance import full_frame
                gray, faces, rot = full_frame(a.image, a.max_height if a.max_height_set else 512)
            else:
                gray, faces, rot = auto_frame(a.image, a.max_height if a.max_height_set else 464)
            gray = cv2.bilateralFilter(gray, 7, 40, 7)
            from auto_enhance import even_tone
            gray, g = even_tone(gray, faces, a.face_tone)
            print(f"auto: {len(faces)} face(s), rotate={'yes' if rot else 'no'}, atkinson, face gamma {g:.2f}")
            img = atkinson(Image.fromarray(gray), scan=a.scan)
        else:
            res = auto_prepare(a.image, a.max_height if a.max_height_set else 464)
            if res is None:
                print("auto: no faces found, using the default style")
                img = None
            else:
                gray, cell, fine, info = res
                print(info)
                img = halftone(gray, cell, False, fine)
    else:
        img = None
    if img is None:
        img = prepare(a.image, a.max_height, a.rotate, a.style, a.cell, a.gamma, a.crop, a.bold, a.lift, a.sharpen, a.clean, a.local, a.fine, a.scan)
    print(f"{WIDTH}x{img.height} dots = {img.height / 8:.0f} mm of paper")
    if a.preview:
        img.save(a.preview)
        print("preview saved:", a.preview)
        return

    data = to_printer_bytes(img)
    # DC2 * r n data: Cashino bitmap command. Unlike GS v 0 (which pauses every 24 rows and
    # leaves white lines) it prints r rows in one continuous run. r <= 255, so tall images are
    # split into equal chunks. The printer's USB takes data far faster than it prints.
    h = img.height
    nchunks = -(-h // 255)
    step = -(-h // nchunks)
    job = b"\x1b@"
    for y in range(0, h, step):
        r = min(step, h - y)
        job += b"\x12*" + bytes([r, ROW_BYTES]) + data[y * ROW_BYTES:(y + r) * ROW_BYTES]
    job += b"\x1bd\x03"                       # feed 3 lines: the head sits behind the tear bar
    try:
        with serial.Serial(a.printer, 115200, timeout=2, write_timeout=30) as up:
            up.write(job)
            up.flush()
    except serial.SerialException as e:
        sys.exit(f"cannot open the printer at {a.printer}: {e}\n(is its USB cable plugged in? see: ls /dev/ttyUSB*)")
    print(f"printed over the printer's USB ({len(job)} bytes)")


if __name__ == "__main__":
    main()
