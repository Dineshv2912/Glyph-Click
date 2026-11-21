#!/usr/bin/env python3
"""Objectively compare print formats: which one reproduces the photo most accurately on paper?

Every format gets the same face-framed input. Each 1-bit result is put through a model of the
CSN-A1 (dot gain + the white pause row every 24 rows), then blurred like the eye at reading
distance, and scored against the photo with SSIM (whole image and faces only).

usage: .venv/bin/python tools/evaluate.py selfie.png
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_enhance as ae          # noqa: E402
from print_image import halftone, atkinson, IMAGES_DIR   # noqa: E402

STALL = 24


# ---------- renderers: gray uint8 (0=black) -> ink array (1=black dot) ----------
def r_halftone(cell, fine):
    def f(gray):
        img = halftone(Image.fromarray(gray), cell, False, fine)
        return 1 - (np.array(img.convert("L")) > 127).astype(np.float32)
    return f


def error_diffusion(gray, kernel, divisor, lift):
    g = gray.astype(np.float32) / 255
    g = lift + (1 - lift) * g                     # dot-gain compensation (fewer dots)
    h, w = g.shape
    out = np.zeros_like(g)
    for y in range(h):
        for x in range(w):
            old = g[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = 1 - new
            err = (old - new) / divisor
            for dx, dy, k in kernel:
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and yy < h:
                    g[yy, xx] += err * k
    return out


FS = [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]
ATK = [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]   # 6/8 of the error


def r_fs(gray):
    return error_diffusion(gray, FS, 16, 0.15)


def r_atkinson(gray):
    return error_diffusion(gray, ATK, 8, 0.05)


def r_atk_scan(n):
    def f(gray):
        img = atkinson(Image.fromarray(gray), scan=n)
        return 1 - (np.array(img.convert("L")) > 127).astype(np.float32)
    return f


# ---------- paper + eye model ----------
GAIN = np.array([[0.10, 0.30, 0.10], [0.30, 1.0, 0.30], [0.10, 0.30, 0.10]], np.float32)


def on_paper(ink):
    paper = np.clip(cv2.filter2D(ink, -1, GAIN), 0, 1)     # dots spread into neighbours
    paper[STALL - 1::STALL, :] = 0                          # printer pause line (measured on
    paper[0::STALL, :][1:] = 0                              # paper: ~2 rows thick)
    return paper


def eye(darkness, sigma=1.6):
    return cv2.GaussianBlur(darkness, (0, 0), sigma)


def ssim(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (0, 0), 3), cv2.GaussianBlur(b, (0, 0), 3)
    va = cv2.GaussianBlur(a * a, (0, 0), 3) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (0, 0), 3) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (0, 0), 3) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def main():
    name = sys.argv[1]
    path = name if os.path.isfile(name) else os.path.join(IMAGES_DIR, name)
    gray, faces, _ = ae.auto_frame(path, 464)
    gray = cv2.bilateralFilter(gray, 7, 40, 7)
    gray = np.array(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1))
    target = eye(1 - gray.astype(np.float32) / 255)          # what the photo looks like

    formats = {
        "halftone cell 8":       r_halftone(8, False),
        "halftone cell 6 fine":  r_halftone(6, True),
        "halftone cell 4 fine":  r_halftone(4, True),
        "dither Floyd-Steinberg": r_fs,
        "dither Atkinson":       r_atkinson,
        "Atkinson scan 3":       r_atk_scan(3),
        "Atkinson scan 4":       r_atk_scan(4),
        "Atkinson scan 6":       r_atk_scan(6),
    }
    face_mask = np.zeros(gray.shape, bool)
    for x, y, w, h in faces:
        face_mask[max(0, y):y + h, max(0, x):x + w] = True

    results = []
    for label, render in formats.items():
        ink = render(gray)
        s = ssim(target, eye(on_paper(ink)))
        whole = float(s.mean())
        face = float(s[face_mask].mean()) if face_mask.any() else whole
        results.append((0.4 * whole + 0.6 * face, whole, face, label, ink))

    results.sort(reverse=True)
    print(f"{'format':26s} {'score':>6s} {'whole':>6s} {'faces':>6s}")
    for score, whole, face, label, _ in results:
        print(f"{label:26s} {score:6.3f} {whole:6.3f} {face:6.3f}")
    best = results[0]
    print(f"\nMOST ACCURATE: {best[3]}")
    out = os.environ.get("EVAL_OUT")
    if out:
        Image.fromarray(((1 - best[4]) * 255).astype(np.uint8)).save(out)


if __name__ == "__main__":
    main()
