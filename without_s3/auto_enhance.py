"""--auto: OpenCV pipeline that picks crop, rotation, exposure and dot size from the faces.

Needs the project venv (.venv: opencv-python-headless + numpy) and the YuNet model in
tools/models/. Returns a grayscale PIL image (384 wide) plus the halftone settings to use.
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

WIDTH = 384
PAPER_GAMMA = 0.7   # approved on paper 2026-11-19 (selfie, cell 6 fine)
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                     "face_detection_yunet_2023mar.onnx")


def detect_faces(bgr):
    """YuNet face detector -> list of (x, y, w, h) in image pixels."""
    if not os.path.isfile(MODEL):
        print("note: face model missing, using the whole image")
        return []
    h, w = bgr.shape[:2]
    # YuNet works best on images around 640 px; detect on a scaled copy.
    s = min(1.0, 640 / max(h, w))
    small = cv2.resize(bgr, (round(w * s), round(h * s))) if s < 1 else bgr
    det = cv2.FaceDetectorYN.create(MODEL, "", (small.shape[1], small.shape[0]),
                                    score_threshold=0.7)
    _, faces = det.detect(small)
    if faces is None:
        return []
    found = [tuple(int(v / s) for v in f[:4]) for f in faces]
    # Keep the main subject(s): drop faces under half the width of the largest one
    # (background people, posters, a small inset copy of the same photo).
    biggest = max(f[2] for f in found)
    return [f for f in found if f[2] >= 0.5 * biggest]


def auto_frame(path, max_height):
    """Face-based crop + rotate + scale to 384 wide. Returns (gray uint8, faces, rotated)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image: {path}")
    H, W = bgr.shape[:2]
    faces = detect_faces(bgr)

    # 1. Crop to the people: union of faces, widened for hair/shoulders.
    if faces:
        fw = sum(f[2] for f in faces) / len(faces)
        fh = sum(f[3] for f in faces) / len(faces)
        x0 = max(0, int(min(f[0] for f in faces) - 0.6 * fw))
        x1 = min(W, int(max(f[0] + f[2] for f in faces) + 0.6 * fw))
        y0 = max(0, int(min(f[1] for f in faces) - 0.8 * fh))
        y1 = min(H, int(max(f[1] + f[3] for f in faces) + 0.8 * fh))
        bgr = bgr[y0:y1, x0:x1]
        faces = [(x - x0, y - y0, w, h) for x, y, w, h in faces]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2. Rotate when the crop is wide: the long side then runs along the paper (bigger print).
    # Choose the orientation that prints the people largest within the paper width and
    # the height limit (a long thin strip turned sideways would hit the height limit).
    ch, cw = gray.shape
    scale_upright = min(WIDTH / cw, max_height / ch)
    scale_sideways = min(WIDTH / ch, max_height / cw)
    fills_width = ch * scale_sideways >= 0.8 * WIDTH      # sideways must still use the paper
    rotate = fills_width and scale_sideways > scale_upright * 1.05
    if rotate:
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h0 = bgr.shape[0]
        faces = [(y, bgr.shape[1] - x - w, h, w) for x, y, w, h in faces]
        del h0

    # 3. Scale to the paper width (fit inside max_height, centred on white).
    h, w = gray.shape
    scale = WIDTH / w
    if h * scale > max_height:
        scale = max_height / h
    nw, nh = round(w * scale), round(h * scale)
    gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    faces = [tuple(round(v * scale) for v in f) for f in faces]
    if nw < WIDTH:
        pad = (WIDTH - nw) // 2
        gray = cv2.copyMakeBorder(gray, 0, 0, pad, WIDTH - nw - pad, cv2.BORDER_CONSTANT, value=255)
        faces = [(x + pad, y, fw_, fh_) for x, y, fw_, fh_ in faces]

    return gray, faces, rotate


def auto_prepare(path, max_height):
    gray, faces, rotate = auto_frame(path, max_height)
    # 4. Edge-preserving denoise: smooth skin/background, keep eyes and mouth sharp.
    gray = cv2.bilateralFilter(gray, 7, 40, 7)

    if not faces:
        return None   # nothing to aim at: caller uses the user's fixed default look

    # 5. Same tone steps as the hand-tuned print the user approved:
    #    autocontrast -> local contrast -> clean (light background to white).
    img = ImageOps.autocontrast(Image.fromarray(gray), cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=24, percent=80, threshold=0))
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3))
    img = img.point(lambda v: 0 if v <= 30 else 255 if v >= 170 else round(255 * (v - 30) / 140))
    gray = np.array(img)

    # 6. Tone: the gamma proven on paper (hand-tuned selfie print the user approved).
    #    A computed "skin target" looked right on screen but printed too light.
    gamma = PAPER_GAMMA
    lut = np.array([round(255 * (i / 255) ** gamma) for i in range(256)], np.uint8)
    gray = cv2.LUT(gray, lut)

    # 7. Fade what is not a person: outside the face ellipses + bodies below them,
    #    blend towards white so leftover clutter loses its dots.
    mask = np.zeros(gray.shape, np.uint8)
    for x, y, w_, h_ in faces:
        cx, cy = x + w_ // 2, y + h_ // 2
        cv2.ellipse(mask, (cx, cy), (int(w_ * 0.85), int(h_ * 0.95)), 0, 0, 360, 255, -1)
        cv2.rectangle(mask, (int(cx - w_ * 1.1), cy), (int(cx + w_ * 1.1), gray.shape[0]), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 12).astype(np.float32) / 255
    faded = cv2.addWeighted(gray, 0.4, np.full_like(gray, 255), 0.6, 0)
    gray = (gray * mask + faded * (1 - mask)).astype(np.uint8)

    # 8. Dot size: big faces get the classic 8-dot cells, small/group faces finer 6-dot
    #    cells with the even scanline texture.
    face_w = max(f[2] for f in faces)
    cell, fine = (8, False) if face_w >= 200 and len(faces) == 1 else (6, True)

    info = (f"auto: {len(faces)} face(s), rotate={'yes' if rotate else 'no'}, "
            f"exposure gamma={gamma:.2f}, cell={cell}{' fine' if fine else ''}")
    return Image.fromarray(gray), cell, fine, info


def even_tone(gray, faces, target=0.62):
    """Same look for every photo: stretch contrast, add mild local contrast (CLAHE), then
    choose a gamma that puts the median face brightness at `target`. Dark photos get
    lifted, washed-out photos get deepened. Returns (gray, gamma)."""
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    lo, hi = np.percentile(g, (1, 99))
    if hi > lo:
        g = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    g = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6)).apply(g)
    vals = [g[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4] for x, y, w, h in faces
            if w > 8 and h > 8]
    vals = np.concatenate([v.ravel() for v in vals if v.size]) if vals else g.ravel()
    m = min(max(float(np.median(vals)) / 255, 0.05), 0.95)
    gamma = min(max(math.log(target) / math.log(m), 0.8), 2.2)   # never lift a lot: washes faces out
    lut = np.array([round(255 * (i / 255) ** gamma) for i in range(256)], np.uint8)
    return cv2.LUT(g, lut), gamma


def full_frame(path, max_height):
    """Whole photo, no crop: turned sideways if that prints it larger, scaled to the paper.
    Faces are detected only so even_tone() can set brightness from them."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image: {path}")
    faces = detect_faces(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    up = min(WIDTH / w, max_height / h)
    side = min(WIDTH / h, max_height / w)
    rotate = side > up * 1.05
    if rotate:
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces = [(y, w - x - fw, fh, fw) for x, y, fw, fh in faces]
        h, w = w, h
    s = min(WIDTH / w, max_height / h)
    nw, nh = round(w * s), round(h * s)
    gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    faces = [tuple(round(v * s) for v in f) for f in faces]
    if nw < WIDTH:
        pad = (WIDTH - nw) // 2
        gray = cv2.copyMakeBorder(gray, 0, 0, pad, WIDTH - nw - pad, cv2.BORDER_CONSTANT, value=255)
        faces = [(x + pad, y, fw, fh) for x, y, fw, fh in faces]
    return gray, faces, rotate
