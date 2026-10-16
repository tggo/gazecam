"""Eye-crop geometry, a line-for-line port of GazeEffect.swift `correct()` (crop box, resize, anchor map)."""
import numpy as np
import cv2

H, W = 48, 64
SEQ = {"R": [0, 1, 2, 3, 4, 5],   # "R" model: image-LEFT eye, points in 6-point order
       "L": [3, 2, 1, 0, 5, 4]}   # "L" model: image-RIGHT eye, mirrored order


def crop_box(pts):
    """pts: 6x2 float image px (6-point order of the eye as listed, i.e. before SEQ). Returns (left,top,right,bottom) ints
    or None. Swift Int() truncates toward zero; coordinates are positive so int() is identical."""
    pts = np.asarray(pts, np.float64)
    cx, cy = pts[[0, 3]].mean(0)               # corner midpoint (Swift: `center`)
    eye_len = abs(pts[3, 0] - pts[0, 0])
    half_w = eye_len * 3 / 4
    box_h = 1.5 * half_w
    top, bottom = int(cy - box_h * 7 / 12), int(cy + box_h * 5 / 12)
    left, right = int(cx - half_w), int(cx + half_w)
    if bottom - top < 12 or right - left < 16:
        return None
    return left, top, right, bottom


def crop_eye(frame_bgr, pts, side):
    """frame_bgr uint8 (h,w,3) BGR; pts 6x2 of that eye in 6-point order; side 'R' (image-left eye) or 'L'.
    Returns (img (48,64,3) float32 BGR [0,1], fp (48,64,12) float32, box) or None if the box leaves the frame."""
    pts = np.asarray(pts, np.float64)
    box = crop_box(pts)
    if box is None:
        return None
    left, top, right, bottom = box
    fh, fw = frame_bgr.shape[:2]
    if top < 0 or left < 0 or bottom > fh or right > fw:
        return None
    ow, oh = right - left, bottom - top
    # float math like the Swift crop (no uint8 rounding of the resampled values)
    img = cv2.resize(frame_bgr[top:bottom, left:right].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) / 255
    fp = np.zeros((H, W, 12), np.float32)
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    for k, i in enumerate(SEQ[side]):
        ax = float(int((pts[i, 0] - left) * W / ow))
        ay = float(int((pts[i, 1] - top) * H / oh))
        fp[..., 2 * k] = cols - ax
        fp[..., 2 * k + 1] = rows - ay
    return img, fp, box


def crop_landmarks(pts, box):
    """6x2 landmarks in 48x64 crop coordinates (float), for alignment / debugging."""
    left, top, right, bottom = box
    p = np.asarray(pts, np.float64) - [left, top]
    return p * [W / (right - left), H / (bottom - top)]
