"""Frame-level data for the per-user gaze estimator (labels from the same capture sessions as the warp pairs).

Label per frame = (gaze_v, gaze_h) = (target_v_deg, target_h_deg)  [camera frames: target == cam].
This is the sign convention of the warp training: warp angle = (cam_v - gaze_v, gaze_h - cam_h).
"""
import glob, os
import numpy as np
import cv2
import dataset as D
from crops import crop_box, SEQ, H, W

MARGIN = 10          # px of raw context kept around each eye box so crop-jitter can re-crop
SESSIONS = {"train": ["2026-10-16_19-06-53", "2026-10-16_19-07-46", "2026-10-16_19-08-34", "2026-10-16_19-10-17"],
            "val": ["2026-10-16_19-09-30"], "test": ["2026-10-16_19-11-19"]}


def _roi(frame, pts):
    """Raw-pixel ROI around the app's crop box (+MARGIN, replicate-padded) and the box in ROI coordinates."""
    box = crop_box(pts)
    if box is None:
        return None
    l, t, r, b = box
    fh, fw = frame.shape[:2]
    if t < 0 or l < 0 or b > fh or r > fw:      # the app skips these frames
        return None
    pad = cv2.copyMakeBorder(frame, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_REPLICATE)
    roi = pad[t:b + 2 * MARGIN, l:r + 2 * MARGIN].copy()      # = frame[t-M : b+M, l-M : r+M]
    return roi, (MARGIN, MARGIN, MARGIN + (r - l), MARGIN + (b - t))


def crop_from_roi(roi, box, dx=0.0, dy=0.0, scale=1.0):
    """Float crop 48x64 BGR [0,1] from an ROI; with dx=dy=0, scale=1 identical to crops.crop_eye."""
    l, t, r, b = box
    cx, cy, hw, hh = (l + r) / 2 + dx, (t + b) / 2 + dy, (r - l) / 2 * scale, (b - t) / 2 * scale
    L, T = int(round(cx - hw)), int(round(cy - hh))
    R, B = L + int(round(2 * hw)), T + int(round(2 * hh)) if scale != 1 or dx or dy else b
    if scale == 1 and not dx and not dy:
        L, T, R, B = l, t, r, b
    L, T = max(L, 0), max(T, 0)
    R, B = min(R, roi.shape[1]), min(B, roi.shape[0])
    return cv2.resize(roi[T:B, L:R].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) / 255


def lm_features(lm, fw, fh):
    """12 landmarks -> 27 numbers: 24 coords relative to the mid-point of the two eye-centres, divided by the inter-ocular
    distance, + (mid_x/fw - .5, mid_y/fh - .5, 10*iod/fw)."""
    cR, cL = lm[[0, 3]].mean(0), lm[[6, 9]].mean(0)
    mid, iod = (cR + cL) / 2, max(np.hypot(*(cL - cR)), 1.0)
    rel = ((lm - mid) / iod).reshape(-1)
    return np.concatenate([rel, [mid[0] / fw - 0.5, mid[1] / fh - 0.5, 10 * iod / fw]]).astype(np.float32)


def load_frames(session_names, root=D.DEFAULT_ROOT, settle=6, min_open=0.5, near_dup=0.002):
    """-> list of dicts(roiR, boxR, roiL, boxL, lm(27), y(2)=[v,h], point, kind, session, pupil_x) and counts."""
    out, cnt = [], {}
    bump = lambda k, n=1: cnt.__setitem__(k, cnt.get(k, 0) + n)
    for name in session_names:
        rows = D.load_session(os.path.join(root, name))
        assert rows is not None, name
        prev = None
        for r in rows:
            bump("frames")
            if r["dup"]: bump("dup_md5"); continue
            if r["lm"] is None: bump("no_landmarks"); continue
            if r["in_block"] < settle: bump("settle"); continue
            if r["open_rel"] < min_open: bump("blink"); continue
            fr = cv2.imread(r["path"])
            a, b = _roi(fr, r["lm"][:6]), _roi(fr, r["lm"][6:])
            if a is None or b is None: bump("edge"); continue
            cR, cL = crop_from_roi(*a), crop_from_roi(*b)
            if prev is not None and prev[2] == r["block"] and np.abs(cR - prev[0]).mean() + np.abs(cL - prev[1]).mean() < near_dup:
                bump("near_dup"); continue
            prev = (cR, cL, r["block"])
            out.append(dict(roiR=a[0], boxR=a[1], roiL=b[0], boxL=b[1], lm=lm_features(r["lm"], fr.shape[1], fr.shape[0]),
                            y=np.array([r["target_v_deg"], r["target_h_deg"]], np.float32), point=r["point"], kind=r["kind"],
                            session=name, pupil_x=float(r["pupil_x"]), eye_dist_cm=float(r["eye_dist_cm"])))
            bump("kept")
    return out, cnt
