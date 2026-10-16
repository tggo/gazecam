"""Per-user training pairs from GazeCam capture sessions (~/Movies/GazeCam/dataset/<session>/).

Pair = (input eye crop while looking at screen point P,  target eye crop of the SAME session while looking at the camera,
        closest in time, optionally similarity-aligned).  angle = sign * (cam_dir - P_dir) in degrees [vertical, horizontal].

Crops replicate GazeEffect.swift correct() exactly (crops.py).  "R" model = image-left eye, "L" = image-right eye.

    python dataset.py --sessions ~/Movies/GazeCam/dataset/* --stats          # how many pairs survive each filter
    python dataset.py --sessions ... --sign-check                              # empirical angle sign with the pretrained model
"""
import argparse, csv, glob, os, sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from crops import crop_eye, crop_box, crop_landmarks, SEQ, H, W

DEFAULT_ROOT = os.path.expanduser("~/Movies/GazeCam/dataset")


# ------------------------------------------------------------------ sessions ----------------------------------
def load_session(path):
    """-> list of row dicts (with parsed 'lm' (12,2) or None), or None if the session has no `kind`/`lm` columns."""
    f = os.path.join(path, "labels.csv")
    if not os.path.exists(f):
        return None
    with open(f, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames is None or "kind" not in rd.fieldnames or "lm" not in rd.fieldnames:
            return None
        rows = list(rd)
    for i, r in enumerate(rows):
        r["idx"] = i
        r["path"] = os.path.join(path, r["file"])
        v = r["lm"].split()
        r["lm"] = np.array(v, np.float64).reshape(12, 2) if len(v) == 24 else None
        for k in ("target_h_deg", "target_v_deg", "cam_h_deg", "cam_v_deg", "openness"):
            r[k] = float(r[k])
    # block id (run of identical point label) + position inside the block, to drop saccade/settling frames
    b = 0
    for i, r in enumerate(rows):
        if i and (r["point"] != rows[i - 1]["point"] or r["kind"] != rows[i - 1]["kind"]):
            b += 1
        r["block"] = b
    start = {}
    for r in rows:
        start.setdefault(r["block"], r["idx"])
    for r in rows:
        r["in_block"] = r["idx"] - start[r["block"]]
    # the app saved at 30 Hz from a ~24 fps camera: ~20% of frames are byte-identical repeats -> mark them (hash of the jpg)
    import hashlib
    seen = set()
    for r in rows:
        h = hashlib.md5(open(r["path"], "rb").read()).hexdigest()
        r["dup"] = h in seen
        seen.add(h)
    med = np.median([r["openness"] for r in rows])
    for r in rows:
        r["open_rel"] = r["openness"] / max(med, 1e-6)
    return rows


def eye_pts(r, side):
    """6 points of the eye the given model handles, in the 6-point order (points as the app passes them)."""
    return r["lm"][0:6] if side == "R" else r["lm"][6:12]


# ---------------------------------------------------------------- pairing -------------------------------------
class Stats(dict):
    def bump(self, k, n=1):
        self[k] = self.get(k, 0) + n


def sim_align_crop(frame, pts_tgt, pts_in, box_in):
    """Crop `box_in` of the INPUT frame geometry out of the target frame after a similarity transform mapping the target
    eye landmarks onto the input eye landmarks.  Returns (img48x64 float BGR [0,1], residual_rms_px, scale, rot_deg)."""
    M, _ = cv2.estimateAffinePartial2D(np.asarray(pts_tgt, np.float32), np.asarray(pts_in, np.float32), method=cv2.LMEDS)
    if M is None:
        return None
    left, top, right, bottom = box_in
    S = np.array([[W / (right - left), 0, 0], [0, H / (bottom - top), 0], [0, 0, 1]])
    T = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
    A = (S @ T @ np.vstack([M, [0, 0, 1]]))[:2]
    out = cv2.warpAffine(frame.astype(np.float32), A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / 255
    q = np.c_[np.asarray(pts_tgt), np.ones(6)] @ M.T
    rms = float(np.sqrt(((q - pts_in) ** 2).sum(1).mean()))
    return out, rms, float(np.hypot(M[0, 0], M[1, 0])), float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def build_pairs(session_dirs, side, tol_px=2.0, align="none", settle=6, min_open=0.5, max_gap=300,
                min_mag=0.0, verbose=True):
    """Returns (pairs, Stats).  pair = dict(img, tgt, anchors(6,2 int), delta=(cam_v-P_v, cam_h-P_h) RAW (unsigned), session, file).
    align: 'none' -> reject if either eye corner moved > tol_px between input and target frame; 'sim' -> similarity-align the
    target onto the input landmarks (reject if residual rms > tol_px or scale/rotation implausible)."""
    pairs, st = [], Stats()
    for sd in session_dirs:
        rows = load_session(sd)
        name = os.path.basename(sd.rstrip("/"))
        if rows is None:
            st.bump("sessions_skipped_old_format"); continue
        st.bump("sessions_used")
        ok = lambda r: not r["dup"] and r["lm"] is not None and r["open_rel"] >= min_open and r["in_block"] >= settle
        cams = [r for r in rows if r["kind"] == "camera" and ok(r)]
        scr = [r for r in rows if r["kind"] == "screen" and ok(r)]
        st.bump("frames_total", len(rows))
        st.bump("frames_duplicate_dropped", sum(r["dup"] for r in rows))
        st.bump("screen_frames_total", sum(r["kind"] == "screen" for r in rows))
        st.bump("screen_frames_after_blink_settle_lm_filter", len(scr))
        st.bump("camera_frames_usable", len(cams))
        cam_idx = np.array([c["idx"] for c in cams])
        for r in scr:
            if np.hypot(r["cam_h_deg"] - r["target_h_deg"], r["cam_v_deg"] - r["target_v_deg"]) < min_mag:
                continue
            p_in = eye_pts(r, side)
            box_in = crop_box(p_in)
            if box_in is None:
                st.bump("reject_box"); continue
            frame = None
            for j in np.argsort(np.abs(cam_idx - r["idx"]), kind="stable"):
                c = cams[j]
                if abs(c["idx"] - r["idx"]) > max_gap:
                    st.bump("reject_no_registered_camera_within_gap"); break
                p_t = eye_pts(c, side)
                if align == "none":
                    corner = max(np.hypot(*(p_t[0] - p_in[0])), np.hypot(*(p_t[3] - p_in[3])))
                    if corner > tol_px:
                        continue
                if frame is None:
                    frame = cv2.imread(r["path"])
                    if frame is None:
                        break
                cf = cv2.imread(c["path"])
                if cf is None:
                    continue
                if align == "none":
                    t = crop_eye(cf, p_t, side)
                    ci = crop_eye(frame, p_in, side)
                    if ci is None or t is None:
                        st.bump("reject_edge"); break
                    # target must be cropped with the INPUT's box so both crops are pixel-registered
                    left, top, right, bottom = box_in
                    tgt = cv2.resize(cf[top:bottom, left:right].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) / 255
                    img, fp, _ = ci
                else:
                    ci = crop_eye(frame, p_in, side)
                    if ci is None:
                        st.bump("reject_edge"); break
                    res = sim_align_crop(cf, p_t, p_in, box_in)
                    if res is None:
                        continue
                    tgt, rms, sc, rot = res
                    if rms > tol_px or not (0.85 < sc < 1.18) or abs(rot) > 10:
                        continue
                    img, fp, _ = ci
                a6 = np.array([[int((p_in[i, 0] - box_in[0]) * W / (box_in[2] - box_in[0])),
                                int((p_in[i, 1] - box_in[1]) * H / (box_in[3] - box_in[1]))] for i in SEQ[side]])
                pairs.append(dict(img=img, tgt=tgt.astype(np.float32), anchors=a6.astype(np.int32),
                                  delta=np.array([c["cam_v_deg"] - r["target_v_deg"], c["cam_h_deg"] - r["target_h_deg"]], np.float32),
                                  session=name, file=r["file"], gap=abs(c["idx"] - r["idx"])))
                st.bump("pairs")
                break
            else:
                st.bump("reject_no_registered_camera_frame")
    return pairs, st


def anchor_map(anchors):
    """anchors (6,2) int (x,y) in crop px -> (48,64,12) like the app."""
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    fp = np.zeros((H, W, 12), np.float32)
    for k, (x, y) in enumerate(anchors):
        fp[..., 2 * k] = cols - x
        fp[..., 2 * k + 1] = rows - y
    return fp


class PairDataset(Dataset):
    def __init__(self, pairs, sign=(1.0, 1.0), augment=False):
        self.p, self.sign, self.augment = pairs, np.array(sign, np.float32), augment

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        p = self.p[i]
        img, tgt = p["img"], p["tgt"]
        if self.augment:  # same photometric gain/offset on both, so the target stays consistent with the input
            g, b = np.random.uniform(0.85, 1.15), np.random.uniform(-0.04, 0.04)
            img, tgt = np.clip(img * g + b, 0, 1), np.clip(tgt * g + b, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(img, np.float32)), torch.from_numpy(anchor_map(p["anchors"])),
                torch.from_numpy(p["delta"] * self.sign), torch.from_numpy(np.ascontiguousarray(tgt, np.float32)))


def resolve_sessions(args):
    out = []
    for s in args:
        out += sorted(glob.glob(os.path.expanduser(s)))
    return [s for s in out if os.path.isdir(s)]


# ------------------------------------------------------------ sign convention ---------------------------------
@torch.no_grad()
def l1_for_sign(net, pairs, sign, n=400, crop=3, seed=0):
    ds = PairDataset(pairs, sign)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    tot, base = 0.0, 0.0
    for i in idx:
        img, fp, ang, tgt = (x[None] for x in ds[i])
        pred = net.render(img, fp, ang)[0]
        tot += (pred - tgt)[:, crop:-crop, crop:-crop].abs().mean().item()
        base += (img - tgt)[:, crop:-crop, crop:-crop].abs().mean().item()
    return tot / len(idx), base / len(idx)


def sign_check(sessions, sides=("R", "L"), **kw):
    """Try all four (sign_v, sign_h) on real pairs with the PRETRAINED weights; lower L1 to the camera target wins.
    Vertical and horizontal are decided independently (each compared with its own flip, other axis held at its best)."""
    import load_tf
    best = {}
    for side in sides:
        pairs, st = build_pairs(sessions, side, min_mag=8.0, **kw)
        net = load_tf.load_side(side)
        print(f"[sign-check] side {side}: {len(pairs)} pairs with |angle|>=8 deg  (identity/no-op L1 = {l1_for_sign(net, pairs, (0, 0))[0]:.5f})")
        res = {}
        for sv in (1.0, -1.0):
            for sh in (1.0, -1.0):
                l1, base = l1_for_sign(net, pairs, (sv, sh))
                res[(sv, sh)] = l1
                print(f"   sign(v)={sv:+.0f} sign(h)={sh:+.0f}  mean L1 to camera target = {l1:.5f}   (input-as-prediction baseline {base:.5f})")
        best[side] = min(res, key=res.get)
        print(f"   -> best for {side}: sign(v)={best[side][0]:+.0f} sign(h)={best[side][1]:+.0f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=[DEFAULT_ROOT + "/*"])
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--align", choices=["none", "sim"], default="none")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--sign-check", action="store_true")
    a = ap.parse_args()
    ss = resolve_sessions(a.sessions)
    if a.stats:
        for side in "RL":
            for al, tol in (("none", a.tol), ("none", 4.0), ("sim", 1.5)):
                pairs, st = build_pairs(ss, side, tol_px=tol, align=al)
                print(f"side {side} align={al} tol={tol}px: {dict(st)}")
    if a.sign_check:
        sign_check(ss, tol_px=a.tol, align=a.align)


if __name__ == "__main__":
    main()
