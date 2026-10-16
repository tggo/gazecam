"""Load the TF1 "Look at Me!" checkpoints into model.GazeWarpNet and run the TF-vs-torch parity test.

    python load_tf.py [--weights DIR] [--repo DIR] [--out parity.json]

Library use:  net = load_tf.load_side("L")   # GazeWarpNet with TF weights, eval mode
"""
import argparse, json, os, sys
import numpy as np
import torch

DEFAULT_WEIGHTS = "/Volumes/A/code/_ext/gcc-weights/weights/warping_model/flx/12"
DEFAULT_REPO = "/Volumes/A/code/_ext/gaze-correction-cam"


def tf_name_to_torch(n):
    """warping_model/<path>/<leaf> -> torch state_dict key (explicit, unknown names raise)."""
    assert n.startswith("warping_model/"), n
    p = n[len("warping_model/"):].split("/")
    leaf = p[-1]
    if p[0] == "encoder":                       # encoder/dnn_blk_i/dnn/{kernel,bias}
        mod = ".".join(["encoder", p[1], p[2]])
    elif p[0] == "warping_module":              # warping_module/{coarse,fine}_level/cnn_blk_i/{cnn,bn_layer}/..  or cnn_4/kernel
        mod = ".".join(["warping_module"] + p[1:-1])
    elif p[0] == "lcm_module":
        mod = ".".join(["lcm_module"] + p[1:-1])
    else:
        raise KeyError(n)
    return {"kernel": mod + ".weight", "bias": mod + ".bias", "gamma": mod + ".weight", "beta": mod + ".bias",
            "moving_mean": mod + ".running_mean", "moving_variance": mod + ".running_var"}[leaf], leaf


def load_checkpoint_into(net, ckpt_dir):
    import tensorflow as tf
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    rd = tf.train.load_checkpoint(ckpt)
    sd, seen = {}, set()
    for name, _ in tf.train.list_variables(ckpt):
        if not name.startswith("warping_model/"):
            continue  # beta1_power, global_step, ...
        if any(s in name for s in ("/Adam", "/RMSProp", "/Momentum")):
            continue  # Adam slots (also /Adam_1) -- not weights
        key, leaf = tf_name_to_torch(name)
        v = torch.from_numpy(np.ascontiguousarray(rd.get_tensor(name)))
        if leaf == "kernel":
            v = v.permute(3, 2, 0, 1) if v.ndim == 4 else v.t()   # HWIO -> OIHW ; (in,out) -> (out,in)
        sd[key] = v.contiguous()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (missing, unexpected)
    return ckpt


def load_side(side, weights=DEFAULT_WEIGHTS):
    from model import GazeWarpNet
    net = GazeWarpNet()
    load_checkpoint_into(net, f"{weights}/{side}/")
    return net.eval()


# ---------------------------------------------------------------- parity ---------------------------------
def build_tf(side, weights, repo):
    import tensorflow as tf
    sys.path.insert(0, repo)
    from tf_models.gaze_corrector_v1 import gaze_warp_model
    tf.compat.v1.disable_eager_execution()
    cfg = gaze_warp_model.ModelConfig(height=48, width=64, encoded_angle_dim=16)
    g = tf.Graph()
    with g.as_default():
        img = tf.compat.v1.placeholder(tf.float32, [None, 48, 64, 3])
        fp = tf.compat.v1.placeholder(tf.float32, [None, 48, 64, 12])
        ang = tf.compat.v1.placeholder(tf.float32, [None, 2])
        pred, raw, lcm = gaze_warp_model.build_inference_graph(img, fp, ang, False, cfg)
        flow = tf.nn.tanh(raw)
        sess = tf.compat.v1.Session(graph=g)
        tf.compat.v1.train.Saver(tf.compat.v1.global_variables()).restore(
            sess, tf.compat.v1.train.latest_checkpoint(f"{weights}/{side}/"))
    return lambda a, b, c: sess.run([pred, flow, lcm], {img: a, fp: b, ang: c})


def anchor_map(pts, h=48, w=64):
    """pts: 6x2 (x,y) in crop px -> (h,w,12) with ch 2k=col-int(px), 2k+1=row-int(py)."""
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    fp = np.zeros((h, w, 12), np.float32)
    for k, (x, y) in enumerate(pts):
        fp[..., 2 * k] = cols - int(x)
        fp[..., 2 * k + 1] = rows - int(y)
    return fp


# Hand-annotated eye landmarks (image px, 6-point order) for frame 30 of ~/Downloads/gaze-sample.mov (1620x1080).
SAMPLE_EYES = {"R": [(662, 675), (688, 664), (714, 664), (738, 677), (714, 684), (688, 684)],   # image-left eye
               "L": [(798, 675), (822, 662), (846, 662), (868, 672), (846, 684), (822, 684)]}   # image-right eye


def real_sample(side="R"):
    """Real eye pixels (gaze-sample.mov frame 30) cropped exactly like the app; landmarks hand-annotated."""
    import cv2
    from crops import crop_eye
    cap = cv2.VideoCapture(os.path.expanduser("~/Downloads/gaze-sample.mov")); cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, fr = cap.read()
    if not ok:
        return None
    c = crop_eye(fr, SEQ_PTS(side), side)
    return c[0][None], c[1][None]


def SEQ_PTS(side):
    return SAMPLE_EYES[side]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="parity.json")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    from model import GazeWarpNet
    res = {}
    for side in "LR":
        tf_run = build_tf(side, a.weights, a.repo)
        net = load_side(side, a.weights)
        rng = np.random.default_rng(0)
        cases = []
        for i in range(a.n):   # random noise inputs, anchor offsets like the converter's check, random angles
            cases.append((rng.random((1, 48, 64, 3), dtype=np.float32),
                          rng.integers(-30, 30, (1, 48, 64, 12)).astype(np.float32),
                          rng.uniform(-20, 20, (1, 2)).astype(np.float32)))
        rs = real_sample(side)
        if rs is not None:
            for ang in ([8, -3], [0, 15], [15, 0], [-10, -12]):
                cases.append((rs[0], rs[1], np.array([ang], np.float32)))
        m = {"flow": 0.0, "lcm": 0.0, "image": 0.0, "image_real_only": 0.0}
        for k, (im, fp, an) in enumerate(cases):
            pt, ft, lt = tf_run(im, fp, an)
            with torch.no_grad():
                pi, fi, li = net.render(torch.from_numpy(im), torch.from_numpy(fp), torch.from_numpy(an))
            m["flow"] = max(m["flow"], float(np.abs(ft - fi.numpy()).max()))
            m["lcm"] = max(m["lcm"], float(np.abs(lt - li.numpy()).max()))
            d = float(np.abs(pt.reshape(pi.shape) - pi.numpy()).max())
            m["image"] = max(m["image"], d)
            if k >= a.n:
                m["image_real_only"] = max(m["image_real_only"], d)
        m["n_cases"] = len(cases); m["real_crop_used"] = rs is not None
        res[side] = m
        print(side, m)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
