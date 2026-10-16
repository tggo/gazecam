"""Convert the L/R gaze-warp TF1 checkpoints (chihfanhsu "Look at Me!", via WangWilly/gaze-correction-cam)
to Core ML .mlpackage files.

usage: python tools/convert_gaze.py <gaze-correction-cam repo> <weights dir> <out dir>
Inputs per model: img (1,48,64,3) BGR in [0,1], fp (1,48,64,12) anchor map, ang (1,2) [vertical, horizontal] degrees.
Outputs: flow (1,48,64,2) = tanh(raw_flow), lcm (1,48,64,2). The bilinear warp + light blend do not convert
(coremltools chokes on the model's hand-written gather sampler), so the app does them; `warp()` below is the
reference that GazeEffect.swift mirrors, and the parity check runs through it.
"""
import sys
import numpy as np
import tensorflow as tf
import coremltools as ct

repo, weights, out = sys.argv[1:4]
sys.path.insert(0, repo)
from tf_models.gaze_corrector_v1 import gaze_warp_model  # noqa: E402

tf.compat.v1.disable_eager_execution()
cfg = gaze_warp_model.ModelConfig(height=48, width=64, encoded_angle_dim=16)

def warp(img, flow, lcm):
    """Backward bilinear warp exactly as tf_models spatial_transform (clamped corners, unclamped weights)."""
    H, W, _ = img.shape
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    px = 0.5 * (gx + flow[..., 0] + 1) * W
    py = 0.5 * (gy + flow[..., 1] + 1) * H
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1); y0 = np.clip(y0, 0, H - 1)
    wa = ((x1 - px) * (y1 - py))[..., None]; wb = ((x1 - px) * (py - y0))[..., None]
    wc = ((px - x0) * (y1 - py))[..., None]; wd = ((px - x0) * (py - y0))[..., None]
    out = wa * img[y0, x0] + wb * img[y1, x0] + wc * img[y0, x1] + wd * img[y1, x1]
    return (out * lcm[..., :1] + lcm[..., 1:2])[None]


for side in ("L", "R"):
    g = tf.Graph()
    with g.as_default():
        img = tf.compat.v1.placeholder(tf.float32, [1, 48, 64, 3], name="img")
        fp = tf.compat.v1.placeholder(tf.float32, [1, 48, 64, 12], name="fp")
        ang = tf.compat.v1.placeholder(tf.float32, [1, 2], name="ang")
        pred, raw_flow, lcm = gaze_warp_model.build_inference_graph(img, fp, ang, False, cfg)
        pred = tf.reshape(pred, [1, 48, 64, 3])
        tf.identity(tf.nn.tanh(raw_flow), name="flow")
        tf.identity(lcm, name="lcm")
        sess = tf.compat.v1.Session(graph=g)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        ckpt = tf.compat.v1.train.get_checkpoint_state(f"{weights}/{side}/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Reference output for parity check.
        rng = np.random.default_rng(0)
        x_img = rng.random((1, 48, 64, 3), dtype=np.float32)
        x_fp = rng.integers(-30, 30, (1, 48, 64, 12)).astype(np.float32)
        x_ang = np.array([[8, -3]], np.float32)
        ref = sess.run(pred, {img: x_img, fp: x_fp, ang: x_ang})

        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ["flow", "lcm"])

    mlmodel = ct.convert(
        frozen,
        source="tensorflow",
        inputs=[ct.TensorType(name="img", shape=(1, 48, 64, 3)),
                ct.TensorType(name="fp", shape=(1, 48, 64, 12)),
                ct.TensorType(name="ang", shape=(1, 2))],
        outputs=[ct.TensorType(name="flow"), ct.TensorType(name="lcm")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )
    path = f"{out}/GazeWarp{side}.mlpackage"
    mlmodel.save(path)
    r = mlmodel.predict({"img": x_img, "fp": x_fp, "ang": x_ang})
    got = warp(x_img[0], r["flow"][0], r["lcm"][0])
    print(side, "saved", path, "max |coreml - tf| =", float(np.abs(got - ref).max()))
