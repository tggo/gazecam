"""Export the flow+lcm subgraph of a GazeWarpNet to Core ML (same I/O contract as GazeCam/Models/GazeWarp{L,R}.mlpackage).

    python export_coreml.py --side L                          # TF-initialised weights (sanity: must equal shipped model)
    python export_coreml.py --side L --ckpt runs/L/best.pt    # fine-tuned weights
Writes <out>/GazeWarp{side}.mlpackage and prints max|diff| vs torch and vs the shipped model.
"""
import argparse, os
import numpy as np
import torch
import coremltools as ct
from model import GazeWarpNet
import load_tf

APP_MODELS = os.path.expanduser("~/code/gazecam/GazeCam/Models")


def load_net(side, ckpt=None):
    if ckpt:
        net = GazeWarpNet()
        sd = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        return net.eval()
    return load_tf.load_side(side)


def export(net, path):
    net = net.eval().cpu()
    ex = (torch.zeros(1, 48, 64, 3), torch.zeros(1, 48, 64, 12), torch.zeros(1, 2))
    traced = torch.jit.trace(net, ex, check_trace=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="img", shape=(1, 48, 64, 3), dtype=np.float32),
                ct.TensorType(name="fp", shape=(1, 48, 64, 12), dtype=np.float32),
                ct.TensorType(name="ang", shape=(1, 2), dtype=np.float32)],
        outputs=[ct.TensorType(name="flow", dtype=np.float32), ct.TensorType(name="lcm", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )
    ml.save(path)
    return ml


def inputs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append((rng.random((1, 48, 64, 3), dtype=np.float32),
                    rng.integers(-30, 30, (1, 48, 64, 12)).astype(np.float32),
                    rng.uniform(-20, 20, (1, 2)).astype(np.float32)))
    rs = load_tf.real_sample("R")
    if rs is not None:
        out.append((rs[0], rs[1], np.array([[8, -3]], np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["L", "R"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="export")
    ap.add_argument("--ref", default=None, help="reference .mlpackage (default: shipped app model)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net = load_net(a.side, a.ckpt)
    path = f"{a.out}/GazeWarp{a.side}.mlpackage"
    ml = export(net, path)
    ref = ct.models.MLModel(a.ref or f"{APP_MODELS}/GazeWarp{a.side}.mlpackage")
    worst = {}
    for k, (im, fp, an) in enumerate(inputs()):
        real = (k == len(inputs()) - 1) and load_tf.real_sample("R") is not None
        tag = "real" if real else "random"
        r = ml.predict({"img": im, "fp": fp, "ang": an})
        q = ref.predict({"img": im, "fp": fp, "ang": an})
        with torch.no_grad():
            f, l = net(*map(torch.from_numpy, (im, fp, an)))
        for name, x, y in (("torch", r, {"flow": f.numpy(), "lcm": l.numpy()}), ("ref", r, q)):
            for o in ("flow", "lcm"):
                key = (name, tag, o)
                worst[key] = max(worst.get(key, 0.0), float(np.abs(x[o] - y[o]).max()))
    for (name, tag, o), v in sorted(worst.items()):
        print(f"  max|coreml - {name:5s}| {tag:6s} inputs {o:4s} = {v:.2e}")
    print(f"{a.side}: saved {path}  (ref = {'given' if a.ref else 'shipped app model'}; ~0 expected only when weights are unchanged)")
    print("  spec outputs:", [o.name for o in ml.get_spec().description.output],
          "inputs:", [(i.name, list(i.type.multiArrayType.shape)) for i in ml.get_spec().description.input])


if __name__ == "__main__":
    main()
