"""Export the gaze estimator to Core ML: inputs eyeR, eyeL (1,48,64,3), lm (1,27); output gaze (1,2) = [v,h] degrees.
    python export_estimator.py --ckpt runs/est3_lm/best.pt --out export_estimator"""
import argparse, os, time
import numpy as np, torch, coremltools as ct
from estimator_model import GazeEstimator

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="export_estimator")
a = ap.parse_args()
ck = torch.load(a.ckpt); net = GazeEstimator(use_lm=ck["use_lm"]); net.load_state_dict(ck["state_dict"]); net.eval()
ex = (torch.zeros(1, 48, 64, 3), torch.zeros(1, 48, 64, 3), torch.zeros(1, 27))
inputs = [ct.TensorType(name="eyeR", shape=(1, 48, 64, 3), dtype=np.float32), ct.TensorType(name="eyeL", shape=(1, 48, 64, 3), dtype=np.float32)]
if ck["use_lm"]:
    inputs.append(ct.TensorType(name="lm", shape=(1, 27), dtype=np.float32))
    wrap = net
else:
    wrap = lambda r, l, lm: net(r, l, lm)
ml = ct.convert(torch.jit.trace(net, ex, check_trace=False), inputs=inputs if ck["use_lm"] else inputs + [ct.TensorType(name="lm", shape=(1, 27), dtype=np.float32)],
                outputs=[ct.TensorType(name="gaze", dtype=np.float32)], minimum_deployment_target=ct.target.macOS14,
                compute_precision=ct.precision.FLOAT32, convert_to="mlprogram")
os.makedirs(a.out, exist_ok=True); path = f"{a.out}/GazeEstimator.mlpackage"; ml.save(path)
rng = np.random.default_rng(0); d = 0
feeds = []
for _ in range(8):
    f = {"eyeR": rng.random((1, 48, 64, 3), dtype=np.float32), "eyeL": rng.random((1, 48, 64, 3), dtype=np.float32), "lm": rng.normal(0, .5, (1, 27)).astype(np.float32)}
    feeds.append(f)
    with torch.no_grad(): t = net(*(torch.from_numpy(f[k]) for k in ("eyeR", "eyeL", "lm"))).numpy()
    d = max(d, float(np.abs(ml.predict(f)["gaze"] - t).max()))
print(f"saved {path}; max|coreml - torch| over 8 random inputs = {d:.2e} deg; params {sum(p.numel() for p in net.parameters())}")
for cu in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL):
    m = ct.models.MLModel(path, compute_units=cu)
    for _ in range(20): m.predict(feeds[0])
    ts = []
    for i in range(300):
        t0 = time.perf_counter(); m.predict(feeds[i % 8]); ts.append((time.perf_counter() - t0) * 1000)
    print(f"  {cu.name:12s} predict median {np.median(ts):.2f} ms  p95 {np.percentile(ts, 95):.2f} ms (python call overhead included)")
