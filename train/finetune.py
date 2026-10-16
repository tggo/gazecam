"""Fine-tune one side (L or R) of the gaze-warp network on a user's GazeCam sessions, starting from the TF weights.

    python finetune.py --side R --val-session 2026-10-16_19-09-30 --out runs/R
    python finetune.py --side L --val-session 2026-10-16_19-09-30 --out runs/L

Loss = paper's compute_image_loss (L1, 3 px border cropped), but averaged per element instead of summed (Adam is scale
invariant; this just keeps the number readable).  BatchNorm is frozen (running stats AND affine): ~1000 strongly
correlated pairs give poor batch statistics and the L model has a dead BN channel (running_var ~1e-37, 316x gain) that
would explode if its input woke up.  Early stopping on the held-out SESSION.  Writes best.pt / last.pt / metrics.json.
"""
import argparse, glob, json, os, pickle, hashlib, time
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset as D
import load_tf
from model import GazeWarpNet

CROP = 3


def l1_loss(pred, tgt):
    return (pred - tgt)[:, CROP:-CROP, CROP:-CROP].abs().mean()


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    tot = n = 0
    for img, fp, ang, tgt in loader:
        img, fp, ang, tgt = (x.to(dev) for x in (img, fp, ang, tgt))
        tot += l1_loss(net.render(img, fp, ang)[0], tgt).item() * len(img)
        n += len(img)
    return tot / max(n, 1)


@torch.no_grad()
def noop_l1(pairs):
    return float(np.mean([np.abs(p["img"] - p["tgt"])[CROP:-CROP, CROP:-CROP].mean() for p in pairs]))


def cached_pairs(sessions, side, kw):
    key = hashlib.md5(json.dumps([sorted(sessions), side, kw], sort_keys=True).encode()).hexdigest()[:12]
    f = f"cache/pairs_{side}_{key}.pkl"
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    out = D.build_pairs(sessions, side, **kw)
    os.makedirs("cache", exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["L", "R"])
    ap.add_argument("--sessions", nargs="+", default=[D.DEFAULT_ROOT + "/*"])
    ap.add_argument("--val-session", default=None, help="session dir name held out for early stopping (default: last one)")
    ap.add_argument("--align", choices=["none", "sim"], default="sim")
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--sign", default="1,-1", help="sign_v,sign_h applied to (cam - P); see dataset.py --sign-check")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-mag", type=float, default=35.0, help="drop pairs whose |angle| exceeds this (deg)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or f"runs/{a.side}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    sign = tuple(float(x) for x in a.sign.split(","))

    sess = [s for s in D.resolve_sessions(a.sessions) if D.load_session(s) is not None]
    val_name = a.val_session or os.path.basename(sess[-1])
    pairs, st = cached_pairs(sess, a.side, dict(tol_px=a.tol, align=a.align))
    pairs = [p for p in pairs if np.hypot(*p["delta"]) <= a.max_mag]
    tr = [p for p in pairs if p["session"] != val_name]
    va = [p for p in pairs if p["session"] == val_name]
    print(f"device {dev}; sessions {[os.path.basename(s) for s in sess]}; val={val_name}\npairs: train {len(tr)}, val {len(va)}; stats {dict(st)}")
    assert tr and va, "need train and val pairs"
    tl = DataLoader(D.PairDataset(tr, sign, augment=True), batch_size=a.bs, shuffle=True, drop_last=True)
    vl = DataLoader(D.PairDataset(va, sign), batch_size=64)

    net = load_tf.load_side(a.side).to(dev)
    for m in net.modules():                      # freeze BN completely
        if isinstance(m, torch.nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    hist = dict(args=vars(a), n_train=len(tr), n_val=len(va), val_noop_l1=noop_l1(va), train_noop_l1=noop_l1(tr))
    hist["val_pretrained_l1"] = evaluate(net, vl, dev)
    print(f"val L1: no-op (input vs target) {hist['val_noop_l1']:.5f} | pretrained {hist['val_pretrained_l1']:.5f}")
    best, bad, hist["epochs"] = hist["val_pretrained_l1"], 0, []
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "side": a.side}, f"{out}/pretrained.pt")
    for ep in range(1, a.epochs + 1):
        net.eval()   # BN frozen -> eval mode everywhere (no dropout in this net)
        t0, tot, n = time.time(), 0.0, 0
        for img, fp, ang, tgt in tl:
            img, fp, ang, tgt = (x.to(dev) for x in (img, fp, ang, tgt))
            loss = l1_loss(net.render(img, fp, ang)[0], tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(img); n += len(img)
        v = evaluate(net, vl, dev)
        hist["epochs"].append(dict(epoch=ep, train_l1=tot / n, val_l1=v, sec=time.time() - t0))
        print(f"epoch {ep}: train L1 {tot / n:.5f}  val L1 {v:.5f}  ({time.time() - t0:.0f}s)")
        sd = {k: x.detach().cpu() for k, x in net.state_dict().items()}
        torch.save({"state_dict": sd, "side": a.side, "sign": sign, "val_l1": v, "epoch": ep}, f"{out}/last.pt")
        if v < best - 1e-5:
            best, bad = v, 0
            torch.save({"state_dict": sd, "side": a.side, "sign": sign, "val_l1": v, "epoch": ep}, f"{out}/best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break
    hist["best_val_l1"] = best
    json.dump(hist, open(f"{out}/metrics.json", "w"), indent=1)
    print(f"best val L1 {best:.5f} (pretrained {hist['val_pretrained_l1']:.5f}, no-op {hist['val_noop_l1']:.5f}) -> {out}/best.pt"
          + ("" if os.path.exists(f"{out}/best.pt") else "  [no epoch beat pretrained; best.pt not written]"))


if __name__ == "__main__":
    main()
