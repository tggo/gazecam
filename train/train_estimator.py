"""Train / evaluate the gaze estimator.   python train_estimator.py [--no-lm] --out runs/est"""
import argparse, json, os, pickle, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import estimator_data as ED
from estimator_model import GazeEstimator


class FrameDS(Dataset):
    def __init__(self, fr, aug):
        self.f, self.aug = fr, aug

    def __len__(self): return len(self.f)

    def __getitem__(self, i):
        f, rng = self.f[i], np.random
        if self.aug:
            def j(roi, box):
                return ED.crop_from_roi(roi, box, rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.95, 1.05))
            R, L = j(f["roiR"], f["boxR"]), j(f["roiL"], f["boxL"])
            g, c, gam = rng.uniform(-0.08, 0.08), rng.uniform(0.8, 1.2), rng.uniform(0.75, 1.3)
            ph = lambda x: np.clip(((np.clip(x, 0, 1) ** gam) - 0.5) * c + 0.5 + g, 0, 1)
            R, L = ph(R), ph(L)
            lm = f["lm"] + rng.normal(0, 0.01, 27).astype(np.float32)
        else:
            R, L, lm = ED.crop_from_roi(f["roiR"], f["boxR"]), ED.crop_from_roi(f["roiL"], f["boxL"]), f["lm"]
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32))
        return t(R), t(L), t(lm), t(f["y"])


@torch.no_grad()
def predict(net, fr, dev):
    net.eval(); out = []
    for R, L, lm, y in DataLoader(FrameDS(fr, False), batch_size=128):
        out.append(net(R.to(dev), L.to(dev), lm.to(dev)).cpu().numpy())
    return np.concatenate(out)


def report(fr, pred, name):
    y = np.stack([f["y"] for f in fr]); e = np.abs(pred - y)
    r = dict(n=len(fr), mae_v=float(e[:, 0].mean()), mae_h=float(e[:, 1].mean()))
    pts = {}
    for p in sorted(set(f["point"] for f in fr)):
        m = np.array([f["point"] == p for f in fr])
        pts[p] = dict(n=int(m.sum()), med_v=float(np.median(e[m, 0])), med_h=float(np.median(e[m, 1])))
    r["per_point_median_abs_err"] = pts
    return r


def baseline(fr_all_sessions_rows, fr):
    """App's current estimator: h = -(pupil_x - pupil_at_camera)/0.0028 (median pupil_x of that session's camera frames), v = 0."""
    ref = {}
    for s in set(f["session"] for f in fr):
        ref[s] = np.median([f["pupil_x"] for f in fr_all_sessions_rows if f["session"] == s and f["kind"] == "camera"])
    return np.array([[0.0, -(f["pupil_x"] - ref[f["session"]]) / 0.0028] for f in fr], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/est")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("cache", exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cf = "cache/est_frames.pkl"
    if os.path.exists(cf):
        data, cnt = pickle.load(open(cf, "rb"))
    else:
        data, cnt = {}, {}
        for k, v in ED.SESSIONS.items():
            data[k], cnt[k] = ED.load_frames(v)
        pickle.dump((data, cnt), open(cf, "wb"))
    tr, va, te = data["train"], data["val"], data["test"]
    print("frame counts:", cnt)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = GazeEstimator(use_lm=not a.no_lm).to(dev)
    print("params", sum(p.numel() for p in net.parameters()))
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.98), use_buffers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (len(tr) // 64))
    dl = DataLoader(FrameDS(tr, True), batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    yv = np.stack([f["y"] for f in va])
    best, bad, hist = 1e9, 0, []
    for ep in range(1, a.epochs + 1):
        net.train(); tot = n = 0
        for R, L, lm, y in dl:
            R, L, lm, y = R.to(dev), L.to(dev), lm.to(dev), y.to(dev)
            loss = torch.nn.functional.smooth_l1_loss(net(R, L, lm) / net.scale, y / net.scale, beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); ema.update_parameters(net)
            tot += loss.item() * len(y); n += len(y)
        e = np.abs(predict(ema.module, va, dev) - yv).mean(0)
        sc = e.sum()
        hist.append(dict(epoch=ep, train=tot / n, val_mae_v=float(e[0]), val_mae_h=float(e[1])))
        print(f"ep {ep}: train {tot / n:.4f} val MAE v {e[0]:.2f} h {e[1]:.2f}")
        if sc < best:
            best, bad = sc, 0
            torch.save({"state_dict": {k: v.cpu() for k, v in ema.module.state_dict().items()}, "use_lm": not a.no_lm, "epoch": ep}, f"{a.out}/best.pt")
        else:
            bad += 1
            if bad >= a.patience: print("early stop"); break
    ck = torch.load(f"{a.out}/best.pt"); net.load_state_dict(ck["state_dict"])
    allfr = tr + va + te
    res = dict(args=vars(a), best_epoch=ck["epoch"], hist=hist, counts=cnt)
    for name, fr in (("val", va), ("test", te)):
        p = predict(net, fr, dev); b = baseline(allfr, fr)
        res[name] = report(fr, p, name); res[name + "_baseline"] = report(fr, b, name)
        const = np.tile(np.median([f["y"] for f in tr], 0), (len(fr), 1)); res[name + "_train_median_const"] = report(fr, const, name)
        print(f"{name}: n={len(fr)}  model MAE v {res[name]['mae_v']:.2f} h {res[name]['mae_h']:.2f} | app baseline v {res[name+'_baseline']['mae_v']:.2f} "
              f"h {res[name+'_baseline']['mae_h']:.2f} | constant(train median) v {res[name+'_train_median_const']['mae_v']:.2f} h {res[name+'_train_median_const']['mae_h']:.2f}")
    json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
