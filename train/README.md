# GazeCam per-user fine-tuning

PyTorch port of the "Look at Me!" (Hsu et al. 2019) gaze warp network, loadable from the TF1 checkpoints, fine-tunable on
GazeCam capture sessions, exportable back to Core ML with the app's exact I/O (`img`, `fp`, `ang` -> `flow`, `lcm`).

Versions used: torch 2.7.0 (mps available), tensorflow 2.20.0, coremltools 9.0, numpy 2.5.4, opencv 5.0.0.93, Python 3.12.
Everything runs in the venv `/Volumes/A/code/_ext/gcc-venv` (`VIRTUAL_ENV=... uv pip install torch==2.7.0` was the only addition).

```
PY=/Volumes/A/code/_ext/gcc-venv/bin/python; cd ~/code/gazecam/train
$PY load_tf.py                                   # parity TF vs torch -> parity.json
$PY export_coreml.py --side L                    # TF weights -> export/GazeWarpL.mlpackage, checks vs torch + shipped model
$PY dataset.py --sessions ~/Movies/GazeCam/dataset/* --stats        # pair counts per filter
$PY dataset.py --sessions ~/Movies/GazeCam/dataset/* --sign-check   # empirical angle sign (pretrained weights)
$PY finetune.py --side R --val-session 2026-10-16_19-09-30 --lr 3e-5 --epochs 12 --patience 4 --out runs/R
$PY finetune.py --side L --val-session 2026-10-16_19-09-30 --lr 3e-5 --epochs 12 --patience 4 --out runs/L
$PY export_coreml.py --side R --ckpt runs/R/best.pt --out export_finetuned
$PY export_coreml.py --side L --ckpt runs/L/best.pt --out export_finetuned
```
`vision_landmarks.swift` (swiftc) dumps the app's 12 eye points for an old session that has no `lm` column. Not needed for new sessions.

## Files
model.py (net + differentiable warp/light), load_tf.py (checkpoint mapping + parity), crops.py (port of GazeEffect.correct crop/anchor
geometry), dataset.py (sessions -> pairs, sign check), finetune.py, export_coreml.py, vision_landmarks.swift.

## Parity (measured)
TF graph vs torch, 16 random inputs (noise image, random anchor offsets, random angles) + 4 angles on a real eye crop
(gaze-sample.mov frame 30, real pixels, hand-annotated landmarks):

| side | flow max abs | lcm max abs | final image max abs | final image, real crop only |
|---|---|---|---|---|
| L | 5.1e-6 | **1.1e-3** | 9.0e-4 | 3.2e-6 |
| R | 6.1e-6 | 6.0e-6 | 9.1e-5 | 3.2e-6 |

L lcm is just over the 1e-3 target on noise inputs. Cause: L `lcm_module/cnn_blk_1` has a dead BN channel (moving_variance 1e-37,
gain 316x), which amplifies float32 rounding. Torch fp32 vs torch fp64 differs by 2.5e-4 to 9.7e-4 on the same inputs, i.e. this
is the fp32 noise floor, not a porting error. On real crops everything is < 4e-6.

torch vs Core ML (export of TF weights, 6 random + 1 real input): L flow 4.8e-6/1.9e-5(real), lcm 8.5e-4/8.2e-6;
R flow 4.4e-6, lcm 2.5e-6. Core ML vs the shipped GazeWarp{L,R}.mlpackage: R flow 1.4e-5, lcm 2.2e-5; L flow 4e-5, lcm 3.8e-5 on the
real crop (5.1e-3 on noise, same dead-channel effect). Output names flow/lcm, input shapes verified in the spec.

Warp mismatch notes: the TF sampler is NOT grid_sample-compatible (scale W not W-1, clamped corner indices with unclamped weights, so
samples outside [0, W-1] come out black, identity flow is not identity, last row/col is 0). `model.warp_image` reproduces it exactly
(it is the same formula as the Swift warp in GazeEffect.swift), so there is no border mismatch to document vs the app.

## Angle sign (measured on real sessions, pretrained weights)
angle = (sign_v, sign_h) * (cam - P) with `[vertical, horizontal]`, cam/P from `cam_*_deg` / `target_*_deg`. Mean L1 to the camera
target, all 6 sessions, |angle| >= 8 deg, similarity-aligned pairs (R: 2111 pairs, L: 2064):

| sign (v,h) | R | L |
|---|---|---|
| (+,+) | 0.0305 | 0.0257 |
| **(+,-)** | **0.0243** | **0.0205** |
| (-,+) | 0.0297 | 0.0285 |
| (-,-) | 0.0267 | 0.0262 |

Same winner with strict no-alignment pairs (764/699 pairs). So: vertical = cam_v - target_v, horizontal = -(cam_h - target_h) = target_h - cam_h.
This is the default `--sign 1,-1`. Caveat: the margin is largest for the winner only; the model-output L1 is a weak metric (see below).

## Dataset
6 new-format sessions (19-06-53 ... 19-11-19), 5288 frames; 1040 byte-identical duplicate frames dropped (md5); old session
19-01-27 skipped (no `kind`). Frames in the first 6 of each block (saccade) and with openness < 0.5 x session median dropped.
Pair = screen frame + nearest-in-time camera frame. Pairs kept (R / L):
- strict, no alignment, corners within 2 px: 740 / 638 (4 px: 1294 / 1241)
- similarity-aligned (default, residual rms <= 1.5 px, scale 0.85-1.18, rot < 10 deg): 1597 / 1577 (before the later blink/settle
  tuning; the finetune run used 2317 / 2270 pairs over all six sessions).
Train = 5 sessions, val = 2026-10-16_19-09-30 (365 val pairs), `--max-mag 35`.

## Fine-tune results (real data, MPS, ~1 s/epoch, BN frozen, Adam, L1 with 3 px border crop)
Val L1 (mean per element) on the held-out session; "no-op" = input crop vs camera target:

| side | no-op | pretrained | fine-tuned lr 1e-4 | fine-tuned lr 3e-5 (best epoch) |
|---|---|---|---|---|
| R | 0.01732 | 0.01956 | 0.01874 | **0.01580** (epoch 1) |
| L | 0.01692 | 0.01722 | 0.01288 | **0.01261** (epoch 2) |

Val rises after epoch 1-2 while train keeps falling (one held-out session, different pose/distance), so use few epochs. The best
epoch is picked on the same val session, so these numbers are optimistic; no separate test session exists yet. Pixel L1 does not
measure "looks at the camera"; **check visually in the app before trusting**. With more sessions, hold out a second one.
Checkpoints: `runs/{R,L}_lr3e-5/{best,last,pretrained}.pt`, `metrics.json`. Fine-tuned Core ML models are in `export_finetuned/`.

## Install into the app
```
cp -R export_finetuned/GazeWarpR.mlpackage export_finetuned/GazeWarpL.mlpackage ~/code/gazecam/GazeCam/Models/   # replace the old ones (back them up first)
cd ~/code/gazecam && make dev
```
Names/shapes are unchanged, so no Swift change. The `maxRedirect = 20` cap in the app still applies.

---
# Gaze estimator (replaces the Vision pupil/eyelid heuristic)

Files: `estimator_data.py` (frame crops/labels), `estimator_model.py` (net), `train_estimator.py`, `export_estimator.py`.
Final run: `runs/est3_lm/` (best.pt, metrics.json with per-point medians); Core ML: `export_estimator/GazeEstimator.mlpackage`.

```
$PY train_estimator.py --epochs 60 --patience 100 --out runs/est3_lm      # add --no-lm for the crops-only variant
$PY export_estimator.py --ckpt runs/est3_lm/best.pt --out export_estimator
```

## Contract (sign convention)
Model output `gaze = [gaze_v, gaze_h]` in degrees, identical to the labels `[target_v_deg, target_h_deg]` (camera frames: target = cam).
The warp angle fed to GazeWarp is therefore
```
ang_v = cam_v - gaze_v        ang_h = gaze_h - cam_h        (measured best sign (+,-), see "Angle sign" above)
```
Inputs (all float32): `eyeR` (1,48,64,3) = crop of the image-left eye, `eyeL` (1,48,64,3) = image-right eye, built exactly like the
warp models' `img` (crops.py = GazeEffect.correct, BGR [0,1], no flipping); `lm` (1,27): from the 12 smoothed-or-raw eye points (image px):
cR = mean(pt0, pt3), cL = mean(pt6, pt9), mid = (cR+cL)/2, iod = |cL-cR|; first 24 numbers = (pt_i - mid)/iod in order x0 y0 x1 y1 ...;
then mid_x/frame_w - 0.5, mid_y/frame_h - 0.5, 10*iod/frame_w. (27, not 24: the last 3 carry head position/distance, which the angle needs.)
Note: trained on RAW per-frame landmarks; the app's EMA-smoothed points will be slightly steadier, not systematically different.

## Data
Split by session. train: 19-06-53, 19-07-46, 19-08-34, 19-10-17 (2415 frames); val: 19-09-30 (572); test: 19-11-19, 52 cm, never used for
any choice (560). Dropped: md5 duplicates, first 6 frames of each point block, blinks (openness < 0.5 x median), frames whose crop leaves the
frame, near-identical consecutive crops. Augmentation (train only): crop shift +-2 px, scale 0.95-1.05, brightness/contrast/gamma, lm noise.
Camera-look frames are ~31% of val/test frames (one long block per cycle), so overall MAE is weighted toward them.

## Results (MAE in degrees, all kept frames of the session; mean over frames)
| | val v | val h | test v | test h |
|---|---|---|---|---|
| **estimator (est3_lm, EMA weights, best epoch 30 chosen on val)** | 2.46 | 2.91 | 4.55 | 2.85 |
| app baseline (h = -(pupil_x - median camera pupil_x)/0.0028, v = 0) | 9.54 | 5.36 | 12.29 | 6.37 |
| constant = train median | 8.25 | 12.16 | 10.77 | 16.00 |

Per-point median |error| is in `runs/est3_lm/metrics.json` (also the baseline's). Test, model vs baseline, selected points (v/h deg):
camera 5.0/0.1 vs 19.4/1.2; right 2.5/4.3 vs 0.0/14.2; top-left 4.4/8.4 vs 16.2/10.5; top-centre 4.3/0.1 vs 16.5/6.9; centre 5.4/1.8 vs 0.0/1.9.
The baseline's v = 0 is exactly right on the three v=0 points (centre/left/right), so its vertical error there is zero by construction.

Caveats, measured: results swing a lot between runs (5 configs/seeds tried; val h 2.2-5.1, test v 3.4-4.8, test h 1.6-2.9), because training has only 4 sessions of
one person. The reported row is the run with the best val among those with the final recipe (EMA); an earlier non-EMA run happened to give a better test
(v 3.42 / h 1.64) but a worse val, so it was not picked. Vertical is the weak axis on test (4.5 deg; the camera-look frames sit ~5 deg off, i.e. the 15 deg
camera-vs-centre difference is only partly resolved at 52 cm). The crops-only variant (`--no-lm`) was worse on val in the same recipe (3.08/3.61).
Expect to retrain per user/setup; there is no temporal smoothing here (the app should low-pass the output).

## Core ML
`GazeEstimator.mlpackage` (291k params, macOS14, fp32): inputs eyeR, eyeL, lm; output `gaze` (1,2) [v,h]. torch vs Core ML max diff 1.3e-5 deg
(8 random inputs). Timing, `predict` median over 300 calls from Python (includes call overhead): CPU+ANE 0.12 ms (p95 0.15), CPU-only 0.12 ms,
ALL 0.47 ms. Well under the 3 ms budget. Not installed in the app.
