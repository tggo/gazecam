# GazeCam

macOS virtual camera that corrects eye contact: it warps your eyes so you appear to look into the lens while you
actually look at the screen. A menu-bar app plus a CoreMediaIO camera extension; pick "GazeCam" as the camera in any
video-call app.

- Apple Vision face/eye landmarks → per-eye crops → Core ML warp network (flow + light-correction map) → pasted back.
- Optional per-user calibration, a learned gaze estimator, and fine-tuned warp weights (see `train/`).
- Apple Silicon (arm64), macOS 14+.

## Build

Requires Xcode and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```
make build   # xcodegen + xcodebuild (Release)
make dev     # build and install to /Applications (system extensions only activate from there)
```

The Makefile signs via an App Store Connect API key; override `ASC_KEY_ID` / `ASC_ISSUER_ID` or adapt signing in
`project.yml` to your own team.

Per-user models (`GazeWarpUser{L,R}`, `GazeEstimator`) are not included — they are trained on your own captures with the
scripts in `train/` and dropped into `GazeCam/Models/`. Without them the app uses the stock warp weights and Vision pupils.

## Credits

The gaze warp network and its pretrained weights (`GazeCam/Models/GazeWarp{L,R}.mlpackage`, converted to Core ML) come
from **"Look at Me! Correcting Eye Gaze in Live Video Communication"**, Chih-Fan Hsu, Yu-Shuen Wang, Chin-Laung Lei,
Kuan-Ta Chen, ACM TOMM 2019 — <https://github.com/chihfanhsu/gaze_correction>. All rights to the original model remain
with its authors.
