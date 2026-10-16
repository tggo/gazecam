"""Small two-eye CNN gaze estimator. Inputs NHWC like the warp models: eyeR/eyeL (B,48,64,3) BGR [0,1], lm (B,27).
Output gaze (B,2) = [gaze_v, gaze_h] degrees, convention: warp angle = (cam_v - gaze_v, gaze_h - cam_h)."""
import torch, torch.nn as nn, torch.nn.functional as F


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class Trunk(nn.Module):
    def __init__(self, f=96):
        super().__init__()
        self.c = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 48), nn.Conv2d(48, 48, 3, padding=1), nn.ReLU())  # 6x8
        self.fc = nn.Linear(48 * 6 * 8, f)

    def forward(self, x):                       # x NCHW
        return F.relu(self.fc(self.c(x).reshape(x.shape[0], -1)))


class GazeEstimator(nn.Module):
    def __init__(self, use_lm=True, f=96):
        super().__init__()
        self.use_lm = use_lm
        self.trunk = Trunk(f)                    # shared by both eyes
        self.side = nn.Parameter(torch.zeros(2, f))   # per-eye learned offset (R / L)
        self.lm = nn.Sequential(nn.Linear(27, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(2 * f + (32 if use_lm else 0), 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))
        self.register_buffer("scale", torch.tensor([15.0, 30.0]))   # head output is in units of (v 15 deg, h 30 deg)

    def forward(self, eyeR, eyeL, lm):
        a = self.trunk(eyeR.permute(0, 3, 1, 2).contiguous()) + self.side[0]
        b = self.trunk(eyeL.permute(0, 3, 1, 2).contiguous()) + self.side[1]
        z = [a, b] + ([self.lm(lm)] if self.use_lm else [])
        return self.head(torch.cat(z, 1)) * self.scale
