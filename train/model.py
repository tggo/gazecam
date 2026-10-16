"""PyTorch port of the "Look at Me!" (Hsu et al. 2019) gaze warping network.

Mirrors tf_models/gaze_corrector_v1/gaze_warp_model.py::build_inference_graph.

Tensor layout at the module boundary is NHWC, exactly like the Core ML models in the app:
    img (B,48,64,3) BGR in [0,1]; fp (B,48,64,12) anchor map; ang (B,2) [vertical, horizontal] degrees.
forward() returns (flow, lcm) = (tanh(raw_flow) (B,48,64,2), softmax LCM weights (B,48,64,2)),
which is the subgraph the app's .mlpackage files contain. The warp and light blend are separate,
differentiable functions (warp_image / light_correct / GazeWarpNet.render) used for training.

Notes on exactness vs. TF:
  * conv 'same' padding: all kernels are odd (5,3,1) so symmetric zero padding == TF 'same'.
  * AveragePooling2D 2x2/stride 2 on 48x64 has no padding. The 2x2/stride 1 'same' smoothing pool pads
    one row/col at the bottom/right and Keras averages over VALID cells only (border divisors 2 and 1);
    reproduced with a zero pad + fixed divisor map.
  * nearest upsample 24x32 -> 48x64 is an exact 2x repeat in both TF and torch.
  * BatchNorm: eps 1e-5, TF momentum 0.9 == torch momentum 0.1. Conv -> ReLU -> BN order (BN AFTER relu).
  * warp: NOT F.grid_sample. The TF sampler maps normalised coords u in [-1,1] to pixel 0.5*(u+1)*W
    (scale W, not W-1, no half-pixel shift), clamps the four corner INDICES but not the weights.
    Consequences: an identity flow is not an identity (pixel i samples i*W/(W-1)), and wherever the sample
    point leaves [0, W-1] the two clamped corners coincide while their weights sum to 0, so the output is 0
    (black) there -- e.g. the last column / last row. grid_sample cannot express that, so warp_image() is a
    hand-written gather bilinear that reproduces it bit-for-bit (differentiable in both image and flow).
    The app's Swift warp (GazeEffect.swift) mirrors the same formula (see tools/convert_gaze.py::warp).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 48, 64


class ConvBlock(nn.Module):
    """Conv2D(no bias) -> ReLU -> BatchNorm (TF: cnn / act / bn_layer)."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.cnn = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.bn_layer = nn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return self.bn_layer(F.relu(self.cnn(x)))


class DenseBlock(nn.Module):
    """Dense -> ReLU (TF: dnn / act)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.dnn = nn.Linear(cin, cout)

    def forward(self, x):
        return F.relu(self.dnn(x))


class TransformModule(nn.Module):
    """Dense-connected conv stack: blk0 -> blk1 -> blk2([0,1]) -> blk3([0,1,2]) -> 1x1 conv."""

    def __init__(self, cin, depths, ks):
        super().__init__()
        d = depths
        self.cnn_blk_0 = ConvBlock(cin, d[0], ks[0])
        self.cnn_blk_1 = ConvBlock(d[0], d[1], ks[1])
        self.cnn_blk_2 = ConvBlock(d[0] + d[1], d[2], ks[2])
        self.cnn_blk_3 = ConvBlock(d[0] + d[1] + d[2], d[3], ks[3])
        self.cnn_4 = nn.Conv2d(d[3], d[4], ks[4], padding=ks[4] // 2, bias=False)

    def forward(self, x):
        b0 = self.cnn_blk_0(x)
        b1 = self.cnn_blk_1(b0)
        b2 = self.cnn_blk_2(torch.cat([b0, b1], 1))
        b3 = self.cnn_blk_3(torch.cat([b0, b1, b2], 1))
        return self.cnn_4(b3)


class LCMModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_blk_0 = ConvBlock(2, 8, 3)
        self.cnn_blk_1 = ConvBlock(8, 8, 3)
        self.cnn_2 = nn.Conv2d(8, 2, 1, bias=False)

    def forward(self, x):
        return torch.softmax(self.cnn_2(self.cnn_blk_1(self.cnn_blk_0(x))), dim=1)


class AngleEncoder(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.dnn_blk_0 = DenseBlock(2, 16)
        self.dnn_blk_1 = DenseBlock(16, 16)
        self.dnn_blk_2 = DenseBlock(16, out_dim)

    def forward(self, ang):
        return self.dnn_blk_2(self.dnn_blk_1(self.dnn_blk_0(ang)))


class WarpingModule(nn.Module):
    def __init__(self):
        super().__init__()
        cin = 3 + 12 + 16
        self.coarse_level = TransformModule(cin, (32, 64, 64, 32, 16), (5, 3, 3, 3, 1))
        self.fine_level = TransformModule(cin + 16, (32, 64, 32, 16, 4), (5, 3, 3, 3, 1))
        # divisor map for the 2x2 stride-1 'same' average pool (valid cells only)
        cnt = torch.full((1, 1, H, W), 4.0)
        cnt[..., -1, :] = 2.0
        cnt[..., :, -1] = 2.0
        cnt[..., -1, -1] = 1.0
        self.register_buffer("pool_div", 4.0 / cnt, persistent=False)

    def forward(self, x):
        coarse = self.coarse_level(F.avg_pool2d(x, 2, 2))
        up = F.interpolate(torch.tanh(coarse), size=(H, W), mode="nearest")
        # TF AveragePooling2D(2, stride 1, 'same'): pad bottom/right by 1, average over valid cells.
        sm = F.avg_pool2d(F.pad(up, (0, 1, 0, 1)), 2, 1) * self.pool_div
        out = self.fine_level(torch.cat([x, sm], 1))
        return out[:, :2], out[:, 2:]  # raw_flow, lcm_input


class GazeWarpNet(nn.Module):
    """forward(img, fp, ang) -> (flow, lcm); all NHWC float32, see module docstring."""

    def __init__(self):
        super().__init__()
        self.encoder = AngleEncoder(16)
        self.warping_module = WarpingModule()
        self.lcm_module = LCMModule()

    def forward(self, img, fp, ang):
        a = self.encoder(ang)[:, :, None, None].expand(-1, -1, H, W)  # tile to spatial map
        x = torch.cat([img.permute(0, 3, 1, 2), fp.permute(0, 3, 1, 2), a], 1)
        raw_flow, lcm_in = self.warping_module(x)
        flow = torch.tanh(raw_flow)
        lcm = self.lcm_module(lcm_in)
        return flow.permute(0, 2, 3, 1), lcm.permute(0, 2, 3, 1)

    def render(self, img, fp, ang):
        """Full differentiable pipeline -> (final image (B,48,64,3), flow, lcm)."""
        flow, lcm = self(img, fp, ang)
        return light_correct(warp_image(img, flow), lcm), flow, lcm


def warp_image(img, flow):
    """Backward bilinear warp identical to tf spatial_transform.apply_optical_flow. NHWC in/out."""
    B, h, w, _ = img.shape
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, h, device=img.device),
                            torch.linspace(-1, 1, w, device=img.device), indexing="ij")
    px = 0.5 * (gx[None] + flow[..., 0] + 1) * w
    py = 0.5 * (gy[None] + flow[..., 1] + 1) * h
    x0f, y0f = torch.floor(px), torch.floor(py)
    x0 = x0f.long().clamp(0, w - 1)
    y0 = y0f.long().clamp(0, h - 1)
    x1 = (x0f.long() + 1).clamp(0, w - 1)
    y1 = (y0f.long() + 1).clamp(0, h - 1)
    bi = torch.arange(B, device=img.device)[:, None, None]
    wa = ((x1 - px) * (y1 - py)).unsqueeze(-1)  # weights use CLAMPED corner coords, like TF
    wb = ((x1 - px) * (py - y0)).unsqueeze(-1)
    wc = ((px - x0) * (y1 - py)).unsqueeze(-1)
    wd = ((px - x0) * (py - y0)).unsqueeze(-1)
    return wa * img[bi, y0, x0] + wb * img[bi, y1, x0] + wc * img[bi, y0, x1] + wd * img[bi, y1, x1]


def light_correct(warped, lcm):
    """out = warped * lcm[...,0] + 1 * lcm[...,1] (white palette)."""
    return warped * lcm[..., :1] + lcm[..., 1:2]
