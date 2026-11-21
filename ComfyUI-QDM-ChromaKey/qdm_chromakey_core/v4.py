"""Global-context / native-detail alpha and foreground matting.

Original implementation inspired by objective decomposition (MODNet), coarse
context plus refinement (BGMv2), and joint alpha/colour supervision (FBA).
No trimap, background plate, pretrained weights or colour threshold is required.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvGNAct, GNResidual, V2DecoderBlock

ARCHITECTURE_ID_V4 = "qdm_keymatte_v4"


def sample_context(value, size, boxes=None):
    """Sample normalized XYXY boxes with pixel-centre alignment, in float32.

    The same transform is used for training crops and inference tiles. Float32
    coordinates are essential: half precision loses individual pixels at 4K.
    """
    if boxes is None:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    height, width = size
    boxes = boxes.to(device=value.device, dtype=torch.float32)
    x = (torch.arange(width, device=value.device).float() + 0.5) / width
    y = (torch.arange(height, device=value.device).float() + 0.5) / height
    xx = boxes[:, 0, None] + x * (boxes[:, 2] - boxes[:, 0])[:, None]
    yy = boxes[:, 1, None] + y * (boxes[:, 3] - boxes[:, 1])[:, None]
    grid = torch.stack((xx[:, None, :].expand(-1, height, -1),
                        yy[:, :, None].expand(-1, -1, width)), dim=-1) * 2 - 1
    with torch.autocast(value.device.type, enabled=False):
        output = F.grid_sample(value.float(), grid, mode="bilinear",
                               padding_mode="border", align_corners=False)
    return output.to(value.dtype)


class LocalBlock(nn.Module):
    """No spatial normalization: local predictions are independent of tile size."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, groups=channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 1), nn.SiLU(),
        )

    def forward(self, value):
        return value + self.body(value)


class KeyMatteV4(nn.Module):
    architecture_id = ARCHITECTURE_ID_V4
    stride = 1
    context_size = 256
    # Local receptive radius is 1 + 1 + 2 + 4 + 1 = 9 pixels.
    tile_halo = 12

    def __init__(self):
        super().__init__()
        self.enc2 = nn.Sequential(ConvGNAct(3, 24, stride=2), GNResidual(24, 24))
        self.enc4 = nn.Sequential(GNResidual(24, 40, stride=2), GNResidual(40, 40))
        self.enc8 = nn.Sequential(GNResidual(40, 64, stride=2), GNResidual(64, 64))
        self.enc16 = nn.Sequential(GNResidual(64, 96, stride=2), GNResidual(96, 96))
        self.enc32 = nn.Sequential(GNResidual(96, 128, stride=2), GNResidual(128, 128))
        self.dec16 = V2DecoderBlock(128, 96, 80)
        self.dec8 = V2DecoderBlock(80, 64, 48)
        self.dec4 = V2DecoderBlock(48, 40, 32)
        self.guide_head = nn.Conv2d(32, 16, 1)
        self.coarse_head = nn.Conv2d(32, 1, 1)
        self.screen_attention = nn.Conv2d(32, 1, 1)
        self.local = nn.Sequential(
            nn.Conv2d(16 + 1 + 3 + 3 + 3, 24, 1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1, groups=24), nn.SiLU(),
            *(LocalBlock(24, dilation) for dilation in (1, 2, 4, 1)),
        )
        self.alpha_head = nn.Conv2d(24, 1, 1)
        self.foreground_head = nn.Conv2d(24, 3, 1)
        self.spill_head = nn.Conv2d(24, 1, 1)
        # Small nonzero heads allow gradients into both branches on step one.
        for head in (self.alpha_head, self.foreground_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def encode_context(self, rgb):
        rgb = F.interpolate(rgb, (self.context_size, self.context_size), mode="area")
        e2 = self.enc2(rgb)
        e4 = self.enc4(e2)
        e8 = self.enc8(e4)
        e16 = self.enc16(e8)
        value = self.dec4(self.dec8(self.dec16(self.enc32(e16), e16), e8), e4)
        coarse = self.coarse_head(value)
        # Attention sees the entire image, including backgrounds disconnected
        # from the border. A character touching a border is not a key sample.
        weights = self.screen_attention(value).float().flatten(2).softmax(-1)
        pixels = F.interpolate(rgb, value.shape[-2:], mode="area").float().flatten(2)
        key = (pixels * weights).sum(-1).to(value.dtype)
        return {"guide": torch.cat((self.guide_head(value), coarse), 1),
                "coarse_logits": coarse, "key": key}

    def refine(self, rgb, context, boxes=None):
        guide = sample_context(context["guide"], rgb.shape[-2:], boxes)
        key = context["key"][:, :, None, None].expand_as(rgb)
        value = self.local(torch.cat((guide, rgb, key, rgb - key), 1))
        logits = guide[:, 16:17] + self.alpha_head(value)
        # Full +/-1 RGB range. There is no division by tiny alpha and no gate
        # tying correction strength to alpha or a synthetic spill coefficient.
        foreground_raw = rgb + torch.tanh(self.foreground_head(value))
        return {"alpha_logits": logits, "alpha": logits.sigmoid(),
                "foreground": foreground_raw.clamp(0, 1),
                "foreground_raw": foreground_raw,
                "rgb_confidence": self.spill_head(value).sigmoid(),
                "coarse_logits": context["coarse_logits"],
                "key_colors": context["key"][:, None]}

    def forward(self, rgb, context_rgb=None, boxes=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected BCHW RGB input")
        return self.refine(rgb, self.encode_context(rgb if context_rgb is None else context_rgb), boxes)
