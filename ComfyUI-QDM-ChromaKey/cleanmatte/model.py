"""Small semantic classifier followed by native-resolution alpha refinement.

Inspired by disentangled matting (AdaMatting; Zhong et al., BMVC 2021).
This is an independent convolutional implementation, not a reproduction of
their OCBlock/ENA model or its reported benchmark results.
"""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "qdm_cleanmatte_v1"


class Separable(nn.Sequential):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cin, 3, stride, dilation, dilation=dilation, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class CleanMatte(nn.Module):
    """RGB -> native BG/FG/mixed classification and conditional coverage.

    There is deliberately no trainable foreground RGB or despill branch.
    The local classifier can override semantic predictions anywhere in the
    image, including holes missed by the low-resolution network.
    """
    context_size = 384
    tile_halo = 32

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(), Separable(24, 24))
        self.enc2 = nn.Sequential(Separable(24, 48, 2), Separable(48, 48))
        self.enc3 = nn.Sequential(Separable(48, 96, 2), Separable(96, 96))
        self.enc4 = nn.Sequential(Separable(96, 128, 2), Separable(128, 128))
        self.dec3 = Separable(128 + 96, 64)
        self.dec2 = Separable(64 + 48, 32)
        self.dec1 = Separable(32 + 24, 16)
        self.context_head = nn.Conv2d(16, 11, 1)  # 3 classes + 8 context features
        self.local1 = nn.Sequential(nn.Conv2d(14, 12, 3, padding=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(), Separable(12, 12))
        self.local2 = nn.Sequential(Separable(12, 24, 2), Separable(24, 24), Separable(24, 24, dilation=2))
        self.local3 = nn.Sequential(Separable(36, 12), Separable(12, 12))
        self.classes = nn.Conv2d(12, 3, 1)
        self.coverage = nn.Conv2d(12, 1, 1)

    def encode(self, rgb):
        h, w = rgb.shape[-2:]
        scale = min(1.0, self.context_size / max(h, w))
        if scale < 1:
            rgb = F.interpolate(rgb, size=(max(1, round(h*scale)), max(1, round(w*scale))),
                                mode="bilinear", align_corners=False, antialias=True)
        x1 = self.enc1(rgb)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        y3 = self.dec3(torch.cat((resize(x4, x3.shape[-2:]), x3), 1))
        y2 = self.dec2(torch.cat((resize(y3, x2.shape[-2:]), x2), 1))
        y1 = self.dec1(torch.cat((resize(y2, x1.shape[-2:]), x1), 1))
        return self.context_head(y1)

    def refine(self, rgb, context):
        x1 = self.local1(torch.cat((rgb, context), 1))
        x2 = self.local2(x1)
        x = self.local3(torch.cat((x1, resize(x2, x1.shape[-2:])), 1))
        logits = self.classes(x) + context[:, :3]
        coverage_logits = self.coverage(x)
        coverage = coverage_logits.float().sigmoid()
        classes = logits.argmax(1, keepdim=True)
        # Class probabilities express uncertainty, not optical transparency.
        alpha = torch.where(classes == 0, 0.0, torch.where(classes == 1, 1.0, coverage))
        return {"alpha": alpha, "coverage": coverage, "coverage_logits": coverage_logits,
                "classes": logits}

    def forward(self, rgb):
        context = self.encode(rgb)
        output = self.refine(rgb, resize(context, rgb.shape[-2:]))
        output["coarse_classes"] = context[:, :3]
        return output

    def fuse_for_inference(self):
        """Fold fixed BatchNorm statistics into convolutions for deployment."""
        self.eval()
        for block in self.modules():
            if isinstance(block, nn.Sequential):
                for i in range(len(block)-1):
                    if isinstance(block[i], nn.Conv2d) and isinstance(block[i+1], nn.BatchNorm2d):
                        block[i] = torch.nn.utils.fuse_conv_bn_eval(block[i], block[i+1])
                        block[i+1] = nn.Identity()
        return self
