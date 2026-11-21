"""Small fully-neural RGB-to-RGBA matting network.

The model deliberately accepts RGB only.  Background colour priors and
software chroma masks are not part of its input contract.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_ID = "qdm_anime_keymatte_v1"
ARCHITECTURE_ID_V2 = "qdm_anime_keymatte_v2"
ARCHITECTURE_ID_V3 = "qdm_anime_keymatte_v3"


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, groups=1, act=True):
        padding = kernel // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.Hardswish(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, value):
        return value * self.net(value)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1, se=False):
        super().__init__()
        hidden = int(expansion)
        layers = []
        if hidden != in_channels:
            layers.append(ConvBNAct(in_channels, hidden, kernel=1))
        layers.append(ConvBNAct(hidden, hidden, stride=stride, groups=hidden))
        if se:
            layers.append(SqueezeExcite(hidden))
        layers.append(ConvBNAct(hidden, out_channels, kernel=1, act=False))
        self.body = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, value):
        output = self.body(value)
        return output + value if self.use_residual else output


class LiteASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel=3, groups=in_channels),
            ConvBNAct(in_channels, out_channels, kernel=1),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, value):
        return self.local(value) * self.global_pool(value)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        merged = in_channels + skip_channels
        self.block = nn.Sequential(
            ConvBNAct(merged, merged, kernel=3, groups=merged),
            ConvBNAct(merged, out_channels, kernel=1),
            InvertedResidual(out_channels, out_channels, out_channels * 2, se=True),
        )

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((value, skip), dim=1))


class AnimeKeyMatte(nn.Module):
    """Aspect-ratio agnostic semantic/detail/fusion matting model."""

    architecture_id = ARCHITECTURE_ID
    stride = 32

    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 16, stride=2)
        self.stage4 = nn.Sequential(
            InvertedResidual(16, 24, 48, stride=2),
            InvertedResidual(24, 24, 72),
        )
        self.stage8 = nn.Sequential(
            InvertedResidual(24, 40, 96, stride=2, se=True),
            InvertedResidual(40, 40, 120, se=True),
        )
        self.stage16 = nn.Sequential(
            InvertedResidual(40, 80, 240, stride=2),
            InvertedResidual(80, 80, 240, se=True),
            InvertedResidual(80, 112, 480, se=True),
        )
        self.stage32 = nn.Sequential(
            InvertedResidual(112, 160, 672, stride=2, se=True),
            InvertedResidual(160, 160, 960, se=True),
            InvertedResidual(160, 160, 960, se=True),
        )
        self.context = LiteASPP(160, 128)
        self.decode16 = DecoderBlock(128, 112, 96)
        self.decode8 = DecoderBlock(96, 40, 64)
        self.decode4 = DecoderBlock(64, 24, 40)
        self.decode2 = DecoderBlock(40, 16, 24)
        self.coarse_head = nn.Conv2d(64, 1, 1)
        self.detail_head = nn.Conv2d(24, 1, 1)
        # Project at half resolution before upsampling.  This keeps the only
        # full-resolution learned feature map narrow, which is what allows
        # large training batches and bounded-memory 4K inference.
        self.fusion_feature = ConvBNAct(24, 8, kernel=1)
        self.fusion_rgb = ConvBNAct(3, 8, kernel=3)
        self.fusion = nn.Sequential(
            InvertedResidual(8, 8, 16, se=True),
            ConvBNAct(8, 8, kernel=3),
        )
        self.alpha_head = nn.Conv2d(8, 1, 1)
        self.rgb_delta_head = nn.Conv2d(8, 3, 1)
        self.rgb_confidence_head = nn.Conv2d(8, 1, 1)
        self._initialize_heads()

    def _initialize_heads(self):
        # Start from "mostly background" instead of an indecisive 0.5 matte.
        # Balanced foreground/boundary losses then only need to discover the
        # subject; trivial flat backgrounds disappear within the early steps.
        nn.init.constant_(self.alpha_head.bias, -2.0)
        nn.init.constant_(self.coarse_head.bias, -2.0)
        nn.init.constant_(self.detail_head.bias, -2.0)
        nn.init.zeros_(self.rgb_delta_head.weight)
        nn.init.zeros_(self.rgb_delta_head.bias)
        nn.init.zeros_(self.rgb_confidence_head.weight)
        nn.init.constant_(self.rgb_confidence_head.bias, -3.0)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, received {tuple(rgb.shape)}")
        e2 = self.stem(rgb)
        e4 = self.stage4(e2)
        e8 = self.stage8(e4)
        e16 = self.stage16(e8)
        e32 = self.stage32(e16)
        value = self.context(e32)
        value = self.decode16(value, e16)
        value = self.decode8(value, e8)
        coarse_logits = self.coarse_head(value)
        value = self.decode4(value, e4)
        value = self.decode2(value, e2)
        detail_logits = self.detail_head(value)
        full = F.interpolate(
            self.fusion_feature(value), size=rgb.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        fused = self.fusion(full + self.fusion_rgb(rgb))
        alpha_logits = self.alpha_head(fused)
        confidence = torch.sigmoid(self.rgb_confidence_head(fused))
        # Strong synthetic spill can exceed 0.5/channel.  Do not artificially
        # cap the learned correction below the corruption used for training.
        delta = torch.tanh(self.rgb_delta_head(fused))
        foreground = (rgb + confidence * delta).clamp(0.0, 1.0)
        return {
            "alpha_logits": alpha_logits,
            "alpha": torch.sigmoid(alpha_logits),
            "foreground": foreground,
            "rgb_delta": delta,
            "rgb_confidence": confidence,
            "coarse_logits": coarse_logits,
            "detail_logits": detail_logits,
        }


class ConvGNAct(nn.Sequential):
    """Batch-size independent block for high-resolution batch-1 training."""

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, groups=1, act=True):
        padding = kernel // 2
        norm_groups = 8
        while out_channels % norm_groups:
            norm_groups //= 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class GNResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=2):
        super().__init__()
        hidden = max(out_channels, in_channels * expansion)
        self.body = nn.Sequential(
            ConvGNAct(in_channels, hidden, 1),
            ConvGNAct(hidden, hidden, 3, stride=stride, groups=hidden),
            ConvGNAct(hidden, out_channels, 1, act=False),
        )
        self.skip = (
            nn.Identity() if stride == 1 and in_channels == out_channels
            else ConvGNAct(in_channels, out_channels, 1, stride=stride, act=False)
        )

    def forward(self, value):
        return F.silu(self.body(value) + self.skip(value), inplace=True)


class BorderKeyEstimator(nn.Module):
    """Learn background prototypes from the frame border using soft attention."""

    def __init__(self, prototypes=3, sample_size=64, border=4):
        super().__init__()
        self.prototypes = int(prototypes)
        self.sample_size = int(sample_size)
        self.border = int(border)
        self.attention = nn.Conv1d(3, self.prototypes, 1)
        nn.init.normal_(self.attention.weight, std=0.01)
        nn.init.zeros_(self.attention.bias)

    def forward(self, rgb):
        sample = F.adaptive_avg_pool2d(rgb, (self.sample_size, self.sample_size))
        b = self.border
        border_pixels = torch.cat((
            sample[:, :, :b, :].flatten(2),
            sample[:, :, -b:, :].flatten(2),
            sample[:, :, b:-b, :b].flatten(2),
            sample[:, :, b:-b, -b:].flatten(2),
        ), dim=2)
        weights = torch.softmax(self.attention(border_pixels), dim=-1)
        colors = torch.einsum("bkn,bcn->bkc", weights, border_pixels)
        difference = border_pixels[:, None] - colors[:, :, :, None]
        variance = torch.einsum("bkn,bkcn->bkc", weights, difference.square())
        tolerance = variance.mean(dim=-1).clamp_min(1e-6).sqrt() * 2.5 + 0.018
        return {"colors": colors, "tolerance": tolerance.clamp(0.018, 0.30)}


class V2DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_channels + skip_channels, out_channels, 3),
            GNResidual(out_channels, out_channels),
        )

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((value, skip), dim=1))


class AnimeKeyMatteV2(nn.Module):
    """Key-conditioned semantic/detail matting with learned RGB despill."""

    architecture_id = ARCHITECTURE_ID_V2
    stride = 32
    feature_channels = 9

    def __init__(self):
        super().__init__()
        # One robust dominant prototype avoids accidentally treating a
        # foreground colour touching the frame as a second background key.
        self.key_estimator = BorderKeyEstimator(prototypes=1)
        self.log_key_scale = nn.Parameter(torch.tensor(3.4))

        # Texture/refinement lives at half resolution.  The previous V2 kept
        # 24--192 channel residual activations at full resolution, so lowering
        # the crop size did not help once the sampler increased the batch to
        # the same megapixel budget.  A tiny full-resolution path below keeps
        # one-pixel hair/gap evidence without retaining a wide autograd graph.
        self.texture_half = nn.Sequential(
            ConvGNAct(self.feature_channels, 24, 3),
            GNResidual(24, 24, expansion=2),
        )
        self.stem = ConvGNAct(self.feature_channels, 32, 3, stride=2)
        self.stage4 = nn.Sequential(
            GNResidual(32, 48, stride=2, expansion=3),
            GNResidual(48, 48, expansion=3),
        )
        self.stage8 = nn.Sequential(
            GNResidual(48, 80, stride=2, expansion=3),
            GNResidual(80, 80, expansion=3),
            GNResidual(80, 80, expansion=3),
        )
        self.stage16 = nn.Sequential(
            GNResidual(80, 144, stride=2, expansion=3),
            GNResidual(144, 144, expansion=3),
            GNResidual(144, 144, expansion=3),
        )
        self.stage32 = nn.Sequential(
            GNResidual(144, 256, stride=2, expansion=3),
            GNResidual(256, 256, expansion=3),
            GNResidual(256, 256, expansion=3),
            GNResidual(256, 256, expansion=3),
        )
        self.context = nn.Sequential(
            GNResidual(256, 320, expansion=3),
            nn.Conv2d(320, 224, 1),
            nn.SiLU(inplace=True),
        )
        self.decode16 = V2DecoderBlock(224, 144, 176)
        self.decode8 = V2DecoderBlock(176, 80, 128)
        self.decode4 = V2DecoderBlock(128, 48, 96)
        self.decode2 = V2DecoderBlock(96, 32, 64)
        self.coarse_head = nn.Conv2d(128, 1, 1)
        self.detail_head = nn.Conv2d(64, 1, 1)
        self.half_fusion = nn.Sequential(
            ConvGNAct(64 + 24 + 1, 48, 3),
            GNResidual(48, 32, expansion=2),
            ConvGNAct(32, 16, 1),
        )
        self.semantic_detail_projection = nn.Conv2d(16, 8, 1)
        self.raw_detail = nn.Sequential(
            ConvGNAct(self.feature_channels, self.feature_channels, 3,
                      groups=self.feature_channels),
            nn.Conv2d(self.feature_channels, 8, 1),
        )
        self.full_detail = ConvGNAct(8, 8, 3, groups=8)
        self.alpha_residual_head = nn.Conv2d(8, 1, 1)
        self.rgb_delta_head = nn.Conv2d(8, 3, 1)
        self.rgb_confidence_head = nn.Conv2d(8, 1, 1)
        nn.init.zeros_(self.alpha_residual_head.weight)
        nn.init.zeros_(self.alpha_residual_head.bias)
        nn.init.zeros_(self.rgb_delta_head.weight)
        nn.init.zeros_(self.rgb_delta_head.bias)
        nn.init.zeros_(self.rgb_confidence_head.weight)
        nn.init.constant_(self.rgb_confidence_head.bias, -2.0)

    def estimate_key_context(self, rgb):
        return self.key_estimator(rgb)

    def _key_features(self, rgb, key_context):
        colors = key_context["colors"]
        tolerance = key_context["tolerance"]
        difference = rgb[:, None] - colors[:, :, :, None, None]
        distance = difference.square().sum(dim=2).clamp_min(1e-8).sqrt()
        normalized = distance / tolerance[:, :, None, None]
        similarity = torch.exp(-0.5 * normalized.square())
        nearest_distance, nearest = normalized.min(dim=1, keepdim=True)
        scale = self.log_key_scale.exp().clamp(8.0, 80.0)
        base_logits = (nearest_distance - 1.0) * scale
        features = torch.cat((
            rgb,
            difference.flatten(1, 2),
            normalized,
            similarity,
            torch.sigmoid(base_logits),
        ), dim=1)
        return features, base_logits, nearest

    def forward(self, rgb, key_context=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, received {tuple(rgb.shape)}")
        if key_context is None:
            key_context = self.estimate_key_context(rgb)
        features, base_logits, nearest = self._key_features(rgb, key_context)
        e2 = self.stem(features)
        e4 = self.stage4(e2)
        e8 = self.stage8(e4)
        e16 = self.stage16(e8)
        e32 = self.stage32(e16)
        value = self.context(e32)
        value = self.decode16(value, e16)
        value = self.decode8(value, e8)
        coarse_logits = self.coarse_head(value)
        value = self.decode4(value, e4)
        value = self.decode2(value, e2)
        half_detail = self.detail_head(value)
        half_features = F.interpolate(features, size=value.shape[-2:], mode="area")
        texture = self.texture_half(half_features)
        base_alpha_half = F.interpolate(
            torch.sigmoid(base_logits), size=value.shape[-2:], mode="area",
        )
        half_fused = self.half_fusion(torch.cat((value, texture, base_alpha_half), dim=1))
        semantic_detail = F.interpolate(
            self.semantic_detail_projection(half_fused),
            size=rgb.shape[-2:], mode="bilinear", align_corners=False,
        )
        fused = self.full_detail(semantic_detail + self.raw_detail(features))
        alpha_logits = base_logits + self.alpha_residual_head(fused)
        alpha = torch.sigmoid(alpha_logits)
        confidence = torch.sigmoid(self.rgb_confidence_head(fused))
        delta = torch.tanh(self.rgb_delta_head(fused))
        foreground = (rgb + confidence * delta).clamp(0.0, 1.0)
        return {
            "alpha_logits": alpha_logits,
            "alpha": alpha,
            "base_alpha": torch.sigmoid(base_logits),
            "foreground": foreground,
            "rgb_delta": delta,
            "rgb_confidence": confidence,
            "coarse_logits": coarse_logits,
            "detail_logits": half_detail,
            "key_colors": key_context["colors"],
            "key_tolerance": key_context["tolerance"],
            "nearest_key": nearest,
        }


class AnimeKeyMatteV3(AnimeKeyMatteV2):
    """Semantic-first matte with a bounded colour cue and explicit despill map.

    V2 added the analytical key logits directly to the final alpha logits.  It
    produced a useful matte immediately, but also let the final alpha remain
    almost identical to that shortcut for thousands of steps.  V3 makes the
    learned half-resolution semantic matte the primary prediction.  The colour
    key contributes at most +/-3 logits, so it accelerates easy screens but
    can always be overridden for hair, holes and foreground/key collisions.
    """

    architecture_id = ARCHITECTURE_ID_V3
    prior_weight = 0.50

    def __init__(self):
        super().__init__()
        # Softplus keeps the distance scale trainable without V2's hard lower
        # clamp, which stopped receiving gradients once it reached 8.
        self.log_key_scale = nn.Parameter(torch.tensor(2.5))
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)
        nn.init.zeros_(self.alpha_residual_head.weight)
        nn.init.zeros_(self.alpha_residual_head.bias)
        nn.init.zeros_(self.rgb_delta_head.weight)
        nn.init.zeros_(self.rgb_delta_head.bias)
        nn.init.zeros_(self.rgb_confidence_head.weight)
        nn.init.constant_(self.rgb_confidence_head.bias, -4.0)

    def _key_features(self, rgb, key_context):
        colors = key_context["colors"]
        tolerance = key_context["tolerance"]
        difference = rgb[:, None] - colors[:, :, :, None, None]
        distance = difference.square().sum(dim=2).clamp_min(1e-8).sqrt()
        normalized = distance / tolerance[:, :, None, None]
        similarity = torch.exp(-0.5 * normalized.square())
        nearest_distance, nearest = normalized.min(dim=1, keepdim=True)
        scale = F.softplus(self.log_key_scale) + 1.0
        base_logits = (nearest_distance - 1.0) * scale
        features = torch.cat((
            rgb,
            difference.flatten(1, 2),
            normalized,
            similarity,
            torch.sigmoid(base_logits),
        ), dim=1)
        return features, base_logits, nearest

    def forward(self, rgb, key_context=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, received {tuple(rgb.shape)}")
        if key_context is None:
            key_context = self.estimate_key_context(rgb)
        features, base_logits, nearest = self._key_features(rgb, key_context)
        e2 = self.stem(features)
        e4 = self.stage4(e2)
        e8 = self.stage8(e4)
        e16 = self.stage16(e8)
        e32 = self.stage32(e16)
        value = self.context(e32)
        value = self.decode16(value, e16)
        value = self.decode8(value, e8)
        coarse_logits = self.coarse_head(value)
        value = self.decode4(value, e4)
        value = self.decode2(value, e2)
        semantic_logits = self.detail_head(value)

        half_features = F.interpolate(features, size=value.shape[-2:], mode="area")
        texture = self.texture_half(half_features)
        base_alpha_half = F.interpolate(
            torch.sigmoid(base_logits), size=value.shape[-2:], mode="area",
        )
        half_fused = self.half_fusion(torch.cat((value, texture, base_alpha_half), dim=1))
        semantic_detail = F.interpolate(
            self.semantic_detail_projection(half_fused),
            size=rgb.shape[-2:], mode="bilinear", align_corners=False,
        )
        fused = self.full_detail(semantic_detail + self.raw_detail(features))

        semantic_full = F.interpolate(
            semantic_logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False,
        )
        key_guide = base_logits.clamp(-6.0, 6.0) * self.prior_weight
        alpha_logits = semantic_full + self.alpha_residual_head(fused) + key_guide
        alpha = torch.sigmoid(alpha_logits)

        # The scalar map explicitly says where screen chroma must be removed.
        # Collision examples supervise it toward zero on naturally green/blue
        # character details, unlike V2's unconstrained global RGB confidence.
        despill_amount = torch.sigmoid(self.rgb_confidence_head(fused))
        key = key_context["colors"][:, 0, :, None, None]
        key_chroma = key - key.mean(dim=1, keepdim=True)
        learned_delta = torch.tanh(self.rgb_delta_head(fused)) * 0.25
        rgb_delta = learned_delta - despill_amount * key_chroma
        foreground = (rgb + rgb_delta).clamp(0.0, 1.0)
        return {
            "alpha_logits": alpha_logits,
            "alpha": alpha,
            "base_alpha": torch.sigmoid(base_logits),
            "foreground": foreground,
            "rgb_delta": rgb_delta,
            "rgb_confidence": despill_amount,
            "coarse_logits": coarse_logits,
            "detail_logits": semantic_logits,
            "key_colors": key_context["colors"],
            "key_tolerance": key_context["tolerance"],
            "nearest_key": nearest,
        }
