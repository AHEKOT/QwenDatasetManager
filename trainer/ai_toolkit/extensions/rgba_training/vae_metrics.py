from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


DEFAULT_READINESS_THRESHOLDS = OrderedDict(
    finite_fraction=(">=", 1.0),
    visible_rgb_mae=("<=", 0.060),
    alpha_mae=("<=", 0.080),
    alpha_edge_mae=("<=", 0.120),
    composite_mae=("<=", 0.050),
    alpha_iou=(">=", 0.900),
    opaque_latent_rmse=("<=", 0.030),
)


def _to_zero_one(image: torch.Tensor) -> torch.Tensor:
    return ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _alpha_edges(alpha: torch.Tensor) -> torch.Tensor:
    """Return a soft Sobel edge magnitude for BCHW alpha tensors."""
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=alpha.device,
        dtype=alpha.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = F.conv2d(alpha, kernel_x, padding=1)
    grad_y = F.conv2d(alpha, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12).clamp(0.0, 1.0)


def rgba_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    opaque_latent_rmse: float | torch.Tensor,
    backgrounds: Sequence[Sequence[float]] = ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)),
) -> dict[str, float]:
    """Compute objective RGBA VAE validation metrics.

    Inputs are normalized to [-1, 1]. RGB error is alpha-weighted so arbitrary
    hidden RGB cannot dominate the score. Composite error checks the rendered
    result over multiple backgrounds, which catches both RGB and alpha defects.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError(
            "prediction and target must be matching BCHW RGBA tensors; "
            f"received {tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    pred = _to_zero_one(prediction)
    tgt = _to_zero_one(target)
    finite = torch.isfinite(prediction).flatten(1).all(dim=1)

    target_alpha = tgt[:, 3:4]
    pred_alpha = pred[:, 3:4]
    visible_denominator = target_alpha.sum() * 3.0 + 1e-8
    visible_rgb_mae = ((pred[:, :3] - tgt[:, :3]).abs() * target_alpha).sum() / visible_denominator
    alpha_mae = (pred_alpha - target_alpha).abs().mean()
    alpha_edge_mae = (_alpha_edges(pred_alpha) - _alpha_edges(target_alpha)).abs().mean()

    pred_mask = pred_alpha >= 0.5
    target_mask = target_alpha >= 0.5
    intersection = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    alpha_iou = torch.where(union > 0, intersection / union, torch.ones_like(union))

    composite_errors = []
    for background in backgrounds:
        if len(background) != 3:
            raise ValueError("each validation background must have exactly three values")
        bg = torch.tensor(background, device=pred.device, dtype=pred.dtype).view(1, 3, 1, 1)
        pred_composite = pred[:, :3] * pred_alpha + bg * (1.0 - pred_alpha)
        target_composite = tgt[:, :3] * target_alpha + bg * (1.0 - target_alpha)
        composite_errors.append((pred_composite - target_composite).abs().mean())
    composite_mae = torch.stack(composite_errors).mean()

    background_mask = target_alpha < 0.5
    foreground_mask = ~background_mask

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1)

    background_alpha_mean = masked_mean(pred_alpha, background_mask)
    foreground_alpha_mean = masked_mean(pred_alpha, foreground_mask)
    predicted_transparent_fraction = (pred_alpha < 0.5).float().mean()

    opaque_latent_rmse_value = float(
        opaque_latent_rmse.detach().float().item()
        if isinstance(opaque_latent_rmse, torch.Tensor)
        else opaque_latent_rmse
    )
    return {
        "finite_fraction": float(finite.float().mean().item()),
        "visible_rgb_mae": float(visible_rgb_mae.item()),
        "alpha_mae": float(alpha_mae.item()),
        "alpha_edge_mae": float(alpha_edge_mae.item()),
        "composite_mae": float(composite_mae.item()),
        "alpha_iou": float(alpha_iou.item()),
        "opaque_latent_rmse": opaque_latent_rmse_value,
        "background_alpha_mean": float(background_alpha_mean.item()),
        "foreground_alpha_mean": float(foreground_alpha_mean.item()),
        "predicted_transparent_fraction": float(predicted_transparent_fraction.item()),
    }


def normalize_readiness_thresholds(
    overrides: Mapping[str, float] | None = None,
) -> OrderedDict[str, tuple[str, float]]:
    thresholds = OrderedDict(DEFAULT_READINESS_THRESHOLDS)
    if overrides:
        unknown = set(overrides).difference(thresholds)
        if unknown:
            raise ValueError(f"unknown readiness thresholds: {sorted(unknown)}")
        for key, value in overrides.items():
            direction = thresholds[key][0]
            thresholds[key] = (direction, float(value))
    return thresholds


def evaluate_readiness(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, tuple[str, float]] | None = None,
) -> tuple[bool, OrderedDict[str, dict[str, float | str | bool]]]:
    thresholds = OrderedDict(thresholds or DEFAULT_READINESS_THRESHOLDS)
    checks: OrderedDict[str, dict[str, float | str | bool]] = OrderedDict()
    passed = True
    for name, (direction, threshold) in thresholds.items():
        value = float(metrics.get(name, math.nan))
        if direction == "<=":
            check_passed = math.isfinite(value) and value <= threshold
        elif direction == ">=":
            check_passed = math.isfinite(value) and value >= threshold
        else:
            raise ValueError(f"unsupported readiness comparison: {direction}")
        checks[name] = {
            "value": value,
            "comparison": direction,
            "threshold": float(threshold),
            "passed": check_passed,
        }
        passed = passed and check_passed
    return passed, checks


@dataclass
class MetricAccumulator:
    totals: dict[str, float]
    count: int = 0

    def __init__(self) -> None:
        self.totals = {}
        self.count = 0

    def update(self, metrics: Mapping[str, float], batch_size: int) -> None:
        self.count += int(batch_size)
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("cannot compute validation metrics for an empty dataset")
        return {key: value / self.count for key, value in self.totals.items()}
