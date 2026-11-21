"""Supervise true coverage and classes. Colour reconstruction cannot pay losses."""
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F


def target_classes(alpha):
    return torch.where(alpha[:, 0] == 0, 0, torch.where(alpha[:, 0] == 1, 1, 2)).long()


def region_mean(values, mask):
    expanded = mask.expand_as(values).to(values.dtype)
    return (values*expanded).sum()/expanded.sum().clamp_min(1)


def balanced(values, labels):
    return sum(region_mean(values, (labels == k)[:, None]) for k in range(3))/3


def class_loss(logits, labels):
    error = F.cross_entropy(logits.float(), labels, reduction="none")[:, None]
    return balanced(error, labels)


def objective(output, target, partner=None, weights=None):
    weights = {"alpha": 5.0, "classification": 1.0, "gradient": 0.5, "consistency": 0.25, **(weights or {})}
    target = target.float()
    labels = target_classes(target)
    coverage = output["coverage"]
    maximum = F.max_pool2d(target, 3, 1, 1)
    minimum = -F.max_pool2d(-target, 3, 1, 1)
    detail = maximum-minimum > 0
    # Coverage is supervised even where the current classifier is wrong.
    # This avoids a hard classification gate starving the alpha head.
    alpha = balanced((coverage-target).abs(), labels) + 0.5*region_mean((coverage-target).abs(), detail)
    classification = class_loss(output["classes"], labels)
    # Opaque one-pixel strands otherwise disappear inside the large FG class.
    # Give the clean contour band its own mean, including BG gaps and hairs.
    edge_ce = F.cross_entropy(output["classes"].float(), labels, reduction="none")[:, None]
    classification = classification + 0.5*region_mean(edge_ce, detail)
    # Area coverage gives honest labels for pixels lost by downsampling.
    coarse_size = output["coarse_classes"].shape[-2:]
    coarse_target = F.interpolate(target, size=coarse_size, mode="area")
    classification = classification + 0.25*class_loss(output["coarse_classes"], target_classes(coarse_target))
    gradient = coverage.new_zeros(())
    for axis in (-1, -2):
        if target.shape[axis] < 2:
            continue
        delta = torch.diff(coverage, dim=axis)-torch.diff(target, dim=axis)
        edges = torch.diff(target, dim=axis).abs() > 0
        gradient = gradient + region_mean(delta.abs(), edges) + 0.25*delta.abs().mean()
    consistency = coverage.new_zeros(())
    if partner is not None:
        # Include class agreement, otherwise equal coverage heads could still
        # yield different hard BG/FG/mixed decisions on two backgrounds.
        consistency = balanced((coverage-partner["coverage"]).abs(), labels)
        consistency = consistency + (output["classes"].float().softmax(1)-partner["classes"].float().softmax(1)).abs().mean()
    pieces = {"alpha": alpha, "classification": classification, "gradient": gradient, "consistency": consistency}
    return sum(weights[k]*v for k, v in pieces.items()), {k: float(v.detach()) for k, v in pieces.items()}


@torch.no_grad()
def metrics(predicted, target):
    labels = target_classes(target)
    error = (predicted-target).abs()
    mixed = (labels == 2)[:, None]
    bg = (labels == 0)[:, None]
    fg = (labels == 1)[:, None]
    # Thin support includes opaque one-pixel strands, not only fractional hair.
    support = (target > 0).float()
    core = -F.max_pool2d(-support, 3, 1, 1)
    thin = (support-core > 0) & (F.avg_pool2d(support, 3, 1, 1) < 0.75)
    holes = []
    for background in bg[:, 0].cpu().numpy():
        # fromarray may expose a read-only buffer; floodfill otherwise silently
        # leaves the external background in place and counts it as a hole.
        flood = Image.fromarray(np.pad(background.astype(np.uint8)*255, 1, constant_values=255)).copy()
        ImageDraw.floodfill(flood, (0, 0), 0)
        holes.append(torch.from_numpy((np.asarray(flood)[1:-1, 1:-1] == 255).copy()))
    holes = torch.stack(holes)[:, None].to(predicted.device)
    return {"alpha_mae": float(error.mean()),
            "background_leak": float(region_mean(predicted, bg)),
            "opaque_error": float(region_mean(error, fg)),
            "mixed_mae": float(region_mean(error, mixed)),
            "thin_mae": float(region_mean(error, thin)),
            "thin_recall": float(region_mean((predicted > 0.02).float(), thin & (target > 0.02))),
            "hole_leak": float(region_mean(predicted, holes)),
            "background_false_positive": float(region_mean((predicted > 0.02).float(), bg))}
