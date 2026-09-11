"""Float32 matting objectives and region-specific evaluation (lower is better)."""
import torch
from torch.nn import functional as F


def masked_mean(value, weight):
    weight = weight.expand_as(value)
    dims = tuple(range(1, value.ndim))
    count = weight.sum(dims)
    per_image = (value * weight).sum(dims) / count.clamp_min(1)
    return per_image.sum() / (count > 0).sum().clamp_min(1)


def regions(alpha, valid, radius=5):
    hi = F.max_pool2d(alpha, radius*2+1, 1, radius)
    lo = -F.max_pool2d(-alpha, radius*2+1, 1, radius)
    opened = F.max_pool2d(-F.max_pool2d(-alpha, 3, 1, 1), 3, 1, 1)
    return {"opaque": (alpha >= .98).float()*valid,
            "background": (alpha <= .02).float()*valid,
            "transition": ((alpha > .005) & (alpha < .995)).float()*valid,
            "edge": ((hi-lo) > .005).float()*valid,
            "thin": ((alpha-opened) > .02).float()*valid}


def laplacian_loss(prediction, target, valid, levels=4):
    loss = prediction.new_zeros(())
    for level in range(levels):
        if min(prediction.shape[-2:]) < 4:
            break
        p_low, t_low = F.avg_pool2d(prediction, 2), F.avg_pool2d(target, 2)
        p_band = prediction-F.interpolate(p_low, prediction.shape[-2:], mode="bilinear", align_corners=False)
        t_band = target-F.interpolate(t_low, target.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + masked_mean((p_band-t_band).abs(), valid) / (2**level)
        prediction, target, valid = p_low, t_low, F.avg_pool2d(valid, 2)
    return loss


def matting_loss(prediction, batch, weights=None, radius=5):
    """Keep tiny regions normalized separately; never train RGB toward screen."""
    weights = weights or {}
    pred = {key: value.float() for key, value in prediction.items()}
    alpha, fg, valid = (batch[name].float() for name in ("alpha", "foreground", "valid"))
    masks = regions(alpha, valid, radius)
    error = (pred["alpha"]-alpha).abs()
    unknown = torch.maximum(masks["transition"], masks["edge"])
    balanced = lambda value: (masked_mean(value, masks["opaque"]) + masked_mean(value, masks["background"]) + 2*masked_mean(value, unknown))/4
    rgb_error = (pred["foreground_raw"]-fg).abs()
    visible = (alpha > .005).float()*valid
    spill_mask = ((batch["spill"] > .001).float()*visible).maximum(masks["transition"])
    clean = masks["opaque"]*(batch["spill"] < .001).float()
    bce = F.binary_cross_entropy_with_logits(pred["alpha_logits"], alpha, reduction="none")
    # Hard-pixel mining is independent of a software key or foreground size.
    hard = (error.detach() >= torch.quantile(error.detach().flatten(1), .9, dim=1)[:, None, None, None]).float()*valid
    coarse = F.binary_cross_entropy_with_logits(pred["coarse_logits"], batch["context_alpha"].float())
    # Both black and white expose errors that a single random backdrop hides.
    premult_error = pred["foreground"]*pred["alpha"]-fg*alpha
    comp = .5*(premult_error.abs() + (premult_error+alpha-pred["alpha"]).abs())
    intersection = (pred["alpha"]*alpha*valid).sum((1, 2, 3))
    denominator = ((pred["alpha"]+alpha)*valid).sum((1, 2, 3))
    losses = {
        "alpha": balanced(error), "bce": balanced(bce),
        "override": masked_mean(error, hard),
        "edge": masked_mean(error, masks["edge"]),
        "laplacian": laplacian_loss(pred["alpha"], alpha, valid),
        "detail": masked_mean(error, masks["thin"]),
        "holes": masked_mean(error, batch["holes"]*valid),
        "semantic": coarse,
        "dice": (1-(2*intersection+1)/(denominator+1)).mean(),
        "key": masked_mean((pred["key_colors"][:, 0]-batch["key_color"]).abs(), batch["key_valid"][:, None]),
        "foreground": masked_mean(rgb_error, visible*alpha.sqrt()),
        "despill": masked_mean(rgb_error, spill_mask),
        "despill_gate": masked_mean((pred["rgb_confidence"]-batch["spill"]).abs(), visible),
        "identity": masked_mean((pred["foreground_raw"]-batch["input"]).abs(), clean),
        "composite": balanced(comp),
    }
    defaults = {"alpha": 1., "bce": .25, "override": 1., "edge": 2., "laplacian": 2.,
                "detail": 2., "holes": 2., "semantic": .5, "dice": 0., "key": .25,
                "foreground": 1., "despill": 2., "despill_gate": .1, "identity": .5, "composite": 2.}
    return sum(value*float(weights.get(name, defaults[name])) for name, value in losses.items()), losses


@torch.no_grad()
def matting_metrics(foreground, prediction, batch):
    alpha, target, valid = (batch[name].float() for name in ("alpha", "foreground", "valid"))
    masks = regions(alpha, valid)
    error = (prediction.float()-alpha).abs()
    premult = foreground.float()*prediction-target*alpha
    composite = .5*(premult.abs() + (premult+alpha-prediction).abs())
    rgb_error = (foreground.float()-target).abs()
    result = {"alpha_mae": masked_mean(error, valid), "alpha_mse": masked_mean(error.square(), valid),
              "edge_mae": masked_mean(error, masks["edge"]),
              "thin_mae": masked_mean(error, masks["thin"]),
              "holes_mae": masked_mean(error, batch["holes"]*valid),
              "foreground_mae": masked_mean(rgb_error, alpha*valid),
              "despill_mae": masked_mean(rgb_error, torch.maximum(masks["transition"], (batch["spill"]>.001).float()*alpha)*valid),
              "composite_mae": masked_mean(composite, valid),
              "edge_composite_mae": masked_mean(composite, torch.maximum(masks["edge"], masks["transition"]))}
    result["selection_score"] = sum(result[name] for name in
        ("alpha_mae", "edge_mae", "thin_mae", "holes_mae", "edge_composite_mae", "despill_mae"))
    return {key: value.item() for key, value in result.items()}
