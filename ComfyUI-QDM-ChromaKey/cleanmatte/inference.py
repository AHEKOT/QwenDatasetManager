"""Portable inference, versioned checkpoints and optional fixed-alpha unmixing."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import CleanMatte, MODEL_ID, resize


def load_model(path, device="cpu"):
    if Path(path).suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            architecture = (handle.metadata() or {}).get("architecture")
        checkpoint = {"architecture": architecture, "model": load_file(str(path))}
    else:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != MODEL_ID:
        raise ValueError("This node requires a CleanMatte v1 checkpoint trained from scratch.")
    model = CleanMatte()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.fuse_for_inference().to(device=device, memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, rgb, tile_size=512):
    """Bounded activation memory; every native pixel is examined.

    Global context is shared between tiles. Even tile boundaries preserve the
    stride-2 phase. Halo pixels are discarded, never blended into soft alpha.
    Padding to even dimensions makes dense and tiled local sampling agree.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 1:
        raise ValueError("Expected nonempty BCHW RGB input.")
    h, w = rgb.shape[-2:]
    tile_size = max(32, int(tile_size) // 2 * 2)
    image = F.pad(rgb, (0, w % 2, 0, h % 2), mode="replicate")
    ph, pw = image.shape[-2:]
    context = model.encode(image)
    # Only eleven cheap guidance channels are resized; expensive local
    # activations stay inside tiles. This also gives exact tile coordinates.
    guidance = resize(context, (ph, pw))
    alpha = torch.empty((rgb.shape[0], 1, ph, pw), device=rgb.device, dtype=torch.float32)
    halo = model.tile_halo
    for y in range(0, ph, tile_size):
        for x in range(0, pw, tile_size):
            y1, x1 = min(y + tile_size, ph), min(x + tile_size, pw)
            top, left = max(0, y-halo), max(0, x-halo)
            bottom, right = min(ph, y1+halo), min(pw, x1+halo)
            out = model.refine(image[..., top:bottom, left:right], guidance[..., top:bottom, left:right])
            alpha[..., y:y1, x:x1] = out["alpha"][..., y-top:y1-top, x-left:x1-left]
    return alpha[..., :h, :w]


@torch.inference_mode()
def recover_foreground(rgb, alpha, screen_rgb=None, strength=1.0):
    """Optional deterministic screen subtraction/despill; alpha is read-only.

    Works in the supplied RGB encoding. A calibrated screen colour is best.
    No claim is made to reconstruct completely occluded foreground colours.
    """
    if strength <= 0:
        return rgb.clone()
    if screen_rgb is None:
        screens, valid = [], []
        for image, matte in zip(rgb, alpha):
            bg = matte[0] == 0
            if bg.sum() < 16:
                screens.append(image.new_zeros(3))
                valid.append(0.0)
            else:
                screens.append(image[:, bg].median(dim=1).values)
                valid.append(1.0)
        screen = torch.stack(screens)[:, :, None, None]
        available = rgb.new_tensor(valid)[:, None, None, None]
    else:
        screen = torch.as_tensor(screen_rgb, device=rgb.device, dtype=rgb.dtype).reshape(-1, 3, 1, 1)
        available = 1.0
    a = alpha.detach().to(rgb.dtype)
    unmix = ((rgb - (1-a)*screen) / a.clamp_min(0.05)).clamp(0, 1)
    # Separate reflected chroma spill from background mixing. Suppress only
    # the dominant green/blue screen component on semi-transparent edges.
    dominant = screen.argmax(1, keepdim=True)
    chromatic = (screen.amax(1, keepdim=True)-screen.amin(1, keepdim=True) > 0.25)
    colour_mask = torch.zeros_like(screen).scatter_(1, dominant, 1)
    ceiling = (unmix*(1-colour_mask)).amax(1, keepdim=True)
    excess = (unmix-ceiling).clamp_min(0)*colour_mask
    unmix = unmix-excess*chromatic*(dominant != 0)
    # Never change solid interiors. Limit ill-conditioned low-alpha recovery.
    amount = ((1-a) * 4).clamp(0, 1) * (a / 0.05).clamp(0, 1) * min(float(strength), 1.0)*available
    result = torch.lerp(rgb, unmix, amount)
    return torch.where(a > 0, result, 0.0)
