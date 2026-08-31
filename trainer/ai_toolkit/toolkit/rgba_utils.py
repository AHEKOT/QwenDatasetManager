from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image


def image_has_alpha(image: Image.Image) -> bool:
    """Return True for explicit alpha bands and palette images with transparency."""
    return "A" in image.getbands() or "transparency" in image.info


def prepare_rgba_image(
    image: Image.Image,
    *,
    require_alpha: bool = True,
    alpha_threshold: float = 1.0 / 255.0,
    hidden_rgb_color: Sequence[int] = (0, 0, 0),
    unblend_background: Sequence[int] | None = None,
    edge_color_correction: str = "none",
    edge_matte_color: Sequence[int] = (0, 255, 0),
    edge_width: float = 3.0,
) -> Image.Image:
    """Convert to RGBA and remove undefined RGB hidden below the alpha threshold.

    PNG files frequently retain a previous matte (often green) in RGB even where
    alpha is zero. A four-channel VAE can see those values, so they must be made
    deterministic before encoding. Pixels with meaningful alpha are deliberately
    left in straight-alpha form.
    """
    if require_alpha and not image_has_alpha(image):
        raise ValueError(
            "RGBA training requires an image with an alpha channel; "
            f"received mode={image.mode!r}"
        )
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must be between 0 and 1")
    if len(hidden_rgb_color) != 3 or any(not 0 <= int(x) <= 255 for x in hidden_rgb_color):
        raise ValueError("hidden_rgb_color must contain three integers in [0, 255]")
    if unblend_background is not None and (
        len(unblend_background) != 3 or any(not 0 <= int(x) <= 255 for x in unblend_background)
    ):
        raise ValueError("unblend_background must contain three integers in [0, 255]")
    if edge_color_correction not in ("none", "nearest_opaque", "matte_despill"):
        raise ValueError("edge_color_correction must be 'none', 'nearest_opaque', or 'matte_despill'")
    if unblend_background is not None and edge_color_correction != "none":
        raise ValueError("unblend_background and edge_color_correction are mutually exclusive")
    if len(edge_matte_color) != 3 or any(not 0 <= int(x) <= 255 for x in edge_matte_color):
        raise ValueError("edge_matte_color must contain three integers in [0, 255]")
    if edge_width <= 0:
        raise ValueError("edge_width must be positive")

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    threshold_u8 = int(round(alpha_threshold * 255.0))
    if unblend_background is not None:
        # Optional de-matting for data whose partially transparent RGB was
        # already composited over a known background: C = a*F + (1-a)*B.
        rgba_float = rgba.astype(np.float32) / 255.0
        alpha = rgba_float[..., 3:4]
        partial = (alpha[..., 0] > alpha_threshold) & (alpha[..., 0] < 1.0)
        background = np.asarray(unblend_background, dtype=np.float32) / 255.0
        recovered = np.divide(
            rgba_float[..., :3] - (1.0 - alpha) * background,
            np.maximum(alpha, max(alpha_threshold, 1.0 / 255.0)),
        )
        rgba_float[..., :3][partial] = np.clip(recovered[partial], 0.0, 1.0)
        rgba = np.clip(np.round(rgba_float * 255.0), 0, 255).astype(np.uint8)
    elif edge_color_correction in ("nearest_opaque", "matte_despill"):
        # Chroma-key extraction often leaves a colored matte in antialiased
        # boundary pixels. Propagate straight RGB from the nearest opaque
        # foreground pixel while preserving the original alpha coverage.
        from scipy.ndimage import distance_transform_edt

        alpha = rgba[..., 3]
        foreground = alpha > threshold_u8
        opaque = alpha == 255
        partial = (alpha > threshold_u8) & (alpha < 255)
        distance_inside = distance_transform_edt(foreground)
        if edge_color_correction == "matte_despill":
            interior = opaque & (distance_inside > edge_width)
            if not interior.any():
                interior = opaque
            rgb_i16 = rgba[..., :3].astype(np.int16)
            matte = np.asarray(edge_matte_color, dtype=np.int16)
            matte_channel = int(np.argmax(matte))
            other_channels = [idx for idx in range(3) if idx != matte_channel]
            matte_like = (
                (rgb_i16[..., matte_channel] > rgb_i16[..., other_channels[0]] + 40)
                & (rgb_i16[..., matte_channel] > rgb_i16[..., other_channels[1]] + 40)
            )
            correction_mask = foreground & (distance_inside <= edge_width) & matte_like
        else:
            interior = opaque
            correction_mask = partial
        if interior.any() and correction_mask.any():
            _, nearest_indices = distance_transform_edt(~interior, return_indices=True)
            nearest_rgba = rgba[tuple(nearest_indices)]
            rgba[..., :3][correction_mask] = nearest_rgba[..., :3][correction_mask]
    hidden = rgba[..., 3] <= threshold_u8
    rgba[hidden, :3] = np.asarray(hidden_rgb_color, dtype=np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def resize_rgba_alpha_safe(
    image: Image.Image,
    size: tuple[int, int],
    resample: int = Image.Resampling.BICUBIC,
    *,
    alpha_epsilon: float = 1.0 / 255.0,
    hidden_rgb_color: Sequence[int] = (0, 0, 0),
) -> Image.Image:
    """Resize straight RGBA through premultiplied RGB to avoid colored fringes."""
    if image.mode != "RGBA":
        raise ValueError(f"alpha-safe resize expects RGBA, received {image.mode!r}")

    rgba = np.asarray(image, dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    premultiplied = np.concatenate((rgba[..., :3] * alpha, alpha), axis=-1)
    premultiplied_u8 = np.clip(np.round(premultiplied * 255.0), 0, 255).astype(np.uint8)
    resized = Image.fromarray(premultiplied_u8, mode="RGBA").resize(size, resample)

    resized_rgba = np.asarray(resized, dtype=np.float32) / 255.0
    resized_alpha = resized_rgba[..., 3:4]
    straight_rgb = np.divide(
        resized_rgba[..., :3],
        np.maximum(resized_alpha, alpha_epsilon),
        out=np.zeros_like(resized_rgba[..., :3]),
        where=resized_alpha > alpha_epsilon,
    )
    hidden = resized_alpha[..., 0] <= alpha_epsilon
    straight_rgb[hidden] = np.asarray(hidden_rgb_color, dtype=np.float32) / 255.0
    output = np.concatenate((np.clip(straight_rgb, 0.0, 1.0), resized_alpha), axis=-1)
    return Image.fromarray(np.clip(np.round(output * 255.0), 0, 255).astype(np.uint8), mode="RGBA")


def ensure_normalized_rgba_tensor(image: torch.Tensor) -> torch.Tensor:
    """Append opaque alpha to normalized RGB image/video tensors."""
    if image.ndim not in (3, 4, 5):
        raise ValueError(f"expected CHW, BCHW, or BCTHW tensor, received shape={tuple(image.shape)}")
    channel_dim = 0 if image.ndim == 3 else 1
    channels = image.shape[channel_dim]
    if channels == 4:
        return image
    if channels != 3:
        raise ValueError(f"expected 3 or 4 channels, received {channels}")

    alpha_shape = list(image.shape)
    alpha_shape[channel_dim] = 1
    # Qwen's VAE inputs are normalized to [-1, 1], so opaque alpha is +1.
    alpha = torch.ones(alpha_shape, device=image.device, dtype=image.dtype)
    return torch.cat((image, alpha), dim=channel_dim)


def rgba_tensor_to_rgb_control(
    image: torch.Tensor,
    background: Sequence[int] = (255, 255, 255),
) -> torch.Tensor:
    """Composite a normalized CHW RGBA target into an RGB [0, 1] control tensor."""
    if image.ndim != 3 or image.shape[0] != 4:
        raise ValueError(f"expected normalized CHW RGBA tensor, received shape={tuple(image.shape)}")
    if len(background) != 3 or any(not 0 <= int(x) <= 255 for x in background):
        raise ValueError("background must contain three integers in [0, 255]")

    rgba = ((image.to(torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)
    alpha = rgba[3:4]
    bg = torch.tensor(background, device=image.device, dtype=torch.float32).view(3, 1, 1) / 255.0
    return (rgba[:3] * alpha + bg * (1.0 - alpha)).to(dtype=image.dtype)


def choose_deterministic_background(
    path: str,
    backgrounds: Iterable[Sequence[int]],
) -> tuple[int, int, int]:
    palette = [tuple(int(x) for x in color) for color in backgrounds]
    if not palette:
        raise ValueError("at least one RGBA control background is required")
    if any(len(color) != 3 or any(not 0 <= x <= 255 for x in color) for color in palette):
        raise ValueError("each RGBA control background must be an RGB triplet")
    digest = hashlib.sha256(path.encode("utf-8")).digest()
    return palette[int.from_bytes(digest[:4], "big") % len(palette)]
