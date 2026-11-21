from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


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
    elif edge_color_correction in ("nearest_opaque", "matte_despill") and np.any(rgba[..., 3] < 255):
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

def rgba_tensor_to_rgb_control_image(
    image: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Composite a normalized CHW RGBA target over an RGB [0, 1] image tensor."""
    if image.ndim != 3 or image.shape[0] != 4:
        raise ValueError(f"expected normalized CHW RGBA tensor, received shape={tuple(image.shape)}")
    if background.ndim != 3 or background.shape[0] != 3:
        raise ValueError(
            f"expected CHW RGB background tensor, received shape={tuple(background.shape)}"
        )
    if tuple(background.shape[1:]) != tuple(image.shape[1:]):
        raise ValueError("RGBA target and RGB background must have matching spatial dimensions")

    rgba = ((image.to(torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)
    alpha = rgba[3:4]
    bg = background.to(device=image.device, dtype=torch.float32).clamp(0.0, 1.0)
    return (rgba[:3] * alpha + bg * (1.0 - alpha)).to(dtype=image.dtype)


def fit_rgb_background(image: Image.Image, size: Sequence[int]) -> Image.Image:
    """Convert any PIL color mode to RGB and resize-to-cover with a centered crop."""
    if len(size) != 2:
        raise ValueError("background target size must contain width and height")
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("background target width and height must be positive")
    return ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def prepare_rgba_validation_pair(
    image: Image.Image,
    size: tuple[int, int],
    *,
    control_mode: str,
    background_image: Image.Image | None = None,
    control_images: list[Image.Image] | None = None,
    alpha_threshold: float = 1.0 / 255.0,
    hidden_rgb_color: Sequence[int] = (0, 0, 0),
    edge_color_correction: str = "none",
    edge_matte_color: Sequence[int] = (0, 255, 0),
    edge_width: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]]:
    """Build the deterministic RGBA target and matching RGB validation control.

    The target uses the exact training cleanup and an alpha-safe resize.  Edit
    mode composites it over a fixed opaque background; when none is supplied an
    intentionally varied deterministic background is generated locally.
    Generation mode uses the same black control as training. Paired mode accepts
    RGB or RGBA targets and returns the supplied RGB controls in their original
    sizes. Returned target is CHW [-1, 1]; controls are CHW RGB [0, 1].
    """

    if control_mode not in {"edit", "generation", "paired"}:
        raise ValueError("RGBA validation control_mode must be edit, generation or paired")
    if control_mode == 'paired' and not control_images:
        raise ValueError('Paired RGBA validation requires input control images')
    prepared = prepare_rgba_image(
        image,
        require_alpha=control_mode != 'paired',
        alpha_threshold=alpha_threshold,
        hidden_rgb_color=hidden_rgb_color,
        edge_color_correction=edge_color_correction,
        edge_matte_color=edge_matte_color,
        edge_width=edge_width,
    )
    prepared = resize_rgba_alpha_safe(
        prepared,
        size,
        alpha_epsilon=alpha_threshold,
        hidden_rgb_color=hidden_rgb_color,
    )
    rgba_array = np.asarray(prepared, dtype=np.float32) / 255.0
    target = torch.from_numpy(rgba_array.copy()).permute(2, 0, 1) * 2.0 - 1.0

    if control_mode == 'paired':
        control = []
        for reference in control_images:
            rgba_reference = reference.convert('RGBA')
            opaque = Image.new('RGBA', reference.size, (0, 0, 0, 255))
            rgb = Image.alpha_composite(opaque, rgba_reference).convert('RGB')
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            control.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    elif control_mode == "generation":
        control = torch.zeros((3, size[1], size[0]), dtype=torch.float32)
    else:
        if background_image is None:
            width, height = size
            yy, xx = np.mgrid[0:height, 0:width]
            tile = max(8, min(width, height) // 12)
            checker = ((xx // tile + yy // tile) % 2).astype(np.float32)
            x_norm = xx.astype(np.float32) / max(width - 1, 1)
            y_norm = yy.astype(np.float32) / max(height - 1, 1)
            color_a = np.stack((
                35.0 + 80.0 * x_norm,
                80.0 + 100.0 * y_norm,
                190.0 - 70.0 * x_norm,
            ), axis=-1)
            color_b = np.stack((
                235.0 - 60.0 * y_norm,
                155.0 + 60.0 * x_norm,
                45.0 + 75.0 * y_norm,
            ), axis=-1)
            generated = color_a * (1.0 - checker[..., None]) + color_b * checker[..., None]
            background_image = Image.fromarray(
                np.clip(generated, 0, 255).astype(np.uint8), mode="RGB"
            )
        fitted = fit_rgb_background(background_image, size)
        background_array = np.asarray(fitted, dtype=np.float32) / 255.0
        background = torch.from_numpy(background_array.copy()).permute(2, 0, 1)
        control = rgba_tensor_to_rgb_control_image(target, background)
    return target, control
