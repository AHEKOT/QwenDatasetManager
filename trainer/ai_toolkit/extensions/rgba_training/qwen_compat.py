from __future__ import annotations

import math


QWEN_IMAGE_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
QWEN_IMAGE_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


def validate_qwen_rgba_vae_config(config) -> None:
    input_channels = getattr(config, "input_channels", None)
    z_dim = getattr(config, "z_dim", None)
    latents_mean = getattr(config, "latents_mean", None)
    latents_std = getattr(config, "latents_std", None)

    if input_channels != 4:
        raise ValueError(
            "The transparent QIE trainer requires a four-channel "
            f"AutoencoderKLQwenImage (input_channels=4), received {input_channels!r}"
        )
    if z_dim != 16:
        raise ValueError(f"QIE2511 requires z_dim=16, received {z_dim!r}")
    if latents_mean is None or len(latents_mean) != 16:
        raise ValueError("Qwen RGBA VAE must provide 16 latents_mean values")
    if latents_std is None or len(latents_std) != 16:
        raise ValueError("Qwen RGBA VAE must provide 16 latents_std values")
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in latents_std):
        raise ValueError("Qwen RGBA VAE latents_std must contain positive finite values")


def validate_qie2511_transformer_config(config) -> None:
    in_channels = getattr(config, "in_channels", None)
    out_channels = getattr(config, "out_channels", None)
    patch_size = getattr(config, "patch_size", None)
    if in_channels != 64 or out_channels != 16 or patch_size != 2:
        raise ValueError(
            "Unexpected QIE transformer latent contract: expected "
            f"in_channels=64, out_channels=16, patch_size=2; received "
            f"{in_channels=}, {out_channels=}, {patch_size=}"
        )

