from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class RGBALoRALossMixin:
    """Cheap alpha-aware weighting for the normal RGBA latent diffusion loss.

    The four-channel VAE already embeds RGB and alpha into the training latent.
    This mixin only reweights that existing element-wise loss around balanced
    foreground/background regions and alpha boundaries. It never runs another
    transformer pass, decodes the VAE, or adds a second backward graph.
    """

    supports_rgba_latent_loss = True

    def _rgba_loss_setting(self, name: str, default: float) -> float:
        key = f"rgba_lora_loss_{name}"
        value = self.model_config.model_kwargs.get(key, default)
        return max(0.0, float(value))

    def get_rgba_latent_loss_multiplier(
        self,
        batch: "DataLoaderBatchDTO",
        *,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target = batch.tensor
        if target is None or target.ndim != 4 or target.shape[1] != 4:
            raise ValueError(
                "Transparent LoRA latent weighting requires the raw BCHW RGBA target tensor"
            )

        with torch.no_grad():
            # Downsample where the raw batch tensor already lives, then transfer
            # only the tiny one-channel latent-sized map to the training device.
            alpha = (target[:, 3:4].detach().to(dtype=torch.float32) + 1.0) * 0.5
            alpha = F.interpolate(alpha, size=size, mode="bilinear", align_corners=False)
            alpha = alpha.clamp(0.0, 1.0)

            foreground_mean = alpha.mean(dim=(2, 3), keepdim=True).clamp_min(0.05)
            background = 1.0 - alpha
            background_mean = background.mean(dim=(2, 3), keepdim=True).clamp_min(0.05)
            balanced = 0.5 * (alpha / foreground_mean + background / background_mean)

            alpha_strength = self._rgba_loss_setting("alpha", 4.0)
            blend = alpha_strength / (alpha_strength + 1.0)
            weight = 1.0 + blend * (balanced - 1.0)

            maximum = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
            minimum = -F.max_pool2d(-alpha, kernel_size=3, stride=1, padding=1)
            edge = (maximum - minimum).clamp(0.0, 1.0)
            edge_strength = self._rgba_loss_setting("alpha_edge", 2.0)
            weight = weight * (1.0 + edge_strength * edge)

            weight = weight.clamp_min(0.05)
            weight = weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            return weight.to(device=device, dtype=dtype, non_blocking=True)
