from __future__ import annotations

import hashlib

import torch

from extensions_built_in.diffusion_models.flux2.flux2_klein_model import (
    Flux2Klein4BModel,
    Flux2Klein9BModel,
)
from toolkit.rgba_utils import ensure_normalized_rgba_tensor


class Flux2KleinRGBAMixin:
    """FLUX.2 Klein backend for a separately trained four-channel VAE.

    Klein uses a 32-channel latent space, so the Qwen RGBA VAE cannot be
    substituted here. The selected checkpoint must be a FLUX.2 autoencoder
    with four-channel input/output boundaries and the original z=32 interior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.model_config.vae_path:
            raise ValueError(
                "Transparent FLUX.2 Klein training requires vae_path pointing "
                "to a compatible four-channel FLUX.2 VAE"
            )
        identity = f"{self.model_config.vae_path}|rgba-preprocess-v1"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        self.latent_space_version = f"flux2-klein-rgba-{digest}"

    def get_flux2_autoencoder_params(self, vae_state_dict):
        params = super().get_flux2_autoencoder_params(vae_state_dict)
        encoder = vae_state_dict.get("encoder.conv_in.weight")
        decoder = vae_state_dict.get("decoder.conv_out.weight")
        decoder_bias = vae_state_dict.get("decoder.conv_out.bias")
        if encoder is None or decoder is None or decoder_bias is None:
            raise ValueError("Selected file is not a FLUX.2 autoencoder checkpoint")
        if encoder.shape[1] != 4 or decoder.shape[0] != 4 or decoder_bias.shape[0] != 4:
            raise ValueError(
                "Transparent FLUX.2 Klein requires an RGBA VAE; received "
                f"encoder={tuple(encoder.shape)}, decoder={tuple(decoder.shape)}"
            )
        if params.z_channels != 32:
            raise ValueError(
                f"Transparent FLUX.2 Klein VAE must use z=32, received {params.z_channels}"
            )
        params.in_channels = 4
        params.out_ch = 4
        return params

    def prepare_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        return ensure_normalized_rgba_tensor(image)

    def decode_latents(self, latents, device=None, dtype=None):
        images = super().decode_latents(latents, device=device, dtype=dtype)
        if images.shape[1] != 4:
            raise ValueError(
                "The configured transparent FLUX.2 VAE decoded "
                f"{images.shape[1]} channels instead of RGBA"
            )
        return images


class Flux2Klein4BRGBAModel(Flux2KleinRGBAMixin, Flux2Klein4BModel):
    arch = "flux2_klein_4b_rgba"


class Flux2Klein9BRGBAModel(Flux2KleinRGBAMixin, Flux2Klein9BModel):
    arch = "flux2_klein_9b_rgba"
