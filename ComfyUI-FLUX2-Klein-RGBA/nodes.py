"""Dedicated loader for four-channel FLUX.2 Klein autoencoders."""

from __future__ import annotations

from collections.abc import Mapping

import comfy.sd
import comfy.utils
import folder_paths


def _shape(state_dict: Mapping, key: str) -> tuple[int, ...] | None:
    tensor = state_dict.get(key)
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return tuple(int(value) for value in tensor.shape)


def is_flux2_klein_rgba_vae(state_dict: Mapping | None) -> bool:
    """Identify the full and small-decoder native FLUX.2 z=32 RGBA layouts."""
    if not isinstance(state_dict, Mapping):
        return False

    encoder = _shape(state_dict, "encoder.conv_in.weight")
    decoder_in = _shape(state_dict, "decoder.conv_in.weight")
    decoder_out = _shape(state_dict, "decoder.conv_out.weight")
    decoder_bias = _shape(state_dict, "decoder.conv_out.bias")
    batch_norm = _shape(state_dict, "bn.running_mean")
    quant = _shape(state_dict, "quant_conv.weight") or _shape(
        state_dict, "encoder.quant_conv.weight"
    )
    post_quant = _shape(state_dict, "post_quant_conv.weight") or _shape(
        state_dict, "decoder.post_quant_conv.weight"
    )

    return (
        encoder is not None
        and len(encoder) == 4
        and encoder[1] == 4
        and decoder_in is not None
        and len(decoder_in) == 4
        and decoder_in[1] == 32
        and decoder_out is not None
        and len(decoder_out) == 4
        and decoder_out[0] == 4
        and decoder_bias == (4,)
        and batch_norm == (128,)
        and quant is not None
        and quant[:2] == (64, 64)
        and post_quant is not None
        and post_quant[:2] == (32, 32)
    )


def normalize_state_dict(state_dict: Mapping) -> dict:
    """Convert AI Toolkit's nested quantizer names to ComfyUI's native names."""
    normalized = dict(state_dict)
    for source_prefix, target_prefix in (
        ("encoder.quant_conv.", "quant_conv."),
        ("decoder.post_quant_conv.", "post_quant_conv."),
    ):
        for key in tuple(normalized):
            if not key.startswith(source_prefix):
                continue
            target = target_prefix + key[len(source_prefix) :]
            if target not in normalized:
                normalized[target] = normalized[key]
            del normalized[key]
    return normalized


def build_comfy_vae_config(state_dict: Mapping) -> dict:
    """Build the exact AutoencoderKL configuration represented by the weights."""
    if not is_flux2_klein_rgba_vae(state_dict):
        raise ValueError(
            "Selected file is not a four-channel FLUX.2 Klein z=32 VAE. "
            "Use a QDM-trained Klein RGBA ae.safetensors checkpoint."
        )

    encoder = _shape(state_dict, "encoder.conv_in.weight")
    decoder_in = _shape(state_dict, "decoder.conv_in.weight")
    post_quant = _shape(state_dict, "post_quant_conv.weight") or _shape(
        state_dict, "decoder.post_quant_conv.weight"
    )
    encoder_channels = encoder[0]
    decoder_channels = decoder_in[0] // 4
    z_channels = decoder_in[1]
    ddconfig = {
        "double_z": True,
        "z_channels": z_channels,
        "resolution": 256,
        "in_channels": 4,
        "out_ch": 4,
        "ch": encoder_channels,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
        "batch_norm_latent": True,
    }
    params = {"ddconfig": ddconfig, "embed_dim": post_quant[1]}
    if decoder_channels != encoder_channels:
        decoder_ddconfig = ddconfig.copy()
        decoder_ddconfig["ch"] = decoder_channels
        params["decoder_ddconfig"] = decoder_ddconfig
    return {"params": params}


def _configure_wrapper(vae, state_dict: Mapping) -> None:
    """Set image and latent geometry skipped by ComfyUI's explicit-config path."""
    vae.latent_channels = int(state_dict["bn.running_mean"].shape[0])
    vae.output_channels = 4
    vae.conv_out_channels = 4
    vae.downscale_ratio = 16
    vae.upscale_ratio = 16
    vae.pad_channel_value = 1.0
    previous_memory_estimator = vae.memory_used_decode
    vae.memory_used_decode = (
        lambda shape, dtype: previous_memory_estimator(shape, dtype) * 4.0
    )


def load_flux2_klein_rgba_vae(
    vae_path: str,
    metadata: dict | None = None,
    device=None,
):
    """Load one disk-backed Klein RGBA VAE without changing ComfyUI globally."""
    if metadata is None:
        state_dict, metadata = comfy.utils.load_torch_file(
            vae_path, return_metadata=True
        )
    else:
        state_dict = comfy.utils.load_torch_file(vae_path)

    config = build_comfy_vae_config(state_dict)
    normalized = normalize_state_dict(state_dict)
    vae = comfy.sd.VAE(
        sd=normalized,
        config=config,
        metadata=metadata,
        device=device,
    )
    _configure_wrapper(vae, normalized)
    vae.throw_exception_if_invalid()
    vae.patcher.cached_patcher_init = (
        load_flux2_klein_rgba_vae_patcher,
        (vae_path, metadata, device),
    )
    return vae


def load_flux2_klein_rgba_vae_patcher(
    vae_path: str,
    metadata: dict | None = None,
    device=None,
    disable_dynamic: bool = False,
):
    """Reload factory used by ComfyUI's VAE device and multi-GPU features."""
    del disable_dynamic
    return load_flux2_klein_rgba_vae(vae_path, metadata, device).patcher


class Flux2KleinRGBAVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (folder_paths.get_filename_list("vae"),)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/FLUX.2 Klein RGBA"
    DESCRIPTION = (
        "Loads a four-channel z=32 FLUX.2 Klein RGBA VAE trained by "
        "Qwen Dataset Manager. Does not patch or replace ComfyUI's VAELoader."
    )

    def load_vae(self, vae_name: str):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        return (load_flux2_klein_rgba_vae(vae_path),)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinRGBAVAELoader": Flux2KleinRGBAVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinRGBAVAELoader": "Load FLUX.2 Klein RGBA VAE",
}

