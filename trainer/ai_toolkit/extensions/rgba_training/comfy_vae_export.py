from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file


_EXACT_KEYS = {
    "quant_conv.weight": "conv1.weight",
    "quant_conv.bias": "conv1.bias",
    "post_quant_conv.weight": "conv2.weight",
    "post_quant_conv.bias": "conv2.bias",
    "encoder.conv_in.weight": "encoder.conv1.weight",
    "encoder.conv_in.bias": "encoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_in.bias": "decoder.conv1.bias",
    "encoder.norm_out.gamma": "encoder.head.0.gamma",
    "encoder.conv_out.weight": "encoder.head.2.weight",
    "encoder.conv_out.bias": "encoder.head.2.bias",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
}

_RESIDUAL_SUFFIXES = {
    "norm1.gamma": "residual.0.gamma",
    "conv1.weight": "residual.2.weight",
    "conv1.bias": "residual.2.bias",
    "norm2.gamma": "residual.3.gamma",
    "conv2.weight": "residual.6.weight",
    "conv2.bias": "residual.6.bias",
    "conv_shortcut.weight": "shortcut.weight",
    "conv_shortcut.bias": "shortcut.bias",
}


def diffusers_qwen_vae_key_to_comfy(key: str) -> str:
    """Map Diffusers AutoencoderKLQwenImage keys to Comfy's native Wan/Qwen keys."""
    if key in _EXACT_KEYS:
        return _EXACT_KEYS[key]

    match = re.fullmatch(r"(encoder|decoder)\.mid_block\.resnets\.(0|1)\.(.+)", key)
    if match:
        side, block, suffix = match.groups()
        if suffix not in _RESIDUAL_SUFFIXES:
            raise KeyError(key)
        return f"{side}.middle.{int(block) * 2}.{_RESIDUAL_SUFFIXES[suffix]}"

    match = re.fullmatch(r"(encoder|decoder)\.mid_block\.attentions\.0\.(.+)", key)
    if match:
        side, suffix = match.groups()
        if suffix not in {
            "norm.gamma",
            "to_qkv.weight",
            "to_qkv.bias",
            "proj.weight",
            "proj.bias",
        }:
            raise KeyError(key)
        return f"{side}.middle.1.{suffix}"

    match = re.fullmatch(r"encoder\.down_blocks\.(\d+)\.(.+)", key)
    if match:
        block, suffix = match.groups()
        native_suffix = _RESIDUAL_SUFFIXES.get(suffix, suffix)
        return f"encoder.downsamples.{block}.{native_suffix}"

    match = re.fullmatch(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", key)
    if match:
        block, resnet, suffix = match.groups()
        if suffix not in _RESIDUAL_SUFFIXES:
            raise KeyError(key)
        native_block = int(block) * 4 + int(resnet)
        return f"decoder.upsamples.{native_block}.{_RESIDUAL_SUFFIXES[suffix]}"

    match = re.fullmatch(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)", key)
    if match:
        block, suffix = match.groups()
        native_block = int(block) * 4 + 3
        return f"decoder.upsamples.{native_block}.{suffix}"

    raise KeyError(f"Unsupported Diffusers Qwen VAE key: {key}")


def convert_diffusers_qwen_vae_to_comfy(
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    encoder_weight = state_dict.get("encoder.conv_in.weight")
    decoder_weight = state_dict.get("decoder.conv_out.weight")
    decoder_bias = state_dict.get("decoder.conv_out.bias")
    if encoder_weight is None or decoder_weight is None or decoder_bias is None:
        raise ValueError("State dict is not an AutoencoderKLQwenImage VAE")
    if encoder_weight.shape[1] != 4 or decoder_weight.shape[0] != 4 or decoder_bias.shape[0] != 4:
        raise ValueError(
            "Expected an RGBA Qwen VAE with four-channel encoder/decoder boundaries; "
            f"received {tuple(encoder_weight.shape)}, {tuple(decoder_weight.shape)}, "
            f"{tuple(decoder_bias.shape)}"
        )

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        native_key = diffusers_qwen_vae_key_to_comfy(key)
        if native_key in converted:
            raise RuntimeError(f"Duplicate native Qwen VAE key: {native_key}")
        converted[native_key] = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return converted


def save_qwen_rgba_vae_for_comfy(
    state_dict: Mapping[str, torch.Tensor],
    output: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    metadata: dict[str, str] | None = None,
) -> Path:
    native_state = convert_diffusers_qwen_vae_to_comfy(state_dict, dtype=dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    file_metadata = {
        "format": "pt",
        "architecture": "qwen_image_vae_rgba",
        "input_channels": "4",
        "output_channels": "4",
    }
    if metadata:
        file_metadata.update({str(key): str(value) for key, value in metadata.items()})
    save_file(native_state, str(temporary), metadata=file_metadata)
    temporary.replace(output)
    return output
