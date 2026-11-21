"""Safetensors loading and bounded-memory tiled inference."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .model import (
    ARCHITECTURE_ID, ARCHITECTURE_ID_V2, ARCHITECTURE_ID_V3,
    AnimeKeyMatte, AnimeKeyMatteV2, AnimeKeyMatteV3,
)
from .v4 import ARCHITECTURE_ID_V4, KeyMatteV4


def load_model(path, device="cpu", dtype=torch.float32):
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    architecture = metadata.get("architecture", ARCHITECTURE_ID)
    model_types = {
        ARCHITECTURE_ID: AnimeKeyMatte,
        ARCHITECTURE_ID_V2: AnimeKeyMatteV2,
        ARCHITECTURE_ID_V3: AnimeKeyMatteV3,
        ARCHITECTURE_ID_V4: KeyMatteV4,
    }
    if architecture not in model_types:
        raise ValueError(f"Unsupported chromakey architecture: {architecture}")
    model = model_types[architecture]()
    missing, unexpected = model.load_state_dict(load_file(str(path), device="cpu"), strict=False)
    if missing or unexpected:
        raise ValueError(f"Invalid checkpoint (missing={missing}, unexpected={unexpected})")
    return model.eval().to(device=device, dtype=dtype), metadata


def _positions(length, tile, overlap):
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    positions = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _window(height, width, device, dtype):
    wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return wy[:, None] * wx[None, :]


@torch.inference_mode()
def run_tiled(model, rgb, tile_size=1024, overlap=96, autocast_dtype=None):
    """Run one CHW RGB image without changing its aspect ratio or dimensions."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image, received {tuple(rgb.shape)}")
    if isinstance(model, KeyMatteV4):
        return _run_v4(model, rgb, tile_size, autocast_dtype)
    tile_size = max(128, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    height, width = rgb.shape[-2:]
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    output_device = rgb.device
    alpha_sum = torch.zeros((1, height, width), dtype=torch.float32, device=output_device)
    rgb_sum = torch.zeros((3, height, width), dtype=torch.float32, device=output_device)
    weight_sum = torch.zeros((1, height, width), dtype=torch.float32, device=output_device)
    key_context = None
    if hasattr(model, "estimate_key_context"):
        context_rgb = rgb.unsqueeze(0)
        context_scale = min(1.0, 256.0 / max(height, width))
        if context_scale < 1.0:
            context_rgb = torch.nn.functional.interpolate(
                context_rgb,
                (max(32, round(height * context_scale)), max(32, round(width * context_scale))),
                mode="area",
            )
        context_rgb = context_rgb.to(device=device, dtype=model_dtype)
        key_context = model.estimate_key_context(context_rgb)
    for top in _positions(height, tile_size, overlap):
        for left in _positions(width, tile_size, overlap):
            patch = rgb[:, top:top + tile_size, left:left + tile_size]
            patch_h, patch_w = patch.shape[-2:]
            pad_h = int(math.ceil(patch_h / model.stride) * model.stride - patch_h)
            pad_w = int(math.ceil(patch_w / model.stride) * model.stride - patch_w)
            mode = "reflect" if patch_h > pad_h and patch_w > pad_w else "replicate"
            patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode=mode)
            patch = patch.unsqueeze(0).to(device=device, dtype=model_dtype)
            enabled = autocast_dtype is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=enabled):
                prediction = model(patch, key_context=key_context) if key_context is not None else model(patch)
            alpha = prediction["alpha"][0, :, :patch_h, :patch_w].float().to(output_device)
            foreground = prediction["foreground"][0, :, :patch_h, :patch_w].float().to(output_device)
            weight = _window(patch_h, patch_w, output_device, torch.float32).unsqueeze(0)
            alpha_sum[:, top:top + patch_h, left:left + patch_w] += alpha * weight
            rgb_sum[:, top:top + patch_h, left:left + patch_w] += foreground * weight
            weight_sum[:, top:top + patch_h, left:left + patch_w] += weight
    weight_sum.clamp_min_(1e-6)
    return rgb_sum / weight_sum, alpha_sum / weight_sum


@torch.inference_mode()
def _run_v4(model, rgb, tile_size, autocast_dtype):
    """Encode once; refine halo tiles and retain only their valid centres.

    Unlike overlap averaging, this also preserves straight foreground colour
    at transparent boundaries. No entire high-resolution feature map on GPU.
    """
    tile_size = max(32, int(tile_size))
    height, width = rgb.shape[-2:]
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    enabled = device.type == "cuda" and autocast_dtype is not None
    small = torch.nn.functional.interpolate(rgb[None],
        (model.context_size, model.context_size), mode="area").to(device=device, dtype=dtype)
    foreground = torch.empty_like(rgb, dtype=torch.float32)
    alpha = torch.empty_like(rgb[:1], dtype=torch.float32)
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=enabled):
        context = model.encode_context(small)
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom, right = min(height, top + tile_size), min(width, left + tile_size)
                y0, x0 = max(0, top - model.tile_halo), max(0, left - model.tile_halo)
                y1, x1 = min(height, bottom + model.tile_halo), min(width, right + model.tile_halo)
                patch = rgb[None, :, y0:y1, x0:x1].to(device=device, dtype=dtype)
                box = torch.tensor([[x0 / width, y0 / height, x1 / width, y1 / height]], device=device)
                prediction = model.refine(patch, context, box)
                crop = (0, slice(None), slice(top-y0, bottom-y0), slice(left-x0, right-x0))
                foreground[:, top:bottom, left:right] = prediction["foreground"][crop].float().to(rgb.device)
                alpha[:, top:bottom, left:right] = prediction["alpha"][crop].float().to(rgb.device)
    return foreground, alpha
