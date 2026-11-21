# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
"""QDM layered QI2 format v1. No dependency on the training application.

Sequence layout follows QDM's qwen_image_2/src/transformer.py and pipeline.py.
ComfyUI owns model loading, weight patches, offloading and actual attention.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from PIL import Image

FORMAT_VERSION = 1


def legacy_timestep_forward(self, timestep, dtype):
    """Match QDM adapters trained before the fp32 timestep-frequency fix."""
    freqs = torch.exp(-math.log(10000) * torch.arange(
        128, device=timestep.device, dtype=torch.float32) / 128).to(torch.bfloat16).float()
    args = (1000 * timestep.float())[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
    return self.timestep_embedder(embedding)


def validate_geometry(width, height, slots):
    if isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 20:
        raise ValueError('Layer slots must equal the training setting (1 to 20).')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 32 or v % 32 for v in (width, height)):
        raise ValueError('Width and height must be positive multiples of 32.')


def flow_sigmas(width, height, slots, steps):
    """Exact QDM preview schedule: dynamic exponential shift, terminal .02, then 0."""
    validate_geometry(width, height, slots)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError('At least two sampling steps are required.')
    tokens = slots * (height // 16) * (width // 16)
    mu = .5 + (tokens - 256) * (.9 - .5) / (8192 - 256)
    sigmas = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float64)
    exp_mu = math.exp(mu)
    sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
    one_minus = 1.0 - sigmas
    sigmas = 1.0 - one_minus / (one_minus[-1] / .98)
    return torch.from_numpy(np.append(sigmas, 0.0).astype(np.float32))


def joint_build_sequence(self, x, context, ref_latents, image_slots, *, layer_slots):
    """Native Comfy QI2 build_sequence replacement, scoped to one ModelPatcher.

Targets live in B,C,L*h,w storage. They have individual RoPE grids, but ONE
final attention segment: every target query sees every target and the prefix.
"""
    if x.ndim != 4 or x.shape[1] != 64 or x.shape[-2] % layer_slots:
        raise ValueError('Expected joint QI2 latent storage B,64,L*h,w.')
    h, w = x.shape[-2] // layer_slots, x.shape[-1]
    validate_geometry(w * 16, h * 16, layer_slots)
    txt = self.txt_in(context)
    refs, slots = list(ref_latents), list(image_slots)
    if len(slots) != len(refs) or slots != sorted(slots) or any(not 0 <= s <= txt.shape[1] for s in slots):
        raise ValueError('Reference image slots do not match the QI2 conditioning. Use the QDM conditioning node.')
    parts, ids, segments = [], [], []
    position, length = 0, 0

    def add_text(start, end):
        nonlocal position, length
        n = end - start
        if not n:
            return
        parts.append(txt[:, start:end])
        ids.append(torch.arange(position, position+n, device=x.device, dtype=torch.float32)[:, None].expand(n, 3))
        segments.append((length, length+n, torch.ones((n, length+n), dtype=torch.bool, device=x.device).tril(length)))
        position += n
        length += n

    def add_image(img, is_target):
        nonlocal position, length
        ih, iw = img.shape[-2:]
        if img.ndim != 4 or img.shape[:2] != x.shape[:2] or ih % 2 or iw % 2:
            raise ValueError('Every reference and target must be a 64-channel image on the 32-pixel grid.')
        parts.append(self.img_in(img.flatten(2).transpose(1, 2)))
        yy = torch.arange(ih, device=x.device, dtype=torch.float32) - (ih - ih // 2)
        xx = torch.arange(iw, device=x.device, dtype=torch.float32) - (iw - iw // 2)
        ids.append(torch.stack((torch.full((ih, iw), position, device=x.device, dtype=torch.float32),
                                yy[:, None].expand(ih, iw), xx[None, :].expand(ih, iw)), dim=-1).flatten(0, 1))
        if not is_target:
            segments.append((length, length+ih*iw, None))
        position += 1 if is_target and layer_slots > 1 else max(ih, iw)
        length += ih * iw

    cursor = 0
    for slot, reference in zip(slots, refs):
        add_text(cursor, slot)
        add_image(reference, False)
        cursor = slot
    add_text(cursor, txt.shape[1])
    target_start = length
    for layer in x.chunk(layer_slots, dim=2):
        add_image(layer, True)
    segments.append((target_start, length, None))
    pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
    return torch.cat(parts, dim=1), pe, segments


def to_pil_layers(rgba):
    if rgba.ndim != 4 or rgba.shape[-1] != 4 or not 1 <= rgba.shape[0] <= 20:
        raise ValueError('Expected one document as L,H,W,4 RGBA layers, bottom to top.')
    if not torch.isfinite(rgba).all():
        raise ValueError('Generated layers contain non-finite pixels.')
    pixels = (rgba.detach().float().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    pixels[pixels[..., 3] == 0, :3] = 0
    return [Image.fromarray(layer) for layer in pixels]


def composite_layers(layers):
    if not layers:
        raise ValueError('Empty document.')
    result = Image.new('RGBA', layers[0].size, (0, 0, 0, 0))
    for layer in layers:
        if layer.mode != 'RGBA' or layer.size != result.size:
            raise ValueError('Layers must be RGBA canvases of the same size.')
        result = Image.alpha_composite(result, layer)
    return result


def save_psd(path, layers):
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise RuntimeError('Install this pack\'s requirements.txt in the ComfyUI Python environment.') from exc
    composite_layers(layers)
    # RGBA document mode keeps per-layer alpha as the native transparency
    # channel. RGB mode in psd-tools 1.22 stores it in a separate pixel mask.
    psd = PSDImage.new('RGBA', layers[0].size, color=(0, 0, 0, 0))
    for index, layer in enumerate(layers):
        psd.create_pixel_layer(layer, name=f'Layer {index+1:02d}', top=0, left=0)
    psd.save(path)
