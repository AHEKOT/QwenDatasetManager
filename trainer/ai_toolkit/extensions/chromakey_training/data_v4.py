"""Reproducible native-detail sampling with matching whole-image context."""
from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import synthesize_background, add_foreground_hard_negative, _chance, image_has_alpha


class CurriculumBatchSampler:
    """Resume at a consumed batch, without counting DataLoader prefetch."""
    def __init__(self, base, skip, start_step, total_steps, accumulation):
        self.base, self.skip, self.start_step = base, skip, start_step
        self.total_steps, self.accumulation = total_steps, accumulation

    def __iter__(self):
        for index, batch in enumerate(self.base):
            if index < self.skip:
                continue
            progress = (self.start_step + (index-self.skip)/self.accumulation) / max(1, self.total_steps)
            yield [(*request, progress) for request in batch]


def srgb_to_linear(value):
    return torch.where(value <= 0.04045, value / 12.92,
                       ((value + 0.055) / 1.055).clamp_min(0).pow(2.4))


def linear_to_srgb(value):
    value = value.clamp_min(0)
    return torch.where(value <= 0.0031308, value * 12.92,
                       1.055 * value.clamp_min(1e-8).pow(1 / 2.4) - 0.055)


def erode_alpha(alpha, radius):
    # OpenCV's separable morphology avoids a Python/PyTorch CPU max-pool over
    # hundreds of neighbours per source pixel for wide spill bands.
    kernel = np.ones((radius*2+1, radius*2+1), dtype=np.uint8)
    return torch.from_numpy(cv2.erode(alpha[0].numpy(), kernel,
                                     borderType=cv2.BORDER_REPLICATE))[None]


def initialize_worker(_worker_id):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def resize_rgba_float(image, size):
    """Premultiply and resample in float, never quantize premultiplied RGB."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    rgba[..., :3] *= rgba[..., 3:4]
    channels = [np.asarray(Image.fromarray(rgba[..., c]).resize(size, Image.Resampling.BILINEAR)).copy()
                for c in range(4)]
    value = torch.from_numpy(np.stack(channels)).clamp(0, 1)
    alpha = value[3:4]
    foreground = torch.where(alpha > 1e-6, value[:3] / alpha.clamp_min(1e-6), 0).clamp(0, 1)
    return foreground, alpha


def enclosed_background(alpha):
    """All disconnected transparent regions, including narrow internal holes."""
    mask = np.pad((alpha[0].numpy() > 0.02).astype(np.uint8), 1)
    cv2.floodFill(mask, None, (0, 0), 2, flags=4)
    return torch.from_numpy((mask[1:-1, 1:-1] == 0).copy()).float()[None]


def split_sources(files, fraction=0.05, seed=42):
    """Content-deduplicated source holdout, before any crops/compositing.

    Exact file duplicates cannot cross splits. Variants of the same character
    must still be curated into a separate evaluation dataset by the operator.
    """
    unique = {}
    for path in files:
        path = Path(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        with Image.open(path) as image:
            if not image_has_alpha(image):
                raise ValueError(f"Missing alpha channel: {path}")
            lo, hi = image.convert("RGBA").getchannel("A").getextrema()
            if lo == hi:
                raise ValueError(f"Alpha must contain foreground and transparency: {path}")
        unique.setdefault(digest, path)
    ranked = sorted(unique, key=lambda digest: hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest())
    if len(ranked) < 2:
        raise ValueError("Need at least two distinct RGBA sources for train/holdout")
    count = min(len(ranked) - 1, max(1, round(len(ranked) * fraction)))
    return [unique[key] for key in ranked[count:]], [unique[key] for key in ranked[:count]]


def add_topology(foreground, alpha, rng):
    """Small supersampled stress patches: fractional 1–2px strands and holes."""
    height, width = alpha.shape[-2:]
    points = torch.nonzero(alpha[0] > 0.8)
    if not len(points):
        return foreground, alpha
    foreground, alpha = foreground.clone(), alpha.clone()
    for _ in range(rng.randint(1, 4)):
        y, x = points[rng.randrange(len(points))].tolist()
        size = min(48, height, width)
        y0, x0 = min(max(0, y - size // 2), height-size), min(max(0, x - size // 2), width-size)
        mask = Image.new("L", (size * 4, size * 4), 0)
        draw = ImageDraw.Draw(mask)
        if rng.random() < 0.5:
            radius = rng.randint(2, max(2, size // 4))
            center = size * 2
            draw.ellipse((center-radius*4, center-radius*2, center+radius*4, center+radius*2), fill=255)
            hole = torch.from_numpy(np.asarray(mask.resize((size, size), Image.Resampling.LANCZOS)).copy()).float()[None] / 255
            alpha[:, y0:y0+size, x0:x0+size] *= 1 - hole
        else:
            draw.line([(rng.randrange(size*4), rng.randrange(size*4)) for _ in range(3)],
                      fill=rng.randint(80, 220), width=rng.choice((4, 8)))
            strand = torch.from_numpy(np.asarray(mask.resize((size, size), Image.Resampling.LANCZOS)).copy()).float()[None] / 255
            old = alpha[:, y0:y0+size, x0:x0+size]
            new = old + strand * (1 - old)
            color = foreground[:, y, x].clone()[:, None, None]
            patch = foreground[:, y0:y0+size, x0:x0+size]
            foreground[:, y0:y0+size, x0:x0+size] = (patch * old + color * strand * (1-old)) / new.clamp_min(1e-6)
            alpha[:, y0:y0+size, x0:x0+size] = new
    return foreground, alpha


class ChromaKeyDatasetV4(Dataset):
    def __init__(self, files, augmentation, flip_probability=0.5):
        self.files = list(files)
        self.augmentation = dict(augmentation)
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.files)

    def __getitem__(self, request):
        # A per-request generator covers *all* torch and Python augmentation,
        # independently of workers, prefetch order or validation frequency.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(request[3]))
            return self._sample(request)

    def _sample(self, request):
        index, height, width, seed, *extra = request
        progress = float(extra[0]) if extra else 1.0
        rng = random.Random(seed)
        config = dict(self.augmentation)
        # First 10%: global structure, then native detail with full difficulty
        # reached by 30%. Never change the alpha definition during curriculum.
        difficulty = min(1.0, max(0.0, (progress - 0.1) / 0.2))
        for name in ("spill_strength_max", "dirt_strength", "noise_strength", "gradient_strength"):
            if name in config:
                config[name] *= 0.25 + difficulty * 0.75
        config["collision_chance"] = config.get("collision_chance", 5) * difficulty
        detail = progress >= 0.1 and _chance({"p": config.get("detail_crop_percent", 70)}, "p", rng)
        with Image.open(self.files[index]) as image:
            if not image_has_alpha(image):
                raise ValueError(f"Missing alpha channel: {self.files[index]}")
            if detail:
                # Keep original pixel scale for hair. Only exceptionally large
                # sources are capped to bound CPU RAM usage.
                scale = min(1.0, float(config.get("source_max_side", 4096)) / max(image.size))
                size = (max(1, round(image.width*scale)), max(1, round(image.height*scale)))
            else:
                scale = min(width/image.width, height/image.height)
                size = (max(1, round(image.width*scale)), max(1, round(image.height*scale)))
            foreground, alpha = resize_rgba_float(image, size)
        pad_h, pad_w = max(0, height-alpha.shape[-2]), max(0, width-alpha.shape[-1])
        foreground = F.pad(foreground, (0, pad_w, 0, pad_h))
        alpha = F.pad(alpha, (0, pad_w, 0, pad_h))
        if rng.random() < self.flip_probability:
            foreground, alpha = foreground.flip(-1), alpha.flip(-1)
        if rng.random() < config.get("topology_chance", 25) / 100 * difficulty:
            foreground, alpha = add_topology(foreground, alpha, rng)
        full_h, full_w = alpha.shape[-2:]
        background, key_color, family = synthesize_background(full_h, full_w, config, rng)
        if _chance(config, "collision_chance", rng):
            points = torch.nonzero(alpha[0] > 0.98)
            if len(points):
                y, x = points[rng.randrange(len(points))].tolist()
                key_color_new = foreground[:, y, x].clone()
                background = (background + (key_color_new-key_color)[:, None, None]).clamp(0, 1)
                key_color, family = key_color_new, "collision"
        foreground = add_foreground_hard_negative(foreground, alpha, key_color, config, rng)
        spill = torch.zeros_like(alpha)
        contaminated = foreground
        if _chance(config, "spill_chance", rng):
            radius = rng.randint(int(config.get("spill_width_min", 1)), int(config.get("spill_width_max", 12)))
            eroded = erode_alpha(alpha, radius)
            band = torch.maximum(alpha-eroded, 4*alpha*(1-alpha))
            spatial = F.interpolate(torch.rand(1, 1, 8, 8), (full_h, full_w), mode="bilinear", align_corners=False)[0]
            low, high = config.get("spill_strength_min", 0.05), config.get("spill_strength_max", 0.65)
            spill = band * spatial * rng.uniform(min(low, high), max(low, high))
            # Spill corrupts F *before* alpha compositing; background mixing
            # and reflected screen light are distinct sources of contamination.
            chroma = key_color[:, None, None] - key_color.mean()
            contaminated = (foreground*(1-spill) + key_color[:, None, None]*spill + chroma*spill*0.2).clamp(0, 1)
        if rng.random() < config.get("linear_composite_percent", 70) / 100:
            composite = linear_to_srgb(srgb_to_linear(contaminated)*alpha + srgb_to_linear(background)*(1-alpha))
        else:
            composite = contaminated*alpha + background*(1-alpha)
        if rng.random() < config.get("jpeg_chance", 25) / 100 * difficulty:
            buffer = io.BytesIO()
            image = Image.fromarray((composite.permute(1, 2, 0).numpy()*255).clip(0, 255).astype(np.uint8))
            image.save(buffer, format="JPEG", quality=rng.randint(65, 98), subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                composite = torch.from_numpy(np.asarray(decoded).copy()).permute(2, 0, 1).float()/255
        holes = enclosed_background(alpha)
        context_rgb = F.interpolate(composite[None], (256, 256), mode="area")[0]
        context_alpha = F.interpolate(alpha[None], (64, 64), mode="area")[0]
        # Key target excludes the subject instead of assuming a clean border.
        bg_weight = 1-alpha
        target_key = (background*bg_weight).sum((1, 2)) / bg_weight.sum().clamp_min(1)
        key_valid = (bg_weight.sum() > 16).float()
        y0 = x0 = 0
        if detail:
            eroded = erode_alpha(alpha, 2)
            candidates = torch.nonzero(((alpha-eroded > 0.01) | (holes > 0))[0])
            if len(candidates):
                y, x = candidates[rng.randrange(len(candidates))].tolist()
                y0 = max(0, min(full_h-height, y-height//2))
                x0 = max(0, min(full_w-width, x-width//2))
        crop = (slice(None), slice(y0, y0+height), slice(x0, x0+width))
        return {"input": composite[crop].contiguous(), "foreground": foreground[crop].contiguous(),
                "alpha": alpha[crop].contiguous(), "valid": torch.ones_like(alpha[crop]),
                "spill": spill[crop].contiguous(), "holes": holes[crop].contiguous(),
                "key_color": target_key, "key_valid": key_valid, "family": family,
                "context_rgb": context_rgb, "context_alpha": context_alpha,
                "context_box": torch.tensor([x0/full_w, y0/full_h, (x0+width)/full_w, (y0+height)/full_h])}
