"""Aspect-preserving RGBA sampling and synthetic chromakey augmentation."""

from __future__ import annotations

import colorsys
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


def image_has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def collect_rgba_files(dataset_configs):
    files = []
    for config in dataset_configs:
        folder = Path(config["folder_path"])
        for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in {".png", ".webp"}:
                files.append(path)
    return files


def _bucket_id(width, height):
    ratio = max(width, 1) / max(height, 1)
    return int(round(math.log2(ratio) * 4.0))


def _image_size(path):
    """Read dimensions without asking Pillow to inspect/decode PNG pixels."""
    if path.suffix.lower() == ".png":
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
                return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
        except OSError:
            pass
    with Image.open(path) as image:
        return image.size


def _bucket_shape(resolution, bucket_id, max_long_side):
    ratio = 2.0 ** (bucket_id / 4.0)
    height = math.sqrt((resolution * resolution) / ratio)
    width = height * ratio
    scale = min(1.0, max_long_side / max(width, height))
    width = max(64, int(round(width * scale / 32.0)) * 32)
    height = max(64, int(round(height * scale / 32.0)) * 32)
    return height, width


class AspectBucketBatchSampler(Sampler):
    """Group similar aspect ratios while limiting pixels per GPU batch."""

    def __init__(self, files, resolutions, max_batch_size, megapixels_per_batch,
                 max_long_side, seed=0):
        self.files = list(files)
        self.resolutions = tuple(int(value) for value in resolutions)
        self.max_batch_size = int(max_batch_size)
        self.pixel_budget = max(1, int(float(megapixels_per_batch) * 1_000_000))
        self.max_long_side = int(max_long_side)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = defaultdict(list)
        for index, path in enumerate(self.files):
            self.groups[_bucket_id(*_image_size(path))].append(index)

    def __len__(self):
        return max(1, len(self.files))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        self.epoch += 1
        batches = []
        for bucket, indices in self.groups.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            cursor = 0
            while cursor < len(shuffled):
                resolution = rng.choice(self.resolutions)
                height, width = _bucket_shape(resolution, bucket, self.max_long_side)
                batch_size = max(1, min(
                    self.max_batch_size,
                    self.pixel_budget // max(1, height * width),
                ))
                chosen = shuffled[cursor:cursor + batch_size]
                cursor += len(chosen)
                seed = rng.randrange(2**31)
                batches.append([(index, height, width, seed + offset) for offset, index in enumerate(chosen)])
        rng.shuffle(batches)
        yield from batches


def _detail_crop(image, target_ratio, rng, scale_min=0.05, scale_max=0.30):
    source_w, source_h = image.size
    alpha_image = image.convert("RGBA").getchannel("A")
    preview_scale = min(1.0, 512.0 / max(source_w, source_h))
    if preview_scale < 1.0:
        alpha_image = alpha_image.resize(
            (max(1, round(source_w * preview_scale)), max(1, round(source_h * preview_scale))),
            Image.Resampling.BILINEAR,
        )
    alpha = np.asarray(alpha_image, dtype=np.uint8)
    edge = np.zeros_like(alpha, dtype=bool)
    edge[:, 1:] |= alpha[:, 1:] != alpha[:, :-1]
    edge[1:, :] |= alpha[1:, :] != alpha[:-1, :]
    points = np.argwhere(edge)
    if not len(points):
        return image
    center_y, center_x = points[rng.randrange(len(points))]
    center_y = int(round(center_y / preview_scale))
    center_x = int(round(center_x / preview_scale))
    area_scale = rng.uniform(min(scale_min, scale_max), max(scale_min, scale_max))
    crop_area = max(64.0, source_w * source_h * area_scale)
    crop_h = max(16, int(round(math.sqrt(crop_area / target_ratio))))
    crop_w = max(16, int(round(crop_h * target_ratio)))
    crop_w = min(crop_w, source_w)
    crop_h = min(crop_h, source_h)
    left = max(0, min(source_w - crop_w, int(center_x - crop_w * rng.uniform(0.3, 0.7))))
    top = max(0, min(source_h - crop_h, int(center_y - crop_h * rng.uniform(0.3, 0.7))))
    return image.crop((left, top, left + crop_w, top + crop_h))


def _resize_rgba_to_canvas(image, height, width, rng, detail_crop_percent=0,
                           detail_crop_scale_min=0.05, detail_crop_scale_max=0.30):
    if rng.random() * 100.0 < float(detail_crop_percent):
        image = _detail_crop(
            image, width / max(height, 1), rng,
            float(detail_crop_scale_min), float(detail_crop_scale_max),
        )
    source_w, source_h = image.size
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, min(width, int(round(source_w * scale))))
    resized_h = max(1, min(height, int(round(source_h * scale))))

    # Resize in Pillow's premultiplied-alpha mode before creating float32
    # tensors. Expanding every 4K source into a float tensor first multiplies
    # RAM traffic across DataLoader workers and is extremely slow on Windows.
    premultiplied_image = image.convert("RGBA").convert("RGBa").resize(
        (resized_w, resized_h), Image.Resampling.LANCZOS, reducing_gap=3.0,
    )
    resized = torch.from_numpy(
        np.asarray(premultiplied_image, dtype=np.uint8).copy()
    ).permute(2, 0, 1).float().div_(255.0)
    resized_alpha = resized[3:4]
    resized_premult = resized[:3]
    resized_rgb = resized_premult / resized_alpha.clamp_min(1.0 / 255.0)
    resized_rgb[:, resized_alpha[0] <= (1.0 / 255.0)] = 0.0
    top = rng.randint(0, height - resized_h) if height > resized_h else 0
    left = rng.randint(0, width - resized_w) if width > resized_w else 0
    rgb = torch.zeros((3, height, width), dtype=torch.float32)
    out_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    # The complete canvas has known supervision: everything outside the
    # resized source is synthetic background with target alpha == 0.  Keeping
    # it valid is important for learning clean frame borders and empty areas.
    valid = torch.ones((1, height, width), dtype=torch.float32)
    rgb[:, top:top + resized_h, left:left + resized_w] = resized_rgb
    out_alpha[:, top:top + resized_h, left:left + resized_w] = resized_alpha
    return rgb.clamp(0, 1), out_alpha.clamp(0, 1), valid


def _weighted_family(config, rng):
    families = ("green", "blue", "white", "black")
    weights = [max(0.0, float(config.get(f"{name}_chance", 0))) for name in families]
    if sum(weights) <= 0:
        weights = [1.0, 1.0, 1.0, 1.0]
    return rng.choices(families, weights=weights, k=1)[0]


def _background_color(config, rng):
    family = _weighted_family(config, rng)
    if family in {"green", "blue"}:
        hue_min = float(config.get(f"{family}_hue_min", 90 if family == "green" else 185))
        hue_max = float(config.get(f"{family}_hue_max", 155 if family == "green" else 255))
        saturation = rng.uniform(float(config.get("saturation_min", 0.65)), float(config.get("saturation_max", 1.0)))
        value = rng.uniform(float(config.get("value_min", 0.55)), float(config.get("value_max", 1.0)))
        return family, torch.tensor(colorsys.hsv_to_rgb(rng.uniform(hue_min, hue_max) / 360.0, saturation, value))
    low = float(config.get(f"{family}_value_min", 0.78 if family == "white" else 0.0))
    high = float(config.get(f"{family}_value_max", 1.0 if family == "white" else 0.2))
    value = rng.uniform(low, high)
    tint = torch.tensor([rng.uniform(-0.035, 0.035) for _ in range(3)])
    return family, (torch.full((3,), value) + tint).clamp(0, 1)


def _chance(config, key, rng):
    return rng.random() * 100.0 < float(config.get(key, 0))


def synthesize_background(height, width, config, rng):
    family, color = _background_color(config, rng)
    background = color[:, None, None].expand(3, height, width).clone()
    if _chance(config, "gradient_chance", rng):
        strength = float(config.get("gradient_strength", 0.15))
        yy = torch.linspace(-1, 1, height).view(1, height, 1)
        xx = torch.linspace(-1, 1, width).view(1, 1, width)
        direction = math.cos(rng.random() * math.tau) * xx + math.sin(rng.random() * math.tau) * yy
        background = background + direction * rng.uniform(-strength, strength)
    if _chance(config, "dirt_chance", rng):
        strength = float(config.get("dirt_strength", 0.12))
        small_h = max(2, height // rng.randint(48, 96))
        small_w = max(2, width // rng.randint(48, 96))
        dirt = torch.randn((1, 1, small_h, small_w))
        dirt = F.interpolate(dirt, (height, width), mode="bicubic", align_corners=False)[0]
        tint = torch.randn((3, 1, 1)) * 0.5 + 0.5
        background = background + dirt * tint * strength
    if _chance(config, "noise_chance", rng):
        background = background + torch.randn_like(background) * float(config.get("noise_strength", 0.025))
    return background.clamp(0, 1), color, family


def add_spill(composite, foreground, alpha, key_color, config, rng):
    spill = torch.zeros_like(alpha)
    if not _chance(config, "spill_chance", rng):
        return composite, spill
    width_min = int(config.get("spill_width_min", 1))
    width_max = int(config.get("spill_width_max", 12))
    width = max(1, rng.randint(min(width_min, width_max), max(width_min, width_max)))
    kernel = width * 2 + 1
    eroded = -F.max_pool2d(-alpha[None], kernel, stride=1, padding=width)[0]
    inner_edge = (alpha - eroded).clamp(0, 1)
    # Very thin hair may consist almost entirely of fractional-alpha pixels.
    # Include that transition explicitly instead of training despill mostly on
    # the opaque side of broad silhouettes.
    transition = (4.0 * alpha * (1.0 - alpha)).clamp(0, 1)
    spill_band = torch.maximum(inner_edge, transition)
    low = float(config.get("spill_strength_min", 0.05))
    high = float(config.get("spill_strength_max", 0.65))
    spatial = torch.rand((1, max(2, alpha.shape[-2] // 64), max(2, alpha.shape[-1] // 64)))
    spatial = F.interpolate(spatial[None], alpha.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
    spill = spill_band * (0.35 + 0.65 * spatial) * rng.uniform(min(low, high), max(low, high))
    key = key_color[:, None, None]
    blended = composite * (1.0 - spill) + key * spill
    additive = (key - key.mean()) * spill * rng.uniform(0.15, 0.6)
    return (blended + additive).clamp(0, 1), spill


def add_foreground_hard_negative(foreground, alpha, key_color, config, rng):
    if not _chance(config, "hard_negative_chance", rng):
        return foreground
    strength = float(config.get("hard_negative_strength", 0.65))
    small_h = max(2, alpha.shape[-2] // 96)
    small_w = max(2, alpha.shape[-1] // 96)
    region = torch.rand((1, 1, small_h, small_w))
    region = F.interpolate(region, alpha.shape[-2:], mode="bicubic", align_corners=False)[0]
    region = ((region - 0.45) * 8.0).sigmoid() * alpha
    amount = region * rng.uniform(strength * 0.4, strength)
    key = key_color[:, None, None]
    return (foreground * (1.0 - amount) + key * amount).clamp(0, 1)


class ChromaKeyDataset(Dataset):
    def __init__(self, files, augmentation, flip_probability=0.5):
        self.files = list(files)
        self.augmentation = dict(augmentation)
        self.flip_probability = float(flip_probability)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, request):
        index, height, width, seed = request
        rng = random.Random(int(seed))
        with Image.open(self.files[index]) as image:
            foreground, alpha, valid = _resize_rgba_to_canvas(
                image, height, width, rng,
                detail_crop_percent=self.augmentation.get("detail_crop_percent", 70),
                detail_crop_scale_min=self.augmentation.get("detail_crop_scale_min", 0.05),
                detail_crop_scale_max=self.augmentation.get("detail_crop_scale_max", 0.30),
            )
        if rng.random() < self.flip_probability:
            foreground = foreground.flip(-1)
            alpha = alpha.flip(-1)
            valid = valid.flip(-1)
        background, key_color, family = synthesize_background(height, width, self.augmentation, rng)
        # Adversarial key collision: sometimes choose the screen colour from
        # the actual character.  The chroma prior will be wrong locally and
        # the semantic/detail branches must learn to preserve green/blue hair,
        # clothing and highlights instead of blindly deleting matching RGB.
        if _chance(self.augmentation, "collision_chance", rng):
            opaque_points = torch.nonzero(alpha[0] > 0.95, as_tuple=False)
            if len(opaque_points):
                point = opaque_points[rng.randrange(len(opaque_points))]
                collision = foreground[:, point[0], point[1]].clone()
                background = (background + (collision - key_color)[:, None, None]).clamp(0, 1)
                key_color = collision
                family = "collision"
        foreground = add_foreground_hard_negative(
            foreground, alpha, key_color, self.augmentation, rng
        )
        # Mixing in both conventions covers physically linear and common sRGB compositors.
        linear_probability = float(self.augmentation.get("linear_composite_percent", 70)) / 100.0
        if rng.random() < linear_probability:
            fg_linear = foreground.clamp_min(0).pow(2.2)
            bg_linear = background.clamp_min(0).pow(2.2)
            composite = (fg_linear * alpha + bg_linear * (1.0 - alpha)).clamp_min(0).pow(1.0 / 2.2)
        else:
            composite = foreground * alpha + background * (1.0 - alpha)
        composite, spill = add_spill(composite, foreground, alpha, key_color, self.augmentation, rng)
        return {
            "input": composite,
            "foreground": foreground,
            "alpha": alpha,
            "valid": valid,
            "spill": spill,
            "key_color": key_color,
            "family": family,
        }
