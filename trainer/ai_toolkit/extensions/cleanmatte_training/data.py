"""Clean target coverage, independent RGB corruption, source-disjoint holdout."""
import colorsys
import hashlib
import io
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F


def discover(datasets):
    paths = []
    for dataset in datasets:
        root = Path(dataset["folder_path"])
        iterator = root.rglob("*") if dataset.get("recursive", False) else root.glob("*")
        paths.extend(p.resolve() for p in iterator if p.suffix.lower() in {".png", ".webp"})
    return sorted(set(paths))


def inventory(paths, seed=42):
    """Group exact pixel duplicates before splitting; never silently fix labels."""
    groups, rejected, audit = {}, [], []
    for path in paths:
        with Image.open(path) as source:
            if "A" not in source.getbands() and "transparency" not in source.info:
                rejected.append(str(path))
                continue
            rgba = np.asarray(source.convert("RGBA"))
        a = rgba[..., 3]
        if a.min() == a.max():
            rejected.append(str(path))
            continue
        digest = hashlib.sha256(str(rgba.shape).encode() + rgba.tobytes()).hexdigest()
        groups.setdefault(digest, []).append(str(path))
        audit.append({"path": str(path), "sha256_pixels": digest,
                      "size": [int(rgba.shape[1]), int(rgba.shape[0])],
                      "background_fraction": float((a == 0).mean()),
                      "opaque_fraction": float((a == 255).mean()),
                      "mixed_fraction": float(((a > 0) & (a < 255)).mean())})
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    if len(keys) < 2:
        raise ValueError("CleanMatte needs at least two distinct RGBA sources with meaningful alpha.")
    nval = min(len(keys)-1, max(1, math.ceil(len(keys) * 0.1)))
    val = [groups[k][0] for k in keys[:nval]]
    train = [groups[k][0] for k in keys[nval:]]
    return {"train": train, "validation": val, "rejected": rejected, "sources": audit,
            "duplicates": {k: v for k, v in groups.items() if len(v) > 1},
            "note": "Statistics are not a label-quality certificate. Inspect clean alpha independently."}


@lru_cache(maxsize=2)
def read_rgba(path):
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def tensor_rgba(image):
    rgba = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return rgba[:3], rgba[3:]


def source_patch(path, size, rng, native=True):
    source = read_rgba(str(path))
    if native:
        a = np.asarray(source)[..., 3]
        edge = (a > 0) & (a < 255)
        edge[1:] |= a[1:] != a[:-1]
        edge[:, 1:] |= a[:, 1:] != a[:, :-1]
        locations = np.flatnonzero(edge)
        if len(locations) and rng.random() < 0.85:
            y, x = divmod(int(locations[rng.randrange(len(locations))]), source.width)
        else:
            x, y = rng.randrange(source.width), rng.randrange(source.height)
        left = x - size//2 + rng.randint(-size//4, size//4)
        top = y - size//2 + rng.randint(-size//4, size//4)
        image = source.crop((left, top, left+size, top+size))
    else:
        # Resize premultiplied colours, then unpremultiply. Transparent RGB
        # padding must not produce a black fringe in the supervised composite.
        scale = min(size/source.width, size/source.height)
        dims = (max(1, round(source.width*scale)), max(1, round(source.height*scale)))
        small = source.convert("RGBa").resize(dims, Image.Resampling.BOX).convert("RGBA")
        image = Image.new("RGBA", (size, size))
        image.paste(small, ((size-dims[0])//2, (size-dims[1])//2))
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    return tensor_rgba(image)


def analytic_patch(size, rng):
    """Coverage at 4x sampling: smooth body, real gaps and attached thin hairs.

    No pixel noise, random erosion or alpha speckle is applied to targets.
    """
    ss = 4
    mask = Image.new("L", (size*ss, size*ss))
    draw = ImageDraw.Draw(mask)
    cx, cy = size*rng.uniform(0.35, 0.65), size*rng.uniform(0.42, 0.62)
    rx, ry = size*rng.uniform(0.16, 0.26), size*rng.uniform(0.2, 0.34)
    draw.ellipse(tuple(int(v*ss) for v in (cx-rx, cy-ry, cx+rx, cy+ry)), fill=255)
    for _ in range(rng.randint(12, 30)):
        angle = rng.uniform(-math.pi, math.pi)
        start = np.array([cx+rx*0.9*math.cos(angle), cy+ry*0.9*math.sin(angle)])
        direction = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        length, bend = size*rng.uniform(0.08, 0.3), size*rng.uniform(-0.12, 0.12)
        points = [tuple((start + direction*length*t + normal*bend*t*t)*ss)
                  for t in np.linspace(0, 1, 30)]
        draw.line(points, fill=255, width=rng.choice([1, 2, 4, 6, 8]), joint="curve")
    for _ in range(rng.randint(1, 4)):
        x, y = cx+rng.uniform(-rx/2, rx/2), cy+rng.uniform(-ry/2, ry/2)
        r = rng.uniform(0.5, max(1, size*0.04))
        draw.ellipse(tuple(int(v*ss) for v in (x-r, y-r, x+r, y+r)), fill=0)
    alpha = torch.from_numpy(np.asarray(mask.resize((size, size), Image.Resampling.BOX)).copy()).float()[None]/255
    colour = torch.tensor([rng.uniform(0.05, 0.95) for _ in range(3)])[:, None, None]
    ramp = torch.linspace(-0.08, 0.08, size)[None, None, :]
    return (colour+ramp).expand(3, size, size).clamp(0, 1).clone(), alpha


def background(size, rng, config):
    choices = ["green", "blue", "white", "black"]
    weights = [config.get(f"{name}_chance", default) for name, default in zip(choices, [70, 30, 0, 0])]
    kind = rng.choices(choices, weights=weights)[0]
    if kind in {"green", "blue"}:
        bounds = (80, 160) if kind == "green" else (180, 265)
        hue = rng.uniform(*bounds)/360
        colour = colorsys.hsv_to_rgb(hue, rng.uniform(0.65, 1), rng.uniform(0.55, 1))
    else:
        value = rng.uniform(0.8, 1) if kind == "white" else rng.uniform(0, 0.2)
        colour = (value, value, value)
    return torch.tensor(colour)[:, None, None].expand(3, size, size).clone()


def composite(foreground, alpha, rng, config, difficulty=0.0):
    """All corruption is on RGB. The caller's clean alpha is never mutated."""
    size = alpha.shape[-1]
    bg = background(size, rng, config)
    if difficulty > 0:
        ramp = torch.linspace(-1, 1, size)[None, None, :]
        bg = (bg + ramp*rng.uniform(-0.12, 0.12)*difficulty).clamp(0, 1)
    fg = foreground
    if rng.random() < config.get("spill_chance", 35)/100*difficulty:
        eroded = -F.max_pool2d(-alpha[None], 7, 1, 3)[0]
        edge = (alpha-eroded).clamp(0, 1)
        fg = torch.lerp(fg, bg, edge*rng.uniform(0.02, 0.25)*difficulty)
    rgb = alpha*fg + (1-alpha)*bg
    if difficulty > 0:
        generator = torch.Generator().manual_seed(rng.randrange(2**31))
        noise = torch.randn(rgb.shape, generator=generator)*config.get("noise_strength", 0.01)*difficulty
        rgb = (rgb+noise).clamp(0, 1)
        if rng.random() < config.get("jpeg_chance", 20)/100*difficulty:
            image = Image.fromarray((rgb.permute(1, 2, 0).numpy()*255).round().astype(np.uint8))
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=rng.randint(75, 97), subsampling=2)
            stream.seek(0)
            with Image.open(stream) as decoded:
                rgb = torch.from_numpy(np.asarray(decoded).copy()).permute(2, 0, 1).float()/255
    return rgb


def make_batch(paths, size, batch_size, seed, config, difficulty=0.0, paired=False,
               analytic_percent=20, detail_percent=70):
    rng = random.Random(seed)
    images, alphas, partners = [], [], []
    for _ in range(batch_size):
        if rng.random() < analytic_percent/100:
            fg, alpha = analytic_patch(size, rng)
        else:
            fg, alpha = source_patch(rng.choice(paths), size, rng, rng.random() < detail_percent/100)
        images.append(composite(fg, alpha, rng, config, difficulty))
        alphas.append(alpha)
        if paired:
            partners.append(composite(fg, alpha, rng, config, difficulty))
    return {"rgb": torch.stack(images), "alpha": torch.stack(alphas),
            "partner": torch.stack(partners) if paired else None}
