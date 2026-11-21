"""PSD layer contract v1: visible, normal RGBA canvases in bottom-to-top order.

No model imports: the reader and writer can also be used by dataset tooling.
Unsupported Photoshop compositing semantics fail instead of silently flattening.
"""
import hashlib
import struct
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_VERSION = 1


def psd_size(path):
    with open(path, "rb") as stream:
        header = stream.read(26)
    if len(header) != 26:
        raise ValueError(f"{path}: truncated PSD header")
    signature, version, reserved, channels, height, width, depth, mode = struct.unpack(
        ">4sH6sHIIHH", header
    )
    if signature != b"8BPS" or version != 1 or reserved != b"\0" * 6:
        raise ValueError(f"{path}: expected a PSD file (PSB is not supported)")
    if depth != 8 or mode != 3 or not 3 <= channels <= 56 or not 1 <= width <= 30000 or not 1 <= height <= 30000:
        raise ValueError(f"{path}: layered targets require an 8-bit RGB PSD")
    return width, height


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _psd_api():
    try:
        from psd_tools import PSDImage
        from psd_tools.constants import BlendMode
    except ImportError as exc:
        raise RuntimeError("PSD training requires psd-tools[composite]==1.22.0 in the trainer environment") from exc
    return PSDImage, BlendMode


def composite_layers(layers):
    if not layers:
        raise ValueError("Cannot composite an empty layer stack")
    result = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
    for layer in layers:
        if layer.size != result.size or layer.mode != "RGBA":
            raise ValueError("Every layer must use the same RGBA canvas")
        result = Image.alpha_composite(result, layer)
    return result


def read_psd_layers(path, layer_slots):
    if isinstance(layer_slots, bool) or not isinstance(layer_slots, int) or not 1 <= layer_slots <= 20:
        raise ValueError("layer_slots must be an integer from 1 to 20")
    size = psd_size(path)
    PSDImage, BlendMode = _psd_api()
    from psd_tools.constants import Tag
    from psd_tools.compression import PSDDecompressionWarning
    psd = PSDImage.open(path)
    selected = []

    def fail(layer, reason):
        raise ValueError(f"{path}: layer '{layer.name}': {reason}. Rasterize this feature into a normal RGBA layer first.")

    def visit(group, prefix=""):
        # psd-tools iterates bottom -> top, including children of a group.
        for layer in group:
            if not layer.is_visible():
                continue
            if layer.clipping or layer.has_clip_layers(visible=True):
                fail(layer, "clipping layers are not independent RGBA canvases")
            if layer.has_effects():
                fail(layer, "live layer effects are unsupported")
            if layer.tagged_blocks.get_data(Tag.KNOCKOUT_SETTING, 0):
                fail(layer, "knockout compositing is unsupported")
            if layer.tagged_blocks.get_data(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING, []):
                fail(layer, "channel blending restrictions are unsupported")
            # Check groups as well as leaves: Blend If depends on the backdrop.
            ranges = layer._record.blending_ranges
            default_range = ((0, 65535), (0, 65535))
            if ranges and any(tuple(tuple(v) for v in entry) != default_range
                              for entry in [ranges.composite_ranges, *(ranges.channel_ranges or [])] if entry):
                fail(layer, "Blend If ranges are unsupported")
            if layer.is_group():
                if layer.kind != "group" or layer.blend_mode not in (BlendMode.NORMAL, BlendMode.PASS_THROUGH):
                    fail(layer, "unsupported group compositing")
                if layer.opacity != 255 or layer.fill_opacity != 255 or layer.has_mask() or layer.has_vector_mask():
                    fail(layer, "group opacity or masks affect overlapping children")
                visit(layer, prefix + layer.name + "/")
                continue
            if layer.blend_mode != BlendMode.NORMAL:
                fail(layer, "only Normal blend mode is supported")
            if layer.kind not in ("pixel", "type", "smartobject"):
                fail(layer, f"unsupported layer kind {layer.kind}")
            if not layer.has_pixels() and layer.width * layer.height:
                fail(layer, "no saved raster pixels")
            selected.append((prefix + layer.name, layer))

    visit(psd)
    if not selected:
        raise ValueError(f"{path}: no visible layers")
    if len(selected) > layer_slots:
        raise ValueError(f"{path}: {len(selected)} visible layers exceed layer_slots={layer_slots}; no layers were discarded")
    images, names = [], []
    for name, layer in selected:
        # Layer.composite applies its opacity, fill opacity and masks; viewport
        # places cropped/offset layer pixels on the document canvas.
        if layer.width == 0 or layer.height == 0:
            image = Image.new("RGBA", size, (0, 0, 0, 0))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PSDDecompressionWarning)
                image = layer.composite(viewport=(0, 0, *size), color=0.0, alpha=0.0)
            if image is None:
                fail(layer, "could not render layer")
            image = image.convert("RGBA")
        pixels = np.array(image)
        pixels[pixels[:, :, 3] == 0, :3] = 0
        images.append(Image.fromarray(pixels))
        names.append(name)
    # Padding has a learned meaning (empty top layer); it is not loss-masked.
    while len(images) < layer_slots:
        images.append(Image.new("RGBA", size, (0, 0, 0, 0)))
        names.append("<empty>")
    return images, names


def save_layered_psd(path, layers):
    PSDImage, _ = _psd_api()
    composite_layers(layers)  # validate mode, size and nonempty stack
    psd = PSDImage.new("RGB", layers[0].size, color=0)
    for index, image in enumerate(layers):
        psd.create_pixel_layer(image, name=f"Layer {index + 1:02d}", top=0, left=0)
    psd.save(Path(path))


def load_psd_target(item, transform, only_load_latents=False):
    import torch
    layers, item.psd_layer_names = read_psd_layers(item.path, item.dataset_config.layer_slots)
    tensors = []
    for image in layers:
        # PIL RGBA resizing uses premultiplied alpha internally.
        image = image.resize((item.scale_to_width, item.scale_to_height), Image.Resampling.BICUBIC)
        image = image.crop((item.crop_x, item.crop_y,
                            item.crop_x + item.crop_width, item.crop_y + item.crop_height))
        pixels = np.array(image)
        pixels[pixels[:, :, 3] == 0, :3] = 0
        tensors.append(transform(Image.fromarray(pixels)))
    item.tensor = torch.cat(tensors, dim=1)
    if not only_load_latents and item.has_control_image:
        item.load_control_image()
