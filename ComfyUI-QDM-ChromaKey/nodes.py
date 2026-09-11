"""ComfyUI node for QDM AnimeKeyMatte safetensors models."""

from __future__ import annotations

from pathlib import Path

import torch
import folder_paths
import comfy.model_management as model_management

from .qdm_chromakey_core import load_model, run_tiled


MODEL_TYPE = "qdm_chromakey"
MODEL_DIR = Path(folder_paths.models_dir) / MODEL_TYPE
MODEL_DIR.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path(MODEL_TYPE, str(MODEL_DIR))


class QDMAnimeChromaKey:
    def __init__(self):
        self._cache_key = None
        self._model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list(MODEL_TYPE),),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 96, "min": 32, "max": 512, "step": 16}),
                "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05}),
            },
            "optional": {"inference_device": (["auto", "cpu"], {"default": "auto"})},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "alpha_mask")
    FUNCTION = "apply"
    CATEGORY = "image/matting"
    DESCRIPTION = "Compact alpha + foreground colour matting. V4 shares global context across native-detail tiles; overlap is used only by legacy models."

    def _load(self, model_name, device):
        path = folder_paths.get_full_path_or_raise(MODEL_TYPE, model_name)
        stat = Path(path).stat()
        key = (path, stat.st_mtime_ns, stat.st_size, str(device))
        if self._cache_key != key:
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            self._model, _metadata = load_model(path, device=device, dtype=dtype)
            self._cache_key = key
        return self._model

    def apply(self, image, model_name, tile_size, overlap, despill_strength, inference_device="auto"):
        device = torch.device("cpu") if inference_device == "cpu" else model_management.get_torch_device()
        intermediate = model_management.intermediate_device()
        model = self._load(model_name, device)
        outputs = []
        masks = []
        for source in image:
            rgb = source[..., :3].permute(2, 0, 1).float()
            clean, alpha = run_tiled(
                model,
                rgb,
                tile_size=tile_size,
                overlap=overlap,
                autocast_dtype=torch.float16 if device.type == "cuda" else None,
            )
            strength = float(despill_strength)
            clean = (rgb + (clean - rgb) * strength).clamp(0.0, 1.0)
            hidden = alpha <= 0.0
            clean = clean.masked_fill(hidden.expand_as(clean), 0.0)
            rgba = torch.cat((clean, alpha.clamp(0.0, 1.0)), dim=0)
            outputs.append(rgba.permute(1, 2, 0).to(intermediate))
            masks.append(alpha[0].to(intermediate))
        return torch.stack(outputs), torch.stack(masks)


NODE_CLASS_MAPPINGS = {"QDMAnimeChromaKey": QDMAnimeChromaKey}
NODE_DISPLAY_NAME_MAPPINGS = {"QDMAnimeChromaKey": "QDM Anime Chroma Key"}
