"""ComfyUI adapter for the independent CleanMatte architecture."""
from pathlib import Path
import torch
import folder_paths
from .cleanmatte import load_model, predict, recover_foreground

folder_paths.add_model_folder_path("cleanmatte", str(Path(folder_paths.models_dir)/"cleanmatte"))


class QDMCleanMatteLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint": (folder_paths.get_filename_list("cleanmatte"),),
                             "device": (["cpu", "cuda"],)}}

    RETURN_TYPES = ("QDM_CLEANMATTE",)
    FUNCTION = "load"
    CATEGORY = "QDM/CleanMatte"

    def load(self, checkpoint, device):
        path = folder_paths.get_full_path("cleanmatte", checkpoint)
        if path is None:
            raise FileNotFoundError(checkpoint)
        return (load_model(path, device),)


class QDMCleanMatteApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("QDM_CLEANMATTE",), "image": ("IMAGE",),
                             "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                             "despill": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("foreground_rgb", "alpha", "rgba")
    FUNCTION = "apply"
    CATEGORY = "QDM/CleanMatte"

    @torch.inference_mode()
    def apply(self, model, image, tile_size, despill):
        device = next(model.parameters()).device
        foregrounds, alphas = [], []
        for frame in image:
            rgb = frame[..., :3].permute(2, 0, 1)[None].to(device)
            alpha = predict(model, rgb, tile_size)
            foreground = recover_foreground(rgb, alpha, strength=despill)
            foregrounds.append(foreground[0].permute(1, 2, 0).cpu())
            alphas.append(alpha[0, 0].cpu())
        foreground, alpha = torch.stack(foregrounds), torch.stack(alphas)
        return foreground, alpha, torch.cat((foreground, alpha[..., None]), -1)


NODE_CLASS_MAPPINGS = {"QDMCleanMatteLoader": QDMCleanMatteLoader, "QDMCleanMatteApply": QDMCleanMatteApply}
NODE_DISPLAY_NAME_MAPPINGS = {"QDMCleanMatteLoader": "QDM Load CleanMatte", "QDMCleanMatteApply": "QDM CleanMatte (alpha first)"}
