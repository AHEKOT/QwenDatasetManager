from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image

from toolkit.rgba_utils import prepare_rgba_image, resize_rgba_alpha_safe


ROOT = Path(r"D:\AiToolkitNew\AI-Toolkit")
SOURCE = ROOT / "datasets" / "ChromaKeyX_New_img" / "image_00001.png"
OUTPUT = ROOT / "output" / "rgba_vae_roundtrip"


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor[0].permute(1, 2, 0).float().cpu().numpy()
    array = np.clip(np.rint((array + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGBA")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image-Layered",
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    vae.eval().requires_grad_(False)
    vae.enable_tiling()

    prepared = prepare_rgba_image(
        Image.open(SOURCE),
        alpha_threshold=1.0 / 255.0,
        hidden_rgb_color=(0, 0, 0),
        edge_color_correction="matte_despill",
        edge_matte_color=(0, 255, 0),
        edge_width=3,
    )
    for resolution in (256, 640, 1024):
        image = resize_rgba_alpha_safe(prepared, (resolution, resolution))
        source = to_tensor(image).to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            latent = vae.encode(source.unsqueeze(2)).latent_dist.mode()
            decoded = vae.decode(latent, return_dict=False)[0][:, :, 0]

        source_f = source.float()
        decoded_f = decoded.float()
        alpha_mae = (source_f[:, 3:4] - decoded_f[:, 3:4]).abs().mean().item()
        visible = ((source_f[:, 3:4] + 1.0) * 0.5).clamp(0, 1)
        rgb_mae = (
            (source_f[:, :3] - decoded_f[:, :3]).abs() * visible
        ).sum().div(visible.sum() * 3.0 + 1e-8).item()
        finite = bool(torch.isfinite(latent).all() and torch.isfinite(decoded).all())
        print(
            f"resolution={resolution} finite={finite} "
            f"latent_range=({latent.float().min().item():.5f},"
            f"{latent.float().max().item():.5f}) "
            f"visible_rgb_mae={rgb_mae:.6f} alpha_mae={alpha_mae:.6f}"
        )
        image.save(OUTPUT / f"source_{resolution}.png")
        to_image(decoded).save(OUTPUT / f"reconstruction_{resolution}.png")


if __name__ == "__main__":
    main()
