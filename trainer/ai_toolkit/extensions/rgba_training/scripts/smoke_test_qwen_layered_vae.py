from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image
from torchvision.transforms import functional as TF

from extensions.rgba_training.qwen_compat import validate_qwen_rgba_vae_config
from toolkit.rgba_utils import prepare_rgba_image


def main():
    parser = argparse.ArgumentParser(description="Run an RGBA reconstruction through Qwen-Image-Layered VAE")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen-Image-Layered")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("qwen_layered_vae_reconstruction.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.model,
        subfolder=args.subfolder or None,
        torch_dtype=dtype,
    ).to(device).eval()
    validate_qwen_rgba_vae_config(vae.config)

    with Image.open(args.image) as source:
        rgba = prepare_rgba_image(source)
    tensor = TF.to_tensor(rgba).mul(2.0).sub(1.0).unsqueeze(0).unsqueeze(2).to(device, dtype)

    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.mode()
        decoded = vae.decode(latent).sample.squeeze(2).float().cpu().clamp(-1.0, 1.0)
    output = decoded[0].add(1.0).mul(127.5).round().byte().permute(1, 2, 0).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(output, dtype=np.uint8), "RGBA").save(args.output)
    print(f"Saved RGBA reconstruction to {args.output}")


if __name__ == "__main__":
    main()

