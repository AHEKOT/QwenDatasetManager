from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import QwenImageTransformer2DModel

from extensions.rgba_training.qwen_image_edit_plus_rgba import (
    QwenImageEditPlusRGBAModel,
    build_qwen_sampling_lora_network,
    validate_qwen_sampling_lora,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate native Qwen sampling-LoRA attachment without loading base weights"
    )
    parser.add_argument("--transformer-config", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--layer-offloading", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.transformer_config).resolve()
    lora_path = validate_qwen_sampling_lora(args.lora)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    started = time.perf_counter()
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(config)

    base_model = object.__new__(QwenImageEditPlusRGBAModel)
    base_model.target_lora_modules = ["QwenImageTransformer2DModel"]
    base_model.torch_dtype = torch.bfloat16
    base_model.use_old_lokr_format = False
    network = build_qwen_sampling_lora_network(
        base_model=base_model,
        transformer=transformer,
        lora_path=lora_path,
        device=torch.device("cuda" if args.layer_offloading else "cpu"),
        use_layer_offloading=args.layer_offloading,
    )

    if len(network.unet_loras) != 720:
        raise RuntimeError(f"Expected 720 Qwen LoRA modules, got {len(network.unet_loras)}")
    if network.is_active:
        raise RuntimeError("Sampling LoRA must be inactive outside sample generation")
    alpha = float(network.unet_loras[0].alpha.float().item())
    rank = int(network.unet_loras[0].lora_dim)
    if (rank, alpha) != (64, 8.0):
        raise RuntimeError(f"Lightning rank/alpha was changed: rank={rank}, alpha={alpha}")
    if not all(not parameter.requires_grad for parameter in network.parameters()):
        raise RuntimeError("Sampling-only LoRA unexpectedly contains trainable parameters")

    elapsed = time.perf_counter() - started
    size_mb = sum(parameter.numel() * parameter.element_size() for parameter in network.parameters()) / 2**20
    print(
        f"PASS: {len(network.unet_loras)} modules, rank={rank}, alpha={alpha:g}, "
        f"{size_mb:.1f} MiB, offloading={args.layer_offloading}, "
        f"attached and loaded in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
