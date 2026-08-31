from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from toolkit.accelerator import unwrap_model
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.memory_management import MemoryManager


def validate_sampling_lora_path(path: str) -> str:
    """Validate a local sampling-only LoRA without materializing its tensors."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Sampling LoRA does not exist: {resolved}")
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError("Sampling LoRA must be a .safetensors file")

    with safe_open(str(resolved), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
    has_peft = any(key.endswith(".lora_A.weight") for key in keys) and any(
        key.endswith(".lora_B.weight") for key in keys
    )
    has_native = any(key.endswith(".lora_down.weight") for key in keys) and any(
        key.endswith(".lora_up.weight") for key in keys
    )
    if not (has_peft or has_native):
        raise ValueError(
            "Sampling LoRA must contain matching lora_A/lora_B or "
            "lora_down/lora_up tensors"
        )
    return str(resolved)


def _converted_key_map(base_model, keys: list[str]) -> dict[str, str]:
    marker = object()
    converted = base_model.convert_lora_weights_before_load(
        OrderedDict((key, marker) for key in keys)
    )
    return {source: target for source, target in zip(keys, converted.keys())}


def _sampling_lora_metadata(base_model, path: str):
    with safe_open(path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        converted_keys = _converted_key_map(base_model, keys)
        modules_dim: dict[str, int] = {}
        modules_alpha: dict[str, float] = {}
        qwen_native_layout = False

        for source_key in keys:
            converted_key = converted_keys[source_key]
            if converted_key.endswith(".lora_A.weight"):
                module_path = converted_key.removesuffix(".lora_A.weight")
                rank = int(weights.get_slice(source_key).get_shape()[0])
            elif converted_key.endswith(".lora_down.weight"):
                module_path = converted_key.removesuffix(".lora_down.weight")
                rank = int(weights.get_slice(source_key).get_shape()[0])
                qwen_native_layout = qwen_native_layout or module_path.startswith(
                    "transformer_blocks."
                )
            else:
                continue

            if module_path.startswith("transformer_blocks."):
                module_path = "transformer." + module_path
            native_name = module_path.replace(".", "$$")
            modules_dim[native_name] = rank

            alpha_key = source_key.rsplit(".", 2)[0] + ".alpha"
            if alpha_key in keys:
                alpha = float(weights.get_tensor(alpha_key).float().item())
            else:
                alpha = float(rank)
            modules_alpha[native_name] = alpha

    if not modules_dim:
        raise ValueError(f"Sampling LoRA contains no transformer modules: {path}")
    return modules_dim, modules_alpha, qwen_native_layout


def build_sampling_lora_network(
    *,
    base_model,
    transformer: torch.nn.Module,
    lora_path: str,
    device: torch.device,
    use_layer_offloading: bool,
) -> LoRASpecialNetwork:
    """Attach an inference adapter through AI Toolkit's native LoRA hooks.

    Both AI Toolkit PEFT-style checkpoints (used by FLUX.2 Klein) and the
    QIE2511 Lightning lora_down/lora_up layout are accepted. The adapter is
    inactive outside the sampling context and never participates in training.
    """
    modules_dim, modules_alpha, qwen_native_layout = _sampling_lora_metadata(
        base_model, lora_path
    )
    first_name = next(iter(modules_dim))
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(base_model.torch_dtype)
    try:
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            multiplier=1.0,
            lora_dim=modules_dim[first_name],
            alpha=modules_alpha[first_name],
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            train_unet=True,
            train_text_encoder=False,
            transformer_only=False,
            target_lin_modules=list(base_model.target_lora_modules),
            is_transformer=True,
            base_model=base_model,
            initialize_weights=False,
        )
    finally:
        torch.set_default_dtype(original_default_dtype)

    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    if qwen_native_layout:
        raw_state = load_file(lora_path, device="cpu")
        load_state = OrderedDict()
        for key, value in raw_state.items():
            if key.startswith("transformer_blocks."):
                key = "transformer." + key
            load_state[key] = value
        del raw_state
        extra = network.load_weights(load_state)
        del load_state
    else:
        extra = network.load_weights(lora_path)
    if extra:
        preview = ", ".join(list(extra.keys())[:5])
        raise ValueError(f"Sampling LoRA has unmatched tensors: {preview}")

    network.requires_grad_(False)
    network.eval()
    network.is_active = False
    network.can_merge_in = False
    if use_layer_offloading:
        MemoryManager.attach(network, device)
    else:
        network.force_to(device, dtype=base_model.torch_dtype)
    network._update_torch_multiplier()
    return network


class SamplingLoRAMixin:
    """Load a LoRA once and activate it only around preview generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configured = getattr(self.model_config, "sample_lora_path", None)
        self.sample_lora_path = (
            validate_sampling_lora_path(configured) if configured else None
        )
        self._sample_lora_network = None

    def _ensure_sampling_lora_network(self) -> LoRASpecialNetwork:
        if self._sample_lora_network is not None:
            return self._sample_lora_network
        started = time.perf_counter()
        self.print_and_status_update("Attaching sampling-only Turbo LoRA")
        self._sample_lora_network = build_sampling_lora_network(
            base_model=self,
            transformer=unwrap_model(self.transformer),
            lora_path=self.sample_lora_path,
            device=self.device_torch,
            use_layer_offloading=bool(self.model_config.layer_offloading),
        )
        elapsed = time.perf_counter() - started
        self.print_and_status_update(
            f"Sampling-only Turbo LoRA ready: "
            f"{len(self._sample_lora_network.unet_loras)} modules in {elapsed:.1f}s"
        )
        return self._sample_lora_network

    def generate_images(self, image_configs, sampler=None, pipeline=None):
        if self.sample_lora_path is None:
            return super().generate_images(
                image_configs, sampler=sampler, pipeline=pipeline
            )
        sampling_network = self._ensure_sampling_lora_network()
        with sampling_network:
            return super().generate_images(
                image_configs, sampler=sampler, pipeline=pipeline
            )
