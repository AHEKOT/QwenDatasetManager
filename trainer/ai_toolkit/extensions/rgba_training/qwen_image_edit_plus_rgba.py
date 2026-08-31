from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from safetensors import safe_open
from safetensors.torch import load_file

from extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus import (
    QwenImageEditPlusModel,
)
from extensions_built_in.diffusion_models.qwen_image.qwen_image_pipelines import (
    QwenImageEditPlusCustomPipeline,
)
from toolkit.accelerator import unwrap_model
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.memory_management import MemoryManager
from toolkit.rgba_utils import ensure_normalized_rgba_tensor

from .qwen_compat import (
    validate_qie2511_transformer_config,
    validate_qwen_rgba_vae_config,
)


def validate_qwen_sampling_lora(path: str) -> str:
    """Validate a local Qwen transformer LoRA without loading its large tensors."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Sampling LoRA does not exist: {resolved}")
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError("Qwen sampling LoRA must be a .safetensors file")

    with safe_open(str(resolved), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        down_suffix = ".lora_down.weight"
        up_suffix = ".lora_up.weight"
        alpha_suffix = ".alpha"
        down_prefixes = {key.removesuffix(down_suffix) for key in keys if key.endswith(down_suffix)}
        up_prefixes = {key.removesuffix(up_suffix) for key in keys if key.endswith(up_suffix)}
        alpha_prefixes = {key.removesuffix(alpha_suffix) for key in keys if key.endswith(alpha_suffix)}
        if not down_prefixes or down_prefixes != up_prefixes or down_prefixes != alpha_prefixes:
            raise ValueError(
                "Sampling LoRA must contain matching lora_down, lora_up, and alpha tensors"
            )
        if not all(prefix.startswith("transformer_blocks.") for prefix in down_prefixes):
            raise ValueError("Sampling LoRA is not in Qwen Image transformer format")

        first_prefix = sorted(down_prefixes)[0]
        down_shape = tuple(weights.get_slice(first_prefix + down_suffix).get_shape())
        up_shape = tuple(weights.get_slice(first_prefix + up_suffix).get_shape())
        if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != up_shape[1]:
            raise ValueError(
                f"Sampling LoRA has incompatible rank shapes: down={down_shape}, up={up_shape}"
            )
    return str(resolved)


def build_qwen_sampling_lora_network(
    *,
    base_model,
    transformer: torch.nn.Module,
    lora_path: str,
    device: torch.device,
    use_layer_offloading: bool,
) -> LoRASpecialNetwork:
    """Attach a Qwen LoRA without asking Diffusers/PEFT to rebuild the model.

    Qwen Lightning checkpoints use Diffusers module names without the leading
    ``transformer.`` component.  AiToolkit's native LoRA network includes that
    component and supports the toolkit layer memory manager, so normalize the
    names while retaining each module's checkpoint alpha.
    """
    modules_dim: dict[str, int] = {}
    modules_alpha: dict[str, float] = {}
    down_suffix = ".lora_down.weight"

    with safe_open(lora_path, framework="pt", device="cpu") as weights:
        for key in weights.keys():
            if not key.endswith(down_suffix):
                continue
            module_path = key.removesuffix(down_suffix)
            native_name = ("transformer." + module_path).replace(".", "$$")
            modules_dim[native_name] = int(weights.get_slice(key).get_shape()[0])
            modules_alpha[native_name] = float(
                weights.get_tensor(module_path + ".alpha").float().item()
            )

    if not modules_dim:
        raise ValueError(f"Sampling LoRA contains no Qwen modules: {lora_path}")

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

    # load_file is memory mapped.  Renaming the dictionary entries does not copy
    # the 850 MB checkpoint; load_state_dict copies each tensor directly into its
    # already-created native LoRA module.
    raw_state = load_file(lora_path, device="cpu")
    native_state = OrderedDict(("transformer." + key, value) for key, value in raw_state.items())
    del raw_state
    extra = network.load_weights(native_state)
    del native_state
    if extra:
        preview = ", ".join(list(extra.keys())[:5])
        raise ValueError(f"Sampling LoRA has unmatched tensors: {preview}")

    network.requires_grad_(False)
    network.eval()
    network.is_active = False
    network.can_merge_in = False

    if use_layer_offloading:
        # Keep the adapter resident in pinned CPU memory and stream its linear
        # weights alongside the already-offloaded Qwen transformer blocks.
        MemoryManager.attach(network, device)
    else:
        network.force_to(device, dtype=base_model.torch_dtype)
    network._update_torch_multiplier()
    return network


class QwenImageEditPlusRGBACustomPipeline(QwenImageEditPlusCustomPipeline):
    """Make RGB edit controls opaque before sending them through the RGBA VAE."""

    def _encode_vae_image(self, image: torch.Tensor, generator):
        image = ensure_normalized_rgba_tensor(image)
        return super()._encode_vae_image(image=image, generator=generator)


class QwenImageEditPlusRGBAModel(QwenImageEditPlusModel):
    """QIE2511 LoRA backend using the toolkit-trained, QIE-compatible RGBA VAE."""

    arch = "qwen_image_edit_plus_rgba"
    default_rgba_vae_path = str(
        Path(__file__).resolve().parents[2] / "models" / "TransparentQIE2511VAE_diffusers"
    )
    _qwen_pipeline = QwenImageEditPlusRGBACustomPipeline

    def __init__(self, device, model_config, dtype="bf16", *args, **kwargs):
        super().__init__(device, model_config, dtype, *args, **kwargs)
        self.sample_lora_path = None
        self._sample_lora_network = None
        if model_config.sample_lora_path:
            self.sample_lora_path = validate_qwen_sampling_lora(model_config.sample_lora_path)
            model_config.sample_lora_path = self.sample_lora_path
        vae_path = model_config.vae_path or self.default_rgba_vae_path
        vae_subfolder = model_config.model_kwargs.get("rgba_vae_subfolder", "vae")
        cache_identity = f"{vae_path}|{vae_subfolder}|rgba-preprocess-v1"
        cache_digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
        # AiToolkit includes this value in every latent cache key.
        self.latent_space_version = f"qwen-image-rgba-{cache_digest}"

    def _load_rgba_vae(self):
        vae_path = self.model_config.vae_path or self.default_rgba_vae_path
        vae_subfolder = self.model_config.model_kwargs.get("rgba_vae_subfolder", "vae")
        load_kwargs = {"torch_dtype": self.vae_torch_dtype}

        # A directly exported Diffusers VAE contains config.json at its root.
        if os.path.isdir(vae_path) and os.path.isfile(os.path.join(vae_path, "config.json")):
            vae = AutoencoderKLQwenImage.from_pretrained(vae_path, **load_kwargs)
        elif vae_subfolder in (None, ""):
            vae = AutoencoderKLQwenImage.from_pretrained(vae_path, **load_kwargs)
        else:
            vae = AutoencoderKLQwenImage.from_pretrained(
                vae_path,
                subfolder=vae_subfolder,
                **load_kwargs,
            )
        validate_qwen_rgba_vae_config(vae.config)
        return vae

    def _load_qwen_vae(self, base_model_path, dtype):
        # QwenImageModel calls this hook while constructing its pipeline. The
        # standard RGB VAE is therefore never loaded for the RGBA architecture.
        self.print_and_status_update("Loading Qwen RGBA VAE")
        return self._load_rgba_vae()

    def load_model(self):
        # Reuse every QIE transformer/text/control path. The overridden VAE hook
        # makes the pipeline RGBA from the moment it is constructed.
        super().load_model()
        validate_qie2511_transformer_config(self.transformer.config)
        validate_qwen_rgba_vae_config(self.vae.config)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.print_and_status_update("Qwen RGBA VAE loaded")

    def get_generation_pipeline(self):
        pipeline = QwenImageEditPlusRGBACustomPipeline(
            scheduler=self.get_train_scheduler(),
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            processor=self.processor,
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )
        return pipeline.to(self.device_torch)

    def _ensure_sampling_lora_network(self) -> LoRASpecialNetwork:
        network = getattr(self, "_sample_lora_network", None)
        if network is not None:
            return network

        started = time.perf_counter()
        self.print_and_status_update("Attaching sampling-only QIE2511 Lightning LoRA")
        network = build_qwen_sampling_lora_network(
            base_model=self,
            transformer=unwrap_model(self.transformer),
            lora_path=self.sample_lora_path,
            device=self.device_torch,
            use_layer_offloading=bool(self.model_config.layer_offloading),
        )
        self._sample_lora_network = network
        elapsed = time.perf_counter() - started
        self.print_and_status_update(
            f"Sampling-only QIE2511 Lightning LoRA ready: "
            f"{len(network.unet_loras)} modules in {elapsed:.1f}s"
        )
        return network

    def generate_images(self, image_configs, sampler=None, pipeline=None):
        # This backend always decodes four-channel images. Old/cloned jobs may
        # not contain `sample.format: png` and would otherwise inherit the
        # toolkit's JPEG default, which Pillow cannot write without destroying
        # transparency. Enforce PNG at the model boundary so every sampling
        # entry point, including baseline samples, is alpha-safe.
        for image_config in image_configs:
            image_config.output_ext = "png"
            output_path = getattr(image_config, "output_path", None)
            if output_path:
                image_config.output_path = os.path.splitext(output_path)[0] + ".png"

        if self.sample_lora_path is None:
            return super().generate_images(image_configs, sampler=sampler, pipeline=pipeline)

        # This checkpoint is distilled specifically for four inference steps at
        # CFG 1. Keep periodic previews representative and deterministic even if
        # an older cloned job still contains the normal QIE sampling defaults.
        for image_config in image_configs:
            image_config.num_inference_steps = 4
            image_config.guidance_scale = 1.0

        if pipeline is None:
            pipeline = self.get_generation_pipeline()

        # QwenImageEditPlusModel's SamplingLoRAMixin owns activation and always
        # removes the adapter again before the training forward pass resumes.
        return super().generate_images(image_configs, sampler=sampler, pipeline=pipeline)

    def encode_images(self, image_list, device=None, dtype=None):
        if isinstance(image_list, torch.Tensor):
            if image_list.ndim == 3:
                image_list = [image_list]
            elif image_list.ndim == 4:
                image_list = list(image_list)
            else:
                raise ValueError(f"Unexpected image tensor shape: {tuple(image_list.shape)}")
        rgba_images = [ensure_normalized_rgba_tensor(image) for image in image_list]
        return super().encode_images(rgba_images, device=device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        images = super().decode_latents(latents, device=device, dtype=dtype)
        if images.shape[1] != 4:
            raise ValueError(
                "The configured transparent Qwen VAE decoded "
                f"{images.shape[1]} channels instead of RGBA"
            )
        return images
