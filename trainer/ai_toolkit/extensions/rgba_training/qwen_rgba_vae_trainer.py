from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jobs.process import BaseTrainProcess
from toolkit.rgba_utils import prepare_rgba_image, resize_rgba_alpha_safe
from toolkit.train_tools import get_torch_dtype

from .comfy_vae_export import save_qwen_rgba_vae_for_comfy
from .vae_metrics import (
    MetricAccumulator,
    evaluate_readiness,
    normalize_readiness_thresholds,
    rgba_reconstruction_metrics,
)


IMAGE_EXTENSIONS = (".png", ".webp", ".tif", ".tiff")


def _stable_fraction(path: str) -> float:
    digest = hashlib.sha256(Path(path).as_posix().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def split_rgba_files(
    files: Sequence[str],
    *,
    validation_fraction: float,
    validation_max_images: int,
    validation_min_images: int,
) -> tuple[list[str], list[str]]:
    """Create a stable validation split that does not change between restarts."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = sorted(files, key=lambda path: (_stable_fraction(path), Path(path).as_posix()))
    desired = max(validation_min_images, round(len(ordered) * validation_fraction))
    validation_count = min(validation_max_images, desired, max(1, len(ordered) - 1))
    validation = ordered[:validation_count]
    training = ordered[validation_count:]
    if not training or not validation:
        raise ValueError("RGBA VAE training requires at least one training and one validation image")
    return training, validation


class RGBAVaeDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        *,
        resolution: int,
        alpha_threshold: float,
        hidden_rgb_color: Sequence[int],
        edge_color_correction: str,
        edge_matte_color: Sequence[int],
        edge_width: float,
        flip_x: bool = False,
    ) -> None:
        self.files = list(files)
        self.resolution = int(resolution)
        self.alpha_threshold = float(alpha_threshold)
        self.hidden_rgb_color = tuple(int(x) for x in hidden_rgb_color)
        self.edge_color_correction = edge_color_correction
        self.edge_matte_color = tuple(int(x) for x in edge_matte_color)
        self.edge_width = float(edge_width)
        self.flip_x = bool(flip_x)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image = prepare_rgba_image(
                image,
                require_alpha=True,
                alpha_threshold=self.alpha_threshold,
                hidden_rgb_color=self.hidden_rgb_color,
                edge_color_correction=self.edge_color_correction,
                edge_matte_color=self.edge_matte_color,
                edge_width=self.edge_width,
            )
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = resize_rgba_alpha_safe(
            image,
            (self.resolution, self.resolution),
            hidden_rgb_color=self.hidden_rgb_color,
        )
        if self.flip_x and _stable_fraction(path + "|flip") < 0.5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, path


def expand_qwen_vae_state_dict_to_rgba(
    rgb_state_dict: dict[str, torch.Tensor],
    *,
    alpha_decoder_scale: float = 0.0,
    alpha_decoder_bias: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Expand only Qwen VAE boundary convolutions from RGB to RGBA.

    Every existing tensor is copied exactly. The encoder's alpha slice starts
    at zero, so an alpha-neutral input reproduces the standard Qwen latent
    space exactly. The decoder alpha head starts close to opaque.
    """
    # ``load_state_dict`` copies tensors into the destination model. Keep a
    # shallow mapping here instead of cloning the complete ~GB checkpoint;
    # only the three boundary tensors are newly allocated below.
    state = dict(rgb_state_dict)
    encoder_key = "encoder.conv_in.weight"
    decoder_weight_key = "decoder.conv_out.weight"
    decoder_bias_key = "decoder.conv_out.bias"
    if encoder_key not in state or decoder_weight_key not in state or decoder_bias_key not in state:
        raise ValueError("source checkpoint is not an AutoencoderKLQwenImage VAE")

    encoder_weight = state[encoder_key]
    decoder_weight = state[decoder_weight_key]
    decoder_bias = state[decoder_bias_key]
    if encoder_weight.shape[1] != 3 or decoder_weight.shape[0] != 3 or decoder_bias.shape[0] != 3:
        raise ValueError(
            "source VAE boundary shapes must be RGB; received "
            f"encoder={tuple(encoder_weight.shape)}, decoder={tuple(decoder_weight.shape)}"
        )

    expanded_encoder = encoder_weight.new_zeros((encoder_weight.shape[0], 4, *encoder_weight.shape[2:]))
    expanded_encoder[:, :3].copy_(encoder_weight)

    expanded_decoder = decoder_weight.new_empty((4, *decoder_weight.shape[1:]))
    expanded_decoder[:3].copy_(decoder_weight)
    luminance_head = (
        decoder_weight[0] * 0.2126 + decoder_weight[1] * 0.7152 + decoder_weight[2] * 0.0722
    )
    expanded_decoder[3].copy_(luminance_head * float(alpha_decoder_scale))
    expanded_bias = decoder_bias.new_empty((4,))
    expanded_bias[:3].copy_(decoder_bias)
    expanded_bias[3] = float(alpha_decoder_bias)

    state[encoder_key] = expanded_encoder
    state[decoder_weight_key] = expanded_decoder
    state[decoder_bias_key] = expanded_bias
    return state


def expand_qwen_vae_config_to_rgba(source_config) -> dict:
    """Return a Qwen VAE config that really constructs four-channel layers.

    Diffusers records constructor arguments omitted by the source checkpoint in
    ``_use_default_values``. Qwen's RGB config omits ``input_channels``, so a
    model loaded from it marks that argument as defaulted. Carrying the marker
    into ``from_config`` makes Diffusers discard our explicit value of 4 and
    silently construct three-channel boundary convolutions again.
    """
    config = dict(source_config)
    config.pop("_use_default_values", None)
    config["input_channels"] = 4
    return config


class AlphaBoundaryGuard:
    """Train alpha slices while restoring all original RGB entries after each step."""

    def __init__(self, vae: AutoencoderKLQwenImage, *, zero_dc_alpha_encoder: bool = True) -> None:
        self.vae = vae
        self.zero_dc_alpha_encoder = bool(zero_dc_alpha_encoder)
        self.encoder_weight = vae.encoder.conv_in.weight
        self.decoder_weight = vae.decoder.conv_out.weight
        self.decoder_bias = vae.decoder.conv_out.bias
        self.encoder_rgb = self.encoder_weight[:, :3].detach().clone()
        self.decoder_rgb = self.decoder_weight[:3].detach().clone()
        self.decoder_rgb_bias = self.decoder_bias[:3].detach().clone()

        vae.requires_grad_(False)
        self.encoder_weight.requires_grad_(True)
        self.decoder_weight.requires_grad_(True)
        self.decoder_bias.requires_grad_(True)

        encoder_mask = torch.zeros_like(self.encoder_weight)
        encoder_mask[:, 3:4] = 1
        decoder_mask = torch.zeros_like(self.decoder_weight)
        decoder_mask[3:4] = 1
        bias_mask = torch.zeros_like(self.decoder_bias)
        bias_mask[3:4] = 1
        self._hooks = [
            self.encoder_weight.register_hook(lambda grad: grad * encoder_mask),
            self.decoder_weight.register_hook(lambda grad: grad * decoder_mask),
            self.decoder_bias.register_hook(lambda grad: grad * bias_mask),
        ]

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.encoder_weight, self.decoder_weight, self.decoder_bias]

    @torch.no_grad()
    def prepare_step(self) -> None:
        return

    @torch.no_grad()
    def restore_rgb(self) -> None:
        self.encoder_weight[:, :3].copy_(self.encoder_rgb)
        self.decoder_weight[:3].copy_(self.decoder_rgb)
        self.decoder_bias[:3].copy_(self.decoder_rgb_bias)
        if self.zero_dc_alpha_encoder:
            # A spatially constant alpha plane (fully opaque input) should not
            # shift the standard RGB latent. Keep every temporal kernel slice
            # zero-mean over XY while still allowing alpha edges to be encoded.
            alpha_kernel = self.encoder_weight[:, 3:4]
            alpha_kernel.sub_(alpha_kernel.mean(dim=(-1, -2), keepdim=True))


class FullRGBAVAEFineTune:
    """Official Qwen/AlphaVAE-style full-model RGBA fine-tuning controller."""

    def __init__(self, vae: AutoencoderKLQwenImage, *, alpha_lr_multiplier: float = 10.0) -> None:
        self.vae = vae
        self.alpha_lr_multiplier = float(alpha_lr_multiplier)
        if self.alpha_lr_multiplier <= 0:
            raise ValueError("train.alpha_lr_multiplier must be greater than zero")
        vae.requires_grad_(True)
        self.encoder_weight = vae.encoder.conv_in.weight
        self.decoder_weight = vae.decoder.conv_out.weight
        self.decoder_bias = vae.decoder.conv_out.bias
        self._alpha_before_step: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.vae.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def prepare_step(self) -> None:
        if self.alpha_lr_multiplier == 1.0:
            return
        self._alpha_before_step = (
            self.encoder_weight[:, 3:4].detach().clone(),
            self.decoder_weight[3:4].detach().clone(),
            self.decoder_bias[3:4].detach().clone(),
        )

    @torch.no_grad()
    def restore_rgb(self) -> None:
        # RGB/RGBA paired reconstruction and reference-latent losses preserve
        # compatibility while allowing the shared VAE body to learn alpha.
        if self._alpha_before_step is None:
            return
        for parameter, before in zip(
            (self.encoder_weight[:, 3:4], self.decoder_weight[3:4], self.decoder_bias[3:4]),
            self._alpha_before_step,
        ):
            parameter.copy_(before + (parameter - before) * self.alpha_lr_multiplier)
        self._alpha_before_step = None


def _posterior_mode(vae: AutoencoderKLQwenImage, image_5d: torch.Tensor) -> torch.Tensor:
    return vae.encode(image_5d).latent_dist.mode()


def _decode_unclamped(vae: AutoencoderKLQwenImage, latent: torch.Tensor) -> torch.Tensor:
    """Decode image latents without Diffusers' final hard clamp.

    AutoencoderKLQwenImage._decode clamps the complete tensor to [-1, 1]. That
    is correct for inference, but an alpha head initialized at +1 receives no
    gradient at all. VAE training uses single-frame image latents, so mirror
    the native non-tiled decode path and leave range limiting to validation.
    """
    if latent.shape[2] != 1:
        raise ValueError("RGBA VAE trainer expects single-frame image latents")
    vae.clear_cache()
    hidden = vae.post_quant_conv(latent)
    vae._conv_idx = [0]
    output = vae.decoder(
        hidden[:, :, :1],
        feat_cache=vae._feat_map,
        feat_idx=vae._conv_idx,
    )
    vae.clear_cache()
    return output


def _latent_std(vae: AutoencoderKLQwenImage, latent: torch.Tensor) -> torch.Tensor:
    return torch.tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)


def _checkerboard(size: tuple[int, int], tile: int = 16) -> np.ndarray:
    height, width = size[1], size[0]
    yy, xx = np.indices((height, width))
    cells = ((xx // tile) + (yy // tile)) % 2
    values = np.where(cells[..., None] == 0, 218, 170).astype(np.uint8)
    return np.repeat(values, 3, axis=2)


def _rgba_tensor_to_preview(image: torch.Tensor) -> Image.Image:
    rgba = ((image.detach().float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5)
    rgba = np.clip(np.rint(rgba), 0, 255).astype(np.uint8)
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    checker = _checkerboard((rgba.shape[1], rgba.shape[0])).astype(np.float32)
    composite = rgb * alpha + checker * (1.0 - alpha)
    return Image.fromarray(np.clip(np.rint(composite), 0, 255).astype(np.uint8), "RGB")


class QwenRGBAVAETrainProcess(BaseTrainProcess):
    """Native RGBA VAE trainer for Qwen Image/Edit models."""

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.device = torch.device(self.get_conf("device", "cuda"))
        self.dtype = get_torch_dtype(self.get_conf("train.dtype", "bf16"))
        self.source_path = self.get_conf("source_vae.name_or_path", "Qwen/Qwen-Image-Edit-2511")
        self.source_subfolder = self.get_conf("source_vae.subfolder", "vae")
        self.local_files_only = self.get_conf("source_vae.local_files_only", False, as_type=bool)
        self.resolution = self.get_conf("train.resolution", 512, as_type=int)
        self.batch_size = self.get_conf("train.batch_size", 1, as_type=int)
        self.gradient_accumulation = self.get_conf("train.gradient_accumulation", 1, as_type=int)
        self.max_steps = self.get_conf("train.steps", 5000, as_type=int)
        self.learning_rate = self.get_conf("train.lr", 1e-5, as_type=float)
        self.weight_decay = self.get_conf("train.weight_decay", 0.0, as_type=float)
        self.max_grad_norm = self.get_conf("train.max_grad_norm", 1.0, as_type=float)
        self.num_workers = self.get_conf("train.num_workers", 2, as_type=int)
        self.gradient_checkpointing = self.get_conf("train.gradient_checkpointing", True, as_type=bool)
        self.train_scope = str(self.get_conf("train.scope", "full")).strip().lower()
        if self.train_scope not in {"full", "alpha_boundary"}:
            raise ValueError("train.scope must be 'full' or 'alpha_boundary'")
        self.zero_dc_alpha_encoder = self.get_conf("train.alpha_encoder_zero_dc", False, as_type=bool)
        self.alpha_lr_multiplier = self.get_conf("train.alpha_lr_multiplier", 10.0, as_type=float)
        self.save_every = self.get_conf("save.every", 250, as_type=int)
        self.max_saves = self.get_conf("save.max_to_keep", 4, as_type=int)
        self.export_comfy_vae = self.get_conf("save.comfy_export", True, as_type=bool)
        self.validation_every = self.get_conf("validation.every", 250, as_type=int)
        self.validation_max_images = self.get_conf("validation.max_images", 32, as_type=int)
        self.validation_min_images = self.get_conf("validation.min_images", 8, as_type=int)
        self.validation_fraction = self.get_conf("validation.fraction", 0.05, as_type=float)
        self.preview_images = self.get_conf("validation.preview_images", 4, as_type=int)
        self.required_passes = self.get_conf("validation.required_consecutive_passes", 2, as_type=int)
        self.stop_when_ready = self.get_conf("validation.stop_when_ready", False, as_type=bool)
        self.thresholds = normalize_readiness_thresholds(self.get_conf("validation.thresholds", {}))
        self.loss_weights = {
            "visible_rgb": self.get_conf("loss.visible_rgb", 1.0, as_type=float),
            "alpha": self.get_conf("loss.alpha", 2.0, as_type=float),
            "alpha_edge": self.get_conf("loss.alpha_edge", 1.0, as_type=float),
            "composite": self.get_conf("loss.composite", 1.0, as_type=float),
            "opaque_latent": self.get_conf("loss.opaque_latent", 5.0, as_type=float),
            "opaque_rgb": self.get_conf("loss.opaque_rgb", 1.0, as_type=float),
            "opaque_alpha": self.get_conf("loss.opaque_alpha", 0.5, as_type=float),
            "latent_delta": self.get_conf("loss.latent_delta", 0.01, as_type=float),
            "perceptual": self.get_conf("loss.perceptual", 0.1, as_type=float),
        }
        self.datasets_config = self.get_conf("datasets", required=True)
        if not isinstance(self.datasets_config, list) or not self.datasets_config:
            raise ValueError("datasets must be a non-empty list")
        self.step_num = 0
        self.consecutive_passes = 0
        self.latest_report: dict | None = None
        self.vae: AutoencoderKLQwenImage | None = None
        self.reference_vae: AutoencoderKLQwenImage | None = None
        self.perceptual_net: torch.nn.Module | None = None
        self.guard: AlphaBoundaryGuard | FullRGBAVAEFineTune | None = None
        self.sqlite_db_path = self.get_conf("sqlite_db_path", None)
        self.job_id = os.environ.get("AITK_JOB_ID")

    def _db_update(self, **values) -> None:
        if not self.sqlite_db_path or not self.job_id or not os.path.isfile(self.sqlite_db_path):
            return
        allowed = {"status", "step", "info", "save_now", "sample_now", "pid"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f'"{key}" = ?' for key in values)
        try:
            with sqlite3.connect(self.sqlite_db_path, timeout=30.0) as connection:
                connection.execute(
                    f'UPDATE "Job" SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                    (*values.values(), self.job_id),
                )
        except sqlite3.Error as exc:
            self.print(f"RGBA VAE trainer database update failed: {exc}")

    def _db_flag(self, column: str, *, consume: bool = False) -> bool:
        if column not in {"stop", "save_now", "sample_now"}:
            raise ValueError(f"Unsupported trainer flag: {column}")
        if not self.sqlite_db_path or not self.job_id or not os.path.isfile(self.sqlite_db_path):
            return False
        try:
            with sqlite3.connect(self.sqlite_db_path, timeout=30.0) as connection:
                row = connection.execute(
                    f'SELECT "{column}" FROM "Job" WHERE id = ?', (self.job_id,)
                ).fetchone()
                active = bool(row and row[0])
                if active and consume:
                    connection.execute(
                        f'UPDATE "Job" SET "{column}" = 0 WHERE id = ?', (self.job_id,)
                    )
                return active
        except sqlite3.Error as exc:
            self.print(f"RGBA VAE trainer database read failed: {exc}")
            return False

    def on_error(self, error: Exception) -> None:
        self._db_update(status="error", info=f"RGBA VAE training failed: {error}")

    def _collect_files(self) -> list[str]:
        files: list[str] = []
        for dataset in self.datasets_config:
            folder = dataset.get("folder_path") or dataset.get("path")
            if not folder or not os.path.isdir(folder):
                raise ValueError(f"RGBA VAE dataset directory does not exist: {folder}")
            recursive = bool(dataset.get("recursive", False))
            iterator: Iterable[Path] = Path(folder).rglob("*") if recursive else Path(folder).iterdir()
            files.extend(str(path) for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        files = sorted(set(files))
        if len(files) < 2:
            raise ValueError("RGBA VAE training needs at least two PNG/WebP/TIFF files")
        return files

    def _dataset(self, files: Sequence[str], training: bool) -> RGBAVaeDataset:
        config = self.datasets_config[0]
        return RGBAVaeDataset(
            files,
            resolution=self.resolution,
            alpha_threshold=float(config.get("rgba_alpha_threshold", 1.0 / 255.0)),
            hidden_rgb_color=config.get("rgba_hidden_rgb_color", [0, 0, 0]),
            edge_color_correction=config.get("rgba_edge_color_correction", "matte_despill"),
            edge_matte_color=config.get("rgba_edge_matte_color", [0, 255, 0]),
            edge_width=float(config.get("rgba_edge_width", 3.0)),
            flip_x=training and bool(config.get("flip_x", False)),
        )

    def _latest_checkpoint(self) -> Path | None:
        root = Path(self.save_root)
        candidates = [path for path in root.glob(f"{self.job.name}_step_*_diffusers") if path.is_dir()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    def _load_source_vae(self) -> AutoencoderKLQwenImage:
        kwargs = {"torch_dtype": torch.float32, "local_files_only": self.local_files_only}
        source_path = Path(str(self.source_path))
        if source_path.is_dir() and (source_path / "config.json").is_file():
            return AutoencoderKLQwenImage.from_pretrained(str(source_path), **kwargs)
        return AutoencoderKLQwenImage.from_pretrained(
            self.source_path,
            subfolder=self.source_subfolder,
            **kwargs,
        )

    @staticmethod
    def _make_encoder_only_reference(source: AutoencoderKLQwenImage) -> AutoencoderKLQwenImage:
        """Keep only the frozen RGB encoder used by compatibility losses."""
        source.requires_grad_(False)
        source.eval()
        # Qwen's encode path calls clear_cache(), which only iterates through
        # decoder.modules(). An empty Identity retains that API while releasing
        # the unused decoder and post-quant convolution weights.
        source.decoder = nn.Identity()
        source.post_quant_conv = nn.Identity()
        return source

    def _load_or_create_vae(self) -> tuple[AutoencoderKLQwenImage, Path | None]:
        checkpoint = self._latest_checkpoint()
        if checkpoint:
            self.print(f"Resuming RGBA VAE from {checkpoint}")
            # Optimizer-owned parameters must stay FP32. The configured dtype
            # is used only by autocast during the forward pass.
            vae = AutoencoderKLQwenImage.from_pretrained(str(checkpoint), torch_dtype=torch.float32)
            if int(getattr(vae.config, "input_channels", 3)) != 4:
                raise ValueError("resume checkpoint is not a four-channel Qwen VAE")
            self.reference_vae = self._make_encoder_only_reference(self._load_source_vae())
            return vae, checkpoint

        self.print(f"Loading standard Qwen VAE from {self.source_path}")
        source = self._load_source_vae()
        source_config = expand_qwen_vae_config_to_rgba(source.config)
        vae = AutoencoderKLQwenImage.from_config(source_config)
        boundary_shapes = (
            int(vae.encoder.conv_in.weight.shape[1]),
            int(vae.decoder.conv_out.weight.shape[0]),
            int(vae.decoder.conv_out.bias.shape[0]),
        )
        if boundary_shapes != (4, 4, 4):
            raise RuntimeError(
                "failed to construct four-channel Qwen VAE boundaries: "
                f"encoder_in={boundary_shapes[0]}, decoder_out={boundary_shapes[1]}, "
                f"decoder_bias={boundary_shapes[2]}"
            )
        rgba_state = expand_qwen_vae_state_dict_to_rgba(source.state_dict())
        missing, unexpected = vae.load_state_dict(rgba_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"could not expand Qwen VAE: missing={missing}, unexpected={unexpected}")
        del rgba_state
        self.reference_vae = self._make_encoder_only_reference(source)
        gc.collect()
        return vae.to(dtype=torch.float32), None

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def _restore_state(self, checkpoint: Path | None, optimizer: torch.optim.Optimizer) -> None:
        if checkpoint is None:
            return
        state_path = checkpoint / "trainer_state.pt"
        if not state_path.is_file():
            match = re.search(r"_step_(\d+)_diffusers$", checkpoint.name)
            self.step_num = int(match.group(1)) if match else 0
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.step_num = int(state.get("step", 0))
        self.consecutive_passes = int(state.get("consecutive_readiness_passes", 0))
        optimizer.load_state_dict(state["optimizer"])

    def _opaque_training_target(self, batch: torch.Tensor) -> torch.Tensor:
        """Composite RGBA over canonical backgrounds and mark it opaque.

        Hidden RGB below transparent pixels is deliberately excluded by alpha
        compositing. This prevents old green-screen pixels from becoming RGB
        supervision for the compatibility branch.
        """
        target_01 = (batch.float() + 1.0) * 0.5
        alpha = target_01[:, 3:4]
        if self.vae.training:
            palette = target_01.new_tensor((0.0, 0.5, 1.0))
            indices = torch.randint(0, len(palette), (batch.shape[0],), device=batch.device)
            background = palette[indices].view(-1, 1, 1, 1)
        else:
            background = target_01.new_full((batch.shape[0], 1, 1, 1), 0.5)
        opaque_rgb = target_01[:, :3] * alpha + background * (1.0 - alpha)
        opaque_01 = torch.cat((opaque_rgb, torch.ones_like(alpha)), dim=1)
        return opaque_01 * 2.0 - 1.0

    def _perceptual_loss(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        opaque_pred_01: torch.Tensor,
        opaque_target_01: torch.Tensor,
    ) -> torch.Tensor:
        if self.perceptual_net is None:
            return pred_01.new_zeros(())

        def composites(rgba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            rgb, alpha = rgba[:, :3], rgba[:, 3:4]
            black = rgb * alpha
            white = black + (1.0 - alpha)
            return black, white

        pred_black, pred_white = composites(pred_01)
        target_black, target_white = composites(target_01)
        perceptual_pred = torch.cat((pred_black, pred_white, opaque_pred_01[:, :3]), dim=0)
        perceptual_target = torch.cat((target_black, target_white, opaque_target_01[:, :3]), dim=0)
        # LPIPS is only a feature-space regularizer. Direct alpha/composite
        # losses remain unclamped and keep gradients alive outside [0, 1].
        perceptual_pred = perceptual_pred.clamp(0.0, 1.0) * 2.0 - 1.0
        perceptual_target = perceptual_target.clamp(0.0, 1.0) * 2.0 - 1.0
        with self._autocast():
            return self.perceptual_net(perceptual_pred, perceptual_target).float().mean()

    def _forward_losses(self, batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        rgba_5d = batch.unsqueeze(2)
        opaque_target = self._opaque_training_target(batch)
        opaque_input = opaque_target.unsqueeze(2)

        with self._autocast():
            with torch.no_grad():
                reference_latent = _posterior_mode(self.reference_vae, opaque_input[:, :3])
            actual_posterior = self.vae.encode(rgba_5d).latent_dist
            opaque_posterior = self.vae.encode(opaque_input).latent_dist
            actual_mode = actual_posterior.mode()
            opaque_mode = opaque_posterior.mode()
            actual_latent = actual_posterior.sample() if self.vae.training else actual_mode
            opaque_latent = opaque_posterior.sample() if self.vae.training else opaque_mode
            prediction = _decode_unclamped(self.vae, actual_latent)[:, :, 0]
            opaque_prediction = _decode_unclamped(self.vae, opaque_latent)[:, :, 0]

        # Do not clamp training predictions. The previous implementation
        # initialized alpha at +1 and then clamped to [0, 1], which zeroed the
        # gradient over nearly the entire background. Validation/output still
        # clamp to the legal image range in rgba_reconstruction_metrics.
        pred_01 = (prediction.float() + 1.0) * 0.5
        target_01 = (batch.float() + 1.0) * 0.5
        opaque_target_01 = (opaque_target.float() + 1.0) * 0.5
        opaque_pred_01 = (opaque_prediction.float() + 1.0) * 0.5
        target_alpha = target_01[:, 3:4]
        pred_alpha = pred_01[:, 3:4]
        visible_rgb = ((pred_01[:, :3] - target_01[:, :3]).abs() * target_alpha).sum() / (
            target_alpha.sum() * 3.0 + 1e-8
        )
        alpha = F.l1_loss(pred_alpha, target_alpha)

        from .vae_metrics import _alpha_edges

        alpha_edge = F.l1_loss(_alpha_edges(pred_alpha), _alpha_edges(target_alpha))
        composite_terms = []
        for value in (1.0, 0.5, 0.0):
            background = torch.full_like(pred_01[:, :3], value)
            pred_composite = pred_01[:, :3] * pred_alpha + background * (1.0 - pred_alpha)
            target_composite = target_01[:, :3] * target_alpha + background * (1.0 - target_alpha)
            composite_terms.append(F.l1_loss(pred_composite, target_composite))
        composite = torch.stack(composite_terms).mean()

        std = _latent_std(self.vae, reference_latent).float()
        opaque_latent_loss = (((opaque_mode.float() - reference_latent.float()) / std).square()).mean()
        latent_delta = (((actual_mode.float() - reference_latent.float()) / std).square()).mean()
        opaque_rgb = F.l1_loss(opaque_pred_01[:, :3], opaque_target_01[:, :3])
        opaque_alpha = F.l1_loss(
            opaque_pred_01[:, 3:4],
            torch.ones_like(target_alpha),
        )
        perceptual = self._perceptual_loss(
            pred_01,
            target_01,
            opaque_pred_01,
            opaque_target_01,
        )
        losses = {
            "visible_rgb": visible_rgb,
            "alpha": alpha,
            "alpha_edge": alpha_edge,
            "composite": composite,
            "opaque_latent": opaque_latent_loss,
            "opaque_rgb": opaque_rgb,
            "opaque_alpha": opaque_alpha,
            "latent_delta": latent_delta,
            "perceptual": perceptual,
        }
        total = sum(losses[name] * self.loss_weights[name] for name in losses)
        opaque_rmse = torch.sqrt(opaque_latent_loss.detach().clamp_min(0.0))
        return total, losses, prediction, opaque_rmse

    @torch.no_grad()
    def validate(self, loader: DataLoader, step: int) -> dict:
        self.vae.eval()
        accumulator = MetricAccumulator()
        previews: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        for batch, paths in tqdm(loader, desc="Validating RGBA VAE", leave=False):
            batch = batch.to(self.device, dtype=torch.float32)
            _, _, prediction, opaque_rmse = self._forward_losses(batch)
            metrics = rgba_reconstruction_metrics(prediction, batch, opaque_latent_rmse=opaque_rmse)
            accumulator.update(metrics, batch.shape[0])
            for idx in range(min(batch.shape[0], self.preview_images - len(previews))):
                previews.append((batch[idx].cpu(), prediction[idx].cpu(), paths[idx]))
        metrics = accumulator.compute()
        passed_now, checks = evaluate_readiness(metrics, self.thresholds)
        self.consecutive_passes = self.consecutive_passes + 1 if passed_now else 0
        ready = passed_now and self.consecutive_passes >= self.required_passes
        report = {
            "schema_version": 2,
            "step": int(step),
            "ready": ready,
            "passed_this_validation": passed_now,
            "consecutive_passes": self.consecutive_passes,
            "required_consecutive_passes": self.required_passes,
            "metrics": metrics,
            "checks": checks,
            "meaning": (
                "ready means the fixed validation split passed RGBA reconstruction, edge, composite, "
                "finite-value, and standard-Qwen opaque-latent compatibility gates"
            ),
        }
        self.latest_report = report
        report_path = Path(self.save_root) / "readiness_latest.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        ready_marker = Path(self.save_root) / "READY"
        if ready:
            ready_marker.write_text(f"step={step}\nreport={report_path.name}\n", encoding="utf-8")
        elif ready_marker.exists():
            ready_marker.unlink()
        self._save_preview(previews, step, ready)
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"validation/{name}", value, step)
            self.writer.add_scalar("validation/ready", float(ready), step)
        self.print(f"RGBA VAE readiness at step {step}: {'READY' if ready else 'NOT READY'}")
        for name, check in checks.items():
            self.print(
                f" - {name}: {check['value']:.6f} {check['comparison']} "
                f"{check['threshold']:.6f} [{'pass' if check['passed'] else 'fail'}]"
            )
        self.print(
            " - alpha diagnostics: "
            f"background_mean={metrics.get('background_alpha_mean', float('nan')):.6f}, "
            f"foreground_mean={metrics.get('foreground_alpha_mean', float('nan')):.6f}, "
            f"transparent_fraction={metrics.get('predicted_transparent_fraction', float('nan')):.6f}"
        )
        self.vae.train()
        return report

    def _save_preview(self, items: Sequence[tuple[torch.Tensor, torch.Tensor, str]], step: int, ready: bool) -> None:
        if not items:
            return
        cell = self.resolution
        header = 32
        canvas = Image.new("RGB", (cell * 4, header + cell * len(items)), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(("target", "reconstruction", "target alpha", "pred alpha")):
            draw.text((column * cell + 8, 8), title, fill=(235, 235, 235))
        for row, (target, prediction, _) in enumerate(items):
            y = header + row * cell
            canvas.paste(_rgba_tensor_to_preview(target), (0, y))
            canvas.paste(_rgba_tensor_to_preview(prediction), (cell, y))
            for column, image in enumerate((target, prediction), start=2):
                alpha = ((image[3].float().numpy() + 1.0) * 127.5)
                alpha_img = Image.fromarray(np.clip(np.rint(alpha), 0, 255).astype(np.uint8), "L").convert("RGB")
                canvas.paste(alpha_img, (column * cell, y))
        # Use the toolkit's existing samples directory so the job viewer can
        # display these round-trip sheets. They are validation previews, not
        # diffusion samples.
        preview_dir = Path(self.save_root) / "samples"
        preview_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(preview_dir / f"vae_{step:09d}_0.png")

    def save(self, optimizer: torch.optim.Optimizer, step: int) -> Path:
        checkpoint = Path(self.save_root) / f"{self.job.name}_step_{step:09d}_diffusers"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.vae.save_pretrained(str(checkpoint), safe_serialization=True)
        if self.export_comfy_vae:
            comfy_path = checkpoint / f"{self.job.name}_step_{step:09d}_ComfyUI_bf16.safetensors"
            save_qwen_rgba_vae_for_comfy(
                self.vae.state_dict(),
                comfy_path,
                dtype=torch.bfloat16,
                metadata={"step": str(step), "source_format": "diffusers"},
            )
            self.print(f"Saved ComfyUI RGBA VAE to {comfy_path}")
        torch.save(
            {
                "step": int(step),
                "optimizer": optimizer.state_dict(),
                "consecutive_readiness_passes": self.consecutive_passes,
            },
            checkpoint / "trainer_state.pt",
        )
        if self.latest_report is not None:
            (checkpoint / "readiness.json").write_text(
                json.dumps(self.latest_report, indent=2), encoding="utf-8"
            )
        checkpoints = sorted(
            (path for path in Path(self.save_root).glob(f"{self.job.name}_step_*_diffusers") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        for old in checkpoints[:-self.max_saves]:
            shutil.rmtree(old)
        self.print(f"Saved RGBA VAE checkpoint to {checkpoint}")
        return checkpoint

    def run(self) -> None:
        super().run()
        self._db_update(status="running", step=self.step_num, info="Preparing RGBA VAE training")
        files = self._collect_files()
        train_files, validation_files = split_rgba_files(
            files,
            validation_fraction=self.validation_fraction,
            validation_max_images=self.validation_max_images,
            validation_min_images=self.validation_min_images,
        )
        self.print(f"RGBA VAE dataset: {len(train_files)} train, {len(validation_files)} validation")
        train_loader = DataLoader(
            self._dataset(train_files, training=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        validation_loader = DataLoader(
            self._dataset(validation_files, training=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.vae, checkpoint = self._load_or_create_vae()
        # Keep trainable weights and Adam moments in FP32. BF16/FP16 is a
        # compute-only autocast setting; direct 1e-5 updates to a BF16 bias near
        # one round to zero and were the reason the old alpha bias never moved.
        self.vae.to(self.device, dtype=torch.float32)
        self.reference_vae.to(self.device, dtype=self.dtype)
        self.reference_vae.requires_grad_(False)
        self.reference_vae.eval()
        if self.gradient_checkpointing and hasattr(self.vae, "enable_gradient_checkpointing"):
            self.vae.enable_gradient_checkpointing()
        if self.loss_weights["perceptual"] > 0:
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError("loss.perceptual > 0 requires the installed 'lpips' package") from exc
            self.print("Loading frozen VGG LPIPS network for VAE perceptual loss")
            self.perceptual_net = lpips.LPIPS(net="vgg").to(self.device).eval()
            self.perceptual_net.requires_grad_(False)

        if self.train_scope == "full":
            self.guard = FullRGBAVAEFineTune(
                self.vae,
                alpha_lr_multiplier=self.alpha_lr_multiplier,
            )
            self.print(
                "RGBA VAE training scope: full model (Qwen-Image-Layered/AlphaVAE strategy), "
                f"alpha LR multiplier={self.alpha_lr_multiplier:g}"
            )
        else:
            self.guard = AlphaBoundaryGuard(
                self.vae,
                zero_dc_alpha_encoder=self.zero_dc_alpha_encoder,
            )
            self.print("RGBA VAE training scope: legacy alpha boundary only")
        optimizer = torch.optim.AdamW(
            self.guard.parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
        )
        self._restore_state(checkpoint, optimizer)
        self.guard.restore_rgb()

        self.validate(validation_loader, self.step_num)
        progress = tqdm(total=self.max_steps, initial=self.step_num, desc=self.job.name)
        optimizer.zero_grad(set_to_none=True)
        stop = False
        stopped_by_user = False
        accumulated_batches = 0
        while self.step_num < self.max_steps and not stop:
            for batch, _ in train_loader:
                if self.step_num >= self.max_steps:
                    break
                self.vae.train()
                batch = batch.to(self.device, dtype=torch.float32, non_blocking=True)
                total, losses, _, _ = self._forward_losses(batch)
                if not torch.isfinite(total):
                    details = ", ".join(f"{name}={value.detach().float().item()}" for name, value in losses.items())
                    raise FloatingPointError(f"RGBA VAE loss is not finite at step {self.step_num}: {details}")
                (total / self.gradient_accumulation).backward()
                accumulated_batches += 1
                if accumulated_batches < self.gradient_accumulation:
                    continue
                encoder_alpha_grad = (
                    self.guard.encoder_weight.grad[:, 3:4].detach().float().norm().item()
                    if self.guard.encoder_weight.grad is not None
                    else 0.0
                )
                decoder_alpha_grad = (
                    self.guard.decoder_weight.grad[3:4].detach().float().norm().item()
                    if self.guard.decoder_weight.grad is not None
                    else 0.0
                )
                alpha_bias_grad = (
                    self.guard.decoder_bias.grad[3:4].detach().float().norm().item()
                    if self.guard.decoder_bias.grad is not None
                    else 0.0
                )
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.guard.parameters, self.max_grad_norm)
                self.guard.prepare_step()
                optimizer.step()
                self.guard.restore_rgb()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                self.step_num += 1
                self._db_update(
                    step=self.step_num,
                    info=f"RGBA VAE training step {self.step_num}/{self.max_steps}",
                )
                progress.update(1)
                progress.set_postfix_str(
                    " ".join(
                        [f"loss={total.detach().float().item():.3e}"]
                        + [f"{name}={value.detach().float().item():.2e}" for name, value in losses.items()]
                        + [f"alpha_bias={self.vae.decoder.conv_out.bias[3].detach().float().item():.5f}"]
                        + [
                            f"grad_enc_a={encoder_alpha_grad:.2e}",
                            f"grad_dec_a={decoder_alpha_grad:.2e}",
                            f"grad_bias_a={alpha_bias_grad:.2e}",
                            f"grad_total={total_grad_norm.detach().float().item():.2e}",
                        ]
                    )
                )
                if self.writer:
                    self.writer.add_scalar("loss/total", total.detach().float().item(), self.step_num)
                    for name, value in losses.items():
                        self.writer.add_scalar(f"loss/{name}", value.detach().float().item(), self.step_num)

                if self.validation_every and self.step_num % self.validation_every == 0:
                    report = self.validate(validation_loader, self.step_num)
                    if report["ready"] and self.stop_when_ready:
                        stop = True
                if self.save_every and self.step_num % self.save_every == 0:
                    self.save(optimizer, self.step_num)
                if self._db_flag("sample_now", consume=True):
                    self.print(f"Running requested RGBA VAE validation at step {self.step_num}")
                    self.validate(validation_loader, self.step_num)
                if self._db_flag("save_now", consume=True):
                    self.print(f"Saving requested RGBA VAE checkpoint at step {self.step_num}")
                    self.save(optimizer, self.step_num)
                if self._db_flag("stop"):
                    self.print(f"Stop requested at RGBA VAE step {self.step_num}")
                    stopped_by_user = True
                    stop = True
                if stop:
                    break
        progress.close()
        if self.latest_report is None or self.latest_report.get("step") != self.step_num:
            self.validate(validation_loader, self.step_num)
        self.save(optimizer, self.step_num)
        if stopped_by_user:
            self._db_update(status="stopped", step=self.step_num, info="RGBA VAE training stopped")
        else:
            self._db_update(status="completed", step=self.step_num, info="RGBA VAE training completed")
