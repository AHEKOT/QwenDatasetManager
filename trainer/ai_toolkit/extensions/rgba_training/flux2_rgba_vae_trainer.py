from __future__ import annotations

import gc
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from extensions_built_in.diffusion_models.flux2.src.autoencoder import (
    AutoEncoder,
    AutoEncoderParams,
    AutoEncoderSmallDecoderParams,
)

from .qwen_rgba_vae_trainer import QwenRGBAVAETrainProcess
from .vae_metrics import _alpha_edges


def flux2_autoencoder_params(state_dict: dict[str, torch.Tensor]):
    """Infer the full or small-decoder FLUX.2 VAE layout from native weights."""
    decoder_probe = state_dict.get("decoder.up.0.block.0.conv1.bias")
    if decoder_probe is None:
        raise ValueError("checkpoint is not a native FLUX.2 AutoEncoder")
    params = AutoEncoderSmallDecoderParams() if decoder_probe.shape[0] == 96 else AutoEncoderParams()
    encoder = state_dict.get("encoder.conv_in.weight")
    decoder = state_dict.get("decoder.conv_out.weight")
    if encoder is None or decoder is None:
        raise ValueError("FLUX.2 VAE checkpoint is missing boundary convolutions")
    params.in_channels = int(encoder.shape[1])
    params.out_ch = int(decoder.shape[0])
    return params


def expand_flux2_vae_state_dict_to_rgba(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expand only native FLUX.2 VAE RGB boundary tensors to RGBA."""
    result = {key: value.detach().clone() for key, value in state_dict.items()}
    encoder = result.get("encoder.conv_in.weight")
    decoder = result.get("decoder.conv_out.weight")
    decoder_bias = result.get("decoder.conv_out.bias")
    if encoder is None or decoder is None or decoder_bias is None:
        raise ValueError("FLUX.2 VAE checkpoint is missing RGB boundary tensors")
    if (encoder.shape[1], decoder.shape[0], decoder_bias.shape[0]) == (4, 4, 4):
        return result
    if (encoder.shape[1], decoder.shape[0], decoder_bias.shape[0]) != (3, 3, 3):
        raise ValueError(
            "FLUX.2 VAE boundaries must be RGB or RGBA; received "
            f"encoder={encoder.shape[1]}, decoder={decoder.shape[0]}, bias={decoder_bias.shape[0]}"
        )

    expanded_encoder = encoder.new_zeros((encoder.shape[0], 4, *encoder.shape[2:]))
    expanded_encoder[:, :3] = encoder
    expanded_decoder = decoder.new_zeros((4, decoder.shape[1], *decoder.shape[2:]))
    expanded_decoder[:3] = decoder
    expanded_bias = decoder_bias.new_zeros((4,))
    expanded_bias[:3] = decoder_bias
    # Start as fully opaque without changing the pretrained RGB reconstruction.
    expanded_bias[3] = 1.0
    result["encoder.conv_in.weight"] = expanded_encoder
    result["decoder.conv_out.weight"] = expanded_decoder
    result["decoder.conv_out.bias"] = expanded_bias
    return result


class Flux2RGBAVAETrainProcess(QwenRGBAVAETrainProcess):
    """RGBA VAE trainer shared by FLUX.2 Klein 4B and 9B (native z=32 VAE)."""

    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.source_filename = str(
            self.get_conf("source_vae.filename", "ae.safetensors")
        ).strip() or "ae.safetensors"

    def _latest_checkpoint(self) -> Path | None:
        root = Path(self.save_root)
        candidates = [
            path
            for path in root.glob(f"{self.job.name}_step_*_flux2")
            if path.is_dir() and (path / "ae.safetensors").is_file()
        ]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    def _resolve_source_file(self) -> Path:
        source = Path(str(self.source_path))
        if source.is_file():
            return source
        if source.is_dir():
            candidates = [
                source / self.source_filename,
                source / self.source_subfolder / self.source_filename
                if self.source_subfolder
                else source / self.source_filename,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                f"FLUX.2 VAE file {self.source_filename!r} was not found in {source}"
            )
        downloaded = hf_hub_download(
            repo_id=str(self.source_path),
            filename=self.source_filename,
            subfolder=self.source_subfolder or None,
            local_files_only=self.local_files_only,
        )
        return Path(downloaded)

    @staticmethod
    def _model_from_state(state_dict: dict[str, torch.Tensor]) -> AutoEncoder:
        params = flux2_autoencoder_params(state_dict)
        vae = AutoEncoder(params)
        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"could not load FLUX.2 VAE: missing={missing}, unexpected={unexpected}"
            )
        return vae

    def _load_source_vae(self) -> AutoEncoder:
        source_file = self._resolve_source_file()
        state = load_file(str(source_file), device="cpu")
        params = flux2_autoencoder_params(state)
        if params.in_channels != 3 or params.out_ch != 3 or params.z_channels != 32:
            raise ValueError(
                "source FLUX.2 VAE must be the standard RGB z=32 AutoEncoder; "
                f"received input={params.in_channels}, output={params.out_ch}, z={params.z_channels}"
            )
        return self._model_from_state(state).to(dtype=torch.float32)

    @staticmethod
    def _make_encoder_only_reference(source: AutoEncoder) -> AutoEncoder:
        source.requires_grad_(False)
        source.eval()
        source.decoder = nn.Identity()
        return source

    def _load_or_create_vae(self) -> tuple[AutoEncoder, Path | None]:
        checkpoint = self._latest_checkpoint()
        if checkpoint:
            self.print(f"Resuming FLUX.2 RGBA VAE from {checkpoint}")
            state = load_file(str(checkpoint / "ae.safetensors"), device="cpu")
            params = flux2_autoencoder_params(state)
            if params.in_channels != 4 or params.out_ch != 4 or params.z_channels != 32:
                raise ValueError("resume checkpoint is not a four-channel FLUX.2 z=32 VAE")
            vae = self._model_from_state(state)
            self.reference_vae = self._make_encoder_only_reference(self._load_source_vae())
            return vae.to(dtype=torch.float32), checkpoint

        self.print(f"Loading standard FLUX.2 VAE from {self.source_path}")
        source = self._load_source_vae()
        rgba_params = replace(source.params, in_channels=4, out_ch=4)
        vae = AutoEncoder(rgba_params)
        rgba_state = expand_flux2_vae_state_dict_to_rgba(source.state_dict())
        missing, unexpected = vae.load_state_dict(rgba_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"could not expand FLUX.2 VAE: missing={missing}, unexpected={unexpected}"
            )
        del rgba_state
        self.reference_vae = self._make_encoder_only_reference(source)
        gc.collect()
        return vae.to(dtype=torch.float32), None

    def _restore_state(self, checkpoint: Path | None, optimizer: torch.optim.Optimizer) -> None:
        if checkpoint is None:
            return
        state_path = checkpoint / "trainer_state.pt"
        if not state_path.is_file():
            match = re.search(r"_step_(\d+)_flux2$", checkpoint.name)
            self.step_num = int(match.group(1)) if match else 0
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.step_num = int(state.get("step", 0))
        self.consecutive_passes = int(state.get("consecutive_readiness_passes", 0))
        optimizer.load_state_dict(state["optimizer"])

    def _forward_losses(self, batch: torch.Tensor):
        opaque_target = self._opaque_training_target(batch)
        with self._autocast():
            with torch.no_grad():
                reference_latent = self.reference_vae.encode(opaque_target[:, :3])
            actual_mode = self.vae.encode(batch)
            opaque_mode = self.vae.encode(opaque_target)
            prediction = self.vae.decode(actual_mode)
            opaque_prediction = self.vae.decode(opaque_mode)

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
        alpha_edge = F.l1_loss(_alpha_edges(pred_alpha), _alpha_edges(target_alpha))
        composite_terms = []
        for value in (1.0, 0.5, 0.0):
            background = torch.full_like(pred_01[:, :3], value)
            pred_composite = pred_01[:, :3] * pred_alpha + background * (1.0 - pred_alpha)
            target_composite = target_01[:, :3] * target_alpha + background * (1.0 - target_alpha)
            composite_terms.append(F.l1_loss(pred_composite, target_composite))
        composite = torch.stack(composite_terms).mean()

        opaque_latent_loss = (opaque_mode.float() - reference_latent.float()).square().mean()
        latent_delta = (actual_mode.float() - reference_latent.float()).square().mean()
        opaque_rgb = F.l1_loss(opaque_pred_01[:, :3], opaque_target_01[:, :3])
        opaque_alpha = F.l1_loss(
            opaque_pred_01[:, 3:4], torch.ones_like(target_alpha)
        )
        perceptual = self._perceptual_loss(
            pred_01, target_01, opaque_pred_01, opaque_target_01
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

    @staticmethod
    def _serializable_state(vae: AutoEncoder, dtype: torch.dtype):
        return {
            key: value.detach().to(device="cpu", dtype=dtype).contiguous()
            for key, value in vae.state_dict().items()
        }

    def save(self, optimizer: torch.optim.Optimizer, step: int) -> Path:
        checkpoint = Path(self.save_root) / f"{self.job.name}_step_{step:09d}_flux2"
        checkpoint.mkdir(parents=True, exist_ok=True)
        save_file(
            self._serializable_state(self.vae, torch.float32),
            str(checkpoint / "ae.safetensors"),
            metadata={"step": str(step), "family": "flux2_klein", "channels": "rgba"},
        )
        if self.export_comfy_vae:
            comfy_path = checkpoint / f"{self.job.name}_step_{step:09d}_ComfyUI_bf16.safetensors"
            save_file(
                self._serializable_state(self.vae, torch.bfloat16),
                str(comfy_path),
                metadata={"step": str(step), "family": "flux2_klein", "channels": "rgba"},
            )
            self.print(f"Saved native FLUX.2 ComfyUI RGBA VAE to {comfy_path}")
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
            (
                path
                for path in Path(self.save_root).glob(f"{self.job.name}_step_*_flux2")
                if path.is_dir()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        for old in checkpoints[:-self.max_saves]:
            shutil.rmtree(old)
        self.print(f"Saved FLUX.2 RGBA VAE checkpoint to {checkpoint}")
        return checkpoint
