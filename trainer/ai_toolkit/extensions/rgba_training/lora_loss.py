from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class RGBALoRALossMixin:
    """Direct, inexpensive alpha supervision for RGBA latent diffusion.

    The RGBA VAE entangles alpha with RGB across all latent channels, so a
    channel-agnostic spatial multiplier cannot specifically reward a correct
    alpha prediction. A single calibrated, frozen probe reads the same alpha
    direction for every image. It supervises the transformer's flow velocity
    directly, so low-noise timesteps cannot make the alpha error vanish before
    the model has learned anything. This adds no VAE decode and no second
    transformer pass.
    """

    supports_rgba_latent_loss = True

    _RGBA_PROBE_FEATURE_LIMIT = 5.0

    def set_rgba_alpha_probe(self, probe: dict[str, torch.Tensor]) -> None:
        required = {"mean", "std", "coefficients"}
        missing = required.difference(probe)
        if missing:
            raise ValueError(f"RGBA alpha probe is missing tensors: {sorted(missing)}")
        mean = probe["mean"].detach().float().cpu()
        std = probe["std"].detach().float().cpu()
        coefficients = probe["coefficients"].detach().float().cpu()
        if mean.ndim != 4 or std.shape != mean.shape:
            raise ValueError("RGBA alpha probe mean/std must have shape [1,C,1,1]")
        expected_features = mean.shape[1] * 2 + 1
        if coefficients.shape != (expected_features, 1):
            raise ValueError(
                "RGBA alpha probe coefficients must have shape "
                f"({expected_features}, 1), received {tuple(coefficients.shape)}"
            )
        self._rgba_alpha_probe_cpu = {
            "mean": mean,
            "std": std.clamp_min(1.0e-3),
            "coefficients": coefficients,
        }
        self._rgba_alpha_probe_device = None

    def _rgba_alpha_probe(self, device: torch.device) -> dict[str, torch.Tensor]:
        probe = getattr(self, "_rgba_alpha_probe_cpu", None)
        if probe is None:
            raise ValueError(
                "Transparent LoRA alpha consistency requires a calibrated fixed alpha probe"
            )
        cached = getattr(self, "_rgba_alpha_probe_device", None)
        if cached is None or cached["mean"].device != device:
            cached = {key: value.to(device=device) for key, value in probe.items()}
            self._rgba_alpha_probe_device = cached
        return cached

    def _rgba_loss_setting(self, name: str, default: float) -> float:
        key = f"rgba_lora_loss_{name}"
        value = self.model_config.model_kwargs.get(key, default)
        return max(0.0, float(value))

    @staticmethod
    def _is_rgba_generation_batch(batch: "DataLoaderBatchDTO") -> bool:
        dataset = getattr(batch, "dataset_config", None)
        return bool(
            dataset is not None
            and getattr(dataset, "rgba_generate_control", False)
            and getattr(dataset, "rgba_control_mode", "edit") == "generation"
        )

    @staticmethod
    def _is_rgba_training_batch(batch: "DataLoaderBatchDTO") -> bool:
        dataset = getattr(batch, "dataset_config", None)
        return bool(
            dataset is not None
            and getattr(dataset, "rgba_generate_control", False)
            and getattr(dataset, "rgba_control_mode", None) in {"edit", "generation"}
        )

    def get_rgba_diffusion_loss_weight(self, batch: "DataLoaderBatchDTO") -> float:
        """Use the exact RGBA latent target for edit, but not generation.

        In edit mode the control image supplies the subject, so ordinary flow
        loss teaches the exact RGBA conversion and preserves its appearance.
        Generation targets, in contrast, would teach the LoRA to reproduce the
        particular people in the dataset. They therefore remain alpha-only.
        Backends without a calibrated probe retain their ordinary loss.
        """

        if (
            self._is_rgba_generation_batch(batch)
            and getattr(self, "_rgba_alpha_probe_cpu", None) is not None
        ):
            return 0.0
        return 1.0

    def get_rgba_validation_metrics(
        self,
        *,
        predicted_clean: torch.Tensor,
        target_clean: torch.Tensor,
        target_rgba: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Measure predicted alpha against the actual validation PNG mask."""

        actual_alpha = ((target_rgba[:, 3:4].float() + 1.0) * 0.5).clamp(0.0, 1.0)
        probe = getattr(self, "_rgba_alpha_probe_cpu", None)
        represented_alpha = None
        if probe is not None:
            probe = self._rgba_alpha_probe(predicted_clean.device)
            mean, std, coefficients = probe["mean"], probe["std"], probe["coefficients"]
            predicted_features = self._rgba_probe_features(
                predicted_clean, mean=mean, std=std
            )
            predicted_alpha = torch.einsum(
                "bchw,co->bohw", predicted_features, coefficients[:-1]
            ) + coefficients[-1].view(1, 1, 1, 1)
            target_features = self._rgba_probe_features(
                target_clean.float(), mean=mean, std=std
            )
            represented_alpha = torch.einsum(
                "bchw,co->bohw", target_features, coefficients[:-1]
            ) + coefficients[-1].view(1, 1, 1, 1)
        else:
            decoded = torch.cat([
                self.decode_latents(
                    chunk,
                    device=predicted_clean.device,
                    dtype=getattr(self, "vae_torch_dtype", predicted_clean.dtype),
                )
                for chunk in predicted_clean.split(1, dim=0)
            ], dim=0)
            if decoded.ndim == 5 and decoded.shape[2] == 1:
                decoded = decoded.squeeze(2)
            if decoded.ndim != 4 or decoded.shape[1] != 4:
                raise ValueError("RGBA validation requires a four-channel VAE decode")
            predicted_alpha = (decoded[:, 3:4].float() + 1.0) * 0.5
            decoded_target = torch.cat([
                self.decode_latents(
                    chunk,
                    device=target_clean.device,
                    dtype=getattr(self, "vae_torch_dtype", target_clean.dtype),
                )
                for chunk in target_clean.split(1, dim=0)
            ], dim=0)
            if decoded_target.ndim == 5 and decoded_target.shape[2] == 1:
                decoded_target = decoded_target.squeeze(2)
            represented_alpha = (decoded_target[:, 3:4].float() + 1.0) * 0.5

        actual_alpha = F.interpolate(
            actual_alpha.to(predicted_alpha.device),
            size=predicted_alpha.shape[-2:],
            mode="area",
        )
        predicted_alpha = predicted_alpha.clamp(0.0, 1.0)
        represented_alpha = represented_alpha.clamp(0.0, 1.0)
        pred_dx, pred_dy = self._rgba_alpha_edges(predicted_alpha)
        target_dx, target_dy = self._rgba_alpha_edges(actual_alpha)
        predicted_mask = predicted_alpha >= 0.5
        actual_mask = actual_alpha >= 0.5
        intersection = (predicted_mask & actual_mask).float().sum()
        union = (predicted_mask | actual_mask).float().sum().clamp_min(1.0)
        background = actual_alpha < 0.5
        foreground = ~background
        background_mean = predicted_alpha[background].mean() if background.any() else predicted_alpha.sum() * 0.0
        foreground_mae = (
            (predicted_alpha[foreground] - actual_alpha[foreground]).abs().mean()
            if foreground.any()
            else predicted_alpha.sum() * 0.0
        )

        # A global MAE can be deceptively good when most pixels are empty.
        # Evaluate a five-latent-pixel boundary band and deep foreground/
        # background separately. Hair wisps, halos and isolated leftovers then
        # affect pass/fail even when they occupy a tiny part of the full canvas.
        actual_binary = (actual_alpha >= 0.5).float()
        dilated = F.max_pool2d(actual_binary, kernel_size=5, stride=1, padding=2)
        eroded = 1.0 - F.max_pool2d(
            1.0 - actual_binary, kernel_size=5, stride=1, padding=2
        )
        boundary = (dilated - eroded) > 0.5
        deep_background = dilated < 0.5
        deep_foreground = eroded > 0.5
        representation_error = (predicted_alpha - represented_alpha).abs()

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return values[mask].mean() if mask.any() else values.sum() * 0.0

        represented_dx, represented_dy = self._rgba_alpha_edges(represented_alpha)
        boundary_dx = boundary[..., :, 1:] | boundary[..., :, :-1]
        boundary_dy = boundary[..., 1:, :] | boundary[..., :-1, :]
        boundary_edge_error = 0.5 * (
            masked_mean((pred_dx - represented_dx).abs(), boundary_dx)
            + masked_mean((pred_dy - represented_dy).abs(), boundary_dy)
        )
        represented_mask = represented_alpha >= 0.5
        represented_intersection = (predicted_mask & represented_mask).float().sum()
        represented_union = (predicted_mask | represented_mask).float().sum().clamp_min(1.0)
        metrics = {
            "alpha_mae": F.l1_loss(predicted_alpha, actual_alpha),
            "alpha_edge_mae": 0.5 * (
                F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
            ),
            "alpha_iou": intersection / union,
            "background_alpha_mean": background_mean,
            "foreground_alpha_mae": foreground_mae,
            "representation_alpha_mae": F.l1_loss(
                predicted_alpha, represented_alpha
            ),
            "representation_boundary_mae": masked_mean(
                representation_error, boundary
            ),
            "representation_boundary_edge_mae": boundary_edge_error,
            "representation_alpha_iou": represented_intersection / represented_union,
            "background_residual_mae": masked_mean(
                representation_error, deep_background
            ),
            "background_false_positive_rate": masked_mean(
                (predicted_alpha > represented_alpha + 0.05).float(),
                deep_background,
            ),
            "foreground_false_negative_rate": masked_mean(
                (predicted_alpha + 0.05 < represented_alpha).float(),
                deep_foreground,
            ),
        }
        metrics["alpha_representation_floor_mae"] = F.l1_loss(
            represented_alpha, actual_alpha
        )
        return metrics

    @classmethod
    def _rgba_probe_features(
        cls,
        latents: torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (latents.float() - mean) / std
        limit = cls._RGBA_PROBE_FEATURE_LIMIT
        normalized = limit * torch.tanh(normalized / limit)
        local = F.avg_pool2d(normalized, kernel_size=3, stride=1, padding=1)
        return torch.cat((normalized, local), dim=1)

    @staticmethod
    def _rgba_alpha_edges(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            alpha[..., :, 1:] - alpha[..., :, :-1],
            alpha[..., 1:, :] - alpha[..., :-1, :],
        )

    @staticmethod
    def _rgba_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values[mask].mean() if mask.any() else values.mean()

    @classmethod
    def _rgba_balanced_pointwise_loss(
        cls,
        pointwise: torch.Tensor,
        target_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Weight empty canvas, subject interior and fine boundary equally."""

        binary = target_alpha.detach().clamp(0.0, 1.0) >= 0.5
        dilated = F.max_pool2d(binary.float(), kernel_size=3, stride=1, padding=1) > 0.5
        eroded = (
            1.0
            - F.max_pool2d(1.0 - binary.float(), kernel_size=3, stride=1, padding=1)
        ) > 0.5
        boundary = dilated & ~eroded
        deep_background = ~dilated
        deep_foreground = eroded
        return torch.stack((
            pointwise.mean(),
            cls._rgba_masked_mean(pointwise, deep_background),
            cls._rgba_masked_mean(pointwise, deep_foreground),
            cls._rgba_masked_mean(pointwise, boundary),
        )).mean()

    @classmethod
    def _rgba_balanced_alpha_loss(
        cls,
        predicted_alpha: torch.Tensor,
        target_alpha: torch.Tensor,
    ) -> torch.Tensor:
        pointwise = F.smooth_l1_loss(
            predicted_alpha,
            target_alpha,
            beta=0.1,
            reduction="none",
        )
        return cls._rgba_balanced_pointwise_loss(pointwise, target_alpha)

    @classmethod
    def _rgba_probe_carrier_loss(
        cls,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        target_alpha: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Prevent alpha-probe channels from cancelling one another.

        Matching only the scalar probe output permits equal and opposite errors
        in alpha-carrying latent features. Their scalar alpha looks correct while
        the VAE still decodes an opaque image. This term matches every feature in
        proportion to its contribution to alpha and ignores unrelated features.
        """

        weights = coefficients[:-1, 0].abs().view(1, -1, 1, 1)
        weight_sum = weights.sum().clamp_min(1.0e-6)
        feature_error = F.smooth_l1_loss(
            predicted_features,
            target_features,
            beta=0.1,
            reduction="none",
        )
        pointwise = (feature_error * weights).sum(dim=1, keepdim=True) / weight_sum
        return cls._rgba_balanced_pointwise_loss(pointwise, target_alpha)

    @classmethod
    def _rgba_balanced_edge_loss(
        cls,
        pred_dx: torch.Tensor,
        pred_dy: torch.Tensor,
        target_dx: torch.Tensor,
        target_dy: torch.Tensor,
        target_alpha: torch.Tensor,
    ) -> torch.Tensor:
        binary = target_alpha.detach().clamp(0.0, 1.0) >= 0.5
        dilated = F.max_pool2d(binary.float(), kernel_size=3, stride=1, padding=1) > 0.5
        eroded = (
            1.0
            - F.max_pool2d(1.0 - binary.float(), kernel_size=3, stride=1, padding=1)
        ) > 0.5
        boundary = dilated & ~eroded
        boundary_dx = boundary[..., :, 1:] | boundary[..., :, :-1]
        boundary_dy = boundary[..., 1:, :] | boundary[..., :-1, :]
        error_dx = F.smooth_l1_loss(
            pred_dx, target_dx, beta=0.05, reduction="none"
        )
        error_dy = F.smooth_l1_loss(
            pred_dy, target_dy, beta=0.05, reduction="none"
        )
        global_error = 0.5 * (error_dx.mean() + error_dy.mean())
        boundary_error = 0.5 * (
            cls._rgba_masked_mean(error_dx, boundary_dx)
            + cls._rgba_masked_mean(error_dy, boundary_dy)
        )
        return 0.5 * (global_error + boundary_error)

    @staticmethod
    def _rgba_smooth_normalized_loss(
        loss: torch.Tensor,
        baseline: torch.Tensor,
        soft_cap: float = 4.0,
    ) -> torch.Tensor:
        # A hard clamp used to make the gradient exactly zero once alpha was
        # sufficiently wrong. Log compression remains bounded in practice while
        # retaining a recovery gradient for arbitrarily bad predictions.
        ratio = loss / baseline
        return soft_cap * torch.log1p(ratio / soft_cap)

    def get_rgba_latent_auxiliary_loss(
        self,
        batch: "DataLoaderBatchDTO",
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise alpha-carrying components of the predicted flow velocity."""

        alpha_strength = self._rgba_loss_setting("alpha", 1.0)
        edge_strength = self._rgba_loss_setting("alpha_edge", 0.5)
        carrier_strength = self._rgba_loss_setting("alpha_carrier", 0.25)
        zero = pred.float().sum() * 0.0
        empty_metrics = {
            "rgba_alpha": zero.detach(),
            "rgba_alpha_edge": zero.detach(),
            "rgba_alpha_carrier": zero.detach(),
            "rgba_probe_r2": zero.detach(),
            "rgba_alpha_ratio": zero.detach(),
            "rgba_alpha_edge_ratio": zero.detach(),
            "rgba_alpha_carrier_ratio": zero.detach(),
            "rgba_base_scale": zero.detach(),
        }
        if alpha_strength == 0.0 and edge_strength == 0.0 and carrier_strength == 0.0:
            return zero, empty_metrics
        if getattr(self, "_rgba_alpha_probe_cpu", None) is None:
            # Other RGBA backends may not yet ship a calibrated fixed probe.
            # Keep their ordinary latent diffusion loss unchanged.
            return zero, empty_metrics

        rgba = batch.tensor
        clean_latents = batch.latents
        if (
            rgba is None
            or rgba.ndim != 4
            or rgba.shape[1] != 4
            or clean_latents is None
            or clean_latents.ndim != 4
            or pred.ndim != 4
        ):
            raise ValueError(
                "Transparent LoRA alpha consistency requires BCHW RGBA targets "
                "and BCHW clean latents"
            )

        clean = clean_latents.detach().to(device=pred.device, dtype=torch.float32)
        alpha = (rgba[:, 3:4].detach().to(dtype=torch.float32) + 1.0) * 0.5
        alpha = F.interpolate(alpha, size=clean.shape[-2:], mode="area")
        alpha = alpha.to(device=pred.device, dtype=torch.float32, non_blocking=True)
        alpha = alpha.clamp(0.0, 1.0)

        probe = self._rgba_alpha_probe(pred.device)
        mean = probe["mean"]
        std = probe["std"]
        coefficients = probe["coefficients"]
        with torch.no_grad():
            clean_features = self._rgba_probe_features(clean, mean=mean, std=std)
            target_alpha = torch.einsum(
                "bchw,co->bohw",
                clean_features,
                coefficients[:-1],
            ) + coefficients[-1].view(1, 1, 1, 1)
        # For rectified flow, target = noise - clean.  The previous loss used
        # noisy - sigma * pred, whose error is sigma * (target - pred).  It was
        # therefore almost zero, with an almost zero gradient, at low-noise
        # timesteps even for an untrained model.  Evaluate the same clean-latent
        # direction in velocity units instead.  It equals `clean` exactly when
        # pred == target and has the same strength at every timestep.
        velocity_implied_clean = clean + target.float() - pred.float()
        predicted_features = self._rgba_probe_features(
            velocity_implied_clean,
            mean=mean,
            std=std,
        )
        predicted_alpha = torch.einsum(
            "bchw,co->bohw",
            predicted_features,
            coefficients[:-1],
        ) + coefficients[-1].view(1, 1, 1, 1)

        # Compare against the probe readout of the exact target latent, not the
        # raw mask. Therefore the auxiliary loss is exactly zero for a correct
        # diffusion prediction and cannot pull latents away from the VAE target.
        alpha_loss = self._rgba_balanced_alpha_loss(predicted_alpha, target_alpha)
        pred_dx, pred_dy = self._rgba_alpha_edges(predicted_alpha)
        target_dx, target_dy = self._rgba_alpha_edges(target_alpha)
        edge_loss = self._rgba_balanced_edge_loss(
            pred_dx,
            pred_dy,
            target_dx,
            target_dy,
            target_alpha,
        )
        carrier_loss = self._rgba_probe_carrier_loss(
            predicted_features,
            clean_features,
            target_alpha,
            coefficients,
        )

        with torch.no_grad():
            alpha_variance = alpha.var(unbiased=False).clamp_min(1.0e-6)
            fit_quality = (
                1.0 - F.mse_loss(target_alpha, alpha) / alpha_variance
            ).clamp(0.0, 1.0)

        # The 1/0.5 defaults keep ordinary diffusion MSE dominant: badly wrong
        # alpha contributes about a quarter base-loss unit, with a smaller edge
        # term. Detached normalization keeps this stable across timesteps and
        # quantization backends.
        base_scale = F.mse_loss(pred.float(), target.float()).detach().clamp(0.1, 0.5)
        alpha_mean = target_alpha.mean(dim=(2, 3), keepdim=True).expand_as(target_alpha)
        alpha_baseline = self._rgba_balanced_alpha_loss(
            alpha_mean, target_alpha
        ).detach().clamp_min(0.05)
        alpha_ratio = alpha_loss / alpha_baseline
        normalized_alpha = self._rgba_smooth_normalized_loss(
            alpha_loss, alpha_baseline
        )

        edge_baseline = self._rgba_balanced_edge_loss(
            torch.zeros_like(target_dx),
            torch.zeros_like(target_dy),
            target_dx,
            target_dy,
            target_alpha,
        ).detach().clamp_min(0.01)
        edge_ratio = edge_loss / edge_baseline
        normalized_edge = self._rgba_smooth_normalized_loss(
            edge_loss, edge_baseline
        )

        mean_features = clean_features.mean(dim=(2, 3), keepdim=True).expand_as(
            clean_features
        )
        carrier_baseline = self._rgba_probe_carrier_loss(
            mean_features,
            clean_features,
            target_alpha,
            coefficients,
        ).detach().clamp_min(0.05)
        carrier_ratio = carrier_loss / carrier_baseline
        normalized_carrier = self._rgba_smooth_normalized_loss(
            carrier_loss, carrier_baseline
        )

        # Edit batches additionally receive their ordinary exact-latent loss.
        # Generation batches receive only these alpha-specific terms, avoiding
        # target-image memorization. All operations reuse the existing forward.
        auxiliary = base_scale * (
            alpha_strength * normalized_alpha
            + edge_strength * normalized_edge
            + carrier_strength * normalized_carrier
        )
        metrics = {
            "rgba_alpha": alpha_loss.detach(),
            "rgba_alpha_edge": edge_loss.detach(),
            "rgba_alpha_carrier": carrier_loss.detach(),
            "rgba_probe_r2": fit_quality.detach(),
            "rgba_alpha_ratio": alpha_ratio.detach(),
            "rgba_alpha_edge_ratio": edge_ratio.detach(),
            "rgba_alpha_carrier_ratio": carrier_ratio.detach(),
            "rgba_base_scale": base_scale.detach(),
        }
        return auxiliary, metrics
