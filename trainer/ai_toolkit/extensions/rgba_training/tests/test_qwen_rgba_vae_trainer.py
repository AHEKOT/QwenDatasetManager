import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from extensions.rgba_training.qwen_rgba_vae_trainer import (
    AlphaBoundaryGuard,
    FullRGBAVAEFineTune,
    expand_qwen_vae_config_to_rgba,
    expand_qwen_vae_state_dict_to_rgba,
    split_rgba_files,
)
from extensions.rgba_training.vae_metrics import (
    evaluate_readiness,
    normalize_readiness_thresholds,
    rgba_reconstruction_metrics,
)


class _Part(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv_in = conv
        self.conv_out = conv


class _TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv3d(4, 2, 1)
        self.decoder = nn.Module()
        self.decoder.conv_out = nn.Conv3d(2, 4, 1)


class QwenRGBAVAETrainerTests(unittest.TestCase):
    def test_rgba_config_removes_diffusers_default_value_marker(self):
        source = {
            "base_dim": 96,
            "input_channels": 3,
            "_use_default_values": ["input_channels", "attn_scales"],
        }
        expanded = expand_qwen_vae_config_to_rgba(source)
        self.assertEqual(expanded["input_channels"], 4)
        self.assertNotIn("_use_default_values", expanded)
        self.assertEqual(source["input_channels"], 3)

    def test_rgb_state_expansion_preserves_rgb_and_adds_alpha(self):
        state = {
            "encoder.conv_in.weight": torch.arange(2 * 3 * 1 * 1 * 1).reshape(2, 3, 1, 1, 1).float(),
            "decoder.conv_out.weight": torch.arange(3 * 2 * 1 * 1 * 1).reshape(3, 2, 1, 1, 1).float(),
            "decoder.conv_out.bias": torch.tensor([0.1, 0.2, 0.3]),
            "unchanged": torch.tensor([7.0]),
        }
        expanded = expand_qwen_vae_state_dict_to_rgba(state)
        self.assertEqual(tuple(expanded["encoder.conv_in.weight"].shape), (2, 4, 1, 1, 1))
        self.assertEqual(tuple(expanded["decoder.conv_out.weight"].shape), (4, 2, 1, 1, 1))
        torch.testing.assert_close(expanded["encoder.conv_in.weight"][:, :3], state["encoder.conv_in.weight"])
        torch.testing.assert_close(expanded["encoder.conv_in.weight"][:, 3], torch.zeros((2, 1, 1, 1)))
        torch.testing.assert_close(expanded["decoder.conv_out.weight"][:3], state["decoder.conv_out.weight"])
        torch.testing.assert_close(expanded["decoder.conv_out.weight"][3], torch.zeros((2, 1, 1, 1)))
        torch.testing.assert_close(expanded["decoder.conv_out.bias"][:3], state["decoder.conv_out.bias"])
        self.assertEqual(expanded["decoder.conv_out.bias"][3].item(), 1.0)
        torch.testing.assert_close(expanded["unchanged"], state["unchanged"])

    def test_boundary_guard_only_allows_alpha_slices_to_change(self):
        vae = _TinyVAE()
        guard = AlphaBoundaryGuard(vae)
        encoder_rgb = vae.encoder.conv_in.weight[:, :3].detach().clone()
        decoder_rgb = vae.decoder.conv_out.weight[:3].detach().clone()
        loss = vae.encoder.conv_in.weight.sum() + vae.decoder.conv_out.weight.sum() + vae.decoder.conv_out.bias.sum()
        loss.backward()
        self.assertEqual(vae.encoder.conv_in.weight.grad[:, :3].abs().sum().item(), 0.0)
        self.assertGreater(vae.encoder.conv_in.weight.grad[:, 3:].abs().sum().item(), 0.0)
        self.assertEqual(vae.decoder.conv_out.weight.grad[:3].abs().sum().item(), 0.0)
        self.assertGreater(vae.decoder.conv_out.weight.grad[3:].abs().sum().item(), 0.0)

        with torch.no_grad():
            vae.encoder.conv_in.weight[:, :3].add_(10)
            vae.decoder.conv_out.weight[:3].add_(10)
        guard.restore_rgb()
        torch.testing.assert_close(vae.encoder.conv_in.weight[:, :3], encoder_rgb)
        torch.testing.assert_close(vae.decoder.conv_out.weight[:3], decoder_rgb)

    def test_full_finetune_enables_the_complete_vae(self):
        vae = _TinyVAE()
        controller = FullRGBAVAEFineTune(vae)
        self.assertEqual(len(controller.parameters), len(list(vae.parameters())))
        self.assertTrue(all(parameter.requires_grad for parameter in vae.parameters()))

    def test_full_finetune_applies_alpha_lr_multiplier_only_to_alpha_slices(self):
        vae = _TinyVAE()
        controller = FullRGBAVAEFineTune(vae, alpha_lr_multiplier=10)
        encoder_rgb = controller.encoder_weight[:, :3].detach().clone()
        encoder_alpha = controller.encoder_weight[:, 3:4].detach().clone()
        controller.prepare_step()
        with torch.no_grad():
            controller.encoder_weight.add_(0.01)
        controller.restore_rgb()
        torch.testing.assert_close(controller.encoder_weight[:, :3], encoder_rgb + 0.01)
        torch.testing.assert_close(controller.encoder_weight[:, 3:4], encoder_alpha + 0.1)

    def test_boundary_guard_projects_encoder_alpha_kernel_to_zero_dc(self):
        vae = _TinyVAE()
        vae.encoder.conv_in = nn.Conv3d(4, 2, 3, padding=1)
        guard = AlphaBoundaryGuard(vae, zero_dc_alpha_encoder=True)
        with torch.no_grad():
            vae.encoder.conv_in.weight[:, 3:4].normal_()
        guard.restore_rgb()
        spatial_sums = vae.encoder.conv_in.weight[:, 3:4].sum(dim=(-1, -2))
        torch.testing.assert_close(spatial_sums, torch.zeros_like(spatial_sums), atol=1e-6, rtol=0)

    def test_identical_rgba_roundtrip_passes_default_metrics(self):
        target = torch.zeros((2, 4, 16, 16))
        target[:, 3] = 1
        metrics = rgba_reconstruction_metrics(target, target, opaque_latent_rmse=0.0)
        passed, checks = evaluate_readiness(metrics)
        self.assertTrue(passed)
        self.assertTrue(all(check["passed"] for check in checks.values()))

    def test_alpha_failure_is_reported_even_when_rgb_matches(self):
        target = torch.zeros((1, 4, 16, 16))
        target[:, 3] = 1
        prediction = target.clone()
        prediction[:, 3] = -1
        metrics = rgba_reconstruction_metrics(prediction, target, opaque_latent_rmse=0.0)
        passed, checks = evaluate_readiness(metrics)
        self.assertFalse(passed)
        self.assertFalse(checks["alpha_mae"]["passed"])
        self.assertFalse(checks["alpha_iou"]["passed"])

    def test_validation_reports_alpha_coverage_diagnostics(self):
        target = torch.full((1, 4, 4, 4), -1.0)
        target[:, :3] = 0
        target[:, 3, :2] = 1
        prediction = target.clone()
        metrics = rgba_reconstruction_metrics(prediction, target, opaque_latent_rmse=0.0)
        self.assertAlmostEqual(metrics["background_alpha_mean"], 0.0)
        self.assertAlmostEqual(metrics["foreground_alpha_mean"], 1.0)
        self.assertAlmostEqual(metrics["predicted_transparent_fraction"], 0.5)

    def test_threshold_overrides_keep_metric_direction(self):
        thresholds = normalize_readiness_thresholds({"alpha_mae": 0.2, "alpha_iou": 0.8})
        self.assertEqual(thresholds["alpha_mae"], ("<=", 0.2))
        self.assertEqual(thresholds["alpha_iou"], (">=", 0.8))

    def test_validation_split_is_stable_and_disjoint(self):
        files = [f"image_{index:03d}.png" for index in range(100)]
        first_train, first_validation = split_rgba_files(
            files,
            validation_fraction=0.1,
            validation_max_images=20,
            validation_min_images=4,
        )
        second_train, second_validation = split_rgba_files(
            list(reversed(files)),
            validation_fraction=0.1,
            validation_max_images=20,
            validation_min_images=4,
        )
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(len(first_validation), 10)
        self.assertFalse(set(first_train).intersection(first_validation))


if __name__ == "__main__":
    unittest.main()
