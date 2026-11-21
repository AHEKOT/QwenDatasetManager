import inspect
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from extensions.rgba_training.lora_loss import RGBALoRALossMixin
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class _DummyRGBAModel(RGBALoRALossMixin):
    def __init__(self, model_kwargs=None):
        self.model_config = SimpleNamespace(model_kwargs=model_kwargs or {})
        coefficients = torch.zeros((9, 1), dtype=torch.float32)
        coefficients[0, 0] = 1.0
        self.set_rgba_alpha_probe({
            "mean": torch.zeros((1, 4, 1, 1), dtype=torch.float32),
            "std": torch.ones((1, 4, 1, 1), dtype=torch.float32),
            "coefficients": coefficients,
        })


class RGBALatentLossTests(unittest.TestCase):
    @staticmethod
    def _make_case():
        alpha = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        alpha[:, :, 2:6, 2:6] = 1.0
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 8),
            torch.linspace(-1, 1, 8),
            indexing="ij",
        )
        clean = torch.stack(
            (alpha[0, 0], xx, yy, xx * yy),
            dim=0,
        ).unsqueeze(0)
        noise = torch.randn_like(clean)
        sigma = 0.5
        noisy = (1.0 - sigma) * clean + sigma * noise
        target = noise - clean
        rgba = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
        rgba[:, 3:4] = F.interpolate(alpha, size=(64, 64), mode="nearest") * 2.0 - 1.0
        batch = SimpleNamespace(
            tensor=rgba,
            latents=clean,
            dataset_config=SimpleNamespace(
                rgba_generate_control=True,
                rgba_control_mode="edit",
            ),
        )
        timesteps = torch.tensor([sigma * 1000.0])
        return batch, clean, noisy, target, timesteps

    def test_auxiliary_loss_rewards_correct_alpha(self):
        model = _DummyRGBAModel({
            "rgba_lora_loss_alpha": 4,
            "rgba_lora_loss_alpha_edge": 2,
        })
        batch, clean, noisy, target, timesteps = self._make_case()
        correct_loss, correct_metrics = model.get_rgba_latent_auxiliary_loss(
            batch,
            pred=target,
            target=target,
            noisy_latents=noisy,
            timesteps=timesteps,
        )
        wrong_clean = clean.clone()
        wrong_clean[:, 0] = 1.0 - wrong_clean[:, 0]
        wrong_pred = (noisy - wrong_clean) / 0.5
        wrong_loss, wrong_metrics = model.get_rgba_latent_auxiliary_loss(
            batch,
            pred=wrong_pred,
            target=target,
            noisy_latents=noisy,
            timesteps=timesteps,
        )

        self.assertGreater(wrong_loss.item(), correct_loss.item() * 5.0)
        self.assertGreater(correct_metrics["rgba_probe_r2"].item(), 0.9)
        self.assertGreater(wrong_metrics["rgba_alpha"].item(), correct_metrics["rgba_alpha"].item())

    def test_auxiliary_loss_backpropagates_to_single_prediction(self):
        model = _DummyRGBAModel()
        batch, clean, noisy, target, timesteps = self._make_case()
        wrong_clean = clean.clone()
        wrong_clean[:, 0] = 1.0 - wrong_clean[:, 0]
        pred = ((noisy - wrong_clean) / 0.5).requires_grad_(True)
        loss, _ = model.get_rgba_latent_auxiliary_loss(
            batch,
            pred=pred,
            target=target,
            noisy_latents=noisy,
            timesteps=timesteps,
        )
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)

    def test_edit_uses_exact_latent_loss_while_generation_stays_alpha_only(self):
        model = _DummyRGBAModel()
        batch, clean, noisy, target, timesteps = self._make_case()
        self.assertEqual(model.get_rgba_diffusion_loss_weight(batch), 1.0)

        wrong_clean = clean.clone()
        wrong_clean[:, 0] = 1.0 - wrong_clean[:, 0]
        pred = ((noisy - wrong_clean) / 0.5).requires_grad_(True)
        batch.dataset_config.rgba_control_mode = "generation"
        self.assertEqual(model.get_rgba_diffusion_loss_weight(batch), 0.0)
        alpha_only_loss, _ = model.get_rgba_latent_auxiliary_loss(
            batch,
            pred=pred,
            target=target,
            noisy_latents=noisy,
            timesteps=timesteps,
        )
        alpha_only_loss.backward()
        self.assertGreater(alpha_only_loss.item(), 0.0)
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)

    def test_alpha_objective_does_not_disappear_at_low_noise(self):
        model = _DummyRGBAModel()
        batch, clean, _noisy, target, _timesteps = self._make_case()
        wrong_pred = target.clone()
        wrong_pred[:, 0] = wrong_pred[:, 0] + 0.75
        losses = []
        gradients = []
        for sigma in (0.01, 0.5, 0.99):
            noise = target + clean
            noisy = (1.0 - sigma) * clean + sigma * noise
            pred = wrong_pred.clone().requires_grad_(True)
            loss, _metrics = model.get_rgba_latent_auxiliary_loss(
                batch,
                pred=pred,
                target=target,
                noisy_latents=noisy,
                timesteps=torch.tensor([sigma * 1000.0]),
            )
            loss.backward()
            losses.append(loss.item())
            gradients.append(pred.grad.abs().sum().item())

        self.assertAlmostEqual(min(losses), max(losses), places=6)
        self.assertAlmostEqual(min(gradients), max(gradients), places=6)

    def test_carrier_loss_catches_probe_feature_cancellation(self):
        target_features = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        predicted_features = torch.ones_like(target_features)
        target_alpha = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        coefficients = torch.tensor([[1.0], [-1.0], [0.0]])

        scalar_target = torch.einsum(
            "bchw,co->bohw", target_features, coefficients[:-1]
        )
        scalar_predicted = torch.einsum(
            "bchw,co->bohw", predicted_features, coefficients[:-1]
        )
        self.assertEqual(
            _DummyRGBAModel._rgba_balanced_alpha_loss(
                scalar_predicted, scalar_target
            ).item(),
            0.0,
        )
        self.assertGreater(
            _DummyRGBAModel._rgba_probe_carrier_loss(
                predicted_features,
                target_features,
                target_alpha,
                coefficients,
            ).item(),
            0.0,
        )

    def test_normalization_retains_gradient_above_old_hard_cap(self):
        raw_loss = torch.tensor(10.0, requires_grad=True)
        normalized = _DummyRGBAModel._rgba_smooth_normalized_loss(
            raw_loss, torch.tensor(1.0)
        )
        normalized.backward()

        self.assertGreater(normalized.item(), 4.0)
        self.assertIsNotNone(raw_loss.grad)
        self.assertGreater(raw_loss.grad.item(), 0.0)

    def test_validation_metrics_compare_prediction_to_actual_png_alpha(self):
        model = _DummyRGBAModel()
        batch, clean, _noisy, _target, _timesteps = self._make_case()
        correct = model.get_rgba_validation_metrics(
            predicted_clean=clean,
            target_clean=clean,
            target_rgba=batch.tensor,
        )
        wrong_clean = clean.clone()
        wrong_clean[:, 0] = 1.0 - wrong_clean[:, 0]
        wrong = model.get_rgba_validation_metrics(
            predicted_clean=wrong_clean,
            target_clean=clean,
            target_rgba=batch.tensor,
        )

        self.assertLess(correct["alpha_mae"].item(), 0.02)
        self.assertGreater(correct["alpha_iou"].item(), 0.99)
        self.assertGreater(wrong["alpha_mae"].item(), correct["alpha_mae"].item() * 10)
        self.assertLess(wrong["alpha_iou"].item(), 0.1)
        self.assertIn("alpha_representation_floor_mae", correct)
        self.assertIn("representation_boundary_mae", correct)
        self.assertIn("background_false_positive_rate", correct)

    def test_rgba_pass_thresholds_use_worst_case_strict_checks(self):
        thresholds = {
            "max": {
                "representation_alpha_mae": 0.015,
                "background_false_positive_rate": 0.001,
            },
            "min": {"representation_alpha_iou": 0.985},
        }
        passed, checks = BaseSDTrainProcess._evaluate_rgba_validation_pass(
            {
                "representation_alpha_mae": 0.014,
                "background_false_positive_rate": 0.0009,
                "representation_alpha_iou": 0.986,
            },
            thresholds,
        )
        self.assertTrue(passed)
        self.assertTrue(all(check[-1] for check in checks))

        failed, checks = BaseSDTrainProcess._evaluate_rgba_validation_pass(
            {
                "representation_alpha_mae": 0.014,
                "background_false_positive_rate": 0.0011,
                "representation_alpha_iou": 0.986,
            },
            thresholds,
        )
        self.assertFalse(failed)
        self.assertEqual(
            [name for name, _value, _operator, _limit, ok in checks if not ok],
            ["background_false_positive_rate"],
        )

    def test_auxiliary_loss_rejects_non_rgba_target(self):
        model = _DummyRGBAModel()
        with self.assertRaisesRegex(ValueError, "BCHW RGBA"):
            model.get_rgba_latent_auxiliary_loss(
                SimpleNamespace(
                    tensor=torch.zeros(1, 3, 32, 32),
                    latents=torch.zeros(1, 4, 4, 4),
                ),
                pred=torch.zeros(1, 4, 4, 4),
                target=torch.zeros(1, 4, 4, 4),
                noisy_latents=torch.zeros(1, 4, 4, 4),
                timesteps=torch.tensor([500.0]),
            )

    def test_training_loop_has_no_rgba_model_or_vae_extra_pass(self):
        init_source = inspect.getsource(SDTrainer.__init__)
        loss_source = inspect.getsource(SDTrainer.calculate_loss)
        process_init_source = inspect.getsource(BaseSDTrainProcess.__init__)
        validation_setup_source = inspect.getsource(BaseSDTrainProcess.setup_validation)
        validation_source = inspect.getsource(BaseSDTrainProcess.validate)

        self.assertNotIn("do_rgba_generation_preservation", init_source)
        self.assertNotIn("get_rgba_training_loss", loss_source)
        self.assertNotIn("get_rgba_generation_preservation_weight", loss_source)
        self.assertIn("get_rgba_latent_auxiliary_loss", loss_source)
        self.assertIn("get_rgba_diffusion_loss_weight", loss_source)
        self.assertNotIn("get_rgba_latent_loss_multiplier", loss_source)
        self.assertIn(
            "if self.train_config.unload_text_encoder:",
            process_init_source,
        )
        self.assertIn(
            "self.train_config.cache_text_embeddings = True",
            process_init_source,
        )
        self.assertIn("supports_rgba_latent_loss", validation_setup_source)
        self.assertIn("prepare_rgba_validation_pair", validation_setup_source)
        self.assertIn("control_images", validation_setup_source)
        self.assertIn("batch=validation_batch", validation_source)
        self.assertIn("get_rgba_validation_metrics", validation_source)


if __name__ == "__main__":
    unittest.main()
