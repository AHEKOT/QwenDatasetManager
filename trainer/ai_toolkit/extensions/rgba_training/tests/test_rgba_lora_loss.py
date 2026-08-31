import inspect
import unittest
from types import SimpleNamespace

import torch

from extensions.rgba_training.lora_loss import RGBALoRALossMixin
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class _DummyRGBAModel(RGBALoRALossMixin):
    def __init__(self, model_kwargs=None):
        self.model_config = SimpleNamespace(model_kwargs=model_kwargs or {})


class RGBALatentLossTests(unittest.TestCase):
    def test_multiplier_balances_alpha_regions_and_keeps_mean_one(self):
        model = _DummyRGBAModel({
            "rgba_lora_loss_alpha": 4,
            "rgba_lora_loss_alpha_edge": 2,
        })
        target = torch.full((1, 4, 64, 64), -1.0)
        target[:, 3, 20:44, 24:40] = 1.0

        weight = model.get_rgba_latent_loss_multiplier(
            SimpleNamespace(tensor=target),
            size=(8, 8),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(weight.shape), (1, 1, 8, 8))
        torch.testing.assert_close(weight.mean(), torch.tensor(1.0))
        self.assertGreater(weight.max().item(), weight.min().item())

    def test_multiplier_rejects_non_rgba_target(self):
        model = _DummyRGBAModel()
        with self.assertRaisesRegex(ValueError, "BCHW RGBA"):
            model.get_rgba_latent_loss_multiplier(
                SimpleNamespace(tensor=torch.zeros(1, 3, 32, 32)),
                size=(4, 4),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_training_loop_has_no_rgba_model_or_vae_extra_pass(self):
        init_source = inspect.getsource(SDTrainer.__init__)
        loss_source = inspect.getsource(SDTrainer.calculate_loss)

        self.assertNotIn("do_rgba_generation_preservation", init_source)
        self.assertNotIn("get_rgba_training_loss", loss_source)
        self.assertNotIn("get_rgba_generation_preservation_weight", loss_source)
        self.assertIn("get_rgba_latent_loss_multiplier", loss_source)


if __name__ == "__main__":
    unittest.main()
