import unittest
from types import SimpleNamespace

from extensions.rgba_training.qwen_compat import (
    QWEN_IMAGE_LATENTS_MEAN,
    QWEN_IMAGE_LATENTS_STD,
    validate_qie2511_transformer_config,
    validate_qwen_rgba_vae_config,
)


class QwenRGBACompatibilityTests(unittest.TestCase):
    def test_qwen_layered_contract_is_accepted(self):
        config = SimpleNamespace(
            input_channels=4,
            z_dim=16,
            latents_mean=QWEN_IMAGE_LATENTS_MEAN,
            latents_std=QWEN_IMAGE_LATENTS_STD,
        )
        validate_qwen_rgba_vae_config(config)

    def test_rgb_qwen_vae_is_rejected(self):
        config = SimpleNamespace(
            input_channels=3,
            z_dim=16,
            latents_mean=QWEN_IMAGE_LATENTS_MEAN,
            latents_std=QWEN_IMAGE_LATENTS_STD,
        )
        with self.assertRaisesRegex(ValueError, "input_channels=4"):
            validate_qwen_rgba_vae_config(config)

    def test_qie2511_transformer_contract(self):
        validate_qie2511_transformer_config(
            SimpleNamespace(in_channels=64, out_channels=16, patch_size=2)
        )


if __name__ == "__main__":
    unittest.main()

