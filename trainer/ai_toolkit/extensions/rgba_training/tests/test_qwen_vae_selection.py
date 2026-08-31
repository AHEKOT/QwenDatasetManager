import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from extensions.rgba_training.qwen_compat import (
    QWEN_IMAGE_LATENTS_MEAN,
    QWEN_IMAGE_LATENTS_STD,
)
from extensions.rgba_training.qwen_image_edit_plus_rgba import (
    QwenImageEditPlusRGBAModel,
    validate_qwen_sampling_lora,
)
from extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusModel
from toolkit.models.base_model import BaseModel


class QwenRGBAVAESelectionTests(unittest.TestCase):
    def test_rgba_arch_loads_trained_rgba_vae_instead_of_standard_qie_vae(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rgba_path = str(Path(temp_dir.name))
        (Path(rgba_path) / "config.json").write_text("{}", encoding="utf-8")
        standard_path = "Qwen/Qwen-Image-Edit-2511"
        fake_vae = SimpleNamespace(
            config=SimpleNamespace(
                input_channels=4,
                z_dim=16,
                latents_mean=QWEN_IMAGE_LATENTS_MEAN,
                latents_std=QWEN_IMAGE_LATENTS_STD,
            )
        )
        model = object.__new__(QwenImageEditPlusRGBAModel)
        model.model_config = SimpleNamespace(
            vae_path=rgba_path,
            model_kwargs={"rgba_vae_subfolder": "vae"},
        )
        model.vae_torch_dtype = torch.bfloat16
        model.print_and_status_update = lambda *_args, **_kwargs: None

        with patch(
            "extensions.rgba_training.qwen_image_edit_plus_rgba."
            "AutoencoderKLQwenImage.from_pretrained",
            return_value=fake_vae,
        ) as from_pretrained:
            loaded = model._load_qwen_vae(standard_path, torch.bfloat16)

        self.assertIs(loaded, fake_vae)
        from_pretrained.assert_called_once_with(
            rgba_path,
            torch_dtype=torch.bfloat16,
        )
        self.assertNotIn(standard_path, from_pretrained.call_args.args)

    def test_sampling_lora_validation_accepts_qwen_transformer_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lightning.safetensors"
            save_file(
                {
                    "transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros(4, 8),
                    "transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros(8, 4),
                    "transformer_blocks.0.attn.to_q.alpha": torch.tensor(2.0),
                },
                str(path),
            )
            self.assertEqual(validate_qwen_sampling_lora(str(path)), str(path.resolve()))

    def test_sampling_lora_native_network_is_only_active_during_generation(self):
        model = object.__new__(QwenImageEditPlusRGBAModel)
        model.sample_lora_path = r"D:\models\lightning.safetensors"
        model.print_and_status_update = lambda *_args, **_kwargs: None
        pipeline = MagicMock()
        sampling_network = MagicMock()
        sampling_network.__enter__.return_value = sampling_network
        model._ensure_sampling_lora_network = MagicMock(return_value=sampling_network)
        image_config = SimpleNamespace(num_inference_steps=25, guidance_scale=3.0)

        with patch.object(BaseModel, "generate_images", return_value="generated") as parent:
            result = model.generate_images([image_config], pipeline=pipeline)

        self.assertEqual(result, "generated")
        self.assertEqual(image_config.num_inference_steps, 4)
        self.assertEqual(image_config.guidance_scale, 1.0)
        self.assertEqual(image_config.output_ext, "png")
        model._ensure_sampling_lora_network.assert_called_once_with()
        sampling_network.__enter__.assert_called_once_with()
        sampling_network.__exit__.assert_called_once_with(None, None, None)
        pipeline.load_lora_weights.assert_not_called()

    def test_rgba_generation_forces_png_without_sampling_lora(self):
        model = object.__new__(QwenImageEditPlusRGBAModel)
        model.sample_lora_path = None
        image_config = SimpleNamespace(
            output_ext="jpg",
            output_path=r"D:\samples\baseline_0.jpg",
        )

        with patch.object(BaseModel, "generate_images", return_value="generated") as parent:
            result = model.generate_images([image_config])

        self.assertEqual(result, "generated")
        self.assertEqual(image_config.output_ext, "png")
        self.assertEqual(image_config.output_path, r"D:\samples\baseline_0.png")
        parent.assert_called_once()
        parent.assert_called_once()

    def test_sampling_lora_native_network_is_disabled_when_generation_fails(self):
        model = object.__new__(QwenImageEditPlusRGBAModel)
        model.sample_lora_path = r"D:\models\lightning.safetensors"
        model.print_and_status_update = lambda *_args, **_kwargs: None
        pipeline = MagicMock()
        sampling_network = MagicMock()
        sampling_network.__enter__.return_value = sampling_network
        model._ensure_sampling_lora_network = MagicMock(return_value=sampling_network)
        image_config = SimpleNamespace(num_inference_steps=25, guidance_scale=3.0)

        with patch.object(BaseModel, "generate_images", side_effect=RuntimeError("sample failed")):
            with self.assertRaisesRegex(RuntimeError, "sample failed"):
                model.generate_images([image_config], pipeline=pipeline)

        sampling_network.__exit__.assert_called_once()
        pipeline.load_lora_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
