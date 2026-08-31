import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file

from toolkit.sampling_lora import _sampling_lora_metadata, validate_sampling_lora_path


class _QwenModel:
    @staticmethod
    def convert_lora_weights_before_load(state):
        return state


class _FluxModel:
    @staticmethod
    def convert_lora_weights_before_load(state):
        return OrderedDict(
            (key.replace("diffusion_model.", "transformer."), value)
            for key, value in state.items()
        )


class SamplingLoRAMetadataTests(unittest.TestCase):
    def test_qwen_lightning_layout_preserves_rank_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen.safetensors"
            save_file({
                "transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros(4, 8),
                "transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros(8, 4),
                "transformer_blocks.0.attn.to_q.alpha": torch.tensor(2.0),
            }, str(path))

            self.assertEqual(validate_sampling_lora_path(str(path)), str(path.resolve()))
            dims, alphas, native = _sampling_lora_metadata(_QwenModel(), str(path))

            key = "transformer$$transformer_blocks$$0$$attn$$to_q"
            self.assertEqual(dims[key], 4)
            self.assertEqual(alphas[key], 2.0)
            self.assertTrue(native)

    def test_flux_peft_layout_is_normalized_to_native_transformer_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "klein.safetensors"
            save_file({
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.zeros(8, 16),
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.zeros(16, 8),
            }, str(path))

            dims, alphas, native = _sampling_lora_metadata(_FluxModel(), str(path))

            key = "transformer$$double_blocks$$0$$img_attn$$qkv"
            self.assertEqual(dims[key], 8)
            self.assertEqual(alphas[key], 8.0)
            self.assertFalse(native)


if __name__ == "__main__":
    unittest.main()
