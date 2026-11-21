import unittest

import torch

from extensions.rgba_training.comfy_vae_export import (
    convert_diffusers_qwen_vae_to_comfy,
    diffusers_qwen_vae_key_to_comfy,
)


class ComfyVAEExportTests(unittest.TestCase):
    def test_representative_key_mapping(self):
        expected = {
            "encoder.conv_in.weight": "encoder.conv1.weight",
            "encoder.mid_block.resnets.1.conv2.bias": "encoder.middle.2.residual.6.bias",
            "encoder.down_blocks.3.conv_shortcut.weight": "encoder.downsamples.3.shortcut.weight",
            "decoder.up_blocks.1.resnets.0.conv_shortcut.bias": "decoder.upsamples.4.shortcut.bias",
            "decoder.up_blocks.2.upsamplers.0.resample.1.weight": "decoder.upsamples.11.resample.1.weight",
            "decoder.conv_out.weight": "decoder.head.2.weight",
            "post_quant_conv.bias": "conv2.bias",
        }
        for source, target in expected.items():
            self.assertEqual(diffusers_qwen_vae_key_to_comfy(source), target)

    def test_rgba_contract_and_dtype(self):
        state = {
            "encoder.conv_in.weight": torch.zeros(2, 4, 3, 3, 3),
            "decoder.conv_out.weight": torch.zeros(4, 2, 3, 3, 3),
            "decoder.conv_out.bias": torch.zeros(4),
        }
        converted = convert_diffusers_qwen_vae_to_comfy(state)
        self.assertEqual(converted["encoder.conv1.weight"].dtype, torch.bfloat16)
        self.assertEqual(tuple(converted["decoder.head.2.weight"].shape), (4, 2, 3, 3, 3))

    def test_rejects_rgb_vae(self):
        state = {
            "encoder.conv_in.weight": torch.zeros(2, 3, 3, 3, 3),
            "decoder.conv_out.weight": torch.zeros(3, 2, 3, 3, 3),
            "decoder.conv_out.bias": torch.zeros(3),
        }
        with self.assertRaisesRegex(ValueError, "RGBA Qwen VAE"):
            convert_diffusers_qwen_vae_to_comfy(state)


if __name__ == "__main__":
    unittest.main()
