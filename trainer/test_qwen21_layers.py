"""CPU regression tests for PSD semantics and cross-layer attention; no weights.

Written with the implementation, not executed during the user's no-run session.
Requires the trainer dependencies, including psd-tools[composite].
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))

import numpy as np
import torch
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import BlendMode
from toolkit.config_modules import DatasetConfig
from toolkit.layered_psd import read_psd_layers, save_layered_psd, load_psd_target, composite_layers
from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import run_transformer
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel
from extensions_built_in.diffusion_models.qwen_image_2.layered import QwenImage2LayeredModel
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model


class PSDLayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'target.psd'

    def test_order_offsets_opacity_hidden_and_padding(self):
        psd = PSDImage.new('RGB', (32, 32))
        psd.create_pixel_layer(Image.new('RGBA', (32, 32), (255, 0, 0, 255)), name='bottom')
        top = psd.create_pixel_layer(Image.new('RGBA', (8, 8), (0, 0, 255, 255)), name='top', left=4, top=6)
        top.opacity = 128
        hidden = psd.create_pixel_layer(Image.new('RGBA', (32, 32), (0, 255, 0, 255)), name='hidden')
        hidden.visible = False
        psd.save(self.path)
        layers, names = read_psd_layers(self.path, 3)
        self.assertEqual(names, ['bottom', 'top', '<empty>'])
        self.assertEqual(layers[0].getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(layers[1].getpixel((0, 0)), (0, 0, 0, 0))
        self.assertAlmostEqual(layers[1].getpixel((4, 6))[3], 128, delta=1)
        self.assertEqual(layers[2].getchannel('A').getbbox(), None)
        merged = composite_layers(layers)
        self.assertAlmostEqual(merged.getpixel((4, 6))[2], 128, delta=1)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            read_psd_layers(self.path, 1)

    def test_unsupported_blend_fails_with_layer_name(self):
        psd = PSDImage.new('RGB', (32, 32))
        layer = psd.create_pixel_layer(Image.new('RGBA', (32, 32)), name='multiply layer')
        layer.blend_mode = BlendMode.MULTIPLY
        psd.save(self.path)
        with self.assertRaisesRegex(ValueError, 'multiply layer.*Normal'):
            read_psd_layers(self.path, 2)

    def test_group_order_and_mask_are_baked_into_layer_alpha(self):
        psd = PSDImage.new('RGB', (32, 32))
        bottom = psd.create_pixel_layer(Image.new('RGBA', (32, 32), (255, 0, 0, 255)), name='bottom')
        top = psd.create_pixel_layer(Image.new('RGBA', (32, 32), (0, 0, 255, 255)), name='top')
        mask = Image.new('L', (32, 32), 0)
        mask.paste(255, (16, 0, 32, 32))
        top.create_mask(mask)
        group = psd.create_group([bottom, top], name='scene')
        psd.save(self.path)
        layers, names = read_psd_layers(self.path, 2)
        self.assertEqual(names, ['scene/bottom', 'scene/top'])
        self.assertEqual(layers[1].getpixel((4, 4))[3], 0)
        self.assertEqual(layers[1].getpixel((24, 4))[3], 255)
        group.opacity = 128
        psd.save(self.path)
        with self.assertRaisesRegex(ValueError, 'group opacity'):
            read_psd_layers(self.path, 2)

    def test_saved_psd_roundtrip_preserves_order_and_alpha(self):
        original = [Image.new('RGBA', (32, 32), c) for c in [(20, 30, 40, 255), (100, 150, 200, 128)]]
        save_layered_psd(self.path, original)
        restored, names = read_psd_layers(self.path, 2)
        self.assertEqual(names, ['Layer 01', 'Layer 02'])
        for before, after in zip(original, restored):
            self.assertLessEqual(np.abs(np.array(before).astype(int) - np.array(after).astype(int)).max(), 1)

    def test_shared_crop_and_layer_storage(self):
        save_layered_psd(self.path, [Image.new('RGBA', (32, 32), c) for c in [(255, 0, 0, 255), (0, 0, 255, 255)]])
        item = SimpleNamespace(path=str(self.path), dataset_config=SimpleNamespace(layer_slots=2),
                               scale_to_width=64, scale_to_height=64, crop_x=16, crop_y=8,
                               crop_width=32, crop_height=32, has_control_image=False)
        load_psd_target(item, lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1)
        self.assertEqual(tuple(item.tensor.shape), (4, 64, 32))
        self.assertEqual(item.tensor[0, :32].mean(), 1)
        self.assertEqual(item.tensor[2, 32:].mean(), 1)
        DatasetConfig(target_format='psd_layers', layer_slots=2)


class JointLayerAttentionTests(unittest.TestCase):
    def test_all_layers_are_noisy_targets_in_one_attention_block(self):
        # text, four reference tokens, text, two four-token target layers
        mask = torch.tensor([False] + [True] * 4 + [False] + [True] * 8)
        ids, target = QwenImage21Transformer2DModel.build_token_metadata(mask, [(1, 2, 2)] * 3, 2)
        self.assertFalse(target[:6].any())
        self.assertTrue(target[6:].all())
        self.assertEqual(ids[6:].unique().numel(), 1)
        self.assertNotEqual(ids[1], ids[6])
        _, ordinary = QwenImage21Transformer2DModel.build_token_metadata(mask, [(1, 2, 2)] * 3)
        self.assertEqual(ordinary.sum(), 4)

    def test_first_layer_output_depends_on_last_layer_and_has_gradient(self):
        torch.manual_seed(7)
        model = QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=1,
                    attention_head_dim=16, num_attention_heads=2, context_in_dim=8,
                    axes_dims_rope=(4, 6, 6), mlp_ratio=2)
        latents = torch.randn(1, 4, 4, 2, requires_grad=True)
        prompt = torch.randn(1, 2, 8)
        prompt_mask = torch.ones(1, 2, dtype=torch.bool)
        slots = torch.zeros_like(prompt_mask)
        result = run_transformer(model, latents, torch.tensor([0.5]), prompt, prompt_mask, slots, num_target_images=2)
        self.assertEqual(result.shape, latents.shape)
        result[:, :, :2].square().sum().backward()
        # This is impossible for independent target images in a batch.
        self.assertGreater(latents.grad[:, :, 2:].abs().sum().item(), 0)

    def test_vae_sees_individual_canvases_not_tall_strip(self):
        model = object.__new__(QwenImage2LayeredModel)
        model.model_config = SimpleNamespace(model_kwargs={'layer_slots': 2})
        seen = []
        def encode(_self, images, **kwargs):
            seen.append(tuple(images[0].shape))
            return images[0].unsqueeze(0)
        with patch.object(QwenImage2Model, 'encode_images', encode):
            result = model.encode_images([torch.ones(4, 64, 32)])
        self.assertEqual(seen, [(4, 32, 32), (4, 32, 32)])
        self.assertEqual(tuple(result.shape), (1, 4, 64, 32))


if __name__ == '__main__':
    unittest.main()
