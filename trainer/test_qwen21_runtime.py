"""Small real-tensor checks; run with trainer/.venv, no downloaded weights."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from toolkit.config_modules import DatasetConfig, ModelConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from extensions_built_in.diffusion_models.qwen_image_2 import QwenImage2Model
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel
from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import pack_latents, tensor_to_pil


class Qwen21RuntimeTests(unittest.TestCase):
    def model(self, rgba=False):
        return QwenImage2Model('cpu', ModelConfig(name_or_path='unused', arch='qwen_image_2',
                              model_kwargs={'rgba': rgba}), dtype='fp32')

    def test_tiny_transformer_training_with_and_without_reference(self):
        torch.set_num_threads(2)
        for reference in (False, True):
            model = self.model()
            model.model = QwenImage21Transformer2DModel(in_channels=4, out_channels=4,
                num_layers=1, attention_head_dim=16, num_attention_heads=2,
                context_in_dim=8, axes_dims_rope=(4, 6, 6))
            slots = torch.tensor([False, reference, False])
            embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(3, 8)],
                attention_mask=[torch.ones(3, dtype=torch.bool)], image_slot_mask=[slots])
            latents = torch.randn(1, 4, 2, 2)
            batch = SimpleNamespace(control_tensor_list=[[torch.rand(3, 32, 32)]], control_tensor=None)
            model.encode_condition_images = Mock(return_value=(pack_latents(torch.randn_like(latents)), [(2, 2)]))
            result = model.get_noise_prediction(latents, torch.tensor([500.]), embeds, batch=batch)
            self.assertEqual(result.shape, latents.shape)
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(model.model.proj_out.weight.grad).all())
            self.assertGreater(model.model.proj_out.weight.grad.norm().item(), 0)
            self.assertEqual(model.encode_condition_images.call_count, int(reference))
            torch.testing.assert_close(model.get_loss_target(noise=latents, batch=SimpleNamespace(latents=latents / 2)), latents / 2)

    def test_rgba_targets_controls_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'controls').mkdir()
            for mode in ('RGB', 'RGBA'):
                Image.new(mode, (64, 64), (20, 40, 60, 128) if mode == 'RGBA' else (20, 40, 60)).save(root / f'{mode}.png')
                Image.new('RGBA', (32, 64), (20, 40, 60, 128)).save(root / 'controls' / f'{mode}.png')
                paths = []
                for rgba in (False, True):
                    model = self.model(rgba)
                    dataset = DatasetConfig(folder_path=str(root), control_path=[str(root / 'controls')], resolution=64)
                    item = FileItemDTO(path=str(root / f'{mode}.png'), dataset_config=dataset,
                        sd=model, encode_control_in_text_embeddings=True,
                        text_embedding_space_version=model.get_text_embedding_space_version(),
                        text_embedding_uses_target_size=True)
                    item.load_and_process_image(ToTensor())
                    self.assertEqual(item.tensor.shape[0], 4 if rgba else 3)
                    self.assertEqual(item.control_tensor.shape[0], 4 if rgba else 3)
                    if rgba:
                        self.assertAlmostEqual(item.tensor[3].mean().item(), 128 / 255 if mode == 'RGBA' else 1, places=5)
                    paths.append(item.get_latent_path())
                    before = item.get_text_embedding_path()
                    item.crop_width = 128
                    self.assertNotEqual(before, item.get_text_embedding_path(recalculate=True))
                self.assertNotEqual(*paths)

    def test_prompt_target_size_and_alpha_compositing(self):
        model = self.model(True)
        # Use an explicit signature, as BaseModel inspects it for plugin compatibility.
        received = {}
        def encode(prompt, control_images=None, target_size=None):
            received.update(target_size=target_size)
        model.get_prompt_embeds = encode
        model.encode_prompt('test', target_size=(512, 768))
        self.assertEqual(received['target_size'], (512, 768))
        image = torch.zeros(4, 32, 32)
        self.assertEqual(tensor_to_pil(image).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(model.get_bucket_divisibility(), 32)

    def test_comfy_convrot_embedding_load(self):
        from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer
        from toolkit.util.comfy_quant_export import export_comfy_quantized_layers
        from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
        from toolkit.models.v2._mixin import OstrisModelMixin
        source = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        quantizer = get_ostris_quantizer('convrot8')
        convert_linear_to_ostris(source[0], quantizer)
        expected = quantizer.dequantize_folded(source[0])
        state, _, _ = export_comfy_quantized_layers(source)
        restored = torch.nn.Sequential(torch.nn.Embedding(32, 64, device='meta'))
        remaining, count = import_comfy_quantized_layers(restored, state, orig_dtype=torch.float32)
        self.assertEqual(count, 1)
        OstrisModelMixin._load_state_dict_with_quantized(restored, remaining)
        indices = torch.tensor([0, 7, 31])
        torch.testing.assert_close(restored(indices), expected[indices])


if __name__ == '__main__':
    unittest.main()
