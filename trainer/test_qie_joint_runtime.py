import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import RescaleTransform
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.rgba_utils import prepare_rgba_image, prepare_rgba_validation_pair
from extensions.rgba_training.qwen_image_edit_plus_rgba import QwenImageEditPlusRGBAModel


class QieJointRuntimeTests(unittest.TestCase):
    def test_disk_cache_disables_autograd_despite_input_gradient_hooks(self):
        from unittest.mock import Mock
        from toolkit.dataloader_mixins import TextEmbeddingCachingMixin
        from toolkit.prompt_utils import PromptEmbeds
        encoder = torch.nn.Sequential(torch.nn.Embedding(4, 8), torch.nn.Linear(8, 8))
        encoder.requires_grad_(False)
        encoder[0].register_forward_hook(lambda module, args, output: output.requires_grad_(True))
        encoder.eval()
        with tempfile.TemporaryDirectory() as folder:
            cfg = DatasetConfig(folder_path=folder, cache_text_embeddings=True, caption_dropout_rate=0)
            item = SimpleNamespace(
                dataset_config=cfg, caption='edit', encode_control_in_text_embeddings=False,
                dopsd_self_ref=False,
                get_text_embedding_path=lambda **kwargs: str(Path(folder) / 'unused.safetensors'),
            )
            def encode(caption):
                self.assertFalse(torch.is_grad_enabled())
                output = encoder(torch.tensor([[0, 1]]))
                self.assertFalse(output.requires_grad)
                self.assertIsNone(output.grad_fn)
                result = PromptEmbeds(output)
                result.save = Mock()
                return result
            sd = SimpleNamespace(device='cpu', set_device_state_preset=Mock(), encode_prompt=encode)
            dataset = SimpleNamespace(sd=sd, dataset_path=folder, dataset_config=cfg, file_list=[item])
            with torch.enable_grad():
                TextEmbeddingCachingMixin.cache_text_embeddings(dataset)
                self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(item.is_text_embedding_cached)

    def test_qie_conditioning_skips_lm_head_and_preserves_padding_and_controls(self):
        from unittest.mock import Mock
        from extensions_built_in.diffusion_models.qwen_image.qwen_image_pipelines import QwenImageEditPlusCustomPipeline
        hidden = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        class Inputs(dict):
            def to(self, device):
                return self
        inputs = Inputs(input_ids=torch.zeros(2, 5, dtype=torch.long), attention_mask=mask,
                        pixel_values=torch.ones(2, 3), image_grid_thw=torch.ones(2, 3))
        core = Mock(return_value=SimpleNamespace(last_hidden_state=hidden))
        outer = Mock(side_effect=AssertionError('Vocabulary logits must not be computed'))
        outer.model = core
        outer.dtype = torch.float32
        processor = Mock(return_value=inputs)
        pipeline = SimpleNamespace(
            _execution_device=torch.device('cpu'), text_encoder=outer, processor=processor,
            prompt_template_encode='{}', prompt_template_encode_start_idx=2,
            _extract_masked_hidden=lambda states, masks: [row[valid.bool()] for row, valid in zip(states, masks)],
        )
        controls = [object(), object()]
        embeds, attention = QwenImageEditPlusCustomPipeline._get_qwen_prompt_embeds(
            pipeline, ['edit A', 'edit B'], image=controls)
        outer.assert_not_called()
        self.assertFalse(core.call_args.kwargs['use_cache'])
        self.assertFalse(core.call_args.kwargs['output_hidden_states'])
        self.assertIs(processor.call_args.kwargs['images'], controls)
        self.assertIn('Picture 2:', processor.call_args.kwargs['text'][0])
        torch.testing.assert_close(embeds[0], hidden[0, 2:])
        torch.testing.assert_close(embeds[1, :1], hidden[1, 2:3])
        self.assertEqual(embeds[1, 1:].abs().sum().item(), 0)
        torch.testing.assert_close(attention, torch.tensor([[1, 1, 1], [1, 0, 0]]))

    def test_real_loader_accepts_rgb_and_rgba_and_preserves_paired_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'Control1').mkdir()
            cfg = DatasetConfig(folder_path=folder, pixel_channels='rgba', rgba_require_alpha=False,
                                rgba_control_mode='paired', control_path=str(root / 'Control1'),
                                resolution=32, num_workers=0)
            self.assertTrue(cfg.load_image_when_caching_latents)
            transform = transforms.Compose([transforms.ToTensor(), RescaleTransform()])
            for mode in ('RGB', 'RGBA'):
                target = Image.new(mode, (32, 32), (200, 20, 40) if mode == 'RGB' else (200, 20, 40, 128))
                target.save(root / f'{mode}.png')
                Image.new('RGB', (32, 32), (20, 40, 200)).save(root / 'Control1' / f'{mode}.png')
                item = FileItemDTO(path=str(root / f'{mode}.png'), dataset_config=cfg)
                item.load_and_process_image(transform)
                self.assertEqual(tuple(item.tensor.shape), (4, 32, 32))
                self.assertAlmostEqual(item.tensor[3].mean().item(), 1.0 if mode == 'RGB' else 128 / 255 * 2 - 1, places=5)
                self.assertGreater(item.control_tensor[2].mean().item(), .7)
                self.assertLess(item.control_tensor[0].mean().item(), .1)
                self.assertGreater(item.tensor[0].mean().item(), .5)

    def test_paired_validation_preserves_rgb_and_fractional_alpha(self):
        controls = [Image.new('RGB', (24, 32), (20, 40, 200)), Image.new('RGB', (48, 24), (40, 60, 80))]
        for mode in ('RGB', 'RGBA'):
            image = Image.new(mode, (32, 32), (200, 20, 40) if mode == 'RGB' else (200, 20, 40, 128))
            target, refs = prepare_rgba_validation_pair(image, (32, 32), control_mode='paired', control_images=controls)
            self.assertEqual(len(refs), 2)
            self.assertEqual(tuple(refs[0].shape), (3, 32, 24))
            self.assertEqual(tuple(refs[1].shape), (3, 24, 48))
            self.assertAlmostEqual(target[3].mean().item(), 1.0 if mode == 'RGB' else 128 / 255 * 2 - 1, places=5)
            self.assertGreater(refs[0][2].mean().item(), .7)

    def test_opaque_green_rgb_is_not_modified_by_legacy_matte_cleanup(self):
        source = np.zeros((32, 32, 3), dtype=np.uint8)
        source[:] = [200, 20, 40]
        source[:3] = [0, 255, 0]
        result = np.asarray(prepare_rgba_image(Image.fromarray(source), require_alpha=False, edge_color_correction='matte_despill'))
        np.testing.assert_array_equal(result[..., :3], source)
        self.assertTrue((result[..., 3] == 255).all())

    def test_qie_keeps_rgb_gradients_even_with_alpha_probe(self):
        model = QwenImageEditPlusRGBAModel.__new__(QwenImageEditPlusRGBAModel)
        model._rgba_alpha_probe_cpu = {'present': True}
        for mode in ('paired', 'edit', 'generation'):
            batch = SimpleNamespace(dataset_config=SimpleNamespace(rgba_control_mode=mode, rgba_generate_control=mode != 'paired'))
            pred = torch.zeros((1, 4, 4, 4), requires_grad=True)
            target = torch.zeros_like(pred)
            target[:, :3] = 1
            loss = torch.nn.functional.mse_loss(pred, target) * model.get_rgba_diffusion_loss_weight(batch)
            loss.backward()
            self.assertGreater(pred.grad[:, :3].abs().sum().item(), 0)
            self.assertEqual(pred.grad[:, 3].abs().sum().item(), 0)

    def test_validation_cache_sends_all_real_controls_to_text_encoder(self):
        from unittest.mock import Mock
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
        from toolkit.config_modules import ValidationItem
        from toolkit.prompt_utils import PromptEmbeds
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'target.jpg'
            Image.new('RGB', (32, 32), (200, 20, 40)).save(target)
            paths = []
            for i in range(3):
                path = root / f'control{i}.png'
                Image.new('RGB', (32+i*8, 24), (20, 40, 120+i*30)).save(path)
                paths.append(str(path))
            sd = SimpleNamespace(
                supports_rgba_latent_loss=True, encode_control_in_text_embeddings=True,
                get_bucket_divisibility=lambda: 8, text_encoder=[], text_encoder_to=Mock(),
                encode_prompt=Mock(return_value=PromptEmbeds(torch.zeros(1, 2, 4))),
                encode_images=lambda images, **kw: torch.stack(images),
                vae=SimpleNamespace(device=torch.device('cpu'), to=Mock()),
            )
            process = SimpleNamespace(
                train_config=SimpleNamespace(dtype='fp32', validation_config=SimpleNamespace(
                    resolution=32, validation_items=[ValidationItem(image_path=str(target),
                        prompt='Change blue to red', rgba_control_mode='paired', control_paths=paths)])),
                accelerator=SimpleNamespace(is_main_process=True), device_torch=torch.device('cpu'),
                sd=sd, trigger_word=None,
            )
            BaseSDTrainProcess.setup_validation(process)
            inputs = sd.encode_prompt.call_args.kwargs['control_images']
            self.assertEqual(len(inputs), 3)
            self.assertEqual([tuple(t.shape) for t in inputs], [(1,3,24,32),(1,3,24,40),(1,3,24,48)])
            self.assertEqual(process._validation_cache['rgba_modes'], ['paired'])
            self.assertTrue((process._validation_cache['rgba_targets'][0][3] == 1).all())


if __name__ == '__main__':
    unittest.main()
