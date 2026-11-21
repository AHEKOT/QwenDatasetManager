"""Small real-tensor checks; run with trainer/.venv, no downloaded weights."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from toolkit.config_modules import DatasetConfig, ModelConfig, GenerateImageConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from extensions_built_in.diffusion_models.qwen_image_2 import QwenImage2Model
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel
from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
    QwenImage21PromptEncoder, pack_latents, tensor_to_pil, prepare_condition_image,
)


class Qwen21RuntimeTests(unittest.TestCase):
    def model(self, rgba=False):
        return QwenImage2Model('cpu', ModelConfig(name_or_path='unused', arch='qwen_image_2',
                              model_kwargs={'rgba': rgba}), dtype='fp32')

    def real_prompt_encoder(self):
        # Use the installed Qwen3-VL processor, including its image resizing and
        # placeholder expansion, without downloading a tokenizer or model weights.
        from transformers import (PreTrainedTokenizerFast, Qwen2VLImageProcessor,
                                  Qwen3VLProcessor, Qwen3VLVideoProcessor)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        special = ['<unk>', '<pad>', '<|image_pad|>', '<|video_pad|>',
                   '<|vision_start|>', '<|vision_end|>', '<|im_start|>', '<|im_end|>']
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(dict(zip(special, range(len(special)))),
                                                unk_token='<unk>')),
            unk_token='<unk>', pad_token='<pad>', additional_special_tokens=special[2:])
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2,
                size={'shortest_edge': 65536, 'longest_edge': 1048576}),
            tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(),
            chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'][0]['text'] + '<|im_end|>\\n' }}{% endfor %}")

        class TinyTextEncoder:
            device = torch.device('cpu')
            model = SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.Identity()))

            def __call__(self, input_ids, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
                return SimpleNamespace(hidden_states=[self.model.language_model.norm(hidden)])

        return QwenImage21PromptEncoder(TinyTextEncoder(), processor)

    def test_real_processor_preserves_reference_slots_below_and_above_pixel_limits(self):
        encoder = self.real_prompt_encoder()
        image = Image.new('RGB', (448, 128), 'white')
        # Reproduce the reported 75-vs-56 failure with the processor's defaults.
        default = encoder.processor(text=['<|image_pad|>'], images=[image], return_tensors='pt')
        self.assertEqual(int((default.input_ids == encoder.image_token_id).sum()), 75)
        for size in ((448, 128), (384, 160), (64, 64), (1536, 768)):
            images = [[Image.new('RGB', size, 'white')],
                      [Image.new('RGB', size, 'black'), Image.new('RGB', size, 'white')], []]
            embeds, masks, slots = encoder.encode(['panorama', '', 'no reference'], images)
            expected = size[0] // 32 * (size[1] // 32)
            self.assertEqual([int(mask.sum()) for mask in slots], [expected, expected * 2, 0])
            for embed, mask, slot in zip(embeds, masks, slots):
                self.assertEqual(embed.shape[0], mask.shape[0])
                self.assertEqual(mask.shape, slot.shape)

    def test_real_processor_panorama_batch_reaches_transformer_and_backward(self):
        torch.set_num_threads(2)
        for align in (False, True):
            model = self.model(True)
            model.model_config.model_kwargs['align_reference_dimensions'] = align
            self.assertTrue(model.get_text_embedding_space_version().endswith('_vision_grid_v2'))
            model.prompt_encoder = self.real_prompt_encoder()
            model.text_encoder = [model.prompt_encoder.text_encoder]
            model.model = QwenImage21Transformer2DModel(in_channels=4, out_channels=4,
                num_layers=1, attention_head_dim=16, num_attention_heads=2,
                context_in_dim=8, axes_dims_rope=(4, 6, 6))
            # Unequal raw reference shapes, two references per sample. With
            # alignment off, references keep their own grid-aligned dimensions.
            controls = [[torch.rand(4, 128, 448), torch.rand(4, 160, 384)],
                        [torch.rand(4, 256, 896), torch.rand(4, 320, 768)]]
            embeds = model.get_prompt_embeds(['panorama', ''], control_images=controls,
                                            target_size=(448, 128))
            expected = 112 if align else 116
            self.assertEqual([int(mask.sum()) for mask in embeds.image_slot_mask], [expected] * 2)
            def vae_encode(model, images, **kwargs):
                height, width = images[0].shape[-2:]
                return torch.zeros(1, 4, height // 16, width // 16)
            latents = torch.randn(2, 4, 8, 28)
            batch = SimpleNamespace(control_tensor_list=controls, control_tensor=None)
            with patch.object(QwenImage2Model, 'encode_images', side_effect=vae_encode):
                result = model.get_noise_prediction(latents, torch.tensor([500., 500.]), embeds, batch)
            self.assertEqual(result.shape, latents.shape)
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(model.model.proj_out.weight.grad).all())
            self.assertGreater(model.model.proj_out.weight.grad.norm().item(), 0)

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
                    self.assertIsNone(item.control_tensor)
                    self.assertEqual(item.control_tensor_list[0].shape[0], 4 if rgba else 3)
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

    def test_reference_canvas_padding_preserves_rgb_rgba_and_aspect(self):
        for channels in (3, 4):
            for height, width in ((20, 40), (40, 20)):
                image = torch.full((1, channels, height, width), 0.5)
                result = prepare_condition_image(image, 4096, target_size=(70, 66))
                self.assertEqual(tuple(result.shape), (1, channels, 64, 64))
                border = result[:, :, :16] if width > height else result[:, :, :, :16]
                self.assertTrue(torch.all(border == (0 if channels == 4 else 1)))
                torch.testing.assert_close(result[:, :, 16:48, 16:48], torch.full((1, channels, 32, 32), 0.5))
                if channels == 4:
                    self.assertEqual(tensor_to_pil(result).getpixel((0, 0)), (255, 255, 255))

    def test_alignment_agrees_for_prompt_training_sampling_and_cache(self):
        model = self.model(True)
        model.model_config.model_kwargs['match_target_res'] = False
        old_key = model.get_text_embedding_space_version()
        model.model_config.model_kwargs['align_reference_dimensions'] = True
        self.assertNotEqual(old_key, model.get_text_embedding_space_version())
        self.assertTrue(model.text_embedding_uses_target_size)
        model.text_encoder = [SimpleNamespace(device=torch.device('cpu'))]
        seen = []
        def encode(prompt, images, **kwargs):
            seen.extend(image.size for sample in images for image in sample)
            return ([torch.zeros(4, 8)] * len(prompt), [torch.ones(4, dtype=torch.bool)] * len(prompt),
                    [torch.ones(4, dtype=torch.bool)] * len(prompt))
        model.prompt_encoder = SimpleNamespace(encode=encode)
        controls = [[torch.full((4, 20, 40), 0.5)], [torch.full((4, 40, 20), 0.5)]]
        embeds = model.get_prompt_embeds(['wide', 'tall'], control_images=controls, target_size=(64, 64))
        self.assertEqual(seen, [(64, 64), (64, 64)])
        model.model = SimpleNamespace(device=torch.device('cpu'), to=lambda *args: None)
        batch = SimpleNamespace(control_tensor_list=controls, control_tensor=None)
        sizes = []
        def vae_encode(model, images, **kwargs):
            sizes.append(tuple(images[0].shape[-2:]))
            return torch.zeros(1, 64, 4, 4)
        with patch.object(QwenImage2Model, 'encode_images', side_effect=vae_encode), patch(
                'extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2.run_transformer',
                return_value=torch.zeros(2, 64, 4, 4)) as transformer:
            model.get_noise_prediction(torch.zeros(2, 64, 4, 4), torch.tensor([500., 500.]), embeds, batch)
        self.assertEqual(sizes, [(64, 64), (64, 64)])
        self.assertEqual(transformer.call_args.kwargs['condition_shapes'], [(4, 4)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.png'
            Image.new('RGBA', (40, 20), (20, 40, 60, 128)).save(path)
            config = GenerateImageConfig(width=64, height=64, ctrl_img=str(path),
                                         output_folder=directory, output_ext='png')
            pipeline = Mock(return_value=['sample'])
            model.generate_single_image(pipeline, config, embeds, embeds, None, {})
            images = pipeline.call_args.kwargs['condition_images']
            self.assertEqual(len(images[0]), 1)
            self.assertEqual(tuple(images[0][0].shape), (1, 4, 64, 64))


if __name__ == '__main__':
    unittest.main()
