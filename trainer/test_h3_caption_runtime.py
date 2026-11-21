"""Regression coverage for bounded H3 dataset vision conditioning; no weights needed."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))

import torch
from PIL import Image
from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin, TextEmbeddingCachingMixin
from extensions_built_in.diffusion_models.minimax_h3 import MinimaxH3Model, MinimaxH3Ref2VAModel
from extensions_built_in.diffusion_models.minimax_h3.src.text_encoder import (
    VideoRef, encode_minimax_h3_prompt, resize_conditioning_image,
)


class StopBeforeEncoder(Exception):
    pass


class H3CaptionTests(unittest.TestCase):
    def test_selected_maximum_survives_resolution_split_per_dataset(self):
        raw = [{'resolution': [256, 512]}, {'resolution': [768, 256]}]
        configs = preprocess_dataset_raw_config(raw)
        configs = preprocess_dataset_raw_config(configs)
        self.assertEqual([DatasetConfig(**c).text_embedding_resolution for c in configs],
                         [512, 512, 768, 768])
        self.assertEqual(raw[0]['resolution'], [256, 512])
        self.assertEqual(DatasetConfig(resolution=1024).text_embedding_resolution, 1024)

    def test_aspect_ratio_and_patch_alignment_without_full_size_vision_tensors(self):
        for size, expected in [((4096, 2048), (512, 256)), ((2048, 4096), (256, 512)),
                               ((4096, 4096), (512, 512)), ((128, 64), (128, 64))]:
            image = Image.new('RGB', size)
            self.assertEqual(resize_conditioning_image(image, 512).size, expected)
            self.assertIs(resize_conditioning_image(image, None), image)

    def test_real_cache_bounds_image_before_tensor_and_restores_context_on_error(self):
        for cls in (MinimaxH3Model, MinimaxH3Ref2VAModel):
            model = cls.__new__(cls)
            model._ref_video_dataset_config = None
            model._text_embedding_resolution = None
            # Delegate only the sizing hooks to the real H3 class; the stand-in
            # records the exact tensors that the dataset sends to the encoder.
            sd = SimpleNamespace(
                text_embedding_dataset_context=model.text_embedding_dataset_context,
                prepare_text_encoder_image=model.prepare_text_encoder_image,
                device='cpu', device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                set_device_state_preset=Mock(), has_multiple_control_images=True,
            )
            cfg = DatasetConfig(**preprocess_dataset_raw_config([{'resolution': [256, 512]}])[0])
            cfg.caption_dropout_rate = 0
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'control.png'
                Image.new('RGB', (4096, 2048)).save(source)
                item = SimpleNamespace(
                    dataset_config=cfg, caption='example', control_path=str(source),
                    encode_control_in_text_embeddings=True, dopsd_self_ref=False,
                    get_text_embedding_path=lambda **kwargs: str(Path(folder) / 'cache.safetensors'),
                )
                def encode(prompt, control_images=None):
                    self.assertEqual(tuple(control_images[0].shape), (1, 3, 256, 512))
                    self.assertEqual(model._text_embedding_resolution, 512)
                    raise StopBeforeEncoder()
                sd.encode_prompt = encode
                dataset = SimpleNamespace(sd=sd, dataset_path=folder, dataset_config=cfg, file_list=[item])
                with self.assertRaises(StopBeforeEncoder):
                    TextEmbeddingCachingMixin.cache_text_embeddings(dataset)
                self.assertIsNone(model._text_embedding_resolution)
                self.assertIsNone(model._ref_video_dataset_config)
                with Image.open(source) as original:
                    self.assertEqual(original.size, (4096, 2048))

    def test_image_and_video_processors_cannot_upscale_to_their_default_pixel_budget(self):
        for video in (False, True):
            processor = SimpleNamespace(
                image_processor=Mock(merge_size=2, side_effect=StopBeforeEncoder),
                video_processor=Mock(side_effect=StopBeforeEncoder),
            )
            encoder = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=50)),
                                      device=torch.device('cpu'))
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda value: 1)
            large = Image.new('RGB', (2048, 1024))
            refs = [VideoRef(frames=[large, large], timestamps=[0, .5])] if video else [large]
            with self.assertRaises(StopBeforeEncoder):
                encode_minimax_h3_prompt(encoder, tokenizer, processor, 'caption',
                                         keyframes=refs, vision_resolution=512)
            kwargs = (processor.video_processor if video else processor.image_processor).call_args.kwargs
            self.assertFalse(kwargs['do_resize'])
            if video:
                self.assertEqual(kwargs['videos'][0].shape, (2, 256, 512, 3))
                self.assertFalse(kwargs['do_sample_frames'])
            else:
                self.assertEqual(kwargs['images'][0].size, (512, 256))

    def test_cache_identity_changes_with_maximum_including_teacher_but_not_other_models(self):
        def key(resolution, arch='minimax_h3_ref2va', **kwargs):
            item = SimpleNamespace(
                caption='caption', text_embedding_space_version=arch, text_embedding_version=1,
                dataset_config=DatasetConfig(resolution=resolution), encode_control_in_text_embeddings=True,
                control_path='control.png',
            )
            return TextEmbeddingFileItemDTOMixin.get_text_embedding_info_dict(item, **kwargs)
        for arch in ('minimax_h3', 'minimax_h3_ref2va:img_as_vid5', 'minimax_h3_rgba', 'minimax_h3_ref2va_rgba'):
            for teacher in (False, True):
                self.assertNotEqual(key(512, arch, dopsd_self_ref=teacher), key(768, arch, dopsd_self_ref=teacher))
            self.assertEqual(key(512, arch, text_only=True), key(768, arch, text_only=True))
        self.assertEqual(key(512, 'qwen_image'), key(768, 'qwen_image'))

    def test_installed_qwen_processors_encode_bounded_images_and_videos(self):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        processor = SimpleNamespace(
            image_processor=Qwen2VLImageProcessor(patch_size=16),
            video_processor=Qwen3VLVideoProcessor(),
            create_mm_token_type_ids=lambda batches: [[0] * len(ids) for ids in batches],
        )
        tokenizer = Mock(return_value={'input_ids': [9]})
        tokenizer.convert_tokens_to_ids.return_value = 1
        encoder = Mock()
        encoder.device = torch.device('cpu')
        encoder.dtype = torch.float32
        encoder.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=50))
        encoder.model.side_effect = lambda **kwargs: SimpleNamespace(
            hidden_states=[torch.zeros(1, kwargs['input_ids'].shape[1], 4)] * 51)
        image = Image.new('RGB', (2048, 1024))
        for video in (False, True):
            refs = [VideoRef(frames=[image, image], timestamps=[0, .5])] if video else [image]
            embeds, tags = encode_minimax_h3_prompt(
                encoder, tokenizer, processor, 'caption', keyframes=refs, vision_resolution=512)
            kwargs = encoder.model.call_args.kwargs
            grid = kwargs['video_grid_thw' if video else 'image_grid_thw']
            self.assertEqual(grid.tolist(), [[1, 16, 32]])
            self.assertEqual(embeds.shape[0], tags.shape[0])


if __name__ == '__main__':
    unittest.main()
