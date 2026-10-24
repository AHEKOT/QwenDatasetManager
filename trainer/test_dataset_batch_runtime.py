"""CPU dataloader checks; run with trainer/.venv, without model weights."""
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))

import torch
from PIL import Image
from toolkit.config_modules import DatasetConfig, ModelConfig, preprocess_dataset_raw_config
from toolkit.data_loader import get_dataloader_from_datasets
from extensions_built_in.diffusion_models.qwen_image_2 import QwenImage2Model


class DatasetBatchRuntimeTests(unittest.TestCase):
    def model(self):
        model = QwenImage2Model('cpu', ModelConfig(name_or_path='unused', arch='qwen_image_2',
                               model_kwargs={'rgba': True}), dtype='fp32')
        model.vae = None
        model.model = None
        return model

    def dataset(self, root, name, count=4, **options):
        folder = root / name
        folder.mkdir()
        for index in range(count):
            Image.new('RGBA', (64, 64), (20, 40, 60, 128)).save(folder / f'{index}.png')
        return dict(folder_path=str(folder), resolution=64, num_workers=0,
                    caption_dropout_rate=0, **options)

    def test_mixed_datasets_use_override_and_global_for_real_batches(self):
        for buckets in (True, False):
            with self.subTest(buckets=buckets), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                options = [self.dataset(root, 'single', batch_size_override=1, buckets=buckets),
                           self.dataset(root, 'global', batch_size=1, buckets=buckets)]
                loader = get_dataloader_from_datasets(options, batch_size=2, sd=self.model())
                self.assertEqual([dataset.batch_size for dataset in loader.dataset.datasets], [1, 2])
                self.assertEqual(len(loader), 6)
                seen = Counter()
                for batch in loader:
                    names = {Path(item.path).parent.name for item in batch.file_items}
                    self.assertEqual(len(names), 1)
                    name = names.pop()
                    self.assertEqual(batch.tensor.shape[0], 1 if name == 'single' else 2)
                    self.assertEqual(batch.tensor.shape[1], 4)
                    seen.update(item.path for item in batch.file_items)
                self.assertEqual(len(seen), 8)
                self.assertTrue(all(count == 1 for count in seen.values()))

    def test_blank_override_and_legacy_batch_size_keep_global_batch(self):
        for buckets in (True, False):
            with self.subTest(buckets=buckets), tempfile.TemporaryDirectory() as directory:
                options = [self.dataset(Path(directory), 'global', batch_size_override='',
                                        batch_size=1, buckets=buckets)]
                loader = get_dataloader_from_datasets(options, batch_size=2, sd=self.model())
                self.assertEqual(len(loader), 2)
                self.assertTrue(all(batch.tensor.shape[0] == 2 for batch in loader))

    def test_override_survives_resolution_splitting_and_validation(self):
        configs = preprocess_dataset_raw_config([{'resolution': [64, 128], 'batch_size_override': 1}])
        self.assertEqual([DatasetConfig(**config).batch_size_override for config in configs], [1, 1])
        for value in (None, '', '  '):
            self.assertIsNone(DatasetConfig(batch_size_override=value).batch_size_override)
        for value in (0, -1, 1.5, 129, True, 'invalid', float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'batch_size_override'):
                DatasetConfig(batch_size_override=value)

    def test_single_sample_batches_allow_480_and_504_reference_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'img'
            control = root / 'Control1'
            target.mkdir()
            control.mkdir()
            for index, size in enumerate(((384, 320), (448, 288))):
                Image.new('RGBA', (352, 352), (20, 40, 60, 128)).save(target / f'{index}.png')
                Image.new('RGBA', size, (20, 40, 60, 128)).save(control / f'{index}.png')
            model = self.model()
            options = [dict(folder_path=str(target), control_path=[str(control)],
                            full_size_control_images=True, resolution=352, num_workers=0,
                            caption_dropout_rate=0, batch_size_override=1)]
            loader = get_dataloader_from_datasets(options, batch_size=2, sd=model)
            prepared = []
            for batch in loader:
                self.assertEqual(len(batch.file_items), 1)
                prepared.extend(model._prepare_control_images(
                    model._normalize_control_images(batch.control_tensor_list, 1), target_pixels=352 * 352))

            # Replace only the heavy VAE with its 16x spatial output shape.
            def encode(model, images, **kwargs):
                height, width = images[0].shape[-2:]
                return torch.zeros(1, 64, height // 16, width // 16)

            with patch.object(QwenImage2Model, 'encode_images', side_effect=encode):
                with self.assertRaisesRegex(ValueError, r'got \[480, 504\]'):
                    model.encode_condition_images(prepared)
                lengths = [model.encode_condition_images([sample])[0].shape[1] for sample in prepared]
            self.assertEqual(sorted(lengths), [480, 504])

    def test_variable_source_sizes_collate_and_align_in_multi_sample_batches(self):
        for references, workers in ((1, 0), (2, 0), (1, 1)):
            with self.subTest(references=references, workers=workers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / 'img'
                target.mkdir()
                controls = [root / f'Control{index + 1}' for index in range(references)]
                for folder in controls:
                    folder.mkdir()
                for index, size in enumerate(((1352, 512), (1024, 1024), (288, 448))):
                    Image.new('RGBA', (352, 352), (20, 40, 60, 128)).save(target / f'{index}.png')
                    for reference_index, folder in enumerate(controls):
                        Image.new('RGBA', size, (20 + reference_index * 40, 40, 60, 128)).save(folder / f'{index}.png')
                model = self.model()
                model.model_config.model_kwargs.update(align_reference_dimensions=True, match_target_res=False)
                loader = get_dataloader_from_datasets([dict(folder_path=str(target),
                    control_path=[str(folder) for folder in controls], full_size_control_images=True,
                    resolution=352, num_workers=workers, caption_dropout_rate=0)], batch_size=3, sd=model)
                for batch in loader:
                    self.assertIsNone(batch.control_tensor)
                    self.assertEqual(len(batch.control_tensor_list), 3)
                    prepared = model._prepare_control_images(batch.control_tensor_list, target_size=(352, 352))
                    self.assertTrue(all(tuple(image.shape) == (1, 4, 352, 352)
                                        for sample in prepared for image in sample))
                    samples = model._normalize_control_images(batch.control_tensor_list, 3)
                    prepared = model._prepare_control_images(samples, target_size=(352, 352))
                    def encode(model, images, **kwargs):
                        return torch.zeros(1, 64, images[0].shape[-2] // 16, images[0].shape[-1] // 16)
                    with patch.object(QwenImage2Model, 'encode_images', side_effect=encode):
                        latents, shapes = model.encode_condition_images(prepared)
                    self.assertEqual(tuple(latents.shape), (3, 484 * references, 64))
                    self.assertEqual(shapes, [(22, 22)] * references)


if __name__ == '__main__':
    unittest.main()
