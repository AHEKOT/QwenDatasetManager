import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[5]
NODE_ROOT = PROJECT_ROOT / 'ComfyUI-QDM-ChromaKey'
AI_TOOLKIT_ROOT = PROJECT_ROOT / 'trainer' / 'ai_toolkit'
for path in (str(NODE_ROOT), str(AI_TOOLKIT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qdm_chromakey_core.inference import load_model, run_tiled
from qdm_chromakey_core.model import (
    ARCHITECTURE_ID,
    ARCHITECTURE_ID_V2,
    ARCHITECTURE_ID_V3,
    AnimeKeyMatte,
    AnimeKeyMatteV2,
    AnimeKeyMatteV3,
)
from extensions.chromakey_training.data import (
    AspectBucketBatchSampler,
    ChromaKeyDataset,
)


class ChromaKeyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.files = []
        for name, size in (('wide', (320, 80)), ('tall', (80, 320)), ('portrait', (120, 200))):
            image = Image.new('RGBA', size, (180, 80, 120, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((size[0] // 4, size[1] // 5, size[0] * 3 // 4, size[1] * 4 // 5), fill=(180, 80, 120, 255))
            path = self.root / f'{name}.png'
            image.save(path)
            self.files.append(path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_aspect_buckets_never_force_wide_or_tall_sources_to_square(self):
        sampler = AspectBucketBatchSampler(
            self.files, [384], max_batch_size=16,
            megapixels_per_batch=4, max_long_side=1280, seed=7,
        )
        requested = [item for batch in sampler for item in batch]
        shapes = {index: (height, width) for index, height, width, _seed in requested}
        self.assertGreater(shapes[0][1] / shapes[0][0], 3.0)
        self.assertGreater(shapes[1][0] / shapes[1][1], 3.0)

    def test_augmentation_returns_rgb_only_input_and_rgba_targets(self):
        dataset = ChromaKeyDataset(self.files, {
            'green_chance': 100, 'blue_chance': 0, 'white_chance': 0, 'black_chance': 0,
            'spill_chance': 100, 'spill_width_min': 2, 'spill_width_max': 5,
            'spill_strength_min': 0.2, 'spill_strength_max': 0.5,
        })
        sample = dataset[(0, 128, 512, 123)]
        self.assertEqual(tuple(sample['input'].shape), (3, 128, 512))
        self.assertEqual(tuple(sample['foreground'].shape), (3, 128, 512))
        self.assertEqual(tuple(sample['alpha'].shape), (1, 128, 512))
        self.assertEqual(sample['valid'].min().item(), 1.0)
        self.assertGreater(sample['spill'].max().item(), 0)

    def test_model_and_safetensors_round_trip_support_rectangles(self):
        model = AnimeKeyMatte().eval()
        source = torch.rand(1, 3, 96, 224)
        with torch.no_grad():
            expected = model(source)
        self.assertEqual(tuple(expected['alpha'].shape), (1, 1, 96, 224))
        path = self.root / 'model.safetensors'
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            str(path), metadata={'architecture': ARCHITECTURE_ID},
        )
        restored, _metadata = load_model(path)
        with torch.no_grad():
            actual = restored(source)
        self.assertTrue(torch.allclose(expected['alpha'], actual['alpha']))

    def test_v2_starts_with_a_strong_learned_border_key_prior(self):
        model = AnimeKeyMatteV2().eval()
        source = torch.zeros(1, 3, 96, 160)
        source[:, 1] = 0.8
        source[:, :, 24:72, 48:112] = torch.tensor([0.8, 0.2, 0.3]).view(1, 3, 1, 1)
        with torch.no_grad():
            prediction = model(source)
        background = prediction['base_alpha'][0, 0, :16, :16].mean().item()
        foreground = prediction['base_alpha'][0, 0, 32:64, 64:96].mean().item()
        self.assertLess(background, 0.05)
        self.assertGreater(foreground, 0.95)

    def test_v2_keeps_the_full_resolution_refinement_path_narrow(self):
        model = AnimeKeyMatteV2()
        self.assertEqual(model.alpha_residual_head.in_channels, 8)
        self.assertEqual(model.rgb_delta_head.in_channels, 8)
        self.assertFalse(hasattr(model, 'full_fusion'))

    def test_v2_safetensors_round_trip(self):
        model = AnimeKeyMatteV2().eval()
        path = self.root / 'model_v2.safetensors'
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            str(path), metadata={'architecture': ARCHITECTURE_ID_V2},
        )
        restored, metadata = load_model(path)
        self.assertIsInstance(restored, AnimeKeyMatteV2)
        self.assertEqual(metadata['architecture'], ARCHITECTURE_ID_V2)

    def test_v3_uses_a_bounded_key_hint_and_a_learned_primary_matte(self):
        model = AnimeKeyMatteV3().train()
        source = torch.zeros(1, 3, 64, 96)
        source[:, 1] = 0.8
        target = torch.zeros(1, 1, 64, 96)
        target[:, :, 16:48, 32:64] = 1.0
        prediction = model(source)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction['alpha_logits'], target,
        )
        loss.backward()
        self.assertIsNotNone(model.detail_head.weight.grad)
        self.assertGreater(model.detail_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(prediction['alpha'].mean().item(), 0.05)

    def test_v3_safetensors_round_trip(self):
        model = AnimeKeyMatteV3().eval()
        path = self.root / 'model_v3.safetensors'
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            str(path), metadata={'architecture': ARCHITECTURE_ID_V3},
        )
        restored, metadata = load_model(path)
        self.assertIsInstance(restored, AnimeKeyMatteV3)
        self.assertEqual(metadata['architecture'], ARCHITECTURE_ID_V3)

    def test_tiled_inference_preserves_exact_dimensions(self):
        model = AnimeKeyMatte().eval()
        clean, alpha = run_tiled(model, torch.rand(3, 137, 353), tile_size=128, overlap=32)
        self.assertEqual(tuple(clean.shape), (3, 137, 353))
        self.assertEqual(tuple(alpha.shape), (1, 137, 353))


if __name__ == '__main__':
    unittest.main()
