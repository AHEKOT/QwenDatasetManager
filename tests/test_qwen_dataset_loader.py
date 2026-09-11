import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


NODE_DIR = Path(__file__).resolve().parents[1] / 'comfyui_qwenDatasetManager'


def load_node(filename, output_dir):
    folder_paths = types.ModuleType('folder_paths')
    folder_paths.get_output_directory = lambda: str(output_dir)
    with patch.dict('sys.modules', {'folder_paths': folder_paths}):
        spec = importlib.util.spec_from_file_location(filename, NODE_DIR / f'{filename}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class QwenDatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / 'source'
        for folder in ('img', 'Control1', 'Control2', 'Control3'):
            (self.dataset / folder).mkdir(parents=True)
        self.loader = load_node('qwen_dataset_loader', self.root).QwenDatasetLoader()

    def assert_pixels(self, tensor, expected):
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (1, *expected.shape))
        np.testing.assert_array_equal(
            (tensor[0].numpy() * 255).round().astype(np.uint8), expected
        )

    def test_rgba_targets_and_all_controls_round_trip_in_every_mode(self):
        pixels = np.array([
            [[12, 34, 56, 0], [78, 90, 123, 64]],
            [[45, 67, 89, 128], [123, 234, 255, 255]],
        ], dtype=np.uint8)
        for folder in ('img', 'Control1', 'Control2', 'Control3'):
            Image.fromarray(pixels).save(self.dataset / folder / 'sample.png')
        (self.dataset / 'img' / 'sample.txt').write_text('alpha test', encoding='utf-8')
        saver = load_node('qwen_dataset_saver', self.root).QwenDatasetSaver()
        for mode in ('Manual', 'List', 'Random'):
            with self.subTest(mode=mode):
                result = self.loader.load_dataset(str(self.dataset), mode, 'sample.png')
                for output in result[:4]:
                    self.assert_pixels(output[0], pixels)
                self.assertEqual(result[4], ['alpha test'])
                saver.save_dataset(
                    result[0][0], mode, result[1][0], result[2][0], result[3][0]
                )
                for folder in ('img', 'Control1', 'Control2', 'Control3'):
                    with Image.open(self.root / mode / folder / 'image_00001.png') as saved:
                        self.assertEqual(saved.mode, 'RGBA')
                        np.testing.assert_array_equal(np.array(saved), pixels)

    def test_palette_grayscale_and_color_key_transparency(self):
        palette = Image.new('P', (2, 1))
        palette.putpalette([20, 40, 60, 80, 100, 120] + [0] * 762)
        palette.putdata([0, 1])
        palette.info['transparency'] = bytes([0, 128])
        grayscale = Image.new('LA', (2, 1))
        grayscale.putdata([(70, 0), (140, 128)])
        color_key = Image.new('RGB', (2, 1))
        color_key.putdata([(20, 40, 60), (80, 100, 120)])
        color_key.info['transparency'] = (20, 40, 60)
        for source in (palette, grayscale, color_key):
            with self.subTest(mode=source.mode):
                for folder in ('img', 'Control1'):
                    source.save(self.dataset / folder / 'sample.png')
                result = self.loader.load_dataset(str(self.dataset), 'Manual', 'sample.png')
                expected = np.array(source.convert('RGBA'))
                self.assert_pixels(result[0][0], expected)
                self.assert_pixels(result[1][0], expected)

    def test_saver_preserves_different_control_sizes_and_alpha_in_batch(self):
        saver = load_node('qwen_dataset_saver', self.root).QwenDatasetSaver()
        rng = np.random.default_rng(42)
        pixels = [
            rng.integers(0, 256, size=shape, dtype=np.uint8)
            for shape in ((2, 8, 6, 4), (1, 3, 9, 4), (2, 12, 4, 3), (2, 5, 7, 4))
        ]
        tensors = [torch.from_numpy(array.astype(np.float32) / 255.0) for array in pixels]
        saver.save_dataset(tensors[0], 'mixed_sizes', *tensors[1:], caption='different sizes')
        for index in range(2):
            filename = f'image_{index + 1:05d}'
            for folder, array in zip(('img', 'Control1', 'Control2', 'Control3'), pixels):
                expected = array[min(index, len(array) - 1)]
                with Image.open(self.root / 'mixed_sizes' / folder / f'{filename}.png') as saved:
                    self.assertEqual(saved.size, (expected.shape[1], expected.shape[0]))
                    self.assertEqual(saved.mode, 'RGBA' if expected.shape[2] == 4 else 'RGB')
                    np.testing.assert_array_equal(np.array(saved), expected)
            caption = self.root / 'mixed_sizes' / 'img' / f'{filename}.txt'
            self.assertEqual(caption.read_text(encoding='utf-8'), 'different sizes')

    def test_rgba_control_padding_is_transparent(self):
        Image.new('RGBA', (4, 4), (10, 20, 30, 128)).save(self.dataset / 'img' / 'sample.png')
        Image.new('RGBA', (4, 2), (40, 50, 60, 128)).save(self.dataset / 'Control1' / 'sample.png')
        result = self.loader.load_dataset(str(self.dataset), 'List')
        expected = np.zeros((4, 4, 4), dtype=np.uint8)
        expected[1:3] = (40, 50, 60, 128)
        self.assert_pixels(result[1][0], expected)

    def test_opaque_images_and_missing_controls_remain_rgb(self):
        Image.new('RGB', (4, 4), (10, 20, 30)).save(self.dataset / 'img' / 'sample.png')
        Image.new('RGB', (4, 2), (40, 50, 60)).save(self.dataset / 'Control1' / 'sample.png')
        result = self.loader.load_dataset(str(self.dataset), 'List')
        self.assert_pixels(result[0][0], np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8))
        expected = np.zeros((4, 4, 3), dtype=np.uint8)
        expected[1:3] = (40, 50, 60)
        self.assert_pixels(result[1][0], expected)
        for output in result[2:4]:
            self.assert_pixels(output[0], np.zeros((4, 4, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
