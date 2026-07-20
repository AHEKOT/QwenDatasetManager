import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename, output_dir, lora_dir):
    folder_paths = types.ModuleType('folder_paths')
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_folder_paths = lambda category: [str(lora_dir)]
    folder_paths.get_full_path = lambda category, value: str(lora_dir / value)
    folder_paths.get_filename_list = lambda category: []
    previous = sys.modules.get('folder_paths')
    previous_numpy = sys.modules.get('numpy')
    sys.modules['folder_paths'] = folder_paths
    if previous_numpy is None:
        sys.modules['numpy'] = types.ModuleType('numpy')
    try:
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop('folder_paths', None)
        else:
            sys.modules['folder_paths'] = previous
        if previous_numpy is None:
            sys.modules.pop('numpy', None)


class ComfyPathSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.output = self.root / 'output'
        self.loras = self.root / 'loras'
        self.output.mkdir()
        self.loras.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_saver_rejects_dataset_traversal(self):
        module = load_module(
            'qdm_saver_test',
            'comfyui_qwenDatasetManager/qwen_dataset_saver.py',
            self.output,
            self.loras
        )
        saver = module.QwenDatasetSaver()
        with self.assertRaises(ValueError):
            saver.resolve_dataset_path('../escaped')
        self.assertEqual(saver.resolve_dataset_path('safe'), (self.output / 'safe').resolve())

    def test_lora_output_subfolder_cannot_escape(self):
        module = load_module(
            'qdm_lora_test',
            'comfyui_qwenDatasetManager/qwen_lora_merge.py',
            self.output,
            self.loras
        )
        with self.assertRaises(ValueError):
            module._resolve_output_dir('ComfyUI output', '../../escaped')
        self.assertEqual(
            Path(module._resolve_output_dir('ComfyUI output', 'merged')),
            (self.output / 'merged').resolve()
        )

    def test_lora_input_is_limited_to_known_roots(self):
        module = load_module(
            'qdm_lora_input_test',
            'comfyui_qwenDatasetManager/qwen_lora_merge.py',
            self.output,
            self.loras
        )
        outside = self.root / 'outside.safetensors'
        outside.write_bytes(b'not a real tensor')
        with self.assertRaises(FileNotFoundError):
            module._resolve_lora_path(str(outside))


if __name__ == '__main__':
    unittest.main()
