import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

import app as manager
from auto_caption import normalize_auto_caption_config, scan_gguf_catalog


class FakeCaptionEngine:
    def __init__(self):
        self.loaded = None

    def ensure_loaded(self, variant):
        self.loaded = variant['filename']

    def caption(self, image_path, config):
        return f"{config['prefix']}caption for {image_path.stem}{config['suffix']}"


class AutoCaptionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.datasets = self.root / 'Datasets'
        self.models = self.root / 'models' / 'llm'
        self.datasets.mkdir()
        self.models.mkdir(parents=True)
        dataset = self.datasets / 'demo'
        for folder in manager.DATASET_IMAGE_FOLDERS:
            (dataset / folder).mkdir(parents=True)
        Image.new('RGB', (12, 8), (20, 40, 60)).save(dataset / 'img' / 'one.png')
        Image.new('RGB', (12, 8), (80, 40, 20)).save(dataset / 'img' / 'two.png')

        self.original_datasets = manager.DATASETS_DIR
        self.original_models = manager.MODELS_LLM_DIR
        self.original_engine = manager.AUTO_CAPTION_ENGINE
        manager.DATASETS_DIR = self.datasets
        manager.MODELS_LLM_DIR = self.models
        manager.AUTO_CAPTION_ENGINE = FakeCaptionEngine()
        manager.ACTIVE_DATASETS.clear()
        manager.AUTO_CAPTION_JOBS.clear()
        manager.app.config.update(TESTING=True)
        self.client = manager.app.test_client()

    def tearDown(self):
        manager.ACTIVE_DATASETS.clear()
        manager.AUTO_CAPTION_JOBS.clear()
        manager.DATASETS_DIR = self.original_datasets
        manager.MODELS_LLM_DIR = self.original_models
        manager.AUTO_CAPTION_ENGINE = self.original_engine
        self.tempdir.cleanup()

    def add_qwen_pair(self):
        (self.models / 'Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf').touch()
        (self.models / 'Qwen2.5-VL-7B-Instruct-Q8_0.gguf').touch()
        (self.models / 'mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf').touch()
        catalog = scan_gguf_catalog(self.models)
        model = catalog['models'][0]
        return catalog, model, model['variants'][0]

    def wait_for_job(self, job_id):
        body = None
        for _ in range(200):
            response = self.client.get(f'/api/auto-caption/jobs/{job_id}')
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            if body['finished']:
                return body
            time.sleep(0.01)
        self.fail(f'Auto Caption job {job_id} did not finish: {body}')

    def test_scanner_groups_quantizations_and_pairs_mmproj(self):
        catalog, model, _variant = self.add_qwen_pair()
        self.assertEqual(catalog['warnings'], [])
        self.assertEqual(model['label'], 'Qwen2.5-VL-7B-Instruct')
        self.assertEqual(
            {variant['quantization'] for variant in model['variants']},
            {'Q4_K_M', 'Q8_0'}
        )
        self.assertTrue(all(variant['handler'] == 'qwen25-vl' for variant in model['variants']))
        self.assertTrue(all(
            variant['mmprojFilename'] == 'mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf'
            for variant in model['variants']
        ))

    def test_unpaired_files_are_not_selectable(self):
        (self.models / 'orphan-Q4_K_M.gguf').touch()
        catalog = scan_gguf_catalog(self.models)
        self.assertEqual(catalog['models'], [])
        self.assertIn('matching mmproj GGUF was not found', catalog['warnings'][0])

    def test_config_requires_a_nonempty_system_prompt(self):
        with self.assertRaisesRegex(ValueError, 'System prompt cannot be empty'):
            normalize_auto_caption_config({'systemPrompt': '   '})

    def test_preview_and_apply_jobs_generate_caption_files_with_backup(self):
        _catalog, model, variant = self.add_qwen_pair()
        config = {
            'systemPrompt': 'Describe the image.',
            'modelId': model['id'],
            'variantId': variant['id'],
            'prefix': 'prefix, ',
            'suffix': ', suffix',
        }

        preview = self.client.post('/api/auto-caption/preview', json={
            'folder': 'demo', 'config': config
        })
        self.assertEqual(preview.status_code, 200)
        preview_job = self.wait_for_job(preview.get_json()['jobId'])
        self.assertEqual(preview_job['status'], 'completed')
        self.assertIn(preview_job['result']['filename'], {'one.png', 'two.png'})
        self.assertTrue(preview_job['result']['caption'].startswith('prefix, caption for'))

        existing = self.datasets / 'demo' / 'img' / 'one.txt'
        existing.write_text('old caption', encoding='utf-8')
        applied = self.client.post('/api/auto-caption/apply', json={
            'folder': 'demo', 'config': config
        })
        self.assertEqual(applied.status_code, 200)
        apply_job = self.wait_for_job(applied.get_json()['jobId'])
        self.assertEqual(apply_job['status'], 'completed')
        self.assertEqual(apply_job['result']['generated'], 2)
        self.assertEqual(
            existing.read_text(encoding='utf-8'),
            'prefix, caption for one, suffix'
        )
        self.assertEqual(
            (self.datasets / 'demo' / apply_job['result']['backup'] / 'one.txt').read_text(encoding='utf-8'),
            'old caption'
        )
        self.assertNotIn('demo', manager.ACTIVE_DATASETS)


if __name__ == '__main__':
    unittest.main()
