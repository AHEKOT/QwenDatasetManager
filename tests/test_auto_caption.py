import base64
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

import app as manager
from auto_caption import _image_data_uri, normalize_auto_caption_config, scan_gguf_catalog


class FakeCaptionEngine:
    def __init__(self):
        self.loaded = None

    def ensure_loaded(self, variant):
        self.loaded = variant['filename']

    def caption(self, image_path, config):
        return f"{config['prefix']}caption for {image_path.stem}{config['suffix']}"


class BlockingCaptionEngine(FakeCaptionEngine):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def caption(self, image_path, config):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError('Test caption engine timed out')
        return super().caption(image_path, config)


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

    def test_scanner_uses_mtmd_handler_for_qwen35(self):
        (self.models / 'Qwen3.5-4B-Q8_0.gguf').touch()
        (self.models / 'mmproj-Qwen3.5-4B-BF16.gguf').touch()

        catalog = scan_gguf_catalog(self.models)

        self.assertEqual(catalog['warnings'], [])
        self.assertEqual(catalog['models'][0]['label'], 'Qwen3.5-4B')
        self.assertEqual(catalog['models'][0]['variants'][0]['handler'], 'mtmd')

    def test_vl_image_is_downscaled_without_changing_source(self):
        image_path = self.root / 'large.png'
        Image.new('RGBA', (2304, 3456), (20, 40, 60, 128)).save(image_path)

        data_uri = _image_data_uri(image_path)

        self.assertTrue(data_uri.startswith('data:image/png;base64,'))
        encoded = data_uri.split(',', 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as vl_image:
            self.assertEqual(vl_image.size, (683, 1024))
            self.assertEqual(vl_image.mode, 'RGBA')
        with Image.open(image_path) as source:
            self.assertEqual(source.size, (2304, 3456))

    def test_unpaired_files_are_not_selectable(self):
        (self.models / 'orphan-Q4_K_M.gguf').touch()
        catalog = scan_gguf_catalog(self.models)
        self.assertEqual(catalog['models'], [])
        self.assertIn('matching mmproj GGUF was not found', catalog['warnings'][0])

    def test_config_requires_a_nonempty_system_prompt(self):
        with self.assertRaisesRegex(ValueError, 'System prompt cannot be empty'):
            normalize_auto_caption_config({'systemPrompt': '   '})

    def test_auto_caption_config_is_persisted_per_dataset(self):
        second = self.datasets / 'second'
        for folder in manager.DATASET_IMAGE_FOLDERS:
            (second / folder).mkdir(parents=True)

        first_config = {
            'systemPrompt': 'Prompt for demo',
            'modelId': '',
            'variantId': '',
            'prefix': 'demo prefix, ',
            'suffix': ', demo suffix',
        }
        second_config = {
            'systemPrompt': 'Prompt for second',
            'modelId': '',
            'variantId': '',
            'prefix': 'second prefix, ',
            'suffix': ', second suffix',
        }

        self.assertEqual(
            self.client.post('/api/auto-caption/config?folder=demo', json=first_config).status_code,
            200,
        )
        self.assertEqual(
            self.client.post('/api/auto-caption/config?folder=second', json=second_config).status_code,
            200,
        )

        self.assertEqual(
            self.client.get('/api/auto-caption/config?folder=demo').get_json(),
            first_config,
        )
        self.assertEqual(
            self.client.get('/api/auto-caption/config?folder=second').get_json(),
            second_config,
        )
        self.assertTrue((self.datasets / 'demo' / '.auto_caption_config.json').is_file())
        self.assertTrue((second / '.auto_caption_config.json').is_file())

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

    def test_apply_missing_skips_existing_nonempty_captions(self):
        _catalog, model, variant = self.add_qwen_pair()
        config = {
            'systemPrompt': 'Describe the image.',
            'modelId': model['id'],
            'variantId': variant['id'],
            'prefix': '',
            'suffix': '',
        }
        existing = self.datasets / 'demo' / 'img' / 'one.txt'
        existing.write_text('keep this caption', encoding='utf-8')

        response = self.client.post('/api/auto-caption/apply', json={
            'folder': 'demo',
            'config': config,
            'missingOnly': True,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['totalItems'], 1)
        job = self.wait_for_job(response.get_json()['jobId'])
        self.assertEqual(job['status'], 'completed')
        self.assertTrue(job['result']['missingOnly'])
        self.assertEqual(job['result']['generated'], 1)
        self.assertEqual(existing.read_text(encoding='utf-8'), 'keep this caption')
        self.assertEqual(
            (self.datasets / 'demo' / 'img' / 'two.txt').read_text(encoding='utf-8'),
            'caption for two',
        )

    def test_stop_finishes_current_image_and_skips_remaining_items(self):
        _catalog, model, variant = self.add_qwen_pair()
        config = {
            'systemPrompt': 'Describe the image.',
            'modelId': model['id'],
            'variantId': variant['id'],
            'prefix': '',
            'suffix': '',
        }
        engine = BlockingCaptionEngine()
        manager.AUTO_CAPTION_ENGINE = engine

        response = self.client.post('/api/auto-caption/apply', json={
            'folder': 'demo',
            'config': config,
        })
        self.assertEqual(response.status_code, 200)
        job_id = response.get_json()['jobId']
        try:
            self.assertTrue(engine.started.wait(timeout=2))
            stopped = self.client.post(f'/api/auto-caption/jobs/{job_id}/stop')
            self.assertEqual(stopped.status_code, 200)
        finally:
            engine.release.set()

        job = self.wait_for_job(job_id)
        self.assertEqual(job['status'], 'stopped')
        self.assertTrue(job['result']['stopped'])
        self.assertEqual(job['result']['generated'], 1)
        self.assertEqual(
            [item['status'] for item in job['items']],
            ['completed', 'stopped'],
        )
        self.assertTrue((self.datasets / 'demo' / 'img' / 'one.txt').is_file())
        self.assertFalse((self.datasets / 'demo' / 'img' / 'two.txt').exists())


if __name__ == '__main__':
    unittest.main()
