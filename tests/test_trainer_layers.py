"""Application contract tests; PSD decoding lives in trainer/test_qwen21_layers.py."""
import tempfile
import json
import unittest
import zipfile
from pathlib import Path

from flask import Flask
from PIL import Image
from trainer_service import TrainerService, TrainerValidationError, create_trainer_blueprint


class LayeredPresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.datasets = self.root / 'Datasets'
        self.dataset = self.datasets / 'layers'
        (self.dataset / 'img').mkdir(parents=True)
        # Saving a job is metadata-only; it must not invoke a PSD decoder.
        for name in ('paired', 'scratch'):
            (self.dataset / 'img' / f'{name}.psd').write_bytes(b'not decoded by web service')
            (self.dataset / 'img' / f'{name}.txt').write_text('A layered scene', encoding='utf-8')
        (self.dataset / 'Control1').mkdir()
        Image.new('RGB', (64, 64)).save(self.dataset / 'Control1' / 'paired.png')
        self.service = TrainerService(self.root, lambda: self.datasets)

    def payload(self, **overrides):
        return dict(name='layer_test', model='qwen_image_2', trainingPreset='qwen_layered_lora',
                    datasets=[{'name': 'layers'}], layerSlots=4, batchSize=1,
                    disableSampling=False, samples=[{'prompt': 'A layered scene'}], **overrides)

    def test_mixed_optional_controls_and_joint_architecture(self):
        config = self.service.build_job_config(self.payload())[2]
        process = config['config']['process'][0]
        self.assertEqual(process['model']['arch'], 'qwen_image_2_layered')
        self.assertEqual(process['model']['model_kwargs']['layer_slots'], 4)
        self.assertTrue(process['model']['model_kwargs']['rgba'])
        self.assertNotIn('vae_path', process['model'])
        self.assertEqual(process['datasets'][0]['target_format'], 'psd_layers')
        self.assertEqual(process['datasets'][0]['layer_slots'], 4)
        self.assertEqual(process['datasets'][0]['control_path'], [str(self.dataset / 'Control1')])
        self.assertEqual(process['sample']['format'], 'png')
        self.assertEqual(process['sample']['samples'], [{'prompt': 'A layered scene'}])
        inspection = self.service.inspect_dataset('layers')
        self.assertEqual(inspection['psdTargetCount'], 2)
        self.assertEqual(inspection['psdPairedCount'], 1)
        self.assertEqual(inspection['psdGenerationCount'], 1)
        self.assertTrue(inspection['psdValid'])
        self.assertEqual(config['meta']['qdm']['layerFormat']['order'], 'bottom_to_top')
        self.assertEqual(process['model']['model_kwargs']['timestep_embedding'], 'fp32')
        self.assertEqual(config['meta']['qdm']['layerFormat']['timestepEmbedding'], 'fp32')

    def test_edit_preserves_old_job_timestep_contract(self):
        job, _ = self.service.create_job(self.payload())
        config = json.loads(self.service._get_job_row(job['id'])['job_config'])
        config['config']['process'][0]['model']['model_kwargs'].pop('timestep_embedding')
        config['meta']['qdm']['layerFormat'].pop('timestepEmbedding')
        with self.service.connect() as db:
            db.execute('UPDATE "Job" SET job_config = ? WHERE id = ?', (json.dumps(config), job['id']))
        self.service.update_job(job['id'], self.payload())
        saved = json.loads(self.service._get_job_row(job['id'])['job_config'])
        self.assertEqual(saved['config']['process'][0]['model']['model_kwargs']['timestep_embedding'], 'legacy_bf16')
        self.assertEqual(saved['meta']['qdm']['layerFormat']['timestepEmbedding'], 'legacy_bf16')

    def test_rgb_targets_do_not_substitute_for_psds(self):
        for path in (self.dataset / 'img').glob('*.psd'):
            path.unlink()
        Image.new('RGB', (64, 64)).save(self.dataset / 'img' / 'plain.png')
        with self.assertRaisesRegex(TrainerValidationError, 'PSD'):
            self.service.build_job_config(self.payload())

    def test_invalid_slot_counts_and_unsafe_options(self):
        for key, value in [('layerSlots', 0), ('layerSlots', 21), ('layerSlots', 2.5),
                           ('layerSlots', True), ('batchSize', 2), ('guidanceLoss', True),
                           ('validationEnabled', True), ('lossType', 'wavelet'),
                           ('advancedProcess', {'model': {}})]:
            payload = self.payload()
            payload[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(TrainerValidationError):
                self.service.build_job_config(payload)

    def test_layered_dataset_batch_override_cannot_exceed_one(self):
        payload = self.payload()
        payload['datasets'][0]['batchSizeOverride'] = 2
        with self.assertRaisesRegex(TrainerValidationError, 'Batch size override'):
            self.service.build_job_config(payload)

    def test_ambiguous_control_and_extra_control_are_rejected(self):
        Image.new('RGB', (64, 64)).save(self.dataset / 'Control1' / 'paired.jpg')
        with self.assertRaisesRegex(TrainerValidationError, 'Ambiguous'):
            self.service.build_job_config(self.payload())
        (self.dataset / 'Control1' / 'paired.jpg').unlink()
        (self.dataset / 'Control2').mkdir()
        Image.new('RGB', (64, 64)).save(self.dataset / 'Control2' / 'paired.png')
        with self.assertRaisesRegex(TrainerValidationError, 'Control2'):
            self.service.build_job_config(self.payload())

    def test_new_preset_does_not_change_standard_qi2_validation(self):
        payload = self.payload()
        payload['trainingPreset'] = 'standard_lora'
        with self.assertRaisesRegex(TrainerValidationError, 'no target images'):
            self.service.build_job_config(payload)

    def test_saved_psd_job_can_be_queued_with_optional_controls(self):
        # Only a temporary database is queued; no worker or training is started.
        job, _ = self.service.create_job(self.payload())
        self.assertEqual(self.service.queue_job(job['id'])['status'], 'queued')
        self.service.stop_job(job['id'])
        (self.dataset / 'Control1' / 'paired.png').unlink()
        self.assertEqual(self.service.queue_job(job['id'])['status'], 'queued')

    def test_queue_rechecks_psd_targets_and_controls_after_saving(self):
        job, _ = self.service.create_job(self.payload())
        Image.new('RGB', (64, 64)).save(self.dataset / 'Control1' / 'paired.jpg')
        with self.assertRaisesRegex(TrainerValidationError, 'Ambiguous'):
            self.service.queue_job(job['id'])
        (self.dataset / 'Control1' / 'paired.jpg').unlink()
        for path in (self.dataset / 'img').glob('*.psd'):
            path.unlink()
        # An ordinary paired image cannot replace PSD targets for a saved job.
        Image.new('RGB', (64, 64)).save(self.dataset / 'img' / 'paired.png')
        with self.assertRaisesRegex(TrainerValidationError, 'No PSD targets'):
            self.service.queue_job(job['id'])
        self.assertEqual(self.service.get_job(job['id'])['status'], 'stopped')

    def test_layered_preflight_counts_psds_and_accepts_optional_controls(self):
        app = Flask(__name__)
        app.register_blueprint(create_trainer_blueprint(self.service))
        client = app.test_client()
        payload = dict(model='qwen_image_2', trainingPreset='qwen_layered_lora', datasets=['layers'])
        for remove_control in (False, True):
            if remove_control:
                (self.dataset / 'Control1' / 'paired.png').unlink()
            response = client.post('/api/trainer/preflight', json=payload)
            self.assertEqual(response.status_code, 200)
            result = response.get_json()
            self.assertTrue(result['valid'])
            self.assertEqual(result['datasets'][0]['targetCount'], 2)
            self.assertEqual(result['datasets'][0]['captionCount'], 2)
            self.assertEqual(result['datasets'][0]['warnings'], [])
        payload['trainingPreset'] = 'standard_lora'
        self.assertFalse(client.post('/api/trainer/preflight', json=payload).get_json()['valid'])
        payload['trainingPreset'] = 'qwen_layered_lora'
        for path in (self.dataset / 'img').glob('*.psd'):
            path.unlink()
        result = client.post('/api/trainer/preflight', json=payload).get_json()
        self.assertFalse(result['valid'])
        self.assertIn('No PSD targets in img/', result['datasets'][0]['psdErrors'])

    def test_psd_sample_download_archive_and_delete(self):
        job, _ = self.service.create_job(self.payload())
        _, folder = self.service._job_samples_dir(job['id'])
        folder.mkdir(parents=True, exist_ok=True)
        preview = folder / '000000010_0.png'
        Image.new('RGBA', (32, 32)).save(preview)
        psd = preview.with_suffix('.psd')
        psd.write_bytes(b'PSD sidecar fixture')
        self.assertEqual(self.service.list_job_samples(job['id'])['samples'][0]['psdFile'], psd.name)
        self.assertEqual(self.service.resolve_job_sample(job['id'], psd.name), psd.resolve())
        archive, _ = self.service.build_samples_archive(job['id'])
        with zipfile.ZipFile(archive) as saved:
            self.assertIn(f'samples/{psd.name}', saved.namelist())
        self.service.delete_job_sample(job['id'], preview.name)
        self.assertFalse(preview.exists())
        self.assertFalse(psd.exists())


if __name__ == '__main__':
    unittest.main()
