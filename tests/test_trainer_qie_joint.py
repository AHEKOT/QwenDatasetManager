import io
import unittest
from pathlib import Path
from PIL import Image
from werkzeug.datastructures import FileStorage
import test_trainer_service as fixtures
from trainer_service import TrainerValidationError


class QieJointServiceTests(unittest.TestCase):
    setUp = fixtures.TrainerServiceTests.setUp
    tearDown = fixtures.TrainerServiceTests.tearDown
    make_dataset = fixtures.TrainerServiceTests.make_dataset
    make_rgba_dataset = fixtures.TrainerServiceTests.make_rgba_dataset
    default_payload = fixtures.TrainerServiceTests.default_payload

    def payload(self, name='demo'):
        vae = self.project_root / 'qie-rgba.safetensors'
        vae.write_bytes(b'test')
        payload = self.default_payload([{'name': name, 'rgbaControlMode': 'paired', 'resolutions': [512]}])
        payload.update(trainingPreset='transparent_lora', vaePath=str(vae),
                       disableSampling=True, cacheTextEmbeddings=True, unloadTextEncoder=True)
        payload['name'] = f'paired_{name}'
        return payload

    def test_rgb_rgba_and_mixed_datasets_keep_real_controls_and_full_targets(self):
        for kind in ('rgb', 'rgba', 'mixed'):
            root = self.make_dataset(kind) if kind != 'rgba' else self.make_rgba_dataset(kind, control_count=2)
            if kind == 'rgb':
                for source in (root / 'img').glob('*.png'):
                    with Image.open(source) as image:
                        image.save(source.with_suffix('.jpg'))
                    source.unlink()
            if kind == 'mixed':
                Image.new('RGBA', (16, 12), (200, 20, 40, 128)).save(root / 'img' / 'two.png')
            payload = self.payload(kind)
            _, _, config, _ = self.service.build_job_config(payload)
            process = config['config']['process'][0]
            ds = process['datasets'][0]
            self.assertEqual(process['model']['arch'], 'qwen_image_edit_plus_rgba')
            self.assertEqual(ds['pixel_channels'], 'rgba')
            self.assertFalse(ds['rgba_require_alpha'])
            self.assertFalse(ds['rgba_generate_control'])
            self.assertEqual(len(ds['control_path']), 2)
            self.assertTrue(ds['load_image_when_caching_latents'])
            self.assertEqual(ds['rgba_edge_color_correction'], 'none')
            self.assertTrue(process['train']['cache_text_embeddings'])
            self.assertFalse(process['train']['unload_text_encoder'])
            self.assertNotIn('rgba_control_background_path', ds)
            job, _ = self.service.create_job(payload)
            self.service.queue_job(job['id'])

    def test_paired_validation_uses_uploaded_rgb_target_and_real_inputs(self):
        self.make_dataset()
        payload = self.payload()
        def upload(rgb, name, method):
            stream = io.BytesIO()
            Image.new('RGB', (20, 16), rgb).save(stream, format='JPEG')
            stream.seek(0)
            return method(FileStorage(stream=stream, filename=name))
        target = upload((200, 20, 40), 'target.jpg', self.service.save_validation_image)
        control = upload((20, 40, 200), 'input.jpg', self.service.save_sample_image)
        payload.update(validationEnabled=True, validationItems=[{
            'targetPath': str(target), 'mode': 'paired', 'ctrlImg1': str(control),
            'prompt': 'Change the blue object to red.',
        }])
        _, _, config, _ = self.service.build_job_config(payload)
        item = config['config']['process'][0]['train']['validation_config']['validation_items'][0]
        self.assertEqual(item['control_paths'], [str(control)])
        self.assertEqual(item['image_path'], str(target))
        payload['validationItems'][0].pop('ctrlImg1')
        with self.assertRaisesRegex(TrainerValidationError, 'input Control'):
            self.service.build_job_config(payload)

    def test_missing_pairs_are_rejected_on_save_and_queue(self):
        root = self.make_dataset()
        payload = self.payload()
        job, _ = self.service.create_job(payload)
        (root / 'Control2' / 'one.jpg').unlink()
        with self.assertRaisesRegex(TrainerValidationError, 'matching Control'):
            self.service.build_job_config(payload)
        with self.assertRaisesRegex(TrainerValidationError, 'not ready'):
            self.service.queue_job(job['id'])

    def test_managed_validation_resolves_all_matching_controls(self):
        root = self.make_dataset()
        payload = self.payload()
        payload.update(validationEnabled=True, validationItems=[{'dataset': 'demo', 'image': 'one.png'}])
        _, _, config, _ = self.service.build_job_config(payload)
        item = config['config']['process'][0]['train']['validation_config']['validation_items'][0]
        self.assertEqual(item['rgba_control_mode'], 'paired')
        self.assertEqual(item['control_paths'], [str(root / 'Control1' / 'one.jpg'), str(root / 'Control2' / 'one.jpg')])
