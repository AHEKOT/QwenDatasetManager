import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from trainer_service import TrainerService, TrainerValidationError, QWEN21_SOURCE_COMMIT


class Qwen21TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.datasets = self.root / 'Datasets'
        target = self.datasets / 'demo' / 'img'
        target.mkdir(parents=True)
        Image.new('RGB', (64, 64)).save(target / 'rgb.png')
        Image.new('RGBA', (64, 64), (50, 80, 100, 128)).save(target / 'rgba.png')
        self.service = TrainerService(self.root, lambda: self.datasets)

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, **kwargs):
        return dict(name='qwen21_test', model='qwen_image_2',
                    datasets=[{'name': 'demo'}], disableSampling=False,
                    samples=[{'prompt': 'A red kite'}], **kwargs)

    def test_text_to_image_defaults_and_native_rgba(self):
        for rgba in (False, True):
            config = self.service.build_job_config(self.payload(nativeRgba=rgba))[2]
            p = config['config']['process'][0]
            self.assertEqual(p['model']['arch'], 'qwen_image_2')
            self.assertEqual(p['model']['name_or_path'], 'Comfy-Org/Qwen-Image-2.1')
            self.assertEqual(p['model']['qtype'], 'convrot8')
            self.assertEqual(p['model']['qtype_te'], 'convrot8')
            self.assertEqual(p['model']['model_kwargs'], {'rgba': rgba, 'match_target_res': True})
            self.assertNotIn('vae_path', p['model'])
            self.assertFalse(p['train']['unload_text_encoder'])
            self.assertEqual(p['train']['timestep_type'], 'shift')
            self.assertEqual(p['sample']['guidance_scale'], 3)
            self.assertEqual(p['sample']['samples'], [{'prompt': 'A red kite'}])
            self.assertEqual(p['datasets'][0]['control_path'], [])
            self.assertNotIn('rgba_generate_control', p['datasets'][0])
            if rgba:
                self.assertEqual(p['sample']['format'], 'png')
            self.assertEqual(config['meta']['qdm']['upstreamCommit'], QWEN21_SOURCE_COMMIT)

    def test_edit_references_require_complete_pairs(self):
        control = self.datasets / 'demo' / 'Control1'
        control.mkdir()
        Image.new('RGBA', (64, 32)).save(control / 'rgb.png')
        with self.assertRaisesRegex(TrainerValidationError, 'matching reference'):
            self.service.build_job_config(self.payload())
        Image.new('RGBA', (64, 32)).save(control / 'rgba.png')
        p = self.service.build_job_config(self.payload())[2]['config']['process'][0]
        self.assertEqual(p['datasets'][0]['control_path'], [str(control)])

    def test_advanced_and_saved_form_keep_native_options(self):
        payload = self.payload(nativeRgba=True, matchTargetResolution=False,
                               cacheTextEmbeddings=True, unloadTextEncoder=True)
        config = self.service.build_job_config(payload)[2]
        advanced = copy.deepcopy(config['config']['process'][0])
        advanced['train']['unload_text_encoder'] = True
        advanced['model']['model_kwargs']['rgba'] = False
        advanced['sample']['format'] = 'jpg'
        payload['advancedProcess'] = advanced
        config = self.service.build_job_config(payload)[2]
        p = config['config']['process'][0]
        self.assertTrue(p['model']['model_kwargs']['rgba'])
        self.assertFalse(p['model']['model_kwargs']['match_target_res'])
        self.assertFalse(p['train']['unload_text_encoder'])
        self.assertEqual(p['sample']['format'], 'png')
        self.assertTrue(config['meta']['qdm']['form']['nativeRgba'])

    def test_native_vae_does_not_accept_legacy_transparent_preset(self):
        with self.assertRaisesRegex(TrainerValidationError, 'native RGBA'):
            self.service.build_job_config(self.payload(trainingPreset='transparent_lora'))


if __name__ == '__main__':
    unittest.main()
