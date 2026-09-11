import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from werkzeug.datastructures import FileStorage

from trainer_service import TrainerService, TrainerValidationError
from trainer_h3 import H3_KEYS


class H3TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'models').mkdir()
        self.datasets = self.root / 'Datasets'
        self.target = self.datasets / 'mixed' / 'img'
        self.target.mkdir(parents=True)
        Image.new('RGB', (32, 32)).save(self.target / 'still.png')
        (self.target / 'still.txt').write_text('A still image')
        (self.target / 'clip.mp4').write_bytes(b'metadata-only-preflight')
        (self.target / 'clip.txt').write_text('A video with audio')
        self.service = TrainerService(self.root, lambda: self.datasets)

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, model='minimax_h3', **kw):
        return {'name': 'h3_test', 'model': model, 'datasets': [{'name': 'mixed'}],
                'disableSampling': False, 'samples': [{'prompt': 'A scene'}], **kw}

    def process(self, payload):
        return self.service.build_job_config(payload)[2]['config']['process'][0]

    def test_both_architecture_defaults_and_text_only_samples(self):
        for model in H3_KEYS:
            with self.subTest(model=model):
                p = self.process(self.payload(model))
                self.assertEqual(p['model']['arch'], model)
                self.assertEqual(p['model']['qtype'], 'convrot8')
                self.assertEqual(p['model']['qtype_te'], 'nvfp4')
                self.assertEqual(p['network']['linear'], 16)
                self.assertEqual(p['network']['network_kwargs']['ignore_if_contains'], ['adaln_proj'])
                self.assertEqual(p['train']['timestep_type'], 'shift')
                self.assertEqual(p['train']['guidance_loss_target'], 3.5)
                self.assertTrue(p['train']['cache_text_embeddings'])
                self.assertEqual(p['sample']['guidance_scale'], 1)
                self.assertEqual(p['sample']['sample_steps'], 28)
                self.assertEqual(p['sample']['num_frames'], 107)
                self.assertEqual(p['sample']['fps'], 24)
                self.assertEqual(p['sample']['samples'], [{'prompt': 'A scene'}])
                self.assertTrue(p['datasets'][0]['do_audio'])
                self.assertTrue(p['datasets'][0]['auto_frame_count'])
                self.assertTrue(p['datasets'][0]['cache_latents_to_disk'])
                self.assertNotIn('control_path', p['datasets'][0])

    def test_distillation_modes_do_not_leave_stale_settings(self):
        for model in H3_KEYS:
            for mode in ('both', 'cg', 'ta', 'none'):
                with self.subTest(model=model, mode=mode):
                    p = self.process(self.payload(model, distillationMethod=mode))
                    self.assertEqual(p['train']['do_guidance_loss'], mode in ('both', 'cg'))
                    self.assertEqual('assistant_lora_path' in p['model'], mode in ('both', 'ta'))
                    self.assertEqual('guidance_loss_target' in p['train'], mode in ('both', 'cg'))
                    if mode in ('both', 'ta'):
                        self.assertEqual('ref2va' in p['model']['assistant_lora_path'], model.endswith('ref2va'))

    def test_dopsd_caches_pixels_and_ignores_external_controls(self):
        control = self.target.parent / 'Control1'
        control.mkdir()
        Image.new('RGB', (32, 32)).save(control / 'still.png')
        p = self.process(self.payload('minimax_h3_ref2va', distillationMethod='dopsd',
                                      cacheTextEmbeddings=False, h3DopsdBleedStrength=0.25))
        self.assertTrue(p['model']['model_kwargs']['dopsd'])
        self.assertEqual(p['model']['model_kwargs']['dopsd_bleed_strength'], 0.25)
        self.assertNotIn('assistant_lora_path', p['model'])
        self.assertFalse(p['train']['do_guidance_loss'])
        self.assertTrue(p['train']['cache_text_embeddings'])
        self.assertTrue(p['datasets'][0]['cache_tensors_to_disk'])
        self.assertNotIn('control_path', p['datasets'][0])
        with self.assertRaises(TrainerValidationError):
            self.process(self.payload(distillationMethod='dopsd'))

    def test_local_components_and_reference_presentation_round_trip(self):
        weights = self.root / 'weights'
        weights.mkdir()
        file = weights / 'local.safetensors'
        file.write_bytes(b'test')
        payload = self.payload('minimax_h3_ref2va', h3Partition='ref2va', h3DitPath=str(file),
                               h3TextEncoderPath=str(file), h3VideoVaePath=str(file), h3AudioVaePath=str(file),
                               assistantLoraPath=str(file), modelsPath=str(weights), h3ConfigPath=str(weights),
                               h3ImageRefsAsVideo=True, h3ImageRefVideoFrames=22, h3MaxTextLength=0,
                               h3LocalOnly=True, h3SampleAudio=False, sampleFrames=1,
                               datasets=[{'name': 'mixed', 'numFrames': 56, 'fps': 12, 'doAudio': False,
                                          'audioNormalize': True, 'audioPreservePitch': True, 'autoFrameCount': False}])
        job, _ = self.service.create_job(payload)
        restored = json.loads(self.service._get_job_row(job['id'])['job_config'])
        p = restored['config']['process'][0]
        kw = p['model']['model_kwargs']
        self.assertEqual(kw['dit_ref2va_path'], str(file.resolve()))
        self.assertTrue(kw['local_files_only'])
        self.assertTrue(kw['image_refs_as_video'])
        self.assertEqual(kw['image_ref_video_frames'], 22)
        self.assertEqual(kw['max_text_length'], 0)
        self.assertFalse(kw['sample_audio'])
        self.assertEqual(p['sample']['format'], 'png')
        self.assertFalse(p['datasets'][0]['do_audio'])
        self.assertTrue(p['datasets'][0]['audio_preserve_pitch'])
        self.assertEqual(p, self.process(restored['meta']['qdm']['form']))

    def test_video_preflight_keeps_image_model_rules(self):
        with patch('trainer_service.Image.open', side_effect=AssertionError('No media decoding in preflight')):
            inspection = self.service.inspect_dataset('mixed')
            self.assertEqual(inspection['targetCount'], 1)
            self.assertEqual(inspection['mediaTargetCount'], 2)
            self.assertTrue(inspection['minimaxValid'])
            self.process(self.payload())
        with self.assertRaises(TrainerValidationError):
            self.process(self.payload('qwen_image_edit_2511'))

    def test_invalid_partition_frames_and_rgba_combination(self):
        for values in ({'h3Partition': 'ref2va'}, {'sampleFrames': 40},
                       {'trainingPreset': 'transparent_lora'}, {'h3DitPath': str(self.root / 'missing')}):
            with self.subTest(values=values), self.assertRaises(TrainerValidationError):
                self.process(self.payload(**values))

    def test_control_selection_and_video_samples(self):
        for i in (1, 2):
            folder = self.target.parent / f'Control{i}'
            folder.mkdir()
            (folder / 'clip.mp4').write_bytes(b'reference')
        uploaded = self.service.save_sample_image(FileStorage(stream=io.BytesIO(b'video'), filename='ref.mp4'))
        payload = self.payload('minimax_h3_ref2va', datasets=[{'name': 'mixed', 'controls': ['Control2']}],
                               samples=[{'prompt': 'Use reference', 'ctrlImg1': str(uploaded)}])
        p = self.process(payload)
        self.assertEqual(p['datasets'][0]['control_path'], [str(self.target.parent / 'Control2')])
        job, _ = self.service.create_job(payload)
        samples = self.service.output_dir / job['name'] / 'samples'
        samples.mkdir(parents=True)
        file = samples / 'h3_test_000000100_0.mp4'
        file.write_bytes(b'generated-video')
        info = self.service.list_job_samples(job['id'])['samples'][0]
        self.assertEqual(info['mediaType'], 'video')
        self.assertEqual(info['controlMediaTypes'], ['video'])
        self.assertEqual(self.service.resolve_job_sample(job['id'], file.name), file)
        self.assertTrue(self.service.build_samples_archive(job['id'])[0].is_file())
        self.service.delete_job_sample(job['id'], file.name)
        self.assertFalse(file.exists())
        with self.assertRaises(TrainerValidationError):
            self.process({**payload, 'model': 'minimax_h3'})

    def test_advanced_h3_settings_survive_but_dopsd_contract_is_checked(self):
        p = self.process(self.payload('minimax_h3_ref2va'))
        p['model']['model_kwargs']['max_text_length'] = 777
        p['model']['model_kwargs']['image_refs_as_video'] = True
        result = self.process(self.payload('minimax_h3_ref2va', advancedProcess=p))
        self.assertEqual(result['model']['model_kwargs']['max_text_length'], 777)
        p['model']['model_kwargs']['dopsd'] = True
        with self.assertRaises(TrainerValidationError):
            self.process(self.payload('minimax_h3_ref2va', advancedProcess=p))

    def rgba_dataset(self):
        root = self.datasets / 'alpha' / 'img'
        root.mkdir(parents=True)
        for i in range(3):
            Image.new('RGBA', (32, 32), (12, 45, 67, 128)).save(root / f'{i}.png')
            (root / f'{i}.txt').write_text('transparent subject')
        vae = self.root / 'models' / 'vae' / 'minimax_h3_rgba_vae.safetensors'
        vae.parent.mkdir(parents=True)
        vae.write_bytes(b'header checked by the runtime')
        return vae

    def test_rgba_generation_and_edit_for_both_architectures(self):
        vae = self.rgba_dataset()
        for model in H3_KEYS:
            p = self.process(self.payload(model, trainingPreset='transparent_lora', vaePath=str(vae),
                datasets=[{'name': 'alpha', 'rgbaControlMode': 'generation'}]))
            self.assertEqual(p['model']['arch'], model + '_rgba')
            self.assertEqual(p['model']['models_path'], str(self.root / 'models'))
            self.assertEqual(p['model']['vae_path'], str(vae))
            self.assertTrue(p['train']['do_guidance_loss'])
            self.assertEqual(p['sample']['format'], 'png')
            ds = p['datasets'][0]
            self.assertEqual(ds['pixel_channels'], 'rgba')
            self.assertEqual(ds['num_frames'], 1)
            self.assertFalse(ds['auto_frame_count'])
            self.assertFalse(ds['do_audio'])
            self.assertFalse(ds['rgba_generate_control'])
            self.assertTrue(ds['cache_tensors_to_disk'])
            p = self.process(self.payload(model, trainingPreset='transparent_lora', vaePath=str(vae),
                datasets=[{'name': 'alpha', 'rgbaControlMode': 'edit', 'rgbaBackgroundDataset': 'mixed'}]))
            self.assertTrue(p['datasets'][0]['rgba_generate_control'])
            self.assertFalse(p['train']['cache_text_embeddings'])
            self.assertFalse(p['train']['unload_text_encoder'])

    def test_rgba_dopsd_self_reference_keeps_caches_without_external_control(self):
        vae = self.rgba_dataset()
        p = self.process(self.payload('minimax_h3_ref2va', trainingPreset='transparent_lora',
            distillationMethod='dopsd', vaePath=str(vae),
            datasets=[{'name': 'alpha', 'rgbaControlMode': 'generation'}]))
        self.assertTrue(p['model']['model_kwargs']['dopsd'])
        self.assertTrue(p['train']['cache_text_embeddings'])
        self.assertFalse(p['datasets'][0]['rgba_generate_control'])
        self.assertNotIn('assistant_lora_path', p['model'])

    def test_h3_vae_preset_uses_project_model_folder_and_queue(self):
        self.rgba_dataset()
        source = self.root / 'models' / 'vae' / 'minimax_h3_video_vae_fp16.safetensors'
        source.write_bytes(b'source')
        payload = self.payload(trainingPreset='h3_rgba_vae', datasets=[{'name': 'alpha'}])
        p = self.process(payload)
        self.assertEqual(p['type'], 'h3_rgba_vae_trainer')
        self.assertEqual(p['source_vae']['name_or_path'], str(source))
        self.assertEqual(p['source_vae']['subfolder'], '')
        self.assertEqual(p['source_vae']['filename'], source.name)
        self.assertEqual(p['train']['scope'], 'full')
        job, _ = self.service.create_job(payload)
        self.service.queue_job(job['id'])
        self.assertTrue(self.service.list_job_samples(job['id'])['isVae'])
        with self.assertRaises(TrainerValidationError):
            self.process({**payload, 'vaeResolution': 100})

    def test_standard_h3_without_controls_can_be_queued(self):
        job, _ = self.service.create_job(self.payload())
        self.service.queue_job(job['id'])
        self.assertEqual(self.service.get_job(job['id'])['status'], 'queued')

    def test_local_adapter_v2_wins_and_explicit_path_survives(self):
        folder = self.root / 'models' / 'loras' / 'training_adapters'
        folder.mkdir(parents=True)
        adapter = folder / 'minimax_h3_training_adapter_v2.safetensors'
        adapter.write_bytes(b'local')
        self.assertEqual(self.process(self.payload())['model']['assistant_lora_path'], str(adapter))
        self.assertEqual(self.process(self.payload(assistantLoraPath='org/repo/custom.safetensors'))['model']['assistant_lora_path'], 'org/repo/custom.safetensors')

    def test_rgba_video_audio_sidecar_is_served_archived_and_deleted(self):
        job, _ = self.service.create_job(self.payload())
        folder = self.service.output_dir / job['name'] / 'samples'
        folder.mkdir(parents=True)
        image = folder / 'h3_test_000000100_0.png'
        Image.new('RGBA', (8, 8)).save(image)
        audio = image.with_suffix('.wav')
        audio.write_bytes(b'RIFF audio')
        self.assertEqual(self.service.list_job_samples(job['id'])['samples'][0]['audioFile'], audio.name)
        self.assertEqual(self.service.resolve_job_sample(job['id'], audio.name), audio)
        archive, _ = self.service.build_samples_archive(job['id'])
        with zipfile.ZipFile(archive) as handle:
            self.assertIn('samples/' + audio.name, handle.namelist())
        self.service.delete_job_sample(job['id'], image.name)
        self.assertFalse(image.exists())
        self.assertFalse(audio.exists())


if __name__ == '__main__':
    unittest.main()
