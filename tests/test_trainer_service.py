import io
import json
import tempfile
import unittest
from pathlib import Path

from flask import Flask
from PIL import Image
from werkzeug.datastructures import FileStorage

from trainer_service import QTYPE_OPTIONS, TrainerService, TrainerValidationError, create_trainer_blueprint


QWEN_2511_ARA = 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors'


class TrainerServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.tempdir.name)
        self.datasets_root = self.project_root / 'Datasets'
        self.datasets_root.mkdir()
        self.service = TrainerService(self.project_root, lambda: self.datasets_root)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_dataset(self, name='demo', control_count=2):
        root = self.datasets_root / name
        (root / 'img').mkdir(parents=True)
        for index in range(1, 4):
            (root / f'Control{index}').mkdir()
        for stem in ('one', 'two'):
            Image.new('RGB', (16, 12), (20, 40, 60)).save(root / 'img' / f'{stem}.png')
            (root / 'img' / f'{stem}.txt').write_text(f'caption {stem}', encoding='utf-8')
            for index in range(1, control_count + 1):
                Image.new('RGB', (12, 16), (index * 20, 30, 40)).save(
                    root / f'Control{index}' / f'{stem}.jpg'
                )
        return root

    def make_rgba_dataset(self, name='rgba', control_count=0):
        root = self.datasets_root / name
        (root / 'img').mkdir(parents=True)
        for index in range(1, 4):
            (root / f'Control{index}').mkdir()
        for stem in ('one', 'two', 'three'):
            Image.new('RGBA', (16, 12), (20, 40, 60, 128)).save(root / 'img' / f'{stem}.png')
            (root / 'img' / f'{stem}.txt').write_text(f'caption {stem}', encoding='utf-8')
            for index in range(1, control_count + 1):
                Image.new('RGB', (16, 12), (30, 40, 50)).save(
                    root / f'Control{index}' / f'{stem}.png'
                )
        return root

    def make_background_dataset(self, name='backgrounds'):
        root = self.datasets_root / name
        (root / 'img').mkdir(parents=True)
        for index, color in enumerate(((30, 60, 90), (180, 140, 100)), start=1):
            Image.new('RGB', (24, 18), color).save(root / 'img' / f'background_{index}.jpg')
        return root

    def default_payload(self, datasets=None):
        return {
            'name': 'edit_lora_v1',
            'gpuIds': '0',
            'model': 'qwen_image_edit_2511',
            'qtype': QWEN_2511_ARA,
            'rank': 64,
            'alpha': 32,
            'steps': 1200,
            'datasets': datasets or [{
                'name': 'demo',
                'resolutions': [512, 768, 1024],
                'repeats': 2,
                'weight': 1,
                'batchSize': 1,
                'captionDropout': 0.05,
            }],
        }

    def test_internal_dataset_maps_to_target_and_multiple_controls(self):
        root = self.make_dataset()

        _name, _gpu_ids, config, inspections = self.service.build_job_config(self.default_payload())

        process = config['config']['process'][0]
        dataset = process['datasets'][0]
        self.assertEqual(dataset['folder_path'], str((root / 'img').resolve()))
        self.assertEqual(dataset['control_path'], [str((root / 'Control1').resolve()), str((root / 'Control2').resolve())])
        self.assertEqual(dataset['num_repeats'], 2)
        self.assertEqual(process['model']['arch'], 'qwen_image_edit_plus')
        self.assertEqual(
            process['model']['qtype'],
            QWEN_2511_ARA
        )
        self.assertTrue(inspections[0]['valid'])

    def test_multiple_managed_datasets_become_multiple_toolkit_datasets(self):
        self.make_dataset('demo')
        self.make_dataset('second', control_count=3)
        payload = self.default_payload([
            {'name': 'demo', 'resolutions': [512], 'repeats': 1},
            {'name': 'second', 'resolutions': [768, 1024], 'repeats': 3},
        ])
        payload['model'] = 'flux2_klein_9b'
        payload['qtype'] = 'qfloat8'

        _name, _gpu_ids, config, inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        self.assertEqual(process['model']['arch'], 'flux2_klein_9b')
        self.assertEqual(len(process['datasets']), 2)
        self.assertEqual(len(process['datasets'][1]['control_path']), 3)
        self.assertEqual([item['name'] for item in inspections], ['demo', 'second'])

    def test_validation_sample_uses_real_control_images(self):
        self.make_dataset('demo')
        payload = self.default_payload()
        payload['disableSampling'] = False
        payload['samples'] = [{'dataset': 'demo', 'image': 'one.png', 'prompt': ''}]

        _name, _gpu_ids, config, _inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        self.assertFalse(process['train']['disable_sampling'])
        self.assertEqual(process['sample']['samples'][0]['prompt'], 'caption one')
        self.assertTrue(process['sample']['samples'][0]['ctrl_img_1'].endswith('Control1/one.jpg'))
        self.assertTrue(process['sample']['samples'][0]['ctrl_img_2'].endswith('Control2/one.jpg'))

    def test_sampling_uses_independent_uploaded_control_images(self):
        self.make_dataset('demo')
        uploaded_paths = []
        for index in range(2):
            stream = io.BytesIO()
            Image.new('RGB', (20, 16), (80 + index, 40, 20)).save(stream, format='PNG')
            stream.seek(0)
            uploaded_paths.append(self.service.save_sample_image(FileStorage(
                stream=stream,
                filename=f'independent-{index}.png',
                content_type='image/png',
            )))
        payload = self.default_payload()
        payload['disableSampling'] = False
        payload['samples'] = [{
            'prompt': 'Remove the background',
            'ctrlImg1': str(uploaded_paths[0]),
            'ctrlImg2': str(uploaded_paths[1]),
            'width': 768,
            'networkMultiplier': '0.75',
        }]

        _name, _gpu_ids, config, _inspections = self.service.build_job_config(payload)

        sample = config['config']['process'][0]['sample']['samples'][0]
        self.assertEqual(sample['prompt'], 'Remove the background')
        self.assertEqual(sample['ctrl_img_1'], str(uploaded_paths[0]))
        self.assertEqual(sample['ctrl_img_2'], str(uploaded_paths[1]))
        self.assertNotIn('ctrl_img_3', sample)
        self.assertEqual(sample['width'], 768)
        self.assertEqual(sample['network_multiplier'], 0.75)

    def test_sample_image_upload_and_preview_routes(self):
        stream = io.BytesIO()
        Image.new('RGBA', (18, 14), (120, 80, 40, 180)).save(stream, format='PNG')
        stream.seek(0)
        app = Flask(__name__)
        app.register_blueprint(create_trainer_blueprint(self.service))
        client = app.test_client()

        response = client.post(
            '/api/trainer/sample-images',
            data={'files': (stream, 'separate-reference.png')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 201)
        uploaded = response.get_json()
        self.assertTrue(Path(uploaded['path']).is_file())
        preview = client.get(uploaded['previewUrl'])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.mimetype, 'image/png')
        preview.close()

    def test_generated_sample_gallery_routes_preserve_rgba_and_metadata(self):
        self.make_dataset('demo')
        payload = self.default_payload()
        payload['disableSampling'] = False
        payload['samples'] = [{'dataset': 'demo', 'image': 'one.png', 'prompt': 'Keep the transparent object'}]
        job, _inspections = self.service.create_job(payload)
        samples_dir = self.service.output_dir / job['name'] / 'samples'
        thumbs_dir = samples_dir / '.thumbs'
        thumbs_dir.mkdir(parents=True)
        sample_name = '1234567890__000000250_0.png'
        Image.new('RGBA', (18, 14), (120, 80, 40, 90)).save(samples_dir / sample_name)
        Image.new('RGBA', (8, 8), (120, 80, 40, 90)).save(thumbs_dir / f'{sample_name}.png')
        Image.new('RGB', (6, 6), (20, 20, 20)).save(thumbs_dir / f'{sample_name}.jpg')

        app = Flask(__name__)
        app.register_blueprint(create_trainer_blueprint(self.service))
        client = app.test_client()

        listed = client.get(f'/api/trainer/jobs/{job["id"]}/samples')
        self.assertEqual(listed.status_code, 200)
        body = listed.get_json()
        self.assertEqual(body['sampleCount'], 1)
        self.assertEqual(body['samples'][0]['name'], sample_name)
        self.assertEqual(body['samples'][0]['step'], 250)
        self.assertEqual(body['samples'][0]['sampleIndex'], 0)
        self.assertEqual(body['samples'][0]['prompt'], 'Keep the transparent object')
        self.assertEqual(body['samples'][0]['controlCount'], 2)

        original = client.get(f'/api/trainer/jobs/{job["id"]}/samples/{sample_name}')
        self.assertEqual(original.status_code, 200)
        with Image.open(io.BytesIO(original.data)) as image:
            self.assertEqual(image.mode, 'RGBA')
            self.assertEqual(image.getpixel((0, 0))[3], 90)
        original.close()

        thumbnail = client.get(f'/api/trainer/jobs/{job["id"]}/samples/{sample_name}?thumb=1')
        self.assertEqual(thumbnail.status_code, 200)
        with Image.open(io.BytesIO(thumbnail.data)) as image:
            self.assertEqual(image.mode, 'RGBA')
            self.assertEqual(image.size, (8, 8))
        thumbnail.close()

        control = client.get(f'/api/trainer/jobs/{job["id"]}/sample-controls/0/0')
        self.assertEqual(control.status_code, 200)
        self.assertEqual(control.mimetype, 'image/jpeg')
        control.close()

        archive = client.get(f'/api/trainer/jobs/{job["id"]}/samples/archive')
        self.assertEqual(archive.status_code, 200)
        self.assertIn(archive.mimetype, {'application/zip', 'application/x-zip-compressed'})
        archive.close()

        deleted = client.delete(f'/api/trainer/jobs/{job["id"]}/samples/{sample_name}')
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse((samples_dir / sample_name).exists())
        self.assertFalse((thumbs_dir / f'{sample_name}.png').exists())
        self.assertFalse((thumbs_dir / f'{sample_name}.jpg').exists())

    def test_full_upstream_simple_settings_are_forwarded(self):
        self.make_dataset('demo', control_count=3)
        payload = self.default_payload()
        payload.update({
            'model': 'flux2_klein_9b',
            'modelPath': 'black-forest-labs/FLUX.2-klein-base-9B',
            'qtype': 'convrotbitnet',
            'qtypeTextEncoder': 'convrotint3',
            'compileModel': True,
            'networkType': 'lokr',
            'lokrFactor': 16,
            'saveDtype': 'fp16',
            'saveEvery': 77,
            'maxSaves': 9,
            'optimizer': 'automagic3',
            'timestepType': 'shift',
            'timestepBias': 'style',
            'lossType': 'stepped',
            'unloadTextEncoder': True,
            'cacheTextEmbeddings': False,
            'useEma': True,
            'emaDecay': 0.975,
            'diffOutputPreservation': True,
            'dopMultiplier': 1.5,
            'dopClass': 'person',
            'blankPromptPreservation': True,
            'bppMultiplier': 2,
            'guidanceLoss': True,
            'guidanceLossTarget': 4.5,
            'differentialGuidance': True,
            'differentialGuidanceScale': 3.5,
            'validationEnabled': True,
            'validateEvery': 20,
            'validationResolution': 768,
            'validationSigmas': [1, 0.5],
            'validationItems': [{'dataset': 'demo', 'image': 'two.png', 'prompt': 'validation'}],
            'disableSampling': False,
            'sampleEvery': 90,
            'sampleStartStep': 10,
            'sampler': 'ddpm',
            'guidanceScale': 5,
            'sampleSteps': 35,
            'sampleWidth': 896,
            'sampleHeight': 1152,
            'sampleSeed': 123,
            'walkSeed': False,
            'skipFirstSample': False,
            'forceFirstSample': True,
            'samples': [{
                'dataset': 'demo', 'image': 'one.png', 'prompt': 'sample prompt',
                'width': 640, 'height': 832, 'seed': 456, 'networkMultiplier': 0.8,
            }],
            'layerOffloading': True,
            'transformerOffload': 0.75,
            'textEncoderOffload': 0.5,
            'datasets': [{
                'name': 'demo', 'resolutions': [512, 1328], 'repeats': 4, 'weight': 0.75,
                'batchSize': 2, 'defaultCaption': 'fallback', 'captionDropout': 0.1,
                'captionExtension': 'json', 'cacheLatents': True, 'isRegularization': True,
                'flipX': True, 'flipY': True,
            }],
        })

        _name, _gpu_ids, config, _inspections = self.service.build_job_config(payload)
        process = config['config']['process'][0]
        model = process['model']
        train = process['train']
        sample = process['sample']
        dataset = process['datasets'][0]

        self.assertEqual(model['name_or_path'], payload['modelPath'])
        self.assertEqual(model['qtype'], 'convrotbitnet')
        self.assertEqual(model['qtype_te'], 'convrotint3')
        self.assertTrue(model['compile'])
        self.assertTrue(model['block_compile'])
        self.assertEqual(model['layer_offloading_transformer_percent'], 0.75)
        self.assertEqual(model['layer_offloading_text_encoder_percent'], 0.5)
        self.assertEqual(process['network']['type'], 'lokr')
        self.assertEqual(process['network']['lokr_factor'], 16)
        self.assertEqual(process['save']['dtype'], 'fp16')
        self.assertEqual(process['save']['save_every'], 77)
        self.assertEqual(train['noise_scheduler'], 'flowmatch')
        self.assertEqual(train['optimizer'], 'automagic3')
        self.assertEqual(train['timestep_type'], 'shift')
        self.assertEqual(train['content_or_style'], 'style')
        self.assertEqual(train['loss_type'], 'stepped')
        self.assertTrue(train['unload_text_encoder'])
        self.assertTrue(train['ema_config']['use_ema'])
        self.assertEqual(train['ema_config']['ema_decay'], 0.975)
        self.assertTrue(train['diff_output_preservation'])
        self.assertFalse(train['blank_prompt_preservation'])
        self.assertTrue(train['do_guidance_loss'])
        self.assertTrue(train['do_differential_guidance'])
        self.assertEqual(train['validation_config']['validation_sigmas'], [1.0, 0.5])
        self.assertEqual(sample['sampler'], 'ddpm')
        self.assertEqual(sample['sample_start_step'], 10)
        self.assertEqual(sample['samples'][0]['network_multiplier'], 0.8)
        self.assertTrue(sample['samples'][0]['ctrl_img_3'].endswith('Control3/one.jpg'))
        self.assertEqual(dataset['caption_ext'], 'json')
        self.assertEqual(dataset['resolution'], [512, 1328])
        self.assertTrue(dataset['is_reg'])

    def test_quantization_options_match_upstream_and_ara_is_separate(self):
        self.make_dataset()
        self.assertEqual(QTYPE_OPTIONS, (
            '', 'qfloat8', 'float8', 'convrot8', 'convrot4', 'nvfp4',
            'convrotint7', 'convrotint6', 'convrotint5', 'convrotint4',
            'convrotint3', 'convrotint2', 'convrotbitnet',
            'uint7', 'uint6', 'uint5', 'uint4', 'uint3', 'uint2',
        ))
        standard = self.default_payload()
        standard['qtype'] = 'uint3'
        ara = self.default_payload()
        ara['name'] = 'edit_lora_ara'
        _name, _gpu, standard_config, _inspection = self.service.build_job_config(standard)
        _name, _gpu, ara_config, _inspection = self.service.build_job_config(ara)
        self.assertEqual(standard_config['config']['process'][0]['model']['qtype'], 'uint3')
        self.assertEqual(ara_config['config']['process'][0]['model']['qtype'], QWEN_2511_ARA)

    def test_model_specific_constraints_match_upstream(self):
        self.make_dataset()
        qwen_payload = self.default_payload()
        qwen_payload['unloadTextEncoder'] = True
        _name, _gpu, qwen_config, _inspection = self.service.build_job_config(qwen_payload)
        qwen_process = qwen_config['config']['process'][0]
        self.assertFalse(qwen_process['train']['unload_text_encoder'])
        self.assertEqual(qwen_process['train']['noise_scheduler'], 'flowmatch')
        self.assertNotIn('conv', qwen_process['network'])

        klein_payload = self.default_payload()
        klein_payload.update({
            'name': 'klein_lora',
            'model': 'flux2_klein_4b',
            'qtype': 'uint3',
            'unloadTextEncoder': True,
        })
        _name, _gpu, klein_config, _inspection = self.service.build_job_config(klein_payload)
        klein_process = klein_config['config']['process'][0]
        self.assertTrue(klein_process['train']['unload_text_encoder'])
        self.assertEqual(klein_process['train']['noise_scheduler'], 'flowmatch')

    def test_gui_exposes_every_scoped_upstream_training_group(self):
        static_root = Path(__file__).resolve().parents[1] / 'static'
        html = (static_root / 'trainer.html').read_text(encoding='utf-8')
        javascript = (static_root / 'trainer.js').read_text(encoding='utf-8')

        required_control_ids = {
            # Model, Hugging Face resolution, quantization and compilation.
            'trainer-model', 'trainer-model-path', 'trainer-model-gate',
            'trainer-qtype', 'trainer-qtype-te', 'trainer-compile',
            # Target and persistence.
            'trainer-network-type', 'trainer-rank', 'trainer-lokr-factor',
            'trainer-save-dtype', 'trainer-save-every', 'trainer-max-saves',
            # Complete SimpleJob training surface for the scoped architectures.
            'trainer-optimizer', 'trainer-timestep', 'trainer-timestep-bias',
            'trainer-loss', 'trainer-noise-scheduler', 'trainer-use-ema',
            'trainer-ema-decay', 'trainer-unload-text', 'trainer-cache-text',
            # Preservation and guidance settings.
            'trainer-dop', 'trainer-dop-multiplier', 'trainer-dop-class',
            'trainer-bpp', 'trainer-bpp-multiplier', 'trainer-guidance-loss',
            'trainer-guidance-loss-target', 'trainer-differential-guidance',
            'trainer-differential-guidance-scale',
            # Validation, sampling and offloading.
            'trainer-validation-enabled', 'trainer-validation-every',
            'trainer-validation-resolution', 'trainer-validation-sigmas',
            'trainer-validation-items', 'trainer-sampler', 'trainer-sample-every',
            'trainer-sample-start', 'trainer-sample-steps', 'trainer-sample-width',
            'trainer-sample-height', 'trainer-sample-seed', 'trainer-walk-seed',
            'trainer-skip-first-sample', 'trainer-force-first-sample',
            'trainer-disable-sampling', 'trainer-sample-items',
            'trainer-layer-offloading', 'trainer-transformer-offload',
            'trainer-text-offload',
            # Full-process escape hatch and runtime actions from AI Toolkit.
            'trainer-advanced-config-json', 'trainer-use-advanced-config',
            'trainer-save-now-btn', 'trainer-sample-now-btn',
            # Modified upstream validation/sample gallery with RGBA viewing.
            'trainer-samples-tab', 'trainer-detail-samples-panel',
            'trainer-samples-gallery', 'trainer-download-samples',
            'trainer-sample-viewer', 'trainer-sample-viewer-image',
            'trainer-sample-controls', 'trainer-sample-delete',
        }
        for control_id in required_control_ids:
            self.assertIn(f'id="{control_id}"', html, control_id)

        for qtype in QTYPE_OPTIONS[1:]:
            self.assertIn(qtype, javascript, qtype)
        self.assertIn('model.accuracyRecoveryAdapters', javascript)
        for optimizer in ('adafactor', 'adam', 'adamw', 'adamw8bit', 'automagic',
                          'automagic2', 'automagic3', 'prodigyopt', 'prodigy8bit'):
            self.assertIn(f'value="{optimizer}"', html, optimizer)
        for sampler in ('flowmatch', 'ddpm'):
            self.assertIn(f'value="{sampler}"', html, sampler)
        self.assertIn('id="trainer-transformer-offload" type="range"', html)
        self.assertIn('id="trainer-text-offload" type="range"', html)
        self.assertIn('id="trainer-transformer-offload-value"', html)
        self.assertIn('id="trainer-text-offload-value"', html)
        self.assertIn("syncSlider('trainer-transformer-offload')", javascript)
        self.assertIn("syncSlider('trainer-text-offload')", javascript)
        self.assertNotIn('trainer-booting', html)
        self.assertNotIn('TRAINER_CLIENT', html)
        self.assertIn('<script src="/static/trainer.js?', html)
        self.assertIn('data-dataset-field="rgbaControlMode"', javascript)
        self.assertIn('data-dataset-field="rgbaBackgroundDataset"', javascript)

        self.assertIn('function backgroundDatasetOptions', javascript)
        self.assertIn('Select a background dataset for', javascript)

        render_samples = javascript[
            javascript.index('function sampleControlPath'):javascript.index('function renderValidationItems()')
        ]
        self.assertIn('Edit instruction', render_samples)
        self.assertIn('Images to edit', render_samples)
        self.assertIn('Add Additional Image', render_samples)
        self.assertIn('ctrlImg', render_samples)
        self.assertNotIn('<span>Dataset</span>', render_samples)
        self.assertNotIn('Target image', render_samples)
        self.assertIn("body.append('files', file)", javascript)

        css = (static_root / 'trainer.css').read_text(encoding='utf-8')
        self.assertIn('color-scheme: dark', css)
        self.assertIn('.trainer-page select option', css)
        self.assertIn('::-webkit-slider-runnable-track', css)
        self.assertIn('.trainer-slider-main input[type="range"]', css)
        self.assertIn('.trainer-mode-toggle', css)
        self.assertIn('.trainer-rgba-settings', css)

        collect_form = javascript[
            javascript.index('function collectForm'):javascript.index('function validateForm')
        ]
        forwarded_payload_keys = {
            'modelPath', 'qtype', 'qtypeTextEncoder', 'lowVram',
            'matchTargetResolution', 'compileModel', 'networkType', 'rank',
            'lokrFactor', 'saveDtype', 'saveEvery', 'maxSaves', 'batchSize',
            'gradientAccumulation', 'steps', 'learningRate', 'optimizer',
            'weightDecay', 'timestepType', 'timestepBias', 'lossType',
            'cacheTextEmbeddings', 'unloadTextEncoder', 'useEma', 'emaDecay',
            'diffOutputPreservation', 'dopMultiplier', 'dopClass',
            'blankPromptPreservation', 'bppMultiplier', 'guidanceLoss',
            'guidanceLossTarget', 'differentialGuidance',
            'differentialGuidanceScale', 'validationEnabled', 'validateEvery',
            'validationResolution', 'validationSigmas', 'validationItems',
            'sampleEvery', 'sampleStartStep', 'sampler', 'guidanceScale',
            'sampleSteps', 'sampleWidth', 'sampleHeight', 'sampleSeed',
            'walkSeed', 'skipFirstSample', 'forceFirstSample',
            'disableSampling', 'samples', 'layerOffloading',
            'transformerOffload', 'textEncoderOffload', 'datasets',
        }
        for payload_key in forwarded_payload_keys:
            self.assertIn(f'{payload_key}:', collect_form, payload_key)
        self.assertIn('payload.advancedProcess =', collect_form)

    def test_trainer_jobs_are_bootstrapped_before_the_full_state_request(self):
        static_root = Path(__file__).resolve().parents[1] / 'static'
        html = (static_root / 'trainer.html').read_text(encoding='utf-8')
        javascript = (static_root / 'trainer.js').read_text(encoding='utf-8')

        self.assertIn('id="trainer-bootstrap" type="application/json"', html)
        self.assertIn('<!--TRAINER_JOB_COUNT-->', html)
        self.assertIn('<!--TRAINER_JOBS-->', html)
        self.assertIn('<!--TRAINER_BOOTSTRAP-->', html)
        self.assertNotIn('<strong>No jobs yet</strong>', html)
        self.assertIn("const bootstrapState = (() =>", javascript)
        self.assertIn('jobs: bootstrapState?.jobs || []', javascript)
        self.assertIn('qtypes: bootstrapState?.qtypes || []', javascript)
        self.assertIn('const bootstrapCanRestoreEditor = Boolean(', javascript)
        self.assertIn('bootstrapState.qtypes.length', javascript)
        self.assertIn('if (!bootstrapCanRestoreEditor)', javascript)
        self.assertIn('if (bootstrapCanRestoreEditor) restoreInitialView();', javascript)
        self.assertIn('if (bootstrapState) {', javascript)
        self.assertIn('renderModelOptions();', javascript)
        self.assertIn('initialViewChosen: false', javascript)
        self.assertIn('function restoreInitialView()', javascript)
        self.assertIn("state.jobs.find(job => job.id === params.get('job'))", javascript)
        self.assertIn("params.get('edit') === '1'", javascript)
        self.assertIn("params.get('new') === '1'", javascript)
        self.assertIn('if (!initial || !bootstrapCanRestoreEditor)', javascript)
        self.assertIn('initial && !state.initialViewChosen', javascript)
        self.assertNotIn('initial && !state.selectedJobId && !state.editingJobId', javascript)

    def test_updating_job_persists_quantization_in_form_and_process(self):
        self.make_dataset()
        payload = self.default_payload()
        payload['qtype'] = 'qfloat8'
        created, _inspections = self.service.create_job(payload)

        payload['qtype'] = 'uint4'
        updated, _inspections = self.service.update_job(created['id'], payload)

        self.assertEqual(updated['form']['qtype'], 'uint4')
        self.assertEqual(
            updated['config']['config']['process'][0]['model']['qtype'],
            'uint4',
        )
        reloaded = self.service.get_job(created['id'])
        self.assertEqual(reloaded['form']['qtype'], 'uint4')
        self.assertEqual(
            reloaded['config']['config']['process'][0]['model']['qtype'],
            'uint4',
        )

    def test_advanced_process_editor_preserves_scope_and_forwards_upstream_fields(self):
        root = self.make_dataset()
        payload = self.default_payload()
        _name, _gpu, generated, _inspection = self.service.build_job_config(payload)
        process_override = generated['config']['process'][0]
        process_override['train']['lr'] = 0.0000123
        process_override['train']['custom_upstream_flag'] = 'kept'
        process_override['datasets'][0]['folder_path'] = '/tmp/not-allowed'
        process_override['device'] = 'mps'
        payload['advancedProcess'] = process_override

        _name, _gpu, config, _inspection = self.service.build_job_config(payload)
        process = config['config']['process'][0]
        self.assertEqual(process['train']['lr'], 0.0000123)
        self.assertEqual(process['train']['custom_upstream_flag'], 'kept')
        self.assertEqual(process['train']['noise_scheduler'], 'flowmatch')
        self.assertEqual(process['device'], 'cuda')
        self.assertEqual(process['datasets'][0]['folder_path'], str((root / 'img').resolve()))

        process_override['model']['arch'] = 'unsupported_model'
        payload['advancedProcess'] = process_override
        with self.assertRaises(TrainerValidationError):
            self.service.build_job_config(payload)

    def test_dataset_without_controls_is_rejected(self):
        self.make_dataset('demo', control_count=0)

        with self.assertRaises(TrainerValidationError):
            self.service.build_job_config(self.default_payload())

    def test_job_lifecycle_persists_in_sqlite(self):
        self.make_dataset()
        job, _inspections = self.service.create_job(self.default_payload())
        self.assertEqual(job['status'], 'stopped')
        queued = self.service.queue_job(job['id'])
        self.assertEqual(queued['status'], 'queued')
        stopped = self.service.stop_job(job['id'])
        self.assertEqual(stopped['status'], 'stopped')
        self.assertEqual(len(self.service.list_jobs()), 1)

    def test_running_job_accepts_save_and_sample_requests(self):
        self.make_dataset()
        job, _inspections = self.service.create_job(self.default_payload())
        with self.service.connect() as connection:
            connection.execute("UPDATE Job SET status = 'running', pid = 999999 WHERE id = ?", (job['id'],))
        self.service.request_runtime_action(job['id'], 'save')
        self.service.request_runtime_action(job['id'], 'sample')
        with self.service.connect() as connection:
            row = connection.execute('SELECT save_now, sample_now FROM Job WHERE id = ?', (job['id'],)).fetchone()
        self.assertEqual((row['save_now'], row['sample_now']), (1, 1))

    def test_transparent_qwen_generation_dataset_uses_rgba_and_black_control_mode(self):
        root = self.make_rgba_dataset()
        vae = self.project_root / 'qwen-rgba-vae'
        vae.mkdir()
        payload = self.default_payload([{
            'name': 'rgba',
            'resolutions': [512],
            'rgbaControlMode': 'generation',
        }])
        payload.update({
            'trainingPreset': 'transparent_lora',
            'vaePath': str(vae),
        })

        _name, _gpu, config, inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        dataset = process['datasets'][0]
        self.assertEqual(process['model']['arch'], 'qwen_image_edit_plus_rgba')
        self.assertEqual(process['model']['vae_path'], str(vae.resolve()))
        self.assertEqual(process['sample']['format'], 'png')
        self.assertEqual(dataset['folder_path'], str((root / 'img').resolve()))
        self.assertEqual(dataset['pixel_channels'], 'rgba')
        self.assertTrue(dataset['rgba_generate_control'])
        self.assertEqual(dataset['rgba_control_mode'], 'generation')
        self.assertNotIn('control_path', dataset)
        self.assertTrue(inspections[0]['transparentValid'])

    def test_transparent_edit_mode_uses_selected_background_dataset(self):
        root = self.make_rgba_dataset(control_count=2)
        backgrounds = self.make_background_dataset()
        vae = self.project_root / 'qwen-rgba-vae'
        vae.mkdir()
        payload = self.default_payload([{
            'name': 'rgba',
            'resolutions': [512],
            'rgbaControlMode': 'edit',
            'rgbaBackgroundDataset': 'backgrounds',
        }])
        payload.update({
            'trainingPreset': 'transparent_lora',
            'vaePath': str(vae),
            'cacheTextEmbeddings': True,
        })

        _name, _gpu, config, _inspections = self.service.build_job_config(payload)

        dataset = config['config']['process'][0]['datasets'][0]
        self.assertEqual(dataset['folder_path'], str((root / 'img').resolve()))
        self.assertEqual(
            dataset['rgba_control_background_path'],
            str((backgrounds / 'img').resolve()),
        )
        self.assertTrue(dataset['rgba_generate_control'])
        self.assertEqual(dataset['rgba_control_mode'], 'edit')
        self.assertNotIn('control_path', dataset)
        self.assertNotIn('rgba_control_backgrounds', dataset)
        self.assertFalse(config['config']['process'][0]['train']['cache_text_embeddings'])
        self.assertFalse(config['meta']['qdm']['form']['cacheTextEmbeddings'])
        self.assertEqual(
            config['meta']['qdm']['form']['datasets'][0]['rgbaBackgroundDataset'],
            'backgrounds',
        )

    def test_transparent_edit_mode_requires_an_opaque_background_dataset(self):
        self.make_rgba_dataset()
        vae = self.project_root / 'qwen-rgba-vae'
        vae.mkdir()
        payload = self.default_payload([{
            'name': 'rgba',
            'resolutions': [512],
            'rgbaControlMode': 'edit',
        }])
        payload.update({'trainingPreset': 'transparent_lora', 'vaePath': str(vae)})

        with self.assertRaisesRegex(TrainerValidationError, 'background dataset'):
            self.service.build_job_config(payload)

    def test_turbo_sampling_lora_is_forwarded_and_forces_qwen_four_step_preview(self):
        self.make_rgba_dataset(control_count=1)
        vae = self.project_root / 'qwen-rgba-vae'
        vae.mkdir()
        turbo = self.project_root / 'qwen-lightning.safetensors'
        turbo.write_bytes(b'test')
        payload = self.default_payload([{'name': 'rgba', 'resolutions': [512], 'rgbaControlMode': 'generation'}])
        payload.update({
            'trainingPreset': 'transparent_lora',
            'vaePath': str(vae),
            'sampleLoraPath': str(turbo),
            'guidanceScale': 7,
            'sampleSteps': 40,
        })

        _name, _gpu, config, _inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        self.assertEqual(process['model']['sample_lora_path'], str(turbo.resolve()))
        self.assertEqual(process['sample']['guidance_scale'], 1.0)
        self.assertEqual(process['sample']['sample_steps'], 4)

    def test_default_turbo_lora_is_forwarded_for_every_model(self):
        self.make_dataset(control_count=1)
        turbo_root = self.project_root / 'models' / 'turbo_lora'
        turbo_root.mkdir(parents=True)
        expected_files = {
            'qwen_image_edit_2511': 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
            'flux2_klein_4b': 'klein4b_turbo_r128.safetensors',
            'flux2_klein_9b': 'Flux_Klein_9b_Turbo_lora_rank_256_fp8_standard.safetensors',
        }
        for filename in expected_files.values():
            (turbo_root / filename).write_bytes(b'test')

        for model_key, filename in expected_files.items():
            with self.subTest(model=model_key):
                payload = self.default_payload()
                payload.update({'model': model_key, 'qtype': 'qfloat8'})

                _name, _gpu, config, _inspections = self.service.build_job_config(payload)
                process = config['config']['process'][0]

                self.assertEqual(
                    process['model']['sample_lora_path'],
                    str((turbo_root / filename).resolve()),
                )

    def test_legacy_job_without_turbo_lora_is_hydrated_and_persisted_on_queue(self):
        self.make_dataset(control_count=1)
        turbo = self.project_root / 'models' / 'turbo_lora' / 'klein4b_turbo_r128.safetensors'
        turbo.parent.mkdir(parents=True)
        turbo.write_bytes(b'test')
        expected = str(turbo.resolve())

        payload = self.default_payload()
        payload.update({'model': 'flux2_klein_4b', 'qtype': 'qfloat8'})
        job, _inspections = self.service.create_job(payload)

        with self.service.connect() as connection:
            row = connection.execute(
                'SELECT job_config FROM "Job" WHERE id = ?', (job['id'],)
            ).fetchone()
            legacy_config = json.loads(row['job_config'])
            legacy_config['config']['process'][0]['model']['sample_lora_path'] = None
            legacy_config['meta']['qdm']['form']['sampleLoraPath'] = ''
            connection.execute(
                'UPDATE "Job" SET job_config = ? WHERE id = ?',
                (json.dumps(legacy_config), job['id']),
            )

        hydrated = self.service.get_job(job['id'])
        self.assertEqual(
            hydrated['config']['config']['process'][0]['model']['sample_lora_path'],
            expected,
        )
        self.assertEqual(hydrated['form']['sampleLoraPath'], expected)

        self.service.queue_job(job['id'])
        with self.service.connect() as connection:
            row = connection.execute(
                'SELECT job_config FROM "Job" WHERE id = ?', (job['id'],)
            ).fetchone()
        persisted = json.loads(row['job_config'])
        self.assertEqual(
            persisted['config']['process'][0]['model']['sample_lora_path'],
            expected,
        )
        self.assertEqual(persisted['meta']['qdm']['form']['sampleLoraPath'], expected)

    def test_klein_transparent_preset_requires_its_own_rgba_vae_and_keeps_turbo_fields(self):
        self.make_rgba_dataset(control_count=1)
        vae = self.project_root / 'flux2-rgba-vae.safetensors'
        vae.write_bytes(b'test')
        turbo = self.project_root / 'klein-turbo.safetensors'
        turbo.write_bytes(b'test')
        payload = self.default_payload([{'name': 'rgba', 'resolutions': [512], 'rgbaControlMode': 'generation'}])
        payload.update({
            'trainingPreset': 'transparent_lora',
            'model': 'flux2_klein_9b',
            'qtype': 'qfloat8',
            'vaePath': str(vae),
            'sampleLoraPath': str(turbo),
            'sampleSteps': 8,
            'guidanceScale': 2,
        })

        _name, _gpu, config, _inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        self.assertEqual(process['model']['arch'], 'flux2_klein_9b_rgba')
        self.assertEqual(process['model']['vae_path'], str(vae.resolve()))
        self.assertEqual(process['model']['sample_lora_path'], str(turbo.resolve()))
        self.assertEqual(process['sample']['sample_steps'], 8)
        self.assertEqual(process['sample']['guidance_scale'], 2)

    def test_default_rgba_vae_is_forwarded_for_both_klein_models(self):
        self.make_rgba_dataset(control_count=1)
        vae = self.project_root / 'models' / 'vae' / 'flux2-klein-rgba.safetensors'
        vae.parent.mkdir(parents=True)
        vae.write_bytes(b'test')

        for model_key, expected_arch in (
            ('flux2_klein_4b', 'flux2_klein_4b_rgba'),
            ('flux2_klein_9b', 'flux2_klein_9b_rgba'),
        ):
            with self.subTest(model=model_key):
                payload = self.default_payload([{
                    'name': 'rgba',
                    'resolutions': [512],
                    'rgbaControlMode': 'generation',
                }])
                payload.update({
                    'trainingPreset': 'transparent_lora',
                    'model': model_key,
                    'qtype': 'qfloat8',
                })

                _name, _gpu, config, _inspections = self.service.build_job_config(payload)
                process = config['config']['process'][0]

                self.assertEqual(process['model']['arch'], expected_arch)
                self.assertEqual(process['model']['vae_path'], str(vae.resolve()))

    def test_qwen_rgba_vae_preset_builds_extension_process_and_readiness_validation(self):
        root = self.make_rgba_dataset()
        payload = self.default_payload([{'name': 'rgba', 'flipX': True}])
        payload.update({
            'trainingPreset': 'qwen_rgba_vae',
            'steps': 700,
            'learningRate': 0.00001,
            'vaeResolution': 512,
            'vaeStopWhenReady': True,
        })

        _name, _gpu, config, inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        self.assertEqual(process['type'], 'qwen_rgba_vae_trainer')
        self.assertEqual(process['datasets'][0]['folder_path'], str((root / 'img').resolve()))
        self.assertEqual(process['train']['steps'], 700)
        self.assertEqual(process['train']['lr'], 0.00001)
        self.assertTrue(process['validation']['stop_when_ready'])
        self.assertEqual(process['sample']['sampler'], 'vae_roundtrip')
        self.assertEqual(process['sample']['format'], 'png')
        self.assertTrue(process['save']['comfy_export'])
        self.assertTrue(inspections[0]['vaeValid'])

    def test_flux2_rgba_vae_preset_builds_native_klein_process(self):
        root = self.make_rgba_dataset()
        payload = self.default_payload([{'name': 'rgba', 'flipX': True}])
        payload.update({
            'trainingPreset': 'flux2_rgba_vae',
            'model': 'flux2_klein_4b',
            'sourceVaePath': 'ai-toolkit/flux2_vae',
            'sourceVaeSubfolder': '',
            'steps': 600,
            'vaeResolution': 512,
        })

        _name, _gpu, config, inspections = self.service.build_job_config(payload)

        process = config['config']['process'][0]
        self.assertEqual(process['type'], 'flux2_rgba_vae_trainer')
        self.assertEqual(process['source_vae']['name_or_path'], 'ai-toolkit/flux2_vae')
        self.assertEqual(process['source_vae']['filename'], 'ae.safetensors')
        self.assertEqual(process['datasets'][0]['folder_path'], str((root / 'img').resolve()))
        self.assertEqual(process['train']['steps'], 600)
        self.assertEqual(config['meta']['qdm']['trainingPreset'], 'flux2_rgba_vae')
        self.assertTrue(inspections[0]['vaeValid'])


if __name__ == '__main__':
    unittest.main()
