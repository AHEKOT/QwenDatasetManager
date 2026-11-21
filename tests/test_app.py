import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

# Importing the Flask module must not start a second queue worker against the
# developer's real trainer database.
os.environ['QDM_DISABLE_TRAINER_WORKER'] = '1'
import app as manager


def image_bytes(image_format='PNG', color=(20, 40, 60), size=(8, 6)):
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, image_format)
    buffer.seek(0)
    return buffer


class DatasetManagerApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / 'Datasets'
        self.root.mkdir()
        manager.DATASETS_DIR = self.root
        manager.app.config.update(TESTING=True)
        manager.ACTIVE_DATASETS.clear()
        manager.IMPORT_JOBS.clear()
        manager.TOOL_JOBS.clear()
        self.client = manager.app.test_client()
        self.make_dataset('demo')

    def tearDown(self):
        manager.ACTIVE_DATASETS.clear()
        self.tempdir.cleanup()

    def make_dataset(self, name):
        dataset = self.root / name
        for folder in manager.DATASET_IMAGE_FOLDERS:
            (dataset / folder).mkdir(parents=True, exist_ok=True)
        return dataset

    def save_image(self, path, image_format='PNG', color=(20, 40, 60), size=(8, 6)):
        Image.new('RGB', size, color).save(path, image_format)

    def test_traversal_is_rejected_for_reads_and_writes(self):
        outside = self.root.parent / 'outside'
        (outside / 'img').mkdir(parents=True)
        self.save_image(outside / 'img' / 'secret.png')

        response = self.client.get('/api/images?folder=../outside')
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            '/api/caption/secret.png?folder=../outside',
            json={'caption': 'overwritten'}
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse((outside / 'img' / 'secret.txt').exists())

    def test_images_endpoint_lists_files_without_renaming_them(self):
        dataset = self.root / 'demo'
        self.save_image(dataset / 'img' / 'second.jpg', 'JPEG')
        self.save_image(dataset / 'img' / 'first.png')
        (dataset / 'img' / 'first.txt').write_text('caption', encoding='utf-8')

        response = self.client.get('/api/images?folder=demo')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'images': ['first.png', 'second.jpg']})
        self.assertTrue((dataset / 'img' / 'first.png').is_file())
        self.assertTrue((dataset / 'img' / 'second.jpg').is_file())
        self.assertTrue((dataset / 'img' / 'first.txt').is_file())

    def test_trainer_screen_and_state_use_managed_edit_datasets(self):
        dataset = self.root / 'demo'
        self.save_image(dataset / 'img' / 'target.png')
        self.save_image(dataset / 'Control1' / 'target.png')

        screen = self.client.get('/trainer')
        self.assertEqual(screen.status_code, 200)
        self.assertIn(b'CUDA Trainer', screen.data)
        self.assertIn(b'id="trainer-bootstrap" type="application/json">{"models":', screen.data)
        self.assertIn(b'id="trainer-job-count">', screen.data)
        self.assertNotIn(b'<!--TRAINER_JOB_COUNT-->', screen.data)
        self.assertNotIn(b'<!--TRAINER_JOBS-->', screen.data)
        self.assertNotIn(b'<!--TRAINER_BOOTSTRAP-->', screen.data)
        self.assertEqual(screen.headers['Cache-Control'], 'no-store')
        screen.close()

        state = self.client.get('/api/trainer/state')
        self.assertEqual(state.status_code, 200)
        body = state.get_json()
        self.assertEqual(
            {model['key'] for model in body['models']},
            {
                'qwen_image_edit_2511', 'flux2_klein_4b', 'flux2_klein_9b',
                'qdm_anime_keymatte_v3',
            },
        )
        self.assertEqual(body['datasets'][0]['name'], 'demo')
        self.assertEqual(body['datasets'][0]['controls'][0]['name'], 'Control1')
        self.assertTrue(body['datasets'][0]['valid'])

    def test_cross_origin_mutation_is_rejected_and_cors_is_absent(self):
        rejected = self.client.post(
            '/api/create-dataset',
            json={'name': 'evil'},
            headers={'Origin': 'https://attacker.example'}
        )
        self.assertEqual(rejected.status_code, 403)
        self.assertNotIn('Access-Control-Allow-Origin', rejected.headers)

        read = self.client.get('/api/folders', headers={'Origin': 'https://attacker.example'})
        self.assertNotIn('Access-Control-Allow-Origin', read.headers)
        self.assertIn("default-src 'self'", read.headers['Content-Security-Policy'])

        hostile_host = self.client.get('/api/folders', headers={'Host': 'attacker.example'})
        self.assertEqual(hostile_host.status_code, 400)

        fetch_metadata_rejected = self.client.post(
            '/api/create-dataset',
            json={'name': 'evil'},
            headers={'Sec-Fetch-Site': 'cross-site'}
        )
        self.assertEqual(fetch_metadata_rejected.status_code, 403)

    def test_upload_is_atomic_and_must_match_extension(self):
        destination = self.root / 'demo' / 'img' / 'sample.png'
        self.save_image(destination, color=(1, 2, 3))
        original = destination.read_bytes()

        bad = self.client.post(
            '/api/save/sample.png?folder=demo&subfolder=img',
            data={'file': (image_bytes('JPEG'), 'sample.png')},
            content_type='multipart/form-data'
        )
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(destination.read_bytes(), original)

        good = self.client.post(
            '/api/save/sample.png?folder=demo&subfolder=img',
            data={'file': (image_bytes('PNG', color=(9, 8, 7)), 'sample.png')},
            content_type='multipart/form-data'
        )
        self.assertEqual(good.status_code, 200)
        with Image.open(destination) as image:
            self.assertEqual(image.format, 'PNG')
            self.assertEqual(image.getpixel((0, 0)), (9, 8, 7))

    def test_delete_moves_complete_set_to_recoverable_trash(self):
        dataset = self.root / 'demo'
        self.save_image(dataset / 'img' / 'entry.png')
        self.save_image(dataset / 'Control1' / 'entry.jpg', 'JPEG')
        (dataset / 'img' / 'entry.txt').write_text('caption', encoding='utf-8')

        response = self.client.delete('/api/delete/entry.png?folder=demo')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body['recoverable'])
        self.assertFalse((dataset / 'img' / 'entry.png').exists())
        self.assertFalse((dataset / 'Control1' / 'entry.jpg').exists())
        trash = dataset / '.trash' / body['trashBatch']
        self.assertTrue((trash / 'img' / 'entry.png').is_file())
        self.assertTrue((trash / 'img' / 'entry.txt').is_file())
        self.assertTrue((trash / 'Control1' / 'entry.jpg').is_file())

    def test_transfer_moves_complete_image_set(self):
        source = self.root / 'demo'
        target = self.make_dataset('target')
        self.save_image(source / 'img' / 'entry.png')
        self.save_image(source / 'Control1' / 'entry.jpg', 'JPEG')
        self.save_image(source / 'Control2' / 'entry.webp', 'WEBP')
        (source / 'img' / 'entry.txt').write_text('caption', encoding='utf-8')

        response = self.client.post(
            '/api/transfer/entry.png?folder=demo',
            json={'targetFolder': 'target', 'operation': 'transfer'}
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['operation'], 'transfer')
        new_stem = Path(body['newFilename']).stem
        self.assertFalse((source / 'img' / 'entry.png').exists())
        self.assertFalse((source / 'img' / 'entry.txt').exists())
        self.assertFalse((source / 'Control1' / 'entry.jpg').exists())
        self.assertFalse((source / 'Control2' / 'entry.webp').exists())
        self.assertTrue((target / 'img' / f'{new_stem}.png').is_file())
        self.assertEqual((target / 'img' / f'{new_stem}.txt').read_text(encoding='utf-8'), 'caption')
        self.assertTrue((target / 'Control1' / f'{new_stem}.jpg').is_file())
        self.assertTrue((target / 'Control2' / f'{new_stem}.webp').is_file())

    def test_copy_keeps_original_complete_image_set(self):
        source = self.root / 'demo'
        target = self.make_dataset('target')
        self.save_image(source / 'img' / 'entry.png')
        self.save_image(source / 'Control1' / 'entry.jpg', 'JPEG')
        self.save_image(source / 'Control2' / 'entry.webp', 'WEBP')
        (source / 'img' / 'entry.txt').write_text('caption', encoding='utf-8')

        response = self.client.post(
            '/api/transfer/entry.png?folder=demo',
            json={'targetFolder': 'target', 'operation': 'copy'}
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['operation'], 'copy')
        self.assertIn('copied', body)
        new_stem = Path(body['newFilename']).stem
        self.assertTrue((source / 'img' / 'entry.png').is_file())
        self.assertEqual((source / 'img' / 'entry.txt').read_text(encoding='utf-8'), 'caption')
        self.assertTrue((source / 'Control1' / 'entry.jpg').is_file())
        self.assertTrue((source / 'Control2' / 'entry.webp').is_file())
        self.assertTrue((target / 'img' / f'{new_stem}.png').is_file())
        self.assertEqual((target / 'img' / f'{new_stem}.txt').read_text(encoding='utf-8'), 'caption')
        self.assertTrue((target / 'Control1' / f'{new_stem}.jpg').is_file())
        self.assertTrue((target / 'Control2' / f'{new_stem}.webp').is_file())

    def test_copy_from_training_dataset_is_allowed(self):
        source = self.root / 'demo'
        target = self.make_dataset('target')
        self.save_image(source / 'img' / 'entry.png')
        (source / 'img' / 'entry.txt').write_text('caption', encoding='utf-8')

        with patch.object(manager.TRAINER_SERVICE, 'active_dataset_names', return_value={'demo'}):
            response = self.client.post(
                '/api/transfer/entry.png?folder=demo',
                json={'targetFolder': 'target', 'operation': 'copy'}
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue((source / 'img' / 'entry.png').is_file())
        copied_stem = Path(response.get_json()['newFilename']).stem
        self.assertTrue((target / 'img' / f'{copied_stem}.png').is_file())

    def test_copy_into_training_dataset_is_blocked(self):
        source = self.root / 'demo'
        self.make_dataset('target')
        self.save_image(source / 'img' / 'entry.png')

        with patch.object(manager.TRAINER_SERVICE, 'active_dataset_names', return_value={'target'}):
            response = self.client.post(
                '/api/transfer/entry.png?folder=demo',
                json={'targetFolder': 'target', 'operation': 'copy'}
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()['datasets'], ['target'])
        self.assertTrue((source / 'img' / 'entry.png').is_file())

    def test_transfer_from_training_dataset_remains_blocked(self):
        source = self.root / 'demo'
        self.make_dataset('target')
        self.save_image(source / 'img' / 'entry.png')

        with patch.object(manager.TRAINER_SERVICE, 'active_dataset_names', return_value={'demo'}):
            response = self.client.post(
                '/api/transfer/entry.png?folder=demo',
                json={'targetFolder': 'target', 'operation': 'transfer'}
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()['datasets'], ['demo'])
        self.assertTrue((source / 'img' / 'entry.png').is_file())

    def test_transfer_rejects_unknown_operation(self):
        self.make_dataset('target')
        response = self.client.post(
            '/api/transfer/entry.png?folder=demo',
            json={'targetFolder': 'target', 'operation': 'clone'}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error'], 'Operation must be transfer or copy')

    def test_reshuffle_keeps_all_related_stems_synchronized(self):
        dataset = self.root / 'demo'
        for index in range(3):
            stem = f'old{index}'
            self.save_image(dataset / 'img' / f'{stem}.png')
            self.save_image(dataset / 'Control1' / f'{stem}.jpg', 'JPEG')
            (dataset / 'img' / f'{stem}.txt').write_text(stem, encoding='utf-8')

        result = manager.process_reshuffle_job(None, 'demo')
        self.assertEqual(result['count'], 3)
        img_stems = {path.stem for path in (dataset / 'img').glob('*.png')}
        control_stems = {path.stem for path in (dataset / 'Control1').glob('*.jpg')}
        caption_stems = {path.stem for path in (dataset / 'img').glob('*.txt')}
        self.assertEqual(img_stems, control_stems)
        self.assertEqual(img_stems, caption_stems)
        self.assertTrue(all(len(stem) == 8 for stem in img_stems))

    def test_two_control_export_is_reported_as_triplet(self):
        export_root = Path(self.tempdir.name) / 'export'
        for suffix in ('img', 'ctr1', 'ctr2'):
            folder = export_root / f'people_{suffix}'
            folder.mkdir(parents=True)
            self.save_image(folder / 'one.png')

        groups = manager.scan_exported_dataset_groups(export_root)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['pairStyle'], 'triplet')
        self.assertEqual(groups[0]['controlCount'], 2)

    def test_process_text_validation_and_backup(self):
        dataset = self.root / 'demo'
        caption = dataset / 'img' / 'one.txt'
        caption.write_text('masterpiece, blue hair', encoding='utf-8')

        rejected = self.client.post(
            '/api/process-text/preview',
            json={
                'folder': 'demo',
                'config': {'drops': ['(a+)+$'], 'slots': [], 'template': '{remainder}'}
            }
        )
        self.assertEqual(rejected.status_code, 400)

        config = {
            'drops': ['^masterpiece$'],
            'slots': [],
            'template': '{remainder}'
        }
        applied = self.client.post(
            '/api/process-text/apply',
            json={'folder': 'demo', 'config': config}
        )
        self.assertEqual(applied.status_code, 200)
        body = applied.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(caption.read_text(encoding='utf-8'), 'blue hair')
        backup = dataset / body['backup'] / 'one.txt'
        self.assertEqual(backup.read_text(encoding='utf-8'), 'masterpiece, blue hair')

    def test_postprocess_preview_and_apply_preserve_literal_text(self):
        dataset = self.root / 'demo'
        original = '  old.* old.*\r\nСтарый текст  \n'
        caption = dataset / 'img' / 'one.txt'
        caption.write_bytes(original.encode('utf-8'))
        empty = dataset / 'img' / 'empty.TXT'
        empty.write_bytes(b'')
        (dataset / 'img' / 'directory.txt').mkdir()
        control_caption = dataset / 'Control1' / 'one.txt'
        control_caption.write_text('old.*', encoding='utf-8')
        image = dataset / 'img' / 'missing.png'
        self.save_image(image)
        image_bytes_before = image.read_bytes()
        config = {'prefix': 'old.*: ', 'suffix': '\n suffix ', 'find': 'old.*', 'replace': 'new $1'}
        expected = config['prefix'] + original.replace('old.*', 'new $1') + config['suffix']

        preview = self.client.post('/api/process-text/postprocess/preview', json={
            'folder': 'demo', 'config': config, 'count': 50
        })
        self.assertEqual(preview.status_code, 200)
        body = preview.get_json()
        self.assertEqual(body['total'], 2)
        self.assertEqual(body['changed'], 2)
        sample = next(item for item in body['results'] if item['filename'] == 'one.txt')
        self.assertEqual(sample['original'], original)
        self.assertEqual(sample['processed'], expected)
        self.assertEqual(caption.read_bytes(), original.encode('utf-8'))
        self.assertFalse((dataset / '.postprocess-backup').exists())

        applied = self.client.post('/api/process-text/postprocess/apply', json={
            'folder': 'demo', 'config': config
        })
        self.assertEqual(applied.status_code, 200)
        body = applied.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['processed'], 2)
        self.assertEqual(caption.read_bytes(), expected.encode('utf-8'))
        self.assertEqual(empty.read_bytes(), (config['prefix'] + config['suffix']).encode('utf-8'))
        self.assertEqual((dataset / body['backup'] / 'one.txt').read_bytes(), original.encode('utf-8'))
        self.assertEqual((dataset / body['backup'] / 'empty.TXT').read_bytes(), b'')
        self.assertEqual(control_caption.read_text(encoding='utf-8'), 'old.*')
        self.assertFalse(image.with_suffix('.txt').exists())
        self.assertEqual(image.read_bytes(), image_bytes_before)

    def test_postprocess_operations_work_independently(self):
        caption = self.root / 'demo' / 'img' / 'one.txt'
        for config, expected in (
            ({'prefix': 'tag, '}, 'tag, old OLD old'),
            ({'suffix': ', end'}, 'old OLD old, end'),
            ({'find': 'old', 'replace': 'new'}, 'new OLD new'),
            ({'find': 'old', 'replace': ''}, ' OLD '),
        ):
            with self.subTest(config=config):
                caption.write_bytes(b'old OLD old')
                response = self.client.post('/api/process-text/postprocess/apply', json={
                    'folder': 'demo', 'config': config
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()['processed'], 1)
                self.assertEqual(caption.read_text(encoding='utf-8'), expected)

    def test_postprocess_samples_changed_captions_and_skips_unchanged_writes(self):
        dataset = self.root / 'demo'
        unchanged = dataset / 'img' / 'a.txt'
        changed = dataset / 'img' / 'z.txt'
        unchanged.write_bytes(b'keep\r\n')
        changed.write_text('old old', encoding='utf-8')
        payload = {'folder': 'demo', 'config': {'find': 'old', 'replace': 'new'}, 'count': 1}
        preview = self.client.post('/api/process-text/postprocess/preview', json=payload).get_json()
        self.assertEqual(preview['changed'], 1)
        self.assertEqual(preview['total'], 2)
        self.assertEqual(preview['results'][0]['filename'], 'z.txt')
        with patch.object(manager, 'atomic_write_text', wraps=manager.atomic_write_text) as write:
            response = self.client.post('/api/process-text/postprocess/apply', json=payload)
        self.assertEqual(response.get_json()['processed'], 1)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(write.call_args.args[0], changed)
        self.assertEqual(unchanged.read_bytes(), b'keep\r\n')

    def test_postprocess_validation_and_empty_dataset(self):
        for endpoint in ('preview', 'apply'):
            url = f'/api/process-text/postprocess/{endpoint}'
            for config in ({}, [], {'prefix': 3}, {'replace': 'new'}, {'find': None}, {'suffix': 'x' * 10001}):
                with self.subTest(endpoint=endpoint, config=repr(config)[:80]):
                    response = self.client.post(url, json={'folder': 'demo', 'config': config})
                    self.assertEqual(response.status_code, 400)
            for folder, status in (('../outside', 400), ('missing', 404), ('', 400)):
                response = self.client.post(url, json={'folder': folder, 'config': {'prefix': 'tag'}})
                self.assertEqual(response.status_code, status)
            response = self.client.post(url, json={'folder': 'demo', 'config': {'prefix': 'tag'}})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['total'], 0)
            self.assertEqual(response.get_json()['results'], [])
            self.assertIsNone(response.get_json()['backup'])

    def test_postprocess_backup_failure_leaves_all_captions_unchanged(self):
        dataset = self.root / 'demo'
        for name in ('a.txt', 'b.txt'):
            (dataset / 'img' / name).write_bytes(b'original')
        with patch.object(manager.shutil, 'copy2', side_effect=OSError('disk full')):
            response = self.client.post('/api/process-text/postprocess/apply', json={
                'folder': 'demo', 'config': {'prefix': 'tag, '}
            })
        self.assertEqual(response.status_code, 500)
        self.assertIn('Backup failed', response.get_json()['error'])
        for name in ('a.txt', 'b.txt'):
            self.assertEqual((dataset / 'img' / name).read_bytes(), b'original')

    def test_postprocess_reports_file_errors_and_retains_backups(self):
        dataset = self.root / 'demo'
        (dataset / 'img' / 'bad.txt').write_bytes(b'\xff')
        (dataset / 'img' / 'good.txt').write_bytes(b'caption')
        response = self.client.post('/api/process-text/postprocess/apply', json={
            'folder': 'demo', 'config': {'suffix': ', end'}
        })
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(body['success'])
        self.assertEqual(body['processed'], 1)
        self.assertEqual(body['errors'][0]['file'], 'bad.txt')
        self.assertEqual((dataset / 'img' / 'good.txt').read_bytes(), b'caption, end')
        self.assertEqual((dataset / 'img' / 'bad.txt').read_bytes(), b'\xff')
        self.assertEqual((dataset / body['backup'] / 'good.txt').read_bytes(), b'caption')
        self.assertEqual((dataset / body['backup'] / 'bad.txt').read_bytes(), b'\xff')

    def test_postprocess_respects_dataset_claims(self):
        self.assertEqual(manager.claim_datasets(['demo'], 'other-job'), [])
        response = self.client.post('/api/process-text/postprocess/apply', json={
            'folder': 'demo', 'config': {'prefix': 'tag, '}
        })
        self.assertEqual(response.status_code, 409)

    def test_stitch_as_is_control_reencodes_to_real_png(self):
        dataset = self.root / 'demo'
        for stem, color in (('one', (255, 0, 0)), ('two', (0, 255, 0))):
            self.save_image(dataset / 'img' / f'{stem}.png', color=color)
            self.save_image(dataset / 'Control1' / f'{stem}.jpg', 'JPEG', color=color)

        response = self.client.post('/api/stitch', json={
            'folder': 'demo',
            'filenames': ['one.png', 'two.png'],
            'direction': 'horizontal',
            'asIsControl': 'Control1'
        })
        self.assertEqual(response.status_code, 200)
        filename = response.get_json()['newFilename']
        control_path = dataset / 'Control1' / filename
        self.assertTrue(control_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        with Image.open(control_path) as image:
            self.assertEqual(image.format, 'PNG')

    def test_dataset_claim_prevents_conflicting_mutation(self):
        self.assertEqual(manager.claim_datasets(['demo'], 'job-one'), [])
        response = self.client.post(
            '/api/caption/entry.png?folder=demo',
            json={'caption': 'blocked'}
        )
        self.assertEqual(response.status_code, 409)
        manager.release_datasets('job-one')

    def test_duplicate_scan_reports_matching_pair(self):
        dataset = self.root / 'demo'
        self.save_image(dataset / 'img' / 'left.png', color=(100, 110, 120))
        self.save_image(dataset / 'img' / 'right.png', color=(100, 110, 120))
        result = manager.process_duplicate_scan_job(None, 'demo', 0)
        self.assertEqual(result['imageCount'], 2)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['pairs'][0]['distance'], 0)

    def test_duplicate_scan_background_job_completes_and_releases_claim(self):
        dataset = self.root / 'demo'
        self.save_image(dataset / 'img' / 'left.png', color=(100, 110, 120))
        self.save_image(dataset / 'img' / 'right.png', color=(100, 110, 120))
        started = self.client.post('/api/tool-jobs/start', json={
            'tool': 'duplicates',
            'payload': {'folderPath': 'demo', 'threshold': 0}
        })
        self.assertEqual(started.status_code, 200)
        job_id = started.get_json()['jobId']

        body = None
        for _ in range(100):
            body = self.client.get(f'/api/tool-jobs/status/{job_id}').get_json()
            if body['finished']:
                break
            time.sleep(0.01)
        self.assertEqual(body['status'], 'completed')
        self.assertEqual(body['result']['count'], 1)
        self.assertNotIn('demo', manager.ACTIVE_DATASETS)

    def test_export_updates_existing_folder_without_removing_unrelated_files(self):
        dataset = self.root / 'demo'
        self.save_image(dataset / 'img' / 'one.png')
        export_root = Path(self.tempdir.name) / 'exports'

        first = self.client.post(
            '/api/export?folder=demo',
            json={'exportPath': str(export_root)}
        )
        self.assertEqual(first.status_code, 200)
        exported_file = export_root / 'demo_img' / 'one.png'
        original = exported_file.read_bytes()
        unrelated_file = export_root / 'demo_img' / 'keep.png'
        self.save_image(unrelated_file, color=(1, 2, 3))

        self.save_image(dataset / 'img' / 'one.png', color=(200, 10, 20))
        second = self.client.post(
            '/api/export?folder=demo',
            json={'exportPath': str(export_root)}
        )
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(exported_file.read_bytes(), original)
        self.assertTrue(unrelated_file.is_file())
        self.assertEqual(second.get_json()['exported']['img']['overwritten'], 1)


if __name__ == '__main__':
    unittest.main()
