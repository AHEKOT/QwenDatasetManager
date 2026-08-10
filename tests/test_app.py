import io
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

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

    def test_export_refuses_to_overwrite_existing_folder(self):
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

        self.save_image(dataset / 'img' / 'one.png', color=(200, 10, 20))
        second = self.client.post(
            '/api/export?folder=demo',
            json={'exportPath': str(export_root)}
        )
        self.assertEqual(second.status_code, 409)
        self.assertEqual(exported_file.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
