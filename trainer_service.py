"""CUDA trainer integration backed by the AI Toolkit job contract.

The web application owns job creation and queueing.  The actual diffusion
training process is launched from the separately installed trainer virtual
environment and updates the same SQLite ``Job`` row that AI Toolkit's
``DiffusionTrainer`` expects.
"""

from __future__ import annotations

import copy
import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from PIL import Image


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
ACTIVE_STATUSES = {'queued', 'running', 'stopping'}
TRAINING_PRESETS = {
    'standard_lora': 'Standard edit LoRA',
    'transparent_lora': 'Transparent RGBA LoRA',
    'qwen_rgba_vae': 'Qwen RGBA VAE',
    'flux2_rgba_vae': 'FLUX.2 Klein RGBA VAE',
    'chromakey_tiny': 'QDM CleanMatte — Alpha First',
}
VAE_TRAINING_PRESETS = {'qwen_rgba_vae', 'flux2_rgba_vae'}
CHROMAKEY_PRESET = 'chromakey_tiny'
CHROMAKEY_MODEL_KEY = 'qdm_cleanmatte_v1'
CHROMAKEY_RESOLUTION_OPTIONS = {256, 320, 384, 448, 512, 640, 768, 896, 1024}
EDIT_MODELS = {
    'qwen_image_edit_2511': {
        'label': 'Qwen Image Edit 2511',
        'modelPath': 'Qwen/Qwen-Image-Edit-2511',
        'arch': 'qwen_image_edit_plus',
        'transparentArch': 'qwen_image_edit_plus_rgba',
        'license': 'Apache-2.0',
        'gated': False,
        'gateUrl': None,
        'defaultQtype': 'qfloat8',
        'noiseScheduler': 'flowmatch',
        'allowUnloadTextEncoder': False,
        'accuracyRecoveryAdapters': {
            '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors'
        },
    },
    'flux2_klein_4b': {
        'label': 'FLUX.2 Klein Base 4B',
        'modelPath': 'black-forest-labs/FLUX.2-klein-base-4B',
        'arch': 'flux2_klein_4b',
        'transparentArch': 'flux2_klein_4b_rgba',
        'license': 'Apache-2.0',
        'gated': False,
        'gateUrl': None,
        'defaultQtype': 'qfloat8',
        'noiseScheduler': 'flowmatch',
        'allowUnloadTextEncoder': True,
        'accuracyRecoveryAdapters': {},
    },
    'flux2_klein_9b': {
        'label': 'FLUX.2 Klein Base 9B',
        'modelPath': 'black-forest-labs/FLUX.2-klein-base-9B',
        'arch': 'flux2_klein_9b',
        'transparentArch': 'flux2_klein_9b_rgba',
        'license': 'FLUX Non-Commercial License',
        'gated': True,
        'gateUrl': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B',
        'defaultQtype': 'qfloat8',
        'noiseScheduler': 'flowmatch',
        'allowUnloadTextEncoder': True,
        'accuracyRecoveryAdapters': {},
    },
}
DEFAULT_TURBO_LORA_FILENAMES = {
    'qwen_image_edit_2511': 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
    'flux2_klein_4b': 'klein4b_turbo_r128.safetensors',
    'flux2_klein_9b': 'Flux_Klein_9b_Turbo_lora_rank_256_fp8_standard.safetensors',
}
QTYPE_OPTIONS = (
    '', 'qfloat8', 'float8', 'convrot8', 'convrot4', 'nvfp4',
    'convrotint7', 'convrotint6', 'convrotint5', 'convrotint4',
    'convrotint3', 'convrotint2', 'convrotbitnet',
    'uint7', 'uint6', 'uint5', 'uint4', 'uint3', 'uint2',
)
OPTIMIZER_OPTIONS = {
    'adafactor', 'adam', 'adamw', 'adamw8bit', 'automagic', 'automagic2',
    'automagic3', 'prodigyopt', 'prodigy8bit',
}
TIMESTEP_OPTIONS = {'sigmoid', 'linear', 'shift', 'weighted'}
TIMESTEP_BIAS_OPTIONS = {'balanced', 'content', 'style'}
LOSS_OPTIONS = {'mse', 'mae', 'wavelet', 'stepped'}
SAMPLER_OPTIONS = {'flowmatch', 'ddpm'}
SAVE_DTYPE_OPTIONS = {'bf16', 'fp16', 'fp32'}
RESOLUTION_OPTIONS = {256, 512, 768, 1024, 1280, 1328, 1536, 2048}
RGBA_LORA_VALIDATION_THRESHOLDS = {
    'max': {
        # Errors are measured against the alpha represented by the exact target
        # VAE latent, so these limits do not demand impossible pixel-perfect
        # inversion from the VAE/probe itself.
        'representation_alpha_mae': 0.015,
        'representation_boundary_mae': 0.035,
        'representation_boundary_edge_mae': 0.025,
        'background_residual_mae': 0.008,
        'background_false_positive_rate': 0.001,
        'foreground_false_negative_rate': 0.002,
    },
    'min': {
        'representation_alpha_iou': 0.985,
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clamp_number(value, minimum, maximum, default, *, integer=False):
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, min(maximum, parsed))
    return int(parsed) if integer else parsed


class TrainerValidationError(ValueError):
    pass


class TrainerService:
    def __init__(self, project_root: Path, datasets_dir_provider):
        self.project_root = Path(project_root).resolve()
        self.datasets_dir_provider = datasets_dir_provider
        self.root = self.project_root / 'trainer'
        self.output_dir = self.root / 'output'
        self.sample_images_dir = self.root / 'sample_images'
        self.validation_images_dir = self.root / 'validation_images'
        self.chromakey_validation_images_dir = self.root / 'chromakey_validation_images'
        self.vendor_root = self.root / 'ai_toolkit'
        self.db_path = self.root / 'trainer.db'
        self.venv_python = (
            self.root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        )
        self._db_lock = threading.RLock()
        self._worker_started = False
        self._stop_event = threading.Event()
        self.root.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_images_dir.mkdir(exist_ok=True)
        self.validation_images_dir.mkdir(exist_ok=True)
        self.chromakey_validation_images_dir.mkdir(exist_ok=True)
        self._init_db()

    @property
    def datasets_dir(self):
        return Path(self.datasets_dir_provider()).resolve()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self):
        with self._db_lock, self.connect() as connection:
            connection.executescript(
                '''
                CREATE TABLE IF NOT EXISTS "Settings" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS "Job" (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    gpu_ids TEXT NOT NULL,
                    job_config TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'stopped',
                    stop INTEGER NOT NULL DEFAULT 0,
                    return_to_queue INTEGER NOT NULL DEFAULT 0,
                    step INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER,
                    info TEXT NOT NULL DEFAULT '',
                    speed_string TEXT NOT NULL DEFAULT '',
                    queue_position INTEGER NOT NULL DEFAULT 0,
                    pid INTEGER,
                    job_type TEXT NOT NULL DEFAULT 'train',
                    job_ref TEXT,
                    save_now INTEGER NOT NULL DEFAULT 0,
                    sample_now INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS job_status_index ON "Job" (status);
                CREATE INDEX IF NOT EXISTS job_gpu_index ON "Job" (gpu_ids);
                '''
            )

    def start_worker(self):
        if self._worker_started:
            return
        self._reconcile_jobs()
        self._worker_started = True
        thread = threading.Thread(target=self._worker_loop, name='qdm-trainer-queue', daemon=True)
        thread.start()

    def _worker_loop(self):
        while not self._stop_event.wait(1.0):
            try:
                self._poll_running_jobs()
                self._start_queued_jobs()
            except Exception as exc:
                print(f'[Trainer] Queue worker error: {exc}')

    @staticmethod
    def _pid_alive(pid):
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except PermissionError:
            # Signal 0 can require more rights than the caller has on Windows.
            # Access denied proves that the process exists; treating it as dead
            # corrupts a live trainer row when a lower-privilege process imports
            # the application (for example, the test runner).
            return True
        except (OSError, ValueError):
            return False

    def _reconcile_jobs(self):
        with self._db_lock, self.connect() as connection:
            rows = connection.execute(
                'SELECT id, pid, status FROM "Job" WHERE status IN (\'running\', \'stopping\')'
            ).fetchall()
            for row in rows:
                if not self._pid_alive(row['pid']):
                    connection.execute(
                        '''UPDATE "Job" SET status = 'error', pid = NULL,
                           info = ?, updated_at = ? WHERE id = ?''',
                        ('Trainer process is no longer running', utc_now(), row['id'])
                    )

    def _poll_running_jobs(self):
        with self._db_lock, self.connect() as connection:
            rows = connection.execute(
                'SELECT id, pid, status FROM "Job" WHERE status IN (\'running\', \'stopping\')'
            ).fetchall()
            for row in rows:
                if not self._pid_alive(row['pid']):
                    connection.execute(
                        '''UPDATE "Job" SET status = 'error', pid = NULL,
                           info = ?, updated_at = ?
                           WHERE id = ? AND status IN ('running', 'stopping')''',
                        ('Trainer process exited without a final status', utc_now(), row['id'])
                    )

    def _start_queued_jobs(self):
        with self._db_lock, self.connect() as connection:
            running_gpu_sets = {
                row['gpu_ids'] for row in connection.execute(
                    'SELECT gpu_ids FROM "Job" WHERE status IN (\'running\', \'stopping\')'
                ).fetchall()
            }
            queued = connection.execute(
                '''SELECT * FROM "Job" WHERE status = 'queued'
                   ORDER BY queue_position ASC, created_at ASC'''
            ).fetchall()
            for row in queued:
                if row['gpu_ids'] in running_gpu_sets:
                    continue
                self._launch_job(connection, row)
                running_gpu_sets.add(row['gpu_ids'])

    def _launch_job(self, connection, row):
        if not self.venv_python.is_file():
            connection.execute(
                '''UPDATE "Job" SET status = 'error', info = ?, updated_at = ? WHERE id = ?''',
                ('Trainer dependencies are not installed. Run the CUDA trainer installer.', utc_now(), row['id'])
            )
            return
        config = json.loads(row['job_config'])
        cleanmatte = config.get('config', {}).get('process', [{}])[0].get('type') == 'qdm_cleanmatte_trainer'
        run_path = self.vendor_root / ('run_cleanmatte.py' if cleanmatte else 'run.py')
        if not run_path.is_file():
            connection.execute(
                '''UPDATE "Job" SET status = 'error', info = ?, updated_at = ? WHERE id = ?''',
                ('AI Toolkit backend is not installed in trainer/ai_toolkit.', utc_now(), row['id'])
            )
            return

        # A stop/requeue race can leave the previous detached Windows runner
        # alive even though the database row is queued again. Never launch a
        # second writer into the same output directory. Restore the truthful
        # runtime state so the existing process can finish or be stopped.
        previous_pid = row['pid']
        if previous_pid and self._pid_alive(previous_pid):
            connection.execute(
                '''UPDATE "Job" SET status = 'running', info = ?, updated_at = ?
                   WHERE id = ?''',
                (f'Trainer process {previous_pid} is already running', utc_now(), row['id'])
            )
            return

        config = json.loads(row['job_config'])
        self._apply_job_asset_defaults(config)
        process_config = config['config']['process'][0]
        process_config['sqlite_db_path'] = str(self.db_path)
        process_config['training_folder'] = str(self.output_dir)
        process_config['device'] = 'cuda'

        job_dir = self.output_dir / row['name']
        job_dir.mkdir(parents=True, exist_ok=True)
        config_path = job_dir / '.job_config.json'
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        # Windows does not permit renaming log.txt while an inherited stdout
        # handle (or a log viewer) has it open. A unique file per launch avoids
        # that lock entirely and also preserves every previous run.
        logs_dir = job_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        suffix = 0
        while (logs_dir / f'{suffix}_log.txt').exists():
            suffix += 1
        log_path = logs_dir / f'{suffix}_log.txt'
        (job_dir / '.active_log').write_text(
            str(log_path.relative_to(job_dir)), encoding='utf-8'
        )

        environment = os.environ.copy()
        environment.update({
            'AITK_JOB_ID': row['id'],
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES': row['gpu_ids'],
            'IS_AI_TOOLKIT_UI': '1',
            'PYTHONUNBUFFERED': '1',
        })
        hf_token = self.get_setting('HF_TOKEN') or os.environ.get('HF_TOKEN', '')
        if hf_token:
            environment['HF_TOKEN'] = hf_token

        # Claim the database row before the child is allowed to start writing
        # progress.  Previously Popen happened first, so a fast child could
        # publish "Loading model" and then have that useful status overwritten
        # by our later "Starting trainer..." update.
        connection.execute(
            '''UPDATE "Job" SET status = 'running', stop = 0, pid = NULL,
               info = 'Starting trainer...', updated_at = ? WHERE id = ?''',
            (utc_now(), row['id'])
        )

        log_handle = open(log_path, 'a', encoding='utf-8')
        kwargs = {
            'cwd': str(self.vendor_root),
            'env': environment,
            'stdin': subprocess.DEVNULL,
            'stdout': log_handle,
            'stderr': subprocess.STDOUT,
            'close_fds': True,
        }
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs['start_new_session'] = True
        try:
            process = subprocess.Popen(
                [str(self.venv_python), '-u', str(run_path), str(config_path)],
                **kwargs,
            )
        except Exception as exc:
            connection.execute(
                '''UPDATE "Job" SET status = 'error', info = ?, updated_at = ? WHERE id = ?''',
                (f'Could not start trainer: {exc}', utc_now(), row['id'])
            )
            return
        finally:
            log_handle.close()

        (job_dir / 'pid.txt').write_text(str(process.pid), encoding='utf-8')
        connection.execute(
            '''UPDATE "Job" SET pid = ?, updated_at = ? WHERE id = ?''',
            (process.pid, utc_now(), row['id'])
        )

    def get_setting(self, key):
        with self._db_lock, self.connect() as connection:
            row = connection.execute('SELECT value FROM "Settings" WHERE key = ?', (key,)).fetchone()
            return row['value'] if row else ''

    def save_settings(self, payload):
        hf_token = payload.get('hfToken')
        with self._db_lock, self.connect() as connection:
            if isinstance(hf_token, str) and hf_token.strip():
                connection.execute(
                    '''INSERT INTO "Settings" (key, value) VALUES ('HF_TOKEN', ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value''',
                    (hf_token.strip(),)
                )
            if payload.get('clearHfToken'):
                connection.execute('DELETE FROM "Settings" WHERE key = \'HF_TOKEN\'')

    def detect_gpus(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        gpus = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 4:
                continue
            gpus.append({
                'index': parts[0],
                'name': parts[1],
                'memoryTotalMb': int(parts[2]),
                'memoryFreeMb': int(parts[3]),
            })
        return gpus

    def _resolve_dataset(self, name):
        if not isinstance(name, str) or not name.strip() or '/' in name or '\\' in name or name in {'.', '..'}:
            raise TrainerValidationError('Invalid dataset name')
        root = self.datasets_dir
        candidate = (root / name.strip()).resolve()
        if candidate == root or root not in candidate.parents or candidate.is_symlink() or not candidate.is_dir():
            raise TrainerValidationError(f'Dataset not found: {name}')
        return candidate

    @staticmethod
    def _image_stems(folder):
        if not folder.is_dir():
            return set()
        return {
            path.stem for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

    @staticmethod
    def _image_files(folder):
        if not folder.is_dir():
            return []
        return sorted(
            (
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )

    def inspect_dataset(self, name):
        dataset_dir = self._resolve_dataset(name)
        target_files = self._image_files(dataset_dir / 'img')
        target_stems = {path.stem for path in target_files}
        # Dataset inspection is deliberately metadata-only. Saving, opening,
        # and queueing a job must never decode thousands of source images.
        # PNG/WebP are treated as alpha-capable; actual pixels are consumed
        # only by the trainer when a sample is requested.
        alpha_count = sum(path.suffix.lower() in {'.png', '.webp'} for path in target_files)
        controls = []
        warnings = []
        for index in range(1, 4):
            folder = dataset_dir / f'Control{index}'
            stems = self._image_stems(folder)
            if not stems:
                continue
            missing = sorted(target_stems - stems)
            extra = sorted(stems - target_stems)
            controls.append({
                'name': f'Control{index}',
                'path': str(folder),
                'count': len(stems),
                'missing': len(missing),
                'extra': len(extra),
                'missingExamples': missing[:5],
            })
            if missing:
                warnings.append(f'Control{index}: {len(missing)} target files have no matching control')
        caption_stems = {
            path.stem for path in (dataset_dir / 'img').iterdir()
            if path.is_file() and path.suffix.lower() == '.txt'
        }
        caption_count = len(target_stems & caption_stems)
        if not target_stems:
            warnings.append('Dataset has no target images')
        if not controls:
            warnings.append('Dataset has no control images')
        if caption_count < len(target_stems):
            warnings.append(f'{len(target_stems) - caption_count} target files have no caption')
        if target_files and alpha_count < len(target_files):
            warnings.append(f'{len(target_files) - alpha_count} target files have no alpha channel')
        return {
            'name': name,
            'targetPath': str(dataset_dir / 'img'),
            'targetCount': len(target_stems),
            'alphaCount': alpha_count,
            'chromaImageCount': alpha_count,
            'meaningfulAlphaCount': alpha_count,
            'opaqueBackgroundCount': len(target_files),
            'captionCount': caption_count,
            'controls': controls,
            'warnings': warnings,
            'valid': bool(target_stems and controls),
            'transparentValid': bool(target_stems and alpha_count == len(target_files)),
            'backgroundValid': bool(
                target_files
            ),
            'vaeValid': bool(len(target_files) >= 2 and alpha_count == len(target_files)),
            'chromakeyValid': bool(
                alpha_count >= 2
            ),
        }

    @staticmethod
    def _first_image(folder):
        images = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name.lower(),
        )
        return images[0] if images else None

    @staticmethod
    def _matching_image(folder, stem):
        if not folder.is_dir():
            return None
        for path in folder.iterdir():
            if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
        return None

    @staticmethod
    def _normalize_resolutions(values):
        if not isinstance(values, list):
            return [512, 768, 1024]
        normalized = set()
        for value in values:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed in RESOLUTION_OPTIONS:
                normalized.add(parsed)
        return sorted(normalized) or [512, 768, 1024]

    def _first_existing_asset(self, *relative_or_absolute):
        for value in relative_or_absolute:
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            if path.exists():
                return str(path.resolve())
        return ''

    def default_turbo_lora(self, model_key):
        filename = DEFAULT_TURBO_LORA_FILENAMES.get(model_key)
        if not filename:
            return ''
        return self._first_existing_asset(
            Path('models') / 'turbo_lora' / filename,
        )

    def default_qwen_turbo_lora(self):
        return self.default_turbo_lora('qwen_image_edit_2511')

    def default_qwen_rgba_vae(self):
        return self._first_existing_asset(
            Path('models') / 'vae' / 'QIE2511-rgba.safetensors',
        )

    def default_flux2_klein_rgba_vae(self):
        return self._first_existing_asset(
            Path('models') / 'vae' / 'flux2-klein-rgba.safetensors',
        )

    @staticmethod
    def _validate_local_asset(value, label, *, directory=False, optional=False):
        text = str(value or '').strip()
        if not text:
            if optional:
                return ''
            raise TrainerValidationError(f'{label} is required')
        path = Path(text).expanduser().resolve()
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            kind = 'directory' if directory else 'file'
            raise TrainerValidationError(f'{label} {kind} does not exist: {path}')
        return str(path)

    def _resolve_managed_target(self, dataset_name, filename=''):
        dataset_dir = self._resolve_dataset(dataset_name)
        target_dir = dataset_dir / 'img'
        if filename:
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise TrainerValidationError('Invalid managed image filename')
            target = target_dir / filename
            if not target.is_file() or target.suffix.lower() not in IMAGE_EXTENSIONS:
                raise TrainerValidationError(f'Image not found in dataset {dataset_name}: {filename}')
        else:
            target = self._first_image(target_dir)
            if target is None:
                raise TrainerValidationError(f'Dataset has no target images: {dataset_name}')
        return dataset_dir, target

    def save_sample_image(self, upload):
        original_name = Path(getattr(upload, 'filename', '') or '').name
        extension = Path(original_name).suffix.lower()
        if not original_name or extension not in IMAGE_EXTENSIONS:
            raise TrainerValidationError('Sample image must be PNG, JPG, JPEG or WebP')
        destination = self.sample_images_dir / f'{uuid.uuid4().hex}{extension}'
        try:
            upload.save(destination)
            with Image.open(destination) as image:
                image.verify()
        except Exception as exc:
            if destination.exists():
                destination.unlink()
            raise TrainerValidationError('Uploaded sample image is invalid') from exc
        return destination.resolve()

    def resolve_sample_image(self, value):
        if not isinstance(value, str) or not value.strip():
            raise TrainerValidationError('Invalid sample image path')
        candidate = Path(value.strip()).resolve()
        root = self.sample_images_dir.resolve()
        if candidate == root or root not in candidate.parents or not candidate.is_file():
            raise TrainerValidationError('Sample image is outside the trainer upload directory')
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise TrainerValidationError('Unsupported sample image type')
        return candidate

    def sample_image_preview(self, filename):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise TrainerValidationError('Invalid sample image filename')
        return self.resolve_sample_image(str(self.sample_images_dir / filename))

    def save_validation_image(self, upload):
        original_name = Path(getattr(upload, 'filename', '') or '').name
        extension = Path(original_name).suffix.lower()
        if not original_name or extension not in {'.png', '.webp'}:
            raise TrainerValidationError('Validation target must be an RGBA PNG or WebP image')
        destination = self.validation_images_dir / f'{uuid.uuid4().hex}{extension}'
        try:
            upload.save(destination)
            with Image.open(destination) as image:
                image.load()
                if 'A' not in image.getbands() and 'transparency' not in image.info:
                    raise TrainerValidationError('Validation target must contain an alpha channel')
                alpha_min, alpha_max = image.convert('RGBA').getchannel('A').getextrema()
                if alpha_min > 1 or alpha_max <= 1:
                    raise TrainerValidationError(
                        'Validation target must contain both visible and transparent pixels'
                    )
        except TrainerValidationError:
            if destination.exists():
                destination.unlink()
            raise
        except Exception as exc:
            if destination.exists():
                destination.unlink()
            raise TrainerValidationError('Uploaded validation target is invalid') from exc
        return destination.resolve()

    def resolve_validation_image(self, value):
        if not isinstance(value, str) or not value.strip():
            raise TrainerValidationError('Upload a validation target image')
        candidate = Path(value.strip()).resolve()
        root = self.validation_images_dir.resolve()
        if candidate == root or root not in candidate.parents or not candidate.is_file():
            raise TrainerValidationError('Validation image is outside the trainer upload directory')
        if candidate.suffix.lower() not in {'.png', '.webp'}:
            raise TrainerValidationError('Unsupported validation target type')
        return candidate

    def validation_image_preview(self, filename):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise TrainerValidationError('Invalid validation image filename')
        return self.resolve_validation_image(str(self.validation_images_dir / filename))

    def save_chromakey_validation_image(self, upload):
        original_name = Path(getattr(upload, 'filename', '') or '').name
        extension = Path(original_name).suffix.lower()
        if not original_name or extension not in IMAGE_EXTENSIONS:
            raise TrainerValidationError('ChromaKey validation image must be PNG, JPG or WebP')
        destination = self.chromakey_validation_images_dir / f'{uuid.uuid4().hex}{extension}'
        try:
            upload.save(destination)
            with Image.open(destination) as image:
                image.verify()
        except Exception as exc:
            if destination.exists():
                destination.unlink()
            raise TrainerValidationError('Uploaded ChromaKey validation image is invalid') from exc
        return destination.resolve()

    def resolve_chromakey_validation_image(self, value):
        if not isinstance(value, str) or not value.strip():
            raise TrainerValidationError('Upload a ChromaKey validation image')
        candidate = Path(value.strip()).resolve()
        root = self.chromakey_validation_images_dir.resolve()
        if candidate == root or root not in candidate.parents or not candidate.is_file():
            raise TrainerValidationError('ChromaKey validation image is outside the upload directory')
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise TrainerValidationError('Unsupported ChromaKey validation image type')
        return candidate

    def chromakey_validation_image_preview(self, filename):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise TrainerValidationError('Invalid ChromaKey validation image filename')
        return self.resolve_chromakey_validation_image(
            str(self.chromakey_validation_images_dir / filename)
        )

    def _black_sample_control(self, width, height):
        width = clamp_number(width, 64, 4096, 1024, integer=True)
        height = clamp_number(height, 64, 4096, 1024, integer=True)
        destination = self.sample_images_dir / f'rgba-generation-black-{width}x{height}.png'
        if not destination.is_file():
            Image.new('RGB', (width, height), (0, 0, 0)).save(destination, format='PNG')
        return destination.resolve()

    def _build_sample_items(self, payload, inspections, preset='standard_lora'):
        raw_samples = payload.get('samples', [])
        if not isinstance(raw_samples, list):
            raise TrainerValidationError('Samples must be a list')
        if len(raw_samples) > 100:
            raise TrainerValidationError('Too many sample prompts')
        selected_names = {item['name'] for item in inspections}
        samples = []
        for raw in raw_samples:
            if not isinstance(raw, dict):
                raise TrainerValidationError('Invalid sample configuration')
            explicit_controls = {}
            for control_index in range(1, 4):
                value = raw.get(f'ctrlImg{control_index}') or raw.get(f'ctrl_img_{control_index}')
                if value:
                    explicit_controls[f'ctrl_img_{control_index}'] = str(self.resolve_sample_image(value))
            if explicit_controls:
                prompt = str(raw.get('prompt', '')).strip() or 'Edit the reference image'
                sample = {'prompt': prompt, **explicit_controls}
                for key in ('width', 'height', 'seed'):
                    value = raw.get(key)
                    if value not in (None, ''):
                        sample[key] = clamp_number(value, 0, 100_000, 0, integer=True)
                multiplier = raw.get('networkMultiplier', raw.get('network_multiplier'))
                if multiplier not in (None, ''):
                    sample['network_multiplier'] = clamp_number(multiplier, -100, 100, 1)
                samples.append(sample)
                continue

            if preset == 'transparent_lora' and not raw.get('image') and not raw.get('dataset'):
                width = raw.get('width') or payload.get('sampleWidth') or 1024
                height = raw.get('height') or payload.get('sampleHeight') or 1024
                sample = {
                    'prompt': str(raw.get('prompt', '')).strip()
                    or 'Generate an isolated RGBA image with a transparent background',
                    'ctrl_img_1': str(self._black_sample_control(width, height)),
                }
                for key in ('width', 'height', 'seed'):
                    value = raw.get(key)
                    if value not in (None, ''):
                        sample[key] = clamp_number(value, 0, 100_000, 0, integer=True)
                multiplier = raw.get('networkMultiplier', raw.get('network_multiplier'))
                if multiplier not in (None, ''):
                    sample['network_multiplier'] = clamp_number(multiplier, -100, 100, 1)
                samples.append(sample)
                continue

            # Compatibility for jobs saved before sample images were separated
            # from managed training datasets.
            dataset_name = raw.get('dataset') or inspections[0]['name']
            if dataset_name not in selected_names:
                raise TrainerValidationError('Sample images must come from a selected dataset')
            dataset_dir, target = self._resolve_managed_target(dataset_name, raw.get('image', ''))
            prompt = str(raw.get('prompt', '')).strip()
            if not prompt:
                caption_path = target.with_suffix('.txt')
                if caption_path.is_file():
                    prompt = caption_path.read_text(encoding='utf-8').strip()
            if not prompt:
                prompt = 'Edit the reference image'
            sample = {'prompt': prompt}
            for control_index in range(1, 4):
                control = self._matching_image(dataset_dir / f'Control{control_index}', target.stem)
                if control is not None:
                    sample[f'ctrl_img_{control_index}'] = control.as_posix()
            if not any(key.startswith('ctrl_img_') for key in sample):
                raise TrainerValidationError(f'No matching control image for sample: {target.name}')
            for key, submitted_key in (('width', 'width'), ('height', 'height'), ('seed', 'seed')):
                value = raw.get(submitted_key)
                if value not in (None, ''):
                    sample[key] = clamp_number(value, 0, 100_000, 0, integer=True)
            multiplier = raw.get('networkMultiplier')
            if multiplier not in (None, ''):
                sample['network_multiplier'] = clamp_number(multiplier, -100, 100, 1)
            samples.append(sample)
        return samples

    def _build_validation_config(
        self,
        payload,
        inspections,
        *,
        transparent=False,
        dataset_configs=None,
    ):
        if not payload.get('validationEnabled', False):
            return None
        raw_items = payload.get('validationItems', [])
        if not isinstance(raw_items, list) or not raw_items:
            raise TrainerValidationError('Add at least one validation image')
        if len(raw_items) > 100:
            raise TrainerValidationError('Too many validation images')
        selected_names = {item['name'] for item in inspections}
        dataset_config_by_name = {
            inspection['name']: dataset_config
            for inspection, dataset_config in zip(inspections, dataset_configs or [])
        }
        items = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise TrainerValidationError('Invalid validation item')
            uploaded_target = raw.get('targetPath', raw.get('target_path'))
            if uploaded_target:
                target = self.resolve_validation_image(uploaded_target)
                dataset_config = {}
            else:
                # Compatibility for jobs saved before validation targets became
                # independent uploads.
                dataset_name = raw.get('dataset') or inspections[0]['name']
                if dataset_name not in selected_names:
                    raise TrainerValidationError('Validation image dataset is unavailable')
                _dataset_dir, target = self._resolve_managed_target(
                    dataset_name, raw.get('image', '')
                )
                dataset_config = dataset_config_by_name.get(dataset_name, {})
            prompt = str(raw.get('prompt', '')).strip()
            if not prompt and dataset_config:
                caption_ext = str(dataset_config.get('caption_ext', 'txt')).lstrip('.')
                caption_path = target.with_suffix(f'.{caption_ext}')
                if caption_path.is_file():
                    prompt = caption_path.read_text(encoding='utf-8').strip()
            if not prompt and dataset_config:
                prompt = str(dataset_config.get('default_caption', '')).strip()
            item = {'image_path': str(target), 'prompt': prompt}
            if transparent:
                with Image.open(target) as image:
                    if 'A' not in image.getbands() and 'transparency' not in image.info:
                        raise TrainerValidationError(
                            f'Transparent validation target has no alpha channel: {target.name}'
                        )
                    alpha_min, alpha_max = image.convert('RGBA').getchannel('A').getextrema()
                alpha_threshold = clamp_number(
                    payload.get('rgbaAlphaThreshold'), 0, 1, 1 / 255
                )
                threshold_u8 = round(alpha_threshold * 255)
                if alpha_min > threshold_u8 or alpha_max <= threshold_u8:
                    raise TrainerValidationError(
                        'Transparent validation target must contain both visible and '
                        f'transparent pixels: {target.name}'
                    )
                item.update({
                    'rgba_control_mode': (
                        str(raw.get('mode', raw.get('rgbaControlMode', 'generation')))
                        if uploaded_target
                        else dataset_config.get('rgba_control_mode', 'edit')
                    ),
                    'rgba_control_background_path': dataset_config.get(
                        'rgba_control_background_path'
                    ),
                    'rgba_alpha_threshold': alpha_threshold,
                    'rgba_hidden_rgb_color': [0, 0, 0],
                    'rgba_edge_color_correction': str(
                        payload.get('rgbaEdgeCorrection', 'matte_despill')
                    ),
                    'rgba_edge_matte_color': dataset_config.get(
                        'rgba_edge_matte_color', [0, 255, 0]
                    ),
                    'rgba_edge_width': clamp_number(
                        payload.get('rgbaEdgeWidth'), 0, 128, 3
                    ),
                })
                if item['rgba_control_mode'] not in {'edit', 'generation'}:
                    raise TrainerValidationError(
                        'Validation mode must be edit or generation'
                    )
            items.append(item)
        raw_sigmas = payload.get('validationSigmas', [0.5])
        if not isinstance(raw_sigmas, list):
            raw_sigmas = [0.5]
        sigmas = []
        for value in raw_sigmas:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if 0 <= parsed <= 1:
                sigmas.append(parsed)
        if not sigmas:
            sigmas = [0.5]
        config = {
            'validation_items': items,
            'resolution': clamp_number(payload.get('validationResolution'), 64, 4096, 1024, integer=True),
            'validate_every_n_steps': clamp_number(payload.get('validateEvery'), 1, 1_000_000, 1, integer=True),
            'validation_sigmas': sigmas,
        }
        if transparent:
            config['rgba_pass_thresholds'] = copy.deepcopy(
                RGBA_LORA_VALIDATION_THRESHOLDS
            )
        return config

    def list_datasets(self):
        root = self.datasets_dir
        if not root.is_dir():
            return []
        datasets = []
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if path.is_dir() and not path.is_symlink() and not path.name.startswith('.') and (path / 'img').is_dir():
                datasets.append(self.inspect_dataset(path.name))
        return datasets

    def list_dataset_names(self):
        """Match AI Toolkit's dataset-list route: enumerate folders without inspecting images."""
        root = self.datasets_dir
        if not root.is_dir():
            return []
        return [
            path.name
            for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
            if path.is_dir()
            and not path.is_symlink()
            and not path.name.startswith('.')
            and (path / 'img').is_dir()
        ]

    def validate_payload(self, payload):
        if not isinstance(payload, dict):
            raise TrainerValidationError('Invalid job payload')
        name = str(payload.get('name', '')).strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}', name):
            raise TrainerValidationError('Job name must use letters, numbers, dots, underscores or hyphens')
        preset = payload.get('trainingPreset', 'standard_lora')
        if preset not in TRAINING_PRESETS:
            raise TrainerValidationError('Unsupported training preset')
        model_key = payload.get('model')
        if preset == CHROMAKEY_PRESET:
            if model_key != CHROMAKEY_MODEL_KEY:
                raise TrainerValidationError('Unsupported ChromaKey model')
        elif model_key not in EDIT_MODELS:
            raise TrainerValidationError('Unsupported edit model')
        if preset == 'qwen_rgba_vae' and model_key != 'qwen_image_edit_2511':
            raise TrainerValidationError('Qwen RGBA VAE training requires the Qwen model family')
        if preset == 'flux2_rgba_vae' and model_key not in {'flux2_klein_4b', 'flux2_klein_9b'}:
            raise TrainerValidationError('FLUX.2 RGBA VAE training requires the Klein model family')
        gpu_ids = str(payload.get('gpuIds', '0')).strip()
        if not re.fullmatch(r'\d+(,\d+)*', gpu_ids):
            raise TrainerValidationError('GPU IDs must be a comma-separated list of numbers')
        datasets = payload.get('datasets')
        if not isinstance(datasets, list) or not datasets:
            raise TrainerValidationError('Select at least one dataset')
        if len(datasets) > 32:
            raise TrainerValidationError('Too many datasets in one job')
        inspections = []
        for dataset in datasets:
            if not isinstance(dataset, dict):
                raise TrainerValidationError('Invalid dataset configuration')
            dataset_name = dataset.get('name')
            inspection = self.inspect_dataset(dataset_name)
            if preset == 'standard_lora' and not inspection['valid']:
                raise TrainerValidationError(f'Dataset is not ready for edit training: {dataset_name}')
            if preset == 'transparent_lora' and not inspection['transparentValid']:
                raise TrainerValidationError(
                    f'Every target image must contain an alpha channel: {dataset_name}'
                )
            if preset in VAE_TRAINING_PRESETS and not inspection['vaeValid']:
                raise TrainerValidationError(
                    f'RGBA VAE training needs at least two alpha-channel images: {dataset_name}'
                )
            if preset == CHROMAKEY_PRESET and not inspection['chromakeyValid']:
                raise TrainerValidationError(
                    f'ChromaKey training needs at least two RGBA images with meaningful alpha: {dataset_name}'
                )
            inspections.append(inspection)
        return name, model_key, gpu_ids, inspections, preset

    @staticmethod
    def _normalize_chromakey_resolutions(values):
        if not isinstance(values, list):
            return [512]
        normalized = []
        for value in values:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed in CHROMAKEY_RESOLUTION_OPTIONS and parsed not in normalized:
                normalized.append(parsed)
        return sorted(normalized) or [512]

    def _build_chromakey_config(self, payload, name, model_key, gpu_ids, inspections):
        if payload.get('advancedProcess'):
            raise TrainerValidationError('Advanced overrides are not supported for CleanMatte')
        steps = clamp_number(payload.get('chromaSteps'), 1, 10_000_000, 50000, integer=True)
        def number(key, low, high, default, integer=False):
            return clamp_number(payload.get(key), low, high, default, integer=integer)
        raw_validation = payload.get('chromaValidationItems', [])
        if not isinstance(raw_validation, list) or len(raw_validation) > 100:
            raise TrainerValidationError('CleanMatte validation images must be a list of at most 100 items')
        validation = []
        for item in raw_validation:
            if not isinstance(item, dict):
                raise TrainerValidationError('Invalid CleanMatte validation image')
            path = self.resolve_chromakey_validation_image(item.get('path'))
            validation.append({'path': str(path), 'name': str(item.get('name') or path.name)[:255]})
        augmentation = {
            f'{colour}_chance': number(f'chroma{colour.title()}Chance', 0, 100, default)
            for colour, default in [('green', 70), ('blue', 30), ('white', 0), ('black', 0)]
        }
        if sum(augmentation.values()) <= 0:
            raise TrainerValidationError('At least one background colour chance must be above zero')
        augmentation.update({
            'noise_strength': number('chromaNoiseStrength', 0, 0.1, 0.01),
            'jpeg_chance': number('chromaJpegChance', 0, 100, 20),
            'spill_chance': number('chromaSpillChance', 0, 100, 35),
        })
        dtype = str(payload.get('chromaDtype', 'bf16')).lower()
        if dtype not in {'bf16', 'fp16', 'fp32'}:
            raise TrainerValidationError('Unsupported CleanMatte compute dtype')
        datasets = [{**submitted, 'name': inspection['name']}
                    for submitted, inspection in zip(payload['datasets'], inspections)]
        process = {
            'type': 'qdm_cleanmatte_trainer',
            'training_folder': str(self.output_dir),
            'sqlite_db_path': str(self.db_path),
            'device': 'cuda',
            'architecture': {'id': CHROMAKEY_MODEL_KEY, 'input_channels': 3, 'output': 'alpha'},
            'datasets': [{'folder_path': item['targetPath'], 'recursive': False} for item in inspections],
            'train': {
                'steps': steps,
                'clean_steps': number('chromaCleanSteps', 0, steps, min(10000, steps//5), True),
                'warmup_steps': number('chromaWarmupSteps', 0, steps, min(1000, steps//10), True),
                'resolutions': self._normalize_chromakey_resolutions(payload.get('chromaResolutions')),
                'batch_size': number('chromaBatchSize', 1, 64, 4, True),
                'megapixels_per_batch': number('chromaMegapixelsPerBatch', 0.25, 16, 1),
                'gradient_accumulation': number('chromaGradientAccumulation', 1, 64, 3, True),
                'lr': number('chromaLearningRate', 1e-8, 0.1, 4e-4),
                'max_grad_norm': number('chromaMaxGradNorm', 0.01, 100, 1),
                'dtype': dtype,
                'seed': number('chromaSeed', 0, 2**31-1, 42, True),
                'analytic_percent': number('chromaAnalyticPercent', 0, 50, 20),
                'detail_percent': number('chromaDetailCropPercent', 0, 100, 70),
            },
            'augmentation': augmentation,
            'loss': {
                'alpha': number('chromaLossAlpha', 0.01, 100, 5),
                'classification': number('chromaLossClassification', 0.01, 100, 1),
                'gradient': number('chromaLossGradient', 0, 100, 0.5),
                'consistency': number('chromaLossConsistency', 0, 100, 0.25),
            },
            'save': {'every': number('chromaSaveEvery', 1, steps, min(500, steps), True)},
            'validation': {
                'every': number('chromaValidateEvery', 1, steps, min(250, steps), True),
                'images': [item['path'] for item in validation],
                'max_side': number('chromaValidationMaxSide', 256, 8192, 4096, True),
                'crop_size': 512,
            },
        }
        config = {
            'job': 'extension', 'config': {'name': name, 'process': [process]},
            'meta': {'name': '[name]', 'version': '1.0', 'qdm': {
                'modelKey': model_key, 'trainingPreset': CHROMAKEY_PRESET, 'gpuIds': gpu_ids,
                'datasets': datasets, 'validationImages': validation, 'form': copy.deepcopy(payload),
                'upstreamCommit': '8a912564ce60047ea44d0f3a98becf3f168d3094',
            }},
        }
        return name, gpu_ids, config, inspections

    def _build_rgba_vae_config(self, payload, name, model_key, gpu_ids, inspections, preset):
        if payload.get('advancedProcess'):
            raise TrainerValidationError(
                'Advanced process override is not available for the RGBA VAE preset'
            )
        steps = clamp_number(payload.get('steps'), 1, 10_000_000, 5000, integer=True)
        save_every = clamp_number(payload.get('saveEvery'), 1, steps, 250, integer=True)
        validate_every = clamp_number(
            payload.get('vaeValidateEvery', save_every), 1, steps, save_every, integer=True
        )
        vae_scope = str(payload.get('vaeTrainScope', 'full')).strip().lower()
        if vae_scope not in {'full', 'alpha_boundary'}:
            raise TrainerValidationError('Unsupported RGBA VAE training scope')
        vae_dtype = str(payload.get('vaeDtype', 'bf16')).strip().lower()
        if vae_dtype not in {'bf16', 'fp16', 'fp32'}:
            raise TrainerValidationError('Unsupported RGBA VAE compute dtype')
        edge_mode = str(payload.get('rgbaEdgeCorrection', 'matte_despill'))
        if edge_mode not in {'none', 'nearest_opaque', 'matte_despill'}:
            raise TrainerValidationError('Unsupported RGBA edge cleanup mode')
        dataset_configs = []
        normalized_datasets = []
        for submitted, inspection in zip(payload['datasets'], inspections):
            dataset_configs.append({
                'folder_path': inspection['targetPath'],
                'recursive': False,
                'rgba_alpha_threshold': clamp_number(
                    payload.get('rgbaAlphaThreshold'), 0, 1, 1 / 255
                ),
                'rgba_hidden_rgb_color': [0, 0, 0],
                'rgba_edge_color_correction': edge_mode,
                'rgba_edge_matte_color': [0, 255, 0],
                'rgba_edge_width': clamp_number(payload.get('rgbaEdgeWidth'), 0.1, 128, 3),
                'flip_x': bool(submitted.get('flipX', False)),
            })
            normalized_datasets.append({**submitted, 'name': inspection['name']})

        is_flux2 = preset == 'flux2_rgba_vae'
        default_source = 'ai-toolkit/flux2_vae' if is_flux2 else 'Qwen/Qwen-Image-Edit-2511'
        default_subfolder = '' if is_flux2 else 'vae'
        process = {
            'type': 'flux2_rgba_vae_trainer' if is_flux2 else 'qwen_rgba_vae_trainer',
            'training_folder': str(self.output_dir),
            'sqlite_db_path': str(self.db_path),
            'device': 'cuda',
            'source_vae': {
                'name_or_path': str(
                    payload.get('sourceVaePath', default_source)
                ).strip() or default_source,
                'subfolder': str(payload.get('sourceVaeSubfolder', default_subfolder)).strip(),
                **({'filename': 'ae.safetensors'} if is_flux2 else {}),
                'local_files_only': bool(payload.get('sourceVaeLocalOnly', False)),
            },
            'datasets': dataset_configs,
            'train': {
                'scope': vae_scope,
                'resolution': clamp_number(payload.get('vaeResolution'), 64, 2048, 512, integer=True),
                'batch_size': clamp_number(payload.get('batchSize'), 1, 128, 1, integer=True),
                'gradient_accumulation': clamp_number(
                    payload.get('gradientAccumulation'), 1, 1024, 1, integer=True
                ),
                'steps': steps,
                'lr': clamp_number(payload.get('learningRate'), 1e-9, 1, 1e-5),
                'weight_decay': clamp_number(payload.get('weightDecay'), 0, 1, 0),
                'max_grad_norm': clamp_number(payload.get('vaeMaxGradNorm'), 0, 1000, 1),
                'dtype': vae_dtype,
                'gradient_checkpointing': bool(payload.get('vaeGradientCheckpointing', True)),
                'alpha_lr_multiplier': clamp_number(
                    payload.get('vaeAlphaLrMultiplier'), 0.1, 1000, 10
                ),
                'alpha_encoder_zero_dc': bool(payload.get('vaeAlphaEncoderZeroDc', False)),
                'num_workers': clamp_number(payload.get('vaeWorkers'), 0, 64, 2, integer=True),
            },
            'loss': {
                'visible_rgb': clamp_number(payload.get('vaeLossVisibleRgb'), 0, 1000, 1),
                'alpha': clamp_number(payload.get('vaeLossAlpha'), 0, 1000, 2),
                'alpha_edge': clamp_number(payload.get('vaeLossAlphaEdge'), 0, 1000, 1),
                'composite': clamp_number(payload.get('vaeLossComposite'), 0, 1000, 1),
                'opaque_latent': clamp_number(payload.get('vaeLossOpaqueLatent'), 0, 1000, 5),
                'opaque_rgb': clamp_number(payload.get('vaeLossOpaqueRgb'), 0, 1000, 1),
                'opaque_alpha': clamp_number(payload.get('vaeLossOpaqueAlpha'), 0, 1000, 0.5),
                'latent_delta': clamp_number(payload.get('vaeLossLatentDelta'), 0, 1000, 0.01),
                'perceptual': clamp_number(payload.get('vaeLossPerceptual'), 0, 1000, 0.1),
            },
            'save': {
                'every': save_every,
                'max_to_keep': clamp_number(payload.get('maxSaves'), 1, 1000, 4, integer=True),
                'comfy_export': bool(payload.get('vaeComfyExport', True)),
            },
            'validation': {
                'every': validate_every,
                'fraction': clamp_number(payload.get('vaeValidationFraction'), 0.001, 0.99, 0.05),
                'min_images': clamp_number(payload.get('vaeValidationMinImages'), 1, 10000, 8, integer=True),
                'max_images': clamp_number(payload.get('vaeValidationMaxImages'), 1, 10000, 32, integer=True),
                'preview_images': clamp_number(payload.get('vaePreviewImages'), 1, 100, 4, integer=True),
                'required_consecutive_passes': clamp_number(
                    payload.get('vaeRequiredPasses'), 1, 100, 2, integer=True
                ),
                'stop_when_ready': bool(payload.get('vaeStopWhenReady', False)),
                'thresholds': {
                    'finite_fraction': 1.0,
                    'visible_rgb_mae': clamp_number(payload.get('vaeReadyVisibleRgb'), 0, 1000, 0.06),
                    'alpha_mae': clamp_number(payload.get('vaeReadyAlpha'), 0, 1000, 0.08),
                    'alpha_edge_mae': clamp_number(payload.get('vaeReadyAlphaEdge'), 0, 1000, 0.12),
                    'composite_mae': clamp_number(payload.get('vaeReadyComposite'), 0, 1000, 0.05),
                    'alpha_iou': clamp_number(payload.get('vaeReadyAlphaIou'), 0, 1, 0.90),
                    'opaque_latent_rmse': clamp_number(
                        payload.get('vaeReadyOpaqueLatent'), 0, 1000, 0.03
                    ),
                },
            },
            'sample': {
                'sampler': 'vae_roundtrip',
                'format': 'png',
                'sample_every': validate_every,
                'sample_start_step': 0,
                'width': clamp_number(payload.get('vaeResolution'), 64, 2048, 512, integer=True),
                'height': clamp_number(payload.get('vaeResolution'), 64, 2048, 512, integer=True),
                'samples': [{'prompt': 'RGBA VAE round-trip validation'}],
                'seed': 0,
                'walk_seed': False,
                'guidance_scale': 0,
                'sample_steps': 0,
            },
        }
        config = {
            'job': 'extension',
            'config': {'name': name, 'process': [process]},
            'meta': {
                'name': '[name]',
                'version': '1.0',
                'qdm': {
                    'modelKey': model_key,
                    'trainingPreset': preset,
                    'gpuIds': gpu_ids,
                    'datasets': normalized_datasets,
                    'form': copy.deepcopy(payload),
                    'upstreamCommit': '8a912564ce60047ea44d0f3a98becf3f168d3094',
                },
            },
        }
        return name, gpu_ids, config, inspections

    def build_job_config(self, payload):
        name, model_key, gpu_ids, inspections, preset = self.validate_payload(payload)
        if preset == CHROMAKEY_PRESET:
            return self._build_chromakey_config(
                payload, name, model_key, gpu_ids, inspections
            )
        model = EDIT_MODELS[model_key]
        if preset in VAE_TRAINING_PRESETS:
            return self._build_rgba_vae_config(
                payload, name, model_key, gpu_ids, inspections, preset
            )
        qtype = payload.get('qtype', model['defaultQtype'])
        allowed_qtypes = set(QTYPE_OPTIONS) | set(model['accuracyRecoveryAdapters'].values())
        if qtype not in allowed_qtypes:
            known_model_adapters = {
                adapter
                for candidate in EDIT_MODELS.values()
                for adapter in candidate['accuracyRecoveryAdapters'].values()
            }
            if qtype in known_model_adapters:
                qtype = model['defaultQtype']
            else:
                raise TrainerValidationError('Unsupported quantization type')
        qtype_te = payload.get('qtypeTextEncoder', model['defaultQtype'])
        if qtype_te not in QTYPE_OPTIONS:
            raise TrainerValidationError('Unsupported text encoder quantization type')
        model_path = str(payload.get('modelPath', model['modelPath'])).strip()
        if not model_path or len(model_path) > 1024 or '\x00' in model_path:
            raise TrainerValidationError('Model name or path is invalid')
        transparent = preset == 'transparent_lora'
        model_arch = model['transparentArch'] if transparent else model['arch']
        vae_path = ''
        if transparent:
            submitted_vae_path = payload.get('vaePath')
            if not submitted_vae_path and model_key == 'qwen_image_edit_2511':
                submitted_vae_path = self.default_qwen_rgba_vae()
            elif not submitted_vae_path and model_key in {'flux2_klein_4b', 'flux2_klein_9b'}:
                submitted_vae_path = self.default_flux2_klein_rgba_vae()
            vae_path = self._validate_local_asset(
                submitted_vae_path,
                'RGBA VAE path',
                directory=Path(str(submitted_vae_path)).expanduser().is_dir(),
            )
        submitted_sample_lora_path = (
            payload.get('sampleLoraPath') or self.default_turbo_lora(model_key)
        )
        sample_lora_path = self._validate_local_asset(
            submitted_sample_lora_path, 'Turbo sampling LoRA', optional=True
        )
        if sample_lora_path and Path(sample_lora_path).suffix.lower() != '.safetensors':
            raise TrainerValidationError('Turbo sampling LoRA must be a .safetensors file')

        dataset_configs = []
        normalized_datasets = []
        dynamic_rgba_backgrounds = False
        for submitted, inspection in zip(payload['datasets'], inspections):
            control_paths = [item['path'] for item in inspection['controls'][:3]]
            resolutions = self._normalize_resolutions(submitted.get('resolutions'))
            caption_ext = str(submitted.get('captionExtension', 'txt')).strip().lstrip('.')
            if caption_ext not in {'txt', 'json', 'caption'}:
                raise TrainerValidationError('Unsupported caption extension')
            dataset_config = {
                'folder_path': inspection['targetPath'],
                'control_path': control_paths,
                'caption_ext': caption_ext,
                'default_caption': str(submitted.get('defaultCaption', '')).strip(),
                'caption_dropout_rate': clamp_number(submitted.get('captionDropout'), 0, 1, 0.05),
                'resolution': resolutions,
                'num_repeats': clamp_number(submitted.get('repeats'), 1, 1000, 1, integer=True),
                'network_weight': clamp_number(submitted.get('weight'), 0, 100, 1),
                'cache_latents_to_disk': bool(submitted.get('cacheLatents', False)),
                'is_reg': bool(submitted.get('isRegularization', False)),
                'flip_x': bool(submitted.get('flipX', False)),
                'flip_y': bool(submitted.get('flipY', False)),
            }
            if transparent:
                rgba_control_mode = str(submitted.get('rgbaControlMode', 'edit')).lower()
                if rgba_control_mode not in {'edit', 'generation'}:
                    raise TrainerValidationError('RGBA dataset mode must be edit or generation')
                edge_mode = str(payload.get('rgbaEdgeCorrection', 'matte_despill'))
                if edge_mode not in {'none', 'nearest_opaque', 'matte_despill'}:
                    raise TrainerValidationError('Unsupported RGBA edge cleanup mode')
                dataset_config.update({
                    'pixel_channels': 'rgba',
                    'rgba_require_alpha': True,
                    'rgba_alpha_threshold': clamp_number(
                        payload.get('rgbaAlphaThreshold'), 0, 1, 1 / 255
                    ),
                    'rgba_hidden_rgb_color': [0, 0, 0],
                    'rgba_edge_color_correction': edge_mode,
                    'rgba_edge_matte_color': [0, 255, 0],
                    'rgba_edge_width': clamp_number(payload.get('rgbaEdgeWidth'), 0.1, 128, 3),
                })
                dataset_config.pop('control_path', None)
                dataset_config['rgba_generate_control'] = True
                dataset_config['rgba_control_mode'] = rgba_control_mode
                dataset_config['rgba_dynamic_control_text_cache_safe'] = (
                    model_key in {'flux2_klein_4b', 'flux2_klein_9b'}
                )
                background_dataset_name = ''
                if rgba_control_mode == 'generation':
                    # The caption identifies where the base model should place
                    # the alpha mask. Dropping it would train an unrelated
                    # unconditional mask and encourage content drift.
                    dataset_config['caption_dropout_rate'] = 0.0
                else:
                    background_dataset_name = str(
                        submitted.get('rgbaBackgroundDataset', '')
                    ).strip()
                    if not background_dataset_name:
                        raise TrainerValidationError(
                            f'Select a background dataset for RGBA edit mode: {inspection["name"]}'
                        )
                    background_inspection = self.inspect_dataset(background_dataset_name)
                    if not background_inspection['backgroundValid']:
                        raise TrainerValidationError(
                            f'Background dataset must contain only readable opaque images: '
                            f'{background_dataset_name}'
                        )
                    dataset_config['rgba_control_background_path'] = background_inspection['targetPath']
                    dynamic_rgba_backgrounds = True
            dataset_configs.append(dataset_config)
            normalized_datasets.append({
                **submitted,
                'name': inspection['name'],
                'resolutions': resolutions,
                **({'rgbaControlMode': str(submitted.get('rgbaControlMode', 'edit')).lower()} if transparent else {}),
                **({'rgbaBackgroundDataset': background_dataset_name} if transparent else {}),
            })

        rank = clamp_number(payload.get('rank'), 1, 1024, 32, integer=True)
        steps = clamp_number(payload.get('steps'), 1, 10_000_000, 3000, integer=True)
        save_every = clamp_number(payload.get('saveEvery'), 1, steps, 250, integer=True)
        network_type = payload.get('networkType', 'lora')
        if network_type not in {'lora', 'lokr'}:
            raise TrainerValidationError('Unsupported training target type')
        lokr_factor = clamp_number(payload.get('lokrFactor'), -1, 32, -1, integer=True)
        if lokr_factor not in {-1, 4, 8, 16, 32}:
            raise TrainerValidationError('Unsupported LoKr factor')
        optimizer = payload.get('optimizer', 'adamw8bit')
        if optimizer not in OPTIMIZER_OPTIONS:
            raise TrainerValidationError('Unsupported optimizer')
        timestep_type = payload.get('timestepType', 'weighted')
        if timestep_type not in TIMESTEP_OPTIONS:
            raise TrainerValidationError('Unsupported timestep type')
        timestep_bias = payload.get('timestepBias', 'balanced')
        if timestep_bias not in TIMESTEP_BIAS_OPTIONS:
            raise TrainerValidationError('Unsupported timestep bias')
        loss_type = payload.get('lossType', 'mse')
        if loss_type not in LOSS_OPTIONS:
            raise TrainerValidationError('Unsupported loss type')
        sampler = payload.get('sampler', 'flowmatch')
        if sampler not in SAMPLER_OPTIONS:
            raise TrainerValidationError('Unsupported sampler')
        save_dtype = payload.get('saveDtype', 'bf16')
        if save_dtype not in SAVE_DTYPE_OPTIONS:
            raise TrainerValidationError('Unsupported save data type')

        disable_sampling = bool(payload.get('disableSampling', not payload.get('sampleEnabled', False)))
        samples = [] if disable_sampling else self._build_sample_items(payload, inspections, preset)
        validation_config = self._build_validation_config(
            payload,
            inspections,
            transparent=transparent,
            dataset_configs=dataset_configs,
        )
        model_kwargs = {'match_target_res': bool(payload.get('matchTargetResolution', False))}
        if transparent:
            model_kwargs.update({
                'rgba_lora_loss_alpha': clamp_number(
                    payload.get('rgbaLoraLossAlpha'), 0, 1000, 1
                ),
                'rgba_lora_loss_alpha_edge': clamp_number(
                    payload.get('rgbaLoraLossAlphaEdge'), 0, 1000, 0.5
                ),
            })
        dynamic_text_cache_safe = model_key in {'flux2_klein_4b', 'flux2_klein_9b'}
        klein_rgba_text_cache_required = transparent and dynamic_text_cache_safe
        # Klein RGBA training never needs a live text encoder after the initial
        # caption pass.  Keep this invariant server-side as well as in the UI so
        # a stale cached client cannot silently save both switches as false.
        unload_text_encoder = (
            bool(payload.get('unloadTextEncoder', False))
            or klein_rgba_text_cache_required
        ) and model['allowUnloadTextEncoder']
        cache_text_embeddings = (
            (
                bool(payload.get('cacheTextEmbeddings', False))
                or unload_text_encoder
                or klein_rgba_text_cache_required
            )
            and (not dynamic_rgba_backgrounds or dynamic_text_cache_safe)
        )
        dop_enabled = bool(payload.get('diffOutputPreservation', False))
        bpp_enabled = bool(payload.get('blankPromptPreservation', False)) and not dop_enabled
        skip_first_sample = bool(payload.get('skipFirstSample', False))
        force_first_sample = bool(payload.get('forceFirstSample', False)) and not skip_first_sample and not disable_sampling

        network_config = {
            'type': network_type,
            'linear': rank,
            'linear_alpha': rank,
            'lokr_factor': lokr_factor,
            'lokr_full_rank': True,
            'network_kwargs': {'ignore_if_contains': []},
        }
        train_config = {
            'batch_size': clamp_number(payload.get('batchSize'), 1, 128, 1, integer=True),
            'bypass_guidance_embedding': True,
            'steps': steps,
            'gradient_accumulation': clamp_number(payload.get('gradientAccumulation'), 1, 1024, 1, integer=True),
            'train_unet': True,
            'train_text_encoder': False,
            'gradient_checkpointing': True,
            'noise_scheduler': model['noiseScheduler'],
            'optimizer': optimizer,
            'timestep_type': timestep_type,
            'content_or_style': timestep_bias,
            'optimizer_params': {'weight_decay': clamp_number(payload.get('weightDecay'), 0, 1, 0.0001)},
            'unload_text_encoder': unload_text_encoder,
            'cache_text_embeddings': cache_text_embeddings,
            'lr': clamp_number(
                payload.get('learningRate'),
                0.000000001,
                1,
                0.00001 if transparent else 0.0001,
            ),
            'ema_config': {
                'use_ema': bool(payload.get('useEma', False)),
                'ema_decay': clamp_number(payload.get('emaDecay'), 0, 1, 0.99),
            },
            'skip_first_sample': skip_first_sample,
            'force_first_sample': force_first_sample,
            'disable_sampling': disable_sampling,
            'dtype': 'bf16',
            'diff_output_preservation': dop_enabled,
            'diff_output_preservation_multiplier': clamp_number(payload.get('dopMultiplier'), 0, 1000, 1),
            'diff_output_preservation_class': str(payload.get('dopClass', 'person')).strip() or 'person',
            'blank_prompt_preservation': bpp_enabled,
            'blank_prompt_preservation_multiplier': clamp_number(payload.get('bppMultiplier'), 0, 1000, 1),
            'do_guidance_loss': bool(payload.get('guidanceLoss', False)),
            'guidance_loss_target': clamp_number(payload.get('guidanceLossTarget'), 0, 1000, 4),
            'do_differential_guidance': bool(payload.get('differentialGuidance', False)),
            'differential_guidance_scale': clamp_number(payload.get('differentialGuidanceScale'), 0, 1000, 3),
            'loss_type': loss_type,
        }
        if validation_config is not None:
            train_config['validation_config'] = validation_config

        compile_model = bool(payload.get('compileModel', False))
        normalized_form = copy.deepcopy(payload)
        normalized_form['datasets'] = copy.deepcopy(normalized_datasets)
        normalized_form['cacheTextEmbeddings'] = cache_text_embeddings
        normalized_form['unloadTextEncoder'] = unload_text_encoder
        normalized_form['qtype'] = qtype
        config = {
            'job': 'extension',
            'config': {
                'name': name,
                'process': [{
                    'type': 'diffusion_trainer',
                    'training_folder': str(self.output_dir),
                    'sqlite_db_path': str(self.db_path),
                    'device': 'cuda',
                    'trigger_word': str(payload.get('triggerWord', '')).strip() or None,
                    'performance_log_every': 10,
                    'network': network_config,
                    'save': {
                        'dtype': save_dtype,
                        'save_every': save_every,
                        'max_step_saves_to_keep': clamp_number(payload.get('maxSaves'), 1, 1000, 4, integer=True),
                        'save_format': 'diffusers',
                        'push_to_hub': False,
                    },
                    'datasets': dataset_configs,
                    'train': train_config,
                    'logging': {'log_every': 1, 'use_ui_logger': True},
                    'model': {
                        'name_or_path': model_path,
                        'arch': model_arch,
                        'quantize': bool(qtype),
                        'qtype': qtype or 'qfloat8',
                        'quantize_te': bool(qtype_te),
                        'qtype_te': qtype_te or 'qfloat8',
                        'low_vram': bool(payload.get('lowVram', True)),
                        'layer_offloading': bool(payload.get('layerOffloading', False)),
                        'layer_offloading_transformer_percent': clamp_number(payload.get('transformerOffload'), 0, 1, 1),
                        'layer_offloading_text_encoder_percent': clamp_number(payload.get('textEncoderOffload'), 0, 1, 1),
                        'model_kwargs': model_kwargs,
                        'compile': compile_model,
                        **({'vae_path': vae_path} if vae_path else {}),
                        **({'sample_lora_path': sample_lora_path} if sample_lora_path else {}),
                        **({'block_compile': True} if compile_model else {}),
                    },
                    'sample': {
                        'sampler': sampler,
                        **({'format': 'png'} if transparent else {}),
                        'sample_every': clamp_number(payload.get('sampleEvery'), 1, steps, save_every, integer=True),
                        'sample_start_step': clamp_number(payload.get('sampleStartStep'), 0, steps, 0, integer=True),
                        'width': clamp_number(payload.get('sampleWidth'), 64, 4096, 1024, integer=True),
                        'height': clamp_number(payload.get('sampleHeight'), 64, 4096, 1024, integer=True),
                        'samples': samples,
                        'neg': '',
                        'seed': clamp_number(payload.get('sampleSeed'), 0, 2**32 - 1, 42, integer=True),
                        'walk_seed': bool(payload.get('walkSeed', True)),
                        'guidance_scale': (
                            1.0 if sample_lora_path and model_key == 'qwen_image_edit_2511'
                            else clamp_number(payload.get('guidanceScale'), 0, 1000, 4)
                        ),
                        'sample_steps': (
                            4 if sample_lora_path and model_key == 'qwen_image_edit_2511'
                            else clamp_number(payload.get('sampleSteps'), 1, 1000, 30, integer=True)
                        ),
                    },
                }],
            },
            'meta': {
                'name': '[name]',
                'version': '1.0',
                'qdm': {
                    'modelKey': model_key,
                    'trainingPreset': preset,
                    'gpuIds': gpu_ids,
                    'datasets': normalized_datasets,
                    'form': normalized_form,
                    'upstreamCommit': '8a912564ce60047ea44d0f3a98becf3f168d3094',
                },
            },
        }
        advanced_process = payload.get('advancedProcess')
        if advanced_process:
            if isinstance(advanced_process, str):
                try:
                    advanced_process = json.loads(advanced_process)
                except json.JSONDecodeError as exc:
                    raise TrainerValidationError(f'Advanced process JSON is invalid: {exc.msg}') from exc
            if not isinstance(advanced_process, dict):
                raise TrainerValidationError('Advanced process config must be an object')
            advanced_process = copy.deepcopy(advanced_process)
            if advanced_process.get('type', 'diffusion_trainer') != 'diffusion_trainer':
                raise TrainerValidationError('Only the LoRA diffusion trainer is supported')
            advanced_model = advanced_process.get('model')
            if not isinstance(advanced_model, dict):
                raise TrainerValidationError('Advanced process config is missing model settings')
            if advanced_model.get('arch', model_arch) != model_arch:
                raise TrainerValidationError('Advanced config cannot change the selected edit architecture')
            advanced_qtype = advanced_model.get('qtype', model['defaultQtype']) if advanced_model.get('quantize', True) else ''
            if advanced_qtype not in allowed_qtypes:
                raise TrainerValidationError('Unsupported transformer quantization in advanced config')
            advanced_qtype_te = advanced_model.get('qtype_te', model['defaultQtype']) if advanced_model.get('quantize_te', True) else ''
            if advanced_qtype_te not in QTYPE_OPTIONS:
                raise TrainerValidationError('Unsupported text encoder quantization in advanced config')
            advanced_network = advanced_process.get('network')
            if not isinstance(advanced_network, dict) or advanced_network.get('type', 'lora') not in {'lora', 'lokr'}:
                raise TrainerValidationError('Advanced config supports only LoRA or LoKr targets')
            advanced_train = advanced_process.get('train')
            if not isinstance(advanced_train, dict):
                raise TrainerValidationError('Advanced process config is missing training settings')
            advanced_train['noise_scheduler'] = model['noiseScheduler']
            if klein_rgba_text_cache_required:
                advanced_train['unload_text_encoder'] = True
                advanced_train['cache_text_embeddings'] = True
            elif dynamic_rgba_backgrounds and not dynamic_text_cache_safe:
                advanced_train['cache_text_embeddings'] = False
            elif advanced_train.get('unload_text_encoder', unload_text_encoder):
                advanced_train['cache_text_embeddings'] = True
            advanced_sample = advanced_process.get('sample')
            if not isinstance(advanced_sample, dict) or advanced_sample.get('sampler', 'flowmatch') not in SAMPLER_OPTIONS:
                raise TrainerValidationError('Unsupported sampler in advanced config')
            if transparent:
                advanced_sample['format'] = 'png'
            if sample_lora_path and model_key == 'qwen_image_edit_2511':
                advanced_sample['sample_steps'] = 4
                advanced_sample['guidance_scale'] = 1.0
            advanced_process.update({
                'type': 'diffusion_trainer',
                'training_folder': str(self.output_dir),
                'sqlite_db_path': str(self.db_path),
                'device': 'cuda',
                'performance_log_every': 10,
                'datasets': dataset_configs,
            })
            advanced_model['arch'] = model_arch
            if transparent:
                advanced_model.setdefault('model_kwargs', {}).update(model_kwargs)
            if vae_path:
                advanced_model['vae_path'] = vae_path
            else:
                advanced_model.pop('vae_path', None)
            if sample_lora_path:
                advanced_model['sample_lora_path'] = sample_lora_path
            else:
                advanced_model.pop('sample_lora_path', None)
            config['config']['process'][0] = advanced_process
        return name, gpu_ids, config, inspections

    def create_job(self, payload):
        name, gpu_ids, config, inspections = self.build_job_config(payload)
        job_id = str(uuid.uuid4())
        now = utc_now()
        with self._db_lock, self.connect() as connection:
            highest = connection.execute('SELECT MAX(queue_position) AS value FROM "Job"').fetchone()['value'] or 0
            try:
                connection.execute(
                    '''INSERT INTO "Job" (
                        id, name, gpu_ids, job_config, created_at, updated_at,
                        status, info, queue_position, total_steps
                    ) VALUES (?, ?, ?, ?, ?, ?, 'stopped', 'Ready', ?, ?)''',
                    (job_id, name, gpu_ids, json.dumps(config), now, now, highest + 1000,
                     config['config']['process'][0]['train']['steps'])
                )
            except sqlite3.IntegrityError as exc:
                raise TrainerValidationError('A job with this name already exists') from exc
        return self.get_job(job_id), inspections

    def update_job(self, job_id, payload):
        existing = self._get_job_row(job_id)
        if existing['status'] in ACTIVE_STATUSES:
            raise TrainerValidationError('Stop the job before editing it')
        name, gpu_ids, config, inspections = self.build_job_config(payload)
        with self._db_lock, self.connect() as connection:
            try:
                connection.execute(
                    '''UPDATE "Job" SET name = ?, gpu_ids = ?, job_config = ?,
                       total_steps = ?, updated_at = ?, info = 'Ready' WHERE id = ?''',
                    (name, gpu_ids, json.dumps(config), config['config']['process'][0]['train']['steps'], utc_now(), job_id)
                )
            except sqlite3.IntegrityError as exc:
                raise TrainerValidationError('A job with this name already exists') from exc
        return self.get_job(job_id), inspections

    @staticmethod
    def _next_clone_name(connection, source_name):
        suffix_index = 1
        while True:
            suffix = '_copy' if suffix_index == 1 else f'_copy_{suffix_index}'
            candidate = f'{source_name[:96 - len(suffix)]}{suffix}'
            exists = connection.execute(
                'SELECT 1 FROM "Job" WHERE name = ?', (candidate,)
            ).fetchone()
            if exists is None:
                return candidate
            suffix_index += 1

    def clone_job(self, job_id):
        source = self._get_job_row(job_id)
        config = json.loads(source['job_config'])
        job_id_copy = str(uuid.uuid4())
        now = utc_now()
        with self._db_lock, self.connect() as connection:
            clone_name = self._next_clone_name(connection, source['name'])
            config.setdefault('config', {})['name'] = clone_name
            qdm = config.setdefault('meta', {}).setdefault('qdm', {})
            form = qdm.get('form')
            if isinstance(form, dict):
                form['name'] = clone_name
            highest = connection.execute(
                'SELECT MAX(queue_position) AS value FROM "Job"'
            ).fetchone()['value'] or 0
            connection.execute(
                '''INSERT INTO "Job" (
                    id, name, gpu_ids, job_config, created_at, updated_at,
                    status, stop, return_to_queue, step, total_steps, info,
                    speed_string, queue_position, pid, job_type, job_ref,
                    save_now, sample_now
                ) VALUES (?, ?, ?, ?, ?, ?, 'stopped', 0, 0, 0, ?, 'Ready',
                          '', ?, NULL, ?, NULL, 0, 0)''',
                (
                    job_id_copy,
                    clone_name,
                    source['gpu_ids'],
                    json.dumps(config),
                    now,
                    now,
                    source['total_steps'],
                    highest + 1000,
                    source['job_type'],
                ),
            )
        return self.get_job(job_id_copy)

    def _get_job_row(self, job_id):
        with self._db_lock, self.connect() as connection:
            row = connection.execute('SELECT * FROM "Job" WHERE id = ?', (job_id,)).fetchone()
        if row is None:
            raise FileNotFoundError('Training job not found')
        return row

    def _apply_job_asset_defaults(self, config):
        """Hydrate assets omitted by jobs saved before model defaults existed."""
        qdm = config.get('meta', {}).get('qdm', {})
        if qdm.get('trainingPreset') in VAE_TRAINING_PRESETS:
            return False
        model_key = qdm.get('modelKey')
        process = config.get('config', {}).get('process', [{}])[0]
        model = process.get('model')
        if not isinstance(model, dict):
            return False
        changed = False
        form = qdm.get('form')

        if qdm.get('trainingPreset') == 'transparent_lora' and model_key == 'qwen_image_edit_2511':
            default_vae = self.default_qwen_rgba_vae()
            configured_vae = str(model.get('vae_path') or '')
            legacy_vae = Path(configured_vae).name == 'TransparentQIE2511VAE_diffusers'
            if default_vae and (not configured_vae or legacy_vae):
                model['vae_path'] = default_vae
                changed = True
            if isinstance(form, dict):
                form_vae = str(form.get('vaePath') or '')
                if default_vae and (
                    not form_vae
                    or Path(form_vae).name == 'TransparentQIE2511VAE_diffusers'
                ):
                    form['vaePath'] = default_vae
                    changed = True

        default_sample_lora = self.default_turbo_lora(model_key)
        if default_sample_lora and not model.get('sample_lora_path'):
            model['sample_lora_path'] = default_sample_lora
            changed = True
        if default_sample_lora and isinstance(form, dict) and not form.get('sampleLoraPath'):
            form['sampleLoraPath'] = default_sample_lora
            changed = True
        return changed

    def _serialize_job(self, row):
        data = dict(row)
        config = json.loads(data.pop('job_config'))
        self._apply_job_asset_defaults(config)
        qdm = config.get('meta', {}).get('qdm', {})
        data['config'] = config
        form = copy.deepcopy(qdm.get('form', {}))
        process = config.get('config', {}).get('process', [{}])[0]
        configured_qtype = process.get('model', {}).get('qtype')
        if isinstance(configured_qtype, str) and '|' in configured_qtype:
            form['qtype'] = configured_qtype
        data['form'] = form
        data['datasets'] = [item.get('name') for item in qdm.get('datasets', [])]
        total = data.get('total_steps') or config['config']['process'][0]['train']['steps']
        data['total_steps'] = total
        data['progress'] = min(100, round((data.get('step', 0) / total) * 100, 2)) if total else 0
        return data

    def get_job(self, job_id):
        return self._serialize_job(self._get_job_row(job_id))

    def list_jobs(self):
        with self._db_lock, self.connect() as connection:
            rows = connection.execute('SELECT * FROM "Job" ORDER BY created_at DESC').fetchall()
        return [self._serialize_job(row) for row in rows]

    def bootstrap_state(self):
        """Return independent, lightweight resources used by the first trainer render."""
        return {
            'models': self.model_options(),
            'trainingPresets': [
                {'key': key, 'label': label}
                for key, label in TRAINING_PRESETS.items()
            ],
            'qtypes': list(QTYPE_OPTIONS),
            'datasets': [
                {
                    'name': name,
                    'inspected': False,
                    'targetPath': str(self.datasets_dir / name / 'img'),
                    # Directory enumeration is cheap and prevents a real,
                    # large dataset from looking empty while the asynchronous
                    # alpha inspection is still running.
                    'targetCount': len(self._image_files(self.datasets_dir / name / 'img')),
                    'alphaCount': None,
                    'chromaImageCount': sum(
                        path.suffix.lower() in {'.png', '.webp'}
                        for path in self._image_files(self.datasets_dir / name / 'img')
                    ),
                    'captionCount': None,
                    'controls': [],
                    'warnings': [],
                }
                for name in self.list_dataset_names()
            ],
            'jobs': self.list_jobs(),
            'gpus': self.detect_gpus(),
            'trainerInstalled': self.venv_python.is_file() and (self.vendor_root / 'run.py').is_file(),
            'hfTokenConfigured': bool(self.get_setting('HF_TOKEN') or os.environ.get('HF_TOKEN')),
            'upstreamCommit': '8a912564ce60047ea44d0f3a98becf3f168d3094',
        }

    def model_options(self):
        models = []
        for key, value in EDIT_MODELS.items():
            item = {**value, 'key': key}
            item['defaultSampleLoraPath'] = self.default_turbo_lora(key)
            item['defaultRgbaVaePath'] = (
                self.default_qwen_rgba_vae()
                if key == 'qwen_image_edit_2511'
                else self.default_flux2_klein_rgba_vae()
            )
            models.append(item)
        models.append({
            'key': CHROMAKEY_MODEL_KEY,
            'label': 'QDM CleanMatte — Alpha First',
            'kind': 'chromakey',
            'modelPath': '',
            'license': 'Project model',
            'gated': False,
            'gateUrl': None,
            'defaultQtype': '',
            'noiseScheduler': 'supervised',
            'allowUnloadTextEncoder': False,
            'accuracyRecoveryAdapters': {},
            'defaultSampleLoraPath': '',
            'defaultRgbaVaePath': '',
        })
        return models

    def active_dataset_names(self):
        names = set()
        with self._db_lock, self.connect() as connection:
            rows = connection.execute(
                'SELECT job_config FROM "Job" WHERE status IN (\'running\', \'stopping\')'
            ).fetchall()
        for row in rows:
            try:
                config = json.loads(row['job_config'])
                for item in config.get('meta', {}).get('qdm', {}).get('datasets', []):
                    if isinstance(item.get('name'), str):
                        names.add(item['name'])
                    if isinstance(item.get('rgbaBackgroundDataset'), str) and item['rgbaBackgroundDataset']:
                        names.add(item['rgbaBackgroundDataset'])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return names

    def queue_job(self, job_id):
        row = self._get_job_row(job_id)
        if row['status'] in {'running', 'stopping'}:
            raise TrainerValidationError('Job is already active')
        config = json.loads(row['job_config'])
        self._apply_job_asset_defaults(config)
        qdm_meta = config.get('meta', {}).get('qdm', {})
        preset = qdm_meta.get('trainingPreset', 'standard_lora')
        for item in qdm_meta.get('datasets', []):
            inspection = self.inspect_dataset(item['name'])
            ready = inspection['valid']
            if preset == 'transparent_lora':
                ready = inspection['transparentValid']
            elif preset in VAE_TRAINING_PRESETS:
                ready = inspection['vaeValid']
            elif preset == CHROMAKEY_PRESET:
                ready = inspection['chromakeyValid']
            if not ready:
                raise TrainerValidationError(f"Dataset is not ready: {item['name']}")
            if preset == 'transparent_lora' and item.get('rgbaControlMode', 'edit') == 'edit':
                background_name = item.get('rgbaBackgroundDataset')
                if not background_name:
                    raise TrainerValidationError(
                        f"Background dataset is not selected: {item['name']}"
                    )
                background = self.inspect_dataset(background_name)
                if not background['backgroundValid']:
                    raise TrainerValidationError(
                        f"Background dataset is not ready: {background_name}"
                    )
        with self._db_lock, self.connect() as connection:
            highest = connection.execute('SELECT MAX(queue_position) AS value FROM "Job"').fetchone()['value'] or 0
            connection.execute(
                '''UPDATE "Job" SET status = 'queued', stop = 0, return_to_queue = 0,
                   queue_position = ?, info = 'Queued', job_config = ?, updated_at = ? WHERE id = ?''',
                (highest + 1000, json.dumps(config), utc_now(), job_id)
            )
        return self.get_job(job_id)

    def stop_job(self, job_id):
        row = self._get_job_row(job_id)
        if row['status'] == 'queued':
            with self._db_lock, self.connect() as connection:
                connection.execute(
                    '''UPDATE "Job" SET status = 'stopped', stop = 1,
                       info = 'Removed from queue', updated_at = ? WHERE id = ?''',
                    (utc_now(), job_id)
                )
            return self.get_job(job_id)
        if row['status'] not in {'running', 'stopping'}:
            raise TrainerValidationError('Job is not running')
        with self._db_lock, self.connect() as connection:
            connection.execute(
                '''UPDATE "Job" SET status = 'stopping', stop = 1,
                   info = 'Stopping trainer...', updated_at = ? WHERE id = ?''',
                (utc_now(), job_id)
            )
        if os.name != 'nt' and row['pid'] and self._pid_alive(row['pid']):
            try:
                os.kill(int(row['pid']), signal.SIGINT)
            except OSError:
                pass
        return self.get_job(job_id)

    def request_runtime_action(self, job_id, action):
        column = {'save': 'save_now', 'sample': 'sample_now'}.get(action)
        if column is None:
            raise TrainerValidationError('Unsupported trainer action')
        row = self._get_job_row(job_id)
        if row['status'] != 'running':
            raise TrainerValidationError('Job must be running')
        with self._db_lock, self.connect() as connection:
            connection.execute(
                f'''UPDATE "Job" SET {column} = 1, info = ?, updated_at = ? WHERE id = ?''',
                (f'{action.capitalize()} requested', utc_now(), job_id),
            )
        return self.get_job(job_id)

    def delete_job(self, job_id):
        row = self._get_job_row(job_id)
        if row['status'] in ACTIVE_STATUSES:
            raise TrainerValidationError('Stop the job before deleting it')
        with self._db_lock, self.connect() as connection:
            connection.execute('DELETE FROM "Job" WHERE id = ?', (job_id,))

    def job_log_path(self, job_id):
        row = self._get_job_row(job_id)
        job_dir = (self.output_dir / row['name']).resolve()
        pointer = job_dir / '.active_log'
        if pointer.is_file():
            try:
                relative = Path(pointer.read_text(encoding='utf-8').strip())
                candidate = (job_dir / relative).resolve()
                if (
                    candidate != job_dir
                    and job_dir in candidate.parents
                    and candidate.suffix.lower() == '.txt'
                ):
                    return candidate
            except OSError:
                pass
        return job_dir / 'log.txt'

    def job_model_artifact(self, job_id):
        row = self._get_job_row(job_id)
        job_dir = (self.output_dir / row['name']).resolve()
        manifest_path = job_dir / 'artifacts.json'
        if not manifest_path.is_file():
            raise FileNotFoundError('This job has no exported model yet')
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainerValidationError('The model artifact manifest is invalid') from exc
        filename = manifest.get('primary')
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise TrainerValidationError('The model artifact filename is invalid')
        candidate = (job_dir / filename).resolve()
        if job_dir not in candidate.parents or candidate.suffix.lower() != '.safetensors':
            raise TrainerValidationError('The model artifact path is invalid')
        if not candidate.is_file():
            raise FileNotFoundError('The exported model file is missing')
        return candidate

    def read_log(self, job_id, max_bytes=200_000):
        path = self.job_log_path(job_id)
        if not path.is_file():
            return ''
        size = path.stat().st_size
        with open(path, 'rb') as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode('utf-8', errors='replace')

    def validation_results(self, job_id):
        row = self._get_job_row(job_id)
        config = json.loads(row['job_config'])
        if config.get('config', {}).get('process', [{}])[0].get('type') == 'qdm_cleanmatte_trainer':
            result_path = self.output_dir / row['name'] / 'cleanmatte_validation.json'
            if not result_path.is_file():
                return {'step': None, 'passed': None, 'items': []}
            return json.loads(result_path.read_text(encoding='utf-8'))
        metrics_path = (self.output_dir / row['name'] / 'loss_log.db').resolve()
        if not metrics_path.is_file():
            return {'step': None, 'passed': None, 'items': []}
        connection = sqlite3.connect(
            f'file:{metrics_path.as_posix()}?mode=ro', uri=True, timeout=5.0
        )
        try:
            latest = connection.execute(
                "SELECT MAX(step) FROM metrics WHERE key = 'val/rgba_pass'"
            ).fetchone()[0]
            if latest is None:
                return {'step': None, 'passed': None, 'items': []}
            metric_rows = connection.execute(
                "SELECT key, value_real, value_text FROM metrics "
                "WHERE step = ? AND (key = 'val/rgba_pass' "
                "OR key = 'val/rgba_failed_checks' OR key LIKE 'val/item_%')",
                (latest,),
            ).fetchall()
        finally:
            connection.close()

        process = config.get('config', {}).get('process', [{}])[0]
        configured_items = (
            process.get('train', {})
            .get('validation_config', {})
            .get('validation_items', [])
        )
        items = {}
        overall_pass = None
        overall_failed = 'none'
        for key, value_real, value_text in metric_rows:
            if key == 'val/rgba_pass':
                overall_pass = bool(value_real)
                continue
            if key == 'val/rgba_failed_checks':
                overall_failed = value_text or 'none'
                continue
            match = re.fullmatch(
                r'val/item_(\d+)_(rgba_pass|failed_checks|worst_(.+))', key
            )
            if not match:
                continue
            index = int(match.group(1)) - 1
            item = items.setdefault(index, {'index': index, 'metrics': {}})
            field = match.group(2)
            if field == 'rgba_pass':
                item['passed'] = bool(value_real)
            elif field == 'failed_checks':
                item['failedChecks'] = value_text or 'none'
            else:
                item['metrics'][match.group(3)] = value_real
        result_items = []
        for index in sorted(items):
            item = items[index]
            configured = configured_items[index] if index < len(configured_items) else {}
            item['name'] = Path(str(configured.get('image_path', ''))).name or (
                f'Validation image {index + 1}'
            )
            item.setdefault('passed', False)
            item.setdefault('failedChecks', 'missing validation result')
            result_items.append(item)
        return {
            'step': int(latest),
            'passed': overall_pass,
            'failedChecks': overall_failed,
            'items': result_items,
        }

    def _job_samples_dir(self, job_id):
        row = self._get_job_row(job_id)
        output_root = self.output_dir.resolve()
        samples_dir = (self.output_dir / row['name'] / 'samples').resolve()
        if samples_dir == output_root or output_root not in samples_dir.parents:
            raise TrainerValidationError('Invalid trainer samples directory')
        return row, samples_dir

    @staticmethod
    def _sample_file_info(path):
        chroma_match = re.fullmatch(
            r'chroma_(\d+)_([0-9]+)(?:_(sheet))?', path.stem
        )
        if chroma_match:
            base_index = int(chroma_match.group(2))
            return {
                'name': path.name,
                'step': int(chroma_match.group(1)),
                'sampleIndex': base_index * 2 + (1 if chroma_match.group(3) else 0),
                'size': path.stat().st_size,
                'modifiedAt': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        match = re.search(r'_(\d+)_([0-9]+)$', path.stem)
        return {
            'name': path.name,
            'step': int(match.group(1)) if match else 0,
            'sampleIndex': int(match.group(2)) if match else 0,
            'size': path.stat().st_size,
            'modifiedAt': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }

    @staticmethod
    def _sample_items(process):
        sample_config = process.get('sample', {}) if isinstance(process, dict) else {}
        items = sample_config.get('samples') or []
        if not items and sample_config.get('prompts'):
            items = [{'prompt': prompt} for prompt in sample_config['prompts']]
        return sample_config, [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _sample_controls(item):
        controls = item.get('ctrl_img')
        if isinstance(controls, str):
            result = [controls]
        elif isinstance(controls, list):
            result = list(controls)
        else:
            result = []
        for index in range(1, 4):
            value = item.get(f'ctrl_img_{index}')
            if value and value not in result:
                result.append(value)
        return [str(value) for value in result if value]

    def list_job_samples(self, job_id):
        row, samples_dir = self._job_samples_dir(job_id)
        config = json.loads(row['job_config'])
        process = config.get('config', {}).get('process', [{}])[0]
        sample_config, sample_items = self._sample_items(process)
        if process.get('type') in {'qdm_chromakey_trainer', 'qdm_cleanmatte_trainer'}:
            validation_images = process.get('validation', {}).get('images', [])
            extra = 2 if process.get('type') == 'qdm_cleanmatte_trainer' else 0
            sample_count = max((len(validation_images) + extra) * 2, 1)
        else:
            sample_count = max(len(sample_items), 1)
        files = []
        if samples_dir.is_dir():
            files = sorted(
                (
                    path for path in samples_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: path.name.lower(),
            )
        samples = []
        for path in files:
            info = self._sample_file_info(path)
            sample_index = info['sampleIndex']
            item = sample_items[sample_index] if sample_index < len(sample_items) else {}
            explicit_seed = item.get('seed')
            if explicit_seed is not None:
                seed = explicit_seed
            elif sample_config.get('walk_seed'):
                seed = int(sample_config.get('seed', 0)) + sample_index
            else:
                seed = sample_config.get('seed')
            info.update({
                'prompt': str(item.get('prompt', '')),
                'seed': seed,
                'controlCount': len(self._sample_controls(item)),
            })
            samples.append(info)
        return {
            'samples': samples,
            'sampleCount': sample_count,
            'isVae': process.get('type') in {'qwen_rgba_vae_trainer', 'flux2_rgba_vae_trainer'},
            'isChroma': process.get('type') in {'qdm_chromakey_trainer', 'qdm_cleanmatte_trainer'},
        }

    def resolve_job_sample(self, job_id, filename, thumbnail=False):
        _row, samples_dir = self._job_samples_dir(job_id)
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise TrainerValidationError('Invalid sample filename')
        candidate = (samples_dir / filename).resolve()
        if samples_dir not in candidate.parents or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise TrainerValidationError('Unsupported sample image')
        if not candidate.is_file():
            raise FileNotFoundError('Validation image not found')
        if thumbnail:
            thumb_base = samples_dir / '.thumbs' / candidate.name
            for suffix in ('.png', '.jpg'):
                thumb = Path(f'{thumb_base}{suffix}')
                if thumb.is_file():
                    return thumb
        return candidate

    def resolve_job_sample_control(self, job_id, sample_index, control_index):
        row = self._get_job_row(job_id)
        config = json.loads(row['job_config'])
        process = config.get('config', {}).get('process', [{}])[0]
        _sample_config, sample_items = self._sample_items(process)
        if sample_index < 0 or sample_index >= len(sample_items):
            raise FileNotFoundError('Sample configuration not found')
        controls = self._sample_controls(sample_items[sample_index])
        if control_index < 0 or control_index >= len(controls):
            raise FileNotFoundError('Control image not found')
        candidate = Path(controls[control_index]).resolve()
        allowed_roots = (self.sample_images_dir.resolve(), self.datasets_dir.resolve())
        if not any(candidate != root and root in candidate.parents for root in allowed_roots):
            raise TrainerValidationError('Control image is outside managed trainer directories')
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError('Control image not found')
        return candidate

    def delete_job_sample(self, job_id, filename):
        path = self.resolve_job_sample(job_id, filename)
        samples_dir = path.parent
        path.unlink()
        caption = path.with_suffix('.txt')
        if caption.is_file():
            caption.unlink()
        thumb_base = samples_dir / '.thumbs' / path.name
        for suffix in ('.png', '.jpg'):
            thumb = Path(f'{thumb_base}{suffix}')
            if thumb.is_file():
                thumb.unlink()

    def build_samples_archive(self, job_id):
        row, samples_dir = self._job_samples_dir(job_id)
        if not samples_dir.is_dir():
            raise FileNotFoundError('No validation images have been generated')
        files = sorted(
            path for path in samples_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError('No validation images have been generated')
        archive_path = samples_dir.parent / 'samples.zip'
        temporary_path = samples_dir.parent / f'.samples-{uuid.uuid4().hex}.zip.tmp'
        try:
            with zipfile.ZipFile(temporary_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    archive.write(path, arcname=f'samples/{path.name}')
            os.replace(temporary_path, archive_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return archive_path, f'{row["name"]}-samples.zip'

    def state(self):
        return {
            'models': self.model_options(),
            'trainingPresets': [
                {'key': key, 'label': label} for key, label in TRAINING_PRESETS.items()
            ],
            'qtypes': list(QTYPE_OPTIONS),
            'optimizers': sorted(OPTIMIZER_OPTIONS),
            'samplers': sorted(SAMPLER_OPTIONS),
            'resolutions': sorted(RESOLUTION_OPTIONS),
            'datasets': self.list_datasets(),
            'jobs': self.list_jobs(),
            'gpus': self.detect_gpus(),
            'trainerInstalled': self.venv_python.is_file() and (self.vendor_root / 'run.py').is_file(),
            'hfTokenConfigured': bool(self.get_setting('HF_TOKEN') or os.environ.get('HF_TOKEN')),
            'upstreamCommit': '8a912564ce60047ea44d0f3a98becf3f168d3094',
        }


def create_trainer_blueprint(service: TrainerService):
    blueprint = Blueprint('trainer_api', __name__)

    def handle_error(exc):
        if isinstance(exc, FileNotFoundError):
            return jsonify({'error': str(exc)}), 404
        if isinstance(exc, TrainerValidationError):
            return jsonify({'error': str(exc)}), 400
        print(f'[Trainer] API error: {exc}')
        return jsonify({'error': str(exc)}), 500

    @blueprint.get('/api/trainer/state')
    def trainer_state():
        try:
            return jsonify(service.state())
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/datasets')
    def list_trainer_datasets():
        try:
            return jsonify({'datasets': service.list_dataset_names()})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/datasets/<name>')
    def inspect_trainer_dataset(name):
        try:
            return jsonify(service.inspect_dataset(name))
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/preflight')
    def trainer_preflight():
        try:
            payload = request.get_json() or {}
            inspections = [service.inspect_dataset(name) for name in payload.get('datasets', [])]
            preset = payload.get('trainingPreset', 'standard_lora')
            validity_key = {
                'standard_lora': 'valid',
                'transparent_lora': 'transparentValid',
                'qwen_rgba_vae': 'vaeValid',
                'flux2_rgba_vae': 'vaeValid',
                CHROMAKEY_PRESET: 'chromakeyValid',
            }.get(preset, 'valid')
            return jsonify({
                'datasets': inspections,
                'valid': bool(inspections) and all(item[validity_key] for item in inspections),
            })
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/config/preview')
    def trainer_config_preview():
        try:
            _name, _gpu_ids, config, inspections = service.build_job_config(request.get_json() or {})
            return jsonify({'process': config['config']['process'][0], 'datasets': inspections})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/sample-images')
    def upload_trainer_sample_image():
        try:
            upload = request.files.get('file')
            if upload is None:
                uploads = request.files.getlist('files')
                upload = uploads[0] if uploads else None
            if upload is None:
                raise TrainerValidationError('Choose an image to upload')
            path = service.save_sample_image(upload)
            return jsonify({
                'path': str(path),
                'previewUrl': f'/api/trainer/sample-images/{path.name}',
            }), 201
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/sample-images/<filename>')
    def preview_trainer_sample_image(filename):
        try:
            return send_file(service.sample_image_preview(filename), conditional=True)
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/jobs')
    def create_trainer_job():
        try:
            job, inspections = service.create_job(request.get_json() or {})
            return jsonify({'job': job, 'datasets': inspections}), 201
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs')
    def list_trainer_jobs():
        try:
            return jsonify({'jobs': service.list_jobs()})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.put('/api/trainer/jobs/<job_id>')
    def update_trainer_job(job_id):
        try:
            job, inspections = service.update_job(job_id, request.get_json() or {})
            return jsonify({'job': job, 'datasets': inspections})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/validation-images')
    def upload_trainer_validation_image():
        try:
            upload = request.files.get('file')
            if upload is None:
                uploads = request.files.getlist('files')
                upload = uploads[0] if uploads else None
            if upload is None:
                raise TrainerValidationError('Choose an RGBA validation target')
            path = service.save_validation_image(upload)
            return jsonify({
                'path': str(path),
                'previewUrl': f'/api/trainer/validation-images/{path.name}',
            }), 201
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/validation-images/<filename>')
    def preview_trainer_validation_image(filename):
        try:
            return send_file(service.validation_image_preview(filename), conditional=True)
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/chromakey-validation-images')
    def upload_trainer_chromakey_validation_image():
        try:
            upload = request.files.get('file')
            if upload is None:
                uploads = request.files.getlist('files')
                upload = uploads[0] if uploads else None
            if upload is None:
                raise TrainerValidationError('Choose a ChromaKey validation image')
            path = service.save_chromakey_validation_image(upload)
            return jsonify({
                'path': str(path),
                'previewUrl': f'/api/trainer/chromakey-validation-images/{path.name}',
            }), 201
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/chromakey-validation-images/<filename>')
    def preview_trainer_chromakey_validation_image(filename):
        try:
            return send_file(
                service.chromakey_validation_image_preview(filename), conditional=True
            )
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/jobs/<job_id>/clone')
    def clone_trainer_job(job_id):
        try:
            return jsonify({'job': service.clone_job(job_id)}), 201
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/jobs/<job_id>/start')
    def start_trainer_job(job_id):
        try:
            return jsonify({'job': service.queue_job(job_id)})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/jobs/<job_id>/stop')
    def stop_trainer_job(job_id):
        try:
            return jsonify({'job': service.stop_job(job_id)})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/jobs/<job_id>/save-now')
    def save_trainer_job_now(job_id):
        try:
            return jsonify({'job': service.request_runtime_action(job_id, 'save')})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/jobs/<job_id>/sample-now')
    def sample_trainer_job_now(job_id):
        try:
            return jsonify({'job': service.request_runtime_action(job_id, 'sample')})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/model')
    def download_trainer_job_model(job_id):
        try:
            path = service.job_model_artifact(job_id)
            return send_file(
                path, as_attachment=True, download_name=path.name, conditional=True
            )
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/samples')
    def list_trainer_job_samples(job_id):
        try:
            return jsonify(service.list_job_samples(job_id))
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/samples/archive')
    def download_trainer_job_samples(job_id):
        try:
            path, download_name = service.build_samples_archive(job_id)
            return send_file(path, as_attachment=True, download_name=download_name, conditional=True)
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/samples/<filename>')
    def serve_trainer_job_sample(job_id, filename):
        try:
            path = service.resolve_job_sample(job_id, filename, thumbnail='thumb' in request.args)
            return send_file(
                path,
                conditional=True,
                as_attachment='download' in request.args,
                download_name=filename,
                max_age=0,
            )
        except Exception as exc:
            return handle_error(exc)

    @blueprint.delete('/api/trainer/jobs/<job_id>/samples/<filename>')
    def delete_trainer_job_sample(job_id, filename):
        try:
            service.delete_job_sample(job_id, filename)
            return jsonify({'success': True})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/sample-controls/<int:sample_index>/<int:control_index>')
    def serve_trainer_job_sample_control(job_id, sample_index, control_index):
        try:
            path = service.resolve_job_sample_control(job_id, sample_index, control_index)
            return send_file(path, conditional=True, max_age=0)
        except Exception as exc:
            return handle_error(exc)

    @blueprint.delete('/api/trainer/jobs/<job_id>')
    def delete_trainer_job(job_id):
        try:
            service.delete_job(job_id)
            return jsonify({'success': True})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/log')
    def trainer_job_log(job_id):
        try:
            return jsonify({'log': service.read_log(job_id)})
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/validation-results')
    def trainer_job_validation_results(job_id):
        try:
            return jsonify(service.validation_results(job_id))
        except Exception as exc:
            return handle_error(exc)

    @blueprint.get('/api/trainer/jobs/<job_id>/log/download')
    def download_trainer_job_log(job_id):
        try:
            path = service.job_log_path(job_id)
            if not path.is_file():
                raise FileNotFoundError('Training log not found')
            return send_file(path, as_attachment=True, download_name=f'{job_id}-trainer.log')
        except Exception as exc:
            return handle_error(exc)

    @blueprint.post('/api/trainer/settings')
    def trainer_settings():
        try:
            service.save_settings(request.get_json() or {})
            return jsonify({'success': True, 'hfTokenConfigured': bool(service.get_setting('HF_TOKEN'))})
        except Exception as exc:
            return handle_error(exc)

    return blueprint
