from flask import Flask, send_from_directory, jsonify, request, send_file
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask_cors import CORS
import os
import re
import json
import random
import shutil
import uuid
import threading
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Base directory for datasets
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / 'Datasets'
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
DATASET_IMAGE_FOLDERS = ('img', 'Control1', 'Control2', 'Control3')
IMPORT_JOBS = {}
IMPORT_JOBS_LOCK = threading.Lock()
TOOL_JOBS = {}
TOOL_JOBS_LOCK = threading.Lock()

# Ensure Datasets directory exists
DATASETS_DIR.mkdir(exist_ok=True)


def iter_dataset_images(dataset_dir, folders=DATASET_IMAGE_FOLDERS):
    for folder_name in folders:
        folder = dataset_dir / folder_name
        if not folder.exists():
            continue
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield file_path


def apply_dataset_blur(image, strength, image_format=None):
    from PIL import ImageFilter

    output = image.copy()
    if strength > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=strength))

    normalized_format = (image_format or '').upper()
    if normalized_format == 'JPG':
        normalized_format = 'JPEG'

    if normalized_format == 'JPEG' and output.mode in ('RGBA', 'LA'):
        output = output.convert('RGB')

    return output, normalized_format


def apply_dataset_mirror(image, horizontal=False, vertical=False, image_format=None):
    from PIL import ImageOps

    output = image.copy()
    if horizontal:
        output = ImageOps.mirror(output)
    if vertical:
        output = ImageOps.flip(output)

    normalized_format = (image_format or '').upper()
    if normalized_format == 'JPG':
        normalized_format = 'JPEG'

    if normalized_format == 'JPEG' and output.mode in ('RGBA', 'LA'):
        output = output.convert('RGB')

    return output, normalized_format


def generate_unique_dataset_basename(target_dataset_dir):
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    target_img_dir = target_dataset_dir / 'img'

    while True:
        name = ''.join(random.choices(chars, k=8))
        is_unique = True
        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
            if (target_img_dir / f"{name}{ext}").exists():
                is_unique = False
                break

        if is_unique:
            return name


def scan_exported_dataset_groups(scan_dir):
    suffix_map = {
        '_img': 'img',
        '_ctr1': 'Control1',
        '_ctr2': 'Control2',
        '_ctr3': 'Control3'
    }
    groups = {}

    for item in scan_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue

        matched_suffix = None
        dataset_key = None
        dataset_folder = None
        for suffix, folder_name in suffix_map.items():
            if item.name.lower().endswith(suffix):
                matched_suffix = suffix
                dataset_key = item.name[:-len(suffix)]
                dataset_folder = folder_name
                break

        if not matched_suffix or not dataset_key:
            continue

        group = groups.setdefault(dataset_key, {
            'sourceName': dataset_key,
            'folders': {},
            'imageCount': 0,
            'controlCount': 0,
            'pairStyle': 'pair'
        })

        image_count = 0
        for file_path in item.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1

        group['folders'][dataset_folder] = {
            'path': str(item),
            'name': item.name,
            'imageCount': image_count
        }

    results = []
    for group in groups.values():
        if 'img' not in group['folders']:
            continue

        control_count = sum(1 for name in ('Control1', 'Control2', 'Control3') if name in group['folders'])
        if control_count < 2:
            continue

        group['controlCount'] = control_count
        group['imageCount'] = group['folders']['img']['imageCount']
        group['pairStyle'] = 'triplet' if control_count >= 3 else 'pair'
        results.append(group)

    results.sort(key=lambda item: item['sourceName'].lower())
    return results


def build_import_plan(dataset, target_dir):
    folder_mapping = {
        'img': 'img',
        'Control1': 'Control1',
        'Control2': 'Control2',
        'Control3': 'Control3'
    }

    folder_plans = []
    total_files = 0
    for source_folder_name, target_folder_name in folder_mapping.items():
        source_info = dataset['folders'].get(source_folder_name)
        if not source_info:
            continue

        source_dir = Path(source_info['path'])
        destination_dir = target_dir / target_folder_name
        directories = []
        files = []

        for path in source_dir.rglob('*'):
            relative_path = path.relative_to(source_dir)
            destination_path = destination_dir / relative_path
            if path.is_dir():
                directories.append(destination_path)
            else:
                files.append((path, destination_path))

        folder_plans.append({
            'sourceFolder': source_folder_name,
            'targetFolder': target_folder_name,
            'sourceDir': source_dir,
            'destinationDir': destination_dir,
            'directories': directories,
            'files': files
        })
        total_files += len(files)

    return folder_plans, total_files


def update_import_job(job_id, **changes):
    with IMPORT_JOBS_LOCK:
        job = IMPORT_JOBS.get(job_id)
        if not job:
            return None
        job.update(changes)
        return dict(job)


def increment_import_job_progress(job_id, folder_name, copied_delta=1):
    with IMPORT_JOBS_LOCK:
        job = IMPORT_JOBS.get(job_id)
        if not job:
            return

        job['copiedFiles'] += copied_delta
        job['currentFolder'] = folder_name
        if folder_name in job['folderProgress']:
            job['folderProgress'][folder_name]['copied'] += copied_delta

        total = job['totalFiles']
        job['progressPercent'] = round((job['copiedFiles'] / total) * 100, 1) if total else 100.0


def run_import_job(job_id, folder_plans, target_dir):
    try:
        for folder_name in DATASET_IMAGE_FOLDERS:
            (target_dir / folder_name).mkdir(parents=True, exist_ok=True)

        update_import_job(job_id, status='running', started=True)

        def copy_single_file(source_path, destination_path, folder_name):
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source_path, destination_path)
            except OSError:
                shutil.copy(source_path, destination_path)
            increment_import_job_progress(job_id, folder_name)

        def copy_folder(folder_plan):
            destination_dir = folder_plan['destinationDir']
            destination_dir.mkdir(parents=True, exist_ok=True)
            for directory in folder_plan['directories']:
                directory.mkdir(parents=True, exist_ok=True)

            files = folder_plan['files']
            if not files:
                return

            max_workers = min(8, max(1, len(files)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(copy_single_file, source_path, destination_path, folder_plan['targetFolder'])
                    for source_path, destination_path in files
                ]
                for future in futures:
                    future.result()

        folder_workers = min(len(folder_plans), 4) or 1
        with ThreadPoolExecutor(max_workers=folder_workers) as executor:
            futures = [executor.submit(copy_folder, folder_plan) for folder_plan in folder_plans]
            for future in futures:
                future.result()

        update_import_job(
            job_id,
            status='completed',
            currentFolder=None,
            progressPercent=100.0,
            finished=True
        )
    except Exception as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        update_import_job(
            job_id,
            status='error',
            currentFolder=None,
            error=str(e),
            finished=True
        )


def update_tool_job(job_id, **changes):
    with TOOL_JOBS_LOCK:
        job = TOOL_JOBS.get(job_id)
        if not job:
            return None
        job.update(changes)
        return dict(job)


def set_tool_job_total(job_id, total_items):
    with TOOL_JOBS_LOCK:
        job = TOOL_JOBS.get(job_id)
        if not job:
            return
        job['totalItems'] = total_items
        job['progressPercent'] = round((job['processedItems'] / total_items) * 100, 1) if total_items else 100.0


def increment_tool_job_progress(job_id, processed_delta=1, current_item=None, metrics=None):
    with TOOL_JOBS_LOCK:
        job = TOOL_JOBS.get(job_id)
        if not job:
            return

        job['processedItems'] += processed_delta
        if current_item is not None:
            job['currentItem'] = current_item

        if metrics:
            for key, value in metrics.items():
                job['metrics'][key] = value

        total = job['totalItems']
        job['progressPercent'] = round((job['processedItems'] / total) * 100, 1) if total else 100.0


def _collect_dataset_file_structure(dataset_dir):
    img_dir = dataset_dir / 'img'
    if not img_dir.exists():
        raise FileNotFoundError('Image directory not found')

    folders_to_process = list(DATASET_IMAGE_FOLDERS)
    primary_basenames = set()
    for file in img_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            primary_basenames.add(file.stem)

    file_structure = {}
    for basename in primary_basenames:
        file_structure[basename] = {}
        for folder_name in folders_to_process:
            folder = dataset_dir / folder_name
            if not folder.exists():
                continue

            extensions = []
            for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
                file_path = folder / f"{basename}{ext}"
                if file_path.exists():
                    extensions.append(ext)

            if extensions:
                file_structure[basename][folder_name] = extensions

    return file_structure


def process_reshuffle_job(job_id, folder_path):
    dataset_dir = DATASETS_DIR / folder_path
    if not dataset_dir.exists():
        raise FileNotFoundError('Dataset not found')

    file_structure = _collect_dataset_file_structure(dataset_dir)
    basenames_list = list(file_structure.keys())
    if not basenames_list:
        raise FileNotFoundError('No images found')

    set_tool_job_total(job_id, len(basenames_list))
    random.shuffle(basenames_list)
    used_names = set()
    rename_count = 0

    def generate_unique_name():
        while True:
            name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            if name not in used_names:
                used_names.add(name)
                return name

    for basename in basenames_list:
        new_basename = generate_unique_name()
        for folder_name, extensions in file_structure[basename].items():
            folder = dataset_dir / folder_name
            for ext in extensions:
                old_path = folder / f"{basename}{ext}"
                new_path = folder / f"{new_basename}{ext}"
                if old_path.exists():
                    old_path.rename(new_path)
                    rename_count += 1

        increment_tool_job_progress(job_id, current_item=basename, metrics={'filesRenamed': rename_count})

    return {
        'count': len(basenames_list),
        'filesRenamed': rename_count,
        'folder': folder_path
    }


def process_compress_job(job_id, folder_path):
    from PIL import Image

    dataset_dir = DATASETS_DIR / folder_path
    folders_to_process = list(DATASET_IMAGE_FOLDERS)
    files_to_compress = []
    for folder_name in folders_to_process:
        folder = dataset_dir / folder_name
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.png':
                files_to_compress.append(file_path)

    set_tool_job_total(job_id, len(files_to_compress))
    if not files_to_compress:
        return {
            'compressed': 0,
            'originalSizeMB': 0,
            'newSizeMB': 0,
            'savingsMB': 0,
            'savingsPercent': 0
        }

    def compress_single_image(file_path):
        orig_size = file_path.stat().st_size
        with Image.open(file_path) as img:
            img.save(file_path, 'PNG', optimize=True, compress_level=9)
        new_size = file_path.stat().st_size
        return file_path.name, orig_size, new_size

    max_workers = min(16, (os.cpu_count() or 4) * 2)
    compressed_count = 0
    original_total_size = 0
    new_total_size = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compress_single_image, file_path) for file_path in files_to_compress]
        for future in as_completed(futures):
            file_name, o_size, n_size = future.result()
            compressed_count += 1
            original_total_size += o_size
            new_total_size += n_size
            increment_tool_job_progress(job_id, current_item=file_name, metrics={
                'compressed': compressed_count,
                'originalTotalSize': original_total_size,
                'newTotalSize': new_total_size
            })

    savings_mb = (original_total_size - new_total_size) / (1024 * 1024)
    savings_percent = ((original_total_size - new_total_size) / original_total_size * 100) if original_total_size > 0 else 0
    return {
        'compressed': compressed_count,
        'originalSizeMB': round(original_total_size / (1024 * 1024), 2),
        'newSizeMB': round(new_total_size / (1024 * 1024), 2),
        'savingsMB': round(savings_mb, 2),
        'savingsPercent': round(savings_percent, 1)
    }


def process_blur_job(job_id, folder_path, strength):
    from PIL import Image

    source_dir = DATASETS_DIR / folder_path
    if not source_dir.exists():
        raise FileNotFoundError('Dataset not found')

    source_images = list(iter_dataset_images(source_dir))
    if not source_images:
        raise FileNotFoundError('No images found in dataset')

    target_dir = source_dir.parent / f"{source_dir.name}-blured"
    if target_dir.exists():
        raise FileExistsError(f'Dataset "{target_dir.name}" already exists')

    try:
        shutil.copytree(source_dir, target_dir)
        target_images = list(iter_dataset_images(target_dir))
        set_tool_job_total(job_id, len(target_images))

        processed_count = 0
        for file_path in target_images:
            with Image.open(file_path) as img:
                output, image_format = apply_dataset_blur(img, strength, img.format or file_path.suffix.lstrip('.'))

            save_kwargs = {}
            if image_format == 'PNG':
                save_kwargs['optimize'] = True

            output.save(file_path, image_format, **save_kwargs)
            processed_count += 1
            increment_tool_job_progress(job_id, current_item=file_path.name)

        return {
            'processed': processed_count,
            'strength': strength,
            'targetFolder': str(target_dir.relative_to(DATASETS_DIR)),
            'targetName': target_dir.name
        }
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise


def process_mirror_job(job_id, folder_path, horizontal, vertical, excluded_controls):
    from PIL import Image

    source_dir = DATASETS_DIR / folder_path
    if not source_dir.exists():
        raise FileNotFoundError('Dataset not found')

    source_images = list(iter_dataset_images(source_dir))
    if not source_images:
        raise FileNotFoundError('No images found in dataset')

    target_dir = source_dir.parent / f"{source_dir.name}-mirror"
    if target_dir.exists():
        raise FileExistsError(f'Dataset "{target_dir.name}" already exists')

    try:
        shutil.copytree(source_dir, target_dir)
        target_images = [
            file_path for file_path in iter_dataset_images(target_dir)
            if file_path.parent.name not in excluded_controls
        ]
        set_tool_job_total(job_id, len(target_images))

        processed_count = 0
        for file_path in target_images:
            with Image.open(file_path) as img:
                output, image_format = apply_dataset_mirror(
                    img,
                    horizontal=horizontal,
                    vertical=vertical,
                    image_format=img.format or file_path.suffix.lstrip('.')
                )

            save_kwargs = {}
            if image_format == 'PNG':
                save_kwargs['optimize'] = True

            output.save(file_path, image_format, **save_kwargs)
            processed_count += 1
            increment_tool_job_progress(job_id, current_item=file_path.name)

        return {
            'processed': processed_count,
            'horizontal': horizontal,
            'vertical': vertical,
            'excludedControls': sorted(excluded_controls),
            'targetFolder': str(target_dir.relative_to(DATASETS_DIR)),
            'targetName': target_dir.name
        }
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise


def process_merge_job(job_id, primary_folder, secondary_folder, target_name):
    primary_dir = DATASETS_DIR / primary_folder
    secondary_dir = DATASETS_DIR / secondary_folder
    target_dir = DATASETS_DIR / target_name

    for source_dir, label in ((primary_dir, 'Primary'), (secondary_dir, 'Secondary')):
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f'{label} dataset not found')
        if not (source_dir / 'img').exists():
            raise ValueError(f'{label} dataset is missing its img folder')

    if target_dir.exists():
        raise FileExistsError(f'Dataset "{target_name}" already exists')

    primary_images = [file_path for file_path in sorted((primary_dir / 'img').iterdir()) if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS]
    secondary_images = [file_path for file_path in sorted((secondary_dir / 'img').iterdir()) if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS]
    total_sets = len(primary_images) + len(secondary_images)
    if total_sets == 0:
        raise ValueError('No image sets found to merge')

    try:
        for folder_name in DATASET_IMAGE_FOLDERS:
            (target_dir / folder_name).mkdir(parents=True, exist_ok=True)

        set_tool_job_total(job_id, total_sets)
        counts = {'primary': 0, 'secondary': 0}

        def copy_dataset_sets(source_dir, bucket_name, image_paths):
            copied_sets = 0
            for image_path in image_paths:
                source_basename = image_path.stem
                target_basename = generate_unique_dataset_basename(target_dir)

                for folder_name in DATASET_IMAGE_FOLDERS:
                    source_folder = source_dir / folder_name
                    target_folder = target_dir / folder_name

                    if folder_name == 'img':
                        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                            src_file = source_folder / f"{source_basename}{ext}"
                            if src_file.exists():
                                shutil.copy2(src_file, target_folder / f"{target_basename}{src_file.suffix}")
                                break

                        caption_path = source_folder / f"{source_basename}.txt"
                        if caption_path.exists():
                            shutil.copy2(caption_path, target_folder / f"{target_basename}.txt")
                        continue

                    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                        src_file = source_folder / f"{source_basename}{ext}"
                        if src_file.exists():
                            shutil.copy2(src_file, target_folder / f"{target_basename}{src_file.suffix}")
                            break

                copied_sets += 1
                counts[bucket_name] = copied_sets
                increment_tool_job_progress(job_id, current_item=image_path.name, metrics={
                    'primaryCount': counts['primary'],
                    'secondaryCount': counts['secondary']
                })

            return copied_sets

        primary_count = copy_dataset_sets(primary_dir, 'primary', primary_images)
        secondary_count = copy_dataset_sets(secondary_dir, 'secondary', secondary_images)

        return {
            'primaryFolder': primary_folder,
            'secondaryFolder': secondary_folder,
            'targetFolder': str(target_dir.relative_to(DATASETS_DIR)),
            'targetName': target_dir.name,
            'primaryCount': primary_count,
            'secondaryCount': secondary_count,
            'totalCount': total_sets
        }
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise


def process_fit_job(job_id, folder_path):
    from PIL import Image, ImageOps

    dataset_dir = DATASETS_DIR / folder_path
    if not dataset_dir.exists():
        raise FileNotFoundError('Dataset not found')

    img_dir = dataset_dir / 'img'
    control_folders = ['Control1', 'Control2', 'Control3']
    if not img_dir.exists():
        raise FileNotFoundError('Primary image directory (img) not found')

    primary_images = {
        f.stem: f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not primary_images:
        raise FileNotFoundError('No primary images found in img folder')

    set_tool_job_total(job_id, len(primary_images))
    processed_count = 0
    updated_count = 0

    for basename, primary_path in primary_images.items():
        processed_count += 1
        with Image.open(primary_path) as p_img:
            target_size = p_img.size

        for ctrl_folder in control_folders:
            ctrl_dir = dataset_dir / ctrl_folder
            if not ctrl_dir.exists():
                continue

            ctrl_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                temp_path = ctrl_dir / f"{basename}{ext}"
                if temp_path.exists():
                    ctrl_path = temp_path
                    break

            if not ctrl_path:
                continue

            with Image.open(ctrl_path) as c_img:
                if c_img.size == target_size:
                    continue

                new_img = ImageOps.pad(c_img, target_size, color=(0, 0, 0), centering=(0.5, 0.5))
                new_img.save(ctrl_path)
                updated_count += 1

        increment_tool_job_progress(job_id, current_item=primary_path.name, metrics={'updated': updated_count})

    return {
        'processed': processed_count,
        'updated': updated_count
    }


def run_tool_job(job_id, tool_name, payload):
    try:
        update_tool_job(job_id, status='running', started=True)

        if tool_name == 'reshuffle':
            result = process_reshuffle_job(job_id, payload['folderPath'])
        elif tool_name == 'compress':
            result = process_compress_job(job_id, payload['folderPath'])
        elif tool_name == 'blur':
            result = process_blur_job(job_id, payload['folderPath'], float(payload['strength']))
        elif tool_name == 'mirror':
            result = process_mirror_job(
                job_id,
                payload['folderPath'],
                bool(payload['horizontal']),
                bool(payload['vertical']),
                set(payload.get('excludedControls') or [])
            )
        elif tool_name == 'merge':
            result = process_merge_job(job_id, payload['primaryFolder'], payload['secondaryFolder'], payload['targetName'])
        elif tool_name == 'fit':
            result = process_fit_job(job_id, payload['folderPath'])
        else:
            raise ValueError('Unsupported tool job')

        update_tool_job(
            job_id,
            status='completed',
            currentItem=None,
            progressPercent=100.0,
            finished=True,
            result=result
        )
    except Exception as e:
        update_tool_job(
            job_id,
            status='error',
            currentItem=None,
            error=str(e),
            finished=True
        )

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/save/<filename>', methods=['POST'])
def save_image(filename):
    """Save an edited image to the dataset"""
    try:
        folder = request.args.get('folder')
        subfolder = request.args.get('subfolder', 'img')
        if not folder:
            return jsonify({'error': 'Folder parameter is required'}), 400
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Construct path
        # We save to the specified subfolder (default 'img')
        save_path = DATASETS_DIR / folder / subfolder / filename
        
        if not save_path.parent.exists():
            return jsonify({'error': 'Dataset folder not found'}), 404
            
        file.save(save_path)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/folders')
def get_folders():
    """Get list of available dataset folders"""
    try:
        folders = []
        for item in DATASETS_DIR.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has img, Control1, Control2 subdirectories
                img_dir = item / 'img'
                control1_dir = item / 'Control1'
                control2_dir = item / 'Control2'
                
                if img_dir.exists() and control1_dir.exists() and control2_dir.exists():
                    folders.append({
                        'name': item.name,
                        'path': str(item.relative_to(DATASETS_DIR))
                    })
        
        # Sort folders alphabetically
        folders.sort(key=lambda x: x['name'].lower())
        
        return jsonify({'folders': folders})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-dataset', methods=['POST'])
def create_dataset():
    """Create a new empty dataset with proper folder structure"""
    try:
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'Dataset name is required'}), 400
        
        # Validate name - only allow safe characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return jsonify({'error': 'Name can only contain letters, numbers, underscores and hyphens'}), 400
        
        dataset_dir = DATASETS_DIR / name
        
        if dataset_dir.exists():
            return jsonify({'error': f'Dataset "{name}" already exists'}), 400
        
        # Create dataset folder and subfolders
        (dataset_dir / 'img').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'Control1').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'Control2').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'Control3').mkdir(parents=True, exist_ok=True)
        
        return jsonify({
            'success': True,
            'name': name,
            'path': str(dataset_dir.relative_to(BASE_DIR))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rename-dataset', methods=['POST'])
def rename_dataset():
    """Rename a dataset by renaming its root folder."""
    try:
        data = request.get_json() or {}
        old_name = data.get('oldName', '').strip()
        new_name = data.get('newName', '').strip()

        if not old_name:
            return jsonify({'error': 'Current dataset name is required'}), 400

        if not new_name:
            return jsonify({'error': 'New dataset name is required'}), 400

        if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
            return jsonify({'error': 'Name can only contain letters, numbers, underscores and hyphens'}), 400

        source_dir = DATASETS_DIR / old_name
        target_dir = DATASETS_DIR / new_name

        if not source_dir.exists() or not source_dir.is_dir():
            return jsonify({'error': 'Dataset not found'}), 404

        if source_dir == target_dir:
            return jsonify({'success': True, 'oldName': old_name, 'newName': new_name, 'path': new_name})

        if target_dir.exists():
            return jsonify({'error': f'Dataset "{new_name}" already exists'}), 400

        source_dir.rename(target_dir)

        return jsonify({
            'success': True,
            'oldName': old_name,
            'newName': new_name,
            'path': str(target_dir.relative_to(DATASETS_DIR))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-datasets', methods=['POST'])
def compare_datasets():
    """Compare two datasets and find orphan files in linked dataset"""
    try:
        data = request.get_json() or {}
        primary_folder = data.get('primaryFolder', '')
        linked_folder = data.get('linkedFolder', '')
        
        if not primary_folder or not linked_folder:
            return jsonify({'error': 'Primary and linked folders are required'}), 400
        
        primary_dir = DATASETS_DIR / primary_folder / 'img'
        linked_dir = DATASETS_DIR / linked_folder / 'img'
        
        if not primary_dir.exists():
            return jsonify({'error': 'Primary dataset not found'}), 404
        if not linked_dir.exists():
            return jsonify({'error': 'Linked dataset not found'}), 404
        
        # Get basenames from both datasets
        primary_basenames = set()
        for f in primary_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                primary_basenames.add(f.stem)
        
        linked_basenames = set()
        for f in linked_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                linked_basenames.add(f.stem)
        
        # Find orphans (in linked but not in primary)
        orphan_basenames = linked_basenames - primary_basenames
        
        # Get full filenames for orphans
        orphans = []
        for basename in orphan_basenames:
            for f in linked_dir.iterdir():
                if f.stem == basename and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    orphans.append(f.name)
                    break
        
        return jsonify({
            'orphans': orphans,
            'primaryCount': len(primary_basenames),
            'linkedCount': len(linked_basenames),
            'orphanCount': len(orphans)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images')
def get_images():
    """Get list of images from the img folder"""
    folder_path = request.args.get('folder', '')
    
    try:
        dataset_dir = DATASETS_DIR / folder_path
        img_dir = dataset_dir / 'img'
        
        if not img_dir.exists():
            return jsonify({'error': 'Image directory not found'}), 404
        
        images = []
        for file in sorted(img_dir.iterdir()):
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                images.append(file.name)
        
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<image_type>/<filename>')
def get_image(image_type, filename):
    """Serve individual image from specified folder type"""
    folder_path = request.args.get('folder', '')
    
    try:
        dataset_dir = DATASETS_DIR / folder_path
        
        # Validate image_type
        if image_type not in ['img', 'Control1', 'Control2', 'Control3']:
            return jsonify({'error': 'Invalid image type'}), 400
        
        image_dir = dataset_dir / image_type
        image_path = image_dir / filename
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/caption/<filename>')
def get_caption(filename):
    """Get caption text for an image"""
    folder_path = request.args.get('folder', '')
    try:
        dataset_dir = DATASETS_DIR / folder_path
        basename = os.path.splitext(filename)[0]
        txt_path = dataset_dir / 'img' / f"{basename}.txt"
        
        if not txt_path.exists():
            return jsonify({'caption': ''})
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read()
            
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/caption/<filename>', methods=['POST'])
def save_caption(filename):
    """Save caption text for an image"""
    folder_path = request.args.get('folder', '')
    try:
        data = request.get_json() or {}
        caption = data.get('caption', '')
        
        dataset_dir = DATASETS_DIR / folder_path
        basename = os.path.splitext(filename)[0]
        txt_path = dataset_dir / 'img' / f"{basename}.txt"
        
        # Ensure img directory exists
        (dataset_dir / 'img').mkdir(parents=True, exist_ok=True)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete all related files (img, Control1-3, txt) with the same filename, optionally from linked dataset too"""
    folder_path = request.args.get('folder', '')
    linked_folder = request.args.get('linkedFolder', '')
    
    try:
        deleted_files = []
        errors = []
        
        # Helper function to delete from a single dataset
        def delete_from_dataset(dataset_path, prefix=''):
            dataset_dir = DATASETS_DIR / dataset_path
            folders_to_check = ['img', 'Control1', 'Control2', 'Control3']
            basename = os.path.splitext(filename)[0]
            txt_filename = f"{basename}.txt"
            
            # Delete txt file from img folder
            txt_path = dataset_dir / 'img' / txt_filename
            if txt_path.exists():
                try:
                    txt_path.unlink()
                    deleted_files.append(f"{prefix}img/{txt_filename}")
                except Exception as e:
                    errors.append(f"Failed to delete {prefix}img/{txt_filename}: {str(e)}")
            
            # Delete image files
            for folder_name in folders_to_check:
                folder = dataset_dir / folder_name
                file_path = folder / filename
                
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_files.append(f"{prefix}{folder_name}/{filename}")
                    except Exception as e:
                        errors.append(f"Failed to delete {prefix}{folder_name}/{filename}: {str(e)}")
        
        # Delete from primary dataset
        delete_from_dataset(folder_path)
        
        # Delete from linked dataset if provided
        if linked_folder:
            delete_from_dataset(linked_folder, f"[linked:{linked_folder}] ")
        
        if deleted_files:
            return jsonify({
                'success': True,
                'deleted': deleted_files,
                'errors': errors if errors else None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No files found to delete',
                'errors': errors
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transfer/<filename>', methods=['POST'])
def transfer_image(filename):
    """Transfer all related files (img, Control1-3, .txt) to another dataset folder, optionally from linked dataset too"""
    source_folder = request.args.get('folder', '')
    data = request.get_json() or {}
    target_folder = data.get('targetFolder', '')
    linked_folder = data.get('linkedFolder', '')
    
    if not source_folder or not target_folder:
        return jsonify({'error': 'Source and target folders are required'}), 400
    
    if source_folder == target_folder:
        return jsonify({'error': 'Source and target folders must be different'}), 400
    
    try:
        target_dir = DATASETS_DIR / target_folder
        
        # Verify target exists
        if not target_dir.exists():
            return jsonify({'error': 'Target directory not found'}), 404
        if not (target_dir / 'img').exists():
            return jsonify({'error': 'Target is not a valid dataset (no img folder)'}), 400
        
        # Get basename without extension
        basename = os.path.splitext(filename)[0]
        original_ext = os.path.splitext(filename)[1]
        
        # Generate unique 8-character name for target
        import string
        chars = string.ascii_lowercase + string.digits
        
        def generate_unique_name(target_dataset_dir):
            """Generate a unique name that doesn't exist in target dataset"""
            while True:
                import random
                name = ''.join(random.choices(chars, k=8))
                
                # Efficient check: if any file with this stem exists in 'img'
                # We check common extensions and the most likely .png
                is_unique = True
                target_img_dir = target_dataset_dir / 'img'
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
                    if (target_img_dir / f"{name}{ext}").exists():
                        is_unique = False
                        break
                
                if is_unique:
                    return name
        
        def transfer_from_dataset(src_folder, target_dataset_dir, src_basename):
            """Transfer files from source to target with new unique name"""
            source_dir = DATASETS_DIR / src_folder
            
            if not source_dir.exists():
                return [], f"Source directory {src_folder} not found"
            
            new_basename = generate_unique_name(target_dataset_dir)
            folders_to_process = ['img', 'Control1', 'Control2', 'Control3']
            
            files_to_transfer = []
            for folder_name in folders_to_process:
                source_subfolder = source_dir / folder_name
                target_subfolder = target_dataset_dir / folder_name
                
                if not source_subfolder.exists():
                    continue
                
                # Check for image files with this basename
                for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                    source_file = source_subfolder / f"{src_basename}{ext}"
                    if source_file.exists():
                        target_subfolder.mkdir(parents=True, exist_ok=True)
                        target_file = target_subfolder / f"{new_basename}{ext}"
                        files_to_transfer.append((source_file, target_file))
                
                # Check for txt caption file (only in img folder)
                if folder_name == 'img':
                    txt_source = source_subfolder / f"{src_basename}.txt"
                    if txt_source.exists():
                        txt_target = target_subfolder / f"{new_basename}.txt"
                        files_to_transfer.append((txt_source, txt_target))
            
            # Move all files
            transferred = []
            for source_file, target_file in files_to_transfer:
                shutil.move(str(source_file), str(target_file))
                transferred.append({
                    'from': str(source_file.relative_to(BASE_DIR)),
                    'to': str(target_file.relative_to(BASE_DIR))
                })
            
            return transferred, new_basename
        
        # Transfer from primary dataset
        primary_transferred, primary_new_name = transfer_from_dataset(source_folder, target_dir, basename)
        
        if not primary_transferred:
            return jsonify({'error': 'No files found to transfer'}), 404
        
        result = {
            'success': True,
            'newFilename': f"{primary_new_name}{original_ext}",
            'transferred': primary_transferred
        }
        
        # Transfer from linked dataset if provided
        if linked_folder:
            linked_transferred, linked_new_name = transfer_from_dataset(linked_folder, target_dir, basename)
            result['linkedTransferred'] = linked_transferred
            result['linkedNewFilename'] = f"{linked_new_name}{original_ext}" if linked_new_name else None
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reshuffle', methods=['POST'])
def reshuffle_dataset():
    """Reshuffle all images in the dataset with random 8-character names while keeping related files synchronized"""
    folder_path = request.args.get('folder', '')
    
    try:
        dataset_dir = DATASETS_DIR / folder_path
        img_dir = dataset_dir / 'img'
        
        if not img_dir.exists():
            return jsonify({'error': 'Image directory not found'}), 404
            
        # 1. Build a mapping of basenames to their file locations
        # Structure: {basename: {folder_name: [extensions]}}
        file_structure = {}
        folders_to_process = ['img', 'Control1', 'Control2', 'Control3']
        
        # Scan img folder to get primary basenames (only image files, not txt)
        primary_basenames = set()
        for file in img_dir.iterdir():
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                primary_basenames.add(file.stem)
        
        if not primary_basenames:
            return jsonify({'error': 'No images found'}), 404
        
        # For each basename, record which extensions exist in which folders
        for basename in primary_basenames:
            file_structure[basename] = {}
            
            for folder_name in folders_to_process:
                folder = dataset_dir / folder_name
                if not folder.exists():
                    continue
                
                extensions = []
                # Check for image and txt files with this basename
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
                    file_path = folder / f"{basename}{ext}"
                    if file_path.exists():
                        extensions.append(ext)
                
                if extensions:
                    file_structure[basename][folder_name] = extensions
        
        # 2. Create random permutation of basenames
        basenames_list = list(file_structure.keys())
        random.shuffle(basenames_list)
        
        # 3. Generate unique 8-character random names for each basename
        import string
        chars = string.ascii_lowercase + string.digits
        used_names = set()
        
        # Helper function to generate unique random name
        def generate_unique_name():
            while True:
                name = ''.join(random.choices(chars, k=8))
                if name not in used_names:
                    used_names.add(name)
                    return name
        
        # 4. Rename all files with random 8-character names
        rename_count = 0
        for basename in basenames_list:
            new_basename = generate_unique_name()
            
            # Rename all files in this set to use the new random basename
            for folder_name, extensions in file_structure[basename].items():
                folder = dataset_dir / folder_name
                for ext in extensions:
                    old_path = folder / f"{basename}{ext}"
                    new_path = folder / f"{new_basename}{ext}"
                    if old_path.exists():
                        old_path.rename(new_path)
                        rename_count += 1
        
        return jsonify({'success': True, 'count': len(basenames_list), 'files_renamed': rename_count})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compress', methods=['POST'])
def compress_dataset():
    """Compress all images in the dataset to 90% quality while keeping PNG format"""
    folder_path = request.args.get('folder', '')
    
    try:
        from PIL import Image
        dataset_dir = DATASETS_DIR / folder_path
        folders_to_process = ['img', 'Control1', 'Control2', 'Control3']
        
        # Collect all files to process
        files_to_compress = []
        for folder_name in folders_to_process:
            folder = dataset_dir / folder_name
            if not folder.exists():
                continue
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.png':
                    files_to_compress.append(file_path)

        if not files_to_compress:
             return jsonify({
                'success': True,
                'compressed': 0,
                'originalSizeMB': 0,
                'newSizeMB': 0,
                'savingsMB': 0,
                'savingsPercent': 0
            })

        def compress_single_image(file_path):
            try:
                orig_size = file_path.stat().st_size
                with Image.open(file_path) as img:
                    # PNG optimization is already lossless, compress_level 9 is best
                    img.save(file_path, 'PNG', optimize=True, compress_level=9)
                new_size = file_path.stat().st_size
                return True, orig_size, new_size
            except Exception as e:
                print(f"Failed to compress {file_path}: {e}")
                return False, 0, 0

        # Execute compression in parallel
        # Use a pool size that's reasonable (max 16 or cpu_count)
        import os
        max_workers = min(16, (os.cpu_count() or 4) * 2)
        
        compressed_count = 0
        original_total_size = 0
        new_total_size = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compress_single_image, files_to_compress))
            
        for success, o_size, n_size in results:
            if success:
                compressed_count += 1
                original_total_size += o_size
                new_total_size += n_size
        
        # Calculate savings
        savings_mb = (original_total_size - new_total_size) / (1024 * 1024)
        savings_percent = ((original_total_size - new_total_size) / original_total_size * 100) if original_total_size > 0 else 0
        
        return jsonify({
            'success': True,
            'compressed': compressed_count,
            'originalSizeMB': round(original_total_size / (1024 * 1024), 2),
            'newSizeMB': round(new_total_size / (1024 * 1024), 2),
            'savingsMB': round(savings_mb, 2),
            'savingsPercent': round(savings_percent, 1)
        })
        
    except ImportError:
        return jsonify({'error': 'Pillow library not installed. Run: pip install Pillow'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/blur-preview/<filename>')
def blur_dataset_preview(filename):
    """Render a blurred preview for a target image without saving it"""
    folder_path = request.args.get('folder', '')

    try:
        from io import BytesIO
        from PIL import Image

        strength = float(request.args.get('strength', 0))
        strength = max(0.0, min(strength, 50.0))

        if not folder_path:
            return jsonify({'error': 'Dataset folder is required'}), 400

        image_path = DATASETS_DIR / folder_path / 'img' / filename
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return jsonify({'error': 'Preview image not found'}), 404

        with Image.open(image_path) as img:
            preview, _ = apply_dataset_blur(img, strength, img.format or image_path.suffix.lstrip('.'))

        if preview.mode not in ('RGB', 'RGBA'):
            preview = preview.convert('RGBA' if 'A' in preview.getbands() else 'RGB')

        buffer = BytesIO()
        preview.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', download_name='blur-preview.png')

    except ValueError:
        return jsonify({'error': 'Blur strength must be a number'}), 400
    except ImportError:
        return jsonify({'error': 'Pillow library not installed. Run: pip install Pillow'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/blur', methods=['POST'])
def blur_dataset():
    """Create a blurred copy of a dataset with a -blured suffix"""
    folder_path = request.args.get('folder', '')
    data = request.get_json() or {}
    target_dir = None

    try:
        from PIL import Image

        if not folder_path:
            return jsonify({'error': 'Dataset folder is required'}), 400

        strength = float(data.get('strength', 0))
        if strength < 0:
            return jsonify({'error': 'Blur strength must be non-negative'}), 400

        source_dir = DATASETS_DIR / folder_path
        if not source_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404

        source_images = list(iter_dataset_images(source_dir))
        if not source_images:
            return jsonify({'error': 'No images found in dataset'}), 404

        target_dir = source_dir.parent / f"{source_dir.name}-blured"
        if target_dir.exists():
            return jsonify({'error': f'Dataset "{target_dir.name}" already exists'}), 400

        shutil.copytree(source_dir, target_dir)

        processed_count = 0
        for file_path in iter_dataset_images(target_dir):
            with Image.open(file_path) as img:
                output, image_format = apply_dataset_blur(img, strength, img.format or file_path.suffix.lstrip('.'))

            save_kwargs = {}
            if image_format == 'PNG':
                save_kwargs['optimize'] = True

            output.save(file_path, image_format, **save_kwargs)
            processed_count += 1

        return jsonify({
            'success': True,
            'processed': processed_count,
            'strength': strength,
            'targetFolder': str(target_dir.relative_to(DATASETS_DIR)),
            'targetName': target_dir.name
        })

    except ValueError:
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return jsonify({'error': 'Blur strength must be a number'}), 400
    except ImportError:
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return jsonify({'error': 'Pillow library not installed. Run: pip install Pillow'}), 500
    except Exception as e:
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/mirror', methods=['POST'])
def mirror_dataset():
    """Create a mirrored copy of a dataset with a -mirror suffix."""
    folder_path = request.args.get('folder', '')
    data = request.get_json() or {}
    target_dir = None

    try:
        from PIL import Image

        if not folder_path:
            return jsonify({'error': 'Dataset folder is required'}), 400

        horizontal = bool(data.get('horizontal'))
        vertical = bool(data.get('vertical'))
        excluded_controls = set(data.get('excludedControls') or [])
        if not horizontal and not vertical:
            return jsonify({'error': 'Select at least one mirror direction'}), 400

        invalid_controls = excluded_controls - {'Control1', 'Control2', 'Control3'}
        if invalid_controls:
            return jsonify({'error': 'Invalid excluded controls specified'}), 400

        source_dir = DATASETS_DIR / folder_path
        if not source_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404

        source_images = list(iter_dataset_images(source_dir))
        if not source_images:
            return jsonify({'error': 'No images found in dataset'}), 404

        target_dir = source_dir.parent / f"{source_dir.name}-mirror"
        if target_dir.exists():
            return jsonify({'error': f'Dataset "{target_dir.name}" already exists'}), 400

        shutil.copytree(source_dir, target_dir)

        processed_count = 0
        for file_path in iter_dataset_images(target_dir):
            folder_name = file_path.parent.name
            if folder_name in excluded_controls:
                continue

            with Image.open(file_path) as img:
                output, image_format = apply_dataset_mirror(
                    img,
                    horizontal=horizontal,
                    vertical=vertical,
                    image_format=img.format or file_path.suffix.lstrip('.')
                )

            save_kwargs = {}
            if image_format == 'PNG':
                save_kwargs['optimize'] = True

            output.save(file_path, image_format, **save_kwargs)
            processed_count += 1

        return jsonify({
            'success': True,
            'processed': processed_count,
            'horizontal': horizontal,
            'vertical': vertical,
            'excludedControls': sorted(excluded_controls),
            'targetFolder': str(target_dir.relative_to(DATASETS_DIR)),
            'targetName': target_dir.name
        })

    except ImportError:
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return jsonify({'error': 'Pillow library not installed. Run: pip install Pillow'}), 500
    except Exception as e:
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/merge', methods=['POST'])
def merge_datasets():
    """Merge two datasets into a new third dataset using fresh unique basenames."""
    data = request.get_json() or {}
    primary_folder = data.get('primaryFolder', '').strip()
    secondary_folder = data.get('secondaryFolder', '').strip()
    target_name = data.get('targetName', '').strip()
    target_dir = None

    try:
        if not primary_folder or not secondary_folder:
            return jsonify({'error': 'Both source datasets are required'}), 400

        if primary_folder == secondary_folder:
            return jsonify({'error': 'Choose two different datasets to merge'}), 400

        if not target_name:
            return jsonify({'error': 'Target dataset name is required'}), 400

        if not re.match(r'^[a-zA-Z0-9_-]+$', target_name):
            return jsonify({'error': 'Name can only contain letters, numbers, underscores and hyphens'}), 400

        primary_dir = DATASETS_DIR / primary_folder
        secondary_dir = DATASETS_DIR / secondary_folder
        target_dir = DATASETS_DIR / target_name

        for source_dir, label in ((primary_dir, 'Primary'), (secondary_dir, 'Secondary')):
            if not source_dir.exists() or not source_dir.is_dir():
                return jsonify({'error': f'{label} dataset not found'}), 404
            if not (source_dir / 'img').exists():
                return jsonify({'error': f'{label} dataset is missing its img folder'}), 400

        if target_dir.exists():
            return jsonify({'error': f'Dataset "{target_name}" already exists'}), 400

        for folder_name in DATASET_IMAGE_FOLDERS:
            (target_dir / folder_name).mkdir(parents=True, exist_ok=True)

        def copy_dataset_sets(source_dir):
            copied_sets = 0
            for image_path in sorted((source_dir / 'img').iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                source_basename = image_path.stem
                target_basename = generate_unique_dataset_basename(target_dir)

                for folder_name in DATASET_IMAGE_FOLDERS:
                    source_folder = source_dir / folder_name
                    target_folder = target_dir / folder_name

                    if folder_name == 'img':
                        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                            src_file = source_folder / f"{source_basename}{ext}"
                            if src_file.exists():
                                shutil.copy2(src_file, target_folder / f"{target_basename}{src_file.suffix}")
                                break

                        caption_path = source_folder / f"{source_basename}.txt"
                        if caption_path.exists():
                            shutil.copy2(caption_path, target_folder / f"{target_basename}.txt")
                        continue

                    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                        src_file = source_folder / f"{source_basename}{ext}"
                        if src_file.exists():
                            shutil.copy2(src_file, target_folder / f"{target_basename}{src_file.suffix}")
                            break

                copied_sets += 1

            return copied_sets

        primary_count = copy_dataset_sets(primary_dir)
        secondary_count = copy_dataset_sets(secondary_dir)
        total_sets = primary_count + secondary_count

        if total_sets == 0:
            shutil.rmtree(target_dir, ignore_errors=True)
            return jsonify({'error': 'No image sets found to merge'}), 400

        return jsonify({
            'success': True,
            'primaryFolder': primary_folder,
            'secondaryFolder': secondary_folder,
            'targetFolder': str(target_dir.relative_to(DATASETS_DIR)),
            'targetName': target_dir.name,
            'primaryCount': primary_count,
            'secondaryCount': secondary_count,
            'totalCount': total_sets
        })
    except Exception as e:
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/import/scan', methods=['POST'])
def scan_import_datasets():
    """Scan an external trainer export folder for reverse-exported datasets."""
    data = request.get_json() or {}
    scan_path = (data.get('path') or '').strip()

    if not scan_path:
        return jsonify({'error': 'Import path is required'}), 400

    try:
        scan_dir = Path(scan_path).expanduser().resolve()
        if not scan_dir.exists() or not scan_dir.is_dir():
            return jsonify({'error': 'Import path does not exist or is not a folder'}), 404

        datasets = scan_exported_dataset_groups(scan_dir)

        return jsonify({
            'success': True,
            'path': str(scan_dir),
            'datasets': datasets
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/import/dataset/start', methods=['POST'])
def import_external_dataset_start():
    """Start importing one reverse-exported dataset group into the local Datasets folder."""
    data = request.get_json() or {}
    base_path = (data.get('basePath') or '').strip()
    source_name = (data.get('sourceName') or '').strip()
    target_name = (data.get('targetName') or source_name).strip()

    if not base_path or not source_name:
        return jsonify({'error': 'Import path and source dataset are required'}), 400

    if not target_name:
        return jsonify({'error': 'Target dataset name is required'}), 400

    if not re.match(r'^[a-zA-Z0-9_-]+$', target_name):
        return jsonify({'error': 'Target name can only contain letters, numbers, underscores and hyphens'}), 400

    try:
        scan_dir = Path(base_path).expanduser().resolve()
        if not scan_dir.exists() or not scan_dir.is_dir():
            return jsonify({'error': 'Import path does not exist or is not a folder'}), 404

        datasets = scan_exported_dataset_groups(scan_dir)
        dataset = next((item for item in datasets if item['sourceName'] == source_name), None)
        if not dataset:
            return jsonify({'error': 'Dataset group not found in import path'}), 404

        target_dir = DATASETS_DIR / target_name
        if target_dir.exists():
            return jsonify({'error': f'Dataset "{target_name}" already exists'}), 400

        folder_plans, total_files = build_import_plan(dataset, target_dir)
        if total_files == 0:
            return jsonify({'error': 'No files found to import'}), 404

        job_id = uuid.uuid4().hex
        folder_progress = {
            folder_plan['targetFolder']: {
                'total': len(folder_plan['files']),
                'copied': 0
            }
            for folder_plan in folder_plans
        }

        with IMPORT_JOBS_LOCK:
            IMPORT_JOBS[job_id] = {
                'jobId': job_id,
                'status': 'queued',
                'sourceName': source_name,
                'targetName': target_name,
                'copiedFiles': 0,
                'totalFiles': total_files,
                'progressPercent': 0.0,
                'currentFolder': None,
                'folderProgress': folder_progress,
                'error': None,
                'started': False,
                'finished': False
            }

        worker = threading.Thread(target=run_import_job, args=(job_id, folder_plans, target_dir), daemon=True)
        worker.start()

        return jsonify({
            'success': True,
            'jobId': job_id,
            'sourceName': source_name,
            'targetName': target_name,
            'totalFiles': total_files,
            'folderProgress': folder_progress
        })
    except Exception as e:
        if 'target_dir' in locals() and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/import/status/<job_id>')
def import_external_dataset_status(job_id):
    """Return the current status of a background dataset import job."""
    with IMPORT_JOBS_LOCK:
        job = IMPORT_JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Import job not found'}), 404
        return jsonify(job)


@app.route('/api/tool-jobs/start', methods=['POST'])
def start_tool_job():
    data = request.get_json() or {}
    tool_name = (data.get('tool') or '').strip()
    payload = data.get('payload') or {}

    if tool_name not in {'reshuffle', 'compress', 'blur', 'mirror', 'merge', 'fit'}:
        return jsonify({'error': 'Unsupported tool job'}), 400

    try:
        if tool_name in {'reshuffle', 'compress', 'blur', 'mirror', 'fit'}:
            folder_path = (payload.get('folderPath') or '').strip()
            if not folder_path:
                return jsonify({'error': 'Dataset folder is required'}), 400
            if not (DATASETS_DIR / folder_path).exists():
                return jsonify({'error': 'Dataset not found'}), 404

        if tool_name == 'blur':
            strength = float(payload.get('strength', 0))
            if strength < 0:
                return jsonify({'error': 'Blur strength must be non-negative'}), 400
            payload['strength'] = strength

        if tool_name == 'mirror':
            horizontal = bool(payload.get('horizontal'))
            vertical = bool(payload.get('vertical'))
            if not horizontal and not vertical:
                return jsonify({'error': 'Select at least one mirror direction'}), 400

            excluded_controls = set(payload.get('excludedControls') or [])
            invalid_controls = excluded_controls - {'Control1', 'Control2', 'Control3'}
            if invalid_controls:
                return jsonify({'error': 'Invalid excluded controls specified'}), 400

        if tool_name == 'merge':
            primary_folder = (payload.get('primaryFolder') or '').strip()
            secondary_folder = (payload.get('secondaryFolder') or '').strip()
            target_name = (payload.get('targetName') or '').strip()

            if not primary_folder or not secondary_folder:
                return jsonify({'error': 'Both source datasets are required'}), 400
            if primary_folder == secondary_folder:
                return jsonify({'error': 'Choose two different datasets to merge'}), 400
            if not target_name:
                return jsonify({'error': 'Target dataset name is required'}), 400
            if not re.match(r'^[a-zA-Z0-9_-]+$', target_name):
                return jsonify({'error': 'Name can only contain letters, numbers, underscores and hyphens'}), 400

        job_id = uuid.uuid4().hex
        with TOOL_JOBS_LOCK:
            TOOL_JOBS[job_id] = {
                'jobId': job_id,
                'tool': tool_name,
                'status': 'queued',
                'processedItems': 0,
                'totalItems': 0,
                'progressPercent': 0.0,
                'currentItem': None,
                'metrics': {},
                'result': None,
                'error': None,
                'started': False,
                'finished': False
            }

        worker = threading.Thread(target=run_tool_job, args=(job_id, tool_name, payload), daemon=True)
        worker.start()

        return jsonify({
            'success': True,
            'jobId': job_id,
            'tool': tool_name
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tool-jobs/status/<job_id>')
def tool_job_status(job_id):
    with TOOL_JOBS_LOCK:
        job = TOOL_JOBS.get(job_id)

    if not job:
        return jsonify({'error': 'Tool job not found'}), 404

    return jsonify(job)

@app.route('/api/export', methods=['POST'])
def export_dataset():
    """Export dataset to AI-Toolkit format with separate folders per control type"""
    folder_path = request.args.get('folder', '')
    data = request.get_json() or {}
    export_path = data.get('exportPath', '')
    
    if not folder_path:
        return jsonify({'error': 'Dataset folder is required'}), 400
    if not export_path:
        return jsonify({'error': 'Export path is required'}), 400
    
    try:
        dataset_dir = DATASETS_DIR / folder_path
        dataset_name = folder_path.replace('/', '_')
        export_base = Path(export_path)
        
        if not dataset_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Create export base directory if it doesn't exist
        export_base.mkdir(parents=True, exist_ok=True)
        
        # Mapping of source folders to export folder suffixes
        folder_mapping = {
            'img': '_img',
            'Control1': '_ctr1',
            'Control2': '_ctr2',
            'Control3': '_ctr3'
        }
        
        exported = {}
        for src_folder, suffix in folder_mapping.items():
            src_dir = dataset_dir / src_folder
            
            if not src_dir.exists():
                continue
            
            # Check if folder has any files
            files = [f for f in src_dir.iterdir() if f.is_file()]
            if not files:
                continue
            
            # Create export folder
            export_folder = export_base / f"{dataset_name}{suffix}"
            export_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy all files
            copied_count = 0
            for file_path in files:
                dest_path = export_folder / file_path.name
                try:
                    shutil.copy2(str(file_path), str(dest_path))
                except OSError:
                     # Fallback if metadata copy fails (e.g. invalid argument on exFAT)
                    shutil.copy(str(file_path), str(dest_path))
                copied_count += 1
            
            exported[src_folder] = {
                'folder': str(export_folder),
                'files': copied_count
            }
        
        if not exported:
            return jsonify({'error': 'No files to export'}), 404
        
        return jsonify({
            'success': True,
            'exportPath': str(export_base),
            'exported': exported
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/augment/crop', methods=['POST'])
def augment_crop():
    """Create a new augmented image set by cropping a region"""
    try:
        data = request.get_json() or {}
        folder_path = data.get('folder', '')
        filename = data.get('filename', '')
        crop_data = data.get('crop', {}) # {x, y, w, h}
        source_exception = data.get('sourceException', '') # folder name like 'Control1'
        
        if not folder_path or not filename:
             return jsonify({'error': 'Folder and filename are required'}), 400
             
        if not crop_data or 'x' not in crop_data or 'w' not in crop_data:
            return jsonify({'error': 'Invalid crop data'}), 400
            
        dataset_dir = DATASETS_DIR / folder_path
        if not dataset_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404
            
        # Get crop coordinates
        x_raw = int(crop_data.get('x', 0))
        y_raw = int(crop_data.get('y', 0))
        w_raw = int(crop_data.get('w', 1))
        h_raw = int(crop_data.get('h', 1))
        
        if w_raw <= 0 or h_raw <= 0:
            return jsonify({'error': 'Invalid crop dimensions'}), 400
            
        crop_mode = data.get('mode', 'keep')
        
        # Generate new unique basename
        import string
        chars = string.ascii_lowercase + string.digits
        
        def generate_unique_name():
            img_dir = dataset_dir / 'img'
            while True:
                name = ''.join(random.choices(chars, k=8))
                is_unique = True
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
                    if (img_dir / f"{name}{ext}").exists():
                        is_unique = False
                        break
                if is_unique:
                    return name
                    
        new_basename = generate_unique_name()
        
        # Process each folder
        from PIL import Image
        
        folders_to_process = ['img', 'Control1', 'Control2', 'Control3']
        processed_files = []
        
        basename = os.path.splitext(filename)[0]
        original_ext = os.path.splitext(filename)[1]
        
        # We need the reference size from 'img' to know how to scale x, y, w, h
        ref_w, ref_h = 1, 1
        ref_img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            temp_path = dataset_dir / 'img' / f"{basename}{ext}"
            if temp_path.exists():
                ref_img_path = temp_path
                break
                
        if ref_img_path:
            try:
                with Image.open(ref_img_path) as ref_img:
                    ref_w, ref_h = ref_img.size
            except Exception:
                pass
        
        for folder_name in folders_to_process:
            src_folder = dataset_dir / folder_name
            
            # Find the file in this folder (might have different extension)
            src_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                temp_path = src_folder / f"{basename}{ext}"
                if temp_path.exists():
                    src_file = temp_path
                    break
            
            if not src_file:
                continue
                
            dest_file = src_folder / f"{new_basename}{src_file.suffix}"
            
            if folder_name == source_exception:
                # Exception case: Keep the full original image
                shutil.copy2(src_file, dest_file)
            else:
                # Normal case: Crop and Resize
                try:
                    img = Image.open(src_file)
                    original_size = img.size # (width, height)
                    cw, ch = original_size
                    
                    # Scale coordinates if this image has a different resolution from ref_img
                    scale_x = cw / ref_w if ref_w > 0 else 1.0
                    scale_y = ch / ref_h if ref_h > 0 else 1.0
                    
                    x = int(x_raw * scale_x)
                    y = int(y_raw * scale_y)
                    w = int(w_raw * scale_x)
                    h = int(h_raw * scale_y)
                    
                    # Validate crop bounds
                    cx = max(0, min(x, cw - 1))
                    cy = max(0, min(y, ch - 1))
                    cw_crop = min(w, cw - cx)
                    ch_crop = min(h, ch - cy)
                    
                    if cw_crop <= 0 or ch_crop <= 0:
                         # Fallback to copy if crop is invalid
                         shutil.copy2(src_file, dest_file)
                    else:
                        # Crop
                        cropped_img = img.crop((cx, cy, cx + cw_crop, cy + ch_crop))
                        
                        if crop_mode == '1:1':
                            target_dim = max(cw_crop, ch_crop)
                            final_img = cropped_img.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
                        else:
                            # Keep Proportion mode: resize back to original_size (which perfectly preserves aspect ratio)
                            final_img = cropped_img.resize(original_size, Image.Resampling.LANCZOS)
                        
                        final_img.save(dest_file)
                except Exception as e:
                    print(f"Error processing {src_file}: {e}")
                    # Fallback copy on error?
                    shutil.copy2(src_file, dest_file)
            
            processed_files.append(str(dest_file.relative_to(BASE_DIR)))
            
            # Handle Caption (only for img folder)
            if folder_name == 'img':
                txt_src = src_folder / f"{basename}.txt"
                if txt_src.exists():
                    txt_dest = src_folder / f"{new_basename}.txt"
                    shutil.copy2(txt_src, txt_dest)

        return jsonify({
            'success': True,
            'newBasename': new_basename,
            'processed': processed_files
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/augment/duplicate', methods=['POST'])
def augment_duplicate():
    """Duplicate the current image pair and caption"""
    try:
        data = request.get_json() or {}
        folder_path = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder_path or not filename:
             return jsonify({'error': 'Folder and filename are required'}), 400
             
        dataset_dir = DATASETS_DIR / folder_path
        if not dataset_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404
            
        # Generate new unique basename
        import string
        chars = string.ascii_lowercase + string.digits
        
        def generate_unique_name():
            img_dir = dataset_dir / 'img'
            while True:
                name = ''.join(random.choices(chars, k=8))
                is_unique = True
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
                    if (img_dir / f"{name}{ext}").exists():
                        is_unique = False
                        break
                if is_unique:
                    return name
                    
        new_basename = generate_unique_name()
        
        # Process each folder
        folders_to_process = ['img', 'Control1', 'Control2', 'Control3']
        processed_files = []
        
        basename = os.path.splitext(filename)[0]
        original_ext = os.path.splitext(filename)[1]
        
        for folder_name in folders_to_process:
            src_folder = dataset_dir / folder_name
            
            # Find the file in this folder (might have different extension)
            src_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                temp_path = src_folder / f"{basename}{ext}"
                if temp_path.exists():
                    src_file = temp_path
                    break
            
            if not src_file:
                continue
                
            dest_file = src_folder / f"{new_basename}{src_file.suffix}"
            shutil.copy2(src_file, dest_file)
            processed_files.append(str(dest_file.relative_to(BASE_DIR)))
            
            # Handle Caption (only for img folder)
            if folder_name == 'img':
                txt_src = src_folder / f"{basename}.txt"
                if txt_src.exists():
                    txt_dest = src_folder / f"{new_basename}.txt"
                    shutil.copy2(txt_src, txt_dest)

        return jsonify({
            'success': True,
            'newBasename': new_basename,
            'newFilename': f"{new_basename}{original_ext}",
            'processed': processed_files
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/fit', methods=['POST'])
def fit_dataset():
    """Resize/Letterbox all control images to match their corresponding primary images (img folder)"""
    folder_path = request.args.get('folder', '')
    
    try:
        from PIL import Image, ImageOps
        
        dataset_dir = DATASETS_DIR / folder_path
        if not dataset_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404
            
        img_dir = dataset_dir / 'img'
        control_folders = ['Control1', 'Control2', 'Control3']
        
        if not img_dir.exists():
            return jsonify({'error': 'Primary image directory (img) not found'}), 404
            
        processed_count = 0
        updated_count = 0
        
        # Get all image files from img folder
        primary_images = {}
        for f in img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                primary_images[f.stem] = f
                
        if not primary_images:
            return jsonify({'error': 'No primary images found in img folder'}), 404
            
        for basename, primary_path in primary_images.items():
            processed_count += 1
            
            # Get target size from primary image
            with Image.open(primary_path) as p_img:
                target_size = p_img.size  # (W, H)
                
            for ctrl_folder in control_folders:
                ctrl_dir = dataset_dir / ctrl_folder
                if not ctrl_dir.exists():
                    continue
                    
                # Find matching control image (any common extension)
                ctrl_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                    temp_path = ctrl_dir / f"{basename}{ext}"
                    if temp_path.exists():
                        ctrl_path = temp_path
                        break
                
                if not ctrl_path:
                    continue
                    
                # Process the control image
                with Image.open(ctrl_path) as c_img:
                    # If already correct size, skip
                    if c_img.size == target_size:
                        continue
                        
                    # Calculate new size with letterboxing (preserving aspect ratio)
                    # ImageOps.pad is perfect for this.
                    # It creates a background of target_size and pastes the resized image centered with bars.
                    # Default is centered, background color black.
                    new_img = ImageOps.pad(c_img, target_size, color=(0, 0, 0), centering=(0.5, 0.5))
                    
                    # Save back to original path
                    # We preserve the original format
                    new_img.save(ctrl_path)
                    updated_count += 1
                    
        return jsonify({
            'success': True, 
            'processed': processed_count, 
            'updated': updated_count
        })
        
    except ImportError:
        return jsonify({'error': 'Pillow library not installed'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/open-in-pixelmator', methods=['POST'])
def open_in_pixelmator():
    """Open all images from a pair as layers in Pixelmator Pro using AppleScript"""
    try:
        data = request.get_json() or {}
        folder_path = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder_path or not filename:
             return jsonify({'error': 'Folder and filename are required'}), 400
             
        dataset_dir = DATASETS_DIR / folder_path
        if not dataset_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404
            
        basename = os.path.splitext(filename)[0]
        folders_to_check = ['img', 'Control1', 'Control2', 'Control3']
        image_paths = []
        
        # Collect all existing images for this pair
        for folder_name in folders_to_check:
            folder = dataset_dir / folder_name
            if not folder.exists():
                continue
                
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                file_path = folder / f"{basename}{ext}"
                if file_path.exists():
                    image_paths.append(str(file_path.absolute()))
                    break
        
        if not image_paths:
            return jsonify({'error': 'No images found to open'}), 404
            
        # We want 'img' to be the first one to open (sets canvas size)
        # But we also want 'img' to be on top at the end.
        
        paths_string = '", "'.join(image_paths)
        
        # AppleScript to open images as layers
        # 1. Open first image (img) to create document and set canvas size
        # 2. Add remaining images as image layers (scale Control1 if needed)
        # 3. Move 'img' to front and 'Control1' to second position
        applescript = f'''
        set imgPaths to {{"{paths_string}"}}
        set imgAliases to {{}}
        repeat with p in imgPaths
            set end of imgAliases to (POSIX file p) as alias
        end repeat

        tell application "Pixelmator Pro"
            activate
            open item 1 of imgAliases
            set doc to front document
            set docWidth to width of doc
            set docHeight to height of doc
            
            repeat with i from 2 to (count of imgAliases)
                set nextAlias to item i of imgAliases
                tell doc
                    set newLayer to make new image layer with properties {{file:nextAlias}}
                    -- If this is Control1 (index 2 in imgPaths), scale it to doc size
                    if i is 2 then
                        set width of newLayer to docWidth
                        set height of newLayer to docHeight
                        set position of newLayer to {{0, 0}}
                    end if
                end tell
            end repeat
            
            -- Reorder so: [img, Control1, Control2, Control3] (top to bottom)
            tell doc
                move last layer to front -- Move 'img' to top
                if (count of layers) > 1 then
                    -- The first control image added (item 2 of imgAliases)
                    -- ended up right above 'img', so now it's at the bottom.
                    move last layer to after layer 1
                end if
            end tell
        end tell
        '''
        
        import subprocess
        process = subprocess.Popen(['osascript', '-e', applescript], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return jsonify({'error': f'AppleScript failed: {stderr}'}), 500
            
        return jsonify({'success': True})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─── Process Text ────────────────────────────────────────────────────────────

def _apply_process_config(caption_text, config):
    """Process a single caption string using the given config.

    Steps:
      1. Split on commas, strip whitespace from each tag.
      2. Strip Stable-Diffusion weight syntax: (tag:1.0) → tag, (tag) → tag.
      3. Drop tags matching any pattern in config['drops'].
      4. Assign remaining tags to named slots via config['slots'] (first match wins).
      5. Render config['template'] substituting {slot} values.
         Optional blocks {? ... {slot} ... } are rendered only when every
         referenced slot is filled; otherwise they collapse to ''.
      6. Replace {remainder} with comma-joined unmatched tags.
    """
    drops = config.get('drops', [])
    slots_cfg = config.get('slots', [])
    template = config.get('template', '{remainder}')

    # 1. Split
    tags = [t.strip() for t in caption_text.split(',') if t.strip()]

    # 2. Strip weight / paren syntax (re.DOTALL handles multiline tags)
    cleaned = []
    for tag in tags:
        m = re.match(r'^\((.+?):\s*[\d.]+\s*\)$', tag, re.DOTALL)
        if m:
            tag = m.group(1).strip()
        else:
            m = re.match(r'^\((.+?)\)$', tag, re.DOTALL)
            if m:
                tag = m.group(1).strip()
        # Collapse any internal newlines (from multiline paren tags)
        tag = re.sub(r'\s*\n\s*', ' ', tag).strip()
        # Strip stray unbalanced parens from comma-split inside (a, b, c) groups
        if tag.startswith('(') and ')' not in tag:
            tag = tag[1:].strip()
        if tag.endswith(')') and '(' not in tag:
            tag = tag[:-1].strip()
        if tag:
            cleaned.append(tag)

    # 3. Drop
    remaining = []
    for tag in cleaned:
        dropped = any(
            p.strip() and re.search(p.strip(), tag, re.IGNORECASE)
            for p in drops if p.strip()
        )
        if not dropped:
            remaining.append(tag)

    # 4. Slot assignment
    slots = {}
    unmatched = []
    for tag in remaining:
        matched = False
        for slot in slots_cfg:
            name = slot.get('name', '').strip()
            pattern = slot.get('pattern', '').strip()
            transform = slot.get('transform', '').strip()
            if not name or not pattern:
                continue
            m = re.search(pattern, tag, re.IGNORECASE)
            if m:
                if name not in slots:          # first match wins
                    if transform:
                        value = transform
                        for i, g in enumerate(m.groups(), 1):
                            if g is not None:
                                value = value.replace(f'${i}', g)
                        value = value.replace('$0', m.group(0))
                    else:
                        value = tag
                    slots[name] = value
                matched = True
                break
        if not matched:
            unmatched.append(tag)

    # 5. Render template – optional blocks first
    # Pattern matches {? ... } where content may contain simple {word} refs
    def _render_optional(m):
        inner = m.group(1)
        refs = re.findall(r'\{(\w+)\}', inner)
        if all(r in slots for r in refs):
            result = inner
            for r in refs:
                result = result.replace(f'{{{r}}}', slots[r])
            return result
        return ''

    result = re.sub(
        r'\{\?([^{}]*(?:\{[^{}]+\}[^{}]*)*)\}',
        _render_optional,
        template
    )

    # Fill remaining {slot} placeholders
    for name, value in slots.items():
        result = result.replace(f'{{{name}}}', value)

    # Remove any unfilled {slot} refs
    result = re.sub(r'\{(?!remainder)\w+\}', '', result)

    # Inject remainder
    result = result.replace('{remainder}', ', '.join(unmatched))

    # Tidy up whitespace / punctuation artifacts
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r' +([,.])', r'\1', result)
    result = re.sub(r'([,.]){2,}', r'\1', result)
    result = re.sub(r',\s*\.', '.', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


DEFAULT_PROCESS_CONFIG = {
    "drops": [
        "^masterpiece$",
        "^solo$",
        "^best quality$",
        "^highres$",
        r"^\d+girl$",
        r"^\d+boy$",
        r"\bbackground\b",
        "^expressionless$",
        "^human race$",
        "^naked$",
        "^nude$",
        "^vagina$",
        "^nipples?$"
    ],
    "slots": [
        # Age: "18yo" → slot age = "18yo"
        {"name": "age",
         "pattern": r"^(\d+)yo$",
         "transform": "$1yo"},

        # Age-body descriptors like "teenager girl", "adult woman", "young_adult woman", "loli" —
        # swallowed intentionally: they duplicate the numeric age and clutter remainder.
        {"name": "age_body",
         "pattern": r"^(?:loli|teenager|adult|young(?:_adult)?)(?:[\s_]\w+)*$",
         "transform": "$0"},

        # Hair colour: "purple hair" → "purple"
        {"name": "hair",
         "pattern": r"^([\w][\w\s-]*)\s+hair$",
         "transform": "$1"},

        # Eyes colour: "white eyes" → "white"
        {"name": "eyes",
         "pattern": r"^(\w+)\s+eyes$",
         "transform": "$1"},

        # Breasts / chest: "big breasts body" → "big breasts", "flat chest" → "flat chest"
        # Group 1 = modifier(s), group 2 = body-part word; trailing " body" stripped.
        {"name": "breasts",
         "pattern": r"^([\w][\w\s-]*?)\s+(chest|breasts?|bust)(?:\s+body)?$",
         "transform": "$1 $2"},

        # Clothes: strips optional leading "wear[ing][ :] ", captures the rest.
        # Handles both "wear white bra" and "wear: one-sleeve velvet corset with slanted hem skirt"
        {"name": "clothes",
         "pattern": r"^(?:wear(?:ing)?[:\s]\s*)?(.*(?:lingerie|dress|gown|shirt|pants|trousers|skirt|blouse|bra|panties|bikini|thong|uniform|coat|jacket|outfit|corset|turtleneck|bodysuit|robe|kimono|shorts|leggings|stockings|swimwear|hoodie|sweater).*)$",
         "transform": "$1"},

        # Location
        {"name": "location",
         "pattern": r"^(?:outdoor|indoor|beach|forest|room|street|park|city)$",
         "transform": "$0"},

        # Posing / action
        {"name": "posing",
         "pattern": r"^(?:posing|sitting|standing|lying|kneeling|crouching)$",
         "transform": "$0"},

        # NOTE: style, race, skin are intentionally NOT listed as slots.
        # They fall through to {remainder} and are preserved in the output.
    ],
    "template": (
        "{age} girl{? with {breasts}}{?, {hair} hair}{? and {eyes} eyes}"
        "{?, {posing}}{? in {location}}.\n"
        "{?She wears {clothes}.}\n"
        "{remainder}"
    )
}


@app.route('/api/process-text/config', methods=['GET', 'POST'])
def process_text_config():
    folder = request.args.get('folder', '')
    if not folder:
        return jsonify({'error': 'folder required'}), 400

    config_path = DATASETS_DIR / folder / '.process_config.json'

    if request.method == 'GET':
        if config_path.exists():
            return jsonify(json.loads(config_path.read_text(encoding='utf-8')))
        return jsonify(DEFAULT_PROCESS_CONFIG)

    data = request.get_json() or {}
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return jsonify({'success': True})


@app.route('/api/process-text/preview', methods=['POST'])
def process_text_preview():
    data = request.get_json() or {}
    folder = data.get('folder', '')
    config = data.get('config', {})
    count = int(data.get('count', 5))

    if not folder:
        return jsonify({'error': 'folder required'}), 400

    img_dir = DATASETS_DIR / folder / 'img'
    if not img_dir.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    all_txt = [f for f in img_dir.iterdir() if f.suffix == '.txt']
    txt_files = random.sample(all_txt, min(count, len(all_txt)))
    results = []
    for f in txt_files:
        try:
            original = f.read_text(encoding='utf-8').strip()
            processed = _apply_process_config(original, config)
            results.append({'filename': f.name, 'original': original, 'processed': processed})
        except Exception as e:
            results.append({'filename': f.name, 'original': '(read error)', 'processed': str(e)})

    return jsonify({'results': results})


@app.route('/api/process-text/apply', methods=['POST'])
def process_text_apply():
    data = request.get_json() or {}
    folder = data.get('folder', '')
    config = data.get('config', {})

    if not folder:
        return jsonify({'error': 'folder required'}), 400

    img_dir = DATASETS_DIR / folder / 'img'
    if not img_dir.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    txt_files = [f for f in img_dir.iterdir() if f.suffix == '.txt']
    processed_count = 0
    errors = []

    for f in txt_files:
        try:
            original = f.read_text(encoding='utf-8').strip()
            result = _apply_process_config(original, config)
            f.write_text(result, encoding='utf-8')
            processed_count += 1
        except Exception as e:
            errors.append({'file': f.name, 'error': str(e)})

    return jsonify({'success': True, 'processed': processed_count, 'errors': errors})


@app.route('/api/stitch', methods=['POST'])
def stitch_images():
    """Stitch multiple image sets into a single new entry"""
    try:
        from PIL import Image
        data = request.get_json() or {}
        folder_path = data.get('folder', '')
        filenames = data.get('filenames', [])   # ordered list
        direction = data.get('direction', 'horizontal')  # horizontal | vertical
        as_is_control = data.get('asIsControl', '')  # Control1/Control2/Control3 or ''

        if not folder_path or len(filenames) < 2:
            return jsonify({'error': 'Folder and at least 2 filenames are required'}), 400

        dataset_dir = DATASETS_DIR / folder_path
        if not dataset_dir.exists():
            return jsonify({'error': 'Dataset not found'}), 404

        # Generate unique basename
        import string
        chars = string.ascii_lowercase + string.digits

        def generate_unique_name():
            img_dir = dataset_dir / 'img'
            while True:
                name = ''.join(random.choices(chars, k=8))
                if all(not (img_dir / f"{name}{ext}").exists()
                       for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt']):
                    return name

        new_basename = generate_unique_name()
        subfolders = ['img', 'Control1', 'Control2', 'Control3']
        processed = []

        for subfolder in subfolders:
            folder_dir = dataset_dir / subfolder
            if not folder_dir.exists():
                continue

            # Collect existing source files in order
            sources = [folder_dir / fn for fn in filenames if (folder_dir / fn).exists()]
            if not sources:
                continue

            new_path = folder_dir / f"{new_basename}.png"

            if subfolder == as_is_control:
                # Keep first image as-is (no stitching); use copy (not copy2) to get fresh timestamps
                shutil.copy(str(sources[0]), str(new_path))
            else:
                # Open all images for this subfolder
                imgs = [Image.open(src).copy() for src in sources]

                # Normalise to RGB (or RGBA if any source has alpha)
                has_alpha = any(img.mode in ('RGBA', 'LA', 'PA') for img in imgs)
                mode = 'RGBA' if has_alpha else 'RGB'
                bg = (0, 0, 0, 255) if has_alpha else (0, 0, 0)
                imgs = [img.convert(mode) for img in imgs]

                if direction == 'grid2x2':
                    # Arrange 4 images in a 2×2 grid: [0][1] / [2][3]
                    if len(imgs) < 4:
                        imgs += [Image.new(mode, imgs[0].size, bg)] * (4 - len(imgs))
                    row_w = imgs[0].width + imgs[1].width
                    row_h = max(imgs[0].height, imgs[1].height)
                    bot_w = imgs[2].width + imgs[3].width
                    bot_h = max(imgs[2].height, imgs[3].height)
                    total_w = max(row_w, bot_w)
                    total_h = row_h + bot_h
                    canvas = Image.new(mode, (total_w, total_h), bg)
                    canvas.paste(imgs[0], (0, 0))
                    canvas.paste(imgs[1], (imgs[0].width, 0))
                    canvas.paste(imgs[2], (0, row_h))
                    canvas.paste(imgs[3], (imgs[2].width, row_h))
                elif direction == 'horizontal':
                    total_w = sum(img.width for img in imgs)
                    max_h = max(img.height for img in imgs)
                    canvas = Image.new(mode, (total_w, max_h), bg)
                    x = 0
                    for img in imgs:
                        canvas.paste(img, (x, 0))
                        x += img.width
                else:  # vertical
                    max_w = max(img.width for img in imgs)
                    total_h = sum(img.height for img in imgs)
                    canvas = Image.new(mode, (max_w, total_h), bg)
                    y = 0
                    for img in imgs:
                        canvas.paste(img, (0, y))
                        y += img.height

                canvas.save(str(new_path), 'PNG')
                for img in imgs:
                    img.close()

            processed.append(f"{subfolder}/{new_basename}.png")

        # Copy caption from first filename
        first_stem = os.path.splitext(filenames[0])[0]
        txt_src = dataset_dir / 'img' / f"{first_stem}.txt"
        if txt_src.exists():
            shutil.copy(str(txt_src), str(dataset_dir / 'img' / f"{new_basename}.txt"))

        return jsonify({'success': True, 'newFilename': f"{new_basename}.png", 'processed': processed})

    except ImportError:
        return jsonify({'error': 'Pillow library not installed'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting Dataset Manager...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
