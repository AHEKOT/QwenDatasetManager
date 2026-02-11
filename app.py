from flask import Flask, send_from_directory, jsonify, request, send_file
from flask_cors import CORS
import os
import random
import shutil
import uuid
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Base directory for datasets
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / 'Datasets'

# Ensure Datasets directory exists
DATASETS_DIR.mkdir(exist_ok=True)

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
            existing_names = set()
            target_img_dir = target_dataset_dir / 'img'
            if target_img_dir.exists():
                for f in target_img_dir.iterdir():
                    existing_names.add(f.stem)
            
            while True:
                import random
                name = ''.join(random.choices(chars, k=8))
                if name not in existing_names:
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
        
        compressed_count = 0
        original_size = 0
        new_size = 0
        
        for folder_name in folders_to_process:
            folder = dataset_dir / folder_name
            if not folder.exists():
                continue
            
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.png':
                    try:
                        original_size += file_path.stat().st_size
                        
                        # Open and re-save with compression
                        img = Image.open(file_path)
                        
                        # Convert to RGB if necessary (PNG can have alpha)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            # Keep alpha channel
                            img.save(file_path, 'PNG', optimize=True, compress_level=9)
                        else:
                            img.save(file_path, 'PNG', optimize=True, compress_level=9)
                        
                        new_size += file_path.stat().st_size
                        compressed_count += 1
                        
                    except Exception as e:
                        print(f"Failed to compress {file_path}: {e}")
        
        # Calculate savings
        savings_mb = (original_size - new_size) / (1024 * 1024)
        savings_percent = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
        
        return jsonify({
            'success': True,
            'compressed': compressed_count,
            'originalSizeMB': round(original_size / (1024 * 1024), 2),
            'newSizeMB': round(new_size / (1024 * 1024), 2),
        'savingsMB': round(savings_mb, 2),
            'savingsPercent': round(savings_percent, 1)
        })
        
    except ImportError:
        return jsonify({'error': 'Pillow library not installed. Run: pip install Pillow'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        x = int(crop_data.get('x', 0))
        y = int(crop_data.get('y', 0))
        w = int(crop_data.get('w', 1))
        h = int(crop_data.get('h', 1))
        
        if w <= 0 or h <= 0:
            return jsonify({'error': 'Invalid crop dimensions'}), 400
            
        # Generate new unique basename
        import string
        chars = string.ascii_lowercase + string.digits
        
        def generate_unique_name():
            existing_names = set()
            img_dir = dataset_dir / 'img'
            if img_dir.exists():
                for f in img_dir.iterdir():
                    existing_names.add(f.stem)
            
            while True:
                name = ''.join(random.choices(chars, k=8))
                if name not in existing_names:
                    return name
                    
        new_basename = generate_unique_name()
        
        # Process each folder
        from PIL import Image
        
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
                
            # Determine logic: Copy original OR Crop & Resize
            # If folder matches source_exception, we keep original (COPY)
            # Otherwise we crop
            
            dest_file = src_folder / f"{new_basename}{src_file.suffix}"
            
            if folder_name == source_exception:
                # Exception case: Keep the full original image
                shutil.copy2(src_file, dest_file)
            else:
                # Normal case: Crop and Resize
                try:
                    img = Image.open(src_file)
                    original_size = img.size # (width, height)
                    
                    # Validate crop bounds
                    cw, ch = original_size
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
                        
                        # Resize back to original size
                        resized_img = cropped_img.resize(original_size, Image.Resampling.LANCZOS)
                        
                        # Save
                        # Preservation of format is handled by save() based on extension or img.format
                        # If original was PNG with P mode (palette), we might want to convert to RGB/RGBA first to avoid issues with resizing?
                        # But Image.open usually handles it.
                        
                        resized_img.save(dest_file)
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
            existing_names = set()
            img_dir = dataset_dir / 'img'
            if img_dir.exists():
                for f in img_dir.iterdir():
                    existing_names.add(f.stem)
            
            while True:
                name = ''.join(random.choices(chars, k=8))
                if name not in existing_names:
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
            
        # Reverse image_paths so the main image ('img') is added last 
        # and appears as the top layer in Pixelmator Pro.
        image_paths.reverse()
        
        paths_string = '", "'.join(image_paths)
        
        # AppleScript to open images as layers
        # 1. Open first image to create document
        # 2. Add remaining as image layers
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
            
            repeat with i from 2 to (count of imgAliases)
                set nextAlias to item i of imgAliases
                tell doc
                    make new image layer with properties {{file:nextAlias}}
                end tell
            end repeat
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

if __name__ == '__main__':
    print(f"Starting Dataset Manager...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
