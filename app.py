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
        if image_type not in ['img', 'Control1', 'Control2']:
            return jsonify({'error': 'Invalid image type'}), 400
        
        image_dir = dataset_dir / image_type
        image_path = image_dir / filename
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete all three images (img, Control1, Control2) with the same filename"""
    folder_path = request.args.get('folder', '')
    
    try:
        dataset_dir = DATASETS_DIR / folder_path
        
        deleted_files = []
        errors = []
        
        # Delete from all relevant folders
        folders_to_check = ['img', 'Control1', 'Control2', 'Control3']
        
        # Also check for .txt caption file in img folder
        basename = os.path.splitext(filename)[0]
        txt_filename = f"{basename}.txt"
        
        # Add txt file to deletion list if it exists
        txt_path = dataset_dir / 'img' / txt_filename
        if txt_path.exists():
            try:
                txt_path.unlink()
                deleted_files.append(f"img/{txt_filename}")
            except Exception as e:
                errors.append(f"Failed to delete img/{txt_filename}: {str(e)}")

        for folder_name in folders_to_check:
            folder = dataset_dir / folder_name
            file_path = folder / filename
            
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_files.append(f"{folder_name}/{filename}")
                except Exception as e:
                    errors.append(f"Failed to delete {folder_name}/{filename}: {str(e)}")
            else:
                errors.append(f"{folder_name}/{filename} not found")
        
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
    """Transfer all related files (img, Control1-3, .txt) to another dataset folder"""
    source_folder = request.args.get('folder', '')
    data = request.get_json() or {}
    target_folder = data.get('targetFolder', '')
    
    if not source_folder or not target_folder:
        return jsonify({'error': 'Source and target folders are required'}), 400
    
    if source_folder == target_folder:
        return jsonify({'error': 'Source and target folders must be different'}), 400
    
    try:
        source_dir = DATASETS_DIR / source_folder
        target_dir = DATASETS_DIR / target_folder
        
        # Verify both directories exist and are valid datasets
        for dir_path, name in [(source_dir, 'Source'), (target_dir, 'Target')]:
            if not dir_path.exists():
                return jsonify({'error': f'{name} directory not found'}), 404
            if not (dir_path / 'img').exists():
                return jsonify({'error': f'{name} is not a valid dataset (no img folder)'}), 400
        
        # Get basename without extension
        basename = os.path.splitext(filename)[0]
        original_ext = os.path.splitext(filename)[1]
        
        # Generate unique 8-character name for target
        import string
        chars = string.ascii_lowercase + string.digits
        
        def generate_unique_name():
            """Generate a unique name that doesn't exist in target dataset"""
            existing_names = set()
            target_img_dir = target_dir / 'img'
            if target_img_dir.exists():
                for f in target_img_dir.iterdir():
                    existing_names.add(f.stem)
            
            while True:
                import random
                name = ''.join(random.choices(chars, k=8))
                if name not in existing_names:
                    return name
        
        new_basename = generate_unique_name()
        
        # Folders to check for related files
        folders_to_process = ['img', 'Control1', 'Control2', 'Control3']
        
        # Collect all files to transfer
        files_to_transfer = []
        for folder_name in folders_to_process:
            source_subfolder = source_dir / folder_name
            target_subfolder = target_dir / folder_name
            
            if not source_subfolder.exists():
                continue
            
            # Check for image files with this basename
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                source_file = source_subfolder / f"{basename}{ext}"
                if source_file.exists():
                    # Create target subfolder if it doesn't exist
                    target_subfolder.mkdir(parents=True, exist_ok=True)
                    target_file = target_subfolder / f"{new_basename}{ext}"
                    files_to_transfer.append((source_file, target_file))
            
            # Check for txt caption file (only in img folder)
            if folder_name == 'img':
                txt_source = source_subfolder / f"{basename}.txt"
                if txt_source.exists():
                    txt_target = target_subfolder / f"{new_basename}.txt"
                    files_to_transfer.append((txt_source, txt_target))
        
        if not files_to_transfer:
            return jsonify({'error': 'No files found to transfer'}), 404
        
        # Move all files
        transferred = []
        for source_file, target_file in files_to_transfer:
            shutil.move(str(source_file), str(target_file))
            transferred.append({
                'from': str(source_file.relative_to(BASE_DIR)),
                'to': str(target_file.relative_to(BASE_DIR))
            })
        
        return jsonify({
            'success': True,
            'newFilename': f"{new_basename}{original_ext}",
            'transferred': transferred
        })
        
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

if __name__ == '__main__':
    print(f"Starting Dataset Manager...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
