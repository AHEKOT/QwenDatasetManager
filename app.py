from flask import Flask, send_from_directory, jsonify, request, send_file
from flask_cors import CORS
import os
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Base directory for datasets
BASE_DIR = Path(__file__).parent

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/folders')
def get_folders():
    """Get list of available dataset folders"""
    try:
        folders = []
        for item in BASE_DIR.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'static':
                # Check if it has img, Control1, Control2 subdirectories
                img_dir = item / 'img'
                control1_dir = item / 'Control1'
                control2_dir = item / 'Control2'
                
                if img_dir.exists() and control1_dir.exists() and control2_dir.exists():
                    folders.append({
                        'name': item.name,
                        'path': str(item.relative_to(BASE_DIR))
                    })
        
        return jsonify({'folders': folders})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images')
def get_images():
    """Get list of images from the img folder"""
    folder_path = request.args.get('folder', '')
    
    try:
        dataset_dir = BASE_DIR / folder_path
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
        dataset_dir = BASE_DIR / folder_path
        
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
        dataset_dir = BASE_DIR / folder_path
        
        deleted_files = []
        errors = []
        
        # Delete from all three folders
        for folder_name in ['img', 'Control1', 'Control2']:
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

if __name__ == '__main__':
    print(f"Starting Dataset Manager...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
