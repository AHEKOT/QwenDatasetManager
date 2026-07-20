# Qwen Dataset Manager

A local web application for managing, reviewing, editing, and transforming Qwen training datasets.

## Features

- 📁 **Folder Selection** - Browse and select dataset folders
- 🖼️ **Image Grid** - View all images from the dataset in a responsive grid
- 🔍 **Fullscreen Preview** - Click any image to view in fullscreen
- 🎨 **Overlay Comparison** - Toggle between normal view and semi-transparent overlay to compare with Control1 images
- ⌨️ **Keyboard Navigation** - Navigate with arrow keys, toggle with space, delete with backspace
- 🧰 **Dataset Tools** - Reshuffle, compress, fit, blur, mirror, merge, import, export, and duplicate review
- ✏️ **Editing** - Paint/crop images and edit captions with synchronized controls
- 🗑️ **Recoverable Deletion** - Move complete image/control/caption sets into a hidden `.trash` folder

## Dataset Structure

Your dataset folder should have this structure:

```
DatasetFolder/
├── img/          # Source images
├── Control1/     # First control images
├── Control2/     # Second control images
└── Control3/     # Optional third control images
```

All three folders must contain images with matching filenames (e.g., `image_00003_.png`).

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server**:
   ```bash
   python app.py
   ```

3. **Open your browser**:
   Navigate to `http://127.0.0.1:5001`

The server binds to localhost by default. To deliberately expose it on another interface, set `QDM_HOST` and list accepted hostnames in `QDM_TRUSTED_HOSTS`; use a trusted network and a reverse proxy with authentication. `QDM_PORT`, `QDM_DEBUG`, `QDM_MAX_UPLOAD_MB`, and `QDM_MAX_IMAGE_PIXELS` are also configurable environment variables.

## Usage

1. **Select a dataset folder** from the dropdown menu
2. **Browse images** in the grid view
3. **Click an image** to open fullscreen preview
4. **Toggle overlay** to compare img with Control1 (img becomes semi-transparent)
5. **Navigate** using arrow keys or on-screen buttons
6. **Delete** mismatched sets by pressing Backspace/Delete

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Navigate between images |
| `Space` | Toggle overlay |
| `Backspace` / `Delete` | Confirm and move the current synchronized set to `.trash` |
| `Shift+P` | Open the current set in Pixelmator Pro (macOS) |
| `Esc` | Close preview |

## Technical Stack

- **Backend**: Python Flask
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Design**: Modern dark theme with glassmorphism and smooth animations

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
