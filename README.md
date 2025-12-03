# Qwen Dataset Manager

A web-based tool for managing and reviewing Qwen training datasets with image comparison features.

## Features

- 📁 **Folder Selection** - Browse and select dataset folders
- 🖼️ **Image Grid** - View all images from the dataset in a responsive grid
- 🔍 **Fullscreen Preview** - Click any image to view in fullscreen
- 🎨 **Overlay Comparison** - Toggle between normal view and semi-transparent overlay to compare with Control1 images
- ⌨️ **Keyboard Navigation** - Navigate with arrow keys, toggle with space, delete with backspace
- 🗑️ **Batch Deletion** - Delete all related images (img, Control1, Control2) at once

## Dataset Structure

Your dataset folder should have this structure:

```
DatasetFolder/
├── img/          # Source images
├── Control1/     # First control images  
└── Control2/     # Second control images
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
   Navigate to `http://localhost:5000`

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
| `Backspace` / `Delete` | Delete current image set (all 3 files) |
| `Esc` | Close preview |

## Technical Stack

- **Backend**: Python Flask
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Design**: Modern dark theme with glassmorphism and smooth animations

## License

MIT