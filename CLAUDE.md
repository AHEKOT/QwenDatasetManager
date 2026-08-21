# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Two related tools for managing Qwen training datasets:

1. **Web GUI** (`app.py` + `static/`) — A Flask web app for reviewing, editing, and managing datasets stored locally in `Datasets/`. Serves on `http://127.0.0.1:5001` by default.

2. **ComfyUI Node** (`comfyui_qwenDatasetManager/`) — A custom ComfyUI node package with two nodes: `QwenDatasetSaver` (saves generated images into dataset format) and `QwenDatasetLoader` (loads dataset images back into ComfyUI workflows).

## Dataset Structure

All tools expect (and create) this folder layout:

```
DatasetFolder/
├── img/           # Source/target images + .txt captions (same basename)
├── Control1/      # Control image set 1
├── Control2/      # Control image set 2
└── Control3/      # Control image set 3
```

Files across folders share the same basename. Captions are stored as `img/<basename>.txt`.

## Running the Web App

```bash
# First-time setup
./install.sh

# Run
./run.sh
# or manually:
source .venv/bin/activate
python app.py
```

Dependencies: Flask 3.1, Werkzeug, and Pillow. No build step is needed and cross-origin access is intentionally disabled.

## ComfyUI Node Installation

Copy `comfyui_qwenDatasetManager/` into `ComfyUI/custom_nodes/` and restart ComfyUI. The package registers saver, loader, LoRA merge, and LoRA save nodes via `__init__.py`.

## Architecture

### Web App (`app.py`)

Single-file Flask backend with these API endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/folders` | List valid dataset folders (must have img + Control1 + Control2) |
| `GET /api/images?folder=X` | List images in a dataset |
| `GET /api/image/<type>/<filename>?folder=X` | Serve image file (type: img/Control1/Control2/Control3) |
| `GET/POST /api/caption/<filename>?folder=X` | Read/write .txt caption |
| `DELETE /api/delete/<filename>` | Delete across all subfolders + optional linked dataset |
| `POST /api/transfer/<filename>` | Move or copy an image set to another dataset with a new random 8-char basename |
| `POST /api/reshuffle` | Rename all files to random 8-char basenames |
| `POST /api/compress` | Lossless PNG recompression using ThreadPoolExecutor |
| `POST /api/augment/crop` | Crop image set; scales crop coords from reference (img) size to each control's actual size |
| `POST /api/compare-datasets` | Find orphan files in a linked dataset |
| `POST /api/create-dataset` | Create empty dataset folder structure |
| `POST /api/export` | Export to AI-Toolkit format (`{name}_img/`, `{name}_ctr1/`, etc.) |
| `POST /api/save/<filename>` | Save edited image back to dataset |

Frontend is `static/index.html` + `static/app.js` + `static/editor.js` (vanilla JS, no framework).

### ComfyUI Nodes (`comfyui_qwenDatasetManager/`)

- **`QwenDatasetSaver`**: Accepts tensors in ComfyUI format `[B, H, W, C]`, converts to PIL, saves PNG with `compress_level=0`. Finds next sequential filename by scanning the `img/` directory.
- **`QwenDatasetLoader`**: Loads datasets in Manual (single file), List (all files), or Random (seeded) mode. Missing control images fall back to black tensors of the same size as the target. Returns `OUTPUT_IS_LIST=True` outputs.

## Key Conventions

- **Filenames**: Sequential `image_NNNNN.png` from the saver node; random 8-char alphanumeric from reshuffle/transfer/copy operations in the web app.
- **Crop scaling**: `augment/crop` receives coordinates relative to the `img/` folder image dimensions and scales them proportionally when applying to control images of different sizes.
- **Datasets dir**: The web app resolves all datasets relative to `Datasets/` (sibling of `app.py`). The ComfyUI saver writes to ComfyUI's output directory.
