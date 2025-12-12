## QwenDatasetManager — Copilot Instructions

This repo is a small Flask-backed web UI for inspecting and managing image datasets and a ComfyUI node that saves images in the same dataset layout. The goal of these instructions is to help an AI coding agent be immediately productive here.

- **Entrypoint / UI**: The main server is [app.py](app.py). It serves `static/index.html` and exposes JSON API endpoints under `/api/*` (e.g., `/api/folders`, `/api/images`, `/api/delete/<filename>`, `/api/transfer/<filename>`, `/api/reshuffle`). Use these endpoints for automated tasks and tests.

- **Frontend code**: Look in `static/` (notably [static/app.js](static/app.js) and [static/index.html](static/index.html)) for UI logic and keyboard shortcuts. Keep UI changes minimal and coordinated with the API shape in `app.py`.

- **Dataset layout / conventions**: Every dataset folder under `Datasets/` must contain subfolders: `img/`, `Control1/`, `Control2/`, `Control3/`. Files share basenames across those folders (e.g., `image_00001.png` and `image_00001.txt`). Caption files are plain `.txt` saved alongside images in `img/`.

- **Naming rules**: The web UI endpoint that creates datasets validates names with `^[a-zA-Z0-9_-]+$` (see `create_dataset` in [app.py](app.py)). The ComfyUI node names files using `image_XXXXX.png` sequential numbering (see `comfyui_qwenDatasetManager/qwen_dataset_saver.py`). Reshuffle/transfer endpoints generate 8-character lowercase+digit names when moving files.

- **Key behaviors to preserve in edits**:
  - `compare_datasets` compares image basenames only and treats extensions `.png, .jpg, .jpeg, .webp` as images. Keep that list consistent if you add new formats.
  - `delete` removes all matching files across `img` and `Control*` and `.txt` captions — preserve the multi-folder delete semantics.
  - `transfer` moves files and ensures unique target names; it also accepts an optional `linkedFolder` to transfer related files.

- **ComfyUI integration**: The node is `QwenDatasetSaver` in `comfyui_qwenDatasetManager/qwen_dataset_saver.py`. It expects a ComfyUI `folder_paths.get_output_directory()` to exist. To test the node locally, copy the folder to your ComfyUI `custom_nodes` directory and ensure the ComfyUI Python environment has dependencies (`Pillow`, `torch`, etc.).

- **Dependencies & runtime**: See [requirements.txt](requirements.txt) — primary runtime is Flask + Flask-CORS. The repository includes `run.sh` which activates `.venv` and runs `python app.py`. Recommended local run steps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

- **Useful API examples**:
  - List datasets: `GET /api/folders`
  - List images in a dataset: `GET /api/images?folder=CH3BB`
  - Delete an image set: `DELETE /api/delete/<filename>?folder=CH3BB&linkedFolder=CH4NB`
  - Transfer: `POST /api/transfer/<filename>?folder=CH3BB` with JSON `{"targetFolder": "CH4NB"}`

- **Search/modify hotspots** (good PR/bugfix targets):
  - [app.py](app.py): all endpoint logic, validation, and filesystem operations
  - [comfyui_qwenDatasetManager/qwen_dataset_saver.py](comfyui_qwenDatasetManager/qwen_dataset_saver.py): ComfyUI node behavior and naming logic
  - `static/app.js`: UI interactions and keyboard handling

- **Non-obvious constraints & pitfalls**:
  - Image filename matching is based on `stem` (basename) comparisons — renaming or changing extension handling affects cross-folder matching.
  - `run.sh` expects a `.venv` at repo root; CI or developer machines may need explicit venv creation.
  - The ComfyUI node assumes an external `folder_paths` helper; ensure it is present in the target ComfyUI environment.

- **When editing**: always run the server and exercise the UI flows that touch your change (browse folders, open image, transfer, delete). For filesystem operations prefer atomic moves/copies and include error handling similar to existing endpoints.

If anything is unclear or you'd like more granular examples (unit test snippets, curl suites, or ComfyUI node tests), tell me which area to expand. 
