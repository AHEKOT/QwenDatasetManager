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
- 🚀 **CUDA Trainer** - Train LoRAs for Qwen Image Edit 2511 and FLUX.2 Klein Base 4B/9B from one or more managed datasets

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

The full installer creates separate virtual environments for the application
and CUDA trainer, installs their pinned dependencies, and runs import,
dependency, and CUDA checks.

### Windows

Requirements: 64-bit Windows, Git, an NVIDIA GPU with a current driver, an
internet connection, and at least 10 GB of free space. Python does not need to
be installed in advance: the installer downloads the signed official Python
3.12.10 installer and installs it for the current user when necessary.

Double-click `install.cmd`, or run it from a terminal:

```cmd
install.cmd
```

Useful variants:

```powershell
# Install only the dataset manager, without CUDA dependencies
.\install.ps1 -SkipTrainer

# Install or repair only the CUDA trainer
install_trainer.cmd

# Do not pause the cmd wrapper (useful for CI and scripts)
set QDM_NO_PAUSE=1
install.cmd
```

If Python 3.12 is installed in a non-standard location, set
`QDM_PYTHON312` to its `python.exe` before running the installer. Re-running
the installer is safe: compatible environments and installed packages are
reused, while incomplete environments are repaired.

### Linux

Install Python 3.12 with its `venv` module, Git, and the current NVIDIA driver,
then run:

```bash
./install.sh
```

Use `./install.sh --skip-trainer` on a machine that only needs the dataset
manager. `./install_trainer.sh` installs or repairs only the trainer. The CUDA
trainer is not supported on macOS.

### Run

```cmd
# Windows
run.cmd
```

```bash
# Linux
./run.sh
```

The server prints both the local URL and every detected LAN URL at startup. It
binds to all local interfaces by default and automatically trusts the machine
hostname and detected local IP addresses. To restrict it to this computer, set
`QDM_HOST=127.0.0.1`. Use `QDM_TRUSTED_HOSTS` to override accepted hostnames.
`QDM_PORT`, `QDM_DEBUG`, `QDM_MAX_UPLOAD_MB`, and `QDM_MAX_IMAGE_PIXELS` are
also configurable environment variables.

LAN access should only be enabled on a trusted network. Internet exposure
requires a reverse proxy or VPN with authentication; do not forward the Flask
development server port directly to the internet.

### CUDA trainer

The trainer remains isolated in `trainer/.venv`. The installer intentionally
keeps the CUDA dependency set used by the vendored AI Toolkit backend and does
not provide a macOS or MPS fallback. After installation, start or restart the
app and open **Trainer** from the main screen.

The trainer supports:

- Qwen Image Edit 2511
- FLUX.2 Klein Base 4B
- FLUX.2 Klein Base 9B (gated model; requires Hugging Face access and is subject to the FLUX non-commercial license)

Each selected dataset is passed to AI Toolkit as a separate dataset entry: `img/` is the target and every present `Control1/`, `Control2/`, and `Control3/` folder is a control-image source. Jobs, queue state, progress, and logs are stored under `trainer/`; model checkpoints are written to `trainer/output/`.

The training screen mirrors the AI Toolkit LoRA Trainer settings that apply to
these three edit architectures: the complete transformer/text-encoder
quantization lists (including Qwen 2511 ARA), LoRA and LoKr, validation,
sampling, schedulers, EMA, regularization, compilation and layer offloading.
`Name or Path` accepts either a local model path or a Hugging Face repository;
the vendored backend downloads Hugging Face models in the same way as AI
Toolkit and uses the token saved in Trainer settings. See
[`trainer/PARITY.md`](trainer/PARITY.md) for the audited field matrix and the
architecture-specific restrictions retained from upstream.

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
