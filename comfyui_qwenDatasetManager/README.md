# Qwen Dataset Manager - ComfyUI Node

ComfyUI custom nodes for saving/loading Qwen datasets and merging Qwen LoRAs.

## Installation

1. Copy the `comfyui_qwenDatasetManager` folder to your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/comfyui_qwenDatasetManager/
   ```

2. Restart ComfyUI

## Node: Qwen Dataset Saver

### Inputs

**Required:**
- `target` (IMAGE) - The main target image to save
- `dataset_name` (STRING) - Name of the dataset folder

**Optional:**
- `control1` (IMAGE) - First control image
- `control2` (IMAGE) - Second control image  
- `control3` (IMAGE) - Third control image
- `caption` (STRING) - Text caption to save as .txt file

### Behavior

1. **Automatic Numbering**: Files are saved with sequential numbering (image_00001.png, image_00002.png, etc.)
   - Scans existing files and continues from the highest number

2. **Directory Structure**: Creates the following structure in ComfyUI's output directory:
   ```
   output/
   └── {dataset_name}/
       ├── img/           # Target images
       ├── Control1/      # Control image 1
       ├── Control2/      # Control image 2
       └── Control3/      # Control image 3
   ```

3. **Black Image Fallback**: If no control images are provided, saves a black image of the same size as target to Control1

4. **Caption Files**: If caption is provided, saves as `{filename}.txt` in the img folder

5. **Format**: All images saved as PNG without compression

6. **Batch handling**: Every target in the input batch is saved. Control batches must contain either one reusable image or the same number of images as the target batch.

Dataset names are single folder names and cannot escape ComfyUI's output directory.

### Example Usage

Connect your workflow outputs to the node:
- Target image from your generation
- Optional control images (depth, canny, etc.)
- Optional caption text
- Specify dataset name

The node will automatically:
- Create the folder structure
- Find the next available number
- Save all files with matching names
- Print confirmation to console

## Node: Qwen Dataset Loader

Loads a dataset in Manual, List, or deterministic Random mode. Controls are matched by basename even when their extensions differ from the target. Missing controls become black images; controls with different dimensions are padded to the target size.

## Nodes: Qwen LoRA Merge / Save

Merge up to four `.safetensors` LoRAs using concat, weighted sum, weighted average, TIES, or DARE Linear. Weighted operations are performed on LoRA deltas; TIES/DARE results are factorized back to LoRA A/B tensors. Inputs are limited to configured ComfyUI LoRA directories or the repository directory, and outputs cannot escape the selected ComfyUI directory.

## Dataset Compatibility

The saved datasets are compatible with:
- Qwen Dataset Manager web GUI (in this repository)
- Standard training scripts expecting img/Control format
