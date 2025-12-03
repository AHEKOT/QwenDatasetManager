# Qwen Dataset Manager - ComfyUI Node

ComfyUI custom node for saving images in Qwen training dataset format.

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

## Dataset Compatibility

The saved datasets are compatible with:
- Qwen Dataset Manager web GUI (in this repository)
- Standard training scripts expecting img/Control format
