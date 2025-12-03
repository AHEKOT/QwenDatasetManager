import os
import re
import torch
import numpy as np
from PIL import Image
import folder_paths


class QwenDatasetSaver:
    """
    ComfyUI node for saving images in Qwen dataset format.
    Saves target image and optional control images with automatic numbering.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE",),
                "dataset_name": ("STRING", {"default": "MyDataset"}),
            },
            "optional": {
                "control1": ("IMAGE",),
                "control2": ("IMAGE",),
                "control3": ("IMAGE",),
                "caption": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_dataset"
    OUTPUT_NODE = True
    CATEGORY = "image/io"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI images are in format [B, H, W, C] with values 0-1
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take first batch
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL
        return Image.fromarray(np_image)
    
    def create_black_image(self, size):
        """Create a black image of specified size"""
        return Image.new('RGB', size, (0, 0, 0))
    
    def get_next_filename(self, directory):
        """Find the next available filename in format image_XXXXX.png"""
        if not os.path.exists(directory):
            return "image_00001.png"
        
        # Find all image files
        pattern = re.compile(r'image_(\d+)\.png')
        max_num = 0
        
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        # Return next number
        next_num = max_num + 1
        return f"image_{next_num:05d}.png"
    
    def save_dataset(self, target, dataset_name, control1=None, control2=None, control3=None, caption=None):
        """Save images in Qwen dataset format"""
        
        # Create dataset directory structure
        dataset_path = os.path.join(self.output_dir, dataset_name)
        img_dir = os.path.join(dataset_path, "img")
        control1_dir = os.path.join(dataset_path, "Control1")
        control2_dir = os.path.join(dataset_path, "Control2")
        control3_dir = os.path.join(dataset_path, "Control3")
        
        # Create directories
        for directory in [img_dir, control1_dir, control2_dir, control3_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Get next filename
        filename = self.get_next_filename(img_dir)
        basename = os.path.splitext(filename)[0]
        
        # Convert target to PIL and save
        target_image = self.tensor_to_pil(target)
        target_path = os.path.join(img_dir, filename)
        target_image.save(target_path, "PNG", compress_level=0)
        
        # Check if any control images are provided
        has_control = control1 is not None or control2 is not None or control3 is not None
        
        # Save control1 (or black image if no control images provided)
        if control1 is not None:
            control1_image = self.tensor_to_pil(control1)
        elif not has_control:
            # No control images at all - create black image same size as target
            control1_image = self.create_black_image(target_image.size)
        else:
            control1_image = None
        
        if control1_image is not None:
            control1_path = os.path.join(control1_dir, filename)
            control1_image.save(control1_path, "PNG", compress_level=0)
        
        # Save control2 if provided
        if control2 is not None:
            control2_image = self.tensor_to_pil(control2)
            control2_path = os.path.join(control2_dir, filename)
            control2_image.save(control2_path, "PNG", compress_level=0)
        
        # Save control3 if provided
        if control3 is not None:
            control3_image = self.tensor_to_pil(control3)
            control3_path = os.path.join(control3_dir, filename)
            control3_image.save(control3_path, "PNG", compress_level=0)
        
        # Save caption if provided
        if caption and caption.strip():
            caption_path = os.path.join(img_dir, f"{basename}.txt")
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption.strip())
        
        print(f"✅ Saved dataset entry: {filename}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Target: {target_path}")
        if control1_image:
            print(f"   Control1: saved")
        if control2 is not None:
            print(f"   Control2: saved")
        if control3 is not None:
            print(f"   Control3: saved")
        if caption and caption.strip():
            print(f"   Caption: saved")
        
        return ()


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenDatasetSaver": QwenDatasetSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenDatasetSaver": "Qwen Dataset Saver"
}
