import os
import torch
import numpy as np
from PIL import Image, ImageOps

import folder_paths

class QwenDatasetLoader:
    """
    ComfyUI node for loading Qwen dataset images and captions.
    Supports Manual (single image) and List (all images) modes.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": "", "multiline": False}),
                "mode": (["Manual", "List"],),
                "manual_filename": ("STRING", {"default": "image_00001.png"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "control1_list", "control2_list", "control3_list", "captions")
    FUNCTION = "load_dataset"
    CATEGORY = "image/io"
    OUTPUT_IS_LIST = (True, True, True, True, True)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor [1, H, W, C]"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        if len(np_image.shape) == 2: # Grayscale
            np_image = np_image[:, :, None]
        tensor = torch.from_numpy(np_image)[None,]
        return tensor
    
    def create_black_image(self, size):
        """Create a black tensor of specified size (W, H)"""
        img = Image.new('RGB', size, (0, 0, 0))
        return self.pil_to_tensor(img)
        
    def load_dataset(self, dataset_path, mode, manual_filename="image_00001.png"):
        dataset_path = dataset_path.strip()
        
        # 1. Try absolute path
        if not os.path.exists(dataset_path):
            # 2. Try relative to ComfyUI output directory
            possible_path = os.path.join(self.output_dir, dataset_path)
            if os.path.exists(possible_path):
                dataset_path = possible_path
            else:
                 raise ValueError(f"Dataset path not found: '{dataset_path}' (checked absolute and relative to output)")
            
        img_dir = os.path.join(dataset_path, "img")
        control1_dir = os.path.join(dataset_path, "Control1")
        control2_dir = os.path.join(dataset_path, "Control2")
        control3_dir = os.path.join(dataset_path, "Control3")
        
        if not os.path.exists(img_dir):
             raise ValueError(f"'img' folder not found at: {img_dir}")
             
        # collect all image files in img_dir
        all_files = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_files.append(f)
        
        all_files.sort()
        
        if not all_files:
            raise ValueError(f"No images found in {img_dir}")
        
        files_to_process = []
        
        if mode == "Manual":
            manual_filename = manual_filename.strip()
            if manual_filename in all_files:
                files_to_process = [manual_filename]
            else:
                 # Try finding it without extension if user didn't provide it
                 found = False
                 for f in all_files:
                     if os.path.splitext(f)[0] == os.path.splitext(manual_filename)[0]:
                         files_to_process = [f]
                         found = True
                         break
                 if not found:
                     print(f"Available files: {all_files[:5]}...") # Debug info
                     raise ValueError(f"Filename '{manual_filename}' not found in dataset. Ensure exact match.")
        else: # List mode
            files_to_process = all_files
            
        images_list = []
        c1_list = []
        c2_list = []
        c3_list = []
        captions_list = []
        
        print(f"Processing {len(files_to_process)} files from {dataset_path}")
        
        for filename in files_to_process:
            # Load Target Image
            img_path = os.path.join(img_dir, filename)
            try:
                pil_img = Image.open(img_path)
                # Ensure RGB
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                target_tensor = self.pil_to_tensor(pil_img)
                target_size = pil_img.size # (W, H)
                
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue # Skip this file if target fails
                
            images_list.append(target_tensor)
            
            # Load Caption
            basename = os.path.splitext(filename)[0]
            caption_path = os.path.join(img_dir, f"{basename}.txt")
            caption_text = ""
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption_text = f.read().strip()
                except:
                    pass
            captions_list.append(caption_text)
            
            # Helper to load control or black
            def load_control(ctrl_dir):
                c_path = os.path.join(ctrl_dir, filename)
                if os.path.exists(c_path):
                    try:
                        c_img = Image.open(c_path)
                        c_img = ImageOps.exif_transpose(c_img)
                        if c_img.mode != 'RGB':
                            c_img = c_img.convert('RGB')
                        return self.pil_to_tensor(c_img)
                    except:
                         return self.create_black_image(target_size)
                else:
                    return self.create_black_image(target_size)

            c1_list.append(load_control(control1_dir))
            c2_list.append(load_control(control2_dir))
            c3_list.append(load_control(control3_dir))
            
        if not images_list:
            raise ValueError(f"Failed to load any images from the selection.")
            
        print(f"QwenDatasetLoader: Successfully loaded {len(images_list)} items.")
        
        return (images_list, c1_list, c2_list, c3_list, captions_list)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenDatasetLoader": QwenDatasetLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenDatasetLoader": "Qwen Dataset Loader"
}
