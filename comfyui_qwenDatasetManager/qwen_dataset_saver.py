import os
import re
import threading
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
import folder_paths


class QwenDatasetSaver:
    """
    ComfyUI node for saving images in Qwen dataset format.
    Saves target image and optional control images with automatic numbering.
    """
    _save_lock = threading.Lock()
    
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
    
    def tensor_to_pil(self, tensor, batch_index=0):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI images are in format [B, H, W, C] with values 0-1
        if len(tensor.shape) == 4:
            if tensor.shape[0] == 0:
                raise ValueError("Image batch is empty")
            tensor = tensor[min(batch_index, tensor.shape[0] - 1)]
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.detach().cpu().numpy().clip(0.0, 1.0) * 255).round().astype(np.uint8)
        
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
            match = pattern.fullmatch(filename)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        # Return next number
        next_num = max_num + 1
        return f"image_{next_num:05d}.png"

    def resolve_dataset_path(self, dataset_name):
        dataset_name = dataset_name.strip()
        if (
            not dataset_name
            or dataset_name in {'.', '..'}
            or '/' in dataset_name
            or '\\' in dataset_name
            or '\x00' in dataset_name
        ):
            raise ValueError("dataset_name must be a single folder name")

        output_root = Path(self.output_dir).resolve()
        dataset_path = (output_root / dataset_name).resolve(strict=False)
        if dataset_path.parent != output_root:
            raise ValueError("Dataset path must stay inside the ComfyUI output directory")
        return dataset_path

    def save_png_atomic(self, image, destination):
        temp_path = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp.png')
        try:
            image.save(temp_path, "PNG", compress_level=0)
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def save_dataset(self, target, dataset_name, control1=None, control2=None, control3=None, caption=None):
        """Save images in Qwen dataset format"""
        
        dataset_path = self.resolve_dataset_path(dataset_name)
        directories = {
            "img": dataset_path / "img",
            "Control1": dataset_path / "Control1",
            "Control2": dataset_path / "Control2",
            "Control3": dataset_path / "Control3",
        }
        batch_size = int(target.shape[0]) if len(target.shape) == 4 else 1
        has_control = any(control is not None for control in (control1, control2, control3))
        for folder_name, control in (
            ("Control1", control1), ("Control2", control2), ("Control3", control3)
        ):
            if control is None:
                continue
            control_batch_size = int(control.shape[0]) if len(control.shape) == 4 else 1
            if control_batch_size not in {1, batch_size}:
                raise ValueError(
                    f"{folder_name} batch must contain either 1 or {batch_size} images"
                )
        saved_entries = []

        with self._save_lock:
            for directory in directories.values():
                directory.mkdir(parents=True, exist_ok=True)

            for batch_index in range(batch_size):
                filename = self.get_next_filename(directories["img"])
                basename = Path(filename).stem
                written_paths = []
                try:
                    target_image = self.tensor_to_pil(target, batch_index)
                    target_path = directories["img"] / filename
                    self.save_png_atomic(target_image, target_path)
                    written_paths.append(target_path)

                    controls = {
                        "Control1": control1,
                        "Control2": control2,
                        "Control3": control3,
                    }
                    for folder_name, control in controls.items():
                        control_image = None
                        if control is not None:
                            control_image = self.tensor_to_pil(control, batch_index)
                        elif folder_name == "Control1" and not has_control:
                            control_image = self.create_black_image(target_image.size)
                        if control_image is None:
                            continue
                        if control_image.size != target_image.size:
                            raise ValueError(f"{folder_name} size does not match target size")
                        control_path = directories[folder_name] / filename
                        self.save_png_atomic(control_image, control_path)
                        written_paths.append(control_path)

                    if caption and caption.strip():
                        caption_path = directories["img"] / f"{basename}.txt"
                        temp_caption = caption_path.with_name(
                            f'.{caption_path.name}.{uuid.uuid4().hex}.tmp'
                        )
                        try:
                            temp_caption.write_text(caption.strip(), encoding='utf-8')
                            os.replace(temp_caption, caption_path)
                        finally:
                            if temp_caption.exists():
                                temp_caption.unlink()
                        written_paths.append(caption_path)

                    saved_entries.append(filename)
                except Exception:
                    for path in written_paths:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
                    raise

        print(f"✅ Saved {len(saved_entries)} dataset entr{'y' if len(saved_entries) == 1 else 'ies'}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Files: {', '.join(saved_entries)}")
        
        return ()


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenDatasetSaver": QwenDatasetSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenDatasetSaver": "Qwen Dataset Saver"
}
