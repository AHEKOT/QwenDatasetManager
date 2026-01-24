"""
Qwen Dataset Manager - ComfyUI Node
Saves images in Qwen training dataset format with automatic numbering.
Loads images from Qwen training dataset.
"""

from .qwen_dataset_saver import NODE_CLASS_MAPPINGS as SAVER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SAVER_DISPLAY
from .qwen_dataset_loader import NODE_CLASS_MAPPINGS as LOADER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY

NODE_CLASS_MAPPINGS = {**SAVER_MAPPINGS, **LOADER_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**SAVER_DISPLAY, **LOADER_DISPLAY}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
