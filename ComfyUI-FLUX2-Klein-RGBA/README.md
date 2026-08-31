# ComfyUI FLUX.2 Klein RGBA

A dedicated ComfyUI node package for loading four-channel FLUX.2 Klein VAEs
trained by Qwen Dataset Manager.

## Installation

Copy this directory to:

```text
ComfyUI/custom_nodes/ComfyUI-FLUX2-Klein-RGBA
```

Restart ComfyUI and use **Load FLUX.2 Klein RGBA VAE** from
`loaders/FLUX.2 Klein RGBA` instead of ComfyUI's standard **Load VAE** node.

The package does not modify, wrap, or monkeypatch the standard loader and is
independent of `comfyui_qwenDatasetManager`.

Supported checkpoints:

- regular float32 `ae.safetensors` checkpoints from the Klein RGBA trainer;
- optional `*_ComfyUI_bf16.safetensors` exports;
- full and small-decoder FLUX.2 z=32 VAE layouts.

For transparent LoRA training, use the float32 master checkpoint at
`QwenDatasetManager/models/vae/flux2-klein-rgba.safetensors`. Both Klein 4B
and 9B trainer presets select that path automatically. The bfloat16 export is
recommended for ComfyUI inference.
