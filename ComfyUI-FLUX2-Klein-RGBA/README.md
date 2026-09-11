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

## MiniMax H3 / Ref2VA

The same package also provides **H3 RGBA VAE Loader (QDM)**,
**H3 RGBA Encode (QDM)** and **H3 RGBA Decode (QDM)** under
`QDM/Transparency/H3`. Requires ComfyUI with native MiniMax H3 support.

Copy the trained `minimax_h3_rgba_vae.safetensors` or its Comfy BF16 export
to `ComfyUI/models/vae`. Stock VAE auto-detection creates RGB boundaries;
use the dedicated H3 loader for RGBA. Its standard `VAE` output works with
the native H3 first-frame/reference conditioning nodes and standard VAE decode.
RGB references are padded opaque. Latent channels remain 24.

The dedicated decoder accepts video or joint H3 audio/video latents and
returns RGBA frames, RGB frames and a Comfy mask (`1 = transparent`). Audio
is not modified and can be decoded separately using the ordinary audio VAE.
The encoder accepts RGBA frames or RGB plus an optional Comfy mask.
Save RGBA frames as PNG/PNG sequences; MP4 export does not preserve alpha.

No global Comfy loader patch is installed. See [the H3 trainer guide](../trainer/H3.md)
for training, checkpoint selection and the current single-frame RGBA dataset scope.
