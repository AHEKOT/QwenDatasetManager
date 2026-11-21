# QDM CleanMatte ComfyUI nodes

The current package uses the independent **CleanMatte** alpha-first model.
Existing V1–V4 ChromaKey checkpoints are incompatible with these nodes.

1. Copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.
2. Place a new `cleanmatte_best.safetensors` in `ComfyUI/models/cleanmatte/`.
3. Add **QDM Load CleanMatte** and **QDM CleanMatte (alpha first)**.
4. Connect IMAGE and model. Outputs: foreground RGB, foreground alpha MASK
   (`1` means foreground), and RGBA IMAGE. Use an RGBA-aware saver for alpha.

CPU and CUDA are supported. Default tiles are 512px and preserve the input
size, with a halo to avoid seams. Every native pixel is refined, including
internal gaps missed by the coarse context network. No heavy backbone or
external segmentation model is loaded. The network has 82,287 parameters.

Despill defaults to 0 so that alpha can be inspected independently. Increasing
it enables separate screen subtraction and green/blue chroma suppression on
semi-transparent edges. It never changes alpha or fully opaque foreground RGB.
Automatic screen colour estimation requires at least 16 predicted background
pixels; otherwise colour correction is skipped for that image. This simple
colour recovery is not a guarantee for coloured lighting or exact foreground/
background colour collisions.

The new trainer is documented in
[`cleanmatte_training/README.md`](../trainer/ai_toolkit/extensions/cleanmatte_training/README.md).
Its latest and best safetensors exports include an explicit architecture tag.
Legacy source files are retained for reference but are not imported by the new
nodes or the independent new training runner.
