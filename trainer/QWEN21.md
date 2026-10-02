# Qwen Image 2.1

Select **Qwen Image 2.1** in Trainer. The same `qwen_image_2` architecture
trains generation from captions when the dataset has no Controls, or editing
when paired Control1–3 images exist. Every used Control folder must have a
matching image for each target. Start with batch size 1; larger batches require
compatible reference token counts and shapes.
For a dataset with varying reference sizes, set its **Batch size override** to
**1** while other datasets keep the global batch size. An empty override inherits
the training batch size. Gradient accumulation remains a global setting.

Enable **Transparency (RGBA)** to preserve alpha in targets, references and PNG
samples. Mixed RGB/RGBA datasets work: RGB images receive opaque alpha. This
uses the model's native 64-channel RGBA VAE, without an external RGBA VAE or
the older Transparent LoRA preset's generated controls/alpha auxiliary losses.
With the switch off, training and output are RGB. Native RGBA and `alpha_mask`
cannot be combined.

Defaults match the source UI: `Comfy-Org/Qwen-Image-2.1`, transformer and text
encoder `convrot8`, low VRAM enabled, FlowMatch, Shift timesteps, CFG 3, and
text encoder unloading disabled. Match target resolution defaults on: references
keep their aspect ratio and are scaled to the target pixel area on a 32-pixel
grid. Turning it off uses the model's reference pixel cap. Text caches include
this rule and the target bucket size; RGBA uses separate latent/reference caches.

The loader reuses/downloads weights in the existing `models/` Comfy layout:

- `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` (or BF16 variant)
- `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (or BF16 variant)
- `vae/qwen_image_2.1_vae_bf16.safetensors`

Configs and processor files come from `Qwen/Qwen-Image-2.1`. A local complete
checkpoint directory or transformer file can be entered in the model path.
Existing trainer dependency pins already match the source checkout.

## Source and local integration

Copied from `D:\AiToolkitNew\AI-Toolkit`, revision
`ecee894ed2b1f3716d9d7326693061ec1a3105bb`, including the Qwen 2.1 commits
`c2622ed`, `f4f5d5d`, `086b663`, and `07abdbe`. Vendored transformer/VAE files
retain their Apache 2.0 notices. Shared prerequisites include convrot embedding
loading, reference sizing during live/cached prompt encoding and sampling,
RGBA dataloading/augmentation, cache separation, and alpha-safe sample format.
Existing QDM model adapters and local memory-management changes are retained.

## Verification

`tests/test_trainer_qwen21.py` and `tests/test_trainer_frontend.py` cover model
defaults, generation without references, paired-reference validation, native
RGBA and Advanced config handling. `trainer/test_qwen21_runtime.py` runs with
`trainer/.venv` and covers a small real transformer forward/backward for
generation and editing, flow targets, RGB/RGBA dataloading and cache identities,
prompt sizing, alpha compositing and convrot embedding import. These tests do
not download checkpoints; full-size GPU training is not covered by them.

## Совместная генерация PSD-слоёв

Отдельный пресет `qwen_layered_lora` реализует общий denoising стека RGBA-слоёв.
Формат PSD-target, optional Control1, ограничения и выдача PSD описаны в [QWEN21_LAYERS.md](QWEN21_LAYERS.md).

