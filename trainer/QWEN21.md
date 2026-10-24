# Qwen Image 2.1

Select **Qwen Image 2.1** in Trainer. The same `qwen_image_2` architecture
trains generation from captions when the dataset has no Controls, or editing
when paired Control1–3 images exist. Every used Control folder must have a
matching image for each target. **Align reference dimensions (padding)** defaults
on in Trainer: each Control is fitted to the target bucket canvas, centered,
without cropping or stretching. RGB borders are white; RGBA borders are transparent.
This allows larger batches with different source dimensions and aspect ratios,
provided every sample has the same number of references. Targets still use normal
resolution buckets; source files are not modified. The same fitting rule is used
for text embeddings, VAE references, and preview sampling.
When alignment is off, references keep their own aspect ratio and larger batches
require compatible reference token counts and shapes. **Batch size override: 1**
remains available for individual datasets. An empty override inherits the global
batch size. Gradient accumulation remains a global setting.

Enable **Transparency (RGBA)** to preserve alpha in targets, references and PNG
samples. Mixed RGB/RGBA datasets work: RGB images receive opaque alpha. This
uses the model's native 64-channel RGBA VAE, without an external RGBA VAE or
the older Transparent LoRA preset's generated controls/alpha auxiliary losses.
With the switch off, training and output are RGB. Native RGBA and `alpha_mask`
cannot be combined.

Defaults match the source UI: `Comfy-Org/Qwen-Image-2.1`, transformer and text
encoder `convrot8`, low VRAM enabled, FlowMatch, Shift timesteps, CFG 3, and
text encoder unloading disabled. With alignment off, Match target resolution
scales references to the target pixel area on a 32-pixel grid; turning both options
off uses the model's reference pixel cap. Alignment takes precedence over this
area-based sizing. Its text caches use a separate version and the target bucket
size, so old embeddings are not reused. RGBA uses separate latent/reference caches.
Existing jobs must be edited and saved with alignment enabled to use the new rule.

Prepared references bypass Qwen3-VL's automatic min/max-pixel resize, keeping the
vision slots on exactly the same 32-pixel canvas as the VAE. This includes small
panoramic buckets such as 448x128 (56 slots) and references above the processor's
usual pixel limit. Text cache versions also change so embeddings made with the
old processor resize are rebuilt automatically.

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
prompt sizing, real Qwen3-VL processor slot counts, panoramic batch forward/backward,
alpha compositing and convrot embedding import. These tests do
not download checkpoints; full-size GPU training is not covered by them.

## Совместная генерация PSD-слоёв

Отдельный пресет `qwen_layered_lora` реализует общий denoising стека RGBA-слоёв.
Формат PSD-target, optional Control1, ограничения и выдача PSD описаны в [QWEN21_LAYERS.md](QWEN21_LAYERS.md).

