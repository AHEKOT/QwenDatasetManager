# QIE2511 transparent LoRA extension

This QDM port also registers `flux2_klein_4b_rgba` and
`flux2_klein_9b_rgba`. They reuse the same alpha-safe dataset and
sampling-only LoRA path, but require a FLUX.2-native four-channel VAE with
z=32. The Qwen z=16 VAE is not compatible and is rejected deliberately. No
Klein RGBA VAE checkpoint is bundled yet. The `flux2_rgba_vae_trainer`
process trains one shared native z=32 RGBA VAE for both Klein 4B and 9B and
saves `ae.safetensors` plus an optional BF16 ComfyUI file.

This opt-in model backend trains the existing AI Toolkit LoRA process against
the toolkit-trained `models/TransparentQIE2511VAE_diffusers` four-channel VAE.

In the job UI select **Qwen-Image-Edit-2511 Transparent (RGBA)**. The preset
sets that local Diffusers directory as `vae_path` and the backend validates
`input_channels=4` and `z_dim=16` before any dataset cache or training step.
The standard QIE VAE is not loaded by this architecture.

For fast workflow-like previews, set `sample_lora_path` to the QIE2511
Lightning 4-step LoRA. It is attached only inside sample generation and removed
in a `finally` block before training resumes. While it is configured, previews
are forced to 4 steps and CFG 1; the trainable LoRA remains active at its normal
sample `network_multiplier`.

## Dataset behavior

Set `pixel_channels: rgba` on the dataset. The loader then:

- requires a real alpha channel instead of silently converting to RGB;
- clears RGB values at or below `rgba_alpha_threshold` (default: 1/255), so a
  hidden green matte cannot enter the VAE latent;
- resizes in premultiplied-alpha space and converts back to straight RGBA,
  preventing green/dark resize fringes;
- keeps control/reference conditioning RGB-only;
- includes all RGBA preprocessing options and the selected RGBA VAE identity
  in cache namespaces.

If green is present only where alpha is zero, the default cleanup is sufficient.
If green was baked into partially transparent edge pixels, first run the audit.
For chroma-key cutouts, the safest correction is:

```yaml
rgba_edge_color_correction: matte_despill
rgba_edge_matte_color: [0, 255, 0]
rgba_edge_width: 3
```

It preserves alpha and replaces only matte-colored pixels in a narrow boundary
band with the nearest uncontaminated interior foreground color. A simpler
`nearest_opaque` mode is also available for partial-alpha-only contamination.
Only when the source is known to be mathematically precomposited over pure
green, use the alternative:

```yaml
rgba_unblend_background: [0, 255, 0]
```

The two correction modes are mutually exclusive. Inverse compositing is
intentionally not automatic because an inexact matte can create magenta fringes.

## Controls

QIE2511 is an edit model and requires an RGB control image. If no paired control
folder exists, use `rgba_generate_control: true`. The loader composites the
processed RGBA target over a deterministic background selected from
`rgba_control_backgrounds`. The generated control receives the same crop and
flip as the target.

## Commands

Run fast unit tests:

```powershell
python -m unittest discover -s extensions/rgba_training/tests -v
```

Audit a dataset before training:

```powershell
python extensions/rgba_training/scripts/audit_rgba_dataset.py D:\path\to\dataset
```

Run a real VAE reconstruction smoke test (downloads the VAE if it is not cached):

```powershell
python extensions/rgba_training/scripts/smoke_test_qwen_layered_vae.py `
  D:\path\to\dataset\example.png --device cuda `
  --output D:\path\to\qwen_layered_reconstruction.png
```

Export a trained RGBA VAE for ComfyUI's standard **Load VAE** node. The
standard native Qwen VAE is read only for its key layout; its weights are not
copied into the result:

```powershell
python extensions/rgba_training/scripts/export_qwen_rgba_vae_for_comfy.py `
  D:\path\to\checkpoint_diffusers `
  D:\ComfyUI\models\vae\qwen_image_vae.safetensors `
  D:\ComfyUI\models\vae\qwen_image_rgba_vae.safetensors
```

The exported file uses ComfyUI's native Wan/Qwen key layout, retains the
trained four-channel encoder and decoder boundaries, and loads through the
ordinary VAE selector without a custom node.

The RGBA VAE trainer also performs this export automatically by default. Every
`*_diffusers` checkpoint directory contains a matching
`*_ComfyUI_bf16.safetensors`; set `save.comfy_export: false` only when the
additional deployment file is not wanted.

The LoRA training example is at
`config/examples/train_lora_qwen_image_edit_2511_rgba.yaml`.
