# AI Toolkit trainer parity

Parity source: `ostris/ai-toolkit` commit
`8a912564ce60047ea44d0f3a98becf3f168d3094`.

The comparison is intentionally limited to the `diffusion_trainer` (LoRA
Trainer) Simple UI for these architecture entries:

- `qwen_image_edit_plus:2511`
- `flux2_klein_4b`
- `flux2_klein_9b`

QDM additionally exposes opt-in local presets that do not replace those
upstream entries:

- `qwen_image_edit_plus_rgba`, `flux2_klein_4b_rgba`, and
  `flux2_klein_9b_rgba` for transparent-target LoRA training;
- `qwen_rgba_vae_trainer` for Qwen four-channel VAE compatibility training;
- `flux2_rgba_vae_trainer` for the shared FLUX.2 Klein 4B/9B native z=32
  four-channel VAE.

## Forwarded settings

| AI Toolkit section | Forwarded settings |
|---|---|
| Job | name, GPU, trigger word |
| Model | architecture, editable Hugging Face name/local path, gated-model link/token, low VRAM, match target resolution, optional sampling-only Turbo LoRA |
| Quantize / Compile | complete upstream transformer qtype list, Qwen 2511 ARA, complete text-encoder qtype list, compile + `block_compile` |
| Layer offloading | enable, transformer percentage, text-encoder percentage |
| Target | LoRA/LoKr, linear rank, LoKr factor |
| Save | dtype, save cadence, retained checkpoint count |
| Training | batch size, gradient accumulation, steps, all upstream optimizers, learning rate, weight decay, timestep type, timestep bias, loss type |
| EMA / TE | EMA + decay, unload text encoder where the architecture allows it, cache text embeddings |
| Regularization | Differential Output Preservation + multiplier/class, Blank Prompt Preservation + multiplier, Contrastive Guidance Loss + target |
| Advanced | differential guidance + scale |
| Validation | cadence, resolution, sigmas, managed validation image/prompt list |
| Dataset | target/control mapping, network weight, repeats, per-dataset batch size, default caption, dropout, caption extension, cache latents, regularization flag, flips, complete resolution list |
| Sampling | cadence/start, FlowMatch/DDPM sampler, guidance, steps, size, seed/walk, skip/force/disable, edit-instruction list, per-sample size/seed/network multiplier, independently uploaded `ctrl_img_1`–`ctrl_img_3` |
| Runtime | queue, stop, save now, sample now, progress, speed and logs |

## Local RGBA preset behavior

- RGBA targets retain their alpha channel through crop, resize, latent cache
  identity, decode and PNG output. Hidden RGB below the alpha threshold is
  cleared and chroma-key spill cleanup is selectable.
- Each transparent dataset visibly selects `edit` or `generation`. Edit uses
  paired Control1–3 when present and otherwise makes an RGB composite;
  generation ignores paired controls and creates an opaque black Control1.
- The Qwen preset accepts the QIE2511 Lightning sampling LoRA, keeps it inactive
  during training, and forces its intended four steps / CFG 1 during previews.
  Klein 4B/9B expose the same sampling-only path and loader; step/CFG values
  remain user-selected because no Klein Turbo checkpoint is bundled.
- Qwen transparent training requires a Qwen z=16 RGBA VAE. Klein transparent
  training requires a separately trained FLUX.2 z=32 RGBA VAE; cross-loading
  either family is rejected before training.
- Qwen VAE readiness uses deterministic RGBA round trips instead of diffusion
  sampling. The queue's “Sample now” action runs validation, “Save now” writes
  a checkpoint, and Stop is polled directly by the VAE process.
| Advanced editor | full process JSON override with model architecture, CUDA device and managed dataset paths locked to the selected QDM job |

## Deliberate model-scoped behavior

- Training noise scheduler is fixed to `flowmatch`, exactly as the three model
  defaults in upstream `options.tsx`; it is displayed read-only in the Simple UI.
- Qwen Image Edit 2511 disables `train.unload_text_encoder`, matching upstream.
  Klein 4B/9B expose it.
- `network.conv` is disabled for all three architectures upstream and is not
  shown here.
- Concept Slider, video/audio controls and unrelated model-specific sections are
  outside this integration.
- AI Toolkit target/control folder selectors are replaced by QDM's managed
  dataset selector. `img` maps to the target and matching `Control1`–`Control3`
  folders map to multi-control inputs.
