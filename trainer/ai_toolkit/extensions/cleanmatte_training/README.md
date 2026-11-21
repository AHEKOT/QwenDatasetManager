# CleanMatte: independent alpha-first training

This replaces the active ChromaKey training path. The old model, trainer,
data generator and objectives were not read or reused for this implementation.
Only the application's queue/configuration/UI interfaces were inspected.

## Architecture and research

`ComfyUI-QDM-ChromaKey/cleanmatte/model.py` contains **82,287 parameters**:

- A small encoder/decoder evaluates global RGB context at at most 384 pixels
  on the long side. Its output contains BG/FG/mixed classification and eight
  guidance channels.
- A native-resolution convolutional refiner sees RGB and this guidance. It
  predicts its own three-class labels and fractional coverage. It can override
  the coarse labels anywhere, including small internal holes.
- BG is exported with alpha 0, opaque FG with alpha 1, mixed pixels with
  learned coverage. Class probabilities are not exported as opacity.
- Coverage receives supervision even when classification is wrong. There is
  no learned foreground-colour output, shared despill branch or RGB loss.

The task decomposition is motivated by
[AdaMatting](https://arxiv.org/abs/1909.04686) and
[Highly Efficient Natural Image Matting](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1642.pdf).
The latter demonstrates a 344k-parameter segmentation/refinement architecture.
**Our convolutional implementation is an adaptation, not a reproduction of its
OCBlock/ENA architecture, weights or published accuracy.** Depthwise/pointwise
convolutions and a narrow native branch were chosen for CPU/weak GPU deployment.

[AIRMatting](https://mdpi-res.com/d_attachment/symmetry/symmetry-13-01454/article_deploy/symmetry-13-01454-v2.pdf)
is a useful chromakey-specific comparison, but adds per-image model fitting.
[PP-MattingV2](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/Matting)
is an external quality baseline, with a substantially larger model budget.
Batch normalization stabilizes training; its statistics are carried into EMA
and folded into convolutions on export loading, avoiding a deployment cost.

No claim of superiority over these methods is made without a matched dataset.

## Clean-label contract

Use RGBA images with independently clean alpha. Input RGB is composed from the
same foreground and coverage. Background gradients, sensor-like noise, JPEG
and colour spill affect RGB only. They never add speckle to the target alpha.
Full-object resampling uses premultiplied colour; native crops preserve the
source pixel scale. Captions and Control directories are not used.

20% of training examples are analytic coverage fixtures rendered at 4x sample
resolution: a smooth body, attached subpixel/1–2px strands and real holes.
They supplement real RGBA subjects; they do not certify real-world quality.
Random erosion, alpha noise, whole-mask binarization and hole filling are not
part of target generation.

The engine audits all PNG/WebP sources, rejects missing/constant alpha and
groups exact decoded-pixel duplicates before reserving a fixed 10% holdout.
`dataset_manifest.json` records the split, hashes, duplicate groups and alpha
fractions. Changed sources invalidate resume. The audit is not an automatic
assessment of whether an artist's intended contour is correctly annotated.

## Default training

Default screen distribution: 70% green, 30% blue. White/black screens are
opt-in and default to zero: white detail on a white screen or black linework
on a black screen can make alpha underdetermined from RGB. Those cases require
separate quality assessment and must not silently dominate chromakey training.

- Adam, LR `4e-4`, betas `(0.5, 0.999)`; 1,000 warmup steps and cosine decay
  with a 10% LR floor. Default run length: 50,000 optimizer steps.
- Start with 512px crops, maximum batch 4, a 1 MP activation budget and
  accumulation 3 on RTX 5070 Ti 16 GB. The budget reduces the batch at larger
  crop sizes and when paired views are enabled. It is not a VRAM guarantee.
- BF16 on CUDA; objective evaluation in FP32. FP16 uses gradient scaling.
- Clean RGB for at least 10,000 steps. Corruption and paired backgrounds
  start only after clean holdout alpha passes the curriculum gate, then ramp
  over 30% of the configured run. Gate limits: alpha MAE 0.025; background,
  opaque and hole errors 0.01; mixed and thin errors 0.1. These are initial
  curriculum thresholds, not a production-quality certification. A short
  run may finish entirely in the clean phase.
- Region-balanced alpha L1 (weight 5), three-class CE (1), target-gradient
  matching (0.5), paired-background consistency (0.25). Alpha and CE each
  include an additional contour-band mean (coefficient 0.5), so opaque 1px
  hairs and narrow BG gaps are not diluted by the large FG/BG interiors.
- 70% native detail crops, 30% full-object examples for the real-source branch.
- EMA, finite-loss/gradient checks, validation every 250 steps, save every 500.

Select **QDM CleanMatte — Alpha First** in the trainer and create a new job.
The existing queue launches `run_cleanmatte.py` directly, without importing
legacy chromakey or diffusion processes. Stop, Save now and Sample now use
the same SQLite request protocol as the existing interface.

For a saved job JSON, from `trainer/ai_toolkit`:

```powershell
& ../.venv/Scripts/python.exe -u run_cleanmatte.py path/to/job.json
```

## Validation and outputs

Fixed full-object and native crops from up to 12 holdout sources are evaluated
with fixed RGB corruption. Reported values include alpha MAE, background alpha
leakage, foreground error, mixed-pixel error, thin-support error/recall and
background false positives and internal-hole leakage. The score used for best export combines regional
alpha errors only. There is no colour metric that can compensate for bad alpha.

Uploaded RGB images are inference-only visual checks. Lossless PNG sheets show
input and alpha; synthetic holdout sheets also show target alpha. No despill is
applied to validation. Metrics appear in the UI as **MEASURED**, not a fabricated
PASS or an assertion of production readiness.

- `cleanmatte_best.safetensors`: best holdout alpha score, primary download.
- `cleanmatte_latest.safetensors`: latest EMA export.
- `training_state.pt`: model, EMA, Adam, scaler, step and best score for resume.
- `training_history.jsonl`, `validation_history.jsonl`: measurements.
- `cleanmatte_validation.json`: latest values displayed by the GUI.
- `artifacts.json`: versioned model download manifest.

Checkpoint writes are atomic. Old keyer weights cannot be loaded into this
architecture. Existing old job artifacts are not removed.

## Deployment and checks

The portable runtime evaluates every original pixel using overlapping tiles
with a discarded halo. It does not restrict refinement to a coarse unknown
band, fill holes or retain only the largest component. Odd-sized images are
padded consistently to preserve the stride phase. CPU runtime uses channels-last
convolutions. Optional deterministic screen subtraction/chroma suppression is
separate, cannot modify alpha, preserves opaque RGB and defaults to disabled.

Run tests from `trainer/ai_toolkit`:

```powershell
& ../.venv/Scripts/python.exe -m unittest extensions.cleanmatte_training.test_cleanmatte -v
& ../.venv/Scripts/python.exe -m extensions.cleanmatte_training.verify --output ../output/CleanMatte_verification --steps 1500
```

Run speed benchmarks from `ComfyUI-QDM-ChromaKey`:

```powershell
& ../trainer/.venv/Scripts/python.exe -m cleanmatte.benchmark --device cpu --output ../trainer/output/cpu.json
& ../trainer/.venv/Scripts/python.exe -m cleanmatte.benchmark --device cuda --output ../trainer/output/gpu.json
```

Single-fixture fitting tests whether the training signal works. A short native
BF16 job tests execution, checkpointing and validation. Neither is evidence
that a production model has already learned the user's difficult subjects.
Read actual benchmark JSON files; do not extrapolate speed on a 5070 Ti to a
weak GPU. CPU and GPU support does not imply real-time performance everywhere.
