# Review of the previous ChromakeyX_v4 run

Reviewed 2026-09-10, after the user corrected a mistaken attribution of their observations to CleanMatte. This review was missing when the replacement architecture was selected. It examines saved outputs, run configuration and recorded metrics, without reading or importing the legacy training/model implementation.

## Evidence inspected

- `trainer/output/ChromakeyX_v4/.job_config.json`
- `trainer/output/ChromakeyX_v4/validation_metrics.jsonl`
- Saved comparison sheets: example 0 at steps 250, 1000 and 2750; example 5 at steps 250, 1000 and 2750; examples 1 and 10 at step 2750.
- Saved RGBA PNGs for example 5 at steps 250, 500, 750, 1000, 1500 and 2750.

This is a targeted review, not an inspection of every example or previous run. Uploaded validation examples have no reference alpha in the inspected artifacts. Their visible defects must not be confused with the separate synthetic holdout metrics.

## Confirmed observations

Example 5 is the character wearing a top hat, blue clothing and red trousers on a bright green background. At step 250 the matte is spatially blurred, retains background and makes large parts of the subject translucent. At step 1000 distant background is almost gone, but broad green remnants persist between strands and along the silhouette. The outline is substantially sharper at 2750.

The RGBA alpha was measured in a visibly empty background strip: x from 0 to floor(0.08 W), y from floor(0.1 H) to floor(0.9 H). W=H=1536. Values below are normalized from the saved 8-bit alpha, not model tensors or a whole-background ground-truth mask.

| Step | Mean alpha | Maximum alpha | Fraction with alpha > 0.02 |
| --- | --- | --- | --- |
| 250 | 0.22769 | 0.38824 | 1.00000 |
| 500 | 0.08697 | 0.23529 | 1.00000 |
| 750 | 0.00732 | 0.06275 | 0.12815 |
| 1000 | 0.00062 | 0.01961 | 0.00000 |
| 1500 | 0 | 0 | 0 |
| 2750 | 0 | 0 | 0 |

This supports the user's report that even obvious background took approximately 1000 steps to become almost transparent in this example. Exact zero throughout this strip is observed at the inspected step 1500, not at 1000.

Example 0 has green hair on a blue background. At step 1000 large holes appear inside hair and clothing. By step 2750 much of the green hair is removed despite the blue screen; the dark strand outlines remain. Thus improving outline sharpness does not establish correct foreground preservation. A rule that removes both green and blue indiscriminately would also fail this example.

Example 1 at step 2750 still has incomplete separation around the bent arm and hair. Example 10 has visible narrow defects inside the foreground matte. These require native-resolution inspection and reference alpha for quantitative scoring.

Recorded synthetic holdout metrics improve unevenly: alpha MAE is approximately 0.26 at 250, 0.09 at 1000 and 0.07 at 2750; thin-region error is approximately 0.37, 0.31 and 0.23. These are legacy metric values whose implementation was not audited. They are not directly comparable to CleanMatte metrics.

## What configuration establishes, and what remains hypothetical

The saved configuration specifies 1000 warmup steps, EMA decay 0.995, 40% green / 30% blue / 15% white / 15% black backgrounds, numerous RGB corruptions, and simultaneous alpha, foreground, despill and compositing objectives. The training source is TransparentVAE, whereas the replacement diagnostic used ChromaKeyX_New. Those experiments are not controlled comparisons.

The warmup duration coincides with the observed early period, but causality cannot be established from this alone. The actual learning-rate and EMA behavior, active augmentation schedule, loss normalization and gradient contributions were not verified. Loss coefficients alone do not establish which objective dominated training. Nor do blurred alpha images prove a particular internal segmentation architecture or insufficient parameter count.

## Requirements for the replacement evaluation

1. Evaluate the same uploaded examples, with early checkpoints and separate alpha/RGB inspection. Preserve their identity across runs.
2. Measure obvious background leakage independently from foreground opacity, internal gaps, boundary accuracy and thin-strand preservation.
3. Include the green-haired character on a blue screen as a required color-preservation case; keying must distinguish the selected screen from foreground colors.
4. Compare any analytic initialization and learned refinement against a plain keying baseline on the same images. Treat an immediate basic matte as an explicit requirement rather than assuming an untrained RGB network provides it.
5. Use labeled held-out examples to quantify accuracy; visual outputs without reference alpha cannot establish subpixel correctness.
6. Establish causes through controlled experiments before attributing failures to capacity, warmup, EMA, color objectives or architecture. The 82k-parameter replacement is not validated by this retrospective review.
