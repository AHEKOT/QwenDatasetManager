# Vendored AI Toolkit trainer

The CUDA training backend in `ai_toolkit/` is a restricted source copy of
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) at commit:

`8a912564ce60047ea44d0f3a98becf3f168d3094`

Upstream is MIT licensed. Its original `LICENSE` is preserved in
`ai_toolkit/LICENSE`.

## Included scope

- shared `toolkit/` training core;
- `jobs/` with the extension job entry point;
- `extensions_built_in/sd_trainer/`;
- Qwen Image Edit Plus model adapter;
- FLUX.2 Klein 4B and 9B model adapters.

The corresponding Simple UI field audit and intentional architecture limits
are recorded in [`PARITY.md`](PARITY.md). The QDM screen replaces only the
upstream folder pickers with managed dataset selection; the applicable trainer
settings continue to produce the same process keys consumed by this vendored
backend.

The upstream Next.js UI, manager, dataset editor, captioning extensions and
unrelated image/video/audio model adapters are intentionally not included.
Qwen Dataset Manager supplies the UI, job queue, dataset mapping and settings.

## Local integration changes

- the model registry exposes only Qwen Image Edit Plus and FLUX.2 Klein;
- built-in legacy model imports were removed from `toolkit/util/get_model.py`;
- `jobs/__init__.py` imports only `BaseJob` and `ExtensionJob`.

Training math, dataloading, quantization, model loading, LoRA creation,
checkpoint saving and sampling remain upstream implementations.
