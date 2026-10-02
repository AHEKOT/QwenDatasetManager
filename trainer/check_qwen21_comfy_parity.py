"""Compare actual QDM/ComfyUI implementations using tiny CPU models, no weights.

Run in trainer/.venv; --comfy-site-packages optionally supplies Comfy-only runtime
libraries from its environment, without installing or changing either environment.
"""
import argparse
import functools
import importlib.util
import json
from pathlib import Path
import sys
import types


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--comfy-site-packages', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'trainer/ai_toolkit'))
    sys.path.insert(0, str(args.comfy_root.resolve()))
    if args.comfy_site_packages:
        sys.path.append(str(args.comfy_site_packages.resolve()))
    sys.argv = [sys.argv[0], '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.ops
    import comfy.model_management as mm
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel as Native
    from comfy.ldm.qwen_image21.model import TimestepProjEmbeddings
    source = root / 'trainer/ai_toolkit/extensions_built_in/diffusion_models/qwen_image_2/src'
    Train = load_module('qdm_audit_transformer', source / 'transformer.py').QwenImage21Transformer2DModel
    pipeline = load_module('qdm_audit_pipeline', source / 'pipeline.py')
    core = load_module('qdm_audit_core', root / 'ComfyUI-QDM-QI2-Layers/core.py')
    torch.set_num_threads(2)
    torch.manual_seed(21)
    kwargs = dict(in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16,
                  num_attention_heads=2, context_in_dim=8, axes_dims_rope=(4, 6, 6), mlp_ratio=2)
    train = Train(**kwargs).eval()
    native = Native(**kwargs, dtype=torch.float32, device='cpu', operations=comfy.ops.disable_weight_init).eval()
    native.load_state_dict(train.state_dict(), strict=True)
    rows = []
    for mode in ('fp32', 'legacy_bf16'):
        train.time_text_embed.time_proj.legacy_bf16 = mode == 'legacy_bf16'
        fn = core.legacy_timestep_forward if mode == 'legacy_bf16' else TimestepProjEmbeddings.forward
        native.time_text_embed.forward = types.MethodType(fn, native.time_text_embed)
        for dtype in (torch.float32, torch.bfloat16):
            train.to(dtype)
            native.to(dtype)
            for slots in (1, 4, 20):
                native.build_sequence = types.MethodType(functools.partial(core.joint_build_sequence, layer_slots=slots), native)
                for reference in (False, True):
                    x = torch.randn(1, 64, slots*2, 4).to(dtype)
                    prompt = torch.randn(1, 4, 8).to(dtype)
                    pmask = torch.ones(1, 4, dtype=torch.bool)
                    smask = torch.tensor([[False, reference, False, False]])
                    ref = torch.randn(1, 64, 2, 2).to(dtype)
                    for sigma in (.02, .137, .5, 1.):
                        t = torch.tensor([sigma])
                        tt = ((t*1000).to(dtype)/1000).to(dtype)
                        with torch.no_grad():
                            expected = pipeline.run_transformer(train, x, tt, prompt, pmask, smask,
                                condition_latents=pipeline.pack_latents(ref) if reference else None,
                                condition_shapes=[(2, 2)] if reference else [], num_target_images=slots)
                            for optimized in (False, True):
                                mm.in_training = not optimized
                                actual = native(x, t, context=prompt[:, ~smask[0]],
                                    ref_latents=[ref] if reference else [], image_slots=[1] if reference else [],
                                    transformer_options={})
                                diff = (actual.float()-expected.float()).abs()
                                rows.append(dict(mode=mode, dtype=str(dtype), slots=slots, reference=reference,
                                    sigma=sigma, optimized=optimized, max_error=diff.max().item(), mean_error=diff.mean().item()))
                                tol = .012 if dtype == torch.bfloat16 else 2e-4
                                torch.testing.assert_close(actual.float(), expected.float(), atol=tol, rtol=tol)
    mm.in_training = False
    result = dict(cases=len(rows), max_error=max(r['max_error'] for r in rows),
                  max_mean_error=max(r['mean_error'] for r in rows), cases_detail=rows)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases_detail'}, indent=2))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
