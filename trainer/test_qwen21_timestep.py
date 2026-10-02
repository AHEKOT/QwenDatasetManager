"""CPU checks for timestep precision, without weights or model downloads."""
import ast
import importlib.util
import math
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'trainer/ai_toolkit/extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenImage21TemporalTimesteps')
scope = {'torch': torch, 'nn': torch.nn, 'math': math}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), scope)
Timesteps = scope['QwenImage21TemporalTimesteps']
spec = importlib.util.spec_from_file_location('qdm_layer_core', ROOT / 'ComfyUI-QDM-QI2-Layers/core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class TimestepPrecisionTests(unittest.TestCase):
    def test_model_cast_does_not_round_frequency_table(self):
        t = torch.tensor([0., .02, .137, .5, 1.], dtype=torch.bfloat16)
        module = Timesteps(256)
        expected = module(t)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            module.to(dtype)
            torch.testing.assert_close(module(t), expected, rtol=0, atol=0)

    def test_native_fp32_frequency_formula(self):
        t = torch.tensor([.02, .137, .5, 1.], dtype=torch.bfloat16)
        freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
        args = (t.float() * 1000)[:, None] * freqs[None]
        expected = torch.cat((args.cos(), args.sin()), dim=-1)
        torch.testing.assert_close(Timesteps(256).to(torch.bfloat16)(t), expected, rtol=0, atol=0)

    def test_legacy_comfy_path_matches_pre_fix_training(self):
        t = torch.tensor([0., .02, .137, .5, 1.], dtype=torch.bfloat16)
        module = Timesteps(256).to(torch.bfloat16)
        module.legacy_bf16 = True
        freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).bfloat16()
        args = (t.float() * 1000)[:, None] * freqs[None]
        expected = torch.cat((args.cos(), args.sin()), dim=-1)
        torch.testing.assert_close(module(t), expected, rtol=0, atol=0)
        projection = type('Probe', (), {'timestep_embedder': torch.nn.Identity()})()
        actual = core.legacy_timestep_forward(projection, t, torch.bfloat16)
        torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
        self.assertGreater((expected - Timesteps(256)(t)).abs().max().item(), 1.)


if __name__ == '__main__':
    unittest.main()
