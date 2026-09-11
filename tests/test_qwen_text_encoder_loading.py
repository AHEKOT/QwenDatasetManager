"""Exercise encoder placement without allocating the multi-billion weight model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'trainer/ai_toolkit/extensions_built_in/diffusion_models/qwen_image/qwen_image.py'


class QwenEncoderLoadingTests(unittest.TestCase):
    def run_placement(self, quantized, offload):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenImageModel')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_text_encoder')
        events = []
        encoder = SimpleNamespace(frozen=False, managed=False)

        def quantize(model, weights):
            self.assertIs(model, encoder)
            self.assertEqual(weights, 'qfloat8')
            self.assertFalse(model.managed, 'Offloading must attach to the replaced quantized layers')
            events.append('quantize')

        def freeze(model):
            model.frozen = True
            events.append('freeze')

        def attach(model, device, offload_percent):
            self.assertEqual(offload_percent, offload)
            model.managed = True
            events.append('offload')

        def move(device, dtype):
            self.assertEqual((device, dtype), ('cuda:0', 'bf16'))
            if quantized:
                self.assertTrue(encoder.frozen, 'Full precision encoder would exhaust CUDA before quantization')
            if offload:
                self.assertTrue(encoder.managed)
            events.append('cuda')

        encoder.to = move
        scope = dict(quantize=quantize, freeze=freeze, get_qtype=lambda q: q,
                     flush=lambda: None, MemoryManager=SimpleNamespace(attach=attach))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)
        model = SimpleNamespace(
            model_config=SimpleNamespace(quantize_te=quantized, qtype_te='qfloat8',
                                         layer_offloading=bool(offload),
                                         layer_offloading_text_encoder_percent=offload),
            device_torch='cuda:0', print_and_status_update=lambda message: None)
        scope['_prepare_text_encoder'](model, encoder, 'bf16')
        return events

    def test_quantized_encoder_frozen_before_cuda(self):
        self.assertEqual(self.run_placement(True, 0), ['quantize', 'freeze', 'cuda'])

    def test_offloading_manages_final_quantized_layers(self):
        self.assertEqual(self.run_placement(True, .75), ['quantize', 'freeze', 'offload', 'cuda'])

    def test_full_precision_offloading_is_preserved(self):
        self.assertEqual(self.run_placement(False, 1), ['offload', 'cuda'])

    def test_does_not_silently_enable_quantization(self):
        self.assertEqual(self.run_placement(False, 0), ['cuda'])


if __name__ == '__main__':
    unittest.main()
