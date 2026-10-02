import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))
import torch
from toolkit.memory_management import MemoryManager


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class PhaseOffloadTests(unittest.TestCase):
    def test_float8_resident_storage_survives_cpu_cuda_roundtrip(self):
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
        device = torch.device('cuda:0')
        module = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)).to(dtype=torch.bfloat16)
        module.requires_grad_(False)
        quantize_(module, Float8WeightOnlyConfig())
        with patch('toolkit.memory_management.manager.random.random', side_effect=[0.1, 0.9]):
            MemoryManager.attach(module, device, offload_percent=.5)
        x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            expected = module(x)
            for _ in range(2):
                module.to('cpu')
                self.assertEqual(module[1].weight.device.type, 'cpu')
                self.assertEqual(module[1].weight.dequantize().device.type, 'cpu')
                module.to(device)
                self.assertEqual(module[1].weight.device.type, 'cuda')
                self.assertEqual(module[1].weight.dequantize().device.type, 'cuda')
                torch.testing.assert_close(module(x), expected)
        MemoryManager.detach(module)

    def test_partial_offload_parks_resident_weights_and_restores_forward_and_gradients(self):
        device = torch.device('cuda:0')
        module = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LayerNorm(64), torch.nn.Linear(64, 64))
        module.register_buffer('container_buffer', torch.ones(4))
        module.requires_grad_(False)
        x = torch.randn(2, 64, device=device, requires_grad=True)
        module.to(device)
        expected = module(x)
        expected_grad, = torch.autograd.grad(expected.sum(), x)
        # One managed linear, one resident linear, plus resident norm/buffer.
        with patch('toolkit.memory_management.manager.random.random', side_effect=[0.1, 0.9]):
            MemoryManager.attach(module, device, offload_percent=.5)
        self.assertEqual(module[0].weight.device.type, 'cpu')
        self.assertEqual(module[2].weight.device.type, 'cuda')
        for _ in range(2):
            module.to('cpu')
            self.assertEqual(module.device.type, 'cpu')
            self.assertTrue(all(p.device.type == 'cpu' for p in module.parameters()))
            self.assertEqual(module.container_buffer.device.type, 'cpu')
            module.to(device)
            self.assertEqual(module.device, device)
            self.assertEqual(module[0].weight.device.type, 'cpu')
            self.assertEqual(module[2].weight.device.type, 'cuda')
            self.assertEqual(module.container_buffer.device.type, 'cuda')
            actual = module(x)
            actual_grad, = torch.autograd.grad(actual.sum(), x)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_grad, expected_grad)
        MemoryManager.detach(module)


if __name__ == '__main__':
    unittest.main()
