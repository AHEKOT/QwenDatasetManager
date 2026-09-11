"""Comfy wrapper contracts with real tensors and a lightweight mock Comfy host."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch, MagicMock

import torch


class FakeWrapper:
    def __init__(self, sd):
        self.disable_offload, self.size = False, None
    def model_size(self):
        return 1
    def encode(self, pixels):
        self.last_pixels = pixels
        return torch.zeros(1, 24, 1, 2, 2)
    def decode(self, latent):
        return torch.full((1, 2, 32, 32, 4), .25)


class FakeCore(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.register_buffer('pixel_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1, 1))
    def load_state_dict(self, state, **kwargs):
        self.load_kwargs = kwargs


class H3ComfyRuntimeTests(unittest.TestCase):
    def setUp(self):
        comfy = ModuleType('comfy')
        comfy.sd = SimpleNamespace(VAE=FakeWrapper)
        comfy.ops = SimpleNamespace(disable_weight_init=object())
        comfy.utils = SimpleNamespace()
        comfy.model_patcher = SimpleNamespace(ModelPatcher=MagicMock())
        mm = SimpleNamespace(vae_device=lambda: torch.device('cpu'),
            vae_dtype=lambda *a: torch.float32, intermediate_device=lambda: torch.device('cpu'),
            vae_offload_device=lambda: torch.device('cpu'), dtype_size=lambda dt: 4)
        modules = {'comfy': comfy, 'comfy.sd': comfy.sd, 'comfy.ops': comfy.ops,
            'comfy.utils': comfy.utils, 'comfy.model_patcher': comfy.model_patcher,
            'comfy.model_management': mm, 'folder_paths': SimpleNamespace(),
            'comfy.ldm': ModuleType('comfy.ldm'), 'comfy.ldm.minimax': ModuleType('comfy.ldm.minimax'),
            'comfy.ldm.minimax.vae': SimpleNamespace(MiniMaxH3VideoVAE=FakeCore)}
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        path = Path(__file__).resolve().parents[1] / 'ComfyUI-FLUX2-Klein-RGBA' / 'h3_nodes.py'
        spec = importlib.util.spec_from_file_location('qdm_h3_nodes_test', path)
        self.nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.nodes)
        self.state = {k: torch.empty(shape, device='meta') for k, shape in {
            'encoder.conv_in.weight': (128, 4, 3, 3, 3),
            'decoder.proj_out.weight': (4096, 2048), 'decoder.proj_out.bias': (4096,),
            'post_quant_conv.weight': (24, 24, 1, 1, 1)}.items()}

    def test_wrapper_geometry_and_alpha_normalization(self):
        vae = self.nodes.H3RGBAVAE(self.state)
        self.assertEqual(vae.output_channels, 4)
        self.assertEqual(vae.latent_channels, 24)
        self.assertEqual(vae.first_stage_model.kwargs['in_channels'], 4)
        self.assertEqual(vae.first_stage_model.kwargs['out_ch'], 4)
        self.assertEqual(vae.first_stage_model.load_kwargs, {'strict': True, 'assign': True})
        self.assertEqual(vae.first_stage_model.pixel_std.flatten()[-1].item(), .5)
        for frames, latents in [(1, 1), (5, 2), (22, 7), (39, 12)]:
            self.assertEqual(vae.downscale_ratio[0](frames), latents)
            self.assertEqual(vae.upscale_ratio[0](latents), frames)
        self.assertEqual(vae.pad_channel_value, 1)

    def test_decode_flattens_video_batches_and_mask_is_transparency(self):
        vae = self.nodes.H3RGBAVAE(self.state)
        rgba, rgb, mask = self.nodes.H3RGBADecode().decode({'samples': torch.zeros(1, 24, 2, 2, 2)}, vae)
        self.assertEqual(rgba.shape, (2, 32, 32, 4))
        self.assertEqual(rgb.shape, (2, 32, 32, 3))
        torch.testing.assert_close(mask, torch.full((2, 32, 32), .75))
        self.nodes.H3RGBAEncode().encode(rgb, vae, mask)
        torch.testing.assert_close(vae.last_pixels, rgba)
        video = torch.zeros(1, 24, 2, 2, 2)
        audio = torch.zeros(1, 32, 2, 4)
        joint = SimpleNamespace(is_nested=True, tensors=(video, audio))
        decoded = self.nodes.H3RGBADecode().decode({'samples': joint}, vae)[0]
        self.assertEqual(decoded.shape, rgba.shape)
        self.assertIs(joint.tensors[1], audio)
        with self.assertRaises(ValueError):
            self.nodes.H3RGBAEncode().encode(rgb, vae, torch.ones(1, 16, 16))

    def test_rgb_or_wrong_family_is_rejected(self):
        self.state['encoder.conv_in.weight'] = torch.empty(128, 3, 3, 3, 3, device='meta')
        with self.assertRaisesRegex(ValueError, 'Not a native H3 RGBA VAE'):
            self.nodes.validate_h3_rgba_state(self.state)


if __name__ == '__main__':
    unittest.main()
