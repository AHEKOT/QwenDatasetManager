"""Offline CPU smoke tests: run with trainer/.venv, no model downloads required."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))

import torch
from toolkit.dto import DTO
from toolkit.config_modules import ModelConfig
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.models.v2 import resolver
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from extensions_built_in.diffusion_models.minimax_h3 import MinimaxH3Model, MinimaxH3Ref2VAModel
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import MiniMaxH3Transformer, MiniMaxH3TransformerParams


class H3RuntimeTests(unittest.TestCase):
    def test_convrot8_forward_backward_save_and_same_qtype_reload(self):
        from toolkit.util.ostris_quant import (convert_linear_to_ostris, get_ostris_quantizer,
                                               save_quantized_layers, load_quantized_layers)
        linear = torch.nn.Linear(64, 64)
        source_weight = linear.weight.detach().clone()
        source_bias = linear.bias.detach().clone()
        self.assertTrue(convert_linear_to_ostris(linear, get_ostris_quantizer('convrot8')))
        x = torch.randn(3, 64, requires_grad=True)
        prediction = linear(x)
        reference = torch.nn.functional.linear(x, source_weight, source_bias)
        self.assertLess(((prediction-reference).square().mean()/reference.square().mean()).item(), .02)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)
        before = {k: v.clone() for k, v in linear._buffers.items() if v is not None}
        convert_linear_to_ostris(linear, get_ostris_quantizer('convrot8'))
        for k, value in before.items():
            torch.testing.assert_close(value, linear._buffers[k], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory) / 'quant.safetensors')
            save_quantized_layers({'0': linear}, file)
            restored = torch.nn.Sequential(torch.nn.Linear(64, 64))
            self.assertEqual(load_quantized_layers(restored, file), 1)
            torch.testing.assert_close(restored(x), prediction)

    def test_audio_cache_collation_gradient_and_rescaling(self):
        video = torch.randn(24, 2, 2, 2, requires_grad=True)
        audio = torch.randn(16, 32, requires_grad=True)
        cached = DTO.from_state_dict(DTO(video, audio=audio).to_state_dict())
        batched = DTO.stack([cached, torch.zeros_like(video)])
        self.assertEqual(batched.audio.shape, (2, 16, 32))
        self.assertEqual(batched.audio[1].count_nonzero().item(), 0)
        batch = DataLoaderBatchDTO.__new__(DataLoaderBatchDTO)
        batch.latents = batched
        batch.latents = batch.latents * 2
        self.assertTrue(torch.equal(batch.audio_latents, batched.audio))
        loss = batch.latents.square().mean() + batch.audio_latents.square().mean()
        loss.backward()
        self.assertIsNotNone(video.grad)
        self.assertIsNotNone(audio.grad)

    def test_both_arches_full_and_pruned_video_audio_forward_backward(self):
        torch.set_num_threads(2)
        for cls in (MinimaxH3Model, MinimaxH3Ref2VAModel):
            for table_size in (None, 4):
                with self.subTest(arch=cls.arch, pruned=table_size is not None):
                    model = cls('cpu', ModelConfig(name_or_path='unused', arch=cls.arch), dtype='fp32')
                    params = MiniMaxH3TransformerParams(hidden_size=32, num_layers=1,
                        token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=12,
                        ffn_hidden_size=48, text_dim=16, timestep_input_dim=8,
                        time_embed_hidden_size=32, time_embed_dim=8, rope_inv_freq_len=2,
                        adaln_t_table_size=table_size)
                    model.model = MiniMaxH3Transformer(params)
                    model.model.enable_gradient_checkpointing()
                    latent = torch.randn(1, 24, 2, 2, 2)
                    audio = torch.randn(1, packing.audio_latent_num_frames(5) * 2, 32)
                    batch = SimpleNamespace(num_frames=5, dataset_config=SimpleNamespace(do_audio=True, do_i2v=False),
                        audio_latents=audio, latents=DTO(latent, audio=audio), control_tensor_list=None,
                        control_tensor=None, file_items=[], dopsd_teacher_pass=False)
                    embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(3, 16)],
                        text_token_tags=[torch.ones(3, dtype=torch.long)])
                    prediction = model.get_noise_prediction(latent, torch.tensor([500.]), embeds, batch=batch)
                    self.assertIsInstance(prediction, DTO)
                    self.assertEqual(prediction.shape, latent.shape)
                    self.assertEqual(prediction.audio.shape, audio.shape)
                    noise = batch.latents.audio_noise.clone()
                    model.get_noise_prediction(latent, torch.tensor([500.]), embeds, batch=batch)
                    self.assertTrue(torch.equal(noise, batch.latents.audio_noise))
                    loss = prediction.square().mean() + (prediction.audio - prediction.audio_target).square().mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.parameters()))

    def test_local_resolver_prefers_existing_nested_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'diffusion_models' / 'nested' / 'model.safetensors'
            file.parent.mkdir(parents=True)
            file.write_bytes(b'test')
            with patch.object(resolver, 'MODELS_PATH', str(root)), patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('network')):
                self.assertEqual(resolver.resolve_comfy_file('diffusion_models/model.safetensors', 'org/repo'), str(file))
                self.assertIsNone(resolver.resolve_comfy_file('vae/missing.safetensors', 'org/repo', local_only=True))
                with self.assertRaises(FileNotFoundError):
                    resolver.resolve_comfy_file('model.safetensors', 'org/repo', override_path=str(root / 'missing'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
