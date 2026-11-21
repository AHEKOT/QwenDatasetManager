"""CPU tests of the real native H3 VAE/training paths at reduced width/depth."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))
import torch
from PIL import Image
from safetensors.torch import load_file
from extensions.rgba_training.h3_rgba_vae_trainer import (
    H3ImageTrainingVAE, H3RGBAVAETrainProcess, expand_h3_vae_state_dict_to_rgba,
)
from extensions.rgba_training.h3_rgba import H3RGBAVAE, H3RGBAMixin, save_rgba_sample
from extensions.rgba_training.h3_rgba import MinimaxH3Ref2VARGBAModel
from extensions.rgba_training.qwen_rgba_vae_trainer import AlphaBoundaryGuard, FullRGBAVAEFineTune

SMALL = dict(block_out_channels=(32,) * 6, layers_per_block=1,
             decoder_num_layers=1, decoder_heads=2, decoder_head_dim=16, tiling=False)


class H3RGBARuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def pair(self):
        rgb = H3ImageTrainingVAE(**SMALL)
        rgba = H3ImageTrainingVAE.load_from_state_dict(
            expand_h3_vae_state_dict_to_rgba(rgb.state_dict()), vae_config=SMALL)
        return rgb, rgba

    def test_rgb_latents_reconstruction_and_normalization_preserved(self):
        rgb, rgba = self.pair()
        x = torch.rand(1, 3, 32, 32) * 2 - 1
        opaque = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        with torch.no_grad():
            z = rgb.encode(x)
            z_rgba = rgba.encode(opaque)
            torch.testing.assert_close(z, z_rgba, atol=1e-6, rtol=1e-5)
            y, y_rgba = rgb.decode(z), rgba.decode(z_rgba)
            torch.testing.assert_close(y, y_rgba[:, :3])
            torch.testing.assert_close(y_rgba[:, 3], torch.ones_like(y_rgba[:, 3]))
        self.assertEqual(z.shape, (1, 24, 2, 2))
        self.assertFalse(any(p.is_meta for p in rgba.parameters()))

    def test_projection_guards_cover_entire_alpha_patch_and_preserve_rgb(self):
        _, vae = self.pair()
        guard = AlphaBoundaryGuard(vae, zero_dc_alpha_encoder=False)
        optimizer = torch.optim.AdamW(guard.parameters, lr=.01, weight_decay=.1)
        before = vae.decoder.proj_out.bias.detach().clone()
        vae.decoder.proj_out.bias.sum().backward()
        optimizer.step()
        guard.restore_rgb()
        torch.testing.assert_close(vae.decoder.proj_out.bias[:3072], before[:3072], rtol=0, atol=0)
        self.assertTrue(torch.all(vae.decoder.proj_out.bias[3072:] != before[3072:]))
        _, vae = self.pair()
        guard = FullRGBAVAEFineTune(vae, alpha_lr_multiplier=3)
        guard.prepare_step()
        before = vae.decoder.proj_out.bias.detach().clone()
        with torch.no_grad():
            vae.decoder.proj_out.bias.add_(.01)
        guard.restore_rgb()
        torch.testing.assert_close(vae.decoder.proj_out.bias[3072:], before[3072:] + .03)
        torch.testing.assert_close(vae.decoder.proj_out.bias[:3072], before[:3072] + .01)

    def test_training_losses_backward_save_and_resume(self):
        rgb, rgba = self.pair()
        # Loading with assign=True can share source storage; the reference factory
        # must clone frozen encoder tensors even for an FP32 original checkpoint.
        reference = H3RGBAVAETrainProcess._reference_from_state(rgb.state_dict(), SMALL)
        self.assertNotEqual(reference.encoder.conv_in.weight.data_ptr(), rgb.encoder.conv_in.weight.data_ptr())
        process = H3RGBAVAETrainProcess.__new__(H3RGBAVAETrainProcess)
        process.vae, process.reference_vae = rgba, reference
        process.device, process.dtype = torch.device('cpu'), torch.float32
        process.perceptual_net = None
        process.loss_weights = dict(visible_rgb=1, alpha=2, alpha_edge=1, composite=1,
                                   opaque_latent=5, opaque_rgb=1, opaque_alpha=.5, latent_delta=.01, perceptual=0)
        guard = FullRGBAVAEFineTune(rgba)
        rgba.enable_gradient_checkpointing()
        optimizer = torch.optim.AdamW(guard.parameters, lr=1e-4)
        batch = torch.rand(1, 4, 32, 32) * 2 - 1
        total, losses, prediction, _ = process._forward_losses(batch)
        self.assertEqual(prediction.shape, batch.shape)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(rgba.decoder.proj_out.bias.grad[3072:].norm().item(), 0)
        optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            process.save_root, process.job = folder, SimpleNamespace(name='smoke')
            process.export_comfy_vae, process.max_saves = True, 2
            process.consecutive_passes, process.latest_report, process.print = 1, {'ready': False}, lambda *a: None
            checkpoint = process.save(optimizer, 7)
            self.assertEqual(process._latest_checkpoint(), checkpoint)
            exported = load_file(str(checkpoint / process.checkpoint_filename))
            self.assertIn('latents_mean', exported)
            restored = H3ImageTrainingVAE.load_from_state_dict(exported, vae_config=SMALL)
            torch.testing.assert_close(restored.decoder.proj_out.bias, rgba.decoder.proj_out.bias)
            process._restore_state(checkpoint, optimizer)
            self.assertEqual(process.step_num, 7)
            self.assertEqual(process.consecutive_passes, 1)

    def test_opaque_reference_padding_and_differentiable_alpha_loss(self):
        _, vae = self.pair()
        vae.__class__ = H3RGBAVAE
        pixels = torch.rand(1, 3, 32, 32) * 2 - 1
        torch.testing.assert_close(vae.encode(pixels, sample=False),
            vae.encode(torch.cat((pixels, torch.ones_like(pixels[:, :1])), 1), sample=False))
        # Make the tiny decoder's alpha depend on latent input as a trained VAE does.
        with torch.no_grad():
            vae.decoder.proj_out.weight[3072:].normal_(std=.02)
        class Model(H3RGBAMixin):
            def __init__(self):
                self.vae = SimpleNamespace(device=torch.device('cpu'), to=lambda *a: None)
                self.vae_device_torch = torch.device('cpu')
                self.video_vae = vae
                self.model_config = SimpleNamespace(model_kwargs={})
        model = Model()
        pred = torch.randn(1, 24, 1, 2, 2, requires_grad=True)
        noisy = torch.randn_like(pred)
        batch = SimpleNamespace(tensor=torch.rand(1, 4, 32, 32) * 2 - 1)
        loss, _ = model.get_rgba_latent_auxiliary_loss(batch, pred=pred, target=pred.detach(),
            noisy_latents=noisy, timesteps=torch.tensor([500.]))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.norm().item(), 0)

    def test_apng_retains_alpha_and_audio_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.png'
            frames = torch.zeros(2, 8, 8, 4, dtype=torch.uint8)
            frames[0, :, :, 3], frames[1, :, :, 3] = 32, 192
            config = SimpleNamespace(get_image_path=lambda *a: str(path))
            save_rgba_sample(config, dict(video=frames, fps=24, audio=torch.zeros(2, 100), audio_sample_rate=32000))
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 2)
                self.assertEqual(image.getpixel((0, 0))[3], 32)
                image.seek(1)
                self.assertEqual(image.getpixel((0, 0))[3], 192)
            self.assertTrue(path.with_suffix('.wav').is_file())

    def test_dopsd_teacher_receives_rgba_but_student_has_no_references(self):
        model = MinimaxH3Ref2VARGBAModel.__new__(MinimaxH3Ref2VARGBAModel)
        model.dopsd = True
        model.model_config = SimpleNamespace(model_kwargs={})
        observed = []
        def encode(frames):
            observed.append(frames)
            return torch.zeros(1, 24, 1, 2, 2)
        model.encode_keyframe_latents = encode
        batch = SimpleNamespace(tensor=torch.rand(1, 4, 32, 32), dopsd_teacher_pass=True,
            num_frames=1, dataset_config=SimpleNamespace(do_audio=False, rgba_control_mode='generation'))
        teacher = model._build_condition(batch, (1, 2, 2), torch.device('cpu'), torch.float32)
        self.assertEqual(observed[0].shape, (1, 4, 1, 32, 32))
        self.assertEqual(teacher[0].shape, (1, 1, 96))
        self.assertEqual(teacher[3], ((1, 2, 2, 0),))
        batch.dopsd_teacher_pass = False
        self.assertEqual(model._build_condition(batch, (1, 2, 2), torch.device('cpu'), torch.float32), (None, None, (), ()))


if __name__ == '__main__':
    unittest.main()
