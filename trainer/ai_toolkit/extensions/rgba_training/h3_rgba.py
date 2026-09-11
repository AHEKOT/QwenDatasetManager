"""Transparent MiniMax H3 / Ref2VA training and lossless sampling."""
import hashlib
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    MinimaxH3Model, MinimaxH3Ref2VAModel,
    KEYFRAME_NOISE_AUG_T, patchify_video_latents,
)
from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE
from .lora_loss import RGBALoRALossMixin


class H3RGBAVAE(MiniMaxH3VideoVAE):
    def encode(self, pixels, **kwargs):
        # RGB reference frames are opaque; RGBA targets retain their own alpha.
        if pixels.shape[1] == 3:
            pixels = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=1)
        return super().encode(pixels, **kwargs)


def save_rgba_sample(config, result, count=0, max_count=0, **kwargs):
    path = Path(config.get_image_path(count, max_count))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(frame.cpu().numpy()).convert('RGBA') for frame in result['video']]
    frames[0].save(path, format='PNG', save_all=True, append_images=frames[1:],
                   duration=1000 / result['fps'], loop=0, disposal=0, blend=0)
    if result.get('audio') is not None:
        import soundfile as sf
        sf.write(str(path.with_suffix('.wav')), result['audio'].cpu().float().numpy().T,
                 result['audio_sample_rate'])


class H3RGBAMixin(RGBALoRALossMixin):
    supports_rgba_video_loss = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(self.model_config.vae_path or '').expanduser()
        if not path.is_file():
            raise ValueError('Transparent H3 requires a trained native H3 RGBA VAE file')
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            enc = handle.get_slice('encoder.conv_in.weight').get_shape()
            dec = handle.get_slice('decoder.proj_out.weight').get_shape()
            latent = handle.get_slice('post_quant_conv.weight').get_shape()
        if enc != [128, 4, 3, 3, 3] or dec != [4096, 2048] or latent != [24, 24, 1, 1, 1]:
            raise ValueError('Selected VAE must be native H3 RGBA (4 pixel / 24 latent channels)')
        self.model_config.model_kwargs['video_vae_path'] = str(path.resolve())
        identity = f'{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}'
        self.latent_space_version = 'minimax-h3-rgba-' + hashlib.sha256(identity.encode()).hexdigest()[:16]
        self.require_pixel_tensor_cache = True

    def _load_vaes(self):
        bundle = super()._load_vaes()
        # Identical parameters; the subclass only pads opaque RGB references.
        bundle.video_vae.__class__ = H3RGBAVAE
        return bundle

    def _present_image_control(self, image):
        # Qwen3-VL sees opaque RGB; the video VAE keeps alpha in latent refs.
        return super()._present_image_control(image.convert('RGB'))

    def _build_condition(self, batch, latent_shape, device, dtype):
        if getattr(self, 'dopsd', False):
            if getattr(batch, 'dopsd_teacher_pass', False):
                return super()._build_condition(batch, latent_shape, device, dtype)
            return None, None, (), ()
        ds = getattr(batch, 'dataset_config', None)
        if ds and getattr(ds, 'rgba_control_mode', '') == 'generation':
            return None, None, (), ()
        if isinstance(self, MinimaxH3Ref2VAModel):
            return super()._build_condition(batch, latent_shape, device, dtype)
        if ds and getattr(ds, 'rgba_generate_control', False):
            frames = batch.control_tensor
            if frames is None:
                raise ValueError('H3 RGBA edit requires the generated opaque control tensor')
            if frames.ndim == 5:
                frames = frames[:, 0]
            latents = self.encode_keyframe_latents((frames * 2 - 1).unsqueeze(2).to(device))
            latents = KEYFRAME_NOISE_AUG_T * latents + (1 - KEYFRAME_NOISE_AUG_T) * torch.randn_like(latents)
            return patchify_video_latents(latents).to(dtype), None, ('first',), ()
        return super()._build_condition(batch, latent_shape, device, dtype)

    def get_rgba_latent_auxiliary_loss(self, batch, *, pred, target, noisy_latents, timesteps):
        strengths = [self._rgba_loss_setting('alpha', 1), self._rgba_loss_setting('alpha_edge', .5)]
        if not any(strengths):
            return pred.sum() * 0, {}
        if pred.ndim != 5 or pred.shape[2] != 1:
            raise ValueError('H3 transparent LoRA targets must be single RGBA images')
        sigma = timesteps.to(pred.device).float().reshape(-1, 1, 1, 1, 1) / 1000
        clean = noisy_latents.float() - sigma * pred.float()
        if self.vae.device == torch.device('cpu'):
            self.vae.to(self.vae_device_torch)
        decoded = self.video_vae.decode(clean.to(self.video_vae.dtype), autocast_fp16=True, clamp=False)
        alpha = (decoded[:, 3:4, 0].float() + 1) * .5
        expected = (batch.tensor[:, 3:4].to(alpha).float() + 1) * .5
        expected = F.interpolate(expected, size=alpha.shape[-2:], mode='area')
        point = self._rgba_balanced_pointwise_loss(F.smooth_l1_loss(alpha, expected, reduction='none'), expected)
        edge = self._rgba_balanced_edge_loss(*self._rgba_alpha_edges(alpha), *self._rgba_alpha_edges(expected), expected)
        return point * strengths[0] + edge * strengths[1], {'rgba_alpha': point.detach(), 'rgba_alpha_edge': edge.detach()}

    def generate_single_image(self, pipeline, gen_config, *args, **kwargs):
        result = super().generate_single_image(pipeline, gen_config, *args, **kwargs)
        if isinstance(result, dict):
            gen_config.output_ext = 'png'
            gen_config.save_image = partial(save_rgba_sample, gen_config)
        return result


class MinimaxH3RGBAModel(H3RGBAMixin, MinimaxH3Model):
    arch = 'minimax_h3_rgba'


class MinimaxH3Ref2VARGBAModel(H3RGBAMixin, MinimaxH3Ref2VAModel):
    arch = 'minimax_h3_ref2va_rgba'
