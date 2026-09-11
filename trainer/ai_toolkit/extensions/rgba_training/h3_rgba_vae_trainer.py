"""Native H3 RGBA boundaries; unchanged 24-channel video latent space."""
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE
from .flux2_rgba_vae_trainer import Flux2RGBAVAETrainProcess


def expand_h3_vae_state_dict_to_rgba(state):
    result = dict(state)
    enc = state['encoder.conv_in.weight']
    dec = state['decoder.proj_out.weight']
    bias = state['decoder.proj_out.bias']
    channels = enc.shape[1]
    if channels not in (3, 4) or dec.shape[0] != bias.shape[0] or dec.shape[0] % channels:
        raise ValueError('Invalid native H3 VAE boundary tensors')
    if channels == 4:
        return result
    patch = dec.shape[0] // 3
    result['encoder.conv_in.weight'] = torch.cat((enc, torch.zeros_like(enc[:, :1])), dim=1)
    result['decoder.proj_out.weight'] = torch.cat((dec, dec.new_zeros((patch, *dec.shape[1:]))))
    result['decoder.proj_out.bias'] = torch.cat((bias, bias.new_ones(patch)))
    return result


class H3ImageTrainingVAE(MiniMaxH3VideoVAE):
    """Image-batch facade for the shared RGBA losses, with native state keys."""
    def encode(self, pixels, **kwargs):
        latent = super().encode(pixels, sample=False)
        return latent.squeeze(2) if pixels.ndim == 4 else latent

    def decode(self, latents, **kwargs):
        still = latents.ndim == 4
        result = super().decode(latents.unsqueeze(2) if still else latents,
                                autocast_fp16=False, clamp=False)
        return result.squeeze(2) if still else result


class H3EncoderReference(H3ImageTrainingVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = nn.Identity()


class H3RGBAVAETrainProcess(Flux2RGBAVAETrainProcess):
    family = 'minimax_h3'
    checkpoint_suffix = 'h3'
    checkpoint_filename = 'minimax_h3_rgba_vae.safetensors'

    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.source_filename = self.get_conf('source_vae.filename', 'minimax_h3_video_vae_fp16.safetensors')

    @staticmethod
    def _model_from_state(state):
        if state['post_quant_conv.weight'].shape[:2] != (24, 24):
            raise ValueError('H3 VAE requires the original 24-channel latent space')
        return H3ImageTrainingVAE.load_from_state_dict(state)

    def _load_source_vae(self):
        state = load_file(str(self._resolve_source_file()), device='cpu')
        if state['encoder.conv_in.weight'].shape[1] != 3:
            raise ValueError('The reference source must be the original RGB H3 VAE')
        # The reference only encodes; discard its large ViT decoder immediately.
        return self._reference_from_state(state)

    @staticmethod
    def _reference_from_state(state, vae_config=None):
        # Independent frozen storage, including when the source is already FP32.
        # Never allocate another multi-billion-parameter decoder for the reference.
        state = {k: v.clone() for k, v in state.items() if not k.startswith('decoder.')}
        return H3EncoderReference.load_from_state_dict(state, vae_config=vae_config or {})

    def _load_or_create_vae(self):
        checkpoint = self._latest_checkpoint()
        source = load_file(str(self._resolve_source_file()), device='cpu')
        if source['encoder.conv_in.weight'].shape[1] != 3:
            raise ValueError('source_vae must point to the original RGB H3 VAE')
        state = (load_file(str(checkpoint / self.checkpoint_filename), device='cpu')
                 if checkpoint else expand_h3_vae_state_dict_to_rgba(source))
        if state['encoder.conv_in.weight'].shape[1] != 4:
            raise ValueError('Resume checkpoint is not an RGBA H3 VAE')
        vae = self._model_from_state(state).to(dtype=torch.float32)
        self.reference_vae = self._reference_from_state(source)
        self.print(f'H3 RGBA VAE: {checkpoint or self.source_path}')
        return vae, checkpoint

    @staticmethod
    def _serializable_state(vae, dtype):
        state = Flux2RGBAVAETrainProcess._serializable_state(vae, dtype)
        # Comfy checkpoints carry these original normalization statistics.
        state.update({key: getattr(vae, key).detach().cpu().float().contiguous()
                      for key in ('latents_mean', 'latents_std')})
        return state
