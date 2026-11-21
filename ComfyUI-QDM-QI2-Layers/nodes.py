"""Native ComfyUI inference for QDM's joint Qwen Image 2.1 layer adapters."""
from __future__ import annotations

import functools
import inspect
import json
import math
from pathlib import Path
import types
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import comfy.lora
import comfy.lora_convert
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers

from .core import (FORMAT_VERSION, validate_geometry, joint_build_sequence,
                   flow_sigmas, to_pil_layers, composite_layers, save_psd, legacy_timestep_forward)

CATEGORY = 'QDM/QI2 Layers'


def require_model(model):
    try:
        from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
    except ImportError as exc:
        raise RuntimeError('This pack requires ComfyUI with native Qwen Image 2.1 support. Update ComfyUI.') from exc
    core = model.model.diffusion_model
    if not isinstance(core, QwenImage21Transformer2DModel):
        raise ValueError('Load Qwen Image 2.1, not Qwen Image 1.x or Qwen-Image-Layered.')
    parameters = inspect.signature(type(core).build_sequence).parameters
    if tuple(parameters) != ('self', 'x', 'context', 'ref_latents', 'image_slots'):
        raise RuntimeError('ComfyUI QI2 sequence API has changed; this pack must be updated before inference.')
    if getattr(core, 'out_channels', None) != 64:
        raise ValueError('Expected the native 64-channel QI2 model.')
    return core


def require_vae(vae):
    if (getattr(vae, 'latent_channels', None) != 64 or
            getattr(vae, 'output_channels', None) != 4 or
            getattr(vae, 'downscale_ratio', None) != 16):
        raise ValueError('Select qwen_image_2.1_vae_bf16.safetensors (native RGBA, 64 latent channels, 16x).')


def declared_layer_format(metadata):
    """Only known QDM metadata fields; no guesses based on the LoRA filename."""
    def decode(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return None
        return value
    qdm = decode(metadata.get('qdm', {}))
    meta = decode(metadata.get('meta', {}))
    if not qdm and isinstance(meta, dict):
        qdm = decode(meta.get('qdm', {}))
    if isinstance(qdm, dict) and qdm.get('trainingPreset') not in (None, 'qwen_layered_lora'):
        raise ValueError('This adapter metadata declares a different training preset, not joint PSD layers.')
    layer_format = decode(qdm.get('layerFormat', {})) if isinstance(qdm, dict) else None
    return layer_format if isinstance(layer_format, dict) else {}


def declared_slots(metadata):
    return declared_layer_format(metadata).get('layerSlots')


def declared_timestep_embedding(metadata):
    mode = declared_layer_format(metadata).get('timestepEmbedding', 'legacy_bf16')
    if mode not in ('fp32', 'legacy_bf16'):
        raise ValueError(f'Unsupported QDM timestep embedding: {mode}')
    return mode


class QDMLayeredQI2LoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'lora_name': (folder_paths.get_filename_list('loras'),),
            'layer_slots': ('INT', {'default': 4, 'min': 1, 'max': 20}),
            'strength': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 2.0, 'step': 0.05}),
        }}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Loads a QDM joint PSD LoRA. Layer slots must match training; 4 and 20 are different inference contracts.'

    def load(self, model, lora_name, layer_slots, strength):
        require_model(model)
        validate_geometry(32, 32, layer_slots)
        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        metadata = metadata or {}
        trained_slots = declared_slots(metadata)
        timestep_embedding = declared_timestep_embedding(metadata)
        if trained_slots is not None and trained_slots != layer_slots:
            raise ValueError(f'This adapter declares {trained_slots} layer slots, but {layer_slots} were selected.')
        state = comfy.lora_convert.convert_lora(state)
        mapping = comfy.lora.model_lora_keys_unet(model.model, {})
        # QDM exports diffusion_model.*; also accept its equivalent transformer.* prefix.
        for key, value in list(mapping.items()):
            if key.startswith('diffusion_model.'):
                mapping['transformer.' + key[len('diffusion_model.'):]] = value
        # Fail on incompatible/partially mapped adapters, rather than quietly generating with missing weights.
        roots = set()
        for key in state:
            candidates = [name for name in mapping if key.startswith(name + '.')]
            if not candidates:
                raise ValueError(f'Adapter tensor cannot be mapped to QI2: {key}')
            roots.add(max(candidates, key=len))
        patches = comfy.lora.load_lora(state, mapping)
        expected = {mapping[root] for root in roots}
        if not patches or not expected.issubset(patches):
            raise ValueError('Adapter is incomplete or unsupported. Export a complete QDM QI2 LoRA/LoKr safetensors file.')
        result = model.clone()
        applied = set(result.add_patches(patches, strength))
        if applied != set(patches):
            raise ValueError('Some QI2 adapter weights could not be applied.')
        result.model_options['qdm_qi2_layers'] = {
            'format_version': FORMAT_VERSION, 'layer_slots': layer_slots,
            'lora_name': lora_name, 'strength': strength,
            'timestep_embedding': timestep_embedding,
        }
        return (result,)


class QDMLayeredQI2Conditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',), 'vae': ('VAE',),
            'prompt': ('STRING', {'default': 'Create layered image from image 1', 'multiline': True}),
            'negative_prompt': ('STRING', {'default': '', 'multiline': True}),
            'width': ('INT', {'default': 1024, 'min': 32, 'max': 4096, 'step': 32}),
            'height': ('INT', {'default': 1024, 'min': 32, 'max': 4096, 'step': 32}),
            'match_reference_area': ('BOOLEAN', {'default': True}),
        }, 'optional': {'image': ('IMAGE',)}}
    RETURN_TYPES = ('QDM_QI2_CONDITIONING',)
    FUNCTION = 'encode'
    CATEGORY = CATEGORY
    DESCRIPTION = 'One flat reference, or no reference for text-to-layers. Matches QDM reference sizing and RGBA VAE input.'

    def encode(self, clip, vae, prompt, negative_prompt, width, height, match_reference_area=True, image=None):
        require_vae(vae)
        validate_geometry(width, height, 1)
        from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
        if not isinstance(clip.tokenizer, QwenImage21Tokenizer):
            raise ValueError('Use CLIPLoader with qwen3vl_8b and type qwen_image.')
        refs, vision = [], []
        if image is not None:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
                raise ValueError('Provide one RGB/RGBA image, not a batch of documents.')
            if not torch.isfinite(image).all():
                raise ValueError('Reference contains non-finite pixels.')
            value = image.detach().float().clamp(0, 1).movedim(-1, 1)
            if value.shape[1] == 3:
                value = torch.cat([value, torch.ones_like(value[:, :1])], dim=1)
            ih, iw = value.shape[-2:]
            # Same cap as QDM's default QI2 model configuration.
            area = width * height if match_reference_area else 1024 * 1024
            if match_reference_area or ih * iw > area:
                fw = math.sqrt(area * iw / ih)
                fh = fw / (iw / ih)
            else:
                fw, fh = iw, ih
            rw, rh = max(32, round(fw / 32) * 32), max(32, round(fh / 32) * 32)
            if (ih, iw) != (rh, rw):
                value = F.interpolate(value, size=(rh, rw), mode='bicubic', antialias=True).clamp(0, 1)
            rgba = value.movedim(1, -1)
            rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
            # Trainer hands 8-bit PIL RGB to Qwen3-VL; keep the same quantization.
            vision = [rgb.mul(255).round().div(255)]
            refs = [vae.encode(rgba)]
            if refs[0].ndim != 4 or tuple(refs[0].shape) != (1, 64, rh//16, rw//16):
                raise ValueError('The selected VAE returned incompatible QI2 reference latents.')
        conds = []
        for text in (prompt, negative_prompt):
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text, images=vision, keep_vision=False, prevent_empty_text=True))
            if refs:
                cond = node_helpers.conditioning_set_values(cond, {'reference_latents': refs})
            conds.append(cond)
        return ({'positive': conds[0], 'negative': conds[1], 'width': width, 'height': height,
                 'prompt': prompt, 'negative_prompt': negative_prompt, 'has_reference': bool(refs),
                 'match_reference_area': match_reference_area},)


class QDMLayeredQI2Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'conditioning': ('QDM_QI2_CONDITIONING',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
            'steps': ('INT', {'default': 40, 'min': 2, 'max': 200}),
            'cfg': ('FLOAT', {'default': 1.0, 'min': 1.0, 'max': 10.0, 'step': 0.1}),
        }}
    RETURN_TYPES = ('QDM_QI2_LAYER_LATENT',)
    FUNCTION = 'sample'
    CATEGORY = CATEGORY
    DESCRIPTION = 'One joint denoising trajectory for all layers. Uses the trainer Euler/shift/terminal schedule, never independent layer sampling.'

    def sample(self, model, conditioning, seed, steps, cfg):
        require_model(model)
        try:
            import psd_tools  # report a missing exporter before denoising
        except ImportError as exc:
            raise RuntimeError('Install this node pack requirements.txt in the ComfyUI environment first.') from exc
        contract = model.model_options.get('qdm_qi2_layers')
        if not contract or contract.get('format_version') != FORMAT_VERSION:
            raise ValueError('Connect QDM QI2 Layer LoRA Loader before this sampler.')
        slots = contract['layer_slots']
        width, height = conditioning['width'], conditioning['height']
        validate_geometry(width, height, slots)
        patched = model.clone()
        core = require_model(patched)
        build = types.MethodType(functools.partial(joint_build_sequence, layer_slots=slots), core)
        patched.add_object_patch('diffusion_model.build_sequence', build)
        if contract['timestep_embedding'] == 'legacy_bf16':
            patched.add_object_patch('diffusion_model.time_text_embed.forward',
                                    types.MethodType(legacy_timestep_forward, core.time_text_embed))
        # No cross-workflow prefix-cache reuse; all layers always remain one target block.
        patched.model_options.setdefault('transformer_options', {})['qwen_image21_cache'] = {'device': 'off'}
        latent = torch.zeros((1, 64, slots * (height//16), width//16), dtype=torch.float32, device=mm.intermediate_device())
        noise = torch.randn(latent.shape, generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32)
        sigmas = flow_sigmas(width, height, slots, steps)
        progress = comfy.utils.ProgressBar(steps)
        def callback(step, denoised, current, total):
            progress.update_absolute(step+1, total)
        samples = comfy.sample.sample_custom(patched, noise, cfg, comfy.samplers.sampler_object('euler'),
                    sigmas, conditioning['positive'], conditioning['negative'], latent,
                    callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        meta = {**contract, **{k: conditioning[k] for k in ('width', 'height', 'prompt', 'negative_prompt', 'has_reference', 'match_reference_area')},
                'seed': seed, 'steps': steps, 'cfg': cfg, 'sampler': 'euler', 'shift_terminal': .02}
        return ({'samples': samples, 'metadata': meta},)


class QDMLayeredQI2Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('QDM_QI2_LAYER_LATENT',), 'vae': ('VAE',)}}
    RETURN_TYPES = ('QDM_RGBA_LAYERS', 'IMAGE', 'IMAGE', 'MASK')
    RETURN_NAMES = ('document', 'composite_rgba', 'layers_rgba', 'transparency_masks')
    FUNCTION = 'decode'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Decodes each latent grid separately with the native RGBA VAE. Output image batch is bottom-to-top layers of one document.'

    def decode(self, samples, vae):
        require_vae(vae)
        meta = samples['metadata']
        slots, width, height = meta['layer_slots'], meta['width'], meta['height']
        validate_geometry(width, height, slots)
        latent = samples['samples']
        if tuple(latent.shape) != (1, 64, slots * (height//16), width//16):
            raise ValueError('Layer latent shape no longer matches the sampled document.')
        decoded = []
        for layer in latent.chunk(slots, dim=2):
            mm.throw_exception_if_processing_interrupted()
            pixels = vae.decode(layer.contiguous())
            if tuple(pixels.shape) != (1, height, width, 4):
                raise ValueError('Native QI2 VAE must decode one RGBA canvas per layer.')
            decoded.append(pixels.detach().float().cpu())
        pil_layers = to_pil_layers(torch.cat(decoded, dim=0))
        merged = composite_layers(pil_layers)
        rgba = torch.from_numpy(np.stack([np.asarray(x).copy() for x in pil_layers])).float().div(255)
        composite = torch.from_numpy(np.asarray(merged).copy()).float().div(255).unsqueeze(0)
        return ({'layers': pil_layers, 'metadata': meta}, composite, rgba, 1-rgba[..., 3])


class QDMLayeredQI2Save:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'document': ('QDM_RGBA_LAYERS',),
                             'filename_prefix': ('STRING', {'default': 'QI2-Layers/layered'})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('psd_path',)
    FUNCTION = 'save'
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = 'Saves PSD, composite PNG, individual RGBA PNGs and metadata. Empty slots and layer order are preserved.'

    def save(self, document, filename_prefix, prompt=None, extra_pnginfo=None):
        from PIL.PngImagePlugin import PngInfo
        layers = document['layers']
        merged = composite_layers(layers)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, str(output_root), *merged.size)
        parent = Path(full).resolve()
        if parent != output_root and output_root not in parent.parents:
            raise ValueError('Output must stay inside the ComfyUI output directory.')
        # Unique document directory prevents overwriting or mixing parallel queues.
        stem = f'{filename}_{counter:05}_{uuid.uuid4().hex[:8]}'
        target = (parent / stem).resolve()
        if output_root not in target.parents:
            raise ValueError('Invalid output filename.')
        target.mkdir(parents=True, exist_ok=False)
        info = PngInfo()
        if prompt is not None:
            info.add_text('prompt', json.dumps(prompt, ensure_ascii=False))
        for key, value in (extra_pnginfo or {}).items():
            info.add_text(key, json.dumps(value, ensure_ascii=False))
        psd = target / 'layers.psd'
        save_psd(psd, layers)
        merged.save(target / 'composite.png', pnginfo=info)
        layer_dir = target / 'layers'
        layer_dir.mkdir()
        for index, layer in enumerate(layers):
            layer.save(layer_dir / f'{index+1:02d}.png')
        meta = {**document['metadata'], 'order': 'bottom_to_top', 'layer_count': len(layers),
                'layer_files': [f'layers/{i+1:02d}.png' for i in range(len(layers))]}
        (target / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'ui': {'images': [{'filename': 'composite.png', 'subfolder': target.relative_to(output_root).as_posix(), 'type': 'output'}]},
                'result': (str(psd),)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (QDMLayeredQI2LoRA, QDMLayeredQI2Conditioning,
                      QDMLayeredQI2Sampler, QDMLayeredQI2Decode, QDMLayeredQI2Save)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'QDMLayeredQI2LoRA': 'QI2 Layer LoRA Loader (QDM)',
    'QDMLayeredQI2Conditioning': 'QI2 Layer Conditioning (QDM)',
    'QDMLayeredQI2Sampler': 'QI2 Joint Layer Sampler (QDM)',
    'QDMLayeredQI2Decode': 'QI2 Decode RGBA Layers (QDM)',
    'QDMLayeredQI2Save': 'QI2 Save Layered PSD (QDM)',
}
