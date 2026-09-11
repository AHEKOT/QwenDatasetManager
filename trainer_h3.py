"""MiniMax H3 form/config contract, kept independent of the CUDA runtime."""

import copy
from pathlib import Path

H3_KEYS = {'minimax_h3', 'minimax_h3_ref2va'}
H3_ARCHES = H3_KEYS | {key + '_rgba' for key in H3_KEYS}
H3_SOURCE_COMMIT = '3e1cb5f67f7bebe6ac74e3da5105d18fb3c1dd5a'
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv'}


def h3_models():
    return {
        key: {
            'label': 'MiniMax-H3 Ref2VA' if key.endswith('ref2va') else 'MiniMax-H3',
            'modelPath': 'Comfy-Org/MiniMax-H3', 'arch': key, 'kind': 'video',
            'transparentArch': key + '_rgba',
            'license': 'See model repository', 'gated': False, 'gateUrl': None,
            'defaultQtype': 'convrot8', 'defaultQtypeTextEncoder': 'nvfp4',
            'noiseScheduler': 'flowmatch', 'allowUnloadTextEncoder': True,
            'accuracyRecoveryAdapters': {}, 'defaults': h3_defaults(key),
        }
        for key in sorted(H3_KEYS)
    }


def h3_defaults(key):
    ref = key.endswith('ref2va')
    return {
        'qtype': 'convrot8', 'qtypeTextEncoder': 'nvfp4', 'lowVram': True,
        'rank': 16, 'timestepType': 'shift', 'cacheTextEmbeddings': True,
        'guidanceLoss': True, 'guidanceLossTarget': 3.5, 'distillationMethod': 'both',
        'assistantLoraPath': 'ostris/minimax_h3_training_adapter/'
        + ('minimax_h3_ref2va_training_adapter_v1.safetensors' if ref
           else 'minimax_h3_training_adapter_v1.safetensors'),
        'sampleFrames': 107, 'sampleFps': 24, 'sampleWidth': 768, 'sampleHeight': 768,
        'guidanceScale': 1, 'sampleSteps': 28, 'audioLossMultiplier': 1,
        'h3Partition': 'ref2va_pruned' if ref else 'fl2va_pruned',
        'h3MaxTextLength': 512, 'h3SampleAudio': True,
        'h3ImageRefsAsVideo': False, 'h3ImageRefVideoFrames': 5,
        'h3DopsdBleedStrength': 1, 'h3DitPath': '', 'h3TextEncoderPath': '',
        'h3VideoVaePath': '', 'h3AudioVaePath': '', 'h3ConfigPath': '',
        'h3LocalOnly': False, 'modelsPath': '',
    }


def local_h3_defaults(key, project_root):
    defaults = h3_defaults(key)
    root = Path(project_root) / 'models'
    defaults['modelsPath'] = str(root)
    ref = key.endswith('ref2va')
    names = (['minimax_h3_ref2va_training_adapter_v1.safetensors'] if ref else
             ['minimax_h3_training_adapter_v2.safetensors', 'minimax_h3_training_adapter_v1.safetensors'])
    for name in names:
        path = root / 'loras' / 'training_adapters' / name
        if path.is_file():
            defaults['assistantLoraPath'] = str(path)
            break
    return defaults


def with_h3_defaults(payload, project_root=None):
    if not isinstance(payload, dict) or payload.get('model') not in H3_KEYS:
        return payload
    defaults = local_h3_defaults(payload['model'], project_root) if project_root else h3_defaults(payload['model'])
    result = {**defaults, **copy.deepcopy(payload)}
    if not result.get('modelsPath'):
        result['modelsPath'] = defaults['modelsPath']
    return result


def frame_count(value, label, error, *, still=True):
    try:
        n = int(value)
        if float(value) != n:
            raise ValueError()
    except (ValueError, TypeError):
        raise error(f'{label}: enter 1 or 17n+5 frames')
    if (still and n == 1) or (5 <= n <= 10000 and (n - 5) % 17 == 0):
        return n
    raise error(f'{label}: use 1 (image), 5, 22, 39, 56, 73, 90, 107… frames')


def configure_h3(process, payload, error, number, local_asset):
    """Apply all Simple UI H3 fields before the optional full process override."""
    ref = payload['model'] == 'minimax_h3_ref2va'
    transparent = payload.get('trainingPreset') == 'transparent_lora'
    method = payload['distillationMethod']
    if method not in ({'cg', 'ta', 'both', 'none', 'dopsd'} if ref else {'cg', 'ta', 'both', 'none'}):
        raise error('Unsupported MiniMax H3 distillation method')
    model, train, sample = (process[k] for k in ('model', 'train', 'sample'))
    kw = model['model_kwargs']
    partition = payload['h3Partition']
    if partition not in ({'ref2va', 'ref2va_pruned'} if ref else {'fl2va', 'fl2va_pruned'}):
        raise error('Partition does not match the selected H3 architecture')
    kw.update(partition=partition, max_text_length=number(payload.get('h3MaxTextLength'), 0, 100000, 512, integer=True),
              sample_audio=bool(payload.get('h3SampleAudio', True)), local_files_only=bool(payload.get('h3LocalOnly')))
    for field, name in [('h3DitPath', f'dit_{partition}_path'), ('h3TextEncoderPath', 'text_encoder_path'),
                        ('h3VideoVaePath', 'video_vae_path'), ('h3AudioVaePath', 'audio_vae_path')]:
        if payload.get(field):
            kw[name] = local_asset(payload[field], field)
    if payload.get('h3ConfigPath'):
        kw['config_path'] = local_asset(payload['h3ConfigPath'], 'H3 tokenizer/config folder', directory=True)
    if payload.get('modelsPath'):
        # Retained in process config so detached runs and saved jobs use the same root.
        model['models_path'] = local_asset(payload['modelsPath'], 'Models folder', directory=True)
    train['audio_loss_multiplier'] = number(payload.get('audioLossMultiplier'), 0, 1000, 1)
    train['do_guidance_loss'] = method in {'cg', 'both'}
    if train['do_guidance_loss']:
        train['guidance_loss_target'] = number(payload.get('guidanceLossTarget'), 0, 1000, 3.5)
    else:
        train.pop('guidance_loss_target', None)
    if method in {'ta', 'both'}:
        adapter = str(payload.get('assistantLoraPath', '')).strip()
        if not adapter or '\x00' in adapter:
            raise error('Set the H3 training adapter path or Hub file')
        model['assistant_lora_path'] = adapter
    else:
        model.pop('assistant_lora_path', None)
    if ref:
        kw['image_refs_as_video'] = bool(payload.get('h3ImageRefsAsVideo'))
        kw['image_ref_video_frames'] = frame_count(payload.get('h3ImageRefVideoFrames', 5), 'Reference clip', error, still=False)
        kw['dopsd'] = method == 'dopsd'
        if method == 'dopsd':
            kw['dopsd_bleed_strength'] = number(payload.get('h3DopsdBleedStrength'), 0, 1000, 1)
            train['cache_text_embeddings'] = True
    process['network']['network_kwargs']['ignore_if_contains'] = ['adaln_proj']
    sample['num_frames'] = frame_count(payload['sampleFrames'], 'Sample', error)
    # The released sampler and rotary/audio clock run at 24 FPS.
    sample['fps'] = 24
    sample['format'] = 'png' if sample['num_frames'] == 1 else 'mp4'
    if transparent:
        sample['format'] = 'png'  # lossless animated PNG preserves video alpha
    for ds, submitted in zip(process['datasets'], payload['datasets']):
        ds.update({
            'num_frames': frame_count(submitted.get('numFrames', 39), 'Dataset', error),
            'fps': number(submitted.get('fps'), 1, 120, 24, integer=True),
            'auto_frame_count': bool(submitted.get('autoFrameCount', True)),
            'do_audio': bool(submitted.get('doAudio', True)),
            'audio_normalize': bool(submitted.get('audioNormalize', False)),
            'audio_preserve_pitch': bool(submitted.get('audioPreservePitch', False)),
            'do_i2v': bool(submitted.get('doI2v', False)) and not ref,
            'cache_latents_to_disk': bool(submitted.get('cacheLatents', True)),
            'shrink_video_to_frames': bool(submitted.get('shrinkVideoToFrames', True)),
            'trim_auto_frame_count_tail': bool(submitted.get('trimAutoFrameCountTail', True)),
        })
        if method == 'dopsd':
            ds['cache_tensors_to_disk'] = True
        selected = submitted.get('controls')
        if selected is not None and not transparent:
            if not isinstance(selected, list) or any(c not in ('Control1', 'Control2', 'Control3') for c in selected):
                raise error('Invalid H3 control folder selection')
            root = Path(ds['folder_path']).parent
            ds['control_path'] = [str(root / c) for c in dict.fromkeys(selected)]
        if not ds.get('control_path') or method == 'dopsd':
            ds.pop('control_path', None)
        if transparent:
            ds.update(num_frames=1, auto_frame_count=False, do_audio=False,
                      do_i2v=False, cache_tensors_to_disk=True)
            # Generation uses captions directly; edit uses the existing opaque
            # composite/background pipeline and its dynamic reference tensors.
            if ds.get('rgba_control_mode') == 'generation' or method == 'dopsd':
                ds['rgba_generate_control'] = False
                ds.pop('control_path', None)
            else:
                train['cache_text_embeddings'] = False
                train['unload_text_encoder'] = False


def validate_h3_process(process, error):
    """Check the effective process too, including values from Advanced JSON."""
    model = process['model']
    kw = model.get('model_kwargs', {})
    ref = model['arch'] in {'minimax_h3_ref2va', 'minimax_h3_ref2va_rgba'}
    transparent = model['arch'].endswith('_rgba')
    partition = kw.get('partition', 'ref2va_pruned' if ref else 'fl2va_pruned')
    if partition not in ({'ref2va', 'ref2va_pruned'} if ref else {'fl2va', 'fl2va_pruned'}):
        raise error('Partition does not match the selected H3 architecture')
    if kw.get('dopsd'):
        if not ref:
            raise error('D-OPSD requires MiniMax H3 Ref2VA')
        if model.get('assistant_lora_path') or process['train'].get('do_guidance_loss'):
            raise error('D-OPSD must be used without the training adapter or contrastive guidance')
        if process['train'].get('diff_output_preservation') or process['train'].get('blank_prompt_preservation'):
            raise error('D-OPSD cannot be combined with output/blank prompt preservation')
        process['train']['cache_text_embeddings'] = True
        for ds in process['datasets']:
            ds.pop('control_path', None)
            ds['cache_tensors_to_disk'] = True
            if transparent:
                ds['rgba_generate_control'] = False
    sample = process['sample']
    if transparent:
        sample['format'] = 'png'
        for ds in process['datasets']:
            if ds.get('num_frames') != 1 or ds.get('auto_frame_count') or ds.get('do_audio'):
                raise error('Transparent H3 training requires single RGBA image targets')
    sample['fps'] = 24
    for item in [sample, *sample.get('samples', [])]:
        if 'num_frames' in item:
            frame_count(item['num_frames'], 'Sample', error)
    if not ref:
        for item in sample.get('samples', []):
            for key in ('ctrl_img', 'ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'):
                paths = item.get(key, [])
                if isinstance(paths, str):
                    paths = [paths]
                if any(Path(p).suffix.lower() in VIDEO_EXTENSIONS for p in paths):
                    raise error('Video references require MiniMax H3 Ref2VA; ordinary H3 uses first-frame images')
