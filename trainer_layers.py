"""Application-side contract for joint QI2 PSD training (no model imports)."""
LAYERED_PRESET = 'qwen_layered_lora'
LAYERED_ARCH = 'qwen_image_2_layered'


def layer_slots(payload, error_type):
    value = payload.get('layerSlots', 4)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
        raise error_type('Layer slots must be an integer from 1 to 20')
    return value


def inspect_psd_dataset(dataset_dir, image_extensions):
    targets = sorted(p for p in (dataset_dir / 'img').iterdir() if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == '.psd')
    controls = sorted(p for p in (dataset_dir / 'Control1').glob('*') if p.is_file() and p.suffix.lower() in image_extensions)
    errors = []
    if not targets:
        errors.append('No PSD targets in img/')
    for paths, label in ((targets, 'PSD target'), (controls, 'Control1')):
        seen = set()
        for path in paths:
            stem = path.stem.casefold()
            if stem in seen:
                errors.append(f'Ambiguous {label} basename: {path.stem}')
            seen.add(stem)
    if targets and any(p.parent != dataset_dir / 'img' and p.suffix.lower() == '.psd' for p in (dataset_dir / 'img').rglob('*')):
        errors.append('Keep PSD targets directly in img/, without nested folders')
    for index in (2, 3):
        if any(p.is_file() and p.suffix.lower() in image_extensions for p in (dataset_dir / f'Control{index}').glob('*')):
            errors.append('PSD layer training supports only optional Control1; remove Control2/Control3 from this dataset')
    control_stems = {p.stem.casefold() for p in controls}
    paired = sum(p.stem.casefold() in control_stems for p in targets)
    return {
        'psdTargetCount': len(targets), 'psdPairedCount': paired,
        'psdGenerationCount': len(targets) - paired,
        'psdCaptionCount': sum(p.with_suffix('.txt').is_file() for p in targets),
        'psdValid': bool(targets) and not errors, 'psdErrors': errors,
    }


def validate_layered_payload(payload, error_type):
    layer_slots(payload, error_type)
    if payload.get('model') != 'qwen_image_2':
        raise error_type('Joint PSD layer training requires Qwen Image 2.1')
    if payload.get('batchSize', 1) != 1:
        raise error_type('Joint PSD layers require batch size 1; use gradient accumulation')
    for key in ('validationEnabled', 'diffOutputPreservation', 'blankPromptPreservation',
                'guidanceLoss', 'differentialGuidance', 'advancedProcess'):
        if payload.get(key):
            raise error_type(f'{key} is not supported by the PSD layer preset')
    if payload.get('lossType', 'mse') != 'mse':
        raise error_type('Joint PSD layers require flow-matching MSE loss')
    datasets = payload.get('datasets', [])
    samples = payload.get('samples', [])
    if not isinstance(datasets, list) or not isinstance(samples, list):
        raise error_type('Datasets and samples must be lists')
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise error_type('Invalid PSD dataset configuration')
        if any(dataset.get(key) for key in ('flipX', 'flipY', 'isRegularization')):
            raise error_type('PSD layer datasets support shared resize/crop; disable flips and regularization')
    for sample in samples:
        if not isinstance(sample, dict):
            raise error_type('Invalid PSD sample configuration')
        if any(sample.get(key) for key in ('ctrlImg2', 'ctrlImg3', 'ctrl_img_2', 'ctrl_img_3', 'image')):
            raise error_type('PSD samples accept an uploaded Control1 or no reference; legacy dataset-image samples are unsupported')


def is_layered_job(row):
    import json
    return json.loads(row['job_config']).get('meta', {}).get('qdm', {}).get('trainingPreset') == LAYERED_PRESET
