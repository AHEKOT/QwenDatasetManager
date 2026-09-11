/* MiniMax H3 controls shared by form restoration and model changes. */
window.QdmH3 = (() => {
    const fields = [
        ['distillationMethod', 'De-turbo / distillation handling', 'select', [['both', 'Contrastive Guidance + Training Adapter'], ['cg', 'Contrastive Guidance'], ['ta', 'Training Adapter'], ['none', 'None'], ['dopsd', 'D-OPSD (Ref2VA)']]],
        ['assistantLoraPath', 'Training adapter - local path or Hub file', 'text'],
        ['modelsPath', 'Models folder (ComfyUI layout)', 'text'],
        ['h3Partition', 'Checkpoint partition', 'select', [['fl2va_pruned', 'FL2VA pruned'], ['fl2va', 'FL2VA full'], ['ref2va_pruned', 'Ref2VA pruned'], ['ref2va', 'Ref2VA full']]],
        ['h3DitPath', 'Transformer override (.safetensors)', 'text'],
        ['h3TextEncoderPath', 'Text encoder override (.safetensors)', 'text'],
        ['h3VideoVaePath', 'Video VAE override', 'text'],
        ['h3AudioVaePath', 'Audio VAE override', 'text'],
        ['h3ConfigPath', 'Tokenizer / processor / config folder (original repo or FL2VA)', 'text'],
        ['h3LocalOnly', 'Local files only', 'checkbox'],
        ['h3MaxTextLength', 'Caption token limit (0 = unlimited)', 'number', 0],
        ['audioLossMultiplier', 'Audio loss multiplier', 'number', 0],
        ['h3ImageRefsAsVideo', 'Present image references as static video clips', 'checkbox'],
        ['h3ImageRefVideoFrames', 'Static reference clip frames (17n+5)', 'number', 5],
        ['h3DopsdBleedStrength', 'D-OPSD normal-target loss weight', 'number', 0],
        ['sampleFrames', 'Sample frames (1 = image; otherwise 17n+5)', 'number', 1],
        ['sampleFps', 'Sample FPS', 'number', 1],
        ['h3SampleAudio', 'Generate sample audio', 'checkbox'],
    ];
    const isH3 = key => ['minimax_h3', 'minimax_h3_ref2va'].includes(key);
    const common = {
        qtype: 'trainer-qtype', qtypeTextEncoder: 'trainer-qtype-te', rank: 'trainer-rank',
        timestepType: 'trainer-timestep', cacheTextEmbeddings: 'trainer-cache-text',
        guidanceLoss: 'trainer-guidance-loss', guidanceLossTarget: 'trainer-guidance-loss-target',
        sampleWidth: 'trainer-sample-width', sampleHeight: 'trainer-sample-height',
        guidanceScale: 'trainer-guidance-scale', sampleSteps: 'trainer-sample-steps', lowVram: 'trainer-low-vram',
    };
    const datasetDefaults = { numFrames: 39, fps: 24, autoFrameCount: true, doAudio: true,
        audioNormalize: false, audioPreservePitch: false, doI2v: false,
        shrinkVideoToFrames: true, trimAutoFrameCountTail: true };
    function mount() {
        document.getElementById('trainer-h3-fields').innerHTML = fields.map(([key, label, type, values]) => {
            const attrs = `id="h3-${key}" data-h3-field="${key}"`;
            if (type === 'checkbox') return `<label class="trainer-switch" data-h3-wrap="${key}"><input ${attrs} type="checkbox"><span></span>${label}</label>`;
            const input = type === 'select'
                ? `<select ${attrs}>${values.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}</select>`
                : `<input ${attrs} type="${type}"${type === 'number' ? ` min="${values}" step="any"` : ' spellcheck="false"'}>`;
            return `<label class="trainer-field" data-h3-wrap="${key}"><span>${label}</span>${input}</label>`;
        }).join('');
    }
    function render(key) {
        const active = isH3(key), ref = key === 'minimax_h3_ref2va';
        const card = document.getElementById('trainer-h3-card');
        card.classList.toggle('hidden', !active);
        card.querySelectorAll('input,select').forEach(el => { el.disabled = !active; });
        document.getElementById('h3-sampleFps').readOnly = true;
        ['h3ImageRefsAsVideo', 'h3ImageRefVideoFrames', 'h3DopsdBleedStrength'].forEach(key => {
            card.querySelector(`[data-h3-wrap="${key}"]`).classList.toggle('hidden', !ref);
        });
        document.querySelectorAll('#h3-h3Partition option').forEach(el => {
            el.disabled = el.value.startsWith('ref2va') !== ref;
            el.hidden = el.disabled;
        });
        document.querySelector('#h3-distillationMethod option[value="dopsd"]').disabled = !ref;
    }
    function collect() {
        return Object.fromEntries(fields.map(([key, , type]) => {
            const el = document.getElementById(`h3-${key}`);
            return [key, type === 'checkbox' ? el.checked : type === 'number' ? Number(el.value) : el.value.trim()];
        }));
    }
    function populate(data) {
        fields.forEach(([key, , type]) => {
            const el = document.getElementById(`h3-${key}`);
            if (type === 'checkbox') el.checked = Boolean(data[key]);
            else el.value = data[key] ?? '';
        });
    }
    function datasetFields(settings, index, key, escapeHtml, controls) {
        if (!isH3(key)) return '';
        const numeric = (field, label) => `<label class="trainer-field"><span>${label}</span><input type="number" min="1" data-dataset-field="${field}" data-dataset-index="${index}" value="${escapeHtml(settings[field])}"></label>`;
        const toggle = (field, label) => `<label class="trainer-switch"><input type="checkbox" data-dataset-field="${field}" data-dataset-index="${index}"${settings[field] ? ' checked' : ''}><span></span>${label}</label>`;
        const dopsd = document.getElementById('h3-distillationMethod').value === 'dopsd';
        const selected = settings.controls ?? controls.map(c => c.name);
        const references = dopsd ? '<p class="trainer-help">D-OPSD: external Control folders are ignored; the target is the teacher reference.</p>'
            : controls.map(c => `<label class="trainer-switch"><input type="checkbox" data-dataset-field="controls" data-dataset-index="${index}" value="${c.name}"${selected.includes(c.name) ? ' checked' : ''}><span></span>${c.name} (${c.count})</label>`).join('');
        return `<div class="trainer-toggle-row trainer-resolution-row">${references || '<span>No paired references — generation from text</span>'}</div>${numeric('numFrames', 'Frames (1 or 17n+5)')}${numeric('fps', 'FPS')}
            <div class="trainer-toggle-row trainer-resolution-row">
            ${toggle('autoFrameCount', 'Auto frame count')}${toggle('doAudio', 'Train audio')}
            ${toggle('audioNormalize', 'Normalize audio')}${toggle('audioPreservePitch', 'Preserve pitch')}
            ${key === 'minimax_h3' ? toggle('doI2v', 'First-frame I2V') : ''}
            ${toggle('shrinkVideoToFrames', 'Fit clip to frame count')}${toggle('trimAutoFrameCountTail', 'Trim short tail')}
            </div><p class="trainer-help trainer-resolution-row">Targets: img/ images or videos with matching captions. Control1-3 are optional paired references; Ref2VA accepts image and video references. At 24 FPS the valid video lengths are 5, 22, 39, 56, 73, 90, 107... frames. Images train as single frames.</p>`;
    }
    return { isH3, common, datasetDefaults, mount, render, collect, populate, datasetFields };
})();
