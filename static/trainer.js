(() => {
    'use strict';

    const state = {
        models: [],
        qtypes: [],
        datasets: [],
        jobs: [],
        gpus: [],
        selectedDatasets: [],
        samples: [],
        validationItems: [],
        selectedJobId: null,
        editingJobId: null,
        filter: 'all',
        trainerInstalled: false,
        pollTimer: null,
    };

    const $ = id => document.getElementById(id);
    const editorView = $('trainer-editor-view');
    const detailView = $('trainer-detail-view');
    const form = $('trainer-job-form');
    const jobsList = $('trainer-jobs-list');
    const formError = $('trainer-form-error');

    const escapeHtml = value => String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');

    async function api(url, options = {}) {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...(options.body ? { 'Content-Type': 'application/json' } : {}),
                ...(options.headers || {}),
            },
        });
        let data = {};
        try { data = await response.json(); } catch (_) { /* empty response */ }
        if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
        return data;
    }

    function showToast(message, isError = false) {
        const toast = $('trainer-toast');
        toast.textContent = message;
        toast.classList.toggle('is-error', isError);
        toast.classList.remove('hidden');
        window.clearTimeout(showToast.timer);
        showToast.timer = window.setTimeout(() => toast.classList.add('hidden'), 3500);
    }

    function showFormError(message = '') {
        formError.textContent = message;
        formError.classList.toggle('hidden', !message);
        if (message) formError.scrollIntoView({ block: 'nearest' });
    }

    function selectedModel() {
        return state.models.find(model => model.key === $('trainer-model').value) || state.models[0];
    }

    function currentJob() {
        return state.jobs.find(job => job.id === state.selectedJobId) || null;
    }

    function renderRuntime() {
        const pill = $('trainer-runtime-status');
        pill.classList.remove('is-checking', 'is-ready', 'is-missing');
        pill.classList.add(state.trainerInstalled ? 'is-ready' : 'is-missing');
        pill.textContent = state.trainerInstalled ? 'CUDA trainer ready' : 'Trainer not installed';
        $('trainer-runtime-banner').classList.toggle('hidden', state.trainerInstalled);
    }

    function renderModelOptions() {
        const modelSelect = $('trainer-model');
        const previous = modelSelect.value;
        modelSelect.innerHTML = state.models.map(model =>
            `<option value="${escapeHtml(model.key)}">${escapeHtml(model.label)}</option>`
        ).join('');
        if (state.models.some(model => model.key === previous)) modelSelect.value = previous;
        renderModelFields();
    }

    function renderModelFields(resetPath = false) {
        const model = selectedModel();
        if (!model) return;
        if (resetPath || !$('trainer-model-path').value) $('trainer-model-path').value = model.modelPath;
        $('trainer-model-license').textContent = model.license;
        $('trainer-model-license').title = model.gated ? 'Gated Hugging Face model' : model.license;
        $('trainer-model-gate').classList.toggle('hidden', !model.gateUrl);
        $('trainer-model-gate').href = model.gateUrl || '#';
        $('trainer-noise-scheduler').value = model.noiseScheduler === 'flowmatch' ? 'FlowMatch' : model.noiseScheduler;
        $('trainer-unload-text-wrap').classList.toggle('hidden', !model.allowUnloadTextEncoder);
        if (!model.allowUnloadTextEncoder) $('trainer-unload-text').checked = false;

        const qtype = $('trainer-qtype');
        const previous = qtype.value || model.defaultQtype;
        const standardOptions = state.qtypes.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(qtypeLabel(value))}</option>`).join('');
        const araOptions = Object.entries(model.accuracyRecoveryAdapters || {}).map(([label, value]) =>
            `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`
        ).join('');
        qtype.innerHTML = araOptions
            ? `<optgroup label="Standard">${state.qtypes.slice(0, 2).map(value => `<option value="${escapeHtml(value)}">${escapeHtml(qtypeLabel(value))}</option>`).join('')}</optgroup><optgroup label="Accuracy Recovery Adapters">${araOptions}</optgroup><optgroup label="Additional quantization options">${state.qtypes.slice(2).map(value => `<option value="${escapeHtml(value)}">${escapeHtml(qtypeLabel(value))}</option>`).join('')}</optgroup>`
            : standardOptions;
        const allowed = new Set([...state.qtypes, ...Object.values(model.accuracyRecoveryAdapters || {})]);
        qtype.value = allowed.has(previous) ? previous : model.defaultQtype;

        const qtypeTe = $('trainer-qtype-te');
        const previousTe = qtypeTe.value || model.defaultQtype;
        qtypeTe.innerHTML = standardOptions;
        qtypeTe.value = state.qtypes.includes(previousTe) ? previousTe : model.defaultQtype;
    }

    function qtypeLabel(value) {
        const labels = {
            '': 'None', qfloat8: 'qfloat8 (default)', float8: 'float8', convrot8: '8bit convrot',
            convrot4: '4bit convrot (nvfp4)', nvfp4: 'nvfp4 (4bit weight only)',
            convrotint7: '7bit convrot', convrotint6: '6bit convrot', convrotint5: '5bit convrot',
            convrotint4: '4bit convrot', convrotint3: '3bit convrot', convrotint2: '2bit convrot',
            convrotbitnet: '1.58bit convrot (bitnet)', uint7: '7 bit', uint6: '6 bit',
            uint5: '5 bit', uint4: '4 bit', uint3: '3 bit', uint2: '2 bit',
        };
        return labels[value] || value;
    }

    function renderGpuOptions() {
        const select = $('trainer-gpu');
        const previous = select.value || '0';
        if (!state.gpus.length) {
            select.innerHTML = '<option value="0">GPU 0 (not detected here)</option>';
            return;
        }
        select.innerHTML = state.gpus.map(gpu =>
            `<option value="${escapeHtml(gpu.index)}">GPU ${escapeHtml(gpu.index)} · ${escapeHtml(gpu.name)} · ${Math.round(gpu.memoryFreeMb / 1024)} / ${Math.round(gpu.memoryTotalMb / 1024)} GB free</option>`
        ).join('');
        if (state.gpus.some(gpu => gpu.index === previous)) select.value = previous;
    }

    function jobVisible(job) {
        if (state.filter === 'active') return ['queued', 'running', 'stopping'].includes(job.status);
        if (state.filter === 'finished') return ['completed', 'error'].includes(job.status);
        return true;
    }

    function renderJobs() {
        const scrollTop = jobsList.scrollTop;
        const jobs = state.jobs.filter(jobVisible);
        $('trainer-job-count').textContent = `${state.jobs.length} training job${state.jobs.length === 1 ? '' : 's'}`;
        if (!jobs.length) {
            jobsList.innerHTML = `<div class="trainer-empty-list"><strong>${state.jobs.length ? 'No matching jobs' : 'No jobs yet'}</strong><span>${state.jobs.length ? 'Try another filter.' : 'Create a LoRA training job from your existing datasets.'}</span></div>`;
            jobsList.scrollTop = scrollTop;
            return;
        }
        jobsList.innerHTML = jobs.map(job => {
            const model = state.models.find(item => item.key === job.form?.model);
            return `<button class="trainer-job-item${job.id === state.selectedJobId ? ' is-selected' : ''}" type="button" data-job-id="${escapeHtml(job.id)}">
                <span class="trainer-job-item-top"><strong>${escapeHtml(job.name)}</strong><span class="trainer-mini-status" data-status="${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></span>
                <span class="trainer-job-item-meta"><span>${escapeHtml(model?.label || job.config?.config?.process?.[0]?.model?.arch || 'Edit model')}</span><span>${escapeHtml(job.step)} / ${escapeHtml(job.total_steps)}</span></span>
                <span class="trainer-job-item-progress"><span style="width:${Number(job.progress) || 0}%"></span></span>
            </button>`;
        }).join('');
        jobsList.scrollTop = scrollTop;
    }

    function renderDatasetPicker() {
        const select = $('trainer-dataset-select');
        const previous = select.value;
        const selectedNames = new Set(state.selectedDatasets.map(item => item.name));
        select.innerHTML = '<option value="">Select dataset…</option>' + state.datasets.map(dataset =>
            `<option value="${escapeHtml(dataset.name)}"${selectedNames.has(dataset.name) || !dataset.valid ? ' disabled' : ''}>${escapeHtml(dataset.name)} · ${dataset.targetCount} targets · ${dataset.controls.length} controls${dataset.valid ? '' : ' · not ready'}</option>`
        ).join('');
        if (state.datasets.some(dataset => dataset.name === previous && !selectedNames.has(previous))) select.value = previous;
    }

    function addDataset(name, settings = {}, allowDuplicate = false) {
        const inspection = state.datasets.find(dataset => dataset.name === name);
        if (!inspection || (!allowDuplicate && state.selectedDatasets.some(dataset => dataset.name === name))) return;
        state.selectedDatasets.push({
            name,
            repeats: Number(settings.repeats ?? 1),
            weight: Number(settings.weight ?? 1),
            batchSize: Number(settings.batchSize ?? 1),
            captionDropout: Number(settings.captionDropout ?? 0.05),
            defaultCaption: settings.defaultCaption || '',
            captionExtension: settings.captionExtension || 'txt',
            resolutions: Array.isArray(settings.resolutions) ? settings.resolutions.map(Number) : [512, 768, 1024],
            cacheLatents: Boolean(settings.cacheLatents),
            isRegularization: Boolean(settings.isRegularization),
            flipX: Boolean(settings.flipX),
            flipY: Boolean(settings.flipY),
        });
        renderSelectedDatasets();
    }

    function resolutionChip(datasetIndex, value, checked) {
        return `<label class="trainer-resolution-chip"><input type="checkbox" data-dataset-field="resolution" data-dataset-index="${datasetIndex}" value="${value}"${checked ? ' checked' : ''}><span>${value}</span></label>`;
    }

    function renderSelectedDatasets() {
        const container = $('trainer-selected-datasets');
        const selectedNames = new Set(state.selectedDatasets.map(item => item.name));
        const fallbackName = state.selectedDatasets[0]?.name || '';
        [...state.samples, ...state.validationItems].forEach(item => {
            if (!selectedNames.has(item.dataset)) item.dataset = fallbackName;
        });
        if (!state.selectedDatasets.length) {
            container.innerHTML = `<div class="trainer-empty-datasets"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M4 7h16v12H4zM7 4h10v3H7z"></path><path d="M8 11h8M8 15h5"></path></svg><strong>No datasets selected</strong><span>Choose a dataset above. Target and control folders will be mapped automatically.</span></div>`;
            $('trainer-dataset-summary').textContent = 'Select one or more managed datasets';
            renderDatasetPicker();
            renderSamples();
            renderValidationItems();
            return;
        }

        container.innerHTML = state.selectedDatasets.map((settings, index) => {
            const dataset = state.datasets.find(item => item.name === settings.name);
            if (!dataset) return '';
            const controls = dataset.controls.map(item => `<span>${escapeHtml(item.name)} · ${item.count}</span>`).join('');
            const warnings = dataset.warnings.length ? `<div class="trainer-dataset-warning">${escapeHtml(dataset.warnings.join(' · '))}</div>` : '';
            return `<article class="trainer-dataset-card" data-dataset-card="${index}">
                <div class="trainer-dataset-title">
                    <strong title="${escapeHtml(dataset.name)}">${escapeHtml(dataset.name)}</strong>
                    <span title="${escapeHtml(dataset.targetPath)}">…/Datasets/${escapeHtml(dataset.name)}/img</span>
                    <div class="trainer-dataset-paths"><span>Target · ${dataset.targetCount}</span>${controls}</div>
                </div>
                <div class="trainer-dataset-metrics">
                    <div class="trainer-dataset-metric"><strong>${dataset.targetCount}</strong><span>targets</span></div>
                    <div class="trainer-dataset-metric"><strong>${dataset.captionCount}</strong><span>captions</span></div>
                    <div class="trainer-dataset-metric"><strong>${dataset.controls.length}</strong><span>controls</span></div>
                </div>
                <div class="trainer-dataset-settings">
                    <label class="trainer-field"><span>Repeats</span><input data-dataset-field="repeats" data-dataset-index="${index}" type="number" min="1" max="1000" value="${settings.repeats}"></label>
                    <label class="trainer-field"><span>LoRA weight</span><input data-dataset-field="weight" data-dataset-index="${index}" type="number" min="0" max="100" step="0.1" value="${settings.weight}"></label>
                    <label class="trainer-field"><span>Batch size</span><input data-dataset-field="batchSize" data-dataset-index="${index}" type="number" min="1" max="128" value="${settings.batchSize}"></label>
                    <label class="trainer-field"><span>Caption dropout</span><input data-dataset-field="captionDropout" data-dataset-index="${index}" type="number" min="0" max="1" step="0.01" value="${settings.captionDropout}"></label>
                    <label class="trainer-field trainer-dataset-caption"><span>Default caption</span><input data-dataset-field="defaultCaption" data-dataset-index="${index}" type="text" value="${escapeHtml(settings.defaultCaption)}"></label>
                    <label class="trainer-field"><span>Caption extension</span><select data-dataset-field="captionExtension" data-dataset-index="${index}"><option value="txt"${settings.captionExtension === 'txt' ? ' selected' : ''}>txt</option><option value="json"${settings.captionExtension === 'json' ? ' selected' : ''}>json</option><option value="caption"${settings.captionExtension === 'caption' ? ' selected' : ''}>caption</option></select></label>
                    <div class="trainer-resolution-row"><span>Resolutions</span>${[256, 512, 768, 1024, 1280, 1328, 1536, 2048].map(value => resolutionChip(index, value, settings.resolutions.includes(value))).join('')}</div>
                    <div class="trainer-toggle-row trainer-resolution-row">
                        <label class="trainer-switch"><input data-dataset-field="cacheLatents" data-dataset-index="${index}" type="checkbox"${settings.cacheLatents ? ' checked' : ''}><span></span>Cache latents</label>
                        <label class="trainer-switch"><input data-dataset-field="isRegularization" data-dataset-index="${index}" type="checkbox"${settings.isRegularization ? ' checked' : ''}><span></span>Is regularization</label>
                        <label class="trainer-switch"><input data-dataset-field="flipX" data-dataset-index="${index}" type="checkbox"${settings.flipX ? ' checked' : ''}><span></span>Flip X</label>
                        <label class="trainer-switch"><input data-dataset-field="flipY" data-dataset-index="${index}" type="checkbox"${settings.flipY ? ' checked' : ''}><span></span>Flip Y</label>
                    </div>
                </div>
                <div class="trainer-dataset-actions">
                    <button class="trainer-icon-btn" type="button" data-duplicate-dataset="${index}" aria-label="Duplicate ${escapeHtml(dataset.name)}" title="Duplicate dataset"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2"></rect><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"></path></svg></button>
                    <button class="trainer-icon-btn trainer-remove-dataset" type="button" data-remove-dataset="${index}" aria-label="Remove ${escapeHtml(dataset.name)}" title="Remove dataset"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg></button>
                </div>
                ${warnings}
            </article>`;
        }).join('');
        const totalTargets = state.selectedDatasets.reduce((sum, item) => sum + (state.datasets.find(dataset => dataset.name === item.name)?.targetCount || 0), 0);
        $('trainer-dataset-summary').textContent = `${state.selectedDatasets.length} dataset${state.selectedDatasets.length === 1 ? '' : 's'} · ${totalTargets} target images`;
        renderDatasetPicker();
        renderSamples();
        renderValidationItems();
    }

    function updateDatasetSetting(input) {
        const index = Number(input.dataset.datasetIndex);
        const field = input.dataset.datasetField;
        const dataset = state.selectedDatasets[index];
        if (!dataset) return;
        if (field === 'resolution') {
            const value = Number(input.value);
            const resolutions = new Set(dataset.resolutions);
            input.checked ? resolutions.add(value) : resolutions.delete(value);
            dataset.resolutions = [...resolutions].sort((a, b) => a - b);
        } else if (input.type === 'checkbox') {
            dataset[field] = input.checked;
        } else if (input.type === 'number') {
            dataset[field] = Number(input.value);
        } else {
            dataset[field] = input.value;
        }
    }

    function selectedDatasetOptions(value) {
        const names = [...new Set(state.selectedDatasets.map(item => item.name))];
        return names.map(name => `<option value="${escapeHtml(name)}"${name === value ? ' selected' : ''}>${escapeHtml(name)}</option>`).join('');
    }

    function sampleControlPath(sample, controlIndex) {
        return sample[`ctrlImg${controlIndex}`] || sample[`ctrl_img_${controlIndex}`] || '';
    }

    function sampleImagePreviewUrl(path) {
        const filename = String(path || '').split(/[\\/]/).pop();
        return filename ? `/api/trainer/sample-images/${encodeURIComponent(filename)}` : '';
    }

    function renderSampleImageSlot(sample, sampleIndex, controlIndex) {
        const path = sampleControlPath(sample, controlIndex);
        const addLabel = controlIndex === 1 ? 'Add Image 1' : `Add Additional Image ${controlIndex}`;
        const preview = sampleImagePreviewUrl(path);
        const previewContent = path
            ? `<img src="${escapeHtml(preview)}" alt="Image to edit ${controlIndex}">`
            : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"></rect><circle cx="8.5" cy="9" r="1.5"></circle><path d="m21 15-5-5L5 20"></path></svg><strong>${escapeHtml(addLabel)}</strong><span>Click or drop</span>`;
        return `<div class="trainer-sample-image-slot${path ? ' has-image' : ''}" data-sample-image-slot="${sampleIndex}-${controlIndex}">
            <button class="trainer-sample-image-picker" type="button" data-pick-sample-image="${sampleIndex}-${controlIndex}" aria-label="${path ? 'Replace' : 'Add'} image to edit ${controlIndex}">${previewContent}</button>
            ${path ? `<button class="trainer-sample-image-clear" type="button" data-clear-sample-image="${sampleIndex}-${controlIndex}" aria-label="Clear image to edit ${controlIndex}" title="Clear image"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg></button>` : ''}
            <input class="hidden" type="file" accept="image/png,image/jpeg,image/webp" data-sample-file="${sampleIndex}-${controlIndex}" tabindex="-1">
            <div class="trainer-sample-upload-overlay hidden" aria-live="polite"><div class="trainer-sample-upload-track"><span></span></div><small>Uploading… <b>0%</b></small></div>
        </div>`;
    }

    function renderSamples() {
        const container = $('trainer-sample-items');
        $('trainer-sample-count').textContent = `Sample prompts (${state.samples.length})`;
        container.innerHTML = state.samples.map((sample, index) => `<article class="trainer-repeat-card trainer-sample-card">
            <button class="trainer-icon-btn trainer-sample-remove" type="button" data-remove-sample="${index}" aria-label="Remove sample ${index + 1}" title="Remove sample"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg></button>
            <div class="trainer-sample-editor">
                <div class="trainer-sample-fields">
                    <label class="trainer-field trainer-sample-instruction"><span>Edit instruction</span><input data-sample-field="prompt" data-sample-index="${index}" type="text" value="${escapeHtml(sample.prompt || '')}" placeholder="Describe the edit" required></label>
                    <div class="trainer-sample-parameters">
                        <label class="trainer-field"><span>Width</span><input data-sample-field="width" data-sample-index="${index}" type="text" inputmode="numeric" value="${escapeHtml(sample.width ?? '')}" placeholder="${escapeHtml($('trainer-sample-width').value)} (default)"></label>
                        <label class="trainer-field"><span>Height</span><input data-sample-field="height" data-sample-index="${index}" type="text" inputmode="numeric" value="${escapeHtml(sample.height ?? '')}" placeholder="${escapeHtml($('trainer-sample-height').value)} (default)"></label>
                        <label class="trainer-field"><span>Seed</span><input data-sample-field="seed" data-sample-index="${index}" type="text" inputmode="numeric" value="${escapeHtml(sample.seed ?? '')}" placeholder="${escapeHtml(Number($('trainer-sample-seed').value) + ($('trainer-walk-seed').checked ? index : 0))} (default)"></label>
                        <label class="trainer-field"><span>LoRA scale</span><input data-sample-field="networkMultiplier" data-sample-index="${index}" type="text" inputmode="decimal" value="${escapeHtml(sample.networkMultiplier ?? sample.network_multiplier ?? '')}" placeholder="1.0 (default)"></label>
                    </div>
                </div>
                <fieldset class="trainer-sample-images"><legend>Images to edit</legend><div>${[1, 2, 3].map(controlIndex => renderSampleImageSlot(sample, index, controlIndex)).join('')}</div></fieldset>
            </div>
        </article>`).join('');
    }

    function renderSamplesPreservingPlace(focusSelector = '') {
        const workspace = document.querySelector('.trainer-workspace');
        const scrollTop = workspace?.scrollTop || 0;
        const windowScrollX = window.scrollX;
        const windowScrollY = window.scrollY;
        renderSamples();
        if (workspace) {
            workspace.scrollTop = scrollTop;
        }
        window.scrollTo(windowScrollX, windowScrollY);
        requestAnimationFrame(() => {
            if (workspace) workspace.scrollTop = scrollTop;
            window.scrollTo(windowScrollX, windowScrollY);
            if (focusSelector) $('trainer-sample-items').querySelector(focusSelector)?.focus({ preventScroll: true });
        });
    }

    function uploadSampleImage(input, droppedFile = null) {
        const file = droppedFile || input.files?.[0];
        if (!file) return;
        const [sampleIndexText, controlIndexText] = input.dataset.sampleFile.split('-');
        const sampleIndex = Number(sampleIndexText);
        const controlIndex = Number(controlIndexText);
        const sample = state.samples[sampleIndex];
        if (!sample) return;
        const slot = input.closest('.trainer-sample-image-slot');
        const overlay = slot.querySelector('.trainer-sample-upload-overlay');
        const progressBar = overlay.querySelector('span');
        const progressText = overlay.querySelector('b');
        overlay.classList.remove('hidden');
        const requestValue = new XMLHttpRequest();
        requestValue.open('POST', '/api/trainer/sample-images');
        requestValue.upload.addEventListener('progress', event => {
            if (!event.lengthComputable) return;
            const percent = Math.round((event.loaded / event.total) * 100);
            progressBar.style.width = `${percent}%`;
            progressText.textContent = `${percent}%`;
        });
        requestValue.addEventListener('load', () => {
            let result = {};
            try { result = JSON.parse(requestValue.responseText || '{}'); } catch (_) { /* handled below */ }
            if (requestValue.status < 200 || requestValue.status >= 300 || !result.path) {
                overlay.classList.add('hidden');
                showToast(result.error || `Upload failed (${requestValue.status})`, true);
                return;
            }
            sample[`ctrlImg${controlIndex}`] = result.path;
            delete sample[`ctrl_img_${controlIndex}`];
            renderSamplesPreservingPlace(`[data-pick-sample-image="${sampleIndex}-${controlIndex}"]`);
        });
        requestValue.addEventListener('error', () => {
            overlay.classList.add('hidden');
            showToast('Sample image upload failed.', true);
        });
        const body = new FormData();
        body.append('files', file);
        requestValue.send(body);
    }

    function renderValidationItems() {
        const container = $('trainer-validation-items');
        container.innerHTML = state.validationItems.map((item, index) => `<article class="trainer-repeat-card">
            <div class="trainer-repeat-card-heading"><strong>Validation image ${index + 1}</strong><button class="trainer-icon-btn" type="button" data-remove-validation="${index}" aria-label="Remove validation image"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg></button></div>
            <div class="trainer-repeat-grid trainer-validation-item-grid">
                <label class="trainer-field"><span>Dataset</span><select data-validation-field="dataset" data-validation-index="${index}">${selectedDatasetOptions(item.dataset)}</select></label>
                <label class="trainer-field"><span>Target image <small>blank = first</small></span><input data-validation-field="image" data-validation-index="${index}" type="text" value="${escapeHtml(item.image || '')}" placeholder="image.png"></label>
                <label class="trainer-field trainer-repeat-prompt"><span>Prompt</span><input data-validation-field="prompt" data-validation-index="${index}" type="text" value="${escapeHtml(item.prompt || '')}"></label>
            </div>
        </article>`).join('');
    }

    function updateRepeatedItem(input, collection, indexKey, fieldKey) {
        const item = collection[Number(input.dataset[indexKey])];
        if (!item) return;
        const field = input.dataset[fieldKey];
        item[field] = input.type === 'number' ? (input.value === '' ? '' : Number(input.value)) : input.value;
    }

    function renderConditionalOptions() {
        $('trainer-lokr-factor-field').classList.toggle('hidden', $('trainer-network-type').value !== 'lokr');
        $('trainer-rank-field').classList.toggle('hidden', $('trainer-network-type').value !== 'lora');
        $('trainer-ema-decay-field').classList.toggle('hidden', !$('trainer-use-ema').checked);
        $('trainer-dop-options').classList.toggle('hidden', !$('trainer-dop').checked);
        $('trainer-bpp-options').classList.toggle('hidden', !$('trainer-bpp').checked);
        $('trainer-guidance-loss-options').classList.toggle('hidden', !$('trainer-guidance-loss').checked);
        $('trainer-differential-guidance-options').classList.toggle('hidden', !$('trainer-differential-guidance').checked);
        $('trainer-validation-options').classList.toggle('hidden', !$('trainer-validation-enabled').checked);
        $('trainer-offload-sliders').classList.toggle('hidden', !$('trainer-layer-offloading').checked);
    }

    function syncSlider(id) {
        const input = $(id);
        const min = Number(input.min || 0);
        const max = Number(input.max || 100);
        const value = Math.min(max, Math.max(min, Number(input.value) || 0));
        const percent = max === min ? 0 : ((value - min) / (max - min)) * 100;
        input.value = String(value);
        input.style.setProperty('--slider-fill', `${percent}%`);
        $(`${id}-value`).textContent = String(Math.round(value));
    }

    function numberValue(id) { return Number($(id).value); }
    function checked(id) { return $(id).checked; }

    function collectForm(includeAdvanced = true) {
        const payload = {
            name: $('trainer-name').value.trim(),
            gpuIds: $('trainer-gpu').value,
            triggerWord: $('trainer-trigger').value.trim(),
            model: $('trainer-model').value,
            modelPath: $('trainer-model-path').value.trim(),
            qtype: $('trainer-qtype').value,
            qtypeTextEncoder: $('trainer-qtype-te').value,
            lowVram: checked('trainer-low-vram'),
            matchTargetResolution: checked('trainer-match-resolution'),
            compileModel: checked('trainer-compile'),
            networkType: $('trainer-network-type').value,
            rank: numberValue('trainer-rank'),
            lokrFactor: numberValue('trainer-lokr-factor'),
            saveDtype: $('trainer-save-dtype').value,
            maxSaves: numberValue('trainer-max-saves'),
            batchSize: numberValue('trainer-batch-size'),
            gradientAccumulation: numberValue('trainer-gradient-accumulation'),
            steps: numberValue('trainer-steps'),
            learningRate: numberValue('trainer-learning-rate'),
            optimizer: $('trainer-optimizer').value,
            weightDecay: numberValue('trainer-weight-decay'),
            timestepType: $('trainer-timestep').value,
            timestepBias: $('trainer-timestep-bias').value,
            lossType: $('trainer-loss').value,
            saveEvery: numberValue('trainer-save-every'),
            cacheTextEmbeddings: checked('trainer-cache-text'),
            unloadTextEncoder: checked('trainer-unload-text'),
            useEma: checked('trainer-use-ema'),
            emaDecay: numberValue('trainer-ema-decay'),
            diffOutputPreservation: checked('trainer-dop'),
            dopMultiplier: numberValue('trainer-dop-multiplier'),
            dopClass: $('trainer-dop-class').value.trim(),
            blankPromptPreservation: checked('trainer-bpp'),
            bppMultiplier: numberValue('trainer-bpp-multiplier'),
            guidanceLoss: checked('trainer-guidance-loss'),
            guidanceLossTarget: numberValue('trainer-guidance-loss-target'),
            differentialGuidance: checked('trainer-differential-guidance'),
            differentialGuidanceScale: numberValue('trainer-differential-guidance-scale'),
            validationEnabled: checked('trainer-validation-enabled'),
            validateEvery: numberValue('trainer-validation-every'),
            validationResolution: numberValue('trainer-validation-resolution'),
            validationSigmas: $('trainer-validation-sigmas').value.split(',').map(Number),
            validationItems: state.validationItems.map(item => ({ ...item })),
            sampleEvery: numberValue('trainer-sample-every'),
            sampleStartStep: numberValue('trainer-sample-start'),
            sampler: $('trainer-sampler').value,
            guidanceScale: numberValue('trainer-guidance-scale'),
            sampleSteps: numberValue('trainer-sample-steps'),
            sampleWidth: numberValue('trainer-sample-width'),
            sampleHeight: numberValue('trainer-sample-height'),
            sampleSeed: numberValue('trainer-sample-seed'),
            walkSeed: checked('trainer-walk-seed'),
            skipFirstSample: checked('trainer-skip-first-sample'),
            forceFirstSample: checked('trainer-force-first-sample'),
            disableSampling: checked('trainer-disable-sampling'),
            samples: state.samples.map(item => ({ ...item })),
            layerOffloading: checked('trainer-layer-offloading'),
            transformerOffload: numberValue('trainer-transformer-offload') / 100,
            textEncoderOffload: numberValue('trainer-text-offload') / 100,
            datasets: state.selectedDatasets.map(item => ({ ...item, resolutions: [...item.resolutions] })),
        };
        if (includeAdvanced && checked('trainer-use-advanced-config') && $('trainer-advanced-config-json').value.trim()) {
            payload.advancedProcess = $('trainer-advanced-config-json').value.trim();
        }
        return payload;
    }

    function validateForm(payload) {
        if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/.test(payload.name)) return 'Training name must use letters, numbers, dots, underscores or hyphens.';
        if (!payload.modelPath) return 'Enter a Hugging Face model name or local path.';
        if (!payload.datasets.length) return 'Select at least one dataset.';
        const noResolutions = payload.datasets.find(dataset => !dataset.resolutions.length);
        if (noResolutions) return `Select at least one resolution for ${noResolutions.name}.`;
        if (!payload.disableSampling && !payload.samples.length) return 'Add at least one sample prompt or disable sampling.';
        if (!payload.disableSampling) {
            const sampleWithoutImage = payload.samples.find(sample => ![1, 2, 3].some(index => sampleControlPath(sample, index)) && !sample.image);
            if (sampleWithoutImage) return 'Add at least one image to edit for every sample.';
            const sampleWithoutInstruction = payload.samples.find(sample => [1, 2, 3].some(index => sampleControlPath(sample, index)) && !String(sample.prompt || '').trim());
            if (sampleWithoutInstruction) return 'Enter an edit instruction for every sample.';
        }
        if (payload.validationEnabled && !payload.validationItems.length) return 'Add at least one validation image.';
        if (payload.advancedProcess) {
            try { JSON.parse(payload.advancedProcess); } catch (error) { return `Advanced process JSON is invalid: ${error.message}`; }
        }
        return '';
    }

    async function saveJob(queueAfter = false) {
        const payload = collectForm();
        const error = validateForm(payload);
        if (error) { showFormError(error); return; }
        showFormError('');
        setSaving(true);
        try {
            const url = state.editingJobId ? `/api/trainer/jobs/${state.editingJobId}` : '/api/trainer/jobs';
            const method = state.editingJobId ? 'PUT' : 'POST';
            const result = await api(url, { method, body: JSON.stringify(payload) });
            state.selectedJobId = result.job.id;
            state.editingJobId = null;
            if (queueAfter) await api(`/api/trainer/jobs/${result.job.id}/start`, { method: 'POST' });
            await refreshState();
            showDetail(result.job.id);
            showToast(queueAfter ? 'Job saved and added to the queue.' : 'Training job saved.');
        } catch (errorValue) {
            showFormError(errorValue.message);
        } finally {
            setSaving(false);
        }
    }

    function setSaving(saving) {
        $('trainer-save-btn').disabled = saving;
        $('trainer-save-start-btn').disabled = saving;
        $('trainer-save-btn').textContent = saving ? 'Saving…' : 'Save job';
    }

    function setInput(id, value) {
        if (value === undefined || value === null) return;
        const input = $(id);
        if (input.type === 'checkbox') input.checked = Boolean(value);
        else input.value = String(value);
    }

    function populateForm(payload = {}) {
        const defaults = {
            name: 'qdm_edit_lora_v1', gpuIds: '0', triggerWord: '', model: state.models[0]?.key || 'qwen_image_edit_2511',
            modelPath: '', qtype: 'qfloat8', qtypeTextEncoder: 'qfloat8', lowVram: true,
            matchTargetResolution: false, compileModel: false, networkType: 'lora', rank: 32, lokrFactor: -1, saveDtype: 'bf16', maxSaves: 4,
            batchSize: 1, gradientAccumulation: 1, steps: 3000, learningRate: 0.0001, optimizer: 'adamw8bit',
            weightDecay: 0.0001, timestepType: 'weighted', timestepBias: 'balanced', lossType: 'mse', saveEvery: 250,
            cacheTextEmbeddings: false, unloadTextEncoder: false, useEma: false, emaDecay: 0.99,
            diffOutputPreservation: false, dopMultiplier: 1, dopClass: 'person', blankPromptPreservation: false,
            bppMultiplier: 1, guidanceLoss: false, guidanceLossTarget: 4, differentialGuidance: false,
            differentialGuidanceScale: 3, validationEnabled: false, validateEvery: 1, validationResolution: 1024,
            validationSigmas: [0.5], validationItems: [], sampleEvery: 250, sampleStartStep: 0,
            sampler: 'flowmatch', guidanceScale: 4, sampleSteps: 30, sampleWidth: 1024, sampleHeight: 1024,
            sampleSeed: 42, walkSeed: true, skipFirstSample: false, forceFirstSample: false, disableSampling: true, samples: [],
            layerOffloading: false, transformerOffload: 1, textEncoderOffload: 1, advancedProcess: '', datasets: [],
        };
        const data = { ...defaults, ...payload };
        setInput('trainer-name', data.name);
        setInput('trainer-gpu', data.gpuIds);
        setInput('trainer-trigger', data.triggerWord);
        setInput('trainer-model', data.model);
        renderModelFields();
        setInput('trainer-model-path', data.modelPath || selectedModel()?.modelPath || '');
        setInput('trainer-qtype', data.qtype);
        setInput('trainer-qtype-te', data.qtypeTextEncoder);
        setInput('trainer-low-vram', data.lowVram);
        setInput('trainer-match-resolution', data.matchTargetResolution);
        setInput('trainer-compile', data.compileModel);
        setInput('trainer-network-type', data.networkType);
        setInput('trainer-rank', data.rank);
        setInput('trainer-lokr-factor', data.lokrFactor);
        setInput('trainer-save-dtype', data.saveDtype);
        setInput('trainer-max-saves', data.maxSaves);
        setInput('trainer-batch-size', data.batchSize);
        setInput('trainer-gradient-accumulation', data.gradientAccumulation);
        setInput('trainer-steps', data.steps);
        setInput('trainer-learning-rate', data.learningRate);
        setInput('trainer-optimizer', data.optimizer);
        setInput('trainer-weight-decay', data.weightDecay);
        setInput('trainer-timestep', data.timestepType);
        setInput('trainer-timestep-bias', data.timestepBias);
        setInput('trainer-loss', data.lossType);
        setInput('trainer-save-every', data.saveEvery);
        setInput('trainer-cache-text', data.cacheTextEmbeddings);
        setInput('trainer-unload-text', data.unloadTextEncoder);
        setInput('trainer-use-ema', data.useEma);
        setInput('trainer-ema-decay', data.emaDecay);
        setInput('trainer-dop', data.diffOutputPreservation);
        setInput('trainer-dop-multiplier', data.dopMultiplier);
        setInput('trainer-dop-class', data.dopClass);
        setInput('trainer-bpp', data.blankPromptPreservation);
        setInput('trainer-bpp-multiplier', data.bppMultiplier);
        setInput('trainer-guidance-loss', data.guidanceLoss);
        setInput('trainer-guidance-loss-target', data.guidanceLossTarget);
        setInput('trainer-differential-guidance', data.differentialGuidance);
        setInput('trainer-differential-guidance-scale', data.differentialGuidanceScale);
        setInput('trainer-validation-enabled', data.validationEnabled);
        setInput('trainer-validation-every', data.validateEvery);
        setInput('trainer-validation-resolution', data.validationResolution);
        setInput('trainer-validation-sigmas', (data.validationSigmas || [0.5]).join(','));
        setInput('trainer-sample-every', data.sampleEvery);
        setInput('trainer-sample-start', data.sampleStartStep);
        setInput('trainer-sampler', data.sampler);
        setInput('trainer-guidance-scale', data.guidanceScale);
        setInput('trainer-sample-steps', data.sampleSteps);
        setInput('trainer-sample-width', data.sampleWidth);
        setInput('trainer-sample-height', data.sampleHeight);
        setInput('trainer-sample-seed', data.sampleSeed);
        setInput('trainer-walk-seed', data.walkSeed);
        setInput('trainer-skip-first-sample', data.skipFirstSample);
        setInput('trainer-force-first-sample', data.forceFirstSample);
        setInput('trainer-disable-sampling', data.disableSampling ?? !data.sampleEnabled);
        setInput('trainer-layer-offloading', data.layerOffloading);
        setInput('trainer-transformer-offload', Number(data.transformerOffload) * 100);
        setInput('trainer-text-offload', Number(data.textEncoderOffload) * 100);
        syncSlider('trainer-transformer-offload');
        syncSlider('trainer-text-offload');
        const advancedText = typeof data.advancedProcess === 'string' ? data.advancedProcess : (data.advancedProcess ? JSON.stringify(data.advancedProcess, null, 2) : '');
        setInput('trainer-advanced-config-json', advancedText);
        setInput('trainer-use-advanced-config', Boolean(advancedText));
        state.selectedDatasets = [];
        (data.datasets || []).forEach(dataset => addDataset(dataset.name, dataset, true));
        state.samples = (data.samples || []).map(sample => ({ ...sample }));
        state.validationItems = (data.validationItems || []).map(item => ({ ...item }));
        renderSelectedDatasets();
        renderConditionalOptions();
    }

    function showEditor(job = null) {
        state.selectedJobId = job?.id || null;
        state.editingJobId = job?.id || null;
        editorView.classList.remove('hidden');
        detailView.classList.add('hidden');
        $('trainer-editor-kicker').textContent = job ? 'Edit training job' : 'New training job';
        $('trainer-editor-title').textContent = job ? job.name : 'Configure LoRA training';
        $('trainer-cancel-edit-btn').classList.toggle('hidden', !job);
        populateForm(job?.form || {});
        showFormError('');
        renderJobs();
        document.querySelector('.trainer-workspace').scrollTop = 0;
    }

    function renderDetail(job) {
        if (!job) return;
        const process = job.config.config.process[0];
        const model = state.models.find(item => item.key === job.form?.model);
        $('trainer-detail-name').textContent = job.name;
        $('trainer-detail-model').textContent = model?.label || process.model.arch;
        $('trainer-detail-kicker').textContent = job.status === 'running' ? 'Active training job' : 'Training job';
        $('trainer-detail-status').textContent = job.status;
        $('trainer-detail-status').dataset.status = job.status;
        $('trainer-detail-info').textContent = job.info || 'Ready';
        $('trainer-detail-percent').textContent = `${Number(job.progress || 0).toFixed(job.progress % 1 ? 1 : 0)}%`;
        $('trainer-detail-progress').style.width = `${job.progress || 0}%`;
        $('trainer-detail-step').textContent = `${job.step || 0} / ${job.total_steps || process.train.steps} steps`;
        $('trainer-detail-speed').textContent = job.speed_string || 'Waiting';
        $('trainer-detail-gpu').textContent = `GPU ${job.gpu_ids}`;
        $('trainer-detail-config').innerHTML = [
            ['Model', model?.label || process.model.arch],
            ['Datasets', (job.datasets || []).join(', ')],
            ['Target', process.network.type === 'lokr' ? `LoKr · factor ${process.network.lokr_factor}` : `LoRA · rank ${process.network.linear}`],
            ['Optimizer', process.train.optimizer],
            ['Learning rate', process.train.lr],
            ['Sampler', process.sample.sampler],
            ['Quantization', process.model.quantize ? process.model.qtype : 'None'],
        ].map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd title="${escapeHtml(value)}">${escapeHtml(value)}</dd>`).join('');
        const active = ['queued', 'running', 'stopping'].includes(job.status);
        const running = job.status === 'running';
        $('trainer-stop-job-btn').classList.toggle('hidden', !active);
        $('trainer-save-now-btn').classList.toggle('hidden', !running);
        $('trainer-sample-now-btn').classList.toggle('hidden', !running || process.train.disable_sampling);
        $('trainer-start-job-btn').classList.toggle('hidden', active);
        $('trainer-edit-job-btn').disabled = active;
        $('trainer-delete-job-btn').disabled = active;
        $('trainer-download-log').href = `/api/trainer/jobs/${job.id}/log/download`;
    }

    async function showDetail(jobId) {
        state.selectedJobId = jobId;
        state.editingJobId = null;
        editorView.classList.add('hidden');
        detailView.classList.remove('hidden');
        renderJobs();
        renderDetail(currentJob());
        await refreshLog(false);
        document.querySelector('.trainer-workspace').scrollTop = 0;
    }

    async function refreshLog(notify = false) {
        const job = currentJob();
        if (!job) return;
        try {
            const result = await api(`/api/trainer/jobs/${job.id}/log`);
            const output = $('trainer-log-output');
            const atBottom = output.scrollTop + output.clientHeight >= output.scrollHeight - 20;
            output.textContent = result.log || 'No log output yet.';
            if (atBottom) output.scrollTop = output.scrollHeight;
            if (notify) showToast('Trainer log refreshed.');
        } catch (error) {
            if (notify) showToast(error.message, true);
        }
    }

    async function refreshState({ initial = false } = {}) {
        const result = await api('/api/trainer/state');
        state.models = result.models || [];
        state.qtypes = result.qtypes || [];
        state.datasets = result.datasets || [];
        state.jobs = result.jobs || [];
        state.gpus = result.gpus || [];
        state.trainerInstalled = Boolean(result.trainerInstalled);
        renderRuntime();
        renderModelOptions();
        renderGpuOptions();
        renderDatasetPicker();
        renderJobs();
        $('trainer-upstream-commit').textContent = result.upstreamCommit || '—';
        if (state.selectedJobId && !currentJob()) showEditor();
        else if (state.selectedJobId && !state.editingJobId && !detailView.classList.contains('hidden')) renderDetail(currentJob());
        if (initial) showEditor();
    }

    async function refreshJobsOnly() {
        const result = await api('/api/trainer/jobs');
        state.jobs = result.jobs || [];
        renderJobs();
        if (state.selectedJobId && !currentJob()) {
            showEditor();
            return;
        }
        if (state.selectedJobId && !state.editingJobId && !detailView.classList.contains('hidden')) {
            renderDetail(currentJob());
        }
    }

    async function runJobAction(action) {
        const job = currentJob();
        if (!job) return;
        try {
            await api(`/api/trainer/jobs/${job.id}/${action}`, { method: 'POST' });
            await refreshState();
            renderDetail(currentJob());
            showToast(action === 'start' ? 'Job added to the queue.' : 'Stop requested.');
        } catch (error) { showToast(error.message, true); }
    }

    async function requestRuntimeAction(action) {
        const job = currentJob();
        if (!job) return;
        try {
            await api(`/api/trainer/jobs/${job.id}/${action}-now`, { method: 'POST' });
            await refreshJobsOnly();
            showToast(action === 'save' ? 'Checkpoint save requested.' : 'Sample requested.');
        } catch (error) { showToast(error.message, true); }
    }

    async function deleteJob() {
        const job = currentJob();
        if (!job || !window.confirm(`Delete job “${job.name}”? Output files will be kept.`)) return;
        try {
            await api(`/api/trainer/jobs/${job.id}`, { method: 'DELETE' });
            state.selectedJobId = null;
            await refreshState();
            showEditor();
            showToast('Training job deleted. Output files were kept.');
        } catch (error) { showToast(error.message, true); }
    }

    function openSettings() {
        $('trainer-hf-token').value = '';
        $('trainer-clear-token').checked = false;
        $('trainer-settings-modal').classList.remove('hidden');
        $('trainer-hf-token').focus();
    }

    function closeSettings() { $('trainer-settings-modal').classList.add('hidden'); }

    async function saveSettings() {
        try {
            await api('/api/trainer/settings', {
                method: 'POST',
                body: JSON.stringify({ hfToken: $('trainer-hf-token').value, clearHfToken: $('trainer-clear-token').checked }),
            });
            closeSettings();
            await refreshState();
            showToast('Trainer settings saved.');
        } catch (error) { showToast(error.message, true); }
    }

    async function generateAdvancedConfig() {
        const payload = collectForm(false);
        const error = validateForm(payload);
        if (error) { showFormError(error); return; }
        try {
            const result = await api('/api/trainer/config/preview', { method: 'POST', body: JSON.stringify(payload) });
            $('trainer-advanced-config-json').value = JSON.stringify(result.process, null, 2);
            $('trainer-use-advanced-config').checked = true;
            showFormError('');
            showToast('Advanced process config generated.');
        } catch (errorValue) { showFormError(errorValue.message); }
    }

    function bindEvents() {
        $('trainer-new-job-btn').addEventListener('click', () => showEditor());
        $('trainer-model').addEventListener('change', () => renderModelFields(true));
        $('trainer-network-type').addEventListener('change', renderConditionalOptions);
        $('trainer-layer-offloading').addEventListener('change', renderConditionalOptions);
        ['trainer-transformer-offload', 'trainer-text-offload'].forEach(id =>
            $(id).addEventListener('input', () => syncSlider(id))
        );
        ['trainer-use-ema', 'trainer-guidance-loss', 'trainer-differential-guidance', 'trainer-validation-enabled'].forEach(id =>
            $(id).addEventListener('change', renderConditionalOptions)
        );
        $('trainer-dop').addEventListener('change', () => {
            if ($('trainer-dop').checked) $('trainer-bpp').checked = false;
            renderConditionalOptions();
        });
        $('trainer-bpp').addEventListener('change', () => {
            if ($('trainer-bpp').checked) $('trainer-dop').checked = false;
            renderConditionalOptions();
        });
        $('trainer-cache-text').addEventListener('change', () => {
            if ($('trainer-cache-text').checked) $('trainer-unload-text').checked = false;
        });
        $('trainer-unload-text').addEventListener('change', () => {
            if ($('trainer-unload-text').checked) $('trainer-cache-text').checked = false;
        });
        $('trainer-skip-first-sample').addEventListener('change', () => {
            if ($('trainer-skip-first-sample').checked) $('trainer-force-first-sample').checked = false;
        });
        $('trainer-force-first-sample').addEventListener('change', () => {
            if ($('trainer-force-first-sample').checked) {
                $('trainer-skip-first-sample').checked = false;
                $('trainer-disable-sampling').checked = false;
            }
        });
        $('trainer-disable-sampling').addEventListener('change', () => {
            if ($('trainer-disable-sampling').checked) $('trainer-force-first-sample').checked = false;
        });
        $('trainer-add-dataset-btn').addEventListener('click', () => {
            const name = $('trainer-dataset-select').value;
            if (!name) return;
            addDataset(name);
            $('trainer-dataset-select').value = '';
        });
        $('trainer-selected-datasets').addEventListener('click', event => {
            const removeButton = event.target.closest('[data-remove-dataset]');
            const duplicateButton = event.target.closest('[data-duplicate-dataset]');
            if (removeButton) state.selectedDatasets.splice(Number(removeButton.dataset.removeDataset), 1);
            else if (duplicateButton) {
                const source = state.selectedDatasets[Number(duplicateButton.dataset.duplicateDataset)];
                if (source) state.selectedDatasets.splice(Number(duplicateButton.dataset.duplicateDataset) + 1, 0, { ...source, resolutions: [...source.resolutions] });
            } else return;
            renderSelectedDatasets();
        });
        $('trainer-selected-datasets').addEventListener('input', event => {
            const input = event.target.closest('[data-dataset-field]');
            if (input) updateDatasetSetting(input);
        });
        $('trainer-add-sample-btn').addEventListener('click', () => {
            state.samples.push({ prompt: '', width: '', height: '', seed: '', networkMultiplier: '', ctrlImg1: '', ctrlImg2: '', ctrlImg3: '' });
            renderSamples();
        });
        $('trainer-sample-items').addEventListener('click', event => {
            const removeButton = event.target.closest('[data-remove-sample]');
            const pickerButton = event.target.closest('[data-pick-sample-image]');
            const clearButton = event.target.closest('[data-clear-sample-image]');
            if (removeButton) {
                state.samples.splice(Number(removeButton.dataset.removeSample), 1);
                renderSamplesPreservingPlace();
                return;
            }
            if (pickerButton) {
                const input = $('trainer-sample-items').querySelector(`[data-sample-file="${pickerButton.dataset.pickSampleImage}"]`);
                input?.click();
                return;
            }
            if (clearButton) {
                const [sampleIndexText, controlIndexText] = clearButton.dataset.clearSampleImage.split('-');
                const sample = state.samples[Number(sampleIndexText)];
                if (!sample) return;
                sample[`ctrlImg${Number(controlIndexText)}`] = '';
                delete sample[`ctrl_img_${Number(controlIndexText)}`];
                renderSamplesPreservingPlace(`[data-pick-sample-image="${sampleIndexText}-${controlIndexText}"]`);
            }
        });
        $('trainer-sample-items').addEventListener('input', event => {
            const input = event.target.closest('[data-sample-field]');
            if (input) updateRepeatedItem(input, state.samples, 'sampleIndex', 'sampleField');
        });
        $('trainer-sample-items').addEventListener('change', event => {
            const input = event.target.closest('[data-sample-file]');
            if (input) uploadSampleImage(input);
        });
        ['dragenter', 'dragover'].forEach(eventName => $('trainer-sample-items').addEventListener(eventName, event => {
            const slot = event.target.closest('[data-sample-image-slot]');
            if (!slot) return;
            event.preventDefault();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
            slot.classList.add('is-dragging');
        }));
        $('trainer-sample-items').addEventListener('dragleave', event => {
            const slot = event.target.closest('[data-sample-image-slot]');
            if (slot && !slot.contains(event.relatedTarget)) slot.classList.remove('is-dragging');
        });
        $('trainer-sample-items').addEventListener('drop', event => {
            const slot = event.target.closest('[data-sample-image-slot]');
            if (!slot) return;
            event.preventDefault();
            slot.classList.remove('is-dragging');
            const file = event.dataTransfer?.files?.[0];
            const input = slot.querySelector('[data-sample-file]');
            if (file && input) uploadSampleImage(input, file);
        });
        $('trainer-add-validation-btn').addEventListener('click', () => {
            state.validationItems.push({ dataset: state.selectedDatasets[0]?.name || '', image: '', prompt: '' });
            renderValidationItems();
        });
        $('trainer-validation-items').addEventListener('click', event => {
            const button = event.target.closest('[data-remove-validation]');
            if (!button) return;
            state.validationItems.splice(Number(button.dataset.removeValidation), 1);
            renderValidationItems();
        });
        $('trainer-validation-items').addEventListener('input', event => {
            const input = event.target.closest('[data-validation-field]');
            if (input) updateRepeatedItem(input, state.validationItems, 'validationIndex', 'validationField');
        });
        form.addEventListener('submit', event => { event.preventDefault(); saveJob(false); });
        $('trainer-save-start-btn').addEventListener('click', () => saveJob(true));
        $('trainer-cancel-edit-btn').addEventListener('click', () => state.selectedJobId ? showDetail(state.selectedJobId) : showEditor());
        jobsList.addEventListener('click', event => {
            const item = event.target.closest('[data-job-id]');
            if (item) showDetail(item.dataset.jobId);
        });
        document.querySelectorAll('[data-job-filter]').forEach(button => button.addEventListener('click', () => {
            state.filter = button.dataset.jobFilter;
            document.querySelectorAll('[data-job-filter]').forEach(item => item.classList.toggle('is-active', item === button));
            renderJobs();
        }));
        $('trainer-edit-job-btn').addEventListener('click', () => showEditor(currentJob()));
        $('trainer-start-job-btn').addEventListener('click', () => runJobAction('start'));
        $('trainer-stop-job-btn').addEventListener('click', () => runJobAction('stop'));
        $('trainer-save-now-btn').addEventListener('click', () => requestRuntimeAction('save'));
        $('trainer-sample-now-btn').addEventListener('click', () => requestRuntimeAction('sample'));
        $('trainer-delete-job-btn').addEventListener('click', deleteJob);
        $('trainer-refresh-log-btn').addEventListener('click', () => refreshLog(true));
        $('trainer-settings-btn').addEventListener('click', openSettings);
        $('trainer-settings-close').addEventListener('click', closeSettings);
        $('trainer-settings-cancel').addEventListener('click', closeSettings);
        $('trainer-settings-save').addEventListener('click', saveSettings);
        $('trainer-generate-config-btn').addEventListener('click', generateAdvancedConfig);
        $('trainer-clear-config-btn').addEventListener('click', () => {
            $('trainer-advanced-config-json').value = '';
            $('trainer-use-advanced-config').checked = false;
            showToast('Advanced override cleared.');
        });
        $('trainer-settings-modal').addEventListener('click', event => { if (event.target === $('trainer-settings-modal')) closeSettings(); });
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape' && !$('trainer-settings-modal').classList.contains('hidden')) closeSettings();
        });
    }

    async function initialize() {
        bindEvents();
        try {
            await refreshState({ initial: true });
            state.pollTimer = window.setInterval(async () => {
                try {
                    await refreshJobsOnly();
                    if (!detailView.classList.contains('hidden') && ['queued', 'running', 'stopping'].includes(currentJob()?.status)) await refreshLog(false);
                } catch (_) { /* keep the last rendered state */ }
            }, 3000);
        } catch (error) {
            showFormError(`Could not load trainer: ${error.message}`);
            $('trainer-runtime-status').textContent = 'Trainer API unavailable';
            $('trainer-runtime-status').classList.add('is-missing');
        }
    }

    initialize();
})();
