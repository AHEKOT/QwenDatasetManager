import shutil
import subprocess
import unittest
from pathlib import Path


class TrainerFrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is required for frontend regression tests')
    def test_dataset_batch_override_defaults_restores_and_clears(self):
        script = r'''
const fs = require('node:fs');
const assert = require('node:assert/strict');
const source = fs.readFileSync('static/trainer.js', 'utf8');
const state = {datasets: [{name: 'demo'}], selectedDatasets: []};
const addSource = source.slice(source.indexOf('    function addDataset('), source.indexOf('    function resolutionChip('));
const add = new Function('state', 'isH3', 'isQieJoint', 'renderSelectedDatasets',
    addSource + '\nreturn addDataset;')(state, () => false, () => false, () => {});
add('demo', {batchSize: 1}); // Legacy values do not become explicit overrides.
assert.equal(state.selectedDatasets[0].batchSizeOverride, '');
add('demo', {batchSizeOverride: 3}, true);
assert.equal(state.selectedDatasets[1].batchSizeOverride, 3);
const updateSource = source.slice(source.indexOf('    function updateDatasetSetting('), source.indexOf('    function renderSamples('));
const update = new Function('state', updateSource + '\nreturn updateDatasetSetting;')(state);
const input = {dataset: {datasetIndex: '0', datasetField: 'batchSizeOverride'}, type: 'number', value: '1'};
update(input);
assert.equal(state.selectedDatasets[0].batchSizeOverride, 1);
input.value = '';
update(input);
assert.equal(state.selectedDatasets[0].batchSizeOverride, '');
const validateSource = source.slice(source.indexOf('    function validateForm('), source.indexOf('    async function saveJob('));
const validate = new Function('isVaePreset', 'isChromaPreset', validateSource + '\nreturn validateForm;')(() => false, () => false);
const payload = {name: 'test', modelPath: 'unused', datasets: [{name: 'demo', resolutions: [512]}], disableSampling: true};
for (const value of ['', undefined, null, 1, 128]) {
    payload.datasets[0].batchSizeOverride = value;
    assert.equal(validate(payload), '');
}
for (const value of [0, -1, 1.5, 129]) {
    payload.datasets[0].batchSizeOverride = value;
    assert.match(validate(payload), /Batch size override for demo/);
}
'''
        result = subprocess.run([shutil.which('node'), '-e', script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for frontend regression tests')
    def test_qwen21_accepts_text_only_samples(self):
        script = r'''
const fs = require('node:fs');
const assert = require('node:assert/strict');
const source = fs.readFileSync('static/trainer.js', 'utf8');
const validation = source.slice(source.indexOf('    function validateForm(payload) {'), source.indexOf('    async function saveJob('));
const validate = new Function('isVaePreset', 'isChromaPreset', 'h3', 'sampleControlPath',
    validation + '\nreturn validateForm;')(() => false, () => false, {isH3: () => false}, () => '');
const payload = {name: 'qwen21', model: 'qwen_image_2', modelPath: 'Comfy-Org/Qwen-Image-2.1',
    trainingPreset: 'standard_lora', nativeRgba: true, datasets: [{name: 'demo', resolutions: [512]}],
    samples: [{prompt: 'A red kite'}], disableSampling: false, validationEnabled: false};
assert.equal(validate(payload), '');
assert.equal(validate({...payload, model: 'qwen_image_edit_2511'}), 'Add at least one image to edit for every sample.');
'''
        result = subprocess.run([shutil.which('node'), '-e', script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for frontend regression tests')
    def test_save_allows_paired_dataset_with_stale_or_missing_inspection(self):
        script = r'''
const fs = require('node:fs');
const assert = require('node:assert/strict');
const source = fs.readFileSync('static/trainer.js', 'utf8');
const validation = source.slice(source.indexOf('    function validateForm(payload) {'), source.indexOf('    async function saveJob('));
const payload = {
    name: 'QIE_PoseStudio', model: 'qwen_image_edit_2511', modelPath: 'Qwen/Qwen-Image-Edit-2511',
    trainingPreset: 'transparent_lora', vaePath: 'models/vae/QIE2511-rgba.safetensors',
    datasets: [{name: 'PoseStudioV7_Transparent', rgbaControlMode: 'paired', resolutions: [512]}],
    disableSampling: true, validationEnabled: false,
};
for (const inspection of [undefined, {name: 'PoseStudioV7_Transparent', inspected: true, targetCount: 122,
    controls: [{name:'Control1',count:122,missing:0},{name:'Control2',count:122,missing:0}]}]) {
    const validate = new Function('state', 'isVaePreset', 'isChromaPreset', 'independentValidationUploads',
        validation + '\nreturn validateForm;')({datasets: inspection ? [inspection] : []}, () => false, () => false, true);
    assert.equal(validate(payload), '');
}
'''
        result = subprocess.run([shutil.which('node'), '-e', script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
