import shutil
import subprocess
import unittest
from pathlib import Path


class TrainerFrontendTests(unittest.TestCase):
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
