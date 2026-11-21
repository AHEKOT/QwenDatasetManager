import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

from .runtime import CleanMatte, MODEL_ID, predict, recover_foreground
from cleanmatte.inference import load_model
from .data import analytic_patch, composite, inventory, source_patch
from .losses import objective, target_classes, metrics


class CleanMatteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tiled_matches_dense_on_odd_native_image(self):
        torch.manual_seed(7)
        model = CleanMatte().eval()
        with torch.no_grad():
            model.classes.bias.copy_(torch.tensor([-10., -10., 10.]))
        rgb = torch.rand(1, 3, 131, 197)
        dense = predict(model, rgb, 512)
        tiled = predict(model, rgb, 64)
        self.assertEqual(tuple(tiled.shape), (1, 1, 131, 197))
        self.assertGreater(float(dense.std()), 1e-7)
        torch.testing.assert_close(tiled, dense, atol=2e-6, rtol=2e-6)

    def test_native_classifier_can_find_hole_despite_wrong_coarse_guidance(self):
        model = CleanMatte().eval()
        rgb = torch.rand(1, 3, 33, 39)
        guidance = torch.zeros(1, 11, 33, 39)
        guidance[:, 1] = 5  # coarse branch confidently says foreground
        with torch.no_grad():
            model.classes.weight.zero_()
            model.classes.bias.copy_(torch.tensor([10., -10., -10.]))
            out = model.refine(rgb, guidance)
        self.assertEqual(out['alpha'].max().item(), 0)

    def test_wrong_classifier_does_not_starve_coverage_training(self):
        model = CleanMatte()
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            model.classes.bias.copy_(torch.tensor([20., -20., -20.]))
        output = model(image)
        target = torch.zeros(1, 1, 64, 64)
        target[..., 12:52, 12:52] = 1
        target[..., 11, 12:52] = 0.25
        loss, _ = objective(output, target)
        loss.backward()
        self.assertGreater(model.coverage.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.classes.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_contour_error_has_weight_independent_of_large_body_area(self):
        target = torch.zeros(1, 1, 32, 32)
        target[..., 4:28, 4:28] = 1
        outputs = []
        for y, x in ((4, 15), (15, 15)):
            coverage = target.clone()
            coverage[..., y, x] = 0.5
            logits = torch.zeros(1, 3, 32, 32)
            logits.scatter_(1, target_classes(target)[:, None], 10)
            logits[..., y, x] = torch.tensor([10., 0., 0.])[None]
            outputs.append({'coverage': coverage, 'classes': logits, 'coarse_classes': logits})
        weights = {'gradient': 0, 'consistency': 0}
        edge, _ = objective(outputs[0], target, weights=weights)
        interior, _ = objective(outputs[1], target, weights=weights)
        self.assertGreater(edge.item(), interior.item())

    def test_rgb_corruption_cannot_change_clean_alpha(self):
        foreground, alpha = analytic_patch(64, random.Random(3))
        before = alpha.clone()
        clean = composite(foreground, alpha, random.Random(9), {}, 0)
        dirty = composite(foreground, alpha, random.Random(9), {'spill_chance': 100, 'jpeg_chance': 100}, 1)
        torch.testing.assert_close(alpha, before, atol=0, rtol=0)
        self.assertGreater(float((clean-dirty).abs().mean()), 0)

    def test_subpixel_alpha_is_mixed_not_uncertain_foreground_probability(self):
        alpha = torch.tensor([[[[0., 1/255, 0.5, 254/255, 1.]]]])
        self.assertEqual(target_classes(alpha).tolist(), [[[0, 2, 2, 2, 1]]])

    def test_despill_preserves_alpha_and_opaque_rgb(self):
        rgb = torch.rand(1, 3, 20, 24)
        alpha = torch.rand(1, 1, 20, 24)
        alpha[..., :4, :] = 1
        before = alpha.clone()
        corrected = recover_foreground(rgb, alpha, [0., 1., 0.])
        torch.testing.assert_close(alpha, before, atol=0, rtol=0)
        torch.testing.assert_close(corrected[..., :4, :], rgb[..., :4, :], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(corrected).all())

    def test_deduplication_prevents_pixel_identical_train_val_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                image = Image.new('RGBA', (8, 8), (100+i, 30, 30, 255))
                image.putpixel((0, 0), (0, 0, 0, 0))
                image.save(root/f'{i}.png')
            with Image.open(root/'0.png') as image:
                image.save(root/'duplicate.png', compress_level=0)
            split = inventory(sorted(root.glob('*.png')))
            self.assertEqual(len(split['train'])+len(split['validation']), 3)
            self.assertEqual(len(split['duplicates']), 1)
            self.assertFalse(set(split['train']) & set(split['validation']))

    def test_premultiplied_resize_does_not_introduce_dark_fringe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rgba.png'
            image = Image.new('RGBA', (63, 63), (0, 0, 0, 0))
            for y in range(10, 54):
                for x in range(10, 54):
                    image.putpixel((x, y), (255, 255, 255, 255))
            image.save(path)
            fg, alpha = source_patch(path, 32, random.Random(1), native=False)
            self.assertTrue(torch.all(fg[:, (alpha[0] > 0)] == 1))

    def test_export_reload_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = CleanMatte().eval()
            save_file(model.state_dict(), str(root/'new.safetensors'), metadata={'architecture': MODEL_ID})
            loaded = load_model(root/'new.safetensors')
            rgb = torch.rand(1, 3, 64, 64)
            torch.testing.assert_close(predict(model, rgb), predict(loaded, rgb), atol=2e-6, rtol=2e-6)
            save_file(model.state_dict(), str(root/'old.safetensors'), metadata={'architecture': 'old'})
            with self.assertRaisesRegex(ValueError, 'trained from scratch'):
                load_model(root/'old.safetensors')

    def test_metrics_penalize_erased_hair_and_leaking_background(self):
        target = torch.zeros(1, 1, 24, 24)
        target[..., 5:19, 12] = 0.5
        perfect = metrics(target, target)
        erased = metrics(torch.zeros_like(target), target)
        dirty = metrics((target+0.1).clamp(0, 1), target)
        self.assertEqual(perfect['thin_mae'], 0)
        self.assertGreater(erased['thin_mae'], 0)
        self.assertGreater(dirty['background_leak'], 0)

    def test_holes_are_measured_separately_from_outer_background(self):
        target = torch.zeros(1, 1, 32, 32)
        target[..., 4:28, 4:28] = 1
        target[..., 15:17, 15:17] = 0
        prediction = target.clone()
        prediction[..., 15:17, 15:17] = 1
        result = metrics(prediction, target)
        self.assertEqual(result['hole_leak'], 1)
        self.assertLess(result['background_leak'], 0.02)

    def test_training_resume_export_and_stop_protocol(self):
        from .engine import Trainer, JobBridge
        import os
        import sqlite3
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root/'sources'
            sources.mkdir()
            for i in range(3):
                fg, alpha = analytic_patch(32, random.Random(i))
                rgba = torch.cat((fg, alpha), 0).permute(1, 2, 0)
                Image.fromarray((rgba.numpy()*255).round().astype(np.uint8)).save(sources/f'{i}.png')
            config = {'architecture': {'id': MODEL_ID}, 'device': 'cpu', 'training_folder': str(root),
                      'datasets': [{'folder_path': str(sources)}],
                      'train': {'steps': 1, 'resolutions': [32], 'batch_size': 1, 'gradient_accumulation': 1,
                                'dtype': 'fp32', 'seed': 4},
                      'validation': {'every': 1, 'crop_size': 32, 'source_limit': 1, 'images': []},
                      'save': {'every': 1}}
            trainer = Trainer(config, 'run')
            trainer.run()
            self.assertEqual(trainer.step, 1)
            self.assertIsNone(trainer.corruption_start)
            state = torch.load(root/'run'/'training_state.pt', weights_only=True)
            self.assertEqual(state['step'], 1)
            self.assertTrue(state['optimizer']['state'])
            resumed = Trainer(config, 'run')
            self.assertEqual(resumed.step, 1)
            for saved, restored in zip(trainer.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(saved, restored)
            exported = load_model(root/'run'/'cleanmatte_best.safetensors')
            self.assertIsInstance(exported, CleanMatte)
            db_path = root/'queue.db'
            with sqlite3.connect(db_path) as db:
                db.execute('CREATE TABLE Job (id TEXT, stop INTEGER, save_now INTEGER, sample_now INTEGER, status TEXT, info TEXT, step INTEGER, speed_string TEXT)')
                db.execute("INSERT INTO Job VALUES ('test', 1, 0, 0, 'running', '', 1, '')")
            db.close()
            config['train']['steps'] = 2
            with patch.dict(os.environ, {'AITK_JOB_ID': 'test'}):
                resumed.bridge = JobBridge(str(db_path))
                resumed.run()
            with sqlite3.connect(db_path) as db:
                self.assertEqual(db.execute('SELECT status FROM Job').fetchone()[0], 'stopped')
            db.close()
            self.assertEqual(resumed.step, 1)


if __name__ == '__main__':
    unittest.main()
