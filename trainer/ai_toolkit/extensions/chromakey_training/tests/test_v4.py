import os
import json
from types import SimpleNamespace
from collections import OrderedDict
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "ComfyUI-QDM-ChromaKey"))
sys.path.insert(0, str(ROOT / "trainer/ai_toolkit"))
from qdm_chromakey_core.v4 import KeyMatteV4, ARCHITECTURE_ID_V4
from qdm_chromakey_core.inference import run_tiled, load_model
from extensions.chromakey_training.data_v4 import (
    ChromaKeyDatasetV4, resize_rgba_float, split_sources, enclosed_background,
    srgb_to_linear, linear_to_srgb, CurriculumBatchSampler,
    erode_alpha,
)
from extensions.chromakey_training.objectives import matting_loss, matting_metrics


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.files = []
        for index in range(4):
            im = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
            draw = ImageDraw.Draw(im)
            draw.rectangle((20, 20, 150, 120), fill=(100+index*20, 70, 160, 255))
            draw.ellipse((50, 40, 80, 70), fill=(0, 0, 0, 0))
            draw.line((10, 30, 160, 10), fill=(160, 120, 80, 90), width=1)
            p = self.root / f"{index}.png"
            im.save(p)
            self.files.append(p)

    def tearDown(self):
        self.temp.cleanup()

    def sample(self):
        return ChromaKeyDatasetV4(self.files, {"green_chance": 100, "spill_chance": 100,
            "detail_crop_percent": 100, "noise_chance": 100, "noise_strength": .01,
            "dirt_chance": 100, "jpeg_chance": 100})[(0, 96, 128, 123, 1.)]

    def test_float_resize_preserves_low_alpha_colour(self):
        image = Image.new("RGBA", (8, 8), (100, 140, 210, 2))
        fg, alpha = resize_rgba_float(image, (13, 11))
        self.assertLess((fg-torch.tensor([100, 140, 210])[:, None, None]/255).abs().max().item(), 1e-5)
        self.assertAlmostEqual(alpha.mean().item(), 2/255, places=6)

    def test_augmentation_is_reproducible_and_does_not_consume_rng(self):
        state = torch.get_rng_state()
        a, b = self.sample(), self.sample()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key in a:
            if torch.is_tensor(a[key]):
                self.assertTrue(torch.equal(a[key], b[key]), key)
                self.assertTrue(torch.isfinite(a[key]).all(), key)
        self.assertEqual(a["input"].shape, (3, 96, 128))
        self.assertEqual(a["context_rgb"].shape, (3, 256, 256))
        self.assertGreater(a["holes"].sum(), 0)

    def test_source_split_deduplicates_and_is_stable(self):
        duplicate = self.root / "duplicate.png"
        duplicate.write_bytes(self.files[0].read_bytes())
        train, val = split_sources(self.files + [duplicate])
        self.assertEqual(len(train)+len(val), 4)
        self.assertFalse(set(p.read_bytes() for p in train) & set(p.read_bytes() for p in val))
        self.assertEqual(split_sources(self.files), split_sources(list(reversed(self.files))))

    def test_rgb_sources_fail_instead_of_becoming_opaque_targets(self):
        path = self.root / "rgb.png"
        Image.new("RGB", (10, 10)).save(path)
        with self.assertRaisesRegex(ValueError, "Missing alpha"):
            split_sources([path, self.files[0]])

    def test_linear_srgb_roundtrip(self):
        value = torch.linspace(0, 1, 1000)
        self.assertLess((linear_to_srgb(srgb_to_linear(value))-value).abs().max(), 2e-6)

    def test_holes_do_not_include_exterior_or_foreground(self):
        alpha = torch.zeros(1, 16, 16)
        alpha[:, 2:14, 2:14] = 1
        alpha[:, 7:9, 7:9] = 0
        mask = enclosed_background(alpha)
        self.assertEqual(mask.sum(), 4)
        self.assertEqual(mask[0, 7, 7], 1)

    def test_fast_spill_erosion_matches_pooling_at_borders(self):
        alpha = torch.rand(1, 31, 43)
        for radius in (1, 5, 12):
            expected = -torch.nn.functional.max_pool2d(-alpha[None], radius*2+1, 1, radius)[0]
            self.assertTrue(torch.equal(expected, erode_alpha(alpha, radius)))

    def test_roundtrip_and_tile_independence_on_odd_rectangle(self):
        torch.manual_seed(12)
        model = KeyMatteV4().eval()
        path = self.root / "v4.safetensors"
        save_file(model.state_dict(), str(path), metadata={"architecture": ARCHITECTURE_ID_V4})
        restored, _ = load_model(path)
        source = torch.rand(3, 97, 173)
        with torch.no_grad():
            expected = model(source[None])
            actual = restored(source[None])
        self.assertTrue(torch.equal(expected["alpha"], actual["alpha"]))
        clean, alpha = run_tiled(restored, source, tile_size=64)
        self.assertLess((alpha-expected["alpha"][0]).abs().max(), 2e-5)
        self.assertLess((clean-expected["foreground"][0]).abs().max(), 2e-5)
        clean2, alpha2 = run_tiled(restored, source, tile_size=96)
        self.assertLess((alpha-alpha2).abs().max(), 2e-5)
        self.assertLess((clean-clean2).abs().max(), 2e-5)

    def test_loss_has_gradients_for_context_alpha_and_foreground(self):
        sample = self.sample()
        batch = {k: v[None] for k, v in sample.items() if torch.is_tensor(v)}
        model = KeyMatteV4()
        prediction = model(batch["input"], batch["context_rgb"], batch["context_box"])
        total, losses = matting_loss(prediction, batch)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for layer in (model.enc2[0][0], model.coarse_head, model.alpha_head, model.foreground_head):
            self.assertGreater(layer.weight.grad.abs().sum(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_metrics_penalize_filled_holes_and_green_hair(self):
        sample = self.sample()
        batch = {k: v[None] for k, v in sample.items() if torch.is_tensor(v)}
        perfect = matting_metrics(batch["foreground"], batch["alpha"], batch)
        self.assertEqual(perfect["selection_score"], 0)
        wrong_alpha = (batch["alpha"] + batch["holes"]).clamp(0, 1)
        bad = matting_metrics(batch["input"], wrong_alpha, batch)
        self.assertGreater(bad["holes_mae"], .99)
        self.assertGreater(bad["despill_mae"], 0)

    def test_curriculum_resume_skips_before_loading(self):
        base = [[(i, 64, 64, i)] for i in range(6)]
        resumed = list(CurriculumBatchSampler(base, 3, 10, 100, 2))
        self.assertEqual(resumed[0][0], (3, 64, 64, 3, .1))
        self.assertEqual(len(resumed), 3)

    @unittest.skipUnless(os.environ.get("QDM_RUN_CUDA_INTEGRATION") == "1", "opt-in actual CUDA trainer integration")
    def test_actual_trainer_stop_resume_matches_uninterrupted_run(self):
        from extensions.chromakey_training.trainer import QDMChromaKeyTrainProcess
        def process(name):
            config = OrderedDict(training_folder=str(self.root/"output"), device="cuda",
                architecture={"id": ARCHITECTURE_ID_V4}, datasets=[{"folder_path": str(self.root)}],
                train={"steps": 4, "resolutions": [64], "max_long_side": 96, "batch_size": 1,
                       "gradient_accumulation": 2, "num_workers": 0, "dtype": "bf16"},
                augmentation={"green_chance": 100, "spill_chance": 100, "source_max_side": 512},
                validation={"every": 0}, save={"every": 0})
            job = SimpleNamespace(name=name, meta=OrderedDict(), raw_config=dict(config))
            return QDMChromaKeyTrainProcess(0, job, config)
        uninterrupted = process("uninterrupted")
        uninterrupted.run()
        interrupted = process("resumed")
        seen = [0]
        def flag(column, consume=False):
            if column == "stop":
                seen[0] += 1
                return seen[0] == 6  # stop after an uncommitted microbatch
            return False
        interrupted._db_flag = flag
        interrupted.run()
        self.assertEqual(interrupted.step_num, 2)
        resumed = process("resumed")
        resumed.run()
        self.assertEqual(resumed.step_num, 4)
        a = torch.load(Path(uninterrupted.save_root)/"trainer_state_v4_step_000000004.pt", weights_only=True, map_location="cpu")
        b = torch.load(Path(resumed.save_root)/"trainer_state_v4_step_000000004.pt", weights_only=True, map_location="cpu")
        for key in a["model"]:
            self.assertTrue(torch.allclose(a["model"][key], b["model"][key], atol=1e-6), key)
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertTrue((Path(resumed.save_root)/"resumed_best.safetensors").is_file())
        manifest = json.loads((Path(resumed.save_root)/"artifacts.json").read_text())
        self.assertEqual(manifest["primary"], "resumed_best.safetensors")


if __name__ == "__main__":
    unittest.main()
