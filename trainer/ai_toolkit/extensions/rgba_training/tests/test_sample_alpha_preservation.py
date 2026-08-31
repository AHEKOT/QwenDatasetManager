import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from toolkit.config_modules import GenerateImageConfig, SampleConfig


class SampleAlphaPreservationTests(unittest.TestCase):
    def test_sample_and_thumbnail_keep_alpha(self):
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[16:48, 16:48] = [240, 40, 20, 255]
        image = Image.fromarray(rgba, mode="RGBA")

        with tempfile.TemporaryDirectory() as tmp:
            output_folder = Path(tmp)
            config = GenerateImageConfig(output_folder=str(output_folder), output_ext="png")
            config.save_image_atomic(image)

            samples = list(output_folder.glob("*.png"))
            self.assertEqual(len(samples), 1)
            with Image.open(samples[0]) as saved:
                self.assertEqual(saved.mode, "RGBA")
                self.assertEqual(saved.getchannel("A").getextrema(), (0, 255))

            thumb = output_folder / ".thumbs" / f"{samples[0].name}.png"
            self.assertTrue(thumb.is_file())
            with Image.open(thumb) as preview:
                self.assertEqual(preview.mode, "RGBA")
                self.assertEqual(preview.size, (300, 300))
                self.assertEqual(preview.getchannel("A").getextrema(), (0, 255))

    def test_sample_format_and_legacy_ext_alias(self):
        self.assertEqual(SampleConfig(format="png", samples=[]).ext, "png")
        self.assertEqual(SampleConfig(ext="png", samples=[]).ext, "png")


if __name__ == "__main__":
    unittest.main()
