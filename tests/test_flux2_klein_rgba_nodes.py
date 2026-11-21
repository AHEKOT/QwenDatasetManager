import importlib.util
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ComfyUI-FLUX2-Klein-RGBA"
    / "nodes.py"
)


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


def state_dict(*, channels=4, decoder_channels=128, nested=True):
    result = {
        "encoder.conv_in.weight": FakeTensor(128, channels, 3, 3),
        "decoder.conv_in.weight": FakeTensor(decoder_channels * 4, 32, 3, 3),
        "decoder.conv_out.weight": FakeTensor(channels, decoder_channels, 3, 3),
        "decoder.conv_out.bias": FakeTensor(channels),
        "bn.running_mean": FakeTensor(128),
    }
    if nested:
        result["encoder.quant_conv.weight"] = FakeTensor(64, 64, 1, 1)
        result["decoder.post_quant_conv.weight"] = FakeTensor(32, 32, 1, 1)
    else:
        result["quant_conv.weight"] = FakeTensor(64, 64, 1, 1)
        result["post_quant_conv.weight"] = FakeTensor(32, 32, 1, 1)
    return result


class Flux2KleinRGBANodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder_paths = mock.MagicMock()
        comfy = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules",
            {
                "folder_paths": folder_paths,
                "comfy": comfy,
                "comfy.sd": comfy.sd,
                "comfy.utils": comfy.utils,
            },
        ):
            spec = importlib.util.spec_from_file_location("flux2_rgba_nodes", MODULE_PATH)
            cls.nodes = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.nodes)

    def test_detects_rgba_but_rejects_rgb(self):
        self.assertTrue(self.nodes.is_flux2_klein_rgba_vae(state_dict()))
        self.assertFalse(
            self.nodes.is_flux2_klein_rgba_vae(state_dict(channels=3))
        )

    def test_normalizes_nested_quantizer_keys(self):
        normalized = self.nodes.normalize_state_dict(state_dict())
        self.assertIn("quant_conv.weight", normalized)
        self.assertIn("post_quant_conv.weight", normalized)
        self.assertNotIn("encoder.quant_conv.weight", normalized)
        self.assertNotIn("decoder.post_quant_conv.weight", normalized)

    def test_builds_full_decoder_config(self):
        params = self.nodes.build_comfy_vae_config(state_dict())["params"]
        self.assertEqual(params["ddconfig"]["in_channels"], 4)
        self.assertEqual(params["ddconfig"]["out_ch"], 4)
        self.assertEqual(params["ddconfig"]["z_channels"], 32)
        self.assertEqual(params["embed_dim"], 32)
        self.assertNotIn("decoder_ddconfig", params)

    def test_builds_small_decoder_config(self):
        params = self.nodes.build_comfy_vae_config(
            state_dict(decoder_channels=96, nested=False)
        )["params"]
        self.assertEqual(params["ddconfig"]["ch"], 128)
        self.assertEqual(params["decoder_ddconfig"]["ch"], 96)


if __name__ == "__main__":
    unittest.main()

