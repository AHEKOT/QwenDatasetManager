import os
import unittest

import torch
from diffusers import AutoencoderKLQwenImage

from extensions.rgba_training.qwen_compat import validate_qwen_rgba_vae_config
from toolkit.rgba_utils import ensure_normalized_rgba_tensor


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_LAYERED_VAE_TESTS") == "1",
    "set RUN_QWEN_LAYERED_VAE_TESTS=1 to load the real Qwen-Image-Layered VAE",
)
class QwenLayeredVAEIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested_device = os.environ.get("QWEN_LAYERED_DEVICE", "cuda")
        cls.device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
        cls.dtype = torch.bfloat16 if cls.device.type == "cuda" else torch.float32
        cls.vae = AutoencoderKLQwenImage.from_pretrained(
            os.environ.get("QWEN_LAYERED_MODEL", "Qwen/Qwen-Image-Layered"),
            subfolder=os.environ.get("QWEN_LAYERED_VAE_SUBFOLDER", "vae"),
            torch_dtype=cls.dtype,
            local_files_only=os.environ.get("QWEN_LAYERED_LOCAL_ONLY") == "1",
        ).to(cls.device).eval()

    def test_real_checkpoint_contract_and_rgba_roundtrip_shape(self):
        validate_qwen_rgba_vae_config(self.vae.config)
        image = torch.zeros((1, 4, 32, 32), dtype=self.dtype, device=self.device).unsqueeze(2)
        image[:, 3] = 1.0
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.mode()
            decoded = self.vae.decode(latent).sample
        self.assertEqual(latent.shape[1], 16)
        self.assertEqual(decoded.shape[:3], (1, 4, 1))
        self.assertTrue(torch.isfinite(latent).all().item())
        self.assertTrue(torch.isfinite(decoded).all().item())

        # QIE control images remain RGB; the extension adds opaque alpha before
        # using this four-channel VAE for control latents.
        rgb_control = torch.zeros((1, 3, 1, 32, 32), dtype=self.dtype, device=self.device)
        rgba_control = ensure_normalized_rgba_tensor(rgb_control)
        with torch.no_grad():
            control_latent = self.vae.encode(rgba_control).latent_dist.mode()
        self.assertEqual(rgba_control.shape[1], 4)
        self.assertEqual(control_latent.shape[1], 16)


if __name__ == "__main__":
    unittest.main()
