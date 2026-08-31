import unittest
from dataclasses import replace

import torch

from extensions_built_in.diffusion_models.flux2.src.autoencoder import AutoEncoder, AutoEncoderParams
from extensions.rgba_training.flux2_rgba_vae_trainer import (
    expand_flux2_vae_state_dict_to_rgba,
    flux2_autoencoder_params,
)


class Flux2RGBAVAETrainerTests(unittest.TestCase):
    def test_rgb_boundary_expansion_preserves_rgb_and_adds_opaque_alpha(self):
        state = {
            "encoder.conv_in.weight": torch.randn(8, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 8, 3, 3),
            "decoder.conv_out.bias": torch.randn(3),
        }

        expanded = expand_flux2_vae_state_dict_to_rgba(state)

        self.assertEqual(tuple(expanded["encoder.conv_in.weight"].shape), (8, 4, 3, 3))
        self.assertEqual(tuple(expanded["decoder.conv_out.weight"].shape), (4, 8, 3, 3))
        torch.testing.assert_close(expanded["encoder.conv_in.weight"][:, :3], state["encoder.conv_in.weight"])
        torch.testing.assert_close(expanded["decoder.conv_out.weight"][:3], state["decoder.conv_out.weight"])
        torch.testing.assert_close(expanded["decoder.conv_out.bias"][:3], state["decoder.conv_out.bias"])
        self.assertEqual(expanded["decoder.conv_out.bias"][3].item(), 1.0)
        self.assertEqual(expanded["encoder.conv_in.weight"][:, 3].count_nonzero().item(), 0)

    def test_native_params_keep_flux2_z32_and_detect_small_decoder(self):
        state = {
            "encoder.conv_in.weight": torch.empty(128, 4, 3, 3),
            "decoder.conv_out.weight": torch.empty(4, 96, 3, 3),
            "decoder.up.0.block.0.conv1.bias": torch.empty(96),
        }

        params = flux2_autoencoder_params(state)

        self.assertEqual(params.in_channels, 4)
        self.assertEqual(params.out_ch, 4)
        self.assertEqual(params.z_channels, 32)
        self.assertEqual(params.ch_encoder, 96)

    def test_expanded_native_autoencoder_runs_rgba_roundtrip(self):
        rgb_params = AutoEncoderParams(
            resolution=32,
            in_channels=3,
            out_ch=3,
            ch=32,
            ch_mult=[1, 1],
            num_res_blocks=1,
            z_channels=32,
        )
        source = AutoEncoder(rgb_params)
        rgba = AutoEncoder(replace(rgb_params, in_channels=4, out_ch=4))

        rgba.load_state_dict(expand_flux2_vae_state_dict_to_rgba(source.state_dict()))
        image = torch.randn(1, 4, 32, 32)
        reconstruction = rgba.decode(rgba.encode(image))

        self.assertEqual(tuple(reconstruction.shape), (1, 4, 32, 32))


if __name__ == "__main__":
    unittest.main()
