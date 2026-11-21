import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from extensions_built_in.diffusion_models.flux2.src.autoencoder import AutoEncoder, AutoEncoderParams
from extensions.rgba_training.flux2_rgba_vae_trainer import (
    expand_flux2_vae_state_dict_to_rgba,
    flux2_autoencoder_params,
)
from extensions.rgba_training.flux2_rgba import (
    Flux2Klein4BRGBAModel,
    Flux2Klein9BRGBAModel,
)
from toolkit.sampling_lora import validate_sampling_lora_path


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

    def test_selected_rgba_vae_overrides_bundled_vae_for_both_klein_models(self):
        selected = str(Path("selected") / "klein-rgba.safetensors")
        local_model = str(Path("models") / "klein")

        for model_type in (Flux2Klein4BRGBAModel, Flux2Klein9BRGBAModel):
            with self.subTest(model_type=model_type.__name__):
                model = object.__new__(model_type)
                model.model_config = SimpleNamespace(vae_path=selected)
                with mock.patch("os.path.exists", return_value=True):
                    resolved = model.get_flux2_vae_source(local_model)
                self.assertEqual(resolved, selected)

    def test_bundled_vae_is_only_used_without_an_explicit_selection(self):
        local_model = str(Path("models") / "klein")
        expected = str(Path(local_model) / "ae.safetensors")
        model = object.__new__(Flux2Klein4BRGBAModel)
        model.model_config = SimpleNamespace(vae_path=None)

        with mock.patch("os.path.exists", return_value=True):
            resolved = model.get_flux2_vae_source(local_model)

        self.assertEqual(resolved, expected)

    def test_klein_sampling_lora_hidden_size_rejects_the_other_model(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "klein4b.safetensors"
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": torch.zeros(2, 3072),
                    "diffusion_model.block.proj.lora_up.weight": torch.zeros(3072, 2),
                },
                str(path),
            )

            self.assertEqual(
                validate_sampling_lora_path(str(path), 3072), str(path.resolve())
            )
            with self.assertRaisesRegex(ValueError, "hidden size 4096"):
                validate_sampling_lora_path(str(path), 4096)


if __name__ == "__main__":
    unittest.main()
