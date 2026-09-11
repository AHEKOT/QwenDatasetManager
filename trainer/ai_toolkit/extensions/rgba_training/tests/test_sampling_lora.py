import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

import torch
from safetensors.torch import save_file

from toolkit.sampling_lora import (
    SamplingLoRAMixin,
    _sampling_lora_metadata,
    build_sampling_lora_network,
    validate_sampling_lora_path,
)
from toolkit.lora_special import FullModule, LoRAModule


class _QwenModel:
    torch_dtype = torch.float32
    target_lora_modules = ["QwenImageTransformer2DModel"]
    use_old_lokr_format = False

    @staticmethod
    def convert_lora_weights_before_load(state):
        return state


class _FluxModel:
    @staticmethod
    def convert_lora_weights_before_load(state):
        return OrderedDict(
            (key.replace("diffusion_model.", "transformer."), value)
            for key, value in state.items()
        )


class SamplingLoRAMetadataTests(unittest.TestCase):
    def test_inference_module_materializes_checkpoint_without_blank_allocation(self):
        network = Mock()
        network.network_type = "lora"
        original = torch.nn.Linear(8, 6, bias=False)
        module = LoRAModule(
            "transformer$$block$$projection",
            original,
            lora_dim=4,
            alpha=4,
            network=network,
            initialize_weights=False,
        )

        self.assertTrue(module.lora_down.weight.is_meta)
        self.assertTrue(module.lora_up.weight.is_meta)

        down = torch.randn(4, 8)
        up = torch.randn(6, 4)
        module.load_state_dict(
            {
                "lora_down.weight": down,
                "lora_up.weight": up,
                "alpha": torch.tensor(4),
            },
            assign=True,
        )

        self.assertFalse(module.lora_down.weight.is_meta)
        self.assertFalse(module.lora_up.weight.is_meta)
        self.assertEqual(module.lora_down.weight.data_ptr(), down.data_ptr())
        self.assertEqual(module.lora_up.weight.data_ptr(), up.data_ptr())

    def test_sampling_network_directly_materializes_safetensors_weights(self):
        class _Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(8, 6, bias=False)

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = _Attention()

        QwenImageTransformer2DModel = type(
            "QwenImageTransformer2DModel",
            (torch.nn.Module,),
            {
                "__init__": lambda self: (
                    torch.nn.Module.__init__(self),
                    setattr(self, "transformer_blocks", torch.nn.ModuleList([_Block()])),
                )[-1],
            },
        )
        transformer = QwenImageTransformer2DModel()
        down = torch.randn(4, 8)
        up = torch.randn(6, 4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen.safetensors"
            save_file({
                "transformer_blocks.0.attn.to_q.lora_down.weight": down,
                "transformer_blocks.0.attn.to_q.lora_up.weight": up,
                "transformer_blocks.0.attn.to_q.alpha": torch.tensor(4.0),
            }, str(path))
            network = build_sampling_lora_network(
                base_model=_QwenModel(),
                transformer=transformer,
                lora_path=str(path),
                device=torch.device("cpu"),
                use_layer_offloading=False,
            )

        self.assertEqual(len(network.unet_loras), 1)
        module = network.unet_loras[0]
        self.assertFalse(module.lora_down.weight.is_meta)
        self.assertFalse(module.lora_up.weight.is_meta)
        torch.testing.assert_close(module.lora_down.weight, down)
        torch.testing.assert_close(module.lora_up.weight, up)

    def test_prepare_materializes_configured_adapter_once(self):
        model = object.__new__(SamplingLoRAMixin)
        model.sample_lora_path = "turbo.safetensors"
        model._ensure_sampling_lora_network = Mock()

        model.prepare_sampling_lora()

        model._ensure_sampling_lora_network.assert_called_once_with()

    def test_prepare_skips_model_without_sampling_adapter(self):
        model = object.__new__(SamplingLoRAMixin)
        model.sample_lora_path = None
        model._ensure_sampling_lora_network = Mock()

        model.prepare_sampling_lora()

        model._ensure_sampling_lora_network.assert_not_called()

    def test_qwen_lightning_layout_preserves_rank_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen.safetensors"
            save_file({
                "transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros(4, 8),
                "transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros(8, 4),
                "transformer_blocks.0.attn.to_q.alpha": torch.tensor(2.0),
            }, str(path))

            self.assertEqual(validate_sampling_lora_path(str(path)), str(path.resolve()))
            dims, alphas, full_modules, native = _sampling_lora_metadata(_QwenModel(), str(path))

            key = "transformer$$transformer_blocks$$0$$attn$$to_q"
            self.assertEqual(dims[key], 4)
            self.assertEqual(alphas[key], 2.0)
            self.assertEqual(full_modules, set())
            self.assertTrue(native)

    def test_flux_peft_layout_is_normalized_to_native_transformer_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "klein.safetensors"
            save_file({
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.zeros(8, 16),
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.zeros(16, 8),
            }, str(path))

            dims, alphas, full_modules, native = _sampling_lora_metadata(_FluxModel(), str(path))

            key = "transformer$$double_blocks$$0$$img_attn$$qkv"
            self.assertEqual(dims[key], 8)
            self.assertEqual(alphas[key], 8.0)
            self.assertEqual(full_modules, set())
            self.assertFalse(native)

    def test_flux_full_weight_diffs_are_attached_and_loaded(self):
        class _ScaleRMSNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(8))

            def forward(self, value):
                return value * self.scale

        class _NormContainer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.key_norm = _ScaleRMSNorm()

        class _Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Linear(8, 8, bias=False)
                self.norm = _NormContainer()

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.img_attn = _Attention()

        class _Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.double_blocks = torch.nn.ModuleList([_Block()])

        class _BuildableFluxModel(_FluxModel):
            torch_dtype = torch.float32
            target_lora_modules = ["_Transformer"]
            use_old_lokr_format = False

        transformer = _Transformer()
        down = torch.randn(4, 8)
        up = torch.randn(8, 4)
        norm_diff = torch.randn(8)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "klein_with_norm_diff.safetensors"
            save_file({
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": down,
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": up,
                "diffusion_model.double_blocks.0.img_attn.norm.key_norm.diff": norm_diff,
            }, str(path))
            network = build_sampling_lora_network(
                base_model=_BuildableFluxModel(),
                transformer=transformer,
                lora_path=str(path),
                device=torch.device("cpu"),
                use_layer_offloading=False,
            )

        self.assertEqual(len(network.unet_loras), 2)
        full_module = next(
            module for module in network.unet_loras if isinstance(module, FullModule)
        )
        self.assertEqual(full_module.parameter_name, "scale")
        torch.testing.assert_close(full_module.diff, norm_diff)
        value = torch.ones(1, 8)
        with network:
            actual = transformer.double_blocks[0].img_attn.norm.key_norm(value)
        torch.testing.assert_close(actual, value * (1 + norm_diff))


if __name__ == "__main__":
    unittest.main()
