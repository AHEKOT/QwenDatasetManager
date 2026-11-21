from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _model_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "diffusion_pytorch_model.safetensors"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not find diffusion_pytorch_model.safetensors below {path}"
    )


def _native_key_map(reference_native_vae: Path) -> dict[str, str]:
    """Return Diffusers-key -> native Wan/Qwen-key without loading reference weights."""
    with safe_open(str(reference_native_vae), framework="pt", device="cpu") as handle:
        native_keys = list(handle.keys())

    # Diffusers already owns the authoritative native-Wan -> Diffusers mapping.
    # Passing key names as values lets us invert it without loading another VAE.
    converted = convert_wan_vae_to_diffusers({key: key for key in native_keys})
    if len(converted) != len(native_keys):
        raise RuntimeError(
            "The reference VAE key layout is not a supported Wan/Qwen VAE layout: "
            f"{len(native_keys)} native keys became {len(converted)} Diffusers keys"
        )
    return {diffusers_key: native_key for diffusers_key, native_key in converted.items()}


def export_qwen_rgba_vae_for_comfy(
    source: Path,
    reference_native_vae: Path,
    output: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    source_file = _model_file(source)
    key_map = _native_key_map(reference_native_vae)
    state = load_file(str(source_file), device="cpu")

    missing = sorted(set(state) - set(key_map))
    extra = sorted(set(key_map) - set(state))
    if missing or extra:
        raise RuntimeError(
            "Source and native Qwen VAE architectures differ. "
            f"Unmapped source keys={missing[:8]}, missing source keys={extra[:8]}"
        )

    encoder_in = state["encoder.conv_in.weight"]
    decoder_out = state["decoder.conv_out.weight"]
    decoder_bias = state["decoder.conv_out.bias"]
    if encoder_in.shape[1] != 4 or decoder_out.shape[0] != 4 or decoder_bias.shape[0] != 4:
        raise ValueError(
            "Expected a trained RGBA Qwen VAE with four-channel encoder/decoder "
            f"boundaries, got {tuple(encoder_in.shape)}, {tuple(decoder_out.shape)}, "
            f"{tuple(decoder_bias.shape)}"
        )

    native_state = {
        key_map[key]: tensor.to(dtype=dtype).contiguous()
        for key, tensor in state.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    save_file(
        native_state,
        str(temporary),
        metadata={
            "format": "pt",
            "architecture": "qwen_image_vae_rgba",
            "input_channels": "4",
            "output_channels": "4",
            "source": str(source_file),
        },
    )
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained Diffusers Qwen RGBA VAE for ComfyUI's standard Load VAE node."
    )
    parser.add_argument("source", type=Path, help="Diffusers VAE directory or its safetensors file")
    parser.add_argument(
        "reference_native_vae",
        type=Path,
        help="A standard native Comfy Qwen/Wan VAE used only as a key-layout reference",
    )
    parser.add_argument("output", type=Path, help="Output .safetensors path")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    args = parser.parse_args()

    result = export_qwen_rgba_vae_for_comfy(
        args.source,
        args.reference_native_vae,
        args.output,
        DTYPES[args.dtype],
    )
    print(f"Saved ComfyUI Qwen RGBA VAE: {result}")


if __name__ == "__main__":
    main()
