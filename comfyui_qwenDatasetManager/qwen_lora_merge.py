import os
import re
import hashlib
import uuid
from pathlib import Path

import folder_paths


LORA_NONE = "None"
MERGE_TYPES = ["concat", "weighted_sum", "weighted_average", "ties", "dare_linear"]


def _import_safetensors():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except Exception as exc:
        raise ImportError(
            "Qwen LoRA merge requires torch and safetensors. Run this node inside ComfyUI "
            "or install safetensors in the same Python environment."
        ) from exc
    return torch, load_file, save_file


def _safe_filename(filename):
    filename = filename.strip() or "qwen_merged_lora.safetensors"
    filename = re.sub(r"[\\/:*?\"<>|]+", "_", filename)
    if not filename.lower().endswith(".safetensors"):
        filename += ".safetensors"
    return filename


def _metadata_to_strings(metadata):
    return {str(k): str(v) for k, v in (metadata or {}).items()}


def _lora_choices():
    choices = [LORA_NONE]
    try:
        choices.extend(folder_paths.get_filename_list("loras"))
    except Exception:
        pass

    # Also expose .safetensors placed next to this custom node repository. This
    # makes local reference files selectable without copying them into ComfyUI.
    repo_root = Path(__file__).resolve().parent.parent
    for path in sorted(repo_root.glob("*.safetensors")):
        if path.name not in choices:
            choices.append(path.name)

    return choices


def _resolve_lora_path(name):
    if not name or name == LORA_NONE:
        return None

    allowed_roots = [Path(__file__).resolve().parent.parent]
    try:
        allowed_roots.extend(Path(path).resolve() for path in folder_paths.get_folder_paths("loras"))
    except Exception:
        pass

    candidate = Path(name).expanduser().resolve(strict=False)
    if candidate.is_file() and any(candidate == root or root in candidate.parents for root in allowed_roots):
        return str(candidate)

    try:
        full_path = folder_paths.get_full_path("loras", name)
        resolved_full_path = Path(full_path).resolve() if full_path else None
        if (
            resolved_full_path
            and resolved_full_path.is_file()
            and any(
                resolved_full_path == root or root in resolved_full_path.parents
                for root in allowed_roots
            )
        ):
            return str(resolved_full_path)
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    repo_path = (repo_root / name).resolve(strict=False)
    if repo_path.is_file() and (repo_path == repo_root or repo_root in repo_path.parents):
        return str(repo_path)

    raise FileNotFoundError(f"LoRA file not found: {name}")


def _resolve_output_dir(save_to, subfolder):
    if save_to == "ComfyUI loras":
        try:
            base_dir = folder_paths.get_folder_paths("loras")[0]
        except Exception as exc:
            raise ValueError("Could not resolve ComfyUI loras directory.") from exc
    else:
        base_dir = folder_paths.get_output_directory()

    base_path = Path(base_dir).resolve()
    subfolder = subfolder.strip()
    if not subfolder:
        return str(base_path)
    if Path(subfolder).is_absolute() or '\x00' in subfolder:
        raise ValueError("LoRA subfolder must be relative")
    output_path = (base_path / subfolder).resolve(strict=False)
    if output_path != base_path and base_path not in output_path.parents:
        raise ValueError("LoRA output path must stay inside the selected ComfyUI directory")
    return str(output_path)


def _is_lora_a_key(key):
    return key.endswith(".lora_A.weight")


def _lora_b_key(a_key):
    return a_key[:-len(".lora_A.weight")] + ".lora_B.weight"


def _same_shape(tensors):
    if not tensors:
        return False
    shape = tuple(tensors[0].shape)
    return all(tuple(t.shape) == shape for t in tensors)


def _weighted_tensor_merge(torch, tensors, strengths, merge_type, density, seed):
    calc_tensors = [tensor.float() * float(strength) for tensor, strength in zip(tensors, strengths)]

    if merge_type == "weighted_average":
        denom = sum(float(strength) for strength in strengths)
        if abs(denom) < 1e-12:
            denom = sum(abs(float(strength)) for strength in strengths) or 1.0
        return sum(calc_tensors) / denom

    if merge_type == "ties":
        density = max(0.0, min(1.0, float(density)))
        pruned = []
        for tensor in calc_tensors:
            if density < 1.0:
                flat_abs = tensor.abs().flatten()
                keep = max(1, int(flat_abs.numel() * density))
                threshold = torch.topk(flat_abs, keep).values[-1]
                tensor = torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))
            pruned.append(tensor)

        stacked = torch.stack(pruned, dim=0)
        elected_sign = torch.sign(stacked.sum(dim=0))
        agreeing = torch.where(
            (torch.sign(stacked) == elected_sign.unsqueeze(0)) | (stacked == 0),
            stacked,
            torch.zeros_like(stacked),
        )
        counts = (agreeing != 0).sum(dim=0).clamp_min(1)
        return agreeing.sum(dim=0) / counts

    if merge_type == "dare_linear":
        density = max(1e-6, min(1.0, float(density)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        dropped = []
        for tensor in calc_tensors:
            mask = torch.rand(tensor.shape, generator=generator, device="cpu") < density
            dropped.append(torch.where(mask, tensor / density, torch.zeros_like(tensor)))
        return sum(dropped)

    return sum(calc_tensors)


def _merge_regular_key(torch, key, states, strengths, merge_type, density, seed):
    tensors = []
    active_strengths = []
    dtypes = []
    for state, strength in zip(states, strengths):
        if key in state:
            tensors.append(state[key])
            active_strengths.append(strength)
            dtypes.append(state[key].dtype)

    if not tensors:
        return None

    if len(tensors) == 1:
        tensor = tensors[0].clone()
        if tensor.is_floating_point():
            tensor = tensor * float(active_strengths[0])
        return tensor

    if not _same_shape(tensors):
        shapes = ', '.join(str(tuple(tensor.shape)) for tensor in tensors)
        raise ValueError(f"Cannot merge '{key}': incompatible tensor shapes ({shapes})")

    merged = _weighted_tensor_merge(torch, tensors, active_strengths, merge_type, density, seed)
    return merged.to(dtype=dtypes[0])


def _pair_alpha_key(a_key):
    return a_key[:-len(".lora_A.weight")] + ".alpha"


def _pair_source_scale(state, a_key, rank):
    alpha = state.get(_pair_alpha_key(a_key))
    if alpha is None:
        return 1.0
    try:
        return float(alpha.item()) / float(rank)
    except (AttributeError, TypeError, ValueError):
        return 1.0


def _normalized_strengths(strengths, merge_type):
    values = [float(strength) for strength in strengths]
    if merge_type != "weighted_average":
        return values
    denominator = sum(values)
    if abs(denominator) < 1e-12:
        denominator = sum(abs(value) for value in values) or 1.0
    return [value / denominator for value in values]


def _merge_concat_pair(torch, a_key, states, strengths, merge_type="weighted_sum"):
    b_key = _lora_b_key(a_key)
    a_tensors = []
    b_tensors = []
    first_dtype = None

    active = [
        (state, strength)
        for state, strength in zip(states, strengths)
        if a_key in state and b_key in state
    ]
    effective_strengths = _normalized_strengths(
        [strength for _, strength in active],
        merge_type
    )

    for (state, _), strength in zip(active, effective_strengths):
        a_tensor = state[a_key]
        b_tensor = state[b_key]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2 or b_tensor.shape[1] != a_tensor.shape[0]:
            raise ValueError(f"Invalid LoRA A/B shapes for '{a_key}'")
        if a_tensors and a_tensor.shape[1:] != a_tensors[0].shape[1:]:
            raise ValueError(f"Incompatible LoRA input dimensions for '{a_key}'")
        if b_tensors and b_tensor.shape[:1] + b_tensor.shape[2:] != b_tensors[0].shape[:1] + b_tensors[0].shape[2:]:
            raise ValueError(f"Incompatible LoRA output dimensions for '{a_key}'")

        first_dtype = first_dtype or a_tensor.dtype
        a_tensors.append(a_tensor)
        source_scale = _pair_source_scale(state, a_key, a_tensor.shape[0])
        b_tensors.append(
            b_tensor.float().mul(float(strength) * source_scale).to(dtype=b_tensor.dtype)
        )

    if not a_tensors:
        return None, None

    return (
        torch.cat(a_tensors, dim=0).to(dtype=first_dtype),
        torch.cat(b_tensors, dim=1).to(dtype=b_tensors[0].dtype),
    )


def _stable_key_seed(seed, key):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:8], "big")) % (2 ** 63 - 1)


def _merge_delta_pair(torch, a_key, states, strengths, merge_type, density, seed):
    b_key = _lora_b_key(a_key)
    deltas = []
    active_strengths = []
    ranks = []
    output_dtype = None

    for state, strength in zip(states, strengths):
        if a_key not in state or b_key not in state:
            continue
        a_tensor = state[a_key]
        b_tensor = state[b_key]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2 or b_tensor.shape[1] != a_tensor.shape[0]:
            raise ValueError(f"Invalid LoRA A/B shapes for '{a_key}'")
        delta = b_tensor.float() @ a_tensor.float()
        if deltas and delta.shape != deltas[0].shape:
            raise ValueError(f"Incompatible LoRA delta shapes for '{a_key}'")
        deltas.append(delta)
        active_strengths.append(float(strength) * _pair_source_scale(state, a_key, a_tensor.shape[0]))
        ranks.append(a_tensor.shape[0])
        output_dtype = output_dtype or a_tensor.dtype

    if not deltas:
        return None, None

    merged_delta = _weighted_tensor_merge(
        torch,
        deltas,
        active_strengths,
        merge_type,
        density,
        _stable_key_seed(seed, a_key)
    )
    max_rank = min(sum(ranks), min(merged_delta.shape))
    u_matrix, singular_values, vh_matrix = torch.linalg.svd(
        merged_delta,
        full_matrices=False
    )
    singular_values = singular_values[:max_rank].clamp_min(0)
    sqrt_values = singular_values.sqrt()
    merged_a = sqrt_values[:, None] * vh_matrix[:max_rank, :]
    merged_b = u_matrix[:, :max_rank] * sqrt_values[None, :]
    return merged_a.to(dtype=output_dtype), merged_b.to(dtype=output_dtype)


def _build_metadata(source_names, source_metadata, merge_type, strengths, density):
    metadata = {}
    for item in source_metadata:
        for key, value in item.items():
            metadata.setdefault(key, value)

    metadata.update(
        {
            "format": metadata.get("format", "pt"),
            "software": "QwenDatasetManager LoRA Merge",
            "merge_type": merge_type,
            "merge_density": str(density),
            "merged_loras": ", ".join(source_names),
            "merged_strengths": ", ".join(str(float(s)) for s in strengths),
            "ss_base_model_version": metadata.get("ss_base_model_version", "qwen_image"),
        }
    )
    return _metadata_to_strings(metadata)


class QwenLoraMerge:
    """
    Merge up to four Qwen Image/Edit LoRA safetensors files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        choices = _lora_choices()
        return {
            "required": {
                "lora_1": (choices,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (choices,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_3": (choices,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_4": (choices,),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "merge_type": (MERGE_TYPES,),
                "density": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("QWEN_LORA_MERGE",)
    RETURN_NAMES = ("merged_lora",)
    FUNCTION = "merge_loras"
    CATEGORY = "Qwen/LoRA"

    def merge_loras(
        self,
        lora_1,
        strength_1,
        lora_2,
        strength_2,
        lora_3,
        strength_3,
        lora_4,
        strength_4,
        merge_type,
        density,
        seed,
    ):
        torch, load_file, _ = _import_safetensors()

        selected = [
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
        ]
        selected = [(name, strength) for name, strength in selected if name and name != LORA_NONE]
        if not selected:
            raise ValueError("Select at least one LoRA.")

        states = []
        metadata = []
        source_names = []
        for name, strength in selected:
            path = _resolve_lora_path(name)
            states.append(load_file(path, device="cpu"))
            source_names.append(os.path.basename(path))
            try:
                from safetensors import safe_open

                with safe_open(path, framework="pt", device="cpu") as handle:
                    metadata.append(_metadata_to_strings(handle.metadata()))
            except Exception:
                metadata.append({})

        strengths = [strength for _, strength in selected]
        all_keys = sorted(set().union(*(state.keys() for state in states)))
        merged = {}
        consumed = set()

        for a_key in [key for key in all_keys if _is_lora_a_key(key)]:
            b_key = _lora_b_key(a_key)
            if b_key not in all_keys:
                continue
            if merge_type in {"ties", "dare_linear"}:
                merged_a, merged_b = _merge_delta_pair(
                    torch, a_key, states, strengths, merge_type, density, seed
                )
            else:
                merged_a, merged_b = _merge_concat_pair(
                    torch, a_key, states, strengths, merge_type
                )
            if merged_a is not None:
                merged[a_key] = merged_a
                merged[b_key] = merged_b
                alpha_key = _pair_alpha_key(a_key)
                merged[alpha_key] = torch.tensor(
                    float(merged_a.shape[0]),
                    dtype=torch.float32
                )
                consumed.update({a_key, b_key, alpha_key})

        for key in all_keys:
            if key in consumed:
                continue
            merged_tensor = _merge_regular_key(torch, key, states, strengths, merge_type, density, seed)
            if merged_tensor is not None:
                merged[key] = merged_tensor

        result = {
            "state_dict": merged,
            "metadata": _build_metadata(source_names, metadata, merge_type, strengths, density),
            "source_names": source_names,
            "merge_type": merge_type,
            "tensor_count": len(merged),
        }
        print(
            f"QwenLoraMerge: merged {len(source_names)} LoRA(s), "
            f"{len(merged)} tensors, type={merge_type}"
        )
        return (result,)


class QwenLoraSave:
    """
    Save merged LoRA safetensors. Marked as OUTPUT_NODE so ComfyUI executes it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_lora": ("QWEN_LORA_MERGE",),
                "filename": ("STRING", {"default": "qwen_merged_lora.safetensors", "multiline": False}),
                "save_to": (["ComfyUI loras", "ComfyUI output"],),
                "subfolder": ("STRING", {"default": "merged", "multiline": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora"
    OUTPUT_NODE = True
    CATEGORY = "Qwen/LoRA"

    def save_lora(self, merged_lora, filename, save_to, subfolder):
        _, _, save_file = _import_safetensors()

        output_dir = _resolve_output_dir(save_to, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, _safe_filename(filename))
        temp_path = os.path.join(output_dir, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
        try:
            save_file(
                merged_lora["state_dict"],
                temp_path,
                metadata=_metadata_to_strings(merged_lora.get("metadata", {})),
            )
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        message = (
            f"Saved merged Qwen LoRA: {path} "
            f"({merged_lora.get('tensor_count', 0)} tensors, "
            f"type={merged_lora.get('merge_type', 'unknown')})"
        )
        print(f"QwenLoraSave: {message}")
        return {"ui": {"text": [message]}}


NODE_CLASS_MAPPINGS = {
    "QwenLoraMerge": QwenLoraMerge,
    "QwenLoraSave": QwenLoraSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenLoraMerge": "Qwen LoRA Merge",
    "QwenLoraSave": "Qwen LoRA Save",
}
