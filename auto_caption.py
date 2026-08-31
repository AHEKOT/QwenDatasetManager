"""Local GGUF vision-language captioning through llama-cpp-python."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import threading
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_AUTO_CAPTION_CONFIG = {
    "systemPrompt": (
        "You create accurate captions for image-model training datasets. Describe only "
        "what is visibly present: the subject, appearance, clothing, pose or action, "
        "composition, lighting, setting, and important details. Do not invent hidden facts. "
        "Return only the caption text, without a title, label, markdown, or commentary."
    ),
    "modelId": "",
    "variantId": "",
    "prefix": "",
    "suffix": "",
}

_QUANT_RE = re.compile(
    r"(?i)(?:^|[-_. ])(IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|"
    r"BF16|F16|F32|FP16|FP32)(?=$|[-_. ])"
)
_PAIR_IGNORED_TOKENS = {
    "gguf", "mmproj", "mm", "proj", "projector", "vision", "model",
    "f16", "f32", "fp16", "fp32", "bf16",
}

_VL_MAX_IMAGE_SIZE = (1024, 1024)


def _stable_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.lower().encode("utf-8")).hexdigest()[:16]
    return f"{kind}-{digest}"


def _display_family_and_quantization(path: Path):
    stem = path.stem
    matches = list(_QUANT_RE.finditer(stem))
    if matches:
        match = matches[-1]
        quantization = match.group(1).upper().replace("FP16", "F16").replace("FP32", "F32")
        family = (stem[:match.start()] + stem[match.end():]).strip("-_. ")
    else:
        quantization = "GGUF"
        family = stem.strip("-_. ")
    family = re.sub(r"[-_. ]{2,}", "-", family).strip("-") or stem
    return family, quantization


def _pair_tokens(path_or_name) -> set[str]:
    stem = Path(path_or_name).stem.lower()
    stem = _QUANT_RE.sub(" ", stem)
    tokens = set(re.findall(r"[a-z0-9]+", stem))
    return {token for token in tokens if token not in _PAIR_IGNORED_TOKENS}


def _projector_rank(path: Path):
    name = path.stem.lower()
    if "f16" in name or "fp16" in name:
        precision = 0
    elif "bf16" in name:
        precision = 1
    elif "f32" in name or "fp32" in name:
        precision = 2
    else:
        precision = 3
    return precision, name


def _select_projector(model_path: Path, projectors: list[Path]):
    if not projectors:
        return None
    if len(projectors) == 1:
        return projectors[0]

    model_tokens = _pair_tokens(model_path.name)
    scored = []
    for projector in projectors:
        projector_tokens = _pair_tokens(projector.name)
        common = len(model_tokens & projector_tokens)
        missing = len(model_tokens - projector_tokens)
        extra = len(projector_tokens - model_tokens)
        score = common * 10 - missing * 2 - extra
        scored.append((score, _projector_rank(projector), projector))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2] if scored[0][0] > 0 else None


def _handler_for_family(family: str) -> str:
    normalized = family.lower().replace("_", "-")
    if "qwen3.5" in normalized or "qwen35" in normalized:
        return "mtmd"
    if "qwen" in normalized and "vl" in normalized:
        return "qwen25-vl"
    if "minicpm" in normalized:
        return "minicpm-v2.6"
    if "moondream" in normalized:
        return "moondream2"
    if "nanollava" in normalized:
        return "nanollava"
    if "llama-3" in normalized and "vision" in normalized:
        return "llama-3-vision-alpha"
    if "llava" in normalized and ("1.6" in normalized or "v1.6" in normalized):
        return "llava-1-6"
    return "llava-1-5"


def scan_gguf_catalog(models_dir: Path):
    """Return selectable model families and quantized variants from models/llm."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    gguf_files = sorted(
        (path for path in models_dir.iterdir() if path.is_file() and path.suffix.lower() == ".gguf"),
        key=lambda path: path.name.lower(),
    )
    projectors = [path for path in gguf_files if "mmproj" in path.name.lower()]
    model_files = [path for path in gguf_files if "mmproj" not in path.name.lower()]

    groups = {}
    warnings = []
    used_projectors = set()
    for model_path in model_files:
        family, quantization = _display_family_and_quantization(model_path)
        projector = _select_projector(model_path, projectors)
        if projector is None:
            warnings.append(f"{model_path.name}: matching mmproj GGUF was not found")
            continue
        used_projectors.add(projector)
        family_key = family.casefold()
        group = groups.setdefault(family_key, {
            "id": _stable_id("model", family),
            "label": family,
            "variants": [],
        })
        relative_model = model_path.relative_to(models_dir).as_posix()
        relative_projector = projector.relative_to(models_dir).as_posix()
        group["variants"].append({
            "id": _stable_id("variant", f"{relative_model}|{relative_projector}"),
            "quantization": quantization,
            "filename": model_path.name,
            "mmprojFilename": projector.name,
            "handler": _handler_for_family(family),
        })

    for projector in projectors:
        if projector not in used_projectors:
            warnings.append(f"{projector.name}: no matching language-model GGUF was found")

    models = sorted(groups.values(), key=lambda group: group["label"].lower())
    for group in models:
        group["variants"].sort(key=lambda item: (item["quantization"], item["filename"].lower()))
    return {
        "directory": "models/llm",
        "models": models,
        "warnings": warnings,
    }


def resolve_catalog_variant(models_dir: Path, model_id: str, variant_id: str):
    catalog = scan_gguf_catalog(models_dir)
    for model in catalog["models"]:
        if model["id"] != model_id:
            continue
        for variant in model["variants"]:
            if variant["id"] == variant_id:
                return {
                    **variant,
                    "modelId": model["id"],
                    "modelLabel": model["label"],
                    "modelPath": Path(models_dir) / variant["filename"],
                    "mmprojPath": Path(models_dir) / variant["mmprojFilename"],
                }
    raise ValueError("Selected GGUF model or quantization is no longer available")


def normalize_auto_caption_config(config, *, catalog=None, require_model=False):
    if not isinstance(config, dict):
        raise ValueError("Auto Caption config must be an object")
    limits = {
        "systemPrompt": 20_000,
        "modelId": 128,
        "variantId": 128,
        "prefix": 10_000,
        "suffix": 10_000,
    }
    normalized = {}
    for key, limit in limits.items():
        value = config.get(key, DEFAULT_AUTO_CAPTION_CONFIG[key])
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if len(value) > limit:
            raise ValueError(f"{key} is too long")
        normalized[key] = value
    if not normalized["systemPrompt"].strip():
        raise ValueError("System prompt cannot be empty")

    if catalog is not None and normalized["modelId"] and normalized["variantId"]:
        valid = any(
            model["id"] == normalized["modelId"]
            and any(variant["id"] == normalized["variantId"] for variant in model["variants"])
            for model in catalog.get("models", [])
        )
        if not valid:
            raise ValueError("Selected GGUF model or quantization is no longer available")
    if require_model and (not normalized["modelId"] or not normalized["variantId"]):
        raise ValueError("Select a GGUF model and quantization first")
    return normalized


def _image_data_uri(image_path: Path) -> str:
    """Encode a VL-only copy, downscaled to fit within 1024x1024 pixels."""
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail(_VL_MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class AutoCaptionEngine:
    """A single cached llama.cpp VLM. Access is serialized by the caller."""

    def __init__(self):
        self._lock = threading.RLock()
        self._llm = None
        self._loaded_key = None

    def close(self):
        with self._lock:
            if self._llm is not None:
                close = getattr(self._llm, "close", None)
                if close:
                    close()
            self._llm = None
            self._loaded_key = None

    @staticmethod
    def _make_handler(handler_name: str, mmproj_path: Path):
        try:
            from llama_cpp import llama_chat_format
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Rerun the application installer."
            ) from exc

        handler_classes = {
            "mtmd": "MTMDChatHandler",
            "qwen25-vl": "Qwen25VLChatHandler",
            "minicpm-v2.6": "MiniCPMv26ChatHandler",
            "moondream2": "MoondreamChatHandler",
            "nanollava": "NanollavaChatHandler",
            "llama-3-vision-alpha": "Llama3VisionAlphaChatHandler",
            "llava-1-6": "Llava16ChatHandler",
            "llava-1-5": "Llava15ChatHandler",
        }
        class_name = handler_classes.get(handler_name, "Llava15ChatHandler")
        handler_class = getattr(llama_chat_format, class_name, None)
        if handler_class is None:
            raise RuntimeError(
                f"Installed llama-cpp-python does not provide {class_name}; rerun the installer"
            )
        return handler_class(clip_model_path=str(mmproj_path), verbose=False)

    def ensure_loaded(self, variant):
        key = (str(variant["modelPath"]), str(variant["mmprojPath"]), variant["handler"])
        with self._lock:
            if self._llm is not None and self._loaded_key == key:
                return
            self.close()
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Rerun the application installer."
                ) from exc
            handler = self._make_handler(variant["handler"], variant["mmprojPath"])
            try:
                self._llm = Llama(
                    model_path=str(variant["modelPath"]),
                    chat_handler=handler,
                    n_ctx=4096,
                    n_batch=512,
                    n_gpu_layers=-1,
                    n_threads=max(1, (os.cpu_count() or 4) // 2),
                    verbose=False,
                )
            except Exception:
                self._llm = None
                raise
            self._loaded_key = key

    def caption(self, image_path: Path, config):
        with self._lock:
            if self._llm is None:
                raise RuntimeError("Caption model is not loaded")
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": config["systemPrompt"].strip()},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_uri(Path(image_path))},
                            },
                            {
                                "type": "text",
                                "text": "Create the requested dataset caption. Return only the caption text.",
                            },
                        ],
                    },
                ],
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.05,
                max_tokens=384,
                seed=0,
            )
            try:
                generated = response["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("llama.cpp returned an invalid caption response") from exc
            generated = re.sub(r"^\s*<think>.*?</think>\s*", "", str(generated), flags=re.DOTALL).strip()
            if not generated:
                raise RuntimeError("The model returned an empty caption")
            return f"{config['prefix']}{generated}{config['suffix']}"
