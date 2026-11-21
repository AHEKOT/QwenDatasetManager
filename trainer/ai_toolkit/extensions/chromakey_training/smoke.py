"""Bounded CUDA overfit check; synthetic fixtures are NOT a quality benchmark.

python -m extensions.chromakey_training.smoke --steps 200 --output smoke.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ComfyUI-QDM-ChromaKey"))
from qdm_chromakey_core.v4 import KeyMatteV4
from .data_v4 import ChromaKeyDatasetV4
from .objectives import matting_loss, matting_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as temporary:
        files = []
        for index in range(4):
            image = Image.new("RGBA", (192, 192), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((30, 20, 170, 180), fill=(180, 90+index*20, 60, 255))
            draw.ellipse((70, 60, 110, 100), fill=(0, 0, 0, 0))
            for offset in range(10):
                draw.line((30+offset*5, 35, 10+offset*9, 5), fill=(120, 80, 40, 80+offset*12), width=1+offset%2)
            path = Path(temporary)/f"fixture_{index}.png"
            image.save(path)
            files.append(path)
        dataset = ChromaKeyDatasetV4(files, {"green_chance": 60, "blue_chance": 40,
            "spill_chance": 100, "spill_strength_min": .3, "spill_strength_max": .6,
            "detail_crop_percent": 0, "topology_chance": 0, "jpeg_chance": 0,
            "gradient_chance": 100, "gradient_strength": .06}, flip_probability=0)
        samples = [dataset[(i, 192, 192, 200+i, 1.)] for i in range(4)]
        batch = {key: torch.stack([row[key] for row in samples]).cuda()
                 for key in samples[0] if torch.is_tensor(samples[0][key])}
        model = KeyMatteV4().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        def evaluate():
            with torch.no_grad():
                result = model(batch["input"], batch["context_rgb"], batch["context_box"])
                return matting_metrics(result["foreground"], result["alpha"], batch)
        before = evaluate()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch["input"], batch["context_rgb"], batch["context_box"])
            loss, _ = matting_loss(prediction, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite smoke loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (step+1) % 25 == 0:
                print(f"smoke {step+1}/{args.steps}: {loss.item():.5f}", flush=True)
        torch.cuda.synchronize()
        after = evaluate()
        report = {"purpose": "synthetic overfit sanity check, not held-out quality",
                  "steps": args.steps, "elapsed_seconds": time.perf_counter()-start,
                  "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                  "before": before, "after": after}
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not (after["alpha_mae"] < before["alpha_mae"]*.5 and after["despill_mae"] < before["despill_mae"]*.8):
            raise RuntimeError("Overfit check did not learn alpha and foreground sufficiently")


if __name__ == "__main__":
    main()
