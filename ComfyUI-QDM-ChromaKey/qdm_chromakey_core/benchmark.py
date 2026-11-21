"""python -m qdm_chromakey_core.benchmark --device cpu --size 1920x1080

Measures complete CPU-input/CPU-output tiled inference, excluding file IO.
Without --checkpoint this measures architecture speed, not trained quality.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from .inference import load_model, run_tiled
from .v4 import KeyMatteV4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", nargs="+", default=["1280x720", "1920x1080"])
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = load_model(args.checkpoint, device, dtype)[0] if args.checkpoint else KeyMatteV4().eval().to(device=device, dtype=dtype)
    count = sum(p.numel() for p in model.parameters())
    report = {"architecture": model.architecture_id, "trained_checkpoint": args.checkpoint,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
              "torch": torch.__version__, "threads": args.threads, "tile": args.tile,
              "parameters": count, "fp16_weights_mib": count*2/1024**2, "measurements": []}
    for size in args.size:
        width, height = map(int, size.lower().split("x"))
        source = torch.rand(3, height, width)
        for _ in range(2):
            run_tiled(model, source, args.tile, autocast_dtype=dtype if device.type == "cuda" else None)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            clean, alpha = run_tiled(model, source, args.tile, autocast_dtype=dtype if device.type == "cuda" else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
            if not torch.isfinite(clean).all() or not torch.isfinite(alpha).all():
                raise RuntimeError("Nonfinite inference output")
        row = {"size": size, "median_ms": statistics.median(durations)*1000,
               "min_ms": min(durations)*1000, "max_ms": max(durations)*1000,
               "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else None}
        report["measurements"].append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
