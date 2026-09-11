import argparse
import json
import time
from pathlib import Path
import torch
from .model import CleanMatte
from .inference import predict, load_model


def benchmark(device="cpu", width=1920, height=1080, repeats=3, checkpoint=None, threads=4):
    torch.set_num_threads(threads)
    model = load_model(checkpoint, device) if checkpoint else CleanMatte().fuse_for_inference().to(device=device, memory_format=torch.channels_last)
    image = torch.rand(1, 3, height, width, device=device).contiguous(memory_format=torch.channels_last)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        predict(model, image)
    times = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(model, image)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return {"device": device, "width": width, "height": height, "threads": threads,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_scope": "deployed convolutions after BatchNorm folding", "warmup_passes": 5,
            "seconds": times, "fps": len(times)/sum(times),
            "peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20 if device.startswith("cuda") else None,
            "includes": "global context, all native tiles, final alpha; excludes file decode and optional despill",
            "weights": str(checkpoint) if checkpoint else "random (speed only)"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = benchmark(args.device, args.width, args.height, args.repeats, args.checkpoint, args.threads)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
