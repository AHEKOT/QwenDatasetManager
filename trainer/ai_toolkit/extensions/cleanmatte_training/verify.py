"""Executable training/CPU/GPU checks. Results are not a quality certificate."""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from .runtime import CleanMatte, predict
from .data import analytic_patch, composite
from .losses import objective, metrics


def run(output, steps=200, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = CleanMatte().to(device)
    foreground, alpha = analytic_patch(96, random.Random(42))
    # A fixed fixture deliberately checks learnability, not generalisation.
    rgb = composite(foreground, alpha, random.Random(7), {"green_chance": 100, "blue_chance": 0,
                                                         "white_chance": 0, "black_chance": 0})
    image, target = rgb[None].to(device), alpha[None].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.999))
    initial = metrics(predict(model.eval(), image), target)
    history = []
    started = time.perf_counter()
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, parts = objective(model(image), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        if (step+1) % 50 == 0:
            print(f"Learnability {step+1}/{steps}: loss={float(loss.detach()):.5f}", flush=True)
            history.append({"step": step+1, "loss": float(loss.detach()), **parts})
    final_alpha = predict(model.eval(), image)
    final = metrics(final_alpha, target)
    from .engine import pil_rgb
    pil_rgb(rgb).save(output/"fixture_input.png")
    pil_rgb(alpha.expand(3, -1, -1)).save(output/"fixture_target.png")
    pil_rgb(final_alpha[0].cpu().expand(3, -1, -1)).save(output/"fixture_prediction.png")
    result = {"parameters": sum(p.numel() for p in model.parameters()), "steps": steps,
              "device": device, "seconds": time.perf_counter()-started,
              "initial": initial, "final": final, "history": history,
              "scope": "Single-fixture overfit tests the training signal, not real-world quality."}
    (output/"learnability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if final["alpha_mae"] >= initial["alpha_mae"]*0.4 or final["thin_recall"] < 0.5:
        raise AssertionError(f"Alpha learnability check failed: {result}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.output, args.steps, args.device)
