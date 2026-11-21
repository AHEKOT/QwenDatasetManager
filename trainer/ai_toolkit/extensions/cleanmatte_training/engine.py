"""Self-contained training engine; no diffusion or legacy keyer dependencies."""
import copy
from contextlib import closing
import json
import math
import os
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import save_file

from .runtime import CleanMatte, MODEL_ID, predict
from .data import discover, inventory, make_batch, source_patch, composite
from .losses import objective, metrics


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def image_tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float()[None]/255


def pil_rgb(tensor):
    return Image.fromarray((tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8))


class JobBridge:
    """Existing queue protocol, including stop/save/sample requests."""
    def __init__(self, path=None):
        self.path = path
        self.job_id = os.environ.get("AITK_JOB_ID")

    def update(self, **values):
        if not self.path or not self.job_id:
            return
        allowed = {"status", "step", "info", "speed_string", "save_now", "sample_now"}
        if not values.keys() <= allowed:
            raise ValueError("Unsupported job status field")
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            with db:
                sql = ', '.join(f'"{key}" = ?' for key in values)
                db.execute(f'UPDATE "Job" SET {sql} WHERE id = ?', (*values.values(), self.job_id))

    def requests(self):
        if not self.path or not self.job_id:
            return (False, False, False)
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            row = db.execute('SELECT stop, save_now, sample_now FROM "Job" WHERE id = ?', (self.job_id,)).fetchone()
        return tuple(bool(v) for v in row) if row else (True, False, False)


class Trainer:
    def __init__(self, config, name):
        self.config, self.name = config, name
        if config.get("architecture", {}).get("id") != MODEL_ID:
            raise ValueError("Start a new CleanMatte job. Old model/config formats are not compatible.")
        self.train = config["train"]
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.seed = int(self.train.get("seed", 42))
        torch.manual_seed(self.seed)
        self.root = Path(config["training_folder"])/name
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples = self.root/"samples"
        self.samples.mkdir(exist_ok=True)
        self.bridge = JobBridge(config.get("sqlite_db_path"))
        self.bridge.update(info="Auditing RGBA sources and building an independent holdout")
        self.split = inventory(discover(config["datasets"]), self.seed)
        manifest_path = self.root/"dataset_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != self.split:
                raise ValueError("Dataset changed since this job was created. Start a new job to keep validation independent.")
        atomic_json(manifest_path, self.split)
        self.model = CleanMatte().to(self.device)
        self.ema = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train.get("lr", 4e-4),
                                           betas=(0.5, 0.999))
        precision = self.train.get("dtype", "bf16")
        self.amp = self.device.type == "cuda" and precision != "fp32"
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.dtype == torch.float16)
        self.step, self.best = 0, float("inf")
        self.corruption_start = None
        checkpoint = self.root/"training_state.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if state["architecture"] != MODEL_ID:
                raise ValueError("Incompatible checkpoint; start a new job.")
            self.model.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            self.step, self.best = int(state["step"]), float(state["best"])
            self.corruption_start = state.get("corruption_start")
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        atomic_json(self.root/"run_config.json", config)
        print(f"CleanMatte: {self.parameters:,} parameters; {len(self.split['train'])} training / "
              f"{len(self.split['validation'])} holdout sources. RGB losses: none.", flush=True)

    def export(self, filename):
        weights = {k: v.detach().cpu().contiguous() for k, v in self.ema.state_dict().items()}
        temporary = self.root/(filename+".tmp")
        save_file(weights, str(temporary), metadata={"architecture": MODEL_ID, "step": str(self.step),
                                                    "output": "alpha", "parameters": str(self.parameters)})
        temporary.replace(self.root/filename)
        primary = "cleanmatte_best.safetensors" if (self.root/"cleanmatte_best.safetensors").exists() else filename
        atomic_json(self.root/"artifacts.json", {"primary": primary, "architecture": MODEL_ID,
                                              "latest": "cleanmatte_latest.safetensors"})

    def save(self):
        state = {"architecture": MODEL_ID, "model": self.model.state_dict(), "ema": self.ema.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                 "step": self.step, "best": self.best, "corruption_start": self.corruption_start}
        temporary = self.root/"training_state.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(self.root/"training_state.pt")
        self.export("cleanmatte_latest.safetensors")
        atomic_json(self.root/"runtime.json", {
            "architecture": MODEL_ID, "parameters": self.parameters, "torch": str(torch.__version__),
            "device": str(self.device), "step": self.step,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(self.device)/2**20
                if self.device.type == "cuda" else None,
        })

    @torch.inference_mode()
    def validate(self):
        self.ema.eval()
        val = self.config.get("validation", {})
        records, clean_records = [], []
        # Fixed identities, crops, colours and corruption across every step.
        paths = self.split["validation"][:int(val.get("source_limit", 12))]
        size = int(val.get("crop_size", 512))
        for index, path in enumerate(paths):
            for native in (False, True):
                rng = random.Random(self.seed+100000+index*2+int(native))
                fg, target = source_patch(path, size, rng, native=native)
                clean_rng = random.Random()
                clean_rng.setstate(rng.getstate())
                clean_rgb = composite(fg, target, clean_rng, self.config.get("augmentation", {}), difficulty=0.0)
                rgb = composite(fg, target, rng, self.config.get("augmentation", {}), difficulty=1.0)
                clean_prediction = predict(self.ema, clean_rgb[None].to(self.device))
                clean_records.append(metrics(clean_prediction, target[None].to(self.device)))
                predicted = predict(self.ema, rgb[None].to(self.device))
                item = metrics(predicted, target[None].to(self.device))
                item["name"] = Path(path).name+(" / native" if native else " / full")
                records.append(item)
                if index == 0:
                    self.preview(rgb[None], predicted.cpu(), len(val.get("images", []))+int(native), target[None])
        keys = [key for key in records[0] if key != "name"]
        aggregate = {key: sum(record[key] for record in records)/len(records) for key in keys}
        clean_aggregate = {key: sum(record[key] for record in clean_records)/len(clean_records) for key in keys}
        gates = {"alpha_mae": 0.025, "background_leak": 0.01, "opaque_error": 0.01,
                 "mixed_mae": 0.1, "thin_mae": 0.1, "hole_leak": 0.01}
        gate_passed = all(clean_aggregate[key] <= limit for key, limit in gates.items())
        minimum_clean = int(self.train.get("clean_steps", max(1, self.train.get("steps", 50000)//5)))
        if self.corruption_start is None and self.step >= minimum_clean and gate_passed:
            self.corruption_start = self.step
        score = sum(aggregate[k] for k in ("background_leak", "opaque_error", "mixed_mae", "thin_mae", "hole_leak"))
        result = {"step": self.step, "scope": "fixed source-disjoint synthetic composites; despill disabled",
                  "score": score, "metrics": aggregate, "clean_metrics": clean_aggregate,
                  "curriculum_gate": {"passed": gate_passed, "thresholds": gates,
                                      "corruption_start": self.corruption_start},
                  "items": [{"name": "Clean / " + item["name"], "metrics": clean}
                            for item, clean in zip(records, clean_records)] +
                           [{"name": "Corrupted / " + item["name"], "metrics": {k: item[k] for k in keys}}
                            for item in records],
                  "passed": None, "failedChecks": "Quality requires review of alpha and native details."}
        atomic_json(self.root/"cleanmatte_validation.json", result)
        with (self.root/"validation_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result)+"\n")
        if score < self.best:
            self.best = score
            self.export("cleanmatte_best.safetensors")
        for index, path in enumerate(val.get("images", [])):
            with Image.open(path) as image:
                image = image.convert("RGB")
                maximum = int(val.get("max_side", 4096))
                if max(image.size) > maximum:
                    image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
                rgb = image_tensor(image)
            alpha = predict(self.ema, rgb.to(self.device))
            self.preview(rgb, alpha.cpu(), index)
        print(f"Validation step {self.step}: {json.dumps(aggregate)}", flush=True)
        return result

    def preview(self, rgb, alpha, index, target=None):
        image = pil_rgb(rgb[0])
        matte = pil_rgb(alpha[0].expand(3, -1, -1))
        rgba = image.convert("RGBA")
        rgba.putalpha(matte.convert("L"))
        stem = f"chroma_{self.step:09d}_{index}"
        rgba.save(self.samples/(stem+".png"))
        # Neutral foreground displays ONLY opacity, completely isolating it
        # from any colour correction or green/blue fringe in source RGB.
        panels = [("Input", image), ("Alpha (no despill)", matte)]
        if target is not None:
            panels.append(("Clean target alpha", pil_rgb(target[0].expand(3, -1, -1))))
        width = min(image.width, 768)
        height = max(1, round(image.height*width/image.width))
        sheet = Image.new("RGB", (width*len(panels), height+24), (32, 32, 32))
        draw = ImageDraw.Draw(sheet)
        for i, (label, panel) in enumerate(panels):
            sheet.paste(panel.resize((width, height)), (i*width, 24))
            draw.text((i*width+5, 5), label, fill="white")
        sheet.save(self.samples/(stem+"_sheet.png"))

    def run(self):
        steps = int(self.train.get("steps", 50000))
        accumulation = int(self.train.get("gradient_accumulation", 3))
        resolutions = self.train.get("resolutions", [512])
        if not resolutions or min(resolutions) < 32 or accumulation < 1:
            raise ValueError("Resolution must be >=32 and gradient accumulation >=1.")
        self.bridge.update(status="running", info="Training clean alpha")
        try:
            while self.step < steps:
                started = time.perf_counter()
                stop, save_now, sample_now = self.bridge.requests()
                if stop:
                    self.save()
                    self.bridge.update(status="stopped", info="CleanMatte stopped; resumable checkpoint saved")
                    return
                if save_now:
                    self.save()
                    self.bridge.update(save_now=0)
                if sample_now:
                    self.validate()
                    self.bridge.update(sample_now=0)
                warmup = int(self.train.get("warmup_steps", min(1000, steps//10)))
                cosine = (1+math.cos(math.pi*max(0, (self.step-warmup)/max(1, steps-warmup))))/2
                lr = float(self.train.get("lr", 4e-4))*max(0.1, cosine)*min(1, (self.step+1)/max(1, warmup))
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                difficulty = (min(1.0, max(0.0, (self.step-self.corruption_start)/max(1, steps*0.3)))
                              if self.corruption_start is not None else 0.0)
                paired = difficulty > 0
                size = int(resolutions[self.step % len(resolutions)])
                max_batch = int(self.train.get("batch_size", 4))
                budget = float(self.train.get("megapixels_per_batch", 1.0))*1e6
                batch_size = max(1, min(max_batch, int(budget/(size*size*(2 if paired else 1)))))
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                values, loss_value = {}, 0.0
                for micro in range(accumulation):
                    batch = make_batch(self.split["train"], size, batch_size,
                                       self.seed+self.step*1009+micro, self.config.get("augmentation", {}),
                                       difficulty=difficulty, paired=paired,
                                       analytic_percent=self.train.get("analytic_percent", 20),
                                       detail_percent=self.train.get("detail_percent", 70))
                    rgb, target = batch["rgb"].to(self.device), batch["alpha"].to(self.device)
                    with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.amp):
                        output = self.model(rgb)
                        partner = self.model(batch["partner"].to(self.device)) if paired else None
                    loss, pieces = objective(output, target, partner, self.config.get("loss"))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite alpha loss at step {self.step}")
                    self.scaler.scale(loss/accumulation).backward()
                    loss_value += float(loss.detach())/accumulation
                    for key, value in pieces.items():
                        values[key] = values.get(key, 0)+value/accumulation
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train.get("max_grad_norm", 1.0))
                if not torch.isfinite(norm):
                    if self.scaler.is_enabled() and self.scaler.get_scale() > 1e-6:
                        # unscale_ recorded the overflow. Reduce the FP16
                        # scale and retry this same step with fresh gradients.
                        self.scaler.update()
                        print(f"FP16 overflow: retrying step {self.step} at scale {self.scaler.get_scale()}", flush=True)
                        continue
                    raise FloatingPointError("Non-finite gradients; no checkpoint was overwritten.")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1
                decay = min(0.995, (1+self.step)/(10+self.step))
                with torch.no_grad():
                    for ema, current in zip(self.ema.parameters(), self.model.parameters()):
                        ema.lerp_(current, 1-decay)
                    for ema, current in zip(self.ema.buffers(), self.model.buffers()):
                        ema.copy_(current)
                duration = time.perf_counter()-started
                phase = "clean alpha gate pending" if self.corruption_start is None else f"RGB corruption {difficulty:.0%}"
                info = f"Alpha-only: loss {loss_value:.4f}; {size}px x {batch_size}; {phase}"
                self.bridge.update(step=self.step, info=info, speed_string=f"{duration:.2f} s/step")
                row = {"step": self.step, "loss": loss_value, "lr": lr, "seconds": duration, **values}
                with (self.root/"training_history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row)+"\n")
                if self.step == 1 or self.step % 10 == 0:
                    print(f"{self.step}/{steps}: {info}; {duration:.2f}s", flush=True)
                if self.step % int(self.config.get("validation", {}).get("every", 250)) == 0 or self.step == steps:
                    self.validate()
                if self.step % int(self.config.get("save", {}).get("every", 500)) == 0 or self.step == steps:
                    self.save()
            phase = "clean phase only; curriculum gate not passed" if self.corruption_start is None else "inspect alpha validation before use"
            self.bridge.update(status="completed", info=f"CleanMatte training complete; {phase}")
        except Exception as exc:
            self.bridge.update(status="error", info=f"CleanMatte: {exc}")
            raise
