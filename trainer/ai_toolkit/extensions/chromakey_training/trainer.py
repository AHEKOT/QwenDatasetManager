"""Source-disjoint, staged training for compact alpha/foreground KeyMatte V4."""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from jobs.process.BaseTrainProcess import BaseTrainProcess
from .data import AspectBucketBatchSampler, collect_rgba_files
from .data_v4 import ChromaKeyDatasetV4, CurriculumBatchSampler, split_sources, initialize_worker
from .objectives import matting_loss, matting_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[4]
NODE_ROOT = PROJECT_ROOT / "ComfyUI-QDM-ChromaKey"
if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

from qdm_chromakey_core.v4 import ARCHITECTURE_ID_V4, KeyMatteV4  # noqa: E402
from qdm_chromakey_core.inference import run_tiled  # noqa: E402


class QDMChromaKeyTrainProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.device = torch.device(self.get_conf("device", "cuda"))
        self.sqlite_db_path = self.get_conf("sqlite_db_path", None)
        self.job_id = os.environ.get("AITK_JOB_ID")
        train = self.get_conf("train", {})
        self.steps = int(train.get("steps", 50000))
        self.step_num = int(train.get("start_step", 0))
        self.max_batch_size = int(train.get("batch_size", 8))
        self.gradient_accumulation = int(train.get("gradient_accumulation", 2))
        self.resolutions = train.get("resolutions", [384, 512, 768])
        self.max_long_side = int(train.get("max_long_side", 1280))
        self.megapixels_per_batch = float(train.get("megapixels_per_batch", 2.0))
        self.num_workers = int(train.get("num_workers", 0 if os.name == "nt" else 4))
        self.learning_rate = float(train.get("lr", 3e-4))
        self.weight_decay = float(train.get("weight_decay", 1e-4))
        self.max_grad_norm = float(train.get("max_grad_norm", 1.0))
        self.dtype_name = str(train.get("dtype", "bf16"))
        self.seed = int(train.get("seed", 42))
        self.use_ema = bool(train.get("use_ema", True))
        self.ema_decay = float(train.get("ema_decay", 0.995))
        self.boundary_width = max(1, int(train.get("boundary_width", 5)))
        save = self.get_conf("save", {})
        self.save_every = int(save.get("every", 500))
        self.max_saves = int(save.get("max_to_keep", 4))
        validation = self.get_conf("validation", {})
        self.validate_every = int(validation.get("every", 250))
        self.validation_paths = [Path(value) for value in validation.get("images", [])]
        self.validation_max_side = int(validation.get("max_side", 1536))
        self.augmentation = self.get_conf("augmentation", {})
        self.loss_weights = self.get_conf("loss", {})
        self.last_time = time.monotonic()
        self.ema_state = None
        self.ema_updates = 0
        self.warmup_steps = min(int(train.get("warmup_steps", 1000)), max(0, self.steps // 10))
        self.data_epoch = 0
        self.batch_cursor = 0
        self.best_score = float("inf")
        self.validation_files = []
        self.scaler = None

    def _db_update(self, **values):
        if not self.sqlite_db_path or not self.job_id or not Path(self.sqlite_db_path).is_file():
            return
        allowed = {"status", "step", "info", "speed_string", "save_now", "sample_now", "pid"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f'"{key}" = ?' for key in values)
        try:
            with sqlite3.connect(self.sqlite_db_path, timeout=30.0) as connection:
                connection.execute(
                    f'UPDATE "Job" SET {assignments}, updated_at = datetime("now") WHERE id = ?',
                    (*values.values(), self.job_id),
                )
        except sqlite3.Error as exc:
            self.print(f"Could not update trainer database: {exc}")

    def _db_flag(self, column, consume=False):
        if column not in {"stop", "save_now", "sample_now"}:
            return False
        if not self.sqlite_db_path or not self.job_id or not Path(self.sqlite_db_path).is_file():
            return False
        try:
            with sqlite3.connect(self.sqlite_db_path, timeout=30.0) as connection:
                row = connection.execute(f'SELECT "{column}" FROM "Job" WHERE id = ?', (self.job_id,)).fetchone()
                enabled = bool(row and row[0])
                if enabled and consume:
                    connection.execute(f'UPDATE "Job" SET "{column}" = 0 WHERE id = ?', (self.job_id,))
                return enabled
        except sqlite3.Error:
            return False

    def on_error(self, error):
        self._db_update(status="error", info=f"ChromaKey training failed: {error}")

    def _autocast_dtype(self):
        if self.dtype_name == "fp16":
            return torch.float16
        if self.dtype_name == "bf16":
            return torch.bfloat16
        return None

    def _loss(self, prediction, batch):
        # Disable autocast for reductions and RGB/alpha objectives.
        with torch.autocast(self.device.type, enabled=False):
            return matting_loss(prediction, batch, self.loss_weights, self.boundary_width)

    def _update_ema(self, model):
        if not self.use_ema:
            return
        with torch.no_grad():
            if self.ema_state is None:
                self.ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                self.ema_updates = 1
            else:
                self.ema_updates += 1
                warm_decay = (1.0 + self.ema_updates) / (10.0 + self.ema_updates)
                decay = min(self.ema_decay, warm_decay)
                for key, value in model.state_dict().items():
                    if torch.is_floating_point(value):
                        self.ema_state[key].lerp_(value.detach(), 1.0 - decay)
                    else:
                        self.ema_state[key].copy_(value)

    def _export_state(self, model):
        state = self.ema_state if self.ema_state is not None else model.state_dict()
        return {
            key: value.detach().to(device="cpu", dtype=torch.float16).contiguous()
            if torch.is_floating_point(value) else value.detach().cpu().contiguous()
            for key, value in state.items()
        }

    def _lr_factor(self, step):
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        fraction = min(1., max(0., (step-self.warmup_steps) / max(1, self.steps-self.warmup_steps)))
        return .05 + .95 * .5 * (1 + math.cos(math.pi*fraction))

    def _restore_latest(self, model, optimizer, scheduler):
        states = sorted(Path(self.save_root).glob("trainer_state_v4_step_*.pt"))
        if not states:
            return
        checkpoint = torch.load(states[-1], map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != ARCHITECTURE_ID_V4:
            raise ValueError("V4 training cannot resume an older architecture")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        self.step_num = int(checkpoint["step"])
        # Recalculate against the requested endpoint if training was extended.
        for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base*self._lr_factor(self.step_num)
        self.ema_state = checkpoint.get("ema")
        if self.ema_state is not None:
            self.ema_state = {key: value.to(self.device) for key, value in self.ema_state.items()}
        self.ema_updates = checkpoint.get("ema_updates", self.step_num)
        self.data_epoch = checkpoint.get("data_epoch", 0)
        self.batch_cursor = checkpoint.get("batch_cursor", 0)
        self.best_score = checkpoint.get("best_score", float("inf"))
        if self.scaler is not None and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.print(f"Resumed V4 at step {self.step_num}, epoch {self.data_epoch}, batch {self.batch_cursor}")

    def _metadata(self, step):
        return {"architecture": ARCHITECTURE_ID_V4, "step": str(step),
                "input": "RGB_0_1_BCHW", "output": "straight_RGBA_and_alpha",
                "stride": "1", "context_size": "256", "default_tile_size": "512",
                "default_overlap": "0", "colour_space": "sRGB"}

    def save(self, model, optimizer, scheduler, step):
        root = Path(self.save_root)
        root.mkdir(parents=True, exist_ok=True)
        weights_path = root / f"{self.job.name}_v4_step_{step:09d}.safetensors"
        temp_path = weights_path.with_suffix(".safetensors.tmp")
        save_file(self._export_state(model), str(temp_path), metadata=self._metadata(step))
        temp_path.replace(weights_path)
        state_path = root / f"trainer_state_v4_step_{step:09d}.pt"
        temp_state = state_path.with_suffix(".pt.tmp")
        torch.save({"architecture": ARCHITECTURE_ID_V4, "step": step,
                    "model": model.state_dict(), "ema": self.ema_state, "ema_updates": self.ema_updates,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": self.scaler.state_dict() if self.scaler else {}, "total_steps": self.steps,
                    "data_epoch": self.data_epoch, "batch_cursor": self.batch_cursor,
                    "best_score": self.best_score}, temp_state)
        temp_state.replace(state_path)
        checkpoints = sorted(root.glob(f"{self.job.name}_v4_step_*.safetensors"))
        for old in checkpoints[:-max(1, self.max_saves)]:
            old.unlink(missing_ok=True)
            old.with_name(old.name.replace(f"{self.job.name}_v4", "trainer_state_v4").replace(".safetensors", ".pt")).unlink(missing_ok=True)
        final_path = root / f"{self.job.name}.safetensors"
        shutil.copy2(weights_path, final_path)
        best_path = root / f"{self.job.name}_best.safetensors"
        (root / "artifacts.json").write_text(json.dumps({
            "architecture": ARCHITECTURE_ID_V4, "step": step,
            "primary": best_path.name if best_path.exists() else final_path.name,
            "latest": final_path.name, "checkpoint": weights_path.name,
            "trainer_state": state_path.name,
        }, indent=2), encoding="utf-8")
        self.print(f"Saved V4 model to {weights_path}")

    def _validate(self, model, step):
        preview = KeyMatteV4().to(self.device)
        preview.load_state_dict(self.ema_state if self.ema_state is not None else model.state_dict())
        preview.eval()
        rows = []
        # Fixed seeds, fixed sources and all four screen families. No RNG or
        # generated crop is shared with optimization.
        for family in ("green", "blue", "white", "black"):
            augmentation = dict(self.augmentation)
            augmentation.update({f"{name}_chance": 100 if name == family else 0
                                 for name in ("green", "blue", "white", "black")})
            augmentation.update(collision_chance=0, detail_crop_percent=50)
            dataset = ChromaKeyDatasetV4(self.validation_files[:8], augmentation, flip_probability=0)
            for index in range(len(dataset)):
                sample = dataset[(index, 512, 512, 900_000+index, 1.)]
                # Include native crops but retain the matching global context,
                # exactly as the deployment tiler does.
                batch = {name: value[None].to(self.device) for name, value in sample.items() if torch.is_tensor(value)}
                with torch.inference_mode(), torch.autocast("cuda", dtype=self._autocast_dtype(), enabled=self._autocast_dtype() is not None):
                    prediction = preview(batch["input"], batch["context_rgb"], batch["context_box"])
                values = matting_metrics(prediction["foreground"], prediction["alpha"], batch)
                if not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError("Nonfinite validation output")
                rows.append({"family": family, "source": str(self.validation_files[index]), **values})
        means = {name: sum(row[name] for row in rows)/len(rows)
                 for name in rows[0] if name not in {"family", "source"}}
        report = {"step": step, "metrics": means, "cases": rows}
        root = Path(self.save_root)
        with (root / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, allow_nan=False)+"\n")
        if means["selection_score"] < self.best_score:
            self.best_score = means["selection_score"]
            best = root / f"{self.job.name}_best.safetensors"
            temporary = best.with_suffix(".safetensors.tmp")
            save_file(self._export_state(model), str(temporary), metadata=self._metadata(step))
            temporary.replace(best)
            (root / "best_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if self.writer:
            for name, value in means.items():
                self.writer.add_scalar(f"validation/{name}", value, step)
        self.print(f"V4 validation step {step}: " + " ".join(f"{k}={v:.5f}" for k, v in means.items()))
        del preview
        self._validation_preview(model, step)

    def _validation_preview(self, model, step):
        if not self.validation_paths:
            self.print(f"ChromaKey validation skipped at step {step}: no validation images configured")
            return
        samples_dir = Path(self.save_root) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        self.print(
            f"Running ChromaKey validation at step {step} "
            f"for {len(self.validation_paths)} image(s)..."
        )
        self._db_update(info=f"Validating {len(self.validation_paths)} ChromaKey image(s) at step {step}")
        was_training = model.training
        preview_model = model
        if self.ema_state is not None:
            preview_model = KeyMatteV4().to(self.device)
            preview_model.load_state_dict(self.ema_state)
        preview_model.eval()
        for index, path in enumerate(self.validation_paths):
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
            scale = min(1.0, self.validation_max_side / max(rgb_image.size))
            if scale < 1:
                rgb_image = rgb_image.resize(
                    (max(1, round(rgb_image.width * scale)), max(1, round(rgb_image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            array = np.asarray(rgb_image, dtype=np.float32) / 255.0
            rgb = torch.from_numpy(array.copy()).permute(2, 0, 1)
            clean, alpha = run_tiled(
                preview_model, rgb, tile_size=min(1024, self.validation_max_side), overlap=96,
                autocast_dtype=self._autocast_dtype(),
            )
            rgba = torch.cat((clean, alpha), dim=0).permute(1, 2, 0).cpu().numpy()
            rgba_u8 = np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)
            output = Image.fromarray(rgba_u8, "RGBA")
            output.save(samples_dir / f"chroma_{step:09d}_{index}.png")
            checker = Image.new("RGB", rgb_image.size, (205, 205, 205))
            draw = ImageDraw.Draw(checker)
            tile = 16
            for y in range(0, checker.height, tile):
                for x in range(0, checker.width, tile):
                    if (x // tile + y // tile) % 2:
                        draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(245, 245, 245))
            checker.paste(output, mask=output.getchannel("A"))
            alpha_rgb = output.getchannel("A").convert("RGB")
            sheet = Image.new("RGB", (rgb_image.width * 3, rgb_image.height), (0, 0, 0))
            sheet.paste(rgb_image, (0, 0))
            sheet.paste(checker, (rgb_image.width, 0))
            sheet.paste(alpha_rgb, (rgb_image.width * 2, 0))
            sheet.save(samples_dir / f"chroma_{step:09d}_{index}_sheet.jpg", quality=92)
        if was_training:
            model.train()
        self.print(
            f"ChromaKey validation step {step} complete: "
            f"saved {len(self.validation_paths) * 2} file(s) to {samples_dir}"
        )

    def run(self):
        super().run()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("QDM ChromaKey training requires CUDA")
        architecture = self.get_conf("architecture", {}).get("id", ARCHITECTURE_ID_V4)
        if architecture != ARCHITECTURE_ID_V4:
            raise RuntimeError("Create a new KeyMatte V4 job. V1–V3 weights/optimizer cannot initialize V4.")
        torch.manual_seed(self.seed)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        files = collect_rgba_files(self.get_conf("datasets", []))
        files, self.validation_files = split_sources(files, seed=self.seed)
        root = Path(self.save_root)
        root.mkdir(parents=True, exist_ok=True)
        split_path = root / "source_split_v4.json"
        manifest = {"train": [str(p) for p in files], "validation": [str(p) for p in self.validation_files]}
        if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Dataset split changed. Create a new job to keep validation independent.")
        split_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.print(f"V4 sources: {len(files)} train / {len(self.validation_files)} holdout")
        dataset = ChromaKeyDatasetV4(files, self.augmentation)
        sampler = AspectBucketBatchSampler(files, self.resolutions, self.max_batch_size,
                                          self.megapixels_per_batch, self.max_long_side, self.seed)
        model = KeyMatteV4().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_factor)
        autocast_dtype = self._autocast_dtype()
        if autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported by this GPU; select FP16 or FP32")
        self.scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)
        self._restore_latest(model, optimizer, scheduler)
        self._db_update(status="running", step=self.step_num, info="Training KeyMatte V4")
        progress = tqdm(total=self.steps, initial=self.step_num, desc=self.job.name)
        optimizer.zero_grad(set_to_none=True)
        self.last_time = time.monotonic()
        accumulated, stopped = 0, False
        accumulation_start = (self.data_epoch, self.batch_cursor)
        # Carry unfinished accumulation across epochs. Only consumed batches
        # are checkpointed, so worker prefetch cannot skip training examples.
        while self.step_num < self.steps and not stopped:
            sampler.epoch = self.data_epoch
            batch_sampler = CurriculumBatchSampler(sampler, self.batch_cursor,
                                                   self.step_num + accumulated/self.gradient_accumulation,
                                                   self.steps, self.gradient_accumulation)
            loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.num_workers,
                                pin_memory=True, worker_init_fn=initialize_worker,
                                prefetch_factor=1 if self.num_workers > 0 else None)
            for batch_index, batch in enumerate(loader, start=self.batch_cursor):
                if self._db_flag("stop"):
                    stopped = True
                    break
                model.train()
                if accumulated == 0:
                    accumulation_start = (self.data_epoch, self.batch_cursor)
                batch = {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                source = batch["input"]
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    prediction = model(source, batch["context_rgb"], batch["context_box"])
                total, losses = self._loss(prediction, batch)
                if not torch.isfinite(total):
                    raise RuntimeError(f"Nonfinite V4 loss at step {self.step_num}")
                self.scaler.scale(total / self.gradient_accumulation).backward()
                accumulated += 1
                self.batch_cursor = batch_index + 1
                if accumulated < self.gradient_accumulation:
                    continue
                self.scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm if self.max_grad_norm > 0 else float("inf"))
                if not torch.isfinite(norm) and not self.scaler.is_enabled():
                    raise RuntimeError(f"Nonfinite V4 gradients at step {self.step_num}")
                old_scale = self.scaler.get_scale()
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                if self.scaler.get_scale() < old_scale:
                    continue  # AMP skipped the optimizer: do not advance LR/EMA/step.
                scheduler.step()
                self.step_num += 1
                self._update_ema(model)
                elapsed = max(1e-6, time.monotonic()-self.last_time)
                self.last_time = time.monotonic()
                self._db_update(step=self.step_num, speed_string=f"{elapsed:.2f}s/step · {source.shape[-1]}×{source.shape[-2]}",
                                info=f"KeyMatte V4 step {self.step_num}/{self.steps}")
                progress.update(1)
                progress.set_postfix_str(f"loss={total.item():.4f} lr={optimizer.param_groups[0]['lr']:.2g}")
                if self.writer:
                    self.writer.add_scalar("loss/total", total.item(), self.step_num)
                    for name, value in losses.items():
                        self.writer.add_scalar(f"loss/{name}", value.item(), self.step_num)
                if (self.validate_every and self.step_num % self.validate_every == 0) or self._db_flag("sample_now", consume=True):
                    self._validate(model, self.step_num)
                if (self.save_every and self.step_num % self.save_every == 0) or self._db_flag("save_now", consume=True):
                    self.save(model, optimizer, scheduler, self.step_num)
                if self.step_num >= self.steps:
                    break
            else:
                self.data_epoch += 1
                self.batch_cursor = 0
        # A stop during accumulation replays precisely those microbatches,
        # including an accumulation that started in the previous epoch.
        if accumulated:
            optimizer.zero_grad(set_to_none=True)
            self.data_epoch, self.batch_cursor = accumulation_start
        progress.close()
        self._validate(model, self.step_num)
        self.save(model, optimizer, scheduler, self.step_num)
        self._db_update(status="stopped" if stopped else "completed", step=self.step_num,
                        info="KeyMatte V4 training stopped" if stopped else "KeyMatte V4 training completed")
