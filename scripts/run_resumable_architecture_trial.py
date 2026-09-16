#!/usr/bin/env python
"""
Run ONE architecture experiment with crash-safe periodic checkpoints.

Normally launched by thesis_architecture_suite.py, not by hand.

Resume behaviour:
- A rolling checkpoint is written atomically to checkpoints/resume_latest.ckpt.
- If the process or PC stops, rerunning this script for the same run directory
  resumes model, optimizer, scheduler, epoch, and global step from that file.
- Metrics from each attempt are stored separately and merged into
  analysis_metrics.csv.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

import grok


ACCURACY_THRESHOLD = 99.0


def atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def global_parameter_l2(model) -> float:
    total = None
    with torch.no_grad():
        for p in model.parameters():
            part = torch.sum(p.detach().float() ** 2)
            total = part if total is None else total + part
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu())


def checkpoint_global_step(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        checkpoint = torch.load(str(path), map_location="cpu")
        return int(checkpoint.get("global_step", 0))
    except Exception:
        return 0


def truncate_eval_history_to_step(path: Path, max_step: int) -> None:
    """Discard metrics from an abandoned post-checkpoint branch before resume."""
    if not path.exists():
        return
    try:
        df = pd.read_csv(path)
        if "step" not in df.columns:
            return
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df.dropna(subset=["step"])
        df = df[df["step"] <= int(max_step)].copy()
        df["step"] = df["step"].astype(int)
        df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step")
        df.to_csv(path, index=False)
    except Exception as exc:
        print(f"WARNING: could not truncate evaluation history: {exc}")


def discover_attempt_metrics(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "attempts").glob("version_*/metrics.csv"))


def merge_attempt_metrics(run_dir: Path) -> Optional[Path]:
    """
    Merge Lightning CSVs across attempts while respecting resume branch points.

    If attempt N resumes from step R, rows >R from all older attempts belong to
    the abandoned branch and are removed before attempt N is added.
    """
    files = discover_attempt_metrics(run_dir)
    if not files:
        return None

    combined = pd.DataFrame()

    for attempt_order, path in enumerate(files):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "step" not in df.columns:
            continue

        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df.dropna(subset=["step"]).copy()
        df["step"] = df["step"].astype(int)

        meta_path = path.parent / "attempt_meta.json"
        resume_step = 0
        if meta_path.exists():
            try:
                resume_step = int(json.loads(meta_path.read_text()).get("resume_step", 0))
            except Exception:
                resume_step = 0

        if not combined.empty and attempt_order > 0:
            combined = combined[combined["step"] <= resume_step].copy()

        df["_attempt_order"] = attempt_order
        df["_row_order"] = np.arange(len(df))
        combined = pd.concat([combined, df], ignore_index=True, sort=False)

    if combined.empty:
        return None

    combined = combined.sort_values(["step", "_attempt_order", "_row_order"])
    data_cols = [c for c in combined.columns if not c.startswith("_") and c != "step"]
    rows = []
    for step, group in combined.groupby("step", sort=True):
        out = {"step": int(step)}
        for col in data_cols:
            vals = group[col].dropna()
            if len(vals):
                out[col] = vals.iloc[-1]
        rows.append(out)

    merged = pd.DataFrame(rows)
    output = run_dir / "analysis_metrics.csv"
    merged.to_csv(output, index=False)
    return output


def load_eval_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "step" in df.columns:
            df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step")
        return df
    except Exception:
        return pd.DataFrame()


def append_eval_history(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def success_row(row, threshold=ACCURACY_THRESHOLD) -> bool:
    try:
        return float(row["full_train_acc"]) >= threshold and float(row["val_accuracy"]) >= threshold
    except Exception:
        return False


def first_step_where(df: pd.DataFrame, predicate) -> Optional[int]:
    if df.empty:
        return None
    for _, row in df.sort_values("step").iterrows():
        if predicate(row):
            return int(row["step"])
    return None


def first_success_streak(df: pd.DataFrame, required: int) -> Optional[int]:
    streak = 0
    start = None
    for _, row in df.sort_values("step").iterrows():
        if success_row(row):
            if streak == 0:
                start = int(row["step"])
            streak += 1
            if streak >= required:
                return start
        else:
            streak = 0
            start = None
    return None


def last_success_streak(df: pd.DataFrame, required: int) -> bool:
    if len(df) < required:
        return False
    tail = df.sort_values("step").tail(required)
    return all(success_row(row) for _, row in tail.iterrows())


def norm_near_step(df: pd.DataFrame, step: Optional[int]) -> Optional[float]:
    if step is None or df.empty or "global_param_l2" not in df.columns:
        return None
    valid = df.dropna(subset=["global_param_l2"]).copy()
    if valid.empty:
        return None
    idx = (valid["step"] - step).abs().idxmin()
    return float(valid.loc[idx, "global_param_l2"])


class ResumableResearchCallback(Callback):
    def __init__(
        self,
        run_dir: Path,
        checkpoint_every_steps: int,
        candidate_success_streak: int,
        confirmation_success_streak: int,
        post_grok_confirmation_steps: int,
        collapse_threshold_pct: float,
    ):
        super().__init__()
        self.run_dir = run_dir
        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_ckpt = self.checkpoint_dir / "resume_latest.ckpt"
        self.checkpoint_state_path = self.checkpoint_dir / "checkpoint_state.json"
        self.eval_path = run_dir / "evaluation_history.csv"
        self.progress_path = run_dir / "progress_state.json"

        self.checkpoint_every_steps = int(checkpoint_every_steps)
        self.candidate_success_streak = int(candidate_success_streak)
        self.confirmation_success_streak = int(confirmation_success_streak)
        self.post_grok_confirmation_steps = int(post_grok_confirmation_steps)
        self.collapse_threshold_pct = float(collapse_threshold_pct)

        self.last_checkpoint_step = 0
        if self.checkpoint_state_path.exists():
            try:
                self.last_checkpoint_step = int(json.loads(self.checkpoint_state_path.read_text())["step"])
            except Exception:
                pass

        self.active_candidate_step = None
        self.confirmation_due_step = None
        self.confirmed_step = None
        self.confirmed_at_step = None
        self.collapse_events = 0

        if self.progress_path.exists():
            try:
                state = json.loads(self.progress_path.read_text())
                self.active_candidate_step = state.get("active_candidate_step")
                self.confirmation_due_step = state.get("confirmation_due_step")
                self.confirmed_step = state.get("confirmed_step")
                self.confirmed_at_step = state.get("confirmed_at_step")
                self.collapse_events = int(state.get("collapse_events", 0))
            except Exception:
                pass

    def reconcile_with_resume_step(self, resume_step: int) -> None:
        """Remove state that was written after the checkpoint we are resuming."""
        if not self.progress_path.exists():
            return
        try:
            state = json.loads(self.progress_path.read_text())
            state_step = int(state.get("global_step", 0))
        except Exception:
            return
        if state_step <= resume_step:
            return

        # A confirmed state is only trusted if its confirmation itself is no
        # later than the checkpoint. Active-candidate state is reconstructed
        # from the truncated evaluation history instead.
        confirmed_at = state.get("confirmed_at_step")
        if confirmed_at is None or int(confirmed_at) > resume_step:
            self.confirmed_step = None
            self.confirmed_at_step = None
        self.active_candidate_step = None
        self.confirmation_due_step = None

    def save_progress(self, trainer) -> None:
        atomic_json(
            self.progress_path,
            {
                "global_step": int(trainer.global_step),
                "active_candidate_step": self.active_candidate_step,
                "confirmation_due_step": self.confirmation_due_step,
                "confirmed_step": self.confirmed_step,
                "confirmed_at_step": self.confirmed_at_step,
                "collapse_events": self.collapse_events,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def save_checkpoint_atomic(self, trainer, reason: str) -> None:
        step = int(trainer.global_step)
        tmp = self.checkpoint_dir / "resume_latest.tmp.ckpt"
        try:
            trainer.save_checkpoint(str(tmp))
            os.replace(str(tmp), str(self.latest_ckpt))
            self.last_checkpoint_step = step
            atomic_json(
                self.checkpoint_state_path,
                {
                    "step": step,
                    "reason": reason,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        step = int(trainer.global_step)
        if step > 0 and step - self.last_checkpoint_step >= self.checkpoint_every_steps:
            self.save_checkpoint_atomic(trainer, "periodic")
            self.save_progress(trainer)
            print(f"[CHECKPOINT] step {step:,} -> {self.latest_ckpt}")

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_acc = scalar(metrics.get("full_train_acc"))
        val_acc = scalar(metrics.get("val_accuracy"))

        # Ignore sanity-check / incomplete validation callbacks.
        if train_acc is None or val_acc is None:
            return

        step = int(trainer.global_step)
        row = {
            "step": step,
            "epoch": int(trainer.current_epoch),
            "full_train_acc": train_acc,
            "val_accuracy": val_acc,
            "full_train_loss": scalar(metrics.get("full_train_loss")),
            "val_loss": scalar(metrics.get("val_loss")),
            "global_param_l2": global_parameter_l2(pl_module),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        existing = load_eval_history(self.eval_path)
        if existing.empty or step not in set(existing.get("step", [])):
            append_eval_history(self.eval_path, row)

        history = load_eval_history(self.eval_path)

        # Candidate: consecutive evaluations with both accuracies >=99%.
        if self.active_candidate_step is None:
            if last_success_streak(history, self.candidate_success_streak):
                tail = history.sort_values("step").tail(self.candidate_success_streak)
                self.active_candidate_step = int(tail.iloc[0]["step"])
                self.confirmation_due_step = self.active_candidate_step + self.post_grok_confirmation_steps
                print(
                    f"\n[CANDIDATE] joint >=99% begins at step {self.active_candidate_step:,}. "
                    f"Will confirm after at least {self.post_grok_confirmation_steps:,} more steps."
                )

        elif self.confirmation_due_step is not None and step >= self.confirmation_due_step:
            if last_success_streak(history, self.confirmation_success_streak):
                self.confirmed_step = int(self.active_candidate_step)
                self.confirmed_at_step = step
                print(
                    f"\n[CONFIRMED] stable grokking from step {self.confirmed_step:,}; "
                    f"confirmation at step {step:,}. Stopping cleanly."
                )
                self.save_checkpoint_atomic(trainer, "confirmed_grokking")
                self.save_progress(trainer)
                trainer.should_stop = True
                return
            else:
                self.collapse_events += 1
                print(
                    f"\n[COLLAPSE/UNSTABLE] Candidate at step {self.active_candidate_step:,} "
                    "did not survive the confirmation window. Continuing training."
                )
                self.active_candidate_step = None
                self.confirmation_due_step = None

        self.save_progress(trainer)

    def on_keyboard_interrupt(self, trainer, pl_module):
        print("\n[INTERRUPT] Saving resumable checkpoint...")
        self.save_checkpoint_atomic(trainer, "keyboard_interrupt")
        self.save_progress(trainer)

    def on_train_end(self, trainer, pl_module):
        self.save_checkpoint_atomic(trainer, "train_end")
        self.save_progress(trainer)


def make_hparams(config: Dict, run_dir: Path):
    parser = grok.training.add_args()
    hparams = parser.parse_args([])

    for key, value in config.items():
        if hasattr(hparams, key):
            setattr(hparams, key, value)

    hparams.logdir = str(run_dir.resolve())
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.checkpoint_path = str((run_dir / "checkpoints").resolve())
    hparams.d_key = hparams.d_model / hparams.n_heads
    return hparams


def save_model_info(model, config: Dict, run_dir: Path) -> None:
    total = count_parameters(model)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    atomic_json(
        run_dir / "model_info.json",
        {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "n_layers": config["n_layers"],
            "n_heads": config["n_heads"],
            "d_model": config["d_model"],
            "dropout": config["dropout"],
            "non_linearity": config["non_linearity"],
        },
    )


def calculate_result(run_dir: Path, config: Dict, model_parameter_count: int, callback: ResumableResearchCallback, final_step: int) -> Dict:
    history = load_eval_history(run_dir / "evaluation_history.csv")

    first_train99 = first_step_where(
        history,
        lambda r: float(r.get("full_train_acc", -1)) >= ACCURACY_THRESHOLD,
    )
    first_val99 = first_step_where(
        history,
        lambda r: float(r.get("val_accuracy", -1)) >= ACCURACY_THRESHOLD,
    )
    first_joint99 = first_success_streak(history, config["candidate_success_streak"])

    confirmed_step = callback.confirmed_step
    if confirmed_step is None:
        # Recover a previously confirmed value if process was resumed after it was written.
        if callback.progress_path.exists():
            try:
                confirmed_step = json.loads(callback.progress_path.read_text()).get("confirmed_step")
            except Exception:
                pass

    first_delay = None
    confirmed_delay = None
    if first_train99 is not None and first_val99 is not None:
        first_delay = first_val99 - first_train99
    if first_train99 is not None and confirmed_step is not None:
        confirmed_delay = int(confirmed_step) - first_train99

    final_train = float(history.iloc[-1]["full_train_acc"]) if not history.empty else None
    final_val = float(history.iloc[-1]["val_accuracy"]) if not history.empty else None
    peak_val = float(history["val_accuracy"].max()) if not history.empty else None

    collapsed_after_success = False
    if first_joint99 is not None and not history.empty:
        later = history[history["step"] > first_joint99]
        if not later.empty:
            bad = (later["full_train_acc"] < config["collapse_threshold_pct"]) | (
                later["val_accuracy"] < config["collapse_threshold_pct"]
            )
            collapsed_after_success = bool(bad.any())

    norm_mem = norm_near_step(history, first_train99)
    norm_grok = norm_near_step(history, int(confirmed_step) if confirmed_step is not None else None)
    norm_log_ratio = None
    if norm_mem and norm_grok and norm_mem > 0 and norm_grok > 0:
        norm_log_ratio = float(math.log((norm_mem ** 2) / (norm_grok ** 2)))

    if confirmed_step is not None:
        status = "CONFIRMED_STABLE_GROKKING"
    elif first_val99 is not None:
        status = "TRANSIENT_OR_UNCONFIRMED_GENERALISATION"
    elif first_train99 is not None:
        status = "MEMORISED_NOT_GENERALISED"
    else:
        status = "NO_MEMORISATION_THRESHOLD"

    complete = bool(confirmed_step is not None or final_step >= int(config["max_steps"]))

    return {
        "complete": complete,
        "status": status,
        "run_name": config["run_name"],
        "phases": config.get("phases", []),
        "seed": config["random_seed"],
        "math_operator": config["math_operator"],
        "train_data_pct": config["train_data_pct"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "d_model": config["d_model"],
        "dropout": config["dropout"],
        "non_linearity": config["non_linearity"],
        "weight_decay": config["weight_decay"],
        "max_lr": config["max_lr"],
        "max_steps": config["max_steps"],
        "last_step": int(final_step),
        "parameter_count": int(model_parameter_count),
        "first_train99_step": first_train99,
        "first_val99_step": first_val99,
        "first_joint99_step": first_joint99,
        "confirmed_stable99_step": confirmed_step,
        "first_crossing_delay_steps": first_delay,
        "confirmed_grokking_delay_steps": confirmed_delay,
        "peak_val_accuracy": peak_val,
        "final_train_accuracy": final_train,
        "final_val_accuracy": final_val,
        "collapsed_after_success": collapsed_after_success,
        "collapse_events": callback.collapse_events,
        "norm_at_train99": norm_mem,
        "norm_at_confirmed99": norm_grok,
        "norm_log_squared_ratio": norm_log_ratio,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing trial config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    done_path = run_dir / "trial_result.json"
    if done_path.exists():
        try:
            previous = json.loads(done_path.read_text())
            if previous.get("complete"):
                print(f"Already complete: {config['run_name']}")
                return
        except Exception:
            pass

    # Reproducibility.
    seed = int(config["random_seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hparams = make_hparams(config, run_dir)
    assert hparams.d_model % hparams.n_heads == 0, "d_model must be divisible by n_heads"

    model = grok.training.TrainableTransformer(hparams).float()
    parameter_count = count_parameters(model)
    save_model_info(model, config, run_dir)

    attempts_root = run_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing_versions = []
    for p in attempts_root.glob("version_*"):
        try:
            existing_versions.append(int(p.name.split("_")[-1]))
        except Exception:
            pass
    attempt_version = (max(existing_versions) + 1) if existing_versions else 0
    logger = CSVLogger(save_dir=str(run_dir), name="attempts", version=attempt_version)

    latest_ckpt = run_dir / "checkpoints" / "resume_latest.ckpt"
    resume_step = checkpoint_global_step(latest_ckpt) if latest_ckpt.exists() else 0
    resume = str(latest_ckpt) if latest_ckpt.exists() else None

    # Critical branch-safety rule: if the previous process logged beyond its
    # last durable checkpoint, those later rows are from an abandoned branch.
    if resume is not None:
        truncate_eval_history_to_step(run_dir / "evaluation_history.csv", resume_step)

    callback = ResumableResearchCallback(
        run_dir=run_dir,
        checkpoint_every_steps=config["checkpoint_every_steps"],
        candidate_success_streak=config["candidate_success_streak"],
        confirmation_success_streak=config["confirmation_success_streak"],
        post_grok_confirmation_steps=config["post_grok_confirmation_steps"],
        collapse_threshold_pct=config["collapse_threshold_pct"],
    )
    callback.reconcile_with_resume_step(resume_step)

    attempt_dir = Path(logger.log_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        attempt_dir / "attempt_meta.json",
        {
            "attempt_version": attempt_version,
            "resume_step": int(resume_step),
            "resume_checkpoint": resume,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    trainer_args = {
        "max_steps": int(config["max_steps"]),
        "min_steps": 0,
        "max_epochs": int(1e8),
        "val_check_interval": 1,
        "profiler": False,
        "logger": logger,
        "log_every_n_steps": 1,
        "flush_logs_every_n_steps": 250,
        "callbacks": [callback],
        "checkpoint_callback": False,
        "resume_from_checkpoint": resume,
        "terminate_on_nan": True,
        "progress_bar_refresh_rate": 0,
    }
    if torch.cuda.is_available() and int(config["gpu"]) >= 0:
        trainer_args["gpus"] = [int(config["gpu"])]

    print("=" * 100)
    print(f"RUN: {config['run_name']}")
    print(
        f"layers={config['n_layers']} heads={config['n_heads']} d_model={config['d_model']} "
        f"wd={config['weight_decay']} activation={config['non_linearity']} seed={seed}"
    )
    print(f"parameters={parameter_count:,} max_steps={config['max_steps']:,}")
    resume_text = f"yes at step {resume_step:,}: {resume}" if resume else "no"
    print(f"resume={resume_text}")
    print("=" * 100)

    trainer = Trainer(**trainer_args)
    start = time.time()
    try:
        trainer.fit(model=model)
    finally:
        # Merge whatever metrics exist even after a clean Python exception.
        merge_attempt_metrics(run_dir)

    final_step = int(trainer.global_step)
    result = calculate_result(run_dir, config, parameter_count, callback, final_step)
    result["runtime_this_attempt_seconds"] = round(time.time() - start, 2)
    atomic_json(done_path, result)
    merge_attempt_metrics(run_dir)

    print("\nRESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
