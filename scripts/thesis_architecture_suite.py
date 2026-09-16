#!/usr/bin/env python
"""
Master runner for the thesis architecture experiments.

Rerun the same command after any crash/reboot. Completed runs are skipped and
incomplete runs resume from their rolling checkpoints.
"""

import argparse
import ctypes
import csv
import json
import os
import subprocess
import sys
import time
import threading
import statistics
from datetime import datetime, timedelta
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "thesis_architecture_suite.json"
CHILD_SCRIPT = ROOT / "scripts" / "run_resumable_architecture_trial.py"


ETA_REFRESH_SECONDS = 60
RUNTIME_HISTORY_FILENAME = "runtime_history.json"


def fmt_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_runtime_history(suite_dir: Path) -> Dict:
    path = suite_dir / RUNTIME_HISTORY_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_runtime_history(suite_dir: Path, history: Dict):
    path = suite_dir / RUNTIME_HISTORY_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_trial_result(run_dir: Path) -> Dict:
    path = run_dir / "trial_result.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def register_attempt_runtime(
    suite_dir: Path,
    history: Dict,
    run_name: str,
    elapsed_seconds: float,
    complete: bool,
):
    entry = history.setdefault(
        run_name,
        {
            "observed_seconds": 0.0,
            "attempt_count": 0,
            "completed": False,
        },
    )
    entry["observed_seconds"] = round(
        float(entry.get("observed_seconds", 0.0)) + float(elapsed_seconds), 2
    )
    entry["attempt_count"] = int(entry.get("attempt_count", 0)) + 1
    entry["completed"] = bool(complete)
    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_runtime_history(suite_dir, history)


def completed_runtime_samples(
    suite_dir: Path,
    manifest: List[Dict],
    history: Dict,
):
    samples = []
    for config in manifest:
        run_name = config["run_name"]
        run_dir = suite_dir / run_name
        if not is_complete(run_dir):
            continue

        seconds = None
        hist = history.get(run_name, {})
        if float(hist.get("observed_seconds", 0.0) or 0.0) > 0:
            seconds = float(hist["observed_seconds"])
        else:
            result = read_trial_result(run_dir)
            fallback = result.get("runtime_this_attempt_seconds")
            if fallback is not None:
                try:
                    seconds = float(fallback)
                except Exception:
                    seconds = None

        if seconds is not None and seconds > 0:
            samples.append(
                {
                    "run_name": run_name,
                    "seconds": seconds,
                    "phases": list(config.get("phases", [])),
                }
            )
    return samples


def estimate_run_seconds(config: Dict, samples) -> float:
    if not samples:
        return None

    phases = set(config.get("phases", []))
    phase_matches = [
        s["seconds"]
        for s in samples
        if phases.intersection(set(s.get("phases", [])))
    ]

    if len(phase_matches) >= 2:
        return float(statistics.median(phase_matches))

    return float(statistics.median([s["seconds"] for s in samples]))


def selected_completion_counts(suite_dir: Path, selected):
    complete = sum(1 for _, c in selected if is_complete(suite_dir / c["run_name"]))
    return complete, len(selected)


def estimate_pending_seconds(
    suite_dir: Path,
    pending_configs: List[Dict],
    manifest: List[Dict],
    history: Dict,
):
    samples = completed_runtime_samples(suite_dir, manifest, history)
    if not samples:
        return None

    total = 0.0
    for config in pending_configs:
        estimate = estimate_run_seconds(config, samples)
        if estimate is None:
            return None
        total += estimate
    return total


def print_suite_eta(
    suite_dir: Path,
    selected,
    manifest: List[Dict],
    history: Dict,
    prefix="ETA",
):
    complete, total = selected_completion_counts(suite_dir, selected)
    pending = [c for _, c in selected if not is_complete(suite_dir / c["run_name"])]
    pct = 100.0 * complete / total if total else 100.0
    remaining = estimate_pending_seconds(suite_dir, pending, manifest, history)

    if remaining is None:
        print(
            f"[{prefix}] progress {complete}/{total} ({pct:.1f}%) | "
            "suite ETA: learning from completed runs"
        )
        return

    finish = datetime.now() + timedelta(seconds=remaining)
    print(
        f"[{prefix}] progress {complete}/{total} ({pct:.1f}%) | "
        f"estimated remaining {fmt_duration(remaining)} | "
        f"estimated finish {finish.strftime('%Y-%m-%d %I:%M %p')}"
    )


def start_eta_monitor(
    suite_dir: Path,
    selected,
    manifest: List[Dict],
    history: Dict,
    current_config: Dict,
    run_started_monotonic: float,
):
    stop_event = threading.Event()

    def monitor():
        while not stop_event.wait(ETA_REFRESH_SECONDS):
            elapsed = time.monotonic() - run_started_monotonic
            complete, total = selected_completion_counts(suite_dir, selected)
            samples = completed_runtime_samples(suite_dir, manifest, history)
            predicted_current = estimate_run_seconds(current_config, samples)

            pending_future = [
                c
                for _, c in selected
                if c["run_name"] != current_config["run_name"]
                and not is_complete(suite_dir / c["run_name"])
            ]

            future_seconds = None
            if samples:
                future_seconds = 0.0
                for c in pending_future:
                    estimate = estimate_run_seconds(c, samples)
                    if estimate is None:
                        future_seconds = None
                        break
                    future_seconds += estimate

            if predicted_current is None or future_seconds is None:
                print(
                    f"\n[ETA] current run elapsed {fmt_duration(elapsed)} | "
                    f"suite progress {complete}/{total} | "
                    "suite ETA: learning from completed runs",
                    flush=True,
                )
                continue

            # If a run is taking longer than its historical estimate, do not let
            # the displayed remaining time collapse to zero. Extend the current
            # estimate as the run continues.
            predicted_total_current = max(float(predicted_current), elapsed * 1.25)
            current_remaining = max(0.0, predicted_total_current - elapsed)
            suite_remaining = current_remaining + future_seconds
            finish = datetime.now() + timedelta(seconds=suite_remaining)

            print(
                f"\n[ETA] current run elapsed {fmt_duration(elapsed)} | "
                f"current est remaining {fmt_duration(current_remaining)} | "
                f"suite progress {complete}/{total} | "
                f"suite est remaining {fmt_duration(suite_remaining)} | "
                f"finish ~ {finish.strftime('%Y-%m-%d %I:%M %p')}",
                flush=True,
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stop_event, thread


def prevent_windows_sleep():
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def restore_windows_sleep():
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def safe_name_number(x) -> str:
    if isinstance(x, float):
        return ("%g" % x).replace("-", "m").replace(".", "p")
    return str(x)


def experiment_key(c: Dict) -> Tuple:
    return (
        c["math_operator"],
        c["train_data_pct"],
        c["max_lr"],
        c["weight_decay"],
        c["weight_decay_kind"],
        c["n_layers"],
        c["n_heads"],
        c["d_model"],
        c["dropout"],
        c["non_linearity"],
        c["random_seed"],
        c["max_steps"],
    )


def make_run_name(index: int, c: Dict) -> str:
    return (
        f"A{index:03d}_"
        f"L{c['n_layers']}_H{c['n_heads']}_D{c['d_model']}_"
        f"WD{safe_name_number(c['weight_decay'])}_"
        f"{c['non_linearity']}_seed{c['random_seed']}"
    )


def transformer_parameter_count(n_layers: int, n_heads: int, d_model: int, vocab_len: int) -> int:
    # Exact for the OpenAI grok Transformer implementation:
    # embedding + output projection = 2*vocab*d
    # per decoder block = QKV (3d^2) + Wo (d^2) + FFN (8d^2)
    #                     + two LayerNorms, each weight+bias (4d total)
    assert d_model % n_heads == 0
    return int(2 * vocab_len * d_model + n_layers * (12 * d_model * d_model + 4 * d_model))


def get_vocab_len(base: Dict) -> int:
    # Use the real repository tokenizer once so parameter matching is exact.
    import grok

    parser = grok.training.add_args()
    hp = parser.parse_args([])
    for key, value in base.items():
        if hasattr(hp, key):
            setattr(hp, key, value)
    hp.logdir = str((ROOT / "runs" / "_parameter_count_probe").resolve())
    hp.datadir = os.path.abspath(hp.datadir)
    model = grok.training.TrainableTransformer(hp).float()
    return int(len(model.train_dataset.tokenizer))


def generate_manifest(settings: Dict) -> List[Dict]:
    base = deepcopy(settings["baseline"])
    seeds = list(settings["seeds"])
    exp = settings["experiments"]

    common_runtime = {
        "checkpoint_every_steps": settings["checkpoint_every_steps"],
        "candidate_success_streak": settings["candidate_success_streak"],
        "confirmation_success_streak": settings["confirmation_success_streak"],
        "post_grok_confirmation_steps": settings["post_grok_confirmation_steps"],
        "collapse_threshold_pct": settings["collapse_threshold_pct"],
    }

    raw = []

    def add(phase: str, **overrides):
        c = deepcopy(base)
        c.update(overrides)
        c.update(common_runtime)
        c["phases"] = [phase]
        raw.append(c)

    # Baseline replicated cleanly with the same runner.
    for seed in seeds:
        add("baseline", random_seed=seed)

    # 1) Attention-head sweep at fixed depth/width.
    for heads in exp["attention_heads"]:
        for seed in seeds:
            add("attention_heads", n_heads=int(heads), random_seed=seed)

    # 2) Fixed-width depth sweep.
    for layers in exp["depth_fixed_width"]:
        for seed in seeds:
            add("depth_fixed_width", n_layers=int(layers), random_seed=seed)

    # 3) Width sweep at fixed depth.
    for width in exp["width_fixed_depth"]:
        for seed in seeds:
            add("width_fixed_depth", d_model=int(width), random_seed=seed)

    # 4) Parameter-matched depth sweep.
    vocab_len = get_vocab_len(base)
    target_params = transformer_parameter_count(
        int(base["n_layers"]), int(base["n_heads"]), int(base["d_model"]), vocab_len
    )
    candidate_widths = list(
        range(
            int(exp["parameter_match_candidate_width_min"]),
            int(exp["parameter_match_candidate_width_max"]) + 1,
            int(exp["parameter_match_width_step"]),
        )
    )
    for layers in exp["parameter_matched_depths"]:
        valid = [w for w in candidate_widths if w % int(base["n_heads"]) == 0]
        width = min(
            valid,
            key=lambda w: abs(
                transformer_parameter_count(int(layers), int(base["n_heads"]), int(w), vocab_len)
                - target_params
            ),
        )
        actual = transformer_parameter_count(int(layers), int(base["n_heads"]), int(width), vocab_len)
        for seed in seeds:
            add(
                "parameter_matched_depth",
                n_layers=int(layers),
                d_model=int(width),
                random_seed=seed,
                parameter_match_target=target_params,
                parameter_match_actual=actual,
                parameter_match_relative_error=(actual - target_params) / target_params,
            )

    # 5) Activation check (repo supports relu and gelu).
    for activation in exp["activations"]:
        for seed in seeds:
            add("activation", non_linearity=str(activation), random_seed=seed)

    # 6) Architecture x regularisation interaction using representative depths.
    for layers in exp["regularisation_depths"]:
        for wd in exp["regularisation_weight_decays"]:
            for seed in seeds:
                add(
                    "depth_x_regularisation",
                    n_layers=int(layers),
                    weight_decay=float(wd),
                    random_seed=seed,
                )

    # Deduplicate identical physical experiments but retain all phase labels.
    unique = {}
    for c in raw:
        key = experiment_key(c)
        if key not in unique:
            unique[key] = c
        else:
            phases = set(unique[key].get("phases", [])) | set(c.get("phases", []))
            unique[key]["phases"] = sorted(phases)
            for k in ["parameter_match_target", "parameter_match_actual", "parameter_match_relative_error"]:
                if k in c:
                    unique[key][k] = c[k]

    phase_order = {
        "baseline": 0,
        "attention_heads": 1,
        "depth_fixed_width": 2,
        "width_fixed_depth": 3,
        "parameter_matched_depth": 4,
        "activation": 5,
        "depth_x_regularisation": 6,
    }

    manifest = list(unique.values())
    manifest.sort(
        key=lambda c: (
            min(phase_order.get(p, 99) for p in c.get("phases", ["zzz"])),
            c["n_layers"],
            c["n_heads"],
            c["d_model"],
            c["weight_decay"],
            c["non_linearity"],
            c["random_seed"],
        )
    )

    for i, c in enumerate(manifest, start=1):
        c["run_name"] = make_run_name(i, c)

    return manifest


def save_environment(suite_dir: Path):
    envdir = suite_dir / "environment"
    envdir.mkdir(parents=True, exist_ok=True)

    def capture(name, command):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            text = result.stdout + "\n" + result.stderr
        except Exception as exc:
            text = repr(exc)
        (envdir / name).write_text(text, encoding="utf-8")

    capture("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"])
    capture("git_commit.txt", ["git", "rev-parse", "HEAD"])
    capture("git_status.txt", ["git", "status", "--short"])
    capture("nvidia_smi.txt", ["nvidia-smi"])
    (envdir / "python.txt").write_text(sys.version, encoding="utf-8")


def is_complete(run_dir: Path) -> bool:
    result = run_dir / "trial_result.json"
    if not result.exists():
        return False
    try:
        return bool(json.loads(result.read_text()).get("complete"))
    except Exception:
        return False


def update_suite_summary(suite_dir: Path, manifest: List[Dict]):
    rows = []
    for config in manifest:
        run_dir = suite_dir / config["run_name"]
        result_path = run_dir / "trial_result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                rows.append(result)
            except Exception:
                pass
    if rows:
        pd.DataFrame(rows).to_csv(suite_dir / "summary.csv", index=False)
        (suite_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--only-phase", default=None)
    ap.add_argument("--from-run", type=int, default=1, help="1-based manifest index to start from")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    suite_dir = ROOT / "runs" / settings["suite_name"]
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = suite_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Using existing manifest: {manifest_path}")
    else:
        print("Generating deterministic experiment manifest...")
        manifest = generate_manifest(settings)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pd.DataFrame(manifest).to_csv(suite_dir / "manifest.csv", index=False)
        (suite_dir / "suite_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        save_environment(suite_dir)

    selected = []
    for i, c in enumerate(manifest, start=1):
        if i < args.from_run:
            continue
        if args.only_phase and args.only_phase not in c.get("phases", []):
            continue
        selected.append((i, c))

    history = load_runtime_history(suite_dir)

    print("=" * 100)
    print("THESIS ARCHITECTURE MASTER SUITE")
    print(f"Suite: {suite_dir}")
    print(f"Manifest experiments: {len(manifest)}")
    print(f"Selected this invocation: {len(selected)}")
    print("Rerunning this command after a crash will skip completed runs and resume incomplete ones.")
    print(f"ETA refresh interval: {ETA_REFRESH_SECONDS} seconds")
    print("=" * 100)

    print_suite_eta(suite_dir, selected, manifest, history, prefix="START")

    prevent_windows_sleep()
    invocation_started = time.monotonic()

    try:
        for manifest_index, config in selected:
            run_dir = suite_dir / config["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

            if is_complete(run_dir):
                print(f"[{manifest_index}/{len(manifest)}] SKIP complete: {config['run_name']}")
                continue

            complete_before, selected_total = selected_completion_counts(suite_dir, selected)
            pct_before = 100.0 * complete_before / selected_total if selected_total else 100.0

            print("\n" + "#" * 100)
            print(f"[{manifest_index}/{len(manifest)}] {config['run_name']}")
            print(f"Phases: {', '.join(config.get('phases', []))}")
            print(f"Selected-suite progress before run: {complete_before}/{selected_total} ({pct_before:.1f}%)")

            samples = completed_runtime_samples(suite_dir, manifest, history)
            expected_run = estimate_run_seconds(config, samples)
            if expected_run is None:
                print("Expected run duration: unknown until at least one run has completed")
            else:
                print(f"Expected run duration from completed-run history: ~{fmt_duration(expected_run)}")

            print_suite_eta(suite_dir, selected, manifest, history, prefix="BEFORE RUN")
            print("#" * 100)

            retries = int(settings["max_auto_retries"])
            success = False

            for attempt in range(1, retries + 1):
                cmd = [
                    sys.executable,
                    str(CHILD_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                ]

                attempt_started = time.monotonic()
                monitor_stop, monitor_thread = start_eta_monitor(
                    suite_dir=suite_dir,
                    selected=selected,
                    manifest=manifest,
                    history=history,
                    current_config=config,
                    run_started_monotonic=attempt_started,
                )

                try:
                    result = subprocess.run(cmd, cwd=ROOT)
                finally:
                    monitor_stop.set()
                    monitor_thread.join(timeout=2)

                attempt_elapsed = time.monotonic() - attempt_started
                completed_now = is_complete(run_dir)
                register_attempt_runtime(
                    suite_dir=suite_dir,
                    history=history,
                    run_name=config["run_name"],
                    elapsed_seconds=attempt_elapsed,
                    complete=completed_now,
                )

                print(
                    f"Attempt {attempt}/{retries} wall time: "
                    f"{fmt_duration(attempt_elapsed)}"
                )

                if result.returncode == 0 and completed_now:
                    success = True
                    break

                print(
                    f"Child ended with code {result.returncode}; "
                    f"automatic retry {attempt}/{retries}. "
                    "The next attempt will resume from resume_latest.ckpt."
                )

                if attempt < retries:
                    delay = int(settings["retry_delay_seconds"])
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

            if not success:
                print(
                    f"WARNING: {config['run_name']} did not complete after {retries} attempts. "
                    "Leaving it resumable and continuing to the next configuration."
                )

            update_suite_summary(suite_dir, manifest)
            print_suite_eta(suite_dir, selected, manifest, history, prefix="UPDATED")

    except KeyboardInterrupt:
        print("\nMaster suite interrupted. Re-run the same command to continue.")
    finally:
        update_suite_summary(suite_dir, manifest)
        save_runtime_history(suite_dir, history)
        restore_windows_sleep()

    invocation_elapsed = time.monotonic() - invocation_started
    print("\n" + "=" * 100)
    print(f"Suite pass finished. This invocation ran for {fmt_duration(invocation_elapsed)}.")
    print_suite_eta(suite_dir, selected, manifest, history, prefix="FINAL")
    print("Run the same command again at any time; completed trials will be skipped.")
    print("=" * 100)


if __name__ == "__main__":
    main()
