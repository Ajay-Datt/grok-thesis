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
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "thesis_architecture_suite.json"
CHILD_SCRIPT = ROOT / "scripts" / "run_resumable_architecture_trial.py"


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

    print("=" * 100)
    print("THESIS ARCHITECTURE MASTER SUITE")
    print(f"Suite: {suite_dir}")
    print(f"Manifest experiments: {len(manifest)}")
    print(f"Selected this invocation: {len(selected)}")
    print("Rerunning this command after a crash will skip completed runs and resume incomplete ones.")
    print("=" * 100)

    prevent_windows_sleep()
    try:
        for manifest_index, config in selected:
            run_dir = suite_dir / config["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

            if is_complete(run_dir):
                print(f"[{manifest_index}/{len(manifest)}] SKIP complete: {config['run_name']}")
                continue

            print("\n" + "#" * 100)
            print(f"[{manifest_index}/{len(manifest)}] {config['run_name']}")
            print(f"Phases: {', '.join(config.get('phases', []))}")
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
                result = subprocess.run(cmd, cwd=ROOT)
                if result.returncode == 0 and is_complete(run_dir):
                    success = True
                    break

                print(
                    f"Child ended with code {result.returncode}; "
                    f"automatic retry {attempt}/{retries}. The next attempt will resume from resume_latest.ckpt."
                )
                if attempt < retries:
                    time.sleep(int(settings["retry_delay_seconds"]))

            if not success:
                print(
                    f"WARNING: {config['run_name']} did not complete after {retries} attempts. "
                    "Leaving it resumable and continuing to the next configuration."
                )

            update_suite_summary(suite_dir, manifest)

    except KeyboardInterrupt:
        print("\nMaster suite interrupted. Re-run the same command to continue.")
    finally:
        update_suite_summary(suite_dir, manifest)
        restore_windows_sleep()

    print("\nSuite pass finished. Run the same command again at any time; completed trials will be skipped.")


if __name__ == "__main__":
    main()
