#!/usr/bin/env python
"""Aggregate completed architecture runs into thesis-friendly CSV summaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "runs" / "thesis_architecture_suite_v1"


def phase_rows(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    def has_phase(value):
        if isinstance(value, list):
            return phase in value
        text = str(value)
        return phase in text
    return df[df["phases"].apply(has_phase)].copy()


def aggregate(df: pd.DataFrame, group_cols, output: Path):
    if df.empty:
        return
    metric = "confirmed_grokking_delay_steps"
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_name", "count"),
            n_confirmed=("status", lambda x: int((x == "CONFIRMED_STABLE_GROKKING").sum())),
            mean_delay=(metric, "mean"),
            std_delay=(metric, "std"),
            median_delay=(metric, "median"),
            mean_train99=("first_train99_step", "mean"),
            mean_parameter_count=("parameter_count", "mean"),
            collapse_rate=("collapsed_after_success", "mean"),
        )
        .reset_index()
    )
    grouped["confirmed_fraction"] = grouped["n_confirmed"] / grouped["n_runs"]
    grouped.to_csv(output, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?", default=str(DEFAULT_SUITE))
    args = ap.parse_args()

    suite = Path(args.suite).resolve()
    results = []
    for path in sorted(suite.glob("A*/trial_result.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

    if not results:
        raise RuntimeError(f"No trial_result.json files found under {suite}")

    outdir = suite / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(outdir / "all_completed_runs.csv", index=False)

    aggregate(phase_rows(df, "attention_heads"), ["n_heads"], outdir / "attention_heads_summary.csv")
    aggregate(phase_rows(df, "depth_fixed_width"), ["n_layers"], outdir / "depth_fixed_width_summary.csv")
    aggregate(phase_rows(df, "width_fixed_depth"), ["d_model"], outdir / "width_fixed_depth_summary.csv")
    aggregate(phase_rows(df, "parameter_matched_depth"), ["n_layers", "d_model"], outdir / "parameter_matched_depth_summary.csv")
    aggregate(phase_rows(df, "activation"), ["non_linearity"], outdir / "activation_summary.csv")
    aggregate(
        phase_rows(df, "depth_x_regularisation"),
        ["n_layers", "weight_decay"],
        outdir / "depth_x_regularisation_summary.csv",
    )

    norm = df.dropna(subset=["confirmed_grokking_delay_steps", "norm_log_squared_ratio"]).copy()
    if len(norm) >= 2:
        x = norm["norm_log_squared_ratio"].astype(float).to_numpy()
        y = norm["confirmed_grokking_delay_steps"].astype(float).to_numpy()
        pearson = float(np.corrcoef(x, y)[0, 1])
    else:
        pearson = None

    norm.to_csv(outdir / "norm_vs_delay_runs.csv", index=False)
    (outdir / "analysis_overview.json").write_text(
        json.dumps(
            {
                "completed_runs": int(len(df)),
                "confirmed_stable_grokking_runs": int((df["status"] == "CONFIRMED_STABLE_GROKKING").sum()),
                "norm_delay_pearson_correlation": pearson,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Analysis written to: {outdir}")


if __name__ == "__main__":
    main()
