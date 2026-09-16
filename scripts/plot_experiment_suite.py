from pathlib import Path
import argparse
import json
import math
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

ACCURACY_THRESHOLD = 99.0
REQUIRED_CONSECUTIVE_SUCCESSES = 2

TRAIN_COLOUR = "#e53935"
VAL_COLOUR = "#2e7d32"

TRAIN_LINEWIDTH = 2.0
VAL_LINEWIDTH = 2.0

DPI = 250


# ============================================================
# FOLDERS WE DO NOT NORMALLY WANT IN --all
# ============================================================

DEFAULT_SKIP_NAMES = {
    "smoke_10step",
    "trial3_canonical_resumed_20260915_082059",
    "runsboundary_calibration_20260915_180629",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def natural_sort_key(path: Path):
    """
    Sort names containing numbers naturally.

    Examples:
        E1 before E2
        T1 before T8
        D3 before D10
    """

    parts = re.split(
        r"(\d+)",
        path.name,
    )

    key = []

    for part in parts:

        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())

    return key


def clean_name(name: str):
    return name.replace("_", " ")


def first_present(
    dictionary,
    keys,
    default=None,
):
    """
    Return the first non-None matching value.
    """

    for key in keys:

        if key in dictionary:

            value = dictionary[key]

            if value is not None:
                return value

    return default


# ============================================================
# FILE LOADING
# ============================================================

def load_stage_result(run_dir: Path):
    """
    Load stage_result.json when available.
    """

    result_file = (
        run_dir
        / "stage_result.json"
    )

    if not result_file.exists():
        return {}

    try:

        with result_file.open(
            "r",
            encoding="utf-8",
        ) as handle:

            return json.load(handle)

    except Exception as exc:

        print(
            f"WARNING: could not read "
            f"{result_file}: {exc}"
        )

        return {}


def load_metrics(run_dir: Path):
    """
    Load analysis_metrics.csv.
    """

    metrics_file = (
        run_dir
        / "analysis_metrics.csv"
    )

    if not metrics_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{metrics_file}"
        )

    df = pd.read_csv(
        metrics_file
    )

    if "step" not in df.columns:

        raise ValueError(
            f"'step' column missing from:\n"
            f"{metrics_file}"
        )

    df["step"] = pd.to_numeric(
        df["step"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["step"])
        .sort_values("step")
        .copy()
    )

    df["step"] = (
        df["step"]
        .astype(int)
    )

    return df


# ============================================================
# ACCURACY DATA
# ============================================================

def get_accuracy_series(df):
    """
    Extract the full-training and validation accuracy curves.
    """

    if "full_train_acc" not in df.columns:

        raise ValueError(
            "'full_train_acc' column missing."
        )

    if "val_accuracy" not in df.columns:

        raise ValueError(
            "'val_accuracy' column missing."
        )

    train = (
        df[
            [
                "step",
                "full_train_acc",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["step"],
            keep="last",
        )
        .sort_values("step")
        .copy()
    )

    val = (
        df[
            [
                "step",
                "val_accuracy",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["step"],
            keep="last",
        )
        .sort_values("step")
        .copy()
    )

    return train, val


def first_threshold_step(
    df,
    column,
    threshold=ACCURACY_THRESHOLD,
):
    """
    First optimisation step where metric >= threshold.
    """

    if column not in df.columns:
        return None

    valid = df[
        df[column].notna()
        & (
            df[column]
            >= threshold
        )
    ]

    if valid.empty:
        return None

    return int(
        valid.iloc[0]["step"]
    )


# ============================================================
# STABLE GROKKING
# ============================================================

def complete_evaluation_rows(df):
    """
    Build rows containing both full training and validation
    accuracy for the same optimisation step.
    """

    train = (
        df[
            [
                "step",
                "full_train_acc",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["step"],
            keep="last",
        )
    )

    val = (
        df[
            [
                "step",
                "val_accuracy",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["step"],
            keep="last",
        )
    )

    merged = pd.merge(
        train,
        val,
        on="step",
        how="inner",
    )

    return (
        merged
        .sort_values("step")
        .reset_index(drop=True)
    )


def first_stable_joint_step(
    df,
    threshold=ACCURACY_THRESHOLD,
    required=REQUIRED_CONSECUTIVE_SUCCESSES,
):
    """
    First step beginning N consecutive evaluations with
    BOTH train and validation >= threshold.
    """

    evaluations = (
        complete_evaluation_rows(
            df
        )
    )

    if evaluations.empty:
        return None

    streak = 0
    streak_start = None

    for _, row in evaluations.iterrows():

        success = (
            row["full_train_acc"]
            >= threshold
            and
            row["val_accuracy"]
            >= threshold
        )

        if success:

            if streak == 0:

                streak_start = int(
                    row["step"]
                )

            streak += 1

            if streak >= required:
                return streak_start

        else:

            streak = 0
            streak_start = None

    return None


# ============================================================
# METADATA
# ============================================================

def get_metadata(
    run_dir,
    df,
):
    """
    Read metadata from stage_result.json where possible.

    Falls back to calculating thresholds directly from
    analysis_metrics.csv.
    """

    result = load_stage_result(
        run_dir
    )

    description = first_present(
        result,
        [
            "description",
            "experiment_description",
        ],
    )

    train_pct = first_present(
        result,
        [
            "train_data_pct",
            "training_percentage",
            "train_percentage",
        ],
    )

    weight_decay = first_present(
        result,
        [
            "weight_decay",
        ],
    )

    learning_rate = first_present(
        result,
        [
            "max_lr",
            "learning_rate",
            "lr",
        ],
    )

    seed = first_present(
        result,
        [
            "seed",
            "random_seed",
        ],
    )

    operator = first_present(
        result,
        [
            "operator",
            "math_operator",
        ],
    )

    # --------------------------------------------------------
    # Threshold fields from old/new result formats
    # --------------------------------------------------------

    train99 = first_present(
        result,
        [
            "first_train99_step",
            "T_train99",
            "train99_step",
            "train99",
        ],
    )

    val99 = first_present(
        result,
        [
            "first_val99_step",
            "T_val99",
            "val99_step",
            "val99",
        ],
    )

    stable99 = first_present(
        result,
        [
            "stable_joint99_step",
            "stable_val99_step",
            "confirmed_grokking_step",
        ],
    )

    # --------------------------------------------------------
    # Calculate missing values directly from metrics
    # --------------------------------------------------------

    if train99 is None:

        train99 = first_threshold_step(
            df,
            "full_train_acc",
        )

    if val99 is None:

        val99 = first_threshold_step(
            df,
            "val_accuracy",
        )

    if stable99 is None:

        stable99 = first_stable_joint_step(
            df
        )

    if train99 is not None:
        train99 = int(train99)

    if val99 is not None:
        val99 = int(val99)

    if stable99 is not None:
        stable99 = int(stable99)

    first_delay = None
    stable_delay = None

    if (
        train99 is not None
        and val99 is not None
    ):

        first_delay = (
            val99
            - train99
        )

    if (
        train99 is not None
        and stable99 is not None
    ):

        stable_delay = (
            stable99
            - train99
        )

    return {
        "description": description,

        "train_pct":
            train_pct,

        "weight_decay":
            weight_decay,

        "learning_rate":
            learning_rate,

        "seed":
            seed,

        "operator":
            operator,

        "train99":
            train99,

        "val99":
            val99,

        "stable99":
            stable99,

        "first_delay":
            first_delay,

        "stable_delay":
            stable_delay,
    }


# ============================================================
# TITLES
# ============================================================

def individual_title(
    run_dir,
    metadata,
):
    """
    Prefer description field when one exists.
    """

    if metadata["description"]:

        return metadata[
            "description"
        ]

    return clean_name(
        run_dir.name
    )


def short_title(
    run_dir,
    metadata,
):
    """
    Short title for a combined multi-panel figure.
    """

    title = clean_name(
        run_dir.name
    )

    if (
        metadata["stable_delay"]
        is not None
    ):

        title += (
            "\nStable delay = "
            f"{metadata['stable_delay']:,}"
            " steps"
        )

    elif (
        metadata["first_delay"]
        is not None
    ):

        title += (
            "\nFirst delay = "
            f"{metadata['first_delay']:,}"
            " steps"
        )

    elif (
        metadata["train99"]
        is not None
    ):

        title += (
            "\nMemorised, not stably "
            "generalised"
        )

    return title


# ============================================================
# PLOT ONE RUN
# ============================================================

def plot_run(
    run_dir,
    output_dir,
    save_pdf=True,
):
    """
    Create the iconic grokking-style graph for one experiment.
    """

    df = load_metrics(
        run_dir
    )

    train, val = (
        get_accuracy_series(
            df
        )
    )

    if train.empty:
        raise RuntimeError(
            "No full training accuracy "
            "data found."
        )

    if val.empty:
        raise RuntimeError(
            "No validation accuracy "
            "data found."
        )

    metadata = get_metadata(
        run_dir,
        df,
    )

    train99 = metadata[
        "train99"
    ]

    val99 = metadata[
        "val99"
    ]

    stable99 = metadata[
        "stable99"
    ]

    first_delay = metadata[
        "first_delay"
    ]

    stable_delay = metadata[
        "stable_delay"
    ]

    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        train["step"],
        train["full_train_acc"],
        label="train",
        color=TRAIN_COLOUR,
        linewidth=TRAIN_LINEWIDTH,
    )

    ax.plot(
        val["step"],
        val["val_accuracy"],
        label="val",
        color=VAL_COLOUR,
        linewidth=VAL_LINEWIDTH,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xscale(
        "log"
    )

    minimum_step = max(
        1,
        int(
            min(
                train["step"].min(),
                val["step"].min(),
            )
        ),
    )

    maximum_step = int(
        max(
            train["step"].max(),
            val["step"].max(),
        )
    )

    ax.set_xlim(
        minimum_step,
        maximum_step * 1.08,
    )

    ax.set_ylim(
        -2,
        104,
    )

    ax.set_xlabel(
        "Optimisation Steps",
        fontsize=13,
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=13,
    )

    ax.set_title(
        individual_title(
            run_dir,
            metadata,
        ),
        fontsize=15,
    )

    ax.grid(
        True,
        which="major",
        alpha=0.30,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.12,
    )

    ax.legend(
        loc="upper left",
        fontsize=12,
    )

    ax.axhline(
        ACCURACY_THRESHOLD,
        color="grey",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )

    # --------------------------------------------------------
    # Training threshold
    # --------------------------------------------------------

    if train99 is not None:

        ax.axvline(
            train99,
            color=TRAIN_COLOUR,
            linestyle="--",
            linewidth=1.1,
            alpha=0.70,
        )

        annotation_x = min(
            train99 * 2.5,
            maximum_step / 3,
        )

        annotation_x = max(
            annotation_x,
            train99 * 1.2,
        )

        ax.annotate(
            (
                "train reaches 99%\n"
                f"at step {train99:,}"
            ),
            xy=(
                train99,
                99,
            ),
            xytext=(
                annotation_x,
                78,
            ),
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
            ),
        )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    if val99 is not None:

        ax.axvline(
            val99,
            color=VAL_COLOUR,
            linestyle="--",
            linewidth=1.1,
            alpha=0.70,
        )

        annotation_x = max(
            val99 / 5,
            minimum_step * 2,
        )

        ax.annotate(
            (
                "validation reaches 99%\n"
                f"at step {val99:,}"
            ),
            xy=(
                val99,
                99,
            ),
            xytext=(
                annotation_x,
                63,
            ),
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
            ),
        )

    else:

        last_val = (
            val.iloc[-1]
        )

        last_step = int(
            last_val["step"]
        )

        last_accuracy = float(
            last_val[
                "val_accuracy"
            ]
        )

        ax.annotate(
            (
                "validation did not "
                "reach 99%\n"
                f"last step = "
                f"{last_step:,}\n"
                f"validation = "
                f"{last_accuracy:.2f}%"
            ),
            xy=(
                last_step,
                last_accuracy,
            ),
            xytext=(
                max(
                    last_step / 8,
                    minimum_step * 2,
                ),
                55,
            ),
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
            ),
        )

    # --------------------------------------------------------
    # Information box
    # --------------------------------------------------------

    info = []

    if first_delay is not None:

        info.append(
            "First-crossing delay: "
            f"{first_delay:,} steps"
        )

    if stable99 is not None:

        info.append(
            "Stable ≥99% step: "
            f"{stable99:,}"
        )

    if stable_delay is not None:

        info.append(
            "Stable grokking delay: "
            f"{stable_delay:,} steps"
        )

    if metadata["train_pct"] is not None:

        info.append(
            "Training data: "
            f"{metadata['train_pct']}%"
        )

    if (
        metadata["weight_decay"]
        is not None
    ):

        info.append(
            "Weight decay: "
            f"{metadata['weight_decay']}"
        )

    if (
        metadata["learning_rate"]
        is not None
    ):

        info.append(
            "Learning rate: "
            f"{metadata['learning_rate']}"
        )

    if info:

        ax.text(
            0.015,
            0.025,
            "\n".join(info),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
                edgecolor="lightgrey",
            ),
        )

    fig.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = (
        f"{run_dir.name}"
        "_grokking_curve"
    )

    png_path = (
        output_dir
        / f"{base_name}.png"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    if save_pdf:

        pdf_path = (
            output_dir
            / f"{base_name}.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(
        f"    Created: "
        f"{png_path.name}"
    )

    return {
        "run":
            run_dir.name,

        "description":
            metadata["description"],

        "train_pct":
            metadata["train_pct"],

        "weight_decay":
            metadata["weight_decay"],

        "learning_rate":
            metadata["learning_rate"],

        "seed":
            metadata["seed"],

        "operator":
            metadata["operator"],

        "train99":
            train99,

        "val99":
            val99,

        "stable99":
            stable99,

        "first_delay":
            first_delay,

        "stable_delay":
            stable_delay,

        "png":
            str(png_path),
    }


# ============================================================
# COMBINED FIGURE
# ============================================================

def make_combined_figure(
    run_dirs,
    output_dir,
    suite_name,
    save_pdf=True,
):
    """
    Create a dynamically sized combined figure.
    """

    count = len(
        run_dirs
    )

    if count == 0:
        return

    ncols = 2

    nrows = math.ceil(
        count / ncols
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            15,
            5 * nrows,
        ),
        sharey=True,
        squeeze=False,
    )

    axes = axes.flatten()

    for index, run_dir in enumerate(
        run_dirs
    ):

        ax = axes[index]

        df = load_metrics(
            run_dir
        )

        train, val = (
            get_accuracy_series(
                df
            )
        )

        metadata = get_metadata(
            run_dir,
            df,
        )

        ax.plot(
            train["step"],
            train["full_train_acc"],
            color=TRAIN_COLOUR,
            linewidth=1.6,
            label="train",
        )

        ax.plot(
            val["step"],
            val["val_accuracy"],
            color=VAL_COLOUR,
            linewidth=1.6,
            label="val",
        )

        ax.set_xscale(
            "log"
        )

        ax.set_ylim(
            -2,
            104,
        )

        ax.grid(
            True,
            which="major",
            alpha=0.28,
        )

        ax.grid(
            True,
            which="minor",
            alpha=0.10,
        )

        ax.axhline(
            ACCURACY_THRESHOLD,
            color="grey",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )

        if (
            metadata["train99"]
            is not None
        ):

            ax.axvline(
                metadata["train99"],
                color=TRAIN_COLOUR,
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )

        if (
            metadata["val99"]
            is not None
        ):

            ax.axvline(
                metadata["val99"],
                color=VAL_COLOUR,
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )

        ax.set_title(
            short_title(
                run_dir,
                metadata,
            ),
            fontsize=10,
        )

        ax.set_xlabel(
            "Optimisation Steps"
        )

        ax.set_ylabel(
            "Accuracy (%)"
        )

    # Hide unused subplot positions.
    for index in range(
        count,
        len(axes),
    ):

        axes[index].axis(
            "off"
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=12,
    )

    fig.suptitle(
        (
            "Grokking Experiment Suite\n"
            f"{suite_name}"
        ),
        fontsize=17,
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.965,
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "combined_grokking_curves.png"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    if save_pdf:

        pdf_path = (
            output_dir
            / "combined_grokking_curves.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(
        "    Created: "
        "combined_grokking_curves.png"
    )


# ============================================================
# DISCOVERY
# ============================================================

def discover_run_dirs(
    suite_dir,
):
    """
    Find experiment directories.

    Supports two arrangements:

    1. Suite folder containing:
       E1_...\analysis_metrics.csv
       E2_...\analysis_metrics.csv

    2. A single run folder containing:
       analysis_metrics.csv
    """

    own_metrics = (
        suite_dir
        / "analysis_metrics.csv"
    )

    if own_metrics.exists():

        return [
            suite_dir
        ]

    run_dirs = []

    for child in suite_dir.iterdir():

        if not child.is_dir():
            continue

        metrics = (
            child
            / "analysis_metrics.csv"
        )

        if metrics.exists():

            run_dirs.append(
                child
            )

    return sorted(
        run_dirs,
        key=natural_sort_key,
    )


def looks_like_suite(
    directory,
):
    """
    Determine whether a top-level runs directory contains
    plottable experiment data.
    """

    if not directory.is_dir():
        return False

    if (
        directory
        / "analysis_metrics.csv"
    ).exists():

        return True

    for child in directory.iterdir():

        if (
            child.is_dir()
            and (
                child
                / "analysis_metrics.csv"
            ).exists()
        ):

            return True

    return False


# ============================================================
# PROCESS ONE SUITE
# ============================================================

def process_suite(
    suite_dir,
    save_pdf=True,
):
    """
    Plot every run belonging to a suite.
    """

    print()
    print(
        "=" * 80
    )

    print(
        f"SUITE: {suite_dir.name}"
    )

    print(
        "=" * 80
    )

    print(
        f"Path:\n{suite_dir}"
    )

    run_dirs = (
        discover_run_dirs(
            suite_dir
        )
    )

    if not run_dirs:

        print(
            "  No analysis_metrics.csv "
            "files found. Skipping."
        )

        return

    print()
    print(
        f"  Found {len(run_dirs)} "
        "plottable run(s):"
    )

    for run_dir in run_dirs:

        print(
            f"    {run_dir.name}"
        )

    output_dir = (
        suite_dir
        / "plots"
    )

    results = []

    print()
    print(
        "  Creating individual plots..."
    )

    for run_dir in run_dirs:

        try:

            result = plot_run(
                run_dir,
                output_dir,
                save_pdf=save_pdf,
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"    ERROR: "
                f"{run_dir.name}: "
                f"{exc}"
            )

    if len(run_dirs) > 1:

        print()
        print(
            "  Creating combined figure..."
        )

        try:

            make_combined_figure(
                run_dirs,
                output_dir,
                suite_dir.name,
                save_pdf=save_pdf,
            )

        except Exception as exc:

            print(
                "    ERROR creating "
                f"combined figure: {exc}"
            )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    if results:

        summary_path = (
            output_dir
            / "plot_summary.csv"
        )

        pd.DataFrame(
            results
        ).to_csv(
            summary_path,
            index=False,
        )

        print()
        print(
            "    Created: "
            "plot_summary.csv"
        )

    print()
    print(
        f"  Output:\n"
        f"  {output_dir}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Plot grokking experiment "
            "accuracy curves."
        )
    )

    group = (
        parser
        .add_mutually_exclusive_group()
    )

    group.add_argument(
        "suite",
        nargs="?",
        help=(
            "Path to one experiment suite. "
            "Example: "
            r".\runs\calibration_20260915_151232"
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Automatically scan the runs "
            "directory and plot every suite."
        ),
    )

    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help=(
            "Include smoke tests and temporary "
            "runs normally skipped by --all."
        ),
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help=(
            "Only save PNG figures."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    save_pdf = (
        not args.no_pdf
    )

    print(
        "=" * 80
    )

    print(
        "GENERAL GROKKING EXPERIMENT PLOTTER"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # --all mode
    # ========================================================

    if args.all:

        if not RUNS_DIR.exists():

            raise FileNotFoundError(
                f"Runs directory not found:\n"
                f"{RUNS_DIR}"
            )

        suites = []

        for directory in RUNS_DIR.iterdir():

            if not directory.is_dir():
                continue

            if (
                not args.include_skipped
                and directory.name
                in DEFAULT_SKIP_NAMES
            ):

                print(
                    f"Skipping test/temporary "
                    f"folder: "
                    f"{directory.name}"
                )

                continue

            try:

                if looks_like_suite(
                    directory
                ):

                    suites.append(
                        directory
                    )

            except Exception as exc:

                print(
                    f"Could not inspect "
                    f"{directory.name}: "
                    f"{exc}"
                )

        suites = sorted(
            suites,
            key=natural_sort_key,
        )

        if not suites:

            raise RuntimeError(
                "No plottable experiment "
                "suites found."
            )

        print()
        print(
            f"Found {len(suites)} "
            "experiment suites."
        )

        for suite in suites:

            process_suite(
                suite,
                save_pdf=save_pdf,
            )

        print()
        print(
            "=" * 80
        )

        print(
            "ALL SUITES COMPLETE"
        )

        print(
            "=" * 80
        )

        return

    # ========================================================
    # Single-suite mode
    # ========================================================

    if args.suite:

        suite_dir = Path(
            args.suite
        )

        if not suite_dir.is_absolute():

            suite_dir = (
                ROOT
                / suite_dir
            ).resolve()

    else:

        print()
        print(
            "No suite specified."
        )

        print()
        print(
            "Examples:"
        )

        print(
            r"  python .\scripts\plot_experiment_suite.py "
            r".\runs\calibration_20260915_151232"
        )

        print()

        print(
            r"  python .\scripts\plot_experiment_suite.py "
            r".\runs\boundary_calibration_20260915_180629"
        )

        print()

        print(
            r"  python .\scripts\plot_experiment_suite.py "
            r".\runs\task_comparison_overnight_20260916_012229"
        )

        print()

        print(
            r"  python .\scripts\plot_experiment_suite.py --all"
        )

        return

    if not suite_dir.exists():

        raise FileNotFoundError(
            f"Experiment suite does not exist:\n"
            f"{suite_dir}"
        )

    process_suite(
        suite_dir,
        save_pdf=save_pdf,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()