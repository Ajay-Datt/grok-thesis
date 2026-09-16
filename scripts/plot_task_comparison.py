from pathlib import Path
import json
import math
import re

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Change ONLY this path when plotting a different experiment suite.
SUITE_DIR = Path(
    r"C:\Users\Ajay\Documents\Thesis\runs"
    r"\boundary_calibration_20260915_180629"
)

# All generated figures are saved here.
OUTPUT_DIR = SUITE_DIR / "plots"

ACCURACY_THRESHOLD = 99.0
REQUIRED_CONSECUTIVE_SUCCESSES = 2

# Save both PNG and PDF copies.
SAVE_PDF = True

# Plot appearance.
TRAIN_COLOUR = "#e53935"
VAL_COLOUR = "#2e7d32"

TRAIN_LINEWIDTH = 2.0
VAL_LINEWIDTH = 2.0

DPI = 250


# ============================================================
# HELPERS
# ============================================================

def run_sort_key(path: Path):
    """
    Sort experiment folders numerically.

    Examples:
        E1_fraction30_wd1
        E2_50pct_wd003
        T1_addition_seed0
        D7_50pct_wd01

    E1 comes before E2, T1 before T2, etc.
    """

    match = re.match(
        r"^[A-Za-z]+(\d+)",
        path.name,
    )

    if match:
        return int(match.group(1))

    return 999999


def clean_name(name: str):
    """
    Convert a folder name into a readable fallback title.
    """

    return name.replace("_", " ")


def get_first_present(dictionary, keys, default=None):
    """
    Return the first non-None value found in a dictionary.
    """

    for key in keys:
        if key in dictionary:
            value = dictionary[key]

            if value is not None:
                return value

    return default


# ============================================================
# LOAD FILES
# ============================================================

def load_stage_result(run_dir: Path):
    """
    Load stage_result.json if present.

    Older and newer experiment scripts may store slightly
    different fields, so this plotter does not depend on it
    being present.
    """

    result_file = run_dir / "stage_result.json"

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

    metrics_file = run_dir / "analysis_metrics.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Missing: {metrics_file}"
        )

    df = pd.read_csv(metrics_file)

    if "step" not in df.columns:
        raise ValueError(
            f"'step' column missing from "
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

    df["step"] = df["step"].astype(int)

    return df


# ============================================================
# ACCURACY SERIES
# ============================================================

def get_accuracy_series(df):
    """
    Extract full-training and validation accuracy.

    analysis_metrics.csv may have sparse rows, so each series
    is filtered independently.
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
            ["step", "full_train_acc"]
        ]
        .dropna()
        .copy()
    )

    val = (
        df[
            ["step", "val_accuracy"]
        ]
        .dropna()
        .copy()
    )

    return train, val


def first_threshold_step(
    df,
    column,
    threshold=ACCURACY_THRESHOLD,
):
    """
    First step where a metric reaches the threshold.
    """

    if column not in df.columns:
        return None

    valid = df[
        df[column].notna()
        & (df[column] >= threshold)
    ]

    if valid.empty:
        return None

    return int(
        valid.iloc[0]["step"]
    )


# ============================================================
# STABLE GROKKING CALCULATION
# ============================================================

def build_complete_evaluation_rows(df):
    """
    Build rows where both full training accuracy and
    validation accuracy exist for the same optimisation step.
    """

    train = (
        df[
            ["step", "full_train_acc"]
        ]
        .dropna()
        .drop_duplicates(
            subset=["step"],
            keep="last",
        )
    )

    val = (
        df[
            ["step", "val_accuracy"]
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

    return merged.sort_values("step")


def first_stable_joint_step(
    df,
    threshold=ACCURACY_THRESHOLD,
    required=REQUIRED_CONSECUTIVE_SUCCESSES,
):
    """
    Find the first step beginning a run of N consecutive
    evaluations where train AND validation are both >=99%.
    """

    evaluations = build_complete_evaluation_rows(
        df
    )

    if evaluations.empty:
        return None

    streak = 0
    streak_start = None

    for _, row in evaluations.iterrows():

        successful = (
            row["full_train_acc"] >= threshold
            and row["val_accuracy"] >= threshold
        )

        if successful:

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
# RUN METADATA
# ============================================================

def get_run_metadata(run_dir, df):
    """
    Gather useful metadata from stage_result.json.

    Falls back safely when older result files use different
    field names.
    """

    result = load_stage_result(
        run_dir
    )

    description = get_first_present(
        result,
        [
            "description",
            "experiment_description",
        ],
        default=None,
    )

    train_pct = get_first_present(
        result,
        [
            "train_data_pct",
            "training_percentage",
            "train_percentage",
        ],
        default=None,
    )

    weight_decay = get_first_present(
        result,
        [
            "weight_decay",
        ],
        default=None,
    )

    learning_rate = get_first_present(
        result,
        [
            "max_lr",
            "learning_rate",
            "lr",
        ],
        default=None,
    )

    seed = get_first_present(
        result,
        [
            "seed",
            "random_seed",
        ],
        default=None,
    )

    operator = get_first_present(
        result,
        [
            "operator",
            "math_operator",
        ],
        default=None,
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    train99 = get_first_present(
        result,
        [
            "first_train99_step",
            "T_train99",
            "train99_step",
            "train99",
        ],
        default=None,
    )

    val99 = get_first_present(
        result,
        [
            "first_val99_step",
            "T_val99",
            "val99_step",
            "val99",
        ],
        default=None,
    )

    stable99 = get_first_present(
        result,
        [
            "stable_joint99_step",
            "stable_val99_step",
            "confirmed_grokking_step",
        ],
        default=None,
    )

    # Calculate directly from metrics if missing.
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

    if (
        train99 is not None
        and val99 is not None
    ):
        first_delay = (
            val99
            - train99
        )

    stable_delay = None

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
        "train_pct": train_pct,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "seed": seed,
        "operator": operator,

        "train99": train99,
        "val99": val99,
        "stable99": stable99,

        "first_delay": first_delay,
        "stable_delay": stable_delay,

        "result": result,
    }


# ============================================================
# TITLES
# ============================================================

def make_plot_title(
    run_dir,
    metadata,
):
    """
    Prefer the experiment description from stage_result.json.
    Otherwise use a readable version of the directory name.
    """

    description = metadata[
        "description"
    ]

    if description:
        return description

    return clean_name(
        run_dir.name
    )


def make_short_title(
    run_dir,
    metadata,
):
    """
    Short title for combined plots.
    """

    title = clean_name(
        run_dir.name
    )

    stable_delay = metadata[
        "stable_delay"
    ]

    if stable_delay is not None:
        title += (
            f"\nStable delay = "
            f"{stable_delay:,} steps"
        )

    elif metadata["first_delay"] is not None:
        title += (
            f"\nFirst delay = "
            f"{metadata['first_delay']:,} steps"
        )

    return title


# ============================================================
# INDIVIDUAL PLOT
# ============================================================

def make_individual_plot(run_dir):
    """
    Generate one full grokking-style plot.
    """

    df = load_metrics(
        run_dir
    )

    train, val = get_accuracy_series(
        df
    )

    metadata = get_run_metadata(
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
        make_plot_title(
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

    # --------------------------------------------------------
    # 99% reference
    # --------------------------------------------------------

    ax.axhline(
        ACCURACY_THRESHOLD,
        color="grey",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )

    # --------------------------------------------------------
    # Train threshold
    # --------------------------------------------------------

    if train99 is not None:

        ax.axvline(
            train99,
            color=TRAIN_COLOUR,
            linestyle="--",
            linewidth=1.1,
            alpha=0.70,
        )

        train_text_x = min(
            train99 * 2.5,
            maximum_step / 3,
        )

        train_text_x = max(
            train_text_x,
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
                train_text_x,
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

        val_text_x = max(
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
                val_text_x,
                63,
            ),
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
            ),
        )

    # --------------------------------------------------------
    # If validation did not reach 99%
    # --------------------------------------------------------

    if val99 is None:

        if not val.empty:

            last_val_step = int(
                val.iloc[-1]["step"]
            )

            last_val_acc = float(
                val.iloc[-1]["val_accuracy"]
            )

            ax.annotate(
                (
                    "validation did not reach 99%\n"
                    f"last step = {last_val_step:,}\n"
                    f"final val = {last_val_acc:.2f}%"
                ),
                xy=(
                    last_val_step,
                    last_val_acc,
                ),
                xytext=(
                    max(
                        last_val_step / 8,
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

    information = []

    if first_delay is not None:
        information.append(
            f"First-crossing delay: "
            f"{first_delay:,} steps"
        )

    if stable99 is not None:
        information.append(
            f"Stable ≥99% step: "
            f"{stable99:,}"
        )

    if stable_delay is not None:
        information.append(
            f"Stable grokking delay: "
            f"{stable_delay:,} steps"
        )

    train_pct = metadata[
        "train_pct"
    ]

    weight_decay = metadata[
        "weight_decay"
    ]

    learning_rate = metadata[
        "learning_rate"
    ]

    if train_pct is not None:
        information.append(
            f"Training data: {train_pct}%"
        )

    if weight_decay is not None:
        information.append(
            f"Weight decay: {weight_decay}"
        )

    if learning_rate is not None:
        information.append(
            f"Learning rate: {learning_rate}"
        )

    if information:

        ax.text(
            0.015,
            0.025,
            "\n".join(
                information
            ),
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

    filename_base = (
        f"{run_dir.name}"
        "_grokking_curve"
    )

    png_path = (
        OUTPUT_DIR
        / f"{filename_base}.png"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    if SAVE_PDF:

        pdf_path = (
            OUTPUT_DIR
            / f"{filename_base}.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(
        f"Created: {png_path.name}"
    )

    return {
        "run": run_dir.name,

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
):
    """
    Create one multi-panel figure containing every experiment
    discovered in the suite.
    """

    number_of_runs = len(
        run_dirs
    )

    ncols = 2

    nrows = math.ceil(
        number_of_runs
        / ncols
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            15,
            5 * nrows,
        ),
        sharey=True,
    )

    # Handle one-row case safely.
    if hasattr(
        axes,
        "flatten",
    ):
        axes = axes.flatten()

    else:
        axes = [axes]

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

        metadata = (
            get_run_metadata(
                run_dir,
                df,
            )
        )

        train99 = metadata[
            "train99"
        ]

        val99 = metadata[
            "val99"
        ]

        # ----------------------------------------------------
        # Curves
        # ----------------------------------------------------

        ax.plot(
            train["step"],
            train["full_train_acc"],
            color=TRAIN_COLOUR,
            linewidth=1.7,
            label="train",
        )

        ax.plot(
            val["step"],
            val["val_accuracy"],
            color=VAL_COLOUR,
            linewidth=1.7,
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

        # ----------------------------------------------------
        # Threshold lines
        # ----------------------------------------------------

        if train99 is not None:

            ax.axvline(
                train99,
                color=TRAIN_COLOUR,
                linestyle="--",
                linewidth=0.9,
                alpha=0.65,
            )

        if val99 is not None:

            ax.axvline(
                val99,
                color=VAL_COLOUR,
                linestyle="--",
                linewidth=0.9,
                alpha=0.65,
            )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ax.set_title(
            make_short_title(
                run_dir,
                metadata,
            ),
            fontsize=11,
        )

        ax.set_xlabel(
            "Optimisation Steps"
        )

        ax.set_ylabel(
            "Accuracy (%)"
        )

    # Hide unused panels.
    for index in range(
        number_of_runs,
        len(axes),
    ):
        axes[index].axis(
            "off"
        )

    # Shared legend.
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
            f"Grokking Experiment Suite\n"
            f"{SUITE_DIR.name}"
        ),
        fontsize=18,
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

    png_path = (
        OUTPUT_DIR
        / "combined_grokking_curves.png"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    if SAVE_PDF:

        pdf_path = (
            OUTPUT_DIR
            / "combined_grokking_curves.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(
        f"Created: {png_path.name}"
    )


# ============================================================
# SUMMARY CSV
# ============================================================

def save_plot_summary(
    results,
):
    """
    Save useful plotting / threshold results.
    """

    summary = pd.DataFrame(
        results
    )

    output = (
        OUTPUT_DIR
        / "plot_summary.csv"
    )

    summary.to_csv(
        output,
        index=False,
    )

    print(
        f"Created: {output.name}"
    )


# ============================================================
# DISCOVER RUNS
# ============================================================

def discover_runs():
    """
    Find every immediate child directory containing an
    analysis_metrics.csv file.

    This means the script works with T*, D*, E*, etc.
    """

    if not SUITE_DIR.exists():

        raise FileNotFoundError(
            f"Suite directory does not exist:\n"
            f"{SUITE_DIR}"
        )

    run_dirs = []

    for path in SUITE_DIR.iterdir():

        if not path.is_dir():
            continue

        metrics_file = (
            path
            / "analysis_metrics.csv"
        )

        if metrics_file.exists():
            run_dirs.append(
                path
            )

    return sorted(
        run_dirs,
        key=run_sort_key,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "GENERAL GROKKING EXPERIMENT PLOTTER"
    )

    print(
        "=" * 80
    )

    print(
        f"Reading from:\n"
        f"{SUITE_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dirs = discover_runs()

    if not run_dirs:

        raise RuntimeError(
            "No experiment folders containing "
            "analysis_metrics.csv were found."
        )

    print()
    print(
        f"Found {len(run_dirs)} runs:"
    )

    for run_dir in run_dirs:

        print(
            f"  {run_dir.name}"
        )

    print()

    results = []

    # --------------------------------------------------------
    # Individual plots
    # --------------------------------------------------------

    for run_dir in run_dirs:

        try:

            result = (
                make_individual_plot(
                    run_dir
                )
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"ERROR plotting "
                f"{run_dir.name}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Combined figure
    # --------------------------------------------------------

    if run_dirs:

        try:

            make_combined_figure(
                run_dirs
            )

        except Exception as exc:

            print(
                "ERROR creating combined "
                f"figure: {exc}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    save_plot_summary(
        results
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

    print(
        f"Figures saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()