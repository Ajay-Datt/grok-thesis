from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing T1_addition_seed0, T2_subtraction_seed0, etc.
SUITE_DIR = Path(
    r"C:\Users\Ajay\Documents\Thesis\runs"
    r"\task_comparison_overnight_20260916_012229"
)

# Figures are written here.
OUTPUT_DIR = SUITE_DIR / "plots"

ACCURACY_THRESHOLD = 99.0

# Save both PNG and PDF versions.
SAVE_PDF = True

# Plot appearance.
TRAIN_COLOUR = "#e53935"   # red, similar to iconic grokking figure
VAL_COLOUR = "#2e7d32"     # green

TRAIN_LINEWIDTH = 2.0
VAL_LINEWIDTH = 2.0

DPI = 250


# ============================================================
# TASK NAMES
# ============================================================

OPERATION_NAMES = {
    "addition": "Modular Addition",
    "subtraction": "Modular Subtraction",
    "multiplication": "Modular Multiplication",
    "division": "Modular Division",
}


# ============================================================
# HELPERS
# ============================================================

def run_sort_key(path: Path):
    """
    Sort T1, T2, ..., T8 numerically.
    """

    match = re.match(
        r"T(\d+)_",
        path.name,
    )

    if match:
        return int(match.group(1))

    return 999


def parse_run_name(run_name: str):
    """
    Example:

        T1_addition_seed0

    becomes:

        task_number = 1
        operation = addition
        seed = 0
    """

    match = re.match(
        r"T(\d+)_([A-Za-z]+)_seed(\d+)",
        run_name,
    )

    if not match:
        return None, run_name, None

    task_number = int(
        match.group(1)
    )

    operation = (
        match.group(2).lower()
    )

    seed = int(
        match.group(3)
    )

    return (
        task_number,
        operation,
        seed,
    )


def first_threshold_step(
    df,
    column,
    threshold=99.0,
):
    """
    Return the first optimisation step where the requested
    accuracy reaches the threshold.
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


def load_stage_result(run_dir):
    """
    Load stage_result.json if available.

    This contains the stable-threshold results produced by the
    overnight experiment script.
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


def load_metrics(run_dir):
    """
    Load analysis_metrics.csv and ensure optimisation steps
    are valid and ordered.
    """

    metrics_file = (
        run_dir
        / "analysis_metrics.csv"
    )

    if not metrics_file.exists():

        raise FileNotFoundError(
            f"Missing: {metrics_file}"
        )

    df = pd.read_csv(
        metrics_file
    )

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

    df["step"] = (
        df["step"]
        .astype(int)
    )

    return df


def get_accuracy_series(df):
    """
    Extract exact full-training and validation accuracy data.

    analysis_metrics.csv contains sparse values, so training
    and validation are filtered independently.
    """

    if "full_train_acc" not in df.columns:

        raise ValueError(
            "'full_train_acc' missing."
        )

    if "val_accuracy" not in df.columns:

        raise ValueError(
            "'val_accuracy' missing."
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


# ============================================================
# INDIVIDUAL GROKKING PLOT
# ============================================================

def make_individual_plot(run_dir):
    """
    Create one iconic grokking-style graph for a run.
    """

    (
        task_number,
        operation,
        seed,
    ) = parse_run_name(
        run_dir.name
    )

    operation_title = (
        OPERATION_NAMES.get(
            operation,
            operation.title(),
        )
    )

    df = load_metrics(
        run_dir
    )

    train, val = (
        get_accuracy_series(df)
    )

    result = load_stage_result(
        run_dir
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    train99 = result.get(
        "first_train99_step"
    )

    val99 = result.get(
        "first_val99_step"
    )

    stable99 = result.get(
        "stable_joint99_step"
    )

    stable_delay = result.get(
        "stable_grokking_delay_steps"
    )

    # Fallback to calculation directly from CSV.
    if train99 is None:

        train99 = first_threshold_step(
            df,
            "full_train_acc",
            ACCURACY_THRESHOLD,
        )

    if val99 is None:

        val99 = first_threshold_step(
            df,
            "val_accuracy",
            ACCURACY_THRESHOLD,
        )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    # Exact raw curves.
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

    maximum_step = max(
        train["step"].max(),
        val["step"].max(),
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
        f"{operation_title} mod 97 "
        f"(training on 50% of data, seed {seed})",
        fontsize=16,
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
    # 99% horizontal reference
    # --------------------------------------------------------

    ax.axhline(
        ACCURACY_THRESHOLD,
        color="grey",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )

    # --------------------------------------------------------
    # Train threshold marker
    # --------------------------------------------------------

    if train99 is not None:

        ax.axvline(
            train99,
            color=TRAIN_COLOUR,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )

        train_text_x = (
            train99 * 2.3
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
    # Validation threshold marker
    # --------------------------------------------------------

    if val99 is not None:

        ax.axvline(
            val99,
            color=VAL_COLOUR,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )

        val_text_x = max(
            val99 / 4,
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
                65,
            ),
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
            ),
        )

    # --------------------------------------------------------
    # Result text
    # --------------------------------------------------------

    information = []

    if (
        train99 is not None
        and val99 is not None
    ):

        first_delay = (
            val99
            - train99
        )

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
                alpha=0.8,
                edgecolor="lightgrey",
            ),
        )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
        "task_number": task_number,
        "operation": operation,
        "seed": seed,
        "train99": train99,
        "val99": val99,
        "stable99": stable99,
        "stable_delay": stable_delay,
        "png": str(png_path),
    }


# ============================================================
# COMBINED 8-PLOT FIGURE
# ============================================================

def make_combined_figure(run_dirs):
    """
    Make one 4 x 2 figure containing all eight experiments.
    """

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(15, 20),
        sharey=True,
    )

    axes = axes.flatten()

    for ax, run_dir in zip(
        axes,
        run_dirs,
    ):

        (
            task_number,
            operation,
            seed,
        ) = parse_run_name(
            run_dir.name
        )

        operation_title = (
            OPERATION_NAMES.get(
                operation,
                operation.title(),
            )
        )

        df = load_metrics(
            run_dir
        )

        train, val = (
            get_accuracy_series(df)
        )

        result = load_stage_result(
            run_dir
        )

        train99 = result.get(
            "first_train99_step"
        )

        val99 = result.get(
            "first_val99_step"
        )

        stable_delay = result.get(
            "stable_grokking_delay_steps"
        )

        if train99 is None:

            train99 = (
                first_threshold_step(
                    df,
                    "full_train_acc",
                )
            )

        if val99 is None:

            val99 = (
                first_threshold_step(
                    df,
                    "val_accuracy",
                )
            )

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
            99,
            color="grey",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )

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

        title = (
            f"T{task_number}: "
            f"{operation_title}, "
            f"seed {seed}"
        )

        if stable_delay is not None:

            title += (
                f"\n"
                f"stable delay = "
                f"{stable_delay:,} steps"
            )

        ax.set_title(
            title,
            fontsize=12,
        )

        ax.set_xlabel(
            "Optimisation Steps"
        )

        ax.set_ylabel(
            "Accuracy (%)"
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
            "Grokking Across Modular Arithmetic Tasks\n"
            "50% training data, "
            "weight decay = 0.1, "
            "learning rate = 0.001"
        ),
        fontsize=18,
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    png_path = (
        OUTPUT_DIR
        / "all_8_task_comparison_grokking_curves.png"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    if SAVE_PDF:

        pdf_path = (
            OUTPUT_DIR
            / "all_8_task_comparison_grokking_curves.pdf"
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

def save_plot_summary(results):

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
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GROKKING TASK-COMPARISON PLOTTER")
    print("=" * 80)

    print(
        f"Reading from:\n{SUITE_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find experiment directories automatically.
    run_dirs = sorted(
        [
            path
            for path in SUITE_DIR.iterdir()
            if (
                path.is_dir()
                and re.match(
                    r"T\d+_",
                    path.name,
                )
                and (
                    path
                    / "analysis_metrics.csv"
                ).exists()
            )
        ],
        key=run_sort_key,
    )

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

    # Individual plots.
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
                f"{run_dir.name}: {exc}"
            )

    # Combined 8-run figure.
    if run_dirs:

        make_combined_figure(
            run_dirs
        )

    # Small CSV containing the important thresholds.
    save_plot_summary(
        results
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Figures saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()