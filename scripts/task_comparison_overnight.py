import csv
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

POLL_SECONDS = 25

ACCURACY_THRESHOLD = 99.0
REQUIRED_CONSECUTIVE_SUCCESSES = 2

# Overnight safeguard.
# At ~23-25 steps/s, the full suite should normally fit inside this.
MAX_SUITE_HOURS = 11.0


# ============================================================
# FIXED EXPERIMENTAL BASELINE
# ============================================================

TRAIN_DATA_PCT = 50
WEIGHT_DECAY = 0.1
MAX_LR = 0.001


# ============================================================
# OVERNIGHT TASK COMPARISON
#
# Only the operation and random seed change.
# ============================================================

STAGES = [
    {
        "name": "T1_addition_seed0",
        "description": "Addition mod 97, seed 0",
        "operator": "+",
        "seed": 0,
        "max_steps": 80_000,
    },
    {
        "name": "T2_subtraction_seed0",
        "description": "Subtraction mod 97, seed 0",
        "operator": "-",
        "seed": 0,
        "max_steps": 80_000,
    },
    {
        "name": "T3_multiplication_seed0",
        "description": "Multiplication mod 97, seed 0",
        "operator": "*",
        "seed": 0,
        "max_steps": 150_000,
    },
    {
        "name": "T4_addition_seed1",
        "description": "Addition mod 97, seed 1",
        "operator": "+",
        "seed": 1,
        "max_steps": 80_000,
    },
    {
        "name": "T5_subtraction_seed1",
        "description": "Subtraction mod 97, seed 1",
        "operator": "-",
        "seed": 1,
        "max_steps": 80_000,
    },
    {
        "name": "T6_multiplication_seed1",
        "description": "Multiplication mod 97, seed 1",
        "operator": "*",
        "seed": 1,
        "max_steps": 150_000,
    },
    {
        "name": "T7_division_seed1",
        "description": "Division mod 97, seed 1",
        "operator": "/",
        "seed": 1,
        "max_steps": 80_000,
    },
    {
        "name": "T8_division_seed2",
        "description": "Division mod 97, seed 2",
        "operator": "/",
        "seed": 2,
        "max_steps": 80_000,
    },
]


# ============================================================
# WINDOWS SLEEP PREVENTION
# ============================================================

def prevent_windows_sleep():
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )


def restore_windows_sleep():
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS
        )


# ============================================================
# HELPERS
# ============================================================

def fmt_time(seconds):
    if seconds is None:
        return "--:--:--"

    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def as_float(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# LIGHTNING METRIC READING
# ============================================================

def find_metrics_file(run_dir):
    candidates = list(
        run_dir.rglob("metrics.csv")
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def read_raw_metrics(run_dir):
    metrics_file = find_metrics_file(
        run_dir
    )

    if metrics_file is None:
        return []

    try:
        with metrics_file.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:

            return list(
                csv.DictReader(handle)
            )

    except (OSError, csv.Error):
        # Lightning may be writing to the file.
        return []


def build_timeline(run_dir):
    """
    Merge sparse Lightning rows into one row per optimisation step.
    """

    rows = read_raw_metrics(
        run_dir
    )

    by_step = {}

    useful_columns = [
        "full_train_acc",
        "val_accuracy",
        "full_train_loss",
        "val_loss",
        "train_accuracy",
        "train_loss",
        "learning_rate",
        "epoch",
    ]

    for row in rows:

        raw_step = as_float(
            row.get("step")
        )

        if raw_step is None:
            continue

        step = int(raw_step)

        if step not in by_step:
            by_step[step] = {
                "step": step
            }

        for column in useful_columns:

            value = as_float(
                row.get(column)
            )

            if value is not None:
                by_step[step][column] = value

    return [
        by_step[step]
        for step in sorted(by_step)
    ]


def evaluation_rows(run_dir):
    """
    Return complete evaluation rows containing both full training
    accuracy and validation accuracy.
    """

    timeline = build_timeline(
        run_dir
    )

    return [
        row
        for row in timeline
        if (
            row.get("full_train_acc") is not None
            and row.get("val_accuracy") is not None
        )
    ]


# ============================================================
# METRIC EXTRACTION
# ============================================================

def latest_status(run_dir):
    timeline = build_timeline(
        run_dir
    )

    if not timeline:
        return 0, None, None

    latest_step = max(
        row["step"]
        for row in timeline
    )

    latest_train = None
    latest_val = None

    for row in reversed(timeline):

        if latest_train is None:
            latest_train = row.get(
                "full_train_acc"
            )

        if latest_val is None:
            latest_val = row.get(
                "val_accuracy"
            )

        if (
            latest_train is not None
            and latest_val is not None
        ):
            break

    return (
        latest_step,
        latest_train,
        latest_val,
    )


def first_threshold_step(
    run_dir,
    metric_name,
):
    timeline = build_timeline(
        run_dir
    )

    for row in timeline:

        value = row.get(
            metric_name
        )

        if (
            value is not None
            and value >= ACCURACY_THRESHOLD
        ):
            return row["step"]

    return None


def peak_metric(
    run_dir,
    metric_name,
):
    timeline = build_timeline(
        run_dir
    )

    values = [
        row[metric_name]
        for row in timeline
        if row.get(metric_name) is not None
    ]

    if not values:
        return None

    return max(values)


def consecutive_success_count(run_dir):
    """
    Count successful complete evaluations from the end of the
    evaluation history.

    Two old successes followed by a later failure therefore do
    NOT count as stable grokking.
    """

    rows = evaluation_rows(
        run_dir
    )

    count = 0

    for row in reversed(rows):

        if (
            row["full_train_acc"] >= ACCURACY_THRESHOLD
            and row["val_accuracy"] >= ACCURACY_THRESHOLD
        ):
            count += 1

        else:
            break

    return count


def first_stable_joint_step(run_dir):
    """
    Find the first evaluation in the first streak of N consecutive
    evaluations where BOTH training and validation are >=99%.

    Returns the first step of that successful streak.
    """

    rows = evaluation_rows(
        run_dir
    )

    streak = 0
    streak_start = None

    for row in rows:

        successful = (
            row["full_train_acc"] >= ACCURACY_THRESHOLD
            and row["val_accuracy"] >= ACCURACY_THRESHOLD
        )

        if successful:

            if streak == 0:
                streak_start = row["step"]

            streak += 1

            if (
                streak
                >= REQUIRED_CONSECUTIVE_SUCCESSES
            ):
                return streak_start

        else:

            streak = 0
            streak_start = None

    return None


def stage_has_grokked(run_dir):
    return (
        consecutive_success_count(run_dir)
        >= REQUIRED_CONSECUTIVE_SUCCESSES
    )


# ============================================================
# ANALYSIS EXPORT
# ============================================================

def export_analysis_metrics(run_dir):
    timeline = build_timeline(
        run_dir
    )

    output = (
        run_dir
        / "analysis_metrics.csv"
    )

    columns = [
        "step",
        "full_train_acc",
        "val_accuracy",
        "full_train_loss",
        "val_loss",
        "train_accuracy",
        "train_loss",
        "learning_rate",
        "epoch",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in timeline:
            writer.writerow(row)

    return output


# ============================================================
# RESULT CALCULATION
# ============================================================

def calculate_result(
    stage,
    run_dir,
    runtime_seconds,
    termination_reason,
):

    first_train99 = first_threshold_step(
        run_dir,
        "full_train_acc",
    )

    first_val99 = first_threshold_step(
        run_dir,
        "val_accuracy",
    )

    stable_joint99 = first_stable_joint_step(
        run_dir
    )

    (
        latest_step,
        latest_train,
        latest_val,
    ) = latest_status(
        run_dir
    )

    peak_train = peak_metric(
        run_dir,
        "full_train_acc",
    )

    peak_val = peak_metric(
        run_dir,
        "val_accuracy",
    )

    first_delay = None
    stable_delay = None

    if (
        first_train99 is not None
        and first_val99 is not None
    ):
        first_delay = (
            first_val99
            - first_train99
        )

    if (
        first_train99 is not None
        and stable_joint99 is not None
    ):
        stable_delay = (
            stable_joint99
            - first_train99
        )

    # --------------------------------------------------------
    # ROBUST VERDICT
    # --------------------------------------------------------

    if stable_joint99 is not None:

        if (
            stable_delay is not None
            and stable_delay >= 10_000
        ):
            verdict = (
                "STABLE_STRONG_DELAYED_GENERALISATION"
            )

        elif (
            stable_delay is not None
            and stable_delay > 0
        ):
            verdict = (
                "STABLE_DELAYED_GENERALISATION"
            )

        else:
            verdict = (
                "STABLE_GENERALISATION"
            )

    elif (
        first_train99 is not None
        and first_val99 is not None
    ):

        verdict = (
            "TRANSIENT_OR_UNSTABLE_THRESHOLD_CROSSING"
        )

    elif (
        first_train99 is not None
        and first_val99 is None
    ):

        verdict = (
            "MEMORISED_NOT_YET_GENERALISED"
        )

    else:

        verdict = (
            "INCONCLUSIVE"
        )

    return {
        "name": stage["name"],
        "description": stage["description"],

        "operator": stage["operator"],
        "seed": stage["seed"],

        "train_data_pct": TRAIN_DATA_PCT,
        "weight_decay": WEIGHT_DECAY,
        "max_lr": MAX_LR,

        "max_steps": stage["max_steps"],
        "last_logged_step": latest_step,

        "runtime_seconds": round(
            runtime_seconds,
            2,
        ),

        "termination_reason": (
            termination_reason
        ),

        "first_train99_step": (
            first_train99
        ),

        "first_val99_step": (
            first_val99
        ),

        "first_threshold_delay_steps": (
            first_delay
        ),

        "stable_joint99_step": (
            stable_joint99
        ),

        "stable_grokking_delay_steps": (
            stable_delay
        ),

        "peak_train_accuracy": (
            peak_train
        ),

        "peak_val_accuracy": (
            peak_val
        ),

        "final_train_accuracy": (
            latest_train
        ),

        "final_val_accuracy": (
            latest_val
        ),

        "verdict": verdict,

        "run_directory": str(
            run_dir
        ),
    }


def print_result(result):

    print()
    print("=" * 100)

    print(
        f"RESULT: {result['name']}"
    )

    print("=" * 100)

    print(
        f"Operation:             "
        f"{result['operator']}"
    )

    print(
        f"Seed:                  "
        f"{result['seed']}"
    )

    print(
        f"Stopped because:       "
        f"{result['termination_reason']}"
    )

    print(
        f"Runtime:               "
        f"{fmt_time(result['runtime_seconds'])}"
    )

    print(
        f"Last logged step:      "
        f"{result['last_logged_step']}"
    )

    print(
        f"First train >=99%:     "
        f"{result['first_train99_step']}"
    )

    print(
        f"First val >=99%:       "
        f"{result['first_val99_step']}"
    )

    print(
        f"Stable joint >=99%:    "
        f"{result['stable_joint99_step']}"
    )

    print(
        f"Stable grokking delay: "
        f"{result['stable_grokking_delay_steps']}"
    )

    print(
        f"Peak train accuracy:   "
        f"{result['peak_train_accuracy']}"
    )

    print(
        f"Peak val accuracy:     "
        f"{result['peak_val_accuracy']}"
    )

    print(
        f"Final train accuracy:  "
        f"{result['final_train_accuracy']}"
    )

    print(
        f"Final val accuracy:    "
        f"{result['final_val_accuracy']}"
    )

    print(
        f"Verdict:               "
        f"{result['verdict']}"
    )

    print("=" * 100)


# ============================================================
# PROCESS MANAGEMENT
# ============================================================

def stop_child_process(process):

    if process.poll() is not None:
        return

    process.terminate()

    try:

        process.wait(
            timeout=30
        )

    except subprocess.TimeoutExpired:

        process.kill()
        process.wait()


# ============================================================
# RUN ONE TASK
# ============================================================

def run_stage(
    stage,
    stage_number,
    total_stages,
    suite_dir,
    suite_deadline,
):

    run_dir = (
        suite_dir
        / stage["name"]
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    console_log = (
        run_dir
        / "training_console.log"
    )

    command = [
        sys.executable,

        str(
            ROOT
            / "scripts"
            / "train.py"
        ),

        "--max_steps",
        str(
            stage["max_steps"]
        ),

        "--gpu",
        "0",

        "--random_seed",
        str(
            stage["seed"]
        ),

        "--math_operator",
        stage["operator"],

        "--train_data_pct",
        str(
            TRAIN_DATA_PCT
        ),

        "--weight_decay",
        str(
            WEIGHT_DECAY
        ),

        "--max_lr",
        str(
            MAX_LR
        ),

        "--logdir",
        str(run_dir),
    ]

    print()
    print("#" * 100)

    print(
        f"STAGE "
        f"{stage_number}/{total_stages}: "
        f"{stage['name']}"
    )

    print(
        stage["description"]
    )

    print()

    print(
        f"Operator:             "
        f"{stage['operator']}"
    )

    print(
        f"Seed:                 "
        f"{stage['seed']}"
    )

    print(
        f"Training data:        "
        f"{TRAIN_DATA_PCT}%"
    )

    print(
        f"Weight decay:         "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Learning rate:        "
        f"{MAX_LR}"
    )

    print(
        f"Maximum steps:        "
        f"{stage['max_steps']:,}"
    )

    print(
        f"Stable threshold:     "
        f"train >= {ACCURACY_THRESHOLD:.0f}% "
        f"AND val >= {ACCURACY_THRESHOLD:.0f}% "
        f"for "
        f"{REQUIRED_CONSECUTIVE_SUCCESSES} "
        "consecutive evaluations"
    )

    print(
        f"Output:               "
        f"{run_dir}"
    )

    print("#" * 100)

    stage_start = time.time()

    env = os.environ.copy()

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    intentional_stop = False
    termination_reason = "UNKNOWN"

    with console_log.open(
        "w",
        encoding="utf-8",
    ) as log:

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

        try:

            while process.poll() is None:

                now = datetime.now()

                elapsed = (
                    time.time()
                    - stage_start
                )

                suite_time_left = (
                    suite_deadline
                    - now
                ).total_seconds()

                (
                    step,
                    train_acc,
                    val_acc,
                ) = latest_status(
                    run_dir
                )

                # ------------------------------------------------
                # STABLE GROKKING DETECTED
                # ------------------------------------------------

                if stage_has_grokked(
                    run_dir
                ):

                    print()
                    print()

                    print(
                        ">>> STABLE GROKKING TARGET REACHED."
                    )

                    print(
                        f">>> Train and validation >= "
                        f"{ACCURACY_THRESHOLD:.0f}% "
                        f"for "
                        f"{REQUIRED_CONSECUTIVE_SUCCESSES} "
                        "consecutive evaluations."
                    )

                    intentional_stop = True

                    termination_reason = (
                        "STABLE_ACCURACY_THRESHOLD_REACHED"
                    )

                    stop_child_process(
                        process
                    )

                    break

                # ------------------------------------------------
                # TOTAL OVERNIGHT SAFETY LIMIT
                # ------------------------------------------------

                if suite_time_left <= 0:

                    print()
                    print()

                    print(
                        ">>> Maximum overnight runtime reached."
                    )

                    intentional_stop = True

                    termination_reason = (
                        "SUITE_TIME_LIMIT"
                    )

                    stop_child_process(
                        process
                    )

                    break

                # ------------------------------------------------
                # TERMINAL STATUS
                # ------------------------------------------------

                if step > 0:

                    step_rate = (
                        step / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining_steps = max(
                        stage["max_steps"]
                        - step,
                        0,
                    )

                    estimated_remaining = (
                        remaining_steps
                        / step_rate
                        if step_rate > 0
                        else None
                    )

                    percent = (
                        100
                        * step
                        / stage["max_steps"]
                    )

                    train_text = (
                        f"{train_acc:6.2f}%"
                        if train_acc is not None
                        else "   N/A "
                    )

                    val_text = (
                        f"{val_acc:6.2f}%"
                        if val_acc is not None
                        else "   N/A "
                    )

                    success_count = (
                        consecutive_success_count(
                            run_dir
                        )
                    )

                    line = (
                        f"\r"
                        f"[{stage_number}/{total_stages}] "
                        f"{stage['name']:<25} | "
                        f"step "
                        f"{step:>8,}/"
                        f"{stage['max_steps']:,} "
                        f"({percent:5.1f}%) | "
                        f"train {train_text} | "
                        f"val {val_text} | "
                        f"{step_rate:5.1f} step/s | "
                        f"elapsed "
                        f"{fmt_time(elapsed)} | "
                        f"stage ETA "
                        f"{fmt_time(estimated_remaining)} | "
                        f"suite left "
                        f"{fmt_time(suite_time_left)} | "
                        f"stable "
                        f"{success_count}/"
                        f"{REQUIRED_CONSECUTIVE_SUCCESSES}"
                    )

                else:

                    line = (
                        f"\r"
                        f"[{stage_number}/{total_stages}] "
                        f"{stage['name']:<25} | "
                        "waiting for first metrics | "
                        f"elapsed "
                        f"{fmt_time(elapsed)} | "
                        f"suite left "
                        f"{fmt_time(suite_time_left)}"
                    )

                print(
                    line.ljust(280),
                    end="",
                    flush=True,
                )

                time.sleep(
                    POLL_SECONDS
                )

        except KeyboardInterrupt:

            print()
            print(
                ">>> User interruption received."
            )

            intentional_stop = True

            termination_reason = (
                "USER_INTERRUPTED"
            )

            stop_child_process(
                process
            )

            raise

    runtime_seconds = (
        time.time()
        - stage_start
    )

    print()

    # ========================================================
    # NATURAL COMPLETION
    # ========================================================

    if not intentional_stop:

        if process.returncode == 0:

            termination_reason = (
                "MAX_STEPS_REACHED"
            )

        else:

            termination_reason = (
                "PROCESS_ERROR"
            )

            print()
            print(
                f"ERROR: training process returned "
                f"{process.returncode}"
            )

            print(
                f"See: {console_log}"
            )

            try:

                lines = (
                    console_log
                    .read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    .splitlines()
                )

                print()
                print(
                    "Last 40 training-log lines:"
                )

                print(
                    "\n".join(
                        lines[-40:]
                    )
                )

            except Exception:
                pass

            raise RuntimeError(
                f"Experiment "
                f"{stage['name']} failed."
            )

    # ========================================================
    # SAVE RESULTS IMMEDIATELY
    # ========================================================

    export_analysis_metrics(
        run_dir
    )

    result = calculate_result(
        stage,
        run_dir,
        runtime_seconds,
        termination_reason,
    )

    result_file = (
        run_dir
        / "stage_result.json"
    )

    with result_file.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
        )

    print_result(
        result
    )

    return result


# ============================================================
# ENVIRONMENT RECORD
# ============================================================

def save_command_output(
    output_path,
    command,
):

    try:

        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        text = (
            result.stdout
            + "\n"
            + result.stderr
        )

    except Exception as exc:

        text = repr(exc)

    output_path.write_text(
        text,
        encoding="utf-8",
    )


def save_environment_info(
    suite_dir,
):

    environment_dir = (
        suite_dir
        / "environment"
    )

    environment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_command_output(
        environment_dir
        / "pip_freeze.txt",
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
    )

    save_command_output(
        environment_dir
        / "git_commit.txt",
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
    )

    save_command_output(
        environment_dir
        / "git_status.txt",
        [
            "git",
            "status",
            "--short",
        ],
    )

    save_command_output(
        environment_dir
        / "nvidia_smi.txt",
        [
            "nvidia-smi"
        ],
    )

    (
        environment_dir
        / "python.txt"
    ).write_text(
        sys.version,
        encoding="utf-8",
    )


# ============================================================
# SAVE SUITE SUMMARY
# ============================================================

def save_suite_summary(
    suite_dir,
    results,
    started,
    deadline,
):

    csv_path = (
        suite_dir
        / "summary.csv"
    )

    columns = [
        "name",
        "description",
        "operator",
        "seed",
        "train_data_pct",
        "weight_decay",
        "max_lr",
        "max_steps",
        "last_logged_step",
        "runtime_seconds",
        "termination_reason",
        "first_train99_step",
        "first_val99_step",
        "first_threshold_delay_steps",
        "stable_joint99_step",
        "stable_grokking_delay_steps",
        "peak_train_accuracy",
        "peak_val_accuracy",
        "final_train_accuracy",
        "final_val_accuracy",
        "verdict",
        "run_directory",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                result
            )

    json_path = (
        suite_dir
        / "summary.json"
    )

    payload = {
        "suite_name": (
            "overnight_task_comparison"
        ),

        "suite_started": (
            started.isoformat()
        ),

        "suite_finished": (
            datetime.now().isoformat()
        ),

        "suite_deadline": (
            deadline.isoformat()
        ),

        "fixed_configuration": {
            "train_data_pct": TRAIN_DATA_PCT,
            "weight_decay": WEIGHT_DECAY,
            "max_lr": MAX_LR,
            "accuracy_threshold": ACCURACY_THRESHOLD,
            "required_consecutive_successes":
                REQUIRED_CONSECUTIVE_SUCCESSES,
        },

        "stages": STAGES,

        "results": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

    return (
        csv_path,
        json_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    prevent_windows_sleep()

    started = datetime.now()

    suite_deadline = (
        started
        + timedelta(
            hours=MAX_SUITE_HOURS
        )
    )

    timestamp = (
        started.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    suite_dir = (
        ROOT
        / "runs"
        / (
            "task_comparison_overnight_"
            + timestamp
        )
    )

    suite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("=" * 100)
    print("OVERNIGHT GROKKING TASK COMPARISON")
    print("=" * 100)

    print(
        f"Started:        "
        f"{started.strftime('%Y-%m-%d %I:%M:%S %p')}"
    )

    print(
        f"Safety cutoff:  "
        f"{suite_deadline.strftime('%Y-%m-%d %I:%M:%S %p')}"
    )

    print(
        f"Output:         "
        f"{suite_dir}"
    )

    print()

    print(
        "FIXED CONFIGURATION"
    )

    print(
        f"Training data:  {TRAIN_DATA_PCT}%"
    )

    print(
        f"Weight decay:   {WEIGHT_DECAY}"
    )

    print(
        f"Learning rate:  {MAX_LR}"
    )

    print(
        "Architecture:   original OpenAI grok Transformer"
    )

    print()

    print(
        "The experimental variables tonight are:"
    )

    print(
        "  1. modular arithmetic operation"
    )

    print(
        "  2. random seed"
    )

    print()

    print(
        f"Each successful run stops after "
        f"{REQUIRED_CONSECUTIVE_SUCCESSES} "
        f"consecutive evaluations where both "
        f"train and validation >= "
        f"{ACCURACY_THRESHOLD:.0f}%."
    )

    print("=" * 100)

    save_environment_info(
        suite_dir
    )

    try:

        for index, stage in enumerate(
            STAGES,
            start=1,
        ):

            if datetime.now() >= suite_deadline:

                print()
                print(
                    "Overnight safety limit reached. "
                    "No additional experiments will start."
                )

                break

            result = run_stage(
                stage=stage,
                stage_number=index,
                total_stages=len(
                    STAGES
                ),
                suite_dir=suite_dir,
                suite_deadline=suite_deadline,
            )

            results.append(
                result
            )

            # Save an updated suite summary after EVERY stage.
            # This means completed results survive an unexpected reboot.
            save_suite_summary(
                suite_dir,
                results,
                started,
                suite_deadline,
            )

        (
            summary_csv,
            summary_json,
        ) = save_suite_summary(
            suite_dir,
            results,
            started,
            suite_deadline,
        )

        print()
        print("=" * 100)
        print("OVERNIGHT TASK COMPARISON COMPLETE")
        print("=" * 100)

        print(
            f"CSV summary:  "
            f"{summary_csv}"
        )

        print(
            f"JSON summary: "
            f"{summary_json}"
        )

        print()
        print("FINAL RESULTS")
        print()

        for result in results:

            print(
                f"{result['name']:<27} | "
                f"operator={result['operator']:<2} | "
                f"seed={result['seed']} | "
                f"train99="
                f"{result['first_train99_step']} | "
                f"val99="
                f"{result['first_val99_step']} | "
                f"stable99="
                f"{result['stable_joint99_step']} | "
                f"stable delay="
                f"{result['stable_grokking_delay_steps']} | "
                f"{result['verdict']}"
            )

        print()
        print("=" * 100)

    finally:

        restore_windows_sleep()


if __name__ == "__main__":
    main()