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

# How often the visible terminal status updates.
POLL_SECONDS = 10

# Power et al. commonly use 99% as the generalisation threshold.
ACCURACY_THRESHOLD = 99.0

# Require more than one successful evaluation so one anomalous
# measurement does not immediately terminate the experiment.
REQUIRED_SUCCESSFUL_EVALUATIONS = 2

# Absolute runtime safeguard.
MAX_SUITE_HOURS = 6.75

# We want everything stopped before you check it at 9 pm.
STOP_AT_HOUR = 20
STOP_AT_MINUTE = 50


# Tonight's baseline verification suite.
#
# IMPORTANT:
# max_steps is a SAFETY CAP.
# Runs stop early once BOTH train and validation accuracy have
# reached >= 99% on two distinct logged evaluation points.
STAGES = [
    {
        "name": "A_regularised_50pct",
        "description": (
            "Positive control: division mod 97, "
            "50% training data, weight decay 1"
        ),
        "max_steps": 50_000,
        "train_pct": 50,
        "weight_decay": 1,
        "seed": 0,
    },
    {
        "name": "B_high_data_80pct",
        "description": (
            "High-data control: division mod 97, "
            "80% training data, weight decay 1"
        ),
        "max_steps": 30_000,
        "train_pct": 80,
        "weight_decay": 1,
        "seed": 0,
    },
    {
        "name": "C_canonical_no_decay",
        "description": (
            "Canonical delayed-generalisation condition: "
            "division mod 97, 50% training data, NO weight decay"
        ),
        "max_steps": 1_000_000,
        "train_pct": 50,
        "weight_decay": 0,
        "seed": 0,
    },
]


# ============================================================
# WINDOWS SLEEP PREVENTION
# ============================================================

def prevent_windows_sleep():
    """
    Prevent Windows from putting the PC to sleep while the suite runs.
    The display itself is still allowed to turn off.
    """
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
# GENERAL HELPERS
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


def calculate_deadline():
    """
    Stop at whichever arrives first:
      - 8:50 pm tonight
      - MAX_SUITE_HOURS from launch
    """
    now = datetime.now()

    evening_deadline = now.replace(
        hour=STOP_AT_HOUR,
        minute=STOP_AT_MINUTE,
        second=0,
        microsecond=0,
    )

    runtime_deadline = now + timedelta(
        hours=MAX_SUITE_HOURS
    )

    if evening_deadline <= now:
        raise RuntimeError(
            "The configured 8:50 pm deadline has already passed."
        )

    return min(
        evening_deadline,
        runtime_deadline,
    )


# ============================================================
# LIGHTNING CSV READING
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
        # Lightning might be writing to the file at this instant.
        return []


def build_timeline(run_dir):
    """
    Lightning sometimes stores train and validation metrics on
    separate rows for the same step.

    This merges all rows belonging to a step into one clean row.
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
    threshold,
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
            and value >= threshold
        ):
            return row["step"]

    return None


def jointly_successful_steps(run_dir):
    """
    Return all logged steps where BOTH:
        train accuracy >= threshold
        validation accuracy >= threshold
    """

    timeline = build_timeline(
        run_dir
    )

    successes = []

    for row in timeline:

        train_acc = row.get(
            "full_train_acc"
        )

        val_acc = row.get(
            "val_accuracy"
        )

        if (
            train_acc is None
            or val_acc is None
        ):
            continue

        if (
            train_acc >= ACCURACY_THRESHOLD
            and val_acc >= ACCURACY_THRESHOLD
        ):
            successes.append(
                row["step"]
            )

    return successes


def stage_has_grokked(run_dir):
    successes = jointly_successful_steps(
        run_dir
    )

    return (
        len(successes)
        >= REQUIRED_SUCCESSFUL_EVALUATIONS
    )


# ============================================================
# CLEAN ANALYSIS FILE
# ============================================================

def export_analysis_metrics(run_dir):
    """
    Produce a clean, dense CSV for later thesis analysis.

    This is easier to plot than Lightning's sparse metrics.csv.
    """

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

    train99 = first_threshold_step(
        run_dir,
        "full_train_acc",
        ACCURACY_THRESHOLD,
    )

    val99 = first_threshold_step(
        run_dir,
        "val_accuracy",
        ACCURACY_THRESHOLD,
    )

    (
        latest_step,
        latest_train,
        latest_val,
    ) = latest_status(
        run_dir
    )

    delay = None
    timing_ratio = None

    if (
        train99 is not None
        and val99 is not None
    ):

        delay = (
            val99 - train99
        )

        timing_ratio = (
            val99
            / max(train99, 1)
        )

    # This is OUR convenient classification,
    # not a definition claimed by Power et al.
    if (
        train99 is not None
        and val99 is not None
    ):

        if (
            delay >= 10_000
            and timing_ratio >= 10
        ):
            verdict = (
                "STRONG_DELAYED_GENERALISATION"
            )

        elif delay > 0:
            verdict = (
                "DELAYED_GENERALISATION"
            )

        else:
            verdict = (
                "NO_MEASURABLE_DELAY"
            )

    elif (
        train99 is not None
        and val99 is None
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
        "seed": stage["seed"],
        "train_data_pct": stage["train_pct"],
        "weight_decay": stage["weight_decay"],
        "max_steps": stage["max_steps"],
        "last_logged_step": latest_step,
        "runtime_seconds": round(
            runtime_seconds,
            2,
        ),
        "termination_reason": (
            termination_reason
        ),
        "train99_step": train99,
        "val99_step": val99,
        "grokking_delay_steps": delay,
        "val_to_train_timing_ratio": (
            timing_ratio
        ),
        "latest_train_accuracy": (
            latest_train
        ),
        "latest_val_accuracy": (
            latest_val
        ),
        "verdict": verdict,
        "run_directory": str(
            run_dir
        ),
    }


def print_result(result):

    print()
    print("=" * 82)

    print(
        f"RESULT: {result['name']}"
    )

    print("=" * 82)

    print(
        f"Stopped because:      "
        f"{result['termination_reason']}"
    )

    print(
        f"Runtime:              "
        f"{fmt_time(result['runtime_seconds'])}"
    )

    print(
        f"Last logged step:     "
        f"{result['last_logged_step']}"
    )

    print(
        f"Latest train acc:     "
        f"{result['latest_train_accuracy']}"
    )

    print(
        f"Latest validation:    "
        f"{result['latest_val_accuracy']}"
    )

    print(
        f"T_train99:            "
        f"{result['train99_step']}"
    )

    print(
        f"T_val99:              "
        f"{result['val99_step']}"
    )

    print(
        f"Grokking delay:       "
        f"{result['grokking_delay_steps']}"
    )

    print(
        f"Val/train time ratio: "
        f"{result['val_to_train_timing_ratio']}"
    )

    print(
        f"Verdict:              "
        f"{result['verdict']}"
    )

    print("=" * 82)


# ============================================================
# STOP CHILD PROCESS
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
# ONE EXPERIMENT STAGE
# ============================================================

def run_stage(
    stage,
    stage_number,
    total_stages,
    suite_dir,
    deadline,
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
        "/",
        "--train_data_pct",
        str(
            stage["train_pct"]
        ),
        "--weight_decay",
        str(
            stage["weight_decay"]
        ),
        "--max_lr",
        "0.001",
        "--logdir",
        str(run_dir),
    ]

    print()
    print("#" * 82)

    print(
        f"STAGE "
        f"{stage_number}/{total_stages}: "
        f"{stage['name']}"
    )

    print(
        stage["description"]
    )

    print(
        f"Maximum step cap: "
        f"{stage['max_steps']:,}"
    )

    print(
        f"Early-stop threshold: "
        f"train >= {ACCURACY_THRESHOLD:.0f}% "
        f"AND validation >= "
        f"{ACCURACY_THRESHOLD:.0f}%"
    )

    print(
        f"Required successful evaluations: "
        f"{REQUIRED_SUCCESSFUL_EVALUATIONS}"
    )

    print(
        f"Output: {run_dir}"
    )

    print("#" * 82)

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

            while (
                process.poll()
                is None
            ):

                now = datetime.now()

                elapsed = (
                    time.time()
                    - stage_start
                )

                seconds_to_deadline = (
                    deadline - now
                ).total_seconds()

                (
                    step,
                    train_acc,
                    val_acc,
                ) = latest_status(
                    run_dir
                )

                # --------------------------------------------
                # SUCCESSFUL GROKKING STOP
                # --------------------------------------------

                if stage_has_grokked(
                    run_dir
                ):

                    print()
                    print()

                    print(
                        ">>> Train and validation "
                        f"have both reached "
                        f"{ACCURACY_THRESHOLD:.0f}% "
                        f"on "
                        f"{REQUIRED_SUCCESSFUL_EVALUATIONS} "
                        "logged evaluations."
                    )

                    print(
                        ">>> Stage complete."
                    )

                    intentional_stop = True

                    termination_reason = (
                        "ACCURACY_THRESHOLD_REACHED"
                    )

                    stop_child_process(
                        process
                    )

                    break

                # --------------------------------------------
                # EVENING DEADLINE
                # --------------------------------------------

                if (
                    seconds_to_deadline
                    <= 0
                ):

                    print()
                    print()

                    print(
                        ">>> Evening suite "
                        "deadline reached."
                    )

                    intentional_stop = True

                    termination_reason = (
                        "SUITE_TIME_LIMIT"
                    )

                    stop_child_process(
                        process
                    )

                    break

                # --------------------------------------------
                # TERMINAL STATUS
                # --------------------------------------------

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

                    if (
                        train_acc
                        is not None
                    ):
                        train_text = (
                            f"{train_acc:6.2f}%"
                        )
                    else:
                        train_text = (
                            "   N/A "
                        )

                    if (
                        val_acc
                        is not None
                    ):
                        val_text = (
                            f"{val_acc:6.2f}%"
                        )
                    else:
                        val_text = (
                            "   N/A "
                        )

                    success_count = len(
                        jointly_successful_steps(
                            run_dir
                        )
                    )

                    line = (
                        f"\r"
                        f"[{stage_number}/{total_stages}] "
                        f"{stage['name']:<24} | "
                        f"step "
                        f"{step:>9,}/"
                        f"{stage['max_steps']:,} "
                        f"({percent:5.1f}%) | "
                        f"train {train_text} | "
                        f"val {val_text} | "
                        f"{step_rate:5.1f} step/s | "
                        f"elapsed "
                        f"{fmt_time(elapsed)} | "
                        f"ETA "
                        f"{fmt_time(estimated_remaining)} | "
                        f"deadline "
                        f"{fmt_time(seconds_to_deadline)} | "
                        f"success "
                        f"{success_count}/"
                        f"{REQUIRED_SUCCESSFUL_EVALUATIONS}"
                    )

                else:

                    line = (
                        f"\r"
                        f"[{stage_number}/{total_stages}] "
                        f"{stage['name']:<24} | "
                        "waiting for first "
                        "logged metrics | "
                        f"elapsed "
                        f"{fmt_time(elapsed)}"
                    )

                print(
                    line.ljust(240),
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

    # Natural completion.
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
                f"ERROR: child process "
                f"returned "
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

    # --------------------------------------------
    # EXPORT CLEAN FILES
    # --------------------------------------------

    export_analysis_metrics(
        run_dir
    )

    result = calculate_result(
        stage,
        run_dir,
        runtime_seconds,
        termination_reason,
    )

    with (
        run_dir
        / "stage_result.json"
    ).open(
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
# SAVE REPRODUCIBILITY INFORMATION
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
# FINAL SUITE SUMMARY
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
        "seed",
        "train_data_pct",
        "weight_decay",
        "max_steps",
        "last_logged_step",
        "runtime_seconds",
        "termination_reason",
        "train99_step",
        "val99_step",
        "grokking_delay_steps",
        "val_to_train_timing_ratio",
        "latest_train_accuracy",
        "latest_val_accuracy",
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
        "suite_started": (
            started.isoformat()
        ),
        "suite_finished": (
            datetime.now().isoformat()
        ),
        "deadline": (
            deadline.isoformat()
        ),
        "accuracy_threshold": (
            ACCURACY_THRESHOLD
        ),
        "required_successful_evaluations": (
            REQUIRED_SUCCESSFUL_EVALUATIONS
        ),
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

    deadline = calculate_deadline()

    timestamp = (
        started.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    suite_dir = (
        ROOT
        / "runs"
        / (
            "grok_verification_"
            + timestamp
        )
    )

    suite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("=" * 82)
    print("GROKKING BASELINE VERIFICATION")
    print("=" * 82)

    print(
        f"Started:  "
        f"{started.strftime('%I:%M:%S %p')}"
    )

    print(
        f"Deadline: "
        f"{deadline.strftime('%I:%M:%S %p')}"
    )

    print(
        f"Output:   {suite_dir}"
    )

    print()

    print(
        "Each run stops as soon as train AND validation "
        f"reach >= {ACCURACY_THRESHOLD:.0f}% "
        f"for {REQUIRED_SUCCESSFUL_EVALUATIONS} "
        "logged evaluations."
    )

    print(
        "Maximum step counts are safety caps only."
    )

    print("=" * 82)

    save_environment_info(
        suite_dir
    )

    try:

        for index, stage in enumerate(
            STAGES,
            start=1,
        ):

            seconds_left = (
                deadline
                - datetime.now()
            ).total_seconds()

            if seconds_left <= 0:

                print()
                print(
                    "Deadline reached. "
                    "No further experiments "
                    "will start."
                )

                break

            result = run_stage(
                stage=stage,
                stage_number=index,
                total_stages=len(
                    STAGES
                ),
                suite_dir=suite_dir,
                deadline=deadline,
            )

            results.append(
                result
            )

        (
            summary_csv,
            summary_json,
        ) = save_suite_summary(
            suite_dir,
            results,
            started,
            deadline,
        )

        print()
        print("=" * 82)
        print("SUITE COMPLETE")
        print("=" * 82)

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
                f"{result['name']}: "
                f"T_train99="
                f"{result['train99_step']}, "
                f"T_val99="
                f"{result['val99_step']}, "
                f"delay="
                f"{result['grokking_delay_steps']}, "
                f"verdict="
                f"{result['verdict']}"
            )

        print()
        print("=" * 82)

    finally:

        restore_windows_sleep()


if __name__ == "__main__":
    main()