import argparse
import ctypes
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

from grok.training import TrainableTransformer


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MAX_STEPS = 1_000_000

TARGET_ACCURACY = 99.0

# Require two successful validation evaluations before stopping.
REQUIRED_SUCCESSES = 2

# Print one terminal progress update every 25 seconds.
STATUS_SECONDS = 25

# Save a recovery checkpoint every 10,000 steps.
CHECKPOINT_EVERY_STEPS = 10_000


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


def get_metric(trainer, name):
    value = trainer.callback_metrics.get(name)

    if value is None:
        return None

    if torch.is_tensor(value):
        return float(value.detach().cpu())

    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# MONITOR CALLBACK
# ============================================================

class RecoveryMonitor(Callback):

    def __init__(self, output_dir):
        super().__init__()

        self.output_dir = Path(output_dir)

        self.start_time = None
        self.start_step = None

        self.last_status_time = 0

        self.next_checkpoint_step = None

        self.latest_train = None
        self.latest_val = None

        self.success_count = 0

        self.train99_step = None
        self.val99_step = None

        self.stop_reason = "MAX_STEPS_REACHED"


    def on_train_start(
        self,
        trainer,
        pl_module,
    ):
        self.start_time = time.time()

        self.start_step = int(
            trainer.global_step
        )

        self.next_checkpoint_step = (
            (
                self.start_step
                // CHECKPOINT_EVERY_STEPS
            )
            + 1
        ) * CHECKPOINT_EVERY_STEPS

        print()
        print("=" * 95)
        print("TRIAL 3: CANONICAL GROKKING RECOVERY")
        print("=" * 95)

        print(
            f"Resumed from global step: "
            f"{self.start_step:,}"
        )

        print(
            f"Maximum step cap:          "
            f"{MAX_STEPS:,}"
        )

        print(
            f"Stop threshold:            "
            f"train >= {TARGET_ACCURACY:.0f}% "
            f"and val >= {TARGET_ACCURACY:.0f}%"
        )

        print(
            f"Required successful vals:  "
            f"{REQUIRED_SUCCESSES}"
        )

        print(
            f"Status update every:       "
            f"{STATUS_SECONDS} seconds"
        )

        print(
            f"Safety checkpoint every:   "
            f"{CHECKPOINT_EVERY_STEPS:,} steps"
        )

        print("=" * 95)


    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        step = int(
            trainer.global_step
        )

        # ----------------------------------------------------
        # SAFETY CHECKPOINT
        # ----------------------------------------------------

        if (
            self.next_checkpoint_step is not None
            and step >= self.next_checkpoint_step
        ):

            checkpoint_dir = (
                self.output_dir
                / "checkpoints"
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint_path = (
                checkpoint_dir
                / f"recovery_step_{step}.ckpt"
            )

            trainer.save_checkpoint(
                str(checkpoint_path)
            )

            print()
            print(
                f">>> Safety checkpoint saved: "
                f"{checkpoint_path.name}"
            )

            while (
                self.next_checkpoint_step
                <= step
            ):
                self.next_checkpoint_step += (
                    CHECKPOINT_EVERY_STEPS
                )

        # ----------------------------------------------------
        # TERMINAL STATUS
        # ----------------------------------------------------

        now = time.time()

        if (
            now - self.last_status_time
            < STATUS_SECONDS
        ):
            return

        self.last_status_time = now

        elapsed = (
            now - self.start_time
        )

        steps_since_resume = max(
            step - self.start_step,
            0,
        )

        rate = (
            steps_since_resume / elapsed
            if elapsed > 0
            else 0
        )

        remaining_steps = max(
            MAX_STEPS - step,
            0,
        )

        eta = (
            remaining_steps / rate
            if rate > 0
            else None
        )

        percent = (
            100
            * step
            / MAX_STEPS
        )

        train_text = (
            f"{self.latest_train:6.2f}%"
            if self.latest_train is not None
            else "   N/A "
        )

        val_text = (
            f"{self.latest_val:6.2f}%"
            if self.latest_val is not None
            else "   N/A "
        )

        line = (
            f"\r"
            f"TRIAL 3 | "
            f"step {step:>9,}/{MAX_STEPS:,} "
            f"({percent:5.1f}%) | "
            f"train {train_text} | "
            f"val {val_text} | "
            f"{rate:5.1f} step/s | "
            f"elapsed {fmt_time(elapsed)} | "
            f"ETA {fmt_time(eta)} | "
            f"success {self.success_count}/"
            f"{REQUIRED_SUCCESSES}"
        )

        print(
            line.ljust(210),
            end="",
            flush=True,
        )


    def on_validation_end(
        self,
        trainer,
        pl_module,
    ):
        # Ignore Lightning's startup sanity check.
        if getattr(
            trainer,
            "sanity_checking",
            False,
        ):
            return

        step = int(
            trainer.global_step
        )

        train_acc = get_metric(
            trainer,
            "full_train_acc",
        )

        val_acc = get_metric(
            trainer,
            "val_accuracy",
        )

        if train_acc is not None:
            self.latest_train = train_acc

        if val_acc is not None:
            self.latest_val = val_acc

        if (
            train_acc is not None
            and train_acc >= TARGET_ACCURACY
            and self.train99_step is None
        ):
            self.train99_step = step

        if (
            val_acc is not None
            and val_acc >= TARGET_ACCURACY
            and self.val99_step is None
        ):
            self.val99_step = step

        # Do not print every validation callback.
        # We only update the stored values here.
        # The terminal status line prints every STATUS_SECONDS.

        if (
            train_acc is not None
            and val_acc is not None
            and train_acc >= TARGET_ACCURACY
            and val_acc >= TARGET_ACCURACY
        ):
            self.success_count += 1

        else:
            self.success_count = 0

        if (
            self.success_count
            >= REQUIRED_SUCCESSES
        ):
            print()
            print()

            print(
                f">>> GROKKING TARGET CONFIRMED "
                f"at step {step:,}"
            )

            print(
                f">>> Train accuracy:      "
                f"{train_acc:.2f}%"
            )

            print(
                f">>> Validation accuracy: "
                f"{val_acc:.2f}%"
            )

            self.stop_reason = (
                "TRAIN_AND_VALIDATION_99"
            )

            trainer.should_stop = True


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint
    ).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    prevent_windows_sleep()

    try:

        # ----------------------------------------------------
        # LOAD CHECKPOINT INFORMATION
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        start_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )

        start_epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        print()
        print(
            f"Checkpoint global step: "
            f"{start_step:,}"
        )

        print(
            f"Checkpoint epoch:       "
            f"{start_epoch:,}"
        )

        print(
            f"Optimizer states:       "
            f"{len(checkpoint.get('optimizer_states', []))}"
        )

        print(
            f"LR schedulers:          "
            f"{len(checkpoint.get('lr_schedulers', []))}"
        )

        # ----------------------------------------------------
        # NEW OUTPUT DIRECTORY
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = (
            ROOT
            / "runs"
            / f"trial3_canonical_resumed_{timestamp}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_dir = (
            output_dir
            / "checkpoints"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # RESTORE ORIGINAL HYPERPARAMETERS
        # ----------------------------------------------------

        hp_dict = checkpoint[
            "hyper_parameters"
        ]

        hp = argparse.Namespace(
            **hp_dict
        )

        # Keep the experiment configuration identical.
        # Only change output locations and maximum run length.

        hp.logdir = str(
            output_dir
        )

        hp.checkpoint_path = str(
            checkpoint_dir
        )

        hp.max_steps = MAX_STEPS
        hp.gpu = 0

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = TrainableTransformer(
            hp
        ).float()

        logger = CSVLogger(
            str(output_dir)
        )

        monitor = RecoveryMonitor(
            output_dir
        )

        # ----------------------------------------------------
        # TRAINER
        # ----------------------------------------------------

        trainer = Trainer(

            max_steps=MAX_STEPS,

            # Allow our early-stop condition to work.
            min_steps=0,

            max_epochs=int(1e8),

            val_check_interval=1,

            profiler=False,

            logger=logger,

            log_every_n_steps=1,

            flush_logs_every_n_steps=1000,

            gpus=[0],

            resume_from_checkpoint=str(
                checkpoint_path
            ),

            callbacks=[
                monitor
            ],

            # We save our own frequent checkpoints.
            checkpoint_callback=False,

            # Disable Lightning's progress bar.
            progress_bar_refresh_rate=0,

            # No startup sanity-validation pass.
            num_sanity_val_steps=0,
        )

        start_time = time.time()

        trainer.fit(
            model=model
        )

        runtime = (
            time.time()
            - start_time
        )

        final_step = int(
            trainer.global_step
        )

        # If training ended below the cap and grokking did not
        # trigger the stop, it was probably interrupted.
        if (
            final_step < MAX_STEPS
            and monitor.stop_reason
            == "MAX_STEPS_REACHED"
        ):
            monitor.stop_reason = (
                "INTERRUPTED_OR_EARLY_TERMINATION"
            )

        # ----------------------------------------------------
        # FINAL CHECKPOINT
        # ----------------------------------------------------

        final_checkpoint = (
            checkpoint_dir
            / f"final_step_{final_step}.ckpt"
        )

        trainer.save_checkpoint(
            str(final_checkpoint)
        )

        final_train = get_metric(
            trainer,
            "full_train_acc",
        )

        final_val = get_metric(
            trainer,
            "val_accuracy",
        )

        result = {
            "source_checkpoint": str(
                checkpoint_path
            ),
            "resume_start_step": start_step,
            "resume_start_epoch": start_epoch,
            "final_step": final_step,
            "runtime_seconds": runtime,
            "termination_reason": (
                monitor.stop_reason
            ),
            "train99_during_resume": (
                monitor.train99_step
            ),
            "val99_step": (
                monitor.val99_step
            ),
            "final_train_accuracy": (
                final_train
            ),
            "final_validation_accuracy": (
                final_val
            ),
            "output_directory": str(
                output_dir
            ),
            "final_checkpoint": str(
                final_checkpoint
            ),
        }

        result_file = (
            output_dir
            / "recovery_result.json"
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

        print()
        print()
        print("=" * 95)
        print("TRIAL 3 RECOVERY FINISHED")
        print("=" * 95)

        print(
            f"Resumed from step: "
            f"{start_step:,}"
        )

        print(
            f"Finished at step:  "
            f"{final_step:,}"
        )

        print(
            f"Final train acc:   "
            f"{final_train}"
        )

        print(
            f"Final val acc:     "
            f"{final_val}"
        )

        print(
            f"Stop reason:       "
            f"{monitor.stop_reason}"
        )

        print(
            f"Results file:      "
            f"{result_file}"
        )

        print(
            f"Final checkpoint:  "
            f"{final_checkpoint}"
        )

        print("=" * 95)

    finally:

        restore_windows_sleep()


if __name__ == "__main__":
    main()