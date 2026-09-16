THESIS ARCHITECTURE TEST SUITE
==============================

FILES TO COPY
-------------
Copy these into your thesis repository:

  scripts\thesis_architecture_suite.py
  scripts\run_resumable_architecture_trial.py
  scripts\analyse_architecture_suite.py
  configs\thesis_architecture_suite.json

You can also put the two .ps1 launchers in the repository root if you want.

WHAT THIS RUNS
--------------
The suite uses division mod 97, 50% training data, LR=0.001, WD=0.1 as the
baseline and cleanly varies model design.

It runs:
  - baseline seeds 0,1,2
  - attention heads: 1,2,4,8 x 3 seeds
  - depth: 1,2,3,4,6 x 3 seeds at fixed d_model=128
  - width: 64,128,256 x 3 seeds at fixed depth=2
  - parameter-matched depth sweep x 3 seeds
  - ReLU vs GELU x 3 seeds
  - depth x weight-decay interaction:
      depths 1,2,4 x WD 0.03,0.1,0.3 x 3 seeds

Duplicate physical configurations are automatically deduplicated.

All trials permit up to 1,000,000 steps, but stop early only after a candidate
99%/99% crossing survives a 5,000-step confirmation window. This avoids the
old LR=0.003 problem where a transient 99% crossing was incorrectly treated as
stable grokking.

CRASH / REBOOT RESUME
---------------------
Each run writes:

  checkpoints\resume_latest.ckpt

atomically every 2,500 optimisation steps. The checkpoint contains model,
optimizer, scheduler, epoch and global-step state through PyTorch Lightning.

If training or Windows crashes:

  1. Reboot/open PowerShell.
  2. Run exactly the same command again:

       python .\scripts\thesis_architecture_suite.py

The master script reads the existing manifest, skips completed experiments,
and resumes the incomplete experiment from resume_latest.ckpt.

Do NOT delete runs\thesis_architecture_suite_v1 between resumes.

START THE FULL SUITE
--------------------
From C:\Users\Ajay\Documents\Thesis with the .venv activated:

  python .\scripts\thesis_architecture_suite.py

or:

  .\run_thesis_architecture_suite.ps1

RUN ONLY ONE PHASE
------------------
Useful for testing the new runner before committing to the full suite:

  python .\scripts\thesis_architecture_suite.py --only-phase baseline
  python .\scripts\thesis_architecture_suite.py --only-phase attention_heads
  python .\scripts\thesis_architecture_suite.py --only-phase depth_fixed_width
  python .\scripts\thesis_architecture_suite.py --only-phase width_fixed_depth
  python .\scripts\thesis_architecture_suite.py --only-phase parameter_matched_depth
  python .\scripts\thesis_architecture_suite.py --only-phase activation
  python .\scripts\thesis_architecture_suite.py --only-phase depth_x_regularisation

ANALYSE CURRENT COMPLETED RUNS
------------------------------
You do not need to wait for every experiment to finish:

  python .\scripts\analyse_architecture_suite.py .\runs\thesis_architecture_suite_v1

Outputs go to:

  runs\thesis_architecture_suite_v1\analysis\

IMPORTANT OUTPUT PER RUN
------------------------
  config.json                  exact experiment configuration
  model_info.json              actual parameter count
  evaluation_history.csv       train/val + global parameter norm evaluations
  analysis_metrics.csv         merged Lightning trajectory across all attempts
  trial_result.json            final classification and timing metrics
  checkpoints\resume_latest.ckpt
  attempts\version_*\metrics.csv

The main fields for the thesis are:
  first_train99_step
  first_val99_step
  confirmed_stable99_step
  confirmed_grokking_delay_steps
  collapsed_after_success
  parameter_count
  norm_at_train99
  norm_at_confirmed99
  norm_log_squared_ratio

GIT
---
Do not commit the .ckpt files. Add this to .gitignore if it is not already
covered:

  runs/**/*.ckpt

The CSV/JSON results, scripts, config and final plots are useful to commit.
