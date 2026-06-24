# Automatic Training Pipeline

A handover-friendly machine-learning training pipeline for machine status prediction.

This project is the **Phase 4 model training and artifact export pipeline** for machine event data. It takes historical machine status logs, checks whether the data is usable, trains machine-learning models, evaluates the models, and exports model artifacts that can later be used by the Phase 5 prediction/inference system.

The goal is simple:

> Given machine event history, predict **what status will happen next** and **how long until it happens**.

The project is designed so that a new user can run it notebook by notebook without needing deep machine-learning knowledge.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Simple Explanation](#simple-explanation)
3. [Project Workflow](#project-workflow)
4. [Input Dataset Requirement](#input-dataset-requirement)
5. [Repository Files](#repository-files)
6. [Installation / Environment Setup](#installation--environment-setup)
7. [How to Use the Pipeline](#how-to-use-the-pipeline)
8. [Step 1 — Pattern Checking Gate](#step-1--pattern-checking-gate)
9. [Step 1.5 — Status Name Check](#step-15--status-name-check)
10. [Step 2 — Automatic Baseline Test](#step-2--automatic-baseline-test)
11. [Step 3 — Automatic Training Pipeline](#step-3--automatic-training-pipeline)
12. [Important Configuration Fields](#important-configuration-fields)
13. [Understanding Normal Statuses and Target Statuses](#understanding-normal-statuses-and-target-statuses)
14. [Understanding the Model Outputs](#understanding-the-model-outputs)
15. [Understanding the Exported Artifacts](#understanding-the-exported-artifacts)
16. [How to Decide Whether a Model Is Good Enough](#how-to-decide-whether-a-model-is-good-enough)
17. [Common Problems and Fixes](#common-problems-and-fixes)
18. [Handover Notes](#handover-notes)
19. [Glossary](#glossary)

---

# What This Project Does

This project trains models from machine status history.

The input is a table of machine events. Each row says:

* when the event happened
* which machine it happened on
* what the machine status was
* which process the machine belongs to

Example:

| occurred            | mc_no | mc_status   | process |
| ------------------- | ----- | ----------- | ------- |
| 2026-05-05 10:12:44 | mc11a | run         | gd      |
| 2026-05-05 10:12:48 | mc11a | mc_waitpart | gd      |
| 2026-05-05 10:13:30 | mc11a | mc_alarm    | gd      |

The pipeline learns from this history and creates models that can answer:

1. **How long until the next important machine status happens?**
2. **What will that next status be?**
3. **How confident is the model about the predicted next status?**

The final output of this project is a set of model artifact files:

```text
lgbm_quantile_p50_final.pkl
lgbm_quantile_p90_final.pkl
lgbm_next_type.pkl
artifacts_phase4.pkl
```

These files are used later by the Phase 5 prediction service.

---

# Simple Explanation

Think of each machine as having a timeline.

Example:

```text
10:00:00   run
10:01:00   run
10:03:00   mc_waitpart
10:05:00   run
10:08:00   mc_alarm
```

The pipeline converts this into a training problem:

```text
At 10:00:00, what is the next important event?
Answer: mc_waitpart, happening in 180 seconds.

At 10:05:00, what is the next important event?
Answer: mc_alarm, happening in 180 seconds.
```

Then the model learns patterns like:

* this machine usually gets `mc_waitpart` after this kind of sequence
* this status often leads to `mc_alarm`
* some machines have shorter or longer time gaps than others
* recent status changes are more important than old ones

The model does **not** know the true physical root cause of a problem because the dataset only contains event logs. It does not contain sensor values such as vibration, current, temperature, pressure, motor load, etc.

So this project should be understood as:

> A machine event-log prediction system, not a full physical root-cause diagnosis system.

---

# Project Workflow

The pipeline has 4 notebooks.

Run them in this order:

```text
1_pattern-checking-gate.ipynb
        ↓
1_5_status_name_check.ipynb
        ↓
2_automatic_baseline_test.ipynb
        ↓
3_automatic_training_pipeline.ipynb
```

Simple view:

| Step     | Notebook                              | Purpose                                                     |
| -------- | ------------------------------------- | ----------------------------------------------------------- |
| Step 1   | `1_pattern-checking-gate.ipynb`       | Check whether the dataset has the correct columns           |
| Step 1.5 | `1_5_status_name_check.ipynb`         | Check what machine status names exist in the dataset        |
| Step 2   | `2_automatic_baseline_test.ipynb`     | Test simple baseline performance before doing full training |
| Step 3   | `3_automatic_training_pipeline.ipynb` | Train final models and export artifacts                     |

---

# Input Dataset Requirement

The dataset must contain these exact required columns:

```text
occurred
mc_no
mc_status
process
```

Optional legacy column:

```text
registered
```

## Required Columns

### `occurred`

The real timestamp of the machine event.

Example:

```text
2026-05-05 10:12:44
```

This column is the true event time used by the model.

### `mc_no`

The machine number or machine ID.

Example:

```text
mc11a
ffl-07-2
or28h
```

### `mc_status`

The status of the machine at that timestamp.

Example:

```text
run
mc_alarm
mc_fullwork
mc_waitpart
mc_stop
no_work
alarm
```

### `process`

The process name or process ID.

Example:

```text
gd
assy
apb
```

This project trains one artifact bundle per process.

---

## Optional Column: `registered`

Some old datasets may contain:

```text
registered
```

This column is allowed, but it is not used by the model.

If the dataset contains `registered`, the notebook will show a warning.

Use `occurred` as the real timestamp.

---

# Repository Files

The main files are:

```text
Automatic-Training-Pipeline/
│
├── 1_pattern-checking-gate.ipynb
├── 1_5_status_name_check.ipynb
├── 2_automatic_baseline_test.ipynb
├── 3_automatic_training_pipeline.ipynb
│
└── README.md
```

After running the final notebook, an artifact folder will be created.

Example:

```text
artifacts_phase4/
│
├── lgbm_quantile_p50_final.pkl
├── lgbm_quantile_p90_final.pkl
├── lgbm_next_type.pkl
├── artifacts_phase4.pkl
├── artifacts_phase4_summary.json
├── model_card_phase4.json
├── model_card_phase4.md
├── phase5_integration_checklist.md
├── artifact_manifest.json
├── feature_importance_p50.csv
├── feature_importance_p90.csv
├── feature_importance_next_type.csv
├── feature_importance_combined.csv
├── feature_importance_aggregate.csv
└── phase4_artifacts_<process_id>.zip
```

The most important files are the four required Phase 5 files:

```text
lgbm_quantile_p50_final.pkl
lgbm_quantile_p90_final.pkl
lgbm_next_type.pkl
artifacts_phase4.pkl
```

---

# Installation / Environment Setup

This project is notebook-based. It is intended to be run in Jupyter Notebook or JupyterLab.

## Recommended Python Version

```text
Python 3.10 or newer
```

## Install Required Packages

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install pandas numpy scikit-learn lightgbm joblib jupyter openpyxl pyarrow
```

Optional packages depending on file format:

```bash
pip install fastparquet
```

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

# How to Use the Pipeline

## Full Operating Flow

1. Put your dataset file in the same folder as the notebooks, or prepare the full file path.
2. Run `1_pattern-checking-gate.ipynb`.
3. If accepted, run `1_5_status_name_check.ipynb`.
4. Use the status names from Step 1.5 to fill `normal_statuses`, `target_statuses`, and `status_alias_map`.
5. Run `2_automatic_baseline_test.ipynb`.
6. Check result at `P4-19`.
7. If the baseline result is reasonable, run `3_automatic_training_pipeline.ipynb`.
8. Check final ETA metrics at `P4-30`.
9. Check final next-status classification metrics at `P4-31`.
10. Check final acceptance gate at `P4-32`.
11. Check artifact save output at `P4-34`.
12. Check Phase 5 contract validation at `P4-37`.
13. Check final ZIP bundle at `P4-40`.

---

# Step 1: Pattern Checking Gate

Notebook:

```text
1_pattern-checking-gate.ipynb
```

## Purpose

This notebook checks whether the dataset has the correct column pattern.

The dataset is accepted only if it contains:

```text
occurred
mc_no
mc_status
process
```

It is also accepted if it contains the legacy optional column:

```text
registered
```

So these are accepted:

```text
occurred, mc_no, mc_status, process
```

or:

```text
occurred, mc_no, mc_status, process, registered
```

Anything else is rejected.

## How to Use

Open the notebook and set:

```python
DATASET_PATH = "your_file_name.csv"
```

If the file is in the same folder as the notebook, just write the file name:

```python
DATASET_PATH = "data.csv"
```

If the file is somewhere else, use the full path:

```python
DATASET_PATH = "/home/user/data/data.csv"
```

Then run the notebook.

## Supported File Formats

The pattern-checking notebook supports:

```text
CSV
TSV / TXT
Parquet
Excel
JSON
Feather
Pickle
```

## Result

If the dataset is valid, the notebook will show that the dataset is accepted.

If invalid, the notebook will show:

* required columns
* your actual columns
* missing columns
* unexpected columns
* example expected table

Do not continue to the next step until this notebook accepts the dataset.

---

# Step 1.5: Status Name Check

Notebook:

```text
1_5_status_name_check.ipynb
```

## Purpose

This notebook shows all machine status names in the dataset.

This is important because before training, you must define:

* which statuses are normal
* which statuses are targets
* whether some status names need aliases

## How to Use

Open the notebook and set:

```python
DATA_PATH = "your_file_name.csv"
```

Then run the notebook.

The notebook prints:

```text
Raw shape
Statuses
Time range
Status counts
First few rows of the dataset
```

## Important Note

By default, this notebook reads CSV:

```python
df_raw = pd.read_csv(DATA_PATH)
```

If your file is Excel, change it to:

```python
df_raw = pd.read_excel(DATA_PATH)
```

If your file is Parquet, change it to:

```python
df_raw = pd.read_parquet(DATA_PATH)
```

## What to Look For

Look at the printed status names.

Example:

```text
run
mc_run
mc_alarm
mc_fullWork
mc_waitpart
mc_stop
```

Use this information when setting the config in the next notebooks.

---

# Step 2: Automatic Baseline Test

Notebook:

```text
2_automatic_baseline_test.ipynb
```

## Purpose

This notebook checks whether the dataset is worth continuing.

It is the first half of the full training pipeline.

It does not export final production artifacts.

It answers:

> Can simple baseline methods already predict the next event reasonably well?

A baseline is a simple method such as:

* use the global median time
* use the machine median time
* use the status median time
* use the machine + status median time

If even the baseline is terrible, the dataset may be too noisy or not useful enough.

## How to Use

Open the notebook.

Go to:

```text
P4-1 — Define AutoTrainingConfig
```

Edit the config.

Then click:

```text
Run All
```

Check the result at:

```text
P4-19 — Baseline evaluation report
```

## Important P4-1 Fields

Example:

```python
process_id = "gd"
data_path = "status_nht_converted.parquet"
normal_statuses = ["run", "mc_run"]
target_statuses = "auto"
```

For an old ASSY-style dataset:

```python
process_id = "assy"
data_path = "/home/micml/Documents/TestML/REAL/MCSTATUS_ASSY.CSV"
normal_statuses = ["run"]
target_statuses = "auto"
```

## What `target_statuses = "auto"` Means

It means:

> Every status that is not normal becomes a target.

Example:

```python
normal_statuses = ["run"]
target_statuses = "auto"
```

If the dataset contains:

```text
run
alarm
fullwork
no_work
mc_stop
```

Then the target statuses become:

```text
alarm
fullwork
no_work
mc_stop
```

This follows the business rule:

> Predict every non-run status.

---

# Step 3: Automatic Training Pipeline

Notebook:

```text
3_automatic_training_pipeline.ipynb
```

## Purpose

This is the final Phase 4 notebook.

It trains the final machine-learning models and exports artifacts for Phase 5.

This notebook does everything Step 2 does, plus:

* trains LightGBM ETA models
* trains next-status classifier
* calibrates ETA range
* evaluates on test data
* saves model artifacts
* reloads artifacts to verify they work
* exports feature importance reports
* exports model card
* exports final ZIP package

## How to Use

Open the notebook.

Go to:

```text
P4-1 — Define AutoTrainingConfig
```

Edit the config the same way as Step 2.

Then click:

```text
Run All
```

## Important Results to Check

### Regression / ETA Model Metrics

Check:

```text
P4-30 — Final ETA test evaluation
```

This tells you how well the model predicts time until next target event.

Important fields:

```text
p50_test_MedAE_sec
p50_test_MAE_sec
p50_test_Hit_5m_percent
p50_test_Hit_10m_percent
p90_test_coverage_raw_percent
p90_test_median_width_sec
```

### Classification / Next Status Metrics

Check:

```text
P4-31 — Final next-type test evaluation
```

This tells you how well the model predicts the next target status type.

Important fields:

```text
type_test_accuracy
type_test_macro_f1
type_test_weighted_f1
displayed_accuracy_at_global_threshold
displayed_percent_at_global_threshold
```

### Acceptance Gate

Check:

```text
P4-32 — Final acceptance gate report
```

This tells you whether the model passed the defined quality gates.

Possible results:

```text
PASS
PASS_WITH_WARNINGS
FAIL
```

### Artifact Save

Check:

```text
P4-34 — Artifact save report
```

This confirms that the required files were created.

### Artifact Reload / Smoke Test

Check:

```text
P4-37 — Phase 5 output contract validation
```

This confirms that the saved artifacts can be loaded and can produce valid outputs.

### Final ZIP

Check:

```text
P4-40 — Final artifact manifest and ZIP bundle
```

This creates a ZIP package containing artifacts and reports.

---

# Important Configuration Fields

The main config is inside:

```text
P4-1 — Define AutoTrainingConfig
```

The most important fields are:

## `process_id`

Name of the process being trained.

Example:

```python
process_id = "gd"
```

or:

```python
process_id = "assy"
```

Use one process per artifact bundle.

## `data_path`

Path to the dataset.

Example:

```python
data_path = "status_nht_converted.parquet"
```

or:

```python
data_path = "/home/micml/Documents/TestML/REAL/MCSTATUS_ASSY.CSV"
```

## `normal_statuses`

Statuses that are considered normal and are not prediction targets.

Example:

```python
normal_statuses = ["run", "mc_run"]
```

or:

```python
normal_statuses = ["run"]
```

## `target_statuses`

Statuses the model should predict.

Usually use:

```python
target_statuses = "auto"
```

This means every non-normal status becomes a target.

You can also manually set:

```python
target_statuses = ["mc_alarm", "mc_stop"]
```

## `status_alias_map`

This maps inconsistent status names into one standard name.

Example:

```python
status_alias_map = {
    "mc_fullWork": "mc_fullwork",
    "mc fullwork": "mc_fullwork",
    "m/c stop": "mc_stop",
}
```

This is important because the same status may appear with different spelling.

Example:

```text
mc_fullWork
mc_fullwork
mc fullwork
```

These should be treated as the same status.

## `hidden_statuses`

Statuses hidden from dashboard display.

This does not affect model training target logic.

Example:

```python
hidden_statuses = ["run", "mc_run"]
```

## `same_timestamp_conflict_policy`

Controls what happens if the same machine has multiple statuses at the exact same timestamp.

Recommended:

```python
same_timestamp_conflict_policy = "drop_conflicts"
```

This removes ambiguous rows.

## `artifact_dir`

Where artifacts are saved.

Example:

```python
artifact_dir = "./artifacts_phase4"
```

Recommended for multiple processes:

```python
artifact_dir = "./artifacts_phase4_gd"
```

or:

```python
artifact_dir = "./artifacts_phase4_assy"
```

Use a different folder for each process to avoid overwriting artifacts.

---

# Understanding Normal Statuses and Target Statuses

This is the most important concept in the project.

## Normal Status

A normal status is a status that means the machine is not in an important target event.

Example:

```text
run
mc_run
```

Normal statuses are used as input history but are not predicted as target events.

## Target Status

A target status is a status that the model should predict.

Example:

```text
mc_alarm
mc_stop
mc_waitpart
mc_fullwork
alarm
fullwork
no_work
```

## Example 1

Dataset statuses:

```text
run
mc_alarm
mc_stop
mc_waitpart
```

Config:

```python
normal_statuses = ["run"]
target_statuses = "auto"
```

Target statuses become:

```text
mc_alarm
mc_stop
mc_waitpart
```

## Example 2

Dataset statuses:

```text
run
mc_run
mc_alarm
mc_fullwork
mc_waitpart
mc_stop
```

Config:

```python
normal_statuses = ["run", "mc_run"]
target_statuses = "auto"
```

Target statuses become:

```text
mc_alarm
mc_fullwork
mc_waitpart
mc_stop
```

## Business Rule Used in This Project

For many processes, the rule is:

> Predict every status except run/normal statuses.

That is why `target_statuses = "auto"` is usually used.

---

# Understanding the Model Outputs

The Phase 4 models are designed to produce four final outputs.

```text
eta_p50_sec
eta_p90_sec
next_type
type_conf
```

## `eta_p50_sec`

The model’s typical estimate of how many seconds until the next target event.

P50 means the middle estimate.

Simple explanation:

> The model thinks the next target event will probably happen around this many seconds from now.

## `eta_p90_sec`

The model’s upper estimate of how many seconds until the next target event.

P90 is usually larger than P50.

Simple explanation:

> The model thinks there is roughly a high chance the next target event should happen within this time.

## `next_type`

The predicted next target status.

Example:

```text
mc_waitpart
mc_alarm
mc_stop
fullwork
no_work
```

## `type_conf`

The model confidence for `next_type`.

Example:

```text
0.91
```

This means the model is about 91% confident in the predicted next status type.

## Display Rule

The dashboard should usually display type predictions only when confidence is high enough.

Default threshold:

```text
type_conf >= 0.60
```

Low-confidence predictions can still be logged for monitoring, but should be treated carefully.

---

# Understanding the Exported Artifacts

The final notebook saves several files.

The four most important files are:

```text
lgbm_quantile_p50_final.pkl
lgbm_quantile_p90_final.pkl
lgbm_next_type.pkl
artifacts_phase4.pkl
```

## `lgbm_quantile_p50_final.pkl`

This is the model for predicting `eta_p50_sec`.

It predicts the typical time until the next target event.

## `lgbm_quantile_p90_final.pkl`

This is the model for predicting `eta_p90_sec`.

It predicts a safer upper-bound time estimate.

## `lgbm_next_type.pkl`

This is the classifier model.

It predicts `next_type`.

## `artifacts_phase4.pkl`

This is the metadata bundle.

It contains important supporting information such as:

* feature column list
* fill values
* status configuration
* normal statuses
* target statuses
* type label encoder
* calibration maps
* post-processing rules
* model metrics
* acceptance results

Do not delete this file. The models need it to work correctly in Phase 5.

---

# Other Exported Files

## `artifacts_phase4_summary.json`

A human-readable summary of the artifact bundle.

## `model_card_phase4.json`

Machine-readable model card.

## `model_card_phase4.md`

Human-readable model card.

This explains:

* process ID
* selected models
* status configuration
* metrics
* warnings
* known limitations

## `phase5_integration_checklist.md`

Checklist for integrating artifacts into the Phase 5 prediction system.

## `artifact_manifest.json`

List of exported files with file sizes and checksums.

## Feature Importance CSV Files

```text
feature_importance_p50.csv
feature_importance_p90.csv
feature_importance_next_type.csv
feature_importance_combined.csv
feature_importance_aggregate.csv
```

These explain which features the models used most.

---

# How to Decide Whether a Model Is Good Enough

The final notebook reports model quality.

Important evaluation points:

## ETA Model

Check:

```text
P4-30
```

Important metrics:

### `MedAE_sec`

Median absolute error.

Simple explanation:

> Typical prediction error in seconds.

Lower is better.

Example:

```text
MedAE = 33 seconds
```

means the typical prediction is off by about 33 seconds.

### `MAE_sec`

Mean absolute error.

This is more affected by large errors.

### `Hit_5m_percent`

Percentage of predictions within ±5 minutes.

Higher is better.

### `Hit_10m_percent`

Percentage of predictions within ±10 minutes.

Higher is better.

### `P90_coverage_raw_percent`

How often the actual event happened before the P90 estimate.

Target range is usually around:

```text
88% to 93%
```

If P90 coverage is too low, the model is too optimistic.

If P90 coverage is too high, the model is too conservative.

---

## Classification Model

Check:

```text
P4-31
```

Important metrics:

### `type_test_accuracy`

Overall next-status prediction accuracy.

### `macro_f1`

Balanced score across all classes.

This is useful when some statuses are rare.

### `weighted_f1`

F1 score weighted by class frequency.

This is useful for overall performance.

### `displayed_accuracy_at_global_threshold`

Accuracy only for rows where confidence is high enough.

Default threshold:

```text
type_conf >= 0.60
```

### `displayed_percent_at_global_threshold`

How many rows are displayed after applying the confidence threshold.

---

## Acceptance Gate

Check:

```text
P4-32
```

Typical gate examples:

```text
ETA MedAE <= 60 seconds
ETA Hit@±5m >= 75%
ETA Hit@±10m >= 80%
P90 global coverage between 88% and 93%
Type displayed accuracy >= 90%
Type displayed percent >= 50%
```

Possible final results:

### `PASS`

The model passed all important gates.

### `PASS_WITH_WARNINGS`

The model passed required gates but has warnings.

This can be acceptable, especially for short data windows or rare machine cases.

### `FAIL`

The model failed one or more hard gates.

This does not always mean the code is broken. It may mean:

* the data is noisy
* the process is hard to predict
* the class distribution is highly imbalanced
* there are not enough examples
* event logs alone are not enough for strong prediction

---

# Common Problems and Fixes

## Problem: Dataset rejected in Step 1

Cause:

The dataset does not have the required columns.

Required:

```text
occurred
mc_no
mc_status
process
```

Fix:

Rename the columns in the dataset before running the pipeline.

---

## Problem: `registered` column warning appears

Cause:

The dataset contains the legacy `registered` column.

Fix:

This is only a warning.

The pipeline uses:

```text
occurred
```

as the real timestamp.

Do not use `registered` as the model timestamp.

---

## Problem: Status names look inconsistent

Example:

```text
mc_fullWork
mc_fullwork
mc fullwork
```

Fix:

Use `status_alias_map` in P4-1.

Example:

```python
status_alias_map = {
    "mc_fullWork": "mc_fullwork",
    "mc fullwork": "mc_fullwork",
}
```

---

## Problem: Wrong target statuses

Cause:

`normal_statuses` was not set correctly.

Fix:

Run Step 1.5 and inspect all status names.

Then set:

```python
normal_statuses = ["run"]
target_statuses = "auto"
```

or:

```python
normal_statuses = ["run", "mc_run"]
target_statuses = "auto"
```

depending on the process.

---

## Problem: Model performance is bad

Possible causes:

1. The process is difficult to predict from event logs alone.
2. The dataset has too few rows.
3. A target status is extremely rare.
4. There are many long time gaps.
5. Machine behavior changed over time.
6. Status labels are inconsistent.
7. The model does not have physical sensor features.

Fixes:

* check status names again
* check target distribution
* check baseline performance at P4-19
* check class frequency at P4-29 or P4-31
* use more data if possible
* make sure aliases are correct
* treat weak models as shadow-test only first

---

## Problem: `mc_stop` or another class has bad classification score

Cause:

The class may be rare.

If there are very few examples, the model cannot learn it well.

Fix:

* check class frequency report
* do not over-trust rare-class predictions
* use confidence threshold
* keep logging predictions for future monitoring
* retrain when more data is available

---

## Problem: P90 coverage is too low

Cause:

The P90 model is too optimistic.

It predicts an upper-bound that is too short.

Fix:

* review P90 calibration report
* monitor per-machine P90 coverage
* adjust P90 multiplier if needed in future serving system
* retrain with more recent data

---

## Problem: P40 / P50 / P90 terminology is confusing

This project mainly uses:

```text
P50
P90
```

P50 means normal/middle estimate.

P90 means safer upper estimate.

The production output uses:

```text
eta_p50_sec
eta_p90_sec
```

---

## Problem: Artifact ZIP takes too long

Cause:

The ZIP may be created inside the same folder being zipped.

Fix:

Use the safe P4-40 cell.

The ZIP should be written outside the artifact directory.

Correct behavior:

```text
artifact_dir/
    model files

phase4_artifacts_<process_id>.zip
```

not:

```text
artifact_dir/
    phase4_artifacts_<process_id>.zip
```

---

# Handover Notes

## What the Next Developer Must Know

This project is Phase 4 only.

It trains and exports artifacts.

It does not run the live prediction service by itself.

The next phase, Phase 5, should load the exported artifacts and use them in a prediction service.

## Do Not Delete These Files

```text
lgbm_quantile_p50_final.pkl
lgbm_quantile_p90_final.pkl
lgbm_next_type.pkl
artifacts_phase4.pkl
```

These are required for Phase 5.

## Always Keep the Same Feature Order

The model expects the feature columns in the exact order stored in:

```text
artifacts_phase4.pkl
```

If the feature order changes, predictions may be wrong.

## One Process = One Artifact Bundle

Do not mix processes inside one artifact bundle.

Recommended:

```text
artifacts_phase4_gd/
artifacts_phase4_assy/
artifacts_phase4_apb/
```

## Manual Cleaning Is Still Required

This project expects the dataset to already be cleaned into the required pattern.

The pipeline does not magically fix raw messy factory data.

Before running this pipeline, make sure the dataset has:

```text
occurred
mc_no
mc_status
process
```

## Recommended Handover Operating Procedure

For each new process:

1. Clean the raw dataset manually.
2. Make sure it has the required four columns.
3. Run `1_pattern-checking-gate.ipynb`.
4. Run `1_5_status_name_check.ipynb`.
5. Decide `normal_statuses`.
6. Decide whether `target_statuses = "auto"` is correct.
7. Add status aliases if needed.
8. Run `2_automatic_baseline_test.ipynb`.
9. Check P4-19 baseline results.
10. Run `3_automatic_training_pipeline.ipynb`.
11. Check P4-30 ETA results.
12. Check P4-31 classification results.
13. Check P4-32 acceptance result.
14. Check P4-34 artifact save result.
15. Check P4-37 Phase 5 contract validation.
16. Use the final ZIP or four required files for Phase 5.

---

# Glossary

## Artifact

A saved file needed for production prediction.

Example:

```text
lgbm_quantile_p50_final.pkl
```

## Baseline

A simple prediction method used as a reference.

Example:

> For this machine and status, use the historical median time.

If the machine-learning model cannot beat the baseline, the data may not be very learnable.

## Classification

Predicting a category.

In this project:

```text
next_type
```

is classification.

Example:

```text
mc_alarm
mc_waitpart
mc_stop
```

## Regression

Predicting a number.

In this project:

```text
eta_p50_sec
eta_p90_sec
```

are regression outputs.

## ETA

Estimated time of arrival.

In this project, ETA means:

> estimated time until the next target machine status.

## P50

The typical/middle estimate.

## P90

The safer upper estimate.

## Target Event

A machine status event that the model should predict.

Example:

```text
mc_alarm
mc_stop
mc_waitpart
```

## Normal Status

A status that is not considered a prediction target.

Example:

```text
run
mc_run
```

## `type_conf`

Confidence score for the predicted next status.

Example:

```text
0.91
```

Higher is better.

## Shadow Testing

Running predictions in the background without using them for real production decisions yet.

This is recommended when the model works technically but quality is not strong enough for full trust.

---

# Final Summary

This repository contains a notebook-based automatic training pipeline for machine status prediction.

It checks the dataset pattern, checks status names, evaluates baselines, trains final models, exports artifacts, and validates that the saved artifacts can produce Phase 5-compatible outputs.

Main input requirement:

```text
occurred
mc_no
mc_status
process
```

Main output artifacts:

```text
lgbm_quantile_p50_final.pkl
lgbm_quantile_p90_final.pkl
lgbm_next_type.pkl
artifacts_phase4.pkl
```

Main prediction outputs for Phase 5:

```text
eta_p50_sec
eta_p90_sec
next_type
type_conf
```

For a new process, only the dataset path and status configuration should normally need to change.
