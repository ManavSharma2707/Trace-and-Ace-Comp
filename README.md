# Trace the Ace — Tutoring Outcomes Competition

Solution repo for the **Trace the Ace** (Tutoring Outcomes Competition), hosted by
DrivenData for the National Tutoring Observatory / K-12 AI Infrastructure Program.

## Task

Given a tutoring session transcript (tutor + student turns) and a short learning-objective
description, predict the probability that the student answers the next assessment
question on that objective correctly. Scored by log loss (lower is better); ROC AUC is
reported for reference only.

## Approach

The final submission blends two models:

- **Feature model** — LightGBM with topic encoding, built from hand-crafted transcript
  features.
- **LLM model** — Qwen2.5-7B fine-tuned with QLoRA on the transcripts, trained via
  cross-validated folds.

The blend weight (0.65 LLM / 0.35 feature model) was chosen deliberately based on
held-out validation performance — see `MEMORY.md` for the full decision log.

## Repository structure

- `Context.md` — single source of truth for competition rules, data schema, and the
  modeling plan.
- `MEMORY.md` — running project log and decision record (scores, experiments, gotchas).
- `notebooks/`
  - `01-eda-features.ipynb` — exploratory data analysis and feature engineering.
  - `02-baseline-model.ipynb` — baseline feature model (LightGBM).
  - `03-llm-finetune.ipynb` — LLM fine-tuning (QLoRA) pipeline.
- `Tutoring_Outcomes_Competition_Problem_Statement.pdf` — official competition problem
  statement.
- `submission.zip` — packaged submission artifact.

## Status

See `MEMORY.md` for the latest scores and current state of the submission.
