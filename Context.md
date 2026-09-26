# Trace the Ace — Project Context & Modeling Plan

This file is the single source of truth for the project. Every notebook prompt below
references it. If anything here changes (file names, fold logic, feature names), update
this file first, then regenerate the affected notebook prompts.

---

## 1. Competition Summary

- **Name:** Trace the Ace (Tutoring Outcomes Competition)
- **Host:** DrivenData, sponsor: National Tutoring Observatory, part of the K-12 AI
  Infrastructure Program
- **URL:** https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/
- **Task:** Given a tutoring session transcript (tutor + student turns) and a short
  learning-objective description, predict the probability that the student answers the
  next assessment question on that objective correctly.
- **Metric:** Log loss (lower is better), averaged across all test responses. ROC AUC is
  shown for reference only and does NOT determine ranking. Calibration matters — a
  predicted 0.80 should correspond to ~80% actual correctness across similar cases.
- **Prize structure:** $50,000 total. 1st $15k / 2nd $10k / 3rd $7k, plus 9× $2,000
  publication bonuses. Winners are chosen from a combination of **leaderboard rank** and
  a **solution write-up** (top 15 teams invited to submit). Write-up rubric: Relevance
  35%, Generalizability 35%, Communication 15%, Rigor 15%.
- **Deadline:** model submissions close Aug 27, 2026; write-ups due ~Sep 15, 2026.
- **Reference baseline (public blog post, 3 hand-built features + logistic regression):**
  validation log loss 0.6054, validation AUC 0.5473 (vs. base-rate log loss 0.6089).
- **Current personal best (starting point for this plan):** AUC 0.6430, log loss 0.5961.

## 2. Data

- `train_features.csv`: columns `response_id`, `session_id`, `learning_objective_id`,
  `learning_objective`.
- `train_labels.csv`: columns `response_id`, `is_correct` (0.0 / 1.0). Join on `response_id`.
- `train_transcripts/{session_id}.csv`: one file per session. Columns: `session_id`,
  `utterance_id`, `role` (`tutor`/`student`), `content`, `timestamp`.
- ~35,072 responses / ~22,821 sessions / 398 distinct learning objectives in the public
  reference numbers. A session can produce multiple responses (one per learning
  objective completed in it) — **this means rows are NOT independent across
  `session_id`,** which drives the validation strategy below.
- Base rate of `is_correct` ≈ 0.70.
- At inference time (competition runtime, not Kaggle) the model will see
  `test_features.csv` and `test_transcripts/` in the same schema, plus
  `submission_format.csv` listing required `response_id`s. Output must be a CSV with
  columns `response_id, probability`.

## 3. Runtime / Submission Constraints (final competition submission only — NOT Kaggle)

- Code-execution competition: submit `submission.zip` containing `main.py` at the root
  plus an `assets/` folder with all model weights/code needed.
- Python 3.12, PyTorch + vLLM + CUDA 12.9 preinstalled in the runtime image.
- **No internet access** during inference — every weight/file must be bundled in the
  zip, or already present under `huggingface_models/{org}/{model}/` (a small set of
  models the organizers pre-load, e.g. `sentence-transformers/all-MiniLM-L6-v2`).
- No test-time cross-sample information: each `response_id` must be scored
  independently (no pseudo-labeling, no batch statistics across the test set).
- Zip size ≤ 60GB. Full run must finish in ≤ 6 hours on 1× A100 80GB + 24 vCPU + 220GB
  RAM. Smoke test (100 sampled training rows) must finish in ≤ 20 minutes.
- Only 3 full submissions per week — validate thoroughly offline before submitting.
- External pretrained models/data are allowed but must be under a license that permits
  commercial use (no CC-NC / research-only) to remain prize-eligible.

## 4. Training Environment (this plan)

- **Platform:** Kaggle Notebooks.
- **Accelerator:** GPU T4 ×2 (16GB each), internet ON during training (Kaggle-only
  restriction — irrelevant to the final no-internet runtime above).
- **Quota:** ~30 GPU-hours/week, 12-hour session cap. Every GPU notebook must checkpoint
  well before 12 hours and be resumable.
- **Base LLM candidates (must be commercially-licensed, open-weight):**
  `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) preferred, or `meta-llama/Llama-3.1-8B-Instruct`
  (Llama 3.1 Community License — check commercial-use terms) as fallback.
- **Fine-tuning method:** QLoRA (4-bit NF4 base weights + LoRA adapters), ideally via
  Unsloth for T4 memory/speed efficiency.

## 5. The 4-Notebook Pipeline

All notebooks share Kaggle Datasets as the hand-off mechanism (Kaggle has no shared
filesystem between separate notebooks). Naming is fixed below — **do not rename these
without updating every downstream notebook.**

| # | Notebook | Purpose | Consumes | Produces (→ published as Kaggle Dataset) |
|---|----------|---------|----------|-------------------------------------------|
| A | `01-eda-features` | EDA + engineered session/turn features + canonical CV folds | raw competition data (`trace-the-ace-raw` dataset) | `trace-the-ace-features` dataset: `session_features.csv`, `train_features_enriched.csv`, `cv_folds.csv` |
| B | `02-baseline-model` | Sklearn baseline (logistic regression) on engineered features only; sanity-checks metric + validation harness | `trace-the-ace-features` | `trace-the-ace-baseline` dataset: `oof_predictions_baseline.csv`, `baseline_model.pkl`, `baseline_metrics.json` |
| C | `03-llm-finetune` | QLoRA fine-tune an LLM directly on transcript+objective → P(correct) | `trace-the-ace-raw`, `trace-the-ace-features` (for `cv_folds.csv`) | `trace-the-ace-llm` dataset: `lora_adapter/`, `oof_predictions_llm.csv`, `llm_train_metrics.json` |
| D | `04-stack-calibrate-submit` | Combine baseline + LLM OOF predictions + engineered features in LightGBM, calibrate (isotonic/Platt), evaluate, and package the final `submission.zip` matching the competition runtime spec | `trace-the-ace-features`, `trace-the-ace-baseline`, `trace-the-ace-llm` | `submission.zip` (`main.py` + `assets/`), `final_metrics.json` |

### Canonical CV fold definition (defined once, in Notebook A, reused everywhere)

- Use `GroupKFold` (5 folds) or `GroupShuffleSplit` (80/20, fixed `random_state=42`)
  grouped by `session_id`, so no session's responses ever appear in both train and
  validation for **any** notebook.
- Notebook A writes `cv_folds.csv` with columns `session_id, fold` (fold ∈ {0..4} for
  GroupKFold, or {train, val} for a single split — pick GroupKFold since it lets B, C, D
  each produce true out-of-fold predictions for stacking in D).
- Every other notebook **must load `cv_folds.csv` and merge on `session_id`** rather than
  re-splitting — this is what keeps B's and C's out-of-fold predictions aligned for D's
  stacking step.

### Shared feature/column naming conventions

- `response_id`, `session_id`, `learning_objective_id`, `learning_objective`, `is_correct`
  — keep these exact names end to end.
- Engineered feature columns are prefixed `feat_` (e.g. `feat_n_student_words`,
  `feat_numeric_turns_per_word`, `feat_digit_chars_per_word`, `feat_hint_count`,
  `feat_correction_count`, `feat_turn_gap_mean_s`, `feat_student_word_share`).
- OOF prediction columns are named `pred_baseline` (Notebook B) and `pred_llm`
  (Notebook C) so Notebook D can join them unambiguously on `response_id`.

## 6. Feature Engineering Backlog (Notebook A)

Extend beyond the reference blog's 3 features:
- `feat_n_student_words`, `feat_n_tutor_words`, `feat_student_word_share`
- `feat_numeric_turns_per_word`, `feat_digit_chars_per_word` (from reference notebook)
- `feat_number_word_count` (spelled-out numbers: "twenty-four" etc., regex/word-list based)
- `feat_hint_count` (tutor turns matching hint/prompt patterns)
- `feat_correction_count` (tutor turns indicating student error / correction)
- `feat_question_count_tutor`, `feat_question_count_student`
- `feat_turn_count`, `feat_avg_turn_length_student`, `feat_avg_turn_length_tutor`
- `feat_turn_gap_mean_s`, `feat_turn_gap_std_s` (from `timestamp`)
- `feat_late_session_student_share` (student word share in last third of session — proxy
  for "did the student take over reasoning toward the end")
- Optional: sentence-embedding similarity between student turns and the learning
  objective (using the pre-loaded `sentence-transformers/all-MiniLM-L6-v2` so it stays
  compatible with the no-internet runtime later).

## 7. Validation & Reporting Standard (apply in every notebook)

- Always report **log loss first**, AUC second.
- Always compare against the base-rate constant prediction (train-set mean of
  `is_correct`) as a floor.
- Log metrics to a `*_metrics.json` file in each notebook's output so Notebook D (and
  the eventual write-up) can pull consistent numbers.
- Never fit anything on a session that appears in the validation fold for that split.

## 8. Open Decisions to Revisit

- Whether to keep the `MIN_STUDENT_WORDS` filtering idea from the reference notebook, and
  what fallback prediction to use for filtered-out / low-signal sessions at inference time.
- Context-window truncation strategy for the LLM stage (long sessions can exceed a
  practical fine-tuning context length — plan is "keep the tail nearest the assessment +
  compressed summary of the earlier turns").
- Whether the final submission ships the merged LoRA weights or base + adapter loaded
  via vLLM's adapter support (check `tutoring-outcomes-runtime` repo before Notebook D).