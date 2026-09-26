# Trace the Ace — Project Log & Decision Record

_Last updated: 2026-08-24 (late evening IST). Model submission deadline: **2026-08-27**. Write-up deadline: ~3 weeks later (mid-Sept)._

---

## 0. TL;DR — where we stand right now

| component | status | score (log loss / AUC) |
|---|---|---|
| Feature model (LightGBM + topic encoding) | **done, verified end-to-end** | **0.5476 / 0.7170** (honest held-out) |
| LLM fold 0 (Qwen2.5-7B QLoRA) | done | 0.6075 / 0.6297 |
| LLM fold 1 | **running** (finishes ~09:00 Aug 25) | — |
| LLM fold 2 | **running** (finishes ~10:00 Aug 25) | — |
| `submission.zip` | built (feature-only version) — **NOT yet submitted** | — |
| Leaderboard top to beat | — | **0.5957 / 0.6413** |

**Nothing has been submitted to the competition yet.** Decision was to wait for folds 1 & 2.

**Agreed plan for tomorrow morning:** collect folds 0/1/2 → rebuild `submission.zip` with **0.65 weight on LLM, 0.35 on feature model** → re-run end-to-end test → submit.

---

## 1. Accounts, credentials, quota

| account | credentials file | GPU quota status (Aug 24) |
|---|---|---|
| `human2706` | `C:\Users\manav\kaggle_creds\friend2.json` | **5.15h left**, refreshes Aug 29 (AFTER deadline — effectively unusable) |
| `manav0607` | `C:\Users\manav\kaggle_creds\friend1.json` | **28.57h left**, refreshes Aug 29 |

**Critical gotchas learned:**
- Both creds files contain a `KGAT_...` token. It must be passed as the **`KAGGLE_API_TOKEN` env var**. Using the old `username`/`key` + `~/.kaggle/kaggle.json` route gives **401 Unauthorized on uploads** (reads work, writes fail) — this cost significant time early on.
- Kaggle CLI has **no share/collaborator command**. Dataset sharing must be done in the web UI (Dataset → Settings → Collaborators). This was used to give `manav0607` access to `human2706`'s private datasets.
- `kaggle quota` shows GPU/TPU hours remaining. Use it before planning runs.
- Kaggle CLI **cannot stop a running kernel** (only `delete`, which is destructive). Use the web UI "Stop Session" button.

---

## 2. Kaggle datasets created (all private, owned by `human2706`)

| dataset | contents |
|---|---|
| `trace-the-ace-raw` | `train_features.csv`, `train_labels.csv`, `submission_format.csv`, `train_transcripts/` (22,821 files) |
| `trace-the-ace-features` | `session_features.csv`, `train_features_enriched.csv`, `cv_folds.csv` |
| `trace-the-ace-baseline` | `oof_predictions_baseline.csv`, `baseline_model.pkl`, `baseline_metrics.json` |
| `hf-token` | `hf_token.txt` (HuggingFace auth for model downloads) |

All three are shared with `manav0607`.

**Dataset mount path quirk (important):** Kaggle mounts these at
`/kaggle/input/datasets/{owner}/{slug}/`, **not** the documented `/kaggle/input/{slug}/`.
All notebooks use a `_resolve_dataset_dir()` helper that auto-detects both layouts.

---

## 3. The data

- **35,072 responses / 22,821 sessions / 398 learning objectives**
- Base rate `is_correct` = **0.7025**
- A session can produce multiple responses (one per objective) → rows are **not independent** → CV must be grouped by `session_id`
- Sessions have 3 roles: `tutor`, `student`, and `background` (audio artifacts like `[unclear]`)
- Median session: 267 turns, ~913 student words, ~43 min duration

**KEY FINDING — all training data is VOICE, not chat.**
The competition says two platforms contribute data: Eedi (chat) and Third Space Learning (voice). Scanned all 22,821 transcripts:
- 100% contain `[unclear]` markers
- 99.3% have a `background` speaker role
- Even the "most chat-like" session says *"Can you hear me?"*

→ **The training set is entirely Third Space Learning (voice). Zero chat sessions.**
Implications: (a) generalizability caveat for the write-up; (b) if the test set contains Eedi chat data, we face distribution shift we cannot validate against.

---

## 4. Phase-by-phase log

### Phase 1 — Notebook A: `01-eda-features` ✅ COMPLETE
Produced 17 `feat_*` session-level features + canonical CV folds.

**Features built:** student/tutor word counts & share, numeric-turn density, digit density, spelled-out numbers, hint count, correction count, question counts (tutor/student), turn count, avg turn lengths, turn-gap mean/std, late-session student share, MiniLM embedding similarity (student turns vs objective).

**CV definition (canonical, reused by every later notebook):** 5-fold `GroupKFold` grouped by `session_id`, seed 42 → `cv_folds.csv`. Verified programmatically that no session spans two folds.

**Bugs fixed during this phase:**
1. Dataset mount path (`/kaggle/input/datasets/{owner}/{slug}/`)
2. `pip install sentence-transformers` pulled a torch rebuild → used `--no-deps`
3. GPU hit `AcceleratorError: no kernel image available` → added health-check + CPU fallback
4. CPU encoding of 2.7M student turns would have taken hours → deduplicated to 1.35M unique strings (50.1%) + multi-process encoding

Runtime: ~2.3h (mostly the CPU embedding fallback).

### Phase 2 — Notebook B: `02-baseline-model` ✅ COMPLETE
Logistic regression (StandardScaler + LR) on the 17 `feat_*` columns, true 5-fold OOF.

**Result: log loss 0.6015, AUC 0.5754** (base-rate floor 0.6088).
Beat the reference blog baseline (0.6054 / 0.5473).

Top standardized coefficients: `feat_numeric_turns_per_word` (−0.207), `feat_question_count_student` (−0.118), `feat_turn_count` (+0.109), `feat_embed_sim_objective_mean` (+0.107).

### Phase 3 — Notebook C: `03-llm-finetune` ⚠️ PARTIALLY COMPLETE
QLoRA fine-tune of **Qwen2.5-7B-Instruct** (4-bit, Unsloth) with a **classification head** (CLST-style) trained with BCE loss — chosen over constrained generation because BCE directly matches the log-loss metric.

**Prompt design:** objective text + transcript, truncated to 3072 tokens keeping the first 5 turns + as much of the tail as fits (tail is nearest the assessment), with `[... N turns omitted ...]` marking the gap.

**This phase consumed ~25 GPU-hours and hit six separate failures before working:**

| # | failure | root cause | fix |
|---|---|---|---|
| 1 | `unsloth_zoo==2024.9.2` not found | stale pinned version | stop pinning stale versions |
| 2 | `import pandas` crashed (numpy ABI) | old pins dragged numpy to 1.26 vs image's 2.x | pin numpy to its **exact** pre-installed version |
| 3 | numpy internally broken (`_center` import) | `numpy>=2` floor let pip upgrade 2.0.2→2.5.2 | pin exact, not a floor |
| 4 | `no kernel image available` | Kaggle assigned a **P100** (CC 6.0), not T4 | set `machine_shape: NvidiaTeslaT4` in kernel metadata |
| 5 | shape-mismatch in `matmul_lora` | HF Trainer auto-wrapped in `DataParallel` across 2 T4s; Unsloth kernels aren't DP-safe | `CUDA_VISIBLE_DEVICES=0` **before** torch imports CUDA |
| 6 | CUDA OOM at inference | trainer memory not freed + larger inference batch | free trainer, `INFERENCE_BATCH_SIZE=2` (16/8/4 all OOM — measured) |
| 7 | **`No space left on device`** after 3h | Trainer checkpoint = **7.63GB after 5 steps** (custom `nn.Module` wrapper isn't a `PreTrainedModel`, so it serialized the whole 7B base) | `save_strategy="no"`; rely on the small manual `peft_model.save_pretrained()` (161MB) |

**THE CRITICAL MODELING FINDING — training size drives everything:**

| fold 0 config | train rows | steps | log loss | AUC |
|---|---|---|---|---|
| `TRAIN_SAMPLE_SIZE=1200` | 1,200 | 75 | 0.6180 | **0.4976** ← coin-flip, collapsed |
| `TRAIN_SAMPLE_SIZE=4000` | 4,000 | 250 | 0.6075 | **0.6297** ← works |

At 1,200 rows the classifier collapsed to near-constant high-confidence predictions (std 0.035). At 4,000 it produced real spread (std 0.164, range 0.082–0.954). **Any future fold must use ≥4,000.**

**Measured performance/cost:**
- ~127.6 s per optimizer step (effective batch 16 = 2 × grad-accum 8)
- Training 4,000 rows (250 steps) ≈ **8.9 h**
- Inference ≈ **2.77 s/row** at batch 2 (batch 4/8/16 all OOM)
- **Total ≈ 10.25 h per fold**
- Full data (28,057 rows) for ONE fold would cost **~72 h** — infeasible

**Why we subsample validation too:** scoring all 35,072 rows would cost ~30 GPU-h for inference alone. `VAL_SAMPLE_SIZE=1500` per fold. Consequence: **`oof_predictions_llm.csv` does NOT cover every response** — Notebook D must LEFT JOIN and fall back where `pred_llm` is missing.

**Fold data hygiene (verified):**
- fold 0 trains on folds {1,2,3,4}, seed 42; fold 1 on {0,2,3,4}, seed 43
- **0 leakage** — no fold's training rows appear in its own validation set
- 452 rows (11.3%) overlap between fold 0's and fold 1's training samples — expected and harmless (shared pool), only validation cleanliness matters
- Together the two folds expose 7,548 distinct responses (21.5% of data)

### Phase 4 — 🔑 THE BIG DISCOVERY: learning-objective target encoding

Objective difficulty varies enormously: **base rates 0.412 → 0.912** across objectives. Notebook B ignored this entirely.

Adding smoothed, cross-fitted target encoding of `learning_objective_id` (+ count + embedding-kNN fallback):

| model | log loss | AUC |
|---|---|---|
| base-rate floor | 0.6088 | 0.500 |
| logistic, 17 `feat_*` (Notebook B repro ✓) | 0.6015 | 0.5753 |
| LightGBM, 17 `feat_*` only | 0.6091 | 0.5575 |
| LightGBM, **objective encoding only** | 0.5593 | 0.6949 |
| **LightGBM, feat_* + objective encoding** | **0.5465** | **0.7197** |
| + isotonic calibration | **0.5443** | 0.7180 |

**Leakage discipline:** encoding for a validation fold uses only its training folds; within the training set the encoding is itself computed out-of-fold (inner 5-fold), so no row sees its own label.

**Tuning:** smoothing sweep → 10 optimal. Capacity sweep → `num_leaves=15, n_estimators=1200`.

**Feature importance (gain):** `obj_te` 36.9%, `obj_count` 3.9% → topic = **~41%**; the 17 transcript features together ≈ **54%**. So transcripts carry the majority collectively, but topic difficulty is the single strongest feature.

**⚠️ THE CENTRAL RISK — does topic encoding transfer to the test set?**

| scenario | log loss | AUC |
|---|---|---|
| **World A** — test topics overlap training (session-grouped CV) | 0.5524 | 0.7095 |
| **World B** — test topics all unseen (objective-grouped CV) | 0.5945 | 0.6214 |

Evidence for World A: only **0.2%** of validation rows had an unseen objective under session-grouped CV; the problem statement's warning against objective-difficulty solutions implies the shortcut exists.
Mitigation built in: **embedding-kNN fallback** — an unseen objective borrows the target rate of its semantically nearest *seen* objectives (MiniLM). This lifted World B from 0.5990 → 0.5945, i.e. **even the worst case narrowly beats the leaderboard top (0.5957)**. Encoding is never harmful because unseen objectives fall back to the prior.

### Phase 5 — Submission engineering ✅ (feature-only version built)

Built **locally**, not on Kaggle (all data is local; LightGBM trains in minutes).

**Files:** `scratchpad/submission/main.py`, `scratchpad/submission/llm_infer.py`, `scratchpad/build_assets.py`, `scratchpad/test_e2e.py`, `scratchpad/test_nested.py`

**Verifications passed:**
1. **Feature parity** — `main.py`'s feature engineering reproduces Notebook A's output to **~1e-15** across all 16 non-embedding features. This eliminates the most dangerous class of bug.
2. **End-to-end container simulation** — fold 0 treated as unseen test data, assets built from folds 1–4 only, `main.py` run exactly as the runtime would: **log loss 0.5476, AUC 0.7170**; format validation passed (columns, row count, IDs, range, no NaNs).
3. **Nested validation** — hyperparameters chosen using *only* folds 1–4, then evaluated once on fold 0: **identical 0.5476 / 0.7170** → no selection leakage.

**Bugs the e2e test caught (CV alone would have missed these):**
- pandas 3.x sums an **empty object Series to `''`**, not `0` → `int('')` crash on any session with no student turns. Fixed with dtype-safe `_nsum`/`_nmean` helpers.
- kNN encoding was special-cased for seen objectives → train/inference mismatch. Now computed identically for all.

**Dependency risk removed** (no inference-time internet):
- Replaced `sentence-transformers` with plain `transformers` (`AutoModel` + mean-pool + L2 normalize — mathematically identical for MiniLM). `transformers` is effectively guaranteed by the vLLM/PyTorch runtime; `sentence-transformers` is not.
- Bundled a manylinux **LightGBM wheel** in `assets/wheels/` with an offline `pip install --no-index` fallback.
- MiniLM weights bundled in `assets/minilm` (88MB), with the runtime's preloaded `huggingface_models/` path as fallback.
- The LLM stage is wrapped in try/except: **any LLM failure degrades to the LightGBM prediction rather than crashing the run** (a crashed submission scores zero, not a bad score).

**Known caveat on a self-test:** the smoothing sweep inside `test_nested.py` was ineffective — `smoothed_map(stats, prior, smooth=SMOOTH)` binds its default at def time, so runtime reassignment of `BA.SMOOTH` did nothing and all 16 configs silently ran at `smooth=10`. The headline nested number is still valid (it ran at the shipped setting), and `exp_tune.py` *did* vary smoothing correctly. **Fix this before reusing that script.**

### Phase 6 — Blend weight analysis

Measured on fold 0's 1,500 scored rows:

**Q: Does calibrating the LLM fix its log loss?** **No** — 0.6075 raw → 0.6119 calibrated (slightly worse). The LLM was already roughly calibrated; its loss reflects genuine discrimination limits. _(This disproved an earlier hypothesis.)_

| | feature model | LLM | best blend | optimal LLM weight |
|---|---|---|---|---|
| **World A** (topics overlap) | 0.5575 / 0.7059 | 0.6119 / 0.6223 | 0.5555 | **0.15** |
| **World B** (topics new) | 0.6051 / 0.5868 | 0.6119 / 0.6223 | **0.5841 / 0.6473** | **0.65** |

The optimal weight swings hugely by world. Robustness table:

| LLM weight | World A | World B | worst case |
|---|---|---|---|
| 0.15 | 0.5555 | ~0.598 | 0.598 |
| 0.40 | 0.5583 | 0.5870 | 0.587 |
| **0.65** | ~0.568 | 0.5841 | **0.584** |

**DECISION (user's call): use 0.65 weight on the LLM, 0.35 on the feature model.**
Rationale: topic encoding is *memorisation* and dies if test topics are new; the LLM *reads the conversation* and should transfer. Accepted cost: ~+0.012 log loss / −0.011 AUC in World A, in exchange for ~−0.021 log loss in World B.

_(Assistant's note for the record: a weight of ~0.40 minimises expected loss under most priors over World A/B, and 0.15 is optimal if World A is near-certain. 0.65 is a deliberate robustness/generalization choice, not the CV-optimal one. Re-examine once 3 folds of data exist.)_

---

## 5. Competition rules (verified from the official site, Aug 24)

- **Code-execution challenge**: submit `main.py` + `assets/`; organizers run it on hidden test data
- **Output**: CSV with `response_id`, `probability`
- **No inference-time internet** — all models/deps must be packaged
- **Each test sample scored independently** — no cross-sample statistics, no pseudo-labeling
- **No restriction on model type** — a CPU-only solution is fully allowed by the rules
- **Prize eligibility**: MIT license; external models must permit commercial use (no NC licenses); declare all external data/models
- **Winners chosen by leaderboard performance + write-up quality combined** (top 15 invited to write up)
- **Metric: log loss** (AUC is reference only, does not determine ranking)
- Two source platforms: **Eedi** (chat, ages 9–16) and **Third Space Learning** (voice) — but see §3: all *training* data is voice
- 1,106 participants
- **Model deadline Aug 27; write-up deadline ~3 weeks later** (these are separate — optimize leaderboard now, write up later)

**Why a purely feature-driven solution was rejected as the final answer (user's decision):**
Winners are chosen partly on the write-up, scored 35% Relevance + 35% Generalizability. The problem statement explicitly says organizers do *not* want "a result based only on inferred difficulty from the learning-objective description, without reference to the tutoring session transcript." A solution whose engine is topic difficulty could top the leaderboard and still lose on the write-up.

---

## 6. Currently running (check these first tomorrow)

| job | account | started | expected done |
|---|---|---|---|
| `manav0607/03-llm-finetune-fold1` | manav0607 | ~22:00 Aug 24 | ~08:15 Aug 25 |
| `manav0607/03-llm-finetune-fold2` | manav0607 | ~23:40 Aug 24 | ~10:00 Aug 25 |

Both at `TRAIN_SAMPLE_SIZE=4000`, `VAL_SAMPLE_SIZE=1500`, single T4, `save_strategy="no"`.

Check with:
```bash
export KAGGLE_API_TOKEN=$(python -c "import json;print(json.load(open(r'C:\Users\manav\kaggle_creds\friend1.json'))['key'])")
kaggle kernels status manav0607/03-llm-finetune-fold1
kaggle kernels status manav0607/03-llm-finetune-fold2
kaggle quota
```

---

## 7. Tomorrow's plan

1. **Confirm folds 1 & 2 completed**; download their `oof_predictions_llm_partial.csv` and `lora_adapter/`.
2. **DISCARD the old weak fold-1 predictions** (the 1,200-row adapter, AUC 0.5355). Use only 4,000-row adapters — folds 0, 1, 2.
3. **Re-run `exp_llm_weight.py` on all 4,500 rows** — this is the key check on whether fold 0's AUC 0.6297 replicates or was a fluke. Report the weight the larger sample favours before finalizing.
4. **Rebuild assets** with `--use-llm --llm-weight 0.65`; bundle the chosen adapter + classifier head (and Qwen base weights — see risk below).
5. **Re-run `test_e2e.py`** to confirm the LLM-inclusive zip still runs and validates.
6. **Submit.** Compare the returned leaderboard score against our ~0.57 estimate — if it comes back ~0.61 that means World B (test topics are new) and we still have 2 submissions to react.

---

## 8. Open risks

| risk | severity | notes |
|---|---|---|
| **Fold 0's LLM result may not replicate** | HIGH | Single fold, 1,500 rows. Its first attempt scored AUC 0.4976. Folds 1/2 resolve this. |
| **Test topics may be unseen (World B)** | HIGH | Moves us from 0.55 → 0.60. Cannot be checked locally; only a real submission reveals it. |
| **Qwen base weights not bundled yet** | HIGH | No inference-time internet. ~5GB in 4-bit (under the 60GB cap). Context.md lists only MiniLM as preloaded — must confirm whether Qwen is available at `huggingface_models/`, else bundle it. **Not yet done.** |
| **Submission runtime** | MEDIUM | 10,508 test rows × 2.77 s/row = ~8h on T4; limit is 6h on A100. Should fit (A100 is faster) but is untested. Must ship ONE adapter, not an ensemble of 3. |
| **Test set may contain Eedi chat data** | MEDIUM | All training data is voice; no chat data to validate against. |
| **Only 3 submissions/week** | MEDIUM | Don't waste one on an unverified zip. |

---

## 9. Key file locations

```
C:\Users\manav\OneDrive\Desktop\Trace and Ace Comp\
  Context.md                      # original project spec
  MEMORY.md                       # this file
  submission.zip                  # built (feature-only, 81MB) - NOT final
  notebooks\01-eda-features.ipynb
  notebooks\02-baseline-model.ipynb
  notebooks\03-llm-finetune.ipynb

<scratchpad>\                     # %LOCALAPPDATA%\Temp\claude\...\scratchpad\
  submission\main.py              # inference entry point
  submission\llm_infer.py         # LLM stage (fail-safe)
  build_assets.py                 # trains final model, writes assets/
  test_e2e.py                     # container simulation + scoring
  test_nested.py                  # nested validation (has the default-arg bug noted above)
  exp_objective_encoding.py       # the discovery experiment
  exp_robustness.py               # World A vs World B
  exp_tune.py                     # smoothing + capacity sweep
  exp_llm_weight.py               # blend weight analysis
  scan_platform.py                # voice-vs-chat detection
  objective_embeddings.npy        # MiniLM embeddings of all 398 objectives
  trace-the-ace-raw\              # full local copy of raw data
  trace-the-ace-features\         # Notebook A outputs
```

---

## 10. Reusable lessons

- **Always run an end-to-end container simulation.** CV cannot catch inference-path bugs; the e2e test found two that would have scored zero.
- **Verify feature parity between training and inference code** (we matched to 1e-15). This is the highest-value single check in a code-execution competition.
- **Pin dependency versions to *exact* pre-installed values**, never floors, and install in a single pip call so the resolver sees all constraints together.
- **Measure before extrapolating.** The "~149 s/step" estimate came from a 5-step smoke test and drove a budget plan that was wrong by 5x on total cost.
- **A custom `nn.Module` wrapper defeats HF Trainer's checkpoint logic** — it will serialize the entire base model. Disable `save_strategy` and save adapters manually.
- **Kaggle log streaming buffers heavily** — silence for 30+ minutes is normal, not a hang. Verify liveness via `kaggle quota` ticking up, not log output.
