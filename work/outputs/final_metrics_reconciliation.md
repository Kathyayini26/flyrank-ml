# Final Metrics Reconciliation & Independent Validation Audit

**Audit Date:** 2026-11-09  
**Evaluation Dataset:** `work/data/capstone_modelling_dataset.parquet` (101,226 rows)  
**Evaluation Cohort:** 29,873 held-out test pages across 8 unseen enterprise client domains  
**Test Base Rate:** 51.55% (unweighted client mean: 31.77%)  
**Random State:** 42 (client-grouped 80/20 split)

---

## 1. Discrepancy Investigation & Root-Cause Analysis

A cross-comparison between `work/outputs/model_results.json`, `work/outputs/validation_audit_results.json`, and the initial notebook execution outputs revealed discrepancies in the **Week 4 Baseline** and **Random Forest** figures, while **HistGradientBoosting** and **Logistic Regression** were 100% identical.

### 1.1 Root-Cause Matrix

| Model | `model_results.json` | `capstone.ipynb` (Initial Run) | Status | Root Cause |
|---|---|---|---|---|
| **Week 4 Baseline (Rule)** | ROC-AUC: 0.5066<br>PR-AUC: 0.5214<br>P@10: 0.70, P@20: 0.65<br>P@50: 0.68, P@100: 0.59 | ROC-AUC: 0.5246<br>PR-AUC: 0.5391<br>P@10: 0.80, P@20: 0.75<br>P@50: 0.80, P@100: 0.88 | **Resolved** | In `build_capstone_notebook.py`, the baseline scoring rule was inadvertently rewritten as `(mom_log_ratio <= -0.10) * (feat_position_volatility >= p75_vol) * feat_impressions`. The canonical Week 4 rule baseline from `w04_baseline_score.ipynb` and `02_train_models.py` uses `(feat_impressions >= p75_imp) & (feat_ctr <= p25_ctr) & (feat_weighted_position <= 20.0)` with continuous impression/CTR tie-breaking. |
| **Random Forest** | ROC-AUC: 0.7353<br>PR-AUC: 0.7428<br>P@10: 1.00, P@20: 0.95<br>P@50: 0.98, P@100: 0.97 | ROC-AUC: 0.7376<br>PR-AUC: 0.7457<br>P@10: 0.90, P@20: 0.95<br>P@50: 0.98, P@100: 0.98 | **Resolved** | In `build_capstone_notebook.py`, Random Forest hyperparameters were inadvertently modified to `n_estimators=150, max_depth=12, min_samples_leaf=20`. The canonical hyperparameters established in `02_train_models.py` and audited in `03_validation_audit.py` are `n_estimators=100, max_depth=10, min_samples_leaf=25`. |
| **HistGradientBoosting** | ROC-AUC: **0.7441**<br>PR-AUC: **0.7521**<br>P@10: **1.00**, P@20: **0.95**<br>P@50: **0.98**, P@100: **0.96** | ROC-AUC: **0.7441**<br>PR-AUC: **0.7521**<br>P@10: **1.00**, P@20: **0.95**<br>P@50: **0.98**, P@100: **0.96** | **Identical** | Fully verified across all runs, scripts, and artifacts down to the 4th decimal place. |
| **Logistic Regression** | ROC-AUC: **0.7250**<br>PR-AUC: **0.7245**<br>P@10: **1.00**, P@20: **0.95**<br>P@50: **0.98**, P@100: **0.97** | ROC-AUC: **0.7250**<br>PR-AUC: **0.7245**<br>P@10: **1.00**, P@20: **0.95**<br>P@50: **0.98**, P@100: **0.97** | **Identical** | Fully verified across all runs, scripts, and artifacts down to the 4th decimal place. |

---

## 2. Authoritative Reconciled Performance Metrics

All models were independently re-evaluated on the exact same 29,873 held-out test rows using the canonical implementations, identical feature matrices, and identical random seeds (`random_state=42`).

### 2.1 The Authoritative Comparison Table

| Model Architecture | Specification / Hyperparameters | Test ROC-AUC | Test PR-AUC | P@10 | P@20 | P@50 | P@100 | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Week 4 Baseline (Rule)** | Train $Q_{75}$ Imp, $Q_{25}$ CTR, Pos $\le 20$, with continuous tie-breaker | 0.5066 | 0.5214 | 0.7000 | 0.6500 | 0.6800 | 0.5900 | N/A |
| **Logistic Regression** | L2 penalty ($C=1.0$), StandardScaler, max_iter=1000 | 0.7250 | 0.7245 | **1.0000** | **0.9500** | **0.9800** | **0.9700** | 0.2160 |
| **Random Forest** | 100 trees, max_depth=10, min_samples_leaf=25 | 0.7353 | 0.7428 | **1.0000** | **0.9500** | **0.9800** | **0.9700** | 0.2129 |
| **HistGradientBoosting (Selected)** | 100 trees, max_depth=6, min_samples_leaf=30, lr=0.08 | **0.7441** | **0.7521** | **1.0000** | **0.9500** | **0.9800** | 0.9600 | **0.2085** |

*Note: Test cohort base rate = 51.55%. Precision@K across all three ML models achieves near-perfect precision ($\ge 95\%$) in the top 50 ranks.*

---

## 3. Subgroup Client Analysis & Sample Size Boundaries

To prevent over-claiming and ensure scientific honesty, subgroup performance is partitioned into **Sufficiently Powered Subgroups ($N \ge 50$)** and **Under-powered Subgroups ($N < 50$)**.

### 3.1 Subgroup Performance Table

| Client Identifier (Hash / Index) | Test Pages ($N$) | Declining Pages ($Y=1$) | Observed Base Rate | Sample Size Status | HistGradientBoosting ROC-AUC | HistGradientBoosting PR-AUC | P@20 |
|---|---|---|---|---|---|---|---|
| `client_62f4a7e64f5e0096` (Client 4) | 17,914 | 11,750 | 65.59% | **Sufficient ($N \ge 50$)** | 0.7551 | 0.8371 | 0.9500 |
| `client_e547b89c05043229` (Client 8) | 8,742 | 2,565 | 29.34% | **Sufficient ($N \ge 50$)** | 0.7287 | 0.5597 | 0.9500 |
| `client_3f0ce4d44fe94f3d` (Client 3) | 2,262 | 372 | 16.45% | **Sufficient ($N \ge 50$)** | 0.7758 | 0.4474 | 0.9500 |
| `client_3197e6291363b4db` (Client 2) | 886 | 691 | 77.99% | **Sufficient ($N \ge 50$)** | 0.7653 | 0.9123 | 1.0000 |
| `client_795153d5b7850ccf` (Client 5) | 48 | 17 | 35.42% | **Insufficient ($N < 50$)** | 0.6490 | 0.5975 | 0.4500 |
| `client_08d2847f24cf89c1` (Client 1) | 17 | 5 | 29.41% | **Insufficient ($N < 50$)** | 0.7500 | 0.6245 | N/A ($N < 20$) |
| `client_8ae2bfb5aa1ffa1e` (Client 6) | 3 | 0 | 0.00% | **Insufficient ($N < 50$)** | *Undefined* (single class) | *Undefined* (single class) | N/A ($N < 20$) |
| `client_9d54435aabd95a6c` (Client 7) | 1 | 0 | 0.00% | **Insufficient ($N < 50$)** | *Undefined* (single class) | *Undefined* (single class) | N/A ($N < 20$) |

### 3.2 Scientific Subgroup Findings & Boundary Language
1. **Sufficiently Powered Cohorts:** Among the 4 sufficiently powered client domains ($N \ge 50$, comprising 29,804 pages or **99.77%** of test records), HistGradientBoosting demonstrates stable discriminative power:
   - ROC-AUC ranges from **0.7287 to 0.7758**.
   - Precision@20 ranges from **0.9500 to 1.0000**, across both high-decline domains (77.99%) and low-decline domains (16.45%).
2. **Under-powered Cohorts ($N < 50$):** The remaining 4 client domains contain between 1 and 48 pages (69 pages total, **0.23%** of test records). Due to extreme small-sample noise and zero positive cases in two domains, these clients are **insufficient for reliable subgroup conclusions**.
3. **No Universal Generalization:** Performance claims apply only to established domains with adequate catalog scale; universal generalization across all possible website architectures is strictly disclaimed.

---

## 4. Probability Terminology & Calibration Review

### 4.1 Calibration Audit
Evaluation of predicted probabilities against empirical frequencies across 10 decile bins reveals:
- **Brier Score:** 0.2085 (HistGradientBoosting), 0.2129 (Random Forest), 0.2160 (Logistic Regression).
- **Decile Monotonicity:** Empirical positive rates scale monotonically with predicted probabilities (from 9.95% in the lowest bin to 94.35% in the highest bin).
- **Systematic Drift:** Due to enterprise catalog skew (Client 4 has a 65.59% base rate), middle decile probabilities slightly under-forecast true empirical decline rates by 5–8%.

### 4.2 Standardized Terminology Directive
- **Prohibited Terms:** "Calibrated confidence score", "true posterior probability", "guaranteed risk percentage".
- **Approved Descriptive Terms:**
  - `"predicted decline probability"`
  - `"model risk score"`
  - `"relative prioritization score"`

---

## 5. Verification Sign-off

- [x] Canonical baseline formula aligned across all scripts and notebook.
- [x] Canonical Random Forest hyperparameters (`n_estimators=100, max_depth=10, min_samples_leaf=25`) restored.
- [x] HistGradientBoosting and Logistic Regression verified 100% identical.
- [x] Subgroup sample size threshold ($N \ge 50$) formally established.
- [x] Probability terminology standardized to non-causal risk scores.
