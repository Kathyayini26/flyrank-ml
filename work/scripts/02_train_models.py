import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

WORK_DIR = Path("work")
DATA_DIR = WORK_DIR / "data"
OUTPUTS_DIR = WORK_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STAGE 1: LOADING MODELLING DATASET & FEATURE SELECTION")
print("=" * 80)

data_path = DATA_DIR / "capstone_modelling_dataset.parquet"
df = pd.read_parquet(data_path)
print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Outlier treatment for mom_impression_change_pct:
# Replace unstable division with symmetric log-ratio momentum
# log10((norm_mar + 10) / (norm_feb + 10))
df["mom_log_ratio"] = np.log10((df["norm_mar_impressions"] + 10.0) / (df["norm_feb_impressions"] + 10.0))

# Clean, non-redundant feature subset
FEATURE_COLS = [
    "log_feat_impressions",       # Scale: log total impressions
    "log_feat_clicks",            # Scale: log total clicks
    "feat_observed_days",         # Tracking maturity (15-59)
    "feat_ctr",                   # Search efficiency
    "is_zero_clicks",             # Zero-click barrier flag
    "feat_weighted_position",     # SERP rank
    "feat_position_volatility",   # SERP stability
    "is_top10_position",          # Page 1 SERP presence
    "is_top20_position",          # Page 1-2 SERP presence
    "has_feb_history",            # Mature vs newly tracked in March
    "mom_log_ratio",              # Smooth symmetric MoM momentum
    "mom_ctr_change",             # MoM CTR direction
    "mom_position_change",        # MoM rank drift
]

TARGET_COL = "target_declining_20pct"
CLIENT_COL = "client_hash_id"

print(f"\nSelected Model Features ({len(FEATURE_COLS)} features):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")

# Verify strictly zero leakage from April / outcomes in features
for col in ["apr_observed_days", "apr_impressions", "apr_clicks", "expected_apr_impressions", "target_declining_20pct", "client_hash_id", "content_hash_id"]:
    assert col not in FEATURE_COLS, f"LEAKAGE ALERT: {col} is in FEATURE_COLS!"

print("\nZero-leakage verification passed: No April outcome or ID columns in X.")

print("=" * 80)
print("STAGE 2: CLIENT-GROUPED TRAIN/TEST SPLIT (80/20)")
print("=" * 80)

unique_clients = df[CLIENT_COL].unique()
rng = np.random.default_rng(RANDOM_STATE)
shuffled_clients = rng.permutation(unique_clients)

test_client_count = max(1, int(round(len(shuffled_clients) * 0.2)))
test_clients = set(shuffled_clients[:test_client_count])
train_clients = set(shuffled_clients[test_client_count:])

train_mask = df[CLIENT_COL].isin(train_clients)
test_mask = df[CLIENT_COL].isin(test_clients)

# Verify disjoint
assert len(train_clients.intersection(test_clients)) == 0, "Clients overlap between train and test!"

X_train_raw = df.loc[train_mask, FEATURE_COLS].copy()
y_train = df.loc[train_mask, TARGET_COL].values
X_test_raw = df.loc[test_mask, FEATURE_COLS].copy()
y_test = df.loc[test_mask, TARGET_COL].values

df_train = df.loc[train_mask].copy()
df_test = df.loc[test_mask].copy()

print(f"Total Clients: {len(unique_clients)}")
print(f"Train Clients ({len(train_clients)}): {len(X_train_raw):,} pages ({len(X_train_raw)/len(df)*100:.1f}%)")
print(f"Test Clients ({len(test_clients)}):  {len(X_test_raw):,} pages ({len(X_test_raw)/len(df)*100:.1f}%)")
print(f"Train Base Rate (Decline >= 20%): {y_train.mean()*100:.2f}%")
print(f"Test Base Rate (Decline >= 20%):  {y_test.mean()*100:.2f}%")

print("=" * 80)
print("STAGE 3: BASELINE REPRODUCTION ON TRAIN-FITTED THRESHOLDS")
print("=" * 80)

# Fit Week 4 baseline thresholds ON TRAIN DATA ONLY
p75_imp_train = df_train["feat_impressions"].quantile(0.75)
p25_ctr_train = df_train["feat_ctr"].quantile(0.25)
print(f"Train-fitted Baseline Thresholds: Q75 Impressions = {p75_imp_train:.1f}, Q25 CTR = {p25_ctr_train:.4f}")

# Score both train and test using train-fitted thresholds
def score_week4_baseline(sub_df, p75_imp, p25_ctr):
    score = np.zeros(len(sub_df))
    # Score 100: high imp + low ctr + pos <= 20
    cond_100 = (sub_df["feat_impressions"] >= p75_imp) & (sub_df["feat_ctr"] <= p25_ctr) & (sub_df["feat_weighted_position"] <= 20.0)
    cond_50 = (sub_df["feat_impressions"] < p75_imp)
    score[cond_100] = 100.0
    score[~cond_100 & cond_50] = 50.0
    return score

baseline_score_train = score_week4_baseline(df_train, p75_imp_train, p25_ctr_train)
baseline_score_test = score_week4_baseline(df_test, p75_imp_train, p25_ctr_train)

# Continuous baseline rank tie-breaker: baseline_score DESC, feat_impressions DESC, feat_ctr ASC
baseline_proxy_test = baseline_score_test + (df_test["feat_impressions"] / (df_test["feat_impressions"].max() + 1.0)) - (df_test["feat_ctr"] * 0.01)

print("Baseline computed for train and test splits.")

print("=" * 80)
print("STAGE 4: PREPROCESSING & MODEL TRAINING")
print("=" * 80)

# Preprocessing: StandardScaler fitted ONLY ON TRAIN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

models = {}

# 1. Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
models["Logistic Regression"] = {
    "model": lr,
    "train_proba": lr.predict_proba(X_train_scaled)[:, 1],
    "test_proba": lr.predict_proba(X_test_scaled)[:, 1],
}

# 2. Random Forest
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=25,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_raw, y_train)
models["Random Forest"] = {
    "model": rf,
    "train_proba": rf.predict_proba(X_train_raw)[:, 1],
    "test_proba": rf.predict_proba(X_test_raw)[:, 1],
}

# 3. Gradient Boosting (HistGradientBoosting)
print("Training Gradient Boosting Classifier...")
gb = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=6,
    min_samples_leaf=30,
    learning_rate=0.08,
    random_state=RANDOM_STATE,
)
gb.fit(X_train_raw, y_train)
models["Gradient Boosting"] = {
    "model": gb,
    "train_proba": gb.predict_proba(X_train_raw)[:, 1],
    "test_proba": gb.predict_proba(X_test_raw)[:, 1],
}

print("All models trained successfully.")

print("=" * 80)
print("STAGE 5: MODEL EVALUATION & COMPARISON TABLE")
print("=" * 80)

def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    top_k = y_true[order[:k]]
    return float(np.mean(top_k))

results = []

# Baseline evaluation on test set
p10_b = precision_at_k(y_test, baseline_proxy_test.values, 10)
p20_b = precision_at_k(y_test, baseline_proxy_test.values, 20)
p50_b = precision_at_k(y_test, baseline_proxy_test.values, 50)
p100_b = precision_at_k(y_test, baseline_proxy_test.values, 100)
roc_b = roc_auc_score(y_test, baseline_proxy_test.values)
pr_auc_b = average_precision_score(y_test, baseline_proxy_test.values)

baseline_pred = (baseline_score_test >= 50).astype(int)
prec_b = precision_score(y_test, baseline_pred, zero_division=0)
rec_b = recall_score(y_test, baseline_pred, zero_division=0)
f1_b = f1_score(y_test, baseline_pred, zero_division=0)

results.append({
    "Model": "Week 4 Baseline (Rule)",
    "Train ROC-AUC": "-",
    "Test ROC-AUC": round(roc_b, 4),
    "Test PR-AUC": round(pr_auc_b, 4),
    "Test Precision@10": round(p10_b, 4),
    "Test Precision@20": round(p20_b, 4),
    "Test Precision@50": round(p50_b, 4),
    "Test Precision@100": round(p100_b, 4),
    "Precision": round(prec_b, 4),
    "Recall": round(rec_b, 4),
    "F1 Score": round(f1_b, 4),
})

for name, m_dict in models.items():
    tr_proba = m_dict["train_proba"]
    te_proba = m_dict["test_proba"]
    
    tr_roc = roc_auc_score(y_train, tr_proba)
    te_roc = roc_auc_score(y_test, te_proba)
    te_pr_auc = average_precision_score(y_test, te_proba)
    
    p10 = precision_at_k(y_test, te_proba, 10)
    p20 = precision_at_k(y_test, te_proba, 20)
    p50 = precision_at_k(y_test, te_proba, 50)
    p100 = precision_at_k(y_test, te_proba, 100)
    
    te_pred = (te_proba >= 0.5).astype(int)
    prec = precision_score(y_test, te_pred, zero_division=0)
    rec = recall_score(y_test, te_pred, zero_division=0)
    f1 = f1_score(y_test, te_pred, zero_division=0)
    
    results.append({
        "Model": name,
        "Train ROC-AUC": round(tr_roc, 4),
        "Test ROC-AUC": round(te_roc, 4),
        "Test PR-AUC": round(te_pr_auc, 4),
        "Test Precision@10": round(p10, 4),
        "Test Precision@20": round(p20, 4),
        "Test Precision@50": round(p50, 4),
        "Test Precision@100": round(p100, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    })

res_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print(f"MODEL COMPARISON ON HELD-OUT CLIENTS (Test Base Rate = {y_test.mean()*100:.2f}%)")
print("=" * 80)
print(res_df.to_string(index=False))

# Confusion Matrices
print("\n" + "=" * 80)
print("CONFUSION MATRICES (Threshold = 0.50)")
print("=" * 80)

conf_matrices = {}
for name, m_dict in models.items():
    te_pred = (m_dict["test_proba"] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, te_pred)
    conf_matrices[name] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    print(f"\nModel: {name}")
    print(f"  True Negatives (TN):  {tn:6d}  |  False Positives (FP): {fp:6d}")
    print(f"  False Negatives (FN): {fn:6d}  |  True Positives (TP):  {tp:6d}")
    print(f"  Accuracy: {(tp+tn)/len(y_test):.4f}  |  Base Rate: {y_test.mean():.4f}")

# Feature Importances for Tree Models
print("\n" + "=" * 80)
print("TOP FEATURE IMPORTANCES (Random Forest & Gradient Boosting)")
print("=" * 80)

rf_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nRandom Forest Gini Importance:")
print(rf_imp.round(4).to_string())

# Save metrics JSON receipt
metrics_payload = {
    "evaluation_split": "client_grouped_holdout",
    "random_state": RANDOM_STATE,
    "train_clients": len(train_clients),
    "test_clients": len(test_clients),
    "train_rows": int(len(X_train_raw)),
    "test_rows": int(len(X_test_raw)),
    "train_base_rate": float(round(y_train.mean(), 4)),
    "test_base_rate": float(round(y_test.mean(), 4)),
    "comparison_table": results,
    "confusion_matrices": conf_matrices,
    "random_forest_feature_importance": rf_imp.to_dict(),
}

with open(OUTPUTS_DIR / "model_results.json", "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print(f"\nMetrics receipt saved to {OUTPUTS_DIR / 'model_results.json'}")

print("=" * 80)
print("STAGE 6: GENERATING EVALUATION CHARTS")
print("=" * 80)

# Chart 1: Precision@K Comparison
plt.figure(figsize=(9, 5))
ks = [10, 20, 50, 100]
plt.axhline(y=y_test.mean(), color="black", linestyle="--", label=f"Test Base Rate ({y_test.mean():.3f})")

plt.plot(ks, [p10_b, p20_b, p50_b, p100_b], marker="o", linewidth=2, label="Week 4 Baseline")
for name, m_dict in models.items():
    probas = m_dict["test_proba"]
    pk_vals = [precision_at_k(y_test, probas, k) for k in ks]
    plt.plot(ks, pk_vals, marker="s", linewidth=2, label=name)

plt.title("Precision@K Across Ranked Review Queues (Held-out Test Clients)", fontsize=13, fontweight="bold")
plt.xlabel("Top-K Content Items Reviewed", fontsize=11)
plt.ylabel("Precision@K (Share Actually Declining >= 20%)", fontsize=11)
plt.xticks(ks)
plt.ylim(0.4, 1.05)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
chart1_path = CHARTS_DIR / "precision_at_k_comparison.png"
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved chart: {chart1_path}")

# Chart 2: ROC Curves
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)")
b_fpr, b_tpr, _ = roc_curve(y_test, baseline_proxy_test.values)
plt.plot(b_fpr, b_tpr, label=f"Week 4 Baseline (AUC = {roc_b:.3f})", linewidth=1.8)

for name, m_dict in models.items():
    fpr, tpr, _ = roc_curve(y_test, m_dict["test_proba"])
    auc_val = roc_auc_score(y_test, m_dict["test_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)

plt.title("ROC Curves on Held-Out Test Clients (8 Unseen Client Portfolios)", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
chart2_path = CHARTS_DIR / "roc_curves_comparison.png"
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved chart: {chart2_path}")

# Chart 3: Top Feature Importances (Random Forest)
plt.figure(figsize=(10, 6))
top_feats = rf_imp.head(10).sort_values(ascending=True)
colors = ["#4a7bb0" if "mom" in f or "ctr" in f else "#5ba381" if "pos" in f else "#9b72aa" for f in top_feats.index]
plt.barh(range(len(top_feats)), top_feats.values, color=colors, align="center")
plt.yticks(range(len(top_feats)), top_feats.index, fontsize=10)
plt.xlabel("Random Forest Feature Importance (MDI / Gini)", fontsize=11)
plt.title("Top 10 Historical Search Signals Predicting April Decline", fontsize=13, fontweight="bold")
plt.grid(True, axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
chart3_path = CHARTS_DIR / "feature_importance_rf.png"
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"Saved chart: {chart3_path}")

print("\nModel training, evaluation, receipts and charts completed successfully.")
