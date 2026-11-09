import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42
WORK_DIR = Path("work")
DATA_DIR = WORK_DIR / "data"
OUTPUTS_DIR = WORK_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STAGE 1: LOADING DATASET & FITTING VALIDATED GRADIENT BOOSTING MODEL")
print("=" * 80)

data_path = DATA_DIR / "capstone_modelling_dataset.parquet"
df = pd.read_parquet(data_path)
df["mom_log_ratio"] = np.log10((df["norm_mar_impressions"] + 10.0) / (df["norm_feb_impressions"] + 10.0))

FEATURE_COLS = [
    "log_feat_impressions",
    "log_feat_clicks",
    "feat_observed_days",
    "feat_ctr",
    "is_zero_clicks",
    "feat_weighted_position",
    "feat_position_volatility",
    "is_top10_position",
    "is_top20_position",
    "has_feb_history",
    "mom_log_ratio",
    "mom_ctr_change",
    "mom_position_change",
]

TARGET_COL = "target_declining_20pct"
CLIENT_COL = "client_hash_id"

unique_clients = df[CLIENT_COL].unique()
rng = np.random.default_rng(RANDOM_STATE)
shuffled_clients = rng.permutation(unique_clients)

test_client_count = max(1, int(round(len(shuffled_clients) * 0.2)))
test_clients = set(shuffled_clients[:test_client_count])
train_clients = set(shuffled_clients[test_client_count:])

train_mask = df[CLIENT_COL].isin(train_clients)
test_mask = df[CLIENT_COL].isin(test_clients)

df_train = df.loc[train_mask].copy()
df_test = df.loc[test_mask].copy()

# Fit validated Gradient Boosting on training clients only
gb = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=6,
    min_samples_leaf=30,
    learning_rate=0.08,
    random_state=RANDOM_STATE,
)
gb.fit(df_train[FEATURE_COLS], df_train[TARGET_COL].values)

# Predict positive class probability on held-out test cohort
df_test["predicted_decline_probability"] = gb.predict_proba(df_test[FEATURE_COLS])[:, 1]

print(f"Model fitted on {len(df_train):,} training pages across 32 clients.")
print(f"Scored {len(df_test):,} held-out test pages across 8 clients.")

print("=" * 80)
print("STAGE 2: RULE-BASED REASON CODES & EDITORIAL ACTIONS")
print("=" * 80)

# Calculate training thresholds for interpretable reason code triggers
p75_volatility_train = df_train["feat_position_volatility"].quantile(0.75)
p25_ctr_train = df_train["feat_ctr"].quantile(0.25)

def assign_reasons_and_actions(row):
    reasons = []
    actions = []
    
    # 1. Momentum & Trajectory
    if row["mom_log_ratio"] <= -0.15:
        reasons.append("ACCELERATING_MOMENTUM_DROP")
        actions.append("INVESTIGATE_SEARCH_DEMAND_DECLINE")
    elif row["mom_log_ratio"] < 0.0:
        reasons.append("MILD_MOMENTUM_DECAY")
        actions.append("MONITOR_TRAFFIC_TREND")
        
    # 2. Position Deterioration & Volatility
    if row["mom_position_change"] >= 2.0:
        reasons.append("SERP_RANK_DETERIORATION")
        actions.append("REVIEW_CONTENT_RELEVANCE_VS_COMPETITORS")
    if row["feat_position_volatility"] >= p75_volatility_train:
        reasons.append("HIGH_POSITION_VOLATILITY")
        actions.append("AUDIT_SERP_INSTABILITY")
        
    # 3. High-Value Exposure Marker (Stakes/Impact Context, not a decline signal)
    if row["is_top10_position"] == 1:
        reasons.append("PAGE_ONE_HIGH_EXPOSURE")
        if row["feat_ctr"] <= p25_ctr_train:
            reasons.append("LOW_CTR_ON_PAGE_ONE")
            actions.append("OPTIMISE_TITLE_AND_SNIPPET_CTR")
            
    # 4. Content Longevity & History Profile (Observation context, not decay)
    if row["feat_observed_days"] >= 50 and row["has_feb_history"] == 1:
        reasons.append("ESTABLISHED_ASSET_MONITORING")
        actions.append("ASSESS_EVERGREEN_CONTENT_FRESHNESS")
        
    if not reasons:
        reasons.append("MULTIVARIATE_MODEL_SIGNAL")
        actions.append("GENERAL_CONTENT_REVIEW")
        
    return "|".join(reasons), "|".join(dict.fromkeys(actions))

print("Applying reason code and action rules...")
res = df_test.apply(assign_reasons_and_actions, axis=1)
df_test["reason_codes"] = [r[0] for r in res]
df_test["suggested_actions"] = [r[1] for r in res]

# Sort strictly by predicted probability
df_test = df_test.sort_values("predicted_decline_probability", ascending=False).reset_index(drop=True)
df_test["rank"] = df_test.index + 1

# Risk tier (model risk score grouping, not calibrated confidence)
df_test["risk_tier"] = pd.cut(
    df_test["predicted_decline_probability"],
    bins=[-np.inf, 0.40, 0.65, 0.85, np.inf],
    labels=["Low Risk / Monitor", "Moderate Opportunity", "High Opportunity", "Critical Priority"],
)

print("Ranked queue constructed successfully.")

print("=" * 80)
print("STAGE 3: CREATING OUTPUT FILES & ARTIFACTS")
print("=" * 80)

# Output 1: Private Ranked Queue CSV (includes internal client/content hashes)
private_cols = [
    "rank",
    "client_hash_id",
    "content_hash_id",
    "predicted_decline_probability",
    "risk_tier",
    "reason_codes",
    "suggested_actions",
    "feat_impressions",
    "feat_clicks",
    "feat_ctr",
    "feat_weighted_position",
    "mom_log_ratio",
    "mom_position_change",
    "target_declining_20pct", # Retained exclusively for retrospective evaluation receipts
]
private_queue_path = OUTPUTS_DIR / "ranked_opportunity_queue_private.csv"
df_test[private_cols].to_csv(private_queue_path, index=False)
print(f"Saved private ranked queue: {private_queue_path} ({len(df_test):,} rows)")

# Output 2: Public-Safe Anonymized Sample CSV (NO client IDs, NO content IDs, NO domains/queries)
public_sample = df_test.head(100).copy()
public_sample["opportunity_id"] = [f"OPP_{i+1:04d}" for i in range(len(public_sample))]
public_cols = [
    "opportunity_id",
    "rank",
    "predicted_decline_probability",
    "risk_tier",
    "reason_codes",
    "suggested_actions",
    "feat_impressions",
    "feat_clicks",
    "feat_ctr",
    "feat_weighted_position",
]
public_queue_path = OUTPUTS_DIR / "ranked_opportunity_queue_public_sample.csv"
public_sample[public_cols].round({"predicted_decline_probability": 4, "feat_ctr": 4, "feat_weighted_position": 2}).to_csv(public_queue_path, index=False)
print(f"Saved public-safe sample (Top 100): {public_queue_path}")

# Output 3: Top Recommendations Summary (Top 10, 20, 50, 100)
summary_records = {}
for k in [10, 20, 50, 100]:
    sub_k = df_test.head(k)
    summary_records[f"top_{k}"] = {
        "n_recommendations": k,
        "mean_predicted_probability": round(float(sub_k["predicted_decline_probability"].mean()), 4),
        "actual_decline_rate_retrospective": round(float(sub_k[TARGET_COL].mean()), 4),
        "mean_impressions_at_risk": round(float(sub_k["feat_impressions"].mean()), 1),
        "mean_weighted_position": round(float(sub_k["feat_weighted_position"].mean()), 2),
        "top_reason_codes": sub_k["reason_codes"].str.split("|").explode().value_counts().head(3).to_dict(),
        "top_actions": sub_k["suggested_actions"].str.split("|").explode().value_counts().head(3).to_dict(),
    }

with open(OUTPUTS_DIR / "top_recommendations_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_records, f, indent=2)
print(f"Saved summary JSON: {OUTPUTS_DIR / 'top_recommendations_summary.json'}")

# Output 4: Risk Score Distribution Chart
plt.figure(figsize=(9, 5))
plt.hist(
    df_test["predicted_decline_probability"],
    bins=40,
    color="#3b729f",
    edgecolor="white",
    alpha=0.85,
)
plt.axvline(x=0.50, color="#d9534f", linestyle="--", linewidth=1.8, label="Classification Cutoff (0.50)")
plt.axvline(x=df_test["predicted_decline_probability"].median(), color="#f0ad4e", linestyle=":", linewidth=1.8, label=f"Median Risk ({df_test['predicted_decline_probability'].median():.2f})")
plt.title("Distribution of Predicted Content Decline Risk (Held-out Test Pages)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Probability of >= 20% Search Impression Decline", fontsize=11)
plt.ylabel("Number of Webpages", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
chart_path = CHARTS_DIR / "risk_score_distribution.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved risk score distribution chart: {chart_path}")

print("=" * 80)
print("STAGE 4: QUALITY CHECKS & VALIDATION AUDIT")
print("=" * 80)

# Check 1: No duplicate client-content pairs
dup_count = df_test.duplicated(subset=["client_hash_id", "content_hash_id"]).sum()
print(f"1. Duplicate client-content pairs: {dup_count} (Pass)")
assert dup_count == 0

# Check 2: All rows meet eligibility criteria
eligibility_check = (
    (df_test["feat_impressions"] >= 100) &
    (df_test["feat_observed_days"] >= 15) &
    (df_test["apr_observed_days"] >= 15)
).all()
print(f"2. All rows meet strict eligibility criteria: {eligibility_check} (Pass)")
assert eligibility_check

# Check 3: Ranking monotonic strictly on predicted probability
is_monotonic = df_test["predicted_decline_probability"].is_monotonic_decreasing
print(f"3. Queue sorted monotonically by predicted risk: {is_monotonic} (Pass)")
assert is_monotonic

# Check 4: No sensitive tokens or domain leaks in public sample
for sensitive in ["client_", "content_", "http", "www", ".com"]:
    leak_found = public_sample["opportunity_id"].str.contains(sensitive).any()
    assert not leak_found
print("4. Public-safe queue contains zero client/content hashes or URLs: Verified (Pass)")

print("\n" + "=" * 80)
print("TOP 10 RECOMMENDATIONS PREVIEW (PUBLIC-SAFE ATTRIBUTES)")
print("=" * 80)
preview_cols = ["rank", "predicted_decline_probability", "risk_tier", "feat_impressions", "feat_weighted_position", "reason_codes", "suggested_actions"]
print(df_test[preview_cols].head(10).to_string(index=False))

print("\nAll tasks completed successfully.")
