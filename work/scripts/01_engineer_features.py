import os
import json
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from huggingface_hub import get_token

token = get_token() or os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found!")

# Ensure work directories exist
WORK_DIR = Path("work")
DATA_DIR = WORK_DIR / "data"
OUTPUTS_DIR = WORK_DIR / "outputs"
SCRIPTS_DIR = WORK_DIR / "scripts"

for d in [DATA_DIR, OUTPUTS_DIR, SCRIPTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
safe_token = token.replace("'", "''")
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{safe_token}')")

BASE_PATH = "hf://datasets/FlyRank/internship-warehouse"

print("=" * 80)
print("STAGE 1: EXTRACTING HISTORICAL FEATURES (Feb 1 - Mar 31, 2026)")
print("=" * 80)

feature_query = f"""
CREATE OR REPLACE TEMP TABLE raw_historical_features AS
SELECT
    client_hash_id,
    content_hash_id,
    
    -- Overall 59d Volume & Traffic Signals
    COUNT(DISTINCT report_date) AS feat_observed_days,
    SUM(gsc_impressions) AS feat_impressions,
    SUM(gsc_clicks) AS feat_clicks,
    
    -- Overall CTR
    CASE 
        WHEN SUM(gsc_impressions) > 0 
        THEN SUM(gsc_clicks) * 1.0 / SUM(gsc_impressions) 
        ELSE 0.0 
    END AS feat_ctr,
    
    -- Weighted Average Search Position & Volatility
    SUM(CASE WHEN gsc_avg_position IS NOT NULL THEN gsc_avg_position * gsc_impressions ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN gsc_avg_position IS NOT NULL THEN gsc_impressions ELSE 0 END), 0) AS feat_weighted_position,
        
    COALESCE(STDDEV(gsc_avg_position), 0.0) AS feat_position_volatility,
    
    -- Daily averages
    SUM(gsc_impressions) * 1.0 / COUNT(DISTINCT report_date) AS daily_avg_impressions,
    SUM(gsc_clicks) * 1.0 / COUNT(DISTINCT report_date) AS daily_avg_clicks,
    
    -- February 2026 Signals (28 days)
    SUM(CASE WHEN month = '2026-02' THEN gsc_impressions ELSE 0 END) AS feb_impressions,
    SUM(CASE WHEN month = '2026-02' THEN gsc_clicks ELSE 0 END) AS feb_clicks,
    COUNT(DISTINCT CASE WHEN month = '2026-02' THEN report_date END) AS feb_observed_days,
    
    CASE 
        WHEN SUM(CASE WHEN month = '2026-02' THEN gsc_impressions ELSE 0 END) > 0 
        THEN SUM(CASE WHEN month = '2026-02' THEN gsc_clicks ELSE 0 END) * 1.0 
             / SUM(CASE WHEN month = '2026-02' THEN gsc_impressions ELSE 0 END) 
        ELSE 0.0 
    END AS feb_ctr,
    
    SUM(CASE WHEN month = '2026-02' AND gsc_avg_position IS NOT NULL THEN gsc_avg_position * gsc_impressions ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN month = '2026-02' AND gsc_avg_position IS NOT NULL THEN gsc_impressions ELSE 0 END), 0) AS feb_weighted_position,
        
    -- March 2026 Signals (31 days)
    SUM(CASE WHEN month = '2026-03' THEN gsc_impressions ELSE 0 END) AS mar_impressions,
    SUM(CASE WHEN month = '2026-03' THEN gsc_clicks ELSE 0 END) AS mar_clicks,
    COUNT(DISTINCT CASE WHEN month = '2026-03' THEN report_date END) AS mar_observed_days,
    
    CASE 
        WHEN SUM(CASE WHEN month = '2026-03' THEN gsc_impressions ELSE 0 END) > 0 
        THEN SUM(CASE WHEN month = '2026-03' THEN gsc_clicks ELSE 0 END) * 1.0 
             / SUM(CASE WHEN month = '2026-03' THEN gsc_impressions ELSE 0 END) 
        ELSE 0.0 
    END AS mar_ctr,
    
    SUM(CASE WHEN month = '2026-03' AND gsc_avg_position IS NOT NULL THEN gsc_avg_position * gsc_impressions ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN month = '2026-03' AND gsc_avg_position IS NOT NULL THEN gsc_impressions ELSE 0 END), 0) AS mar_weighted_position

FROM read_parquet([
    '{BASE_PATH}/fact_content_daily_performance/month=2026-02/**/*.parquet',
    '{BASE_PATH}/fact_content_daily_performance/month=2026-03/**/*.parquet'
], hive_partitioning = true)
WHERE gsc_data_available IS TRUE
GROUP BY client_hash_id, content_hash_id
HAVING SUM(gsc_impressions) >= 100 AND COUNT(DISTINCT report_date) >= 15;
"""
con.execute(feature_query)
print("Historical features extracted.")

print("=" * 80)
print("STAGE 2: EXTRACTING APRIL OUTCOME & OBSERVATION AUDIT")
print("=" * 80)

outcome_query = f"""
CREATE OR REPLACE TEMP TABLE raw_april_outcomes AS
SELECT
    client_hash_id,
    content_hash_id,
    COUNT(DISTINCT report_date) AS apr_observed_days,
    SUM(gsc_impressions) AS apr_impressions,
    SUM(gsc_clicks) AS apr_clicks
FROM read_parquet('{BASE_PATH}/fact_content_daily_performance/month=2026-04/**/*.parquet', hive_partitioning = true)
WHERE gsc_data_available IS TRUE
GROUP BY client_hash_id, content_hash_id;

CREATE OR REPLACE TEMP TABLE raw_april_clients AS
SELECT DISTINCT client_hash_id
FROM read_parquet('{BASE_PATH}/fact_content_daily_performance/month=2026-04/**/*.parquet', hive_partitioning = true)
WHERE gsc_data_available IS TRUE;
"""
con.execute(outcome_query)
print("April outcomes extracted.")

print("=" * 80)
print("STAGE 3: JOINING, NORMALIZING TRENDS & BUILDING FINAL DATASET")
print("=" * 80)

final_join_query = """
CREATE OR REPLACE TEMP TABLE capstone_master AS
SELECT
    f.client_hash_id,
    f.content_hash_id,
    
    -- 1. Scale & Volume Features
    f.feat_impressions,
    f.feat_clicks,
    f.feat_observed_days,
    f.daily_avg_impressions,
    f.daily_avg_clicks,
    
    -- Log transformed scale features
    LOG10(f.feat_impressions + 1.0) AS log_feat_impressions,
    LOG10(f.feat_clicks + 1.0) AS log_feat_clicks,
    
    -- 2. Efficiency Features
    f.feat_ctr,
    CASE WHEN f.feat_clicks = 0 THEN 1 ELSE 0 END AS is_zero_clicks,
    
    -- 3. SERP Position Features
    f.feat_weighted_position,
    f.feat_position_volatility,
    CASE WHEN f.feat_weighted_position <= 10.0 THEN 1 ELSE 0 END AS is_top10_position,
    CASE WHEN f.feat_weighted_position <= 20.0 THEN 1 ELSE 0 END AS is_top20_position,
    
    -- 4. Monthly Sub-window Breakdowns
    f.feb_impressions,
    f.mar_impressions,
    f.feb_clicks,
    f.mar_clicks,
    f.feb_observed_days,
    f.mar_observed_days,
    f.feb_ctr,
    f.mar_ctr,
    COALESCE(f.feb_weighted_position, f.feat_weighted_position) AS feb_weighted_position,
    COALESCE(f.mar_weighted_position, f.feat_weighted_position) AS mar_weighted_position,
    CASE WHEN f.feb_observed_days > 0 THEN 1 ELSE 0 END AS has_feb_history,
    
    -- 5. Month-over-Month Dynamics (rate normalized for 28 vs 31 days)
    (f.feb_impressions * (30.0 / 28.0)) AS norm_feb_impressions,
    (f.mar_impressions * (30.0 / 31.0)) AS norm_mar_impressions,
    
    ((f.mar_impressions * (30.0 / 31.0)) - (f.feb_impressions * (30.0 / 28.0)))
        / ((f.feb_impressions * (30.0 / 28.0)) + 1.0) AS mom_impression_change_pct,
        
    (f.mar_ctr - f.feb_ctr) AS mom_ctr_change,
    
    (COALESCE(f.mar_weighted_position, f.feat_weighted_position) 
     - COALESCE(f.feb_weighted_position, f.feat_weighted_position)) AS mom_position_change,
     
    -- 6. Target Construction (Strictly quarantined ground truth)
    a.apr_observed_days,
    a.apr_impressions,
    a.apr_clicks,
    (f.feat_impressions * (30.0 / 59.0)) AS expected_apr_impressions,
    
    CASE 
        WHEN a.apr_impressions < 0.80 * (f.feat_impressions * (30.0 / 59.0)) 
        THEN 1 
        ELSE 0 
    END AS target_declining_20pct

FROM raw_historical_features f
JOIN raw_april_clients c ON f.client_hash_id = c.client_hash_id
JOIN raw_april_outcomes a ON f.client_hash_id = a.client_hash_id AND f.content_hash_id = a.content_hash_id
WHERE a.apr_observed_days >= 15;
"""
con.execute(final_join_query)

# Verify population count
n_rows = con.execute("SELECT COUNT(*) FROM capstone_master").fetchone()[0]
n_distinct = con.execute("SELECT COUNT(DISTINCT client_hash_id || '___' || content_hash_id) FROM capstone_master").fetchone()[0]
print(f"Master Dataset Created: {n_rows:,} rows, Unique Keys: {n_distinct:,}")
assert n_rows == 101226, f"Expected 101,226 rows, got {n_rows}"
assert n_rows == n_distinct, "Duplicate keys detected!"

# Save Parquet to work/data/
output_parquet_path = DATA_DIR / "capstone_modelling_dataset.parquet"
con.execute(f"COPY capstone_master TO '{output_parquet_path.as_posix()}' (FORMAT PARQUET)")
print(f"Modelling dataset saved to {output_parquet_path} ({os.path.getsize(output_parquet_path)/(1024*1024):.2f} MB)")

# Pull pandas dataframe for detailed feature audit
df = con.execute("SELECT * FROM capstone_master").df()

# Feature list (excluding keys and target/outcome fields)
KEYS = ["client_hash_id", "content_hash_id"]
OUTCOME_COLS = ["apr_observed_days", "apr_impressions", "apr_clicks", "expected_apr_impressions", "target_declining_20pct"]
FEATURE_COLS = [c for c in df.columns if c not in KEYS and c not in OUTCOME_COLS]

print(f"\nEngineered Features count: {len(FEATURE_COLS)}")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i}. {f}")

# Save feature metadata definition JSON
definitions = {
    "feat_impressions": "Total GSC impressions across February 1 - March 31 (59 days)",
    "feat_clicks": "Total GSC clicks across February 1 - March 31 (59 days)",
    "feat_observed_days": "Number of distinct dates with active GSC data in Feb-Mar",
    "daily_avg_impressions": "Average impressions per observed active day in Feb-Mar",
    "daily_avg_clicks": "Average clicks per observed active day in Feb-Mar",
    "log_feat_impressions": "Log10(1 + feat_impressions)",
    "log_feat_clicks": "Log10(1 + feat_clicks)",
    "feat_ctr": "Total clicks divided by total impressions over Feb-Mar",
    "is_zero_clicks": "Binary flag indicating 0 clicks received during Feb-Mar",
    "feat_weighted_position": "Impression-weighted average SERP position over Feb-Mar",
    "feat_position_volatility": "Standard deviation of daily average SERP position over Feb-Mar",
    "is_top10_position": "Binary flag for weighted average position <= 10.0",
    "is_top20_position": "Binary flag for weighted average position <= 20.0",
    "feb_impressions": "Total GSC impressions in February 2026 (28 days)",
    "mar_impressions": "Total GSC impressions in March 2026 (31 days)",
    "feb_clicks": "Total GSC clicks in February 2026",
    "mar_clicks": "Total GSC clicks in March 2026",
    "feb_observed_days": "Active GSC days in February 2026",
    "mar_observed_days": "Active GSC days in March 2026",
    "feb_ctr": "February CTR (feb_clicks / feb_impressions)",
    "mar_ctr": "March CTR (mar_clicks / mar_impressions)",
    "feb_weighted_position": "Impression-weighted average position in February",
    "mar_weighted_position": "Impression-weighted average position in March",
    "has_feb_history": "Binary flag indicating whether content was tracked in February",
    "norm_feb_impressions": "February impressions scaled to 30-day equivalent (feb_imp * 30/28)",
    "norm_mar_impressions": "March impressions scaled to 30-day equivalent (mar_imp * 30/31)",
    "mom_impression_change_pct": "Percentage change from normalized February to March impressions",
    "mom_ctr_change": "Difference between March CTR and February CTR (mar_ctr - feb_ctr)",
    "mom_position_change": "Change in weighted position from Feb to Mar (mar_pos - feb_pos)"
}

with open(OUTPUTS_DIR / "feature_definitions.json", "w", encoding="utf-8") as f:
    json.dump(definitions, f, indent=2)

print("\n" + "=" * 80)
print("STAGE 4: FEATURE AUDIT & DISTRIBUTIONS")
print("=" * 80)

# 1. Missing values check
missing_series = df[FEATURE_COLS].isna().sum()
print("\nMissing Values per Feature:")
print(missing_series)

# 2. Distribution statistics table
stats_df = df[FEATURE_COLS].describe(percentiles=[0.25, 0.50, 0.75]).T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
print("\nFeature Distributions:")
print(stats_df.round(4).to_string())

# 3. Correlation with Target
target_corr = df[FEATURE_COLS].apply(lambda s: s.corr(df["target_declining_20pct"])).sort_values(ascending=False)
print("\nFeature Correlation with Target (target_declining_20pct):")
print(target_corr.round(4).to_string())

# 4. Multicollinearity / Redundancy Check (|r| >= 0.85)
corr_matrix = df[FEATURE_COLS].corr()
high_corr_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i + 1, len(FEATURE_COLS)):
        col1 = FEATURE_COLS[i]
        col2 = FEATURE_COLS[j]
        val = corr_matrix.loc[col1, col2]
        if abs(val) >= 0.85:
            high_corr_pairs.append((col1, col2, round(val, 4)))

print(f"\nHighly Correlated Pairs (|r| >= 0.85): {len(high_corr_pairs)} pairs found")
for c1, c2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {c1:28} <-> {c2:28} : r = {r}")

print("\nFeature engineering and audit completed successfully.")
