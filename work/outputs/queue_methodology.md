# Content Opportunity Queue Methodology & Reason Code Verification

## 1. Operational Scoring vs Retrospective Evaluation

In this capstone, we distinguish between two deployment contexts:

1. **Retrospective Evaluation Queue (Current Deliverable)**:
   - Built strictly on the **29,873 held-out test pages** across 8 unseen client domains.
   - Scores are generated from the model trained on the 32 training clients (grouped split, random_state=42).
   - This guarantees that reported precision and ranking metrics are genuine out-of-domain evaluation results.
   - Ground truth April outcome labels are retained in the private artifact solely to verify retrospective precision and are strictly excluded from features.

2. **Future Operational Scoring (Production Pipeline)**:
   - When deploying to score future upcoming months (e.g., scoring May 2026 using March–April historical features), the engineering pipeline fits the model on 100% of available historical data across all active clients.
   - Predictions are then generated for all active portfolio pages for the upcoming editorial cycle.
   - Training predictions and held-out evaluation metrics are never conflated.

---

## 2. Reason-Code Validation & Theoretical Justification

Each recommended page in the opportunity queue is annotated with rule-based reason codes to provide editorial teams with transparent, human-interpretable context. In accordance with honest evaluation practices:
- Reason codes describe **observed historical signals**, not proven causal mechanisms.
- Asset context attributes (e.g. Page 1 placement, longitudinal tracking) are strictly distinguished from decline signals.

### Verified Reason-Code Definitions & Thresholds

| Reason Code | Category | Trigger Rule | Statistical & Operational Rationale | Signal Interpretation |
|---|---|---|---|---|
| `ACCELERATING_MOMENTUM_DROP` | Decline Signal | `mom_log_ratio <= -0.15` | $\log_{10}((\text{norm\_mar} + 10) / (\text{norm\_feb} + 10)) \le -0.15$ corresponds to a $\ge 29.2\%$ month-over-month decline in normalized impressions. | The content page experienced a steep downward trajectory in search exposure between February and March, exceeding normal weekly variance. |
| `MILD_MOMENTUM_DECAY` | Decline Signal | `-0.15 < mom_log_ratio < 0.0` | Negative momentum between $0\%$ and $-29.2\%$. | Observed mild deceleration in search impression volume. |
| `SERP_RANK_DETERIORATION` | Decline Signal | `mom_position_change >= 2.0` | In Google Search Console, average position is a 1-based rank (1 is top, 10 is page 1 bottom). Therefore, $\text{mar\_pos} - \text{feb\_pos} \ge +2.0$ represents a numerical increase, indicating the page slipped $\ge 2.0$ ranks lower on search results. | The page's average SERP ranking worsened substantially month-over-month, reducing search visibility and click-through opportunity. |
| `HIGH_POSITION_VOLATILITY` | Risk Signal | `feat_position_volatility >= 11.23` | Exceeds the 75th percentile ($Q_{75} = 11.23$) of daily SERP position standard deviation across training pages. | Indicates severe ranking turbulence, typically caused by multi-intent query competition, keyword cannibalization, or algorithmic testing. |
| `PAGE_ONE_HIGH_EXPOSURE` | Exposure Context (Not Decline) | `is_top10_position == 1` (`feat_weighted_position <= 10.0`) | Ranks 1 through 10 represent Page 1 of search engine results. | **Stakes/Impact Context**: This page sits on Page 1, representing high business exposure and prime search real estate. When accompanied by decline signals, it warrants urgent attention because losing Page 1 ranking carries high business cost. |
| `LOW_CTR_ON_PAGE_ONE` | Efficiency Signal | `is_top10_position == 1` AND `feat_ctr <= 0.0000` | Exceeds bottom quartile threshold ($Q_{25} = 0.0000$) for click-through efficiency among Page 1 content. | Despite achieving Page 1 exposure, searchers rarely click through, indicating search snippet / title mismatch, competitor snippet dominance, or zero-click SERP feature displacement. |
| `ESTABLISHED_ASSET_MONITORING` | Asset Context (Not Decay) | `feat_observed_days >= 50` AND `has_feb_history == 1` | Observed on $\ge 50$ of 59 days with active tracking spanning both February and March 2026. | **Longevity Profile**: The page represents an established, evergreen asset with stable tracking history, rather than a newly indexed or ephemeral post. Its ranking loss is not due to indexing volatility. |
| `MULTIVARIATE_MODEL_SIGNAL` | Model Signal | Default fallback | Triggered when gradient boosting flags high decline probability based on non-linear feature interactions not captured by isolated univariate rules. | Broad multi-feature risk identified by the non-linear gradient boosting ensemble. |

---

## 3. Editorial Action Playbook Mapping

Each reason code maps directly to an actionable, decision-support task for content editors. None of these actions claim guaranteed traffic recovery; they provide prioritized investigative starting points:

| Reason Code | Recommended Editorial Action | Verification & Next-Step Protocol |
|---|---|---|
| `ACCELERATING_MOMENTUM_DROP` | `INVESTIGATE_SEARCH_DEMAND_DECLINE` | Audit Google Trends and primary query intent to determine whether broader search demand dropped or if competitor URLs captured category share. |
| `SERP_RANK_DETERIORATION` | `REVIEW_CONTENT_RELEVANCE_VS_COMPETITORS` | Inspect top 3 ranking competitor pages; identify missing sections, outdated citations, thinner depth, or weaker topical coverage. |
| `HIGH_POSITION_VOLATILITY` | `AUDIT_SERP_INSTABILITY` | Audit site architecture for internal keyword cannibalization between multiple competing URLs; check for oscillating query intents. |
| `PAGE_ONE_HIGH_EXPOSURE` | *(Context Marker)* | Informs review priority: prioritize high-exposure assets first in the editorial backlog before reviewing lower-visibility pages. |
| `LOW_CTR_ON_PAGE_ONE` | `OPTIMISE_TITLE_AND_SNIPPET_CTR` | Rewrite title tags and meta descriptions to improve click appeal, align snippet copy with search intent, and incorporate structured data. |
| `ESTABLISHED_ASSET_MONITORING` | `ASSESS_EVERGREEN_CONTENT_FRESHNESS` | Review core evergreen content: update dated statistics, replace broken links, refresh examples, and enhance visual media. |
| `MULTIVARIATE_MODEL_SIGNAL` | `GENERAL_CONTENT_REVIEW` | Conduct standard content health audit: check indexing status, technical errors, mobile usability, and basic readability. |

---

## 4. Auditing Protocol & Fallback Steps

Editorial teams using this opportunity queue should follow this verification workflow:

1. **Step 1: Check External Demand (Macro vs Page Issue)**
   - Before editing content, verify whether category-level query volume dropped (e.g. seasonal demand decline in April).
   - If macro demand dropped across the industry, do not rewrite content unnecessarily.

2. **Step 2: Inspect Competitor Displacement**
   - Query the target keyword in an incognito browser. Check if competitors launched comprehensive guides, interactive tools, or video embeds that displaced the page.

3. **Step 3: What to do if a recommendation proves unhelpful (False Positive Handling)**:
   - If editorial audit reveals the page was intentionally deprecated, superseded by a canonical URL, or experienced a planned keyword migration, flag the URL with feedback code `INTENTIONAL_CHANGE`.
   - Log unhelpful picks to calibrate future model training filters and adjust the client-level minimum impression threshold.

---

## 5. Quality & Safety Verification

1. **Monotonic Ranking**: Verified sorted strictly by `predicted_decline_probability DESC`.
2. **Eligibility Compliance**: 100% of rows meet historical impression ($\ge 100$), historical observation ($\ge 15\text{ days}$), and April active observation ($\ge 15\text{ days}$) thresholds.
3. **Zero Data Leakage**: Ground truth labels and April performance metrics are strictly omitted from scoring inputs.
4. **Public Privacy**: Public sample uses synthetic IDs (`OPP_0001` to `OPP_0100`) with all client, domain, content, and query identifiers stripped.
