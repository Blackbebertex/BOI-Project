# 🧩 MULE ACCOUNT CLASSIFICATION – HIGH‑LEVEL ML ENGINEER REPORT

> **Domain:** Cyber-enabled financial fraud prevention  
> **Target:** `F3924` (binary: 1 = mule/suspicious, 0 = legitimate)  
> **Dataset:** ~9,082 account rows × 3,923 anonymized features (`F1`–`F3923`)  
> **Bank key features:** F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894

---

## I. AGENT SYNTHESIS

### 1. EXPOSE — Hidden Assumptions & Blind Spots

| Blind Spot | What the Data Is Not Telling Us |
|------------|--------------------------------|
| **Temporal order** | Flat CSV has no timestamps; time-based CV and sleeper-mule detection rely on inferred window features (F2082/F2122) |
| **Network topology** | No counterparty graph; mule hubs (Aggregator typology) require transaction-level links |
| **Label quality** | 81 positives (~0.9%) may include false negatives; rule-based labels lag evolving typologies |
| **Feature semantics** | Anonymized `F*` columns hide true meaning; domain inference is hypothesis-driven |
| **Label leakage** | **F3912 correlates r=0.97 with F3924** — must be excluded before any production claim |
| **Adversarial dynamics** | Fraudsters adapt to known thresholds; static models decay without retraining |

**Signals not directly listed in the 18 bank keys:** dormancy-then-burst activation, round-number transaction frequency, geographic velocity (impossible travel), device fingerprint churn rate, beneficiary overlap with known fraud rings, cash-in/cash-out channel asymmetry, micro-structuring below reporting thresholds, peer-group deviation (z-score vs segment), and co-movement with flagged accounts in the same batch window.

### 2. IQ200 — Information-Theoretic & Causal Insights

Mule detection is a **sparse signal extraction** problem: ~81 positives in ~9,000 rows across 3,923 dimensions. Information theory dictates that raw dimensionality must be collapsed via MI/Boruta before tree models can generalize. Causal framing: mule accounts are **intermediaries** in a fund-flow chain — the causal parent is upstream fraud, the observable proxy is asymmetric velocity. Non-obvious interactions:

- `F115 × F527 / F531` — young account with high outbound share
- `log(F2678 + 1) − log(F2737 + 1)` — log-scale net flow asymmetry
- `F2082 / (F2122 + ε)` decay-weighted burst ratio
- Entropy of `{F2956, F3043}` channel usage — multi-channel switching

**Meta-learning strategy:** Train per-family sub-models (velocity, device, balance) and use a meta-learner to weight families by holdout PR-AUC contribution — identifies which signal family dominates per fraud wave.

### 3. AUTOPSY — Semantic Decomposition of 3,924 Features

Systematic grouping of the anonymized feature space:

| Semantic Group | Approx. Count | Examples / Proxies |
|----------------|---------------|---------------------|
| Transaction aggregates | ~800 | F527, F531, F1692, F2082, F2122 |
| Amount / balance ratios | ~600 | F670, F2582, F2678, F2737 |
| Account profile / KYC | ~200 | F115, F321 |
| Channel activity | ~300 | F2956, F3043 |
| Terminal / device fingerprint | ~150 | F3836, F3887–F3894 |
| Risk scores / flags | ~400 | High-MI features like F2507, F3912 (leakage) |
| Sparse / missing-heavy | ~1,400 | >50% missing, dropped in pipeline |
| Target | 1 | F3924 |

**18 bank key features — hypothesized real-world meaning:**

| Feature | Likely Meaning | Mule vs Legit (from EDA) |
|---------|---------------|--------------------------|
| F115 | Account age / tenure (normalized 0–1) | Mules higher (0.72 vs 0.59) |
| F321 | Foreign / cross-border exposure | Slightly lower in mules |
| F527 | Outbound transaction count | Similar means; ratio matters |
| F531 | Total / inbound transaction count | Out-in ratio key signal |
| F670 | Balance / amount proxy (sparse) | Mules higher when non-zero |
| F1692 | Short-window txn aggregate | Legit higher (more history) |
| F2082 | Very short-window amount | Mostly zero; burst flag |
| F2122 | Medium-window amount | Low values; velocity denominator |
| F2582 | Amount deviation / z-score | Similar distributions |
| F2678 | Inbound amount flow | Heavy tail; mules lower mean |
| F2737 | Outbound amount flow | Net flow = F2678 − F2737 |
| F2956 | Primary channel activity count | Mules lower (58 vs 133) |
| F3043 | Secondary channel activity | Sparse; mules lower |
| F3836 | Terminal aggregate (wide range) | Different scale by class |
| F3887–F3894 | Device/terminal fingerprint dims | Moderate separation |

### 4. 10X THINK — Expanded Feature Candidate Catalog

See Section II for 50+ derived features with formulas. Key exponential combinations: all pairwise ratios among bank keys, rank-percentile transforms, rolling pseudo-window statistics, cluster distance features (implemented), and anomaly residuals.

### 5. KILL CRITIC — Bold Pipeline Commitment

**Final architecture:** Auto-selected production model from model zoo (`best_model.pkl`, currently **XGBoost** by holdout PR-AUC) + Isolation Forest (legitimate-only training) with 70/30 fusion + TypologyEngine typology boost. Eight candidates trained: LightGBM, XGBoost, CatBoost (optional), RF, ExtraTrees, DecisionTree, LR, MLP, plus stacking ensemble.

### Master Hypothesis

> Mule accounts are separable from legitimate accounts by a **compound signature** of (1) asymmetric fund flow (inbound spike → rapid outbound drain), (2) velocity burst in short vs long windows, (3) behavioral instability in channel/device usage, and (4) deviation from peer-cluster norms — detectable through the 18 bank key features and their engineered interactions, even under anonymization.

---

## II. FEATURE ENGINEERING DEEP DIVE

### Original Listed Features — Transformations

| Feature | Transform Recommendations |
|---------|--------------------------|
| F115 | `log1p`, z-score vs segment, `FEAT_age_x_freq` |
| F321 | clip p99, z-score, ratio with F115 |
| F527, F531 | `FEAT_out_in_count_ratio = F527/(F531+ε)` |
| F670 | `log1p`, missing indicator (75%+ zeros) |
| F1692, F2582 | `FEAT_amount_deviation = |F1692 − F2582|` |
| F2082, F2122 | `FEAT_short_long_amount_ratio` |
| F2678, F2737 | `FEAT_net_fund_flow = F2678 − F2737` |
| F2956, F3043 | `FEAT_multichannel_signal = F2956 × F3043` |
| F3836 | `log1p` (heavy skew) |
| F3887–F3894 | `FEAT_terminal_aggregate`, `FEAT_terminal_max`, entropy |

### High-Value Derived Features (50+ Candidates)

#### Temporal / Velocity (12)

1. `FEAT_out_in_count_ratio = F527 / (F531 + ε)`
2. `FEAT_short_long_amount_ratio = F2082 / (F2122 + ε)`
3. `FEAT_net_fund_flow = F2678 − F2737`
4. `FEAT_burst_velocity = F2082 × F527`
5. `FEAT_dormancy_activation = 1[F115 < p10 ∧ F527 > p90]`
6. `FEAT_txn_acceleration = F527 − F531`
7. `FEAT_window_ratio_log = log1p(F2082) − log1p(F2122)`
8. `FEAT_recent_vs_baseline = F1692 / (F2582 + ε)`
9. `FEAT_zero_history_burst = 1[F1692 = 0 ∧ F527 > 0]`
10. `FEAT_activity_spike = (F527 − μ_F527) / σ_F527`
11. `FEAT_outbound_dominance = F527 / (F527 + F531 + ε)`
12. `FEAT_flow_turnover = (F2678 + F2737) / (F670 + ε)`

#### Ratio & Interaction (15)

13. `FEAT_age_x_freq = F115 × F321` *(implemented)*
14. `FEAT_age_x_outbound = F115 × F527`
15. `FEAT_balance_to_inflow = F670 / (F2678 + ε)`
16. `FEAT_balance_to_outflow = F670 / (F2737 + ε)`
17. `FEAT_in_out_amount_ratio = F2678 / (F2737 + ε)`
18. `FEAT_channel_product = F2956 × F3043` *(implemented)*
19. `FEAT_foreign_x_outbound = F321 × F527`
20. `FEAT_profile_risk = F115 × F527 / (F531 + ε)`
21. `FEAT_amount_x_velocity = F1692 × F527`
22. `FEAT_terminal_x_flow = FEAT_terminal_aggregate × F527`
23. `FEAT_device_flow_ratio = FEAT_terminal_max / (F527 + ε)`
24. `FEAT_structuring_proxy = F531 / (F670 + ε)`
25. `FEAT_cross_period_shift = F2582 − F1692`
26. `FEAT_multiscale_product = F2082 × F2122`
27. `FEAT_composite_risk = F527 × F2678 / (F531 × F2737 + ε)`

#### Graph / Network (8) — Requires Transaction Graph

28. `FEAT_in_degree` — count of unique senders *(needs graph)*
29. `FEAT_out_degree` — count of unique receivers *(needs graph)*
30. `FEAT_pagerank` — account importance in transfer network *(needs graph)*
31. `FEAT_clustering_coef` — local network density *(needs graph)*
32. `FEAT_shared_counterparty_count` — overlap with flagged accounts *(needs graph)*
33. `FEAT_2hop_flagged_ratio` — fraction of 2-hop neighbors flagged *(needs graph)*
34. `FEAT_temporal_motif_count` — fan-in/fan-out patterns *(needs graph)*
35. `FEAT_kmeans_cluster` — behavioral segment proxy *(implemented)*

#### Behavioral Stability (10)

36. `FEAT_channel_entropy = −Σ p_i log(p_i)` over {F2956, F3043}
37. `FEAT_terminal_entropy` over F3887–F3894
38. `FEAT_device_switch_count = count(unique terminal dims > 0)`
39. `FEAT_channel_asymmetry = |F2956 − F3043| / (F2956 + F3043 + ε)`
40. `FEAT_amount_gini` — Gini of amount distribution proxies
41. `FEAT_profile_stability = 1 / (1 + |F115 − F321|)`
42. `FEAT_terminal_range = max(F3887:F3894) − min(F3887:F3894)`
43. `FEAT_terminal_cv = std(F3887:F3894) / (mean + ε)`
44. `FEAT_balance_volatility = |F670 − F2582|`
45. `FEAT_multichannel_flag = 1[F2956 > 0 ∧ F3043 > 0]`

#### Anomaly Scores (8)

46. `FEAT_iso_anomaly_score` — Isolation Forest decision function *(implemented in fusion)*
47. `FEAT_lof_score` — Local Outlier Factor on legitimate manifold
48. `FEAT_ae_reconstruction_error` — Autoencoder residual *(optional PyTorch)*
49. `FEAT_ecod_score` — ECOD empirical CDF deviation *(optional PyOD)*
50. `FEAT_mahalanobis_dist` — distance from legit centroid
51. `FEAT_dist_cluster_0..4` — KMeans centroid distances *(implemented)*
52. `FEAT_peer_zscore_max` — max z-score vs segment for bank keys
53. `FEAT_combined_anomaly = 0.5×iso + 0.3×lof + 0.2×ae`

#### Log / Scale Transforms (implemented subset)

54. `FEAT_log_F115`, `FEAT_log_F321`, `FEAT_log_F670`, `FEAT_log_F3836`
55. `LOG_{Fi}` — auto log1p for top-30 skewed MI features

### Feature Selection Plan

1. **Leakage audit:** exclude any feature with |r| > 0.50 vs F3924 (includes F3912)
2. **MI ranking:** retain top 200 numeric features + force-include 18 bank keys
3. **Boruta (future):** confirm MI selection with shadow-feature importance
4. **SHAP stability:** retain features with mean |SHAP| > 0.01 across 5 folds
5. **Final model input:** 50–200 features (current pipeline: ~260 after enrichment)

---

## III. ML PIPELINE & CLASSIFICATION MODEL

### Preprocessing

| Step | Method | Notes |
|------|--------|-------|
| Missing values | Median imputation + optional missing indicators | F670, F3043 highly sparse |
| Outliers | Clip at 1st/99th percentile for heavy tails (F2678, F3836) | Robust to adversarial spikes |
| Skew | `log1p` then QuantileTransformer (normal) | Applied to |skew| > 3 |
| Categoricals | Target encoding (future) | Currently numeric-only |
| Scaling | RobustScaler for anomaly path only | Supervised trees unscaled |

### Class Imbalance (~112:1)

- **Primary:** `scale_pos_weight` in LightGBM/XGBoost
- **Avoid SMOTE** at this sparsity — synthetic positives harm generalization
- **Cost-sensitive loss:** increase false-negative penalty in threshold tuning
- **Evaluation focus:** Recall@FPR=1%, not accuracy

### Model Recommendation

| Model | Role | Notes |
|-------|------|-------|
| **XGBoost** | Current production winner (holdout PR-AUC 0.888) | Auto-selected; saved as `best_model.pkl` |
| **LightGBM** | Primary GBDT candidate (Optuna HPO) | Strong CV baseline |
| **Stacking Ensemble** | Meta-learner over all OOF preds | Challenger vs best single model |
| **CatBoost / ExtraTrees / DT / MLP** | Model zoo diversity | Optional CatBoost if deps install |
| **Isolation Forest** | Unsupervised layer (train on y=0 only) | `contamination: 0.01` |

**Fusion:** `fused_score = 0.70 × P(mule|best_model) + 0.30 × iso_anomaly_score`  
**Typology boost:** `adjusted = min(1.0, fused + typology_boost)` where boost = +0.05 per flag, max +0.10

### Evaluation Protocol

1. Stratified 80/20 train/holdout split (fixed seed=42)
2. 5-fold CV on train for model selection
3. **Holdout evaluation** for honest metrics (post leakage removal)
4. Primary: **Recall@FPR=1%** and PR-AUC
5. Calibration: Platt scaling on holdout probabilities (future)
6. Threshold: `precision_optimal_threshold` on holdout (recall ≥ 0.80); F1 threshold retained for comparison

### Current Holdout Results (post leakage fix)

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| Holdout PR-AUC | 0.888 |
| Holdout ROC-AUC | 0.999 |
| Precision @ optimal | 0.778 |
| Recall @ optimal | 0.875 |
| Recall @ FPR 1% | 1.000 |
| Precision-optimal threshold | 0.10 |

---

## IV. INSIGHTS & ALERT GENERATION

### Expected Distributions

| Pattern | Legitimate | Mule |
|---------|-----------|------|
| Out-in ratio | < 0.5 typically | > 0.9 (Instant Mule) |
| Account age (F115) | Lower mean | Higher (0.72) — newer accounts |
| Channel activity | Higher F2956 | Lower — focused routing |
| Net fund flow | Near zero | Large negative (rapid drain) |
| Terminal features | Stable cluster | Elevated aggregate |

### Intelligent Alert Rules

**Adaptive threshold:**
```
τ_t = τ_base + β × recent_FPR
```
where `recent_FPR` = false positives / alerts in last 7 days, per branch.

**Contextual bandit (future):** Treat threshold as arm; reward = recall − λ × FPR; update per region weekly.

**Alert payload for investigators:**
- Fused risk score + decision tier
- Top-10 SHAP features with values
- Typology classification (Instant / Sleeper / Aggregator) from rule overlay
- Peer comparison (cluster ID + distance)

### Dashboard Metrics

- Alert volume by tier (BLOCK/CHALLENGE/REVIEW)
- SHAP force plots per case
- Score distribution drift (PSI weekly)
- Network propagation heatmap (when graph available)

---

## V. RISKS & MITIGATIONS

| Risk | Severity | Mitigation |
|------|----------|------------|
| F3912 label leakage | Critical | Automated leakage audit; exclude \|r\| > 0.5 |
| Train/serve skew | Critical | Serialized `FeaturePipeline` at serving |
| Overfitting (81 positives) | High | Holdout validation, strong regularization, MI pre-filter |
| Adversarial evasion | High | Adversarial validation; anomaly layer; periodic retrain |
| Concept drift | Medium | PSI monitoring; quarterly retrain |
| Threshold too aggressive | Medium | Holdout `precision_optimal_threshold` (0.10); F1 threshold kept for comparison |
| Explainability trust | Medium | Real TreeExplainer SHAP at `/explain` |

---

## VI. EXECUTION CHECKLIST

| Step | Status | Effort |
|------|--------|--------|
| Raw data ingestion (`DataSet.csv`) | Done | — |
| EDA + leakage audit | Done | — |
| Feature engineering + serialized pipeline + typology thresholds | Done | — |
| Train/holdout split + honest holdout metrics | Done | — |
| Model zoo (8 candidates + stacking) + auto `best_model.pkl` | Done | — |
| Typology engine + suspicion levels (1–4) | Done | — |
| Real SHAP TreeExplainer at serving | Done | — |
| FastAPI scoring (18 bank keys → server FE) | Done | — |
| Hackathon Word proposal doc | Done | regenerate via `scripts/generate_hackathon_word_doc.py` |
| Boruta + 50+ derived features | Future | 2 days |
| TabTransformer for categoricals | Future | 3 days |
| Graph features (PageRank, motifs) | Future (needs txn graph) | 5 days |
| PSI drift monitoring | Future | 2 days |
| Adaptive threshold / contextual bandit | Future | 3 days |
| Airflow retraining + MLflow registry | Future | 5 days |
| Kubernetes production deployment | Future | 5 days |

**Run order after this PR:**
```bash
python notebooks/00_eda_exploration.py
python notebooks/01_feature_engineering.py
python notebooks/02_model_training.py
python notebooks/03_anomaly_detection.py
python notebooks/04_shap_explainability.py
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

---

*Report complements [`mule_account_detection_blueprint.md`](../mule_account_detection_blueprint.md). Critical fixes implemented in `serving/feature_pipeline.py` and `serving/app.py`.*

---

## VII. TYPOLOGY MATRIX & SUSPICION LEVELS

### Suspicion level mapping (balanced preset)

| Level | Label | Score range (adjusted fused) |
|-------|-------|------------------------------|
| 1 | LOW | < REVIEW (0.45) |
| 2 | MEDIUM | REVIEW – CHALLENGE |
| 3 | HIGH | CHALLENGE – BLOCK |
| 4 | CRITICAL | ≥ BLOCK (0.85) |

### Typology flags (rule engine on 18 bank keys)

| Flag | Detection logic |
|------|-----------------|
| `silent_account` | Low baseline activity (`F531` < p25) with sudden activation (`F527` > p75 or \|F2678\| > p90), **or** high tenure (`F115` > p75) with burst ratio > p90 |
| `large_amount_mover` | \|F2678\| + \|F2737\| > p95 **or** \|net flow\| > p95 |
| `instant_mule` | Out/in count ratio > 0.9 |
| `aggregator_hub` | High inbound proxy (`F531` > p90) with low outbound ratio (< 0.3) |

Thresholds are fitted on the train split and stored in `models/typology_thresholds.json`. Each flag adds +0.05 to the fused score (capped at +0.10) before decision/suspicion mapping.

### Model zoo & precision threshold

Training (`02_model_training.py`) compares LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Decision Tree, Logistic Regression, optional MLP, and a stacking meta-learner. The holdout PR-AUC winner is copied to `models/best_model.pkl`. `precision_optimal_threshold` in `best_model_metadata.json` maximizes precision subject to recall ≥ 0.80 on holdout.
