# REPO_ONBOARDING — Reviewer Onboarding Guide

> **Purpose:** Get teammates up to speed quickly so they can review, critique, and provide feedback on this SIOP 2026 Master Tutorial codebase.

---

## 1. What This Project Does (30-Second Summary)

This is a **hands-on workshop** for teaching causal inference methods to I/O psychology practitioners at SIOP 2026. It evaluates whether a fictional **leadership development program** causally improves manager outcomes (efficacy, workload, stay intention, retention) using **observational (non-randomized) data**.

The full pipeline:

```
Synthetic data generation → Pre-modeling diagnostics → IPTW + covariate-adjusted GEE (survey)
→ IPTW + Cox PH with Time Interaction + RMST (retention/survival) → Sensitivity analysis → DML / HTE exploration (optional)
```

**Key teaching points:** IPTW weighting, covariate adjustment in both stages, ATE vs ATT estimands, covariate balance, E-value sensitivity, survival analysis via Cox PH with time interaction plus RMST, and heterogeneous treatment effects via Double Machine Learning.

---

## 2. Repository Map

```
├── causal_inference_workshop.ipynb   # Main teaching notebook (run this)
├── requirements.txt                  # Pinned Python dependencies
├── README.md                         # Public-facing README (Colab setup instructions)
├── REPO_ONBOARDING.md                # ← You are here
│
├── supp_functions/
│   ├── causal_diagnostics.py         # CausalDiagnostics class (~2,092 lines)
│   └── causal_inference_modelling.py # CausalInferenceModel class (~4,102 lines)
│
├── data/
│   ├── generate_data.py              # Deterministic synthetic data generator (seed=42)
│   ├── manager_data.csv              # Pre-generated dataset (9,000 rows × 24 cols)
│   └── data_descriptives.xlsx        # Formatted Excel descriptives (8 sheets)
│
├── diagrams/                         # Checkpoint images, DAG, diagnostic flow
│   ├── checkpoint_overview.png      # Five-checkpoint learning flow
│   ├── checkpoint1.png … checkpoint5.png
│   ├── manager_training_dag.png     # Causal DAG
│   └── DiagnosticFlow.png           # Overlap diagnostics logic
│
├── pregenerated_results/s2/
│   └── s2_overlap_diagnostics_summary.txt  # Reference overlap diagnostics
│
└── results/                          # Workshop-generated output (created at runtime)
```

### File Size Reference

| File | Lines | Role |
|------|-------|------|
| `data/generate_data.py` | ~1,100 | Data generation + Excel reporting |
| `causal_diagnostics.py` | ~2,092 | All pre-modeling & balance diagnostics |
| `causal_inference_modelling.py` | ~4,102 | IPTW/GEE, Cox PH survival, DML, summary tables, sensitivity, reports |
| `causal_inference_workshop.ipynb` | 45 cells | Interactive walkthrough of the full pipeline |

---

## 3. Quick Start

### Local Setup

```bash
# 1. Clone
git clone https://github.com/mlpost/2026-siop-causal-inference-master-tutorial.git
cd 2026-siop-causal-inference-master-tutorial

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate synthetic data
python data/generate_data.py

# 4. Open and run the workshop notebook
#    Run cells sequentially in VS Code or Jupyter
```

### Google Colab (Primary Target)

See the public [README.md](README.md) for Colab clone + setup instructions.

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.2.2 | Data manipulation |
| `numpy` | ≥2.0.0 | Numerical operations |
| `statsmodels` | ≥0.14.2 | GEE, propensity score models, VIF |
| `scikit-learn` | ≥1.6.0 | Random Forest (diagnostics + DML nuisance models) |
| `econml` | ≥0.15.0 | Causal Forest DML for CATE exploration |
| `doubleml` | current | Cluster-robust DoubleML PLR for ATE robustness checks |
| `scipy` | ≥1.14.0 | Statistical tests |
| `matplotlib` / `seaborn` | ≥3.9.0 / ≥0.13.2 | Plotting |
| `lifelines` | ≥0.27.0 | Survival analysis (Cox PH, Kaplan-Meier) |
| `openpyxl` | ≥3.1.5 | Excel export with formatting |

---

## 4. Architecture Deep Dive

### Two-Class Design

All reusable logic lives in two stateful classes, instantiated in the notebook:

```python
import sys
sys.path.append('./supp_functions')

from causal_diagnostics import CausalDiagnostics
from causal_inference_modelling import CausalInferenceModel

cd = CausalDiagnostics()
causal_model = CausalInferenceModel()
```

> **Note:** There is no package install — imports rely on `sys.path.append`. The class was renamed from `IPTWGEEModel` to `CausalInferenceModel` to reflect the addition of survival analysis alongside GEE. A backward-compatible alias `IPTWGEEModel = CausalInferenceModel` is provided.

### `CausalDiagnostics` (causal_diagnostics.py)

Organized into five method groups:

| Group | Methods | What It Does |
|-------|---------|-------------|
| **A) Pre-Modeling** | `check_vif()`, `check_high_intercorrelations()`, `show_low_proportion_groups()` | Multicollinearity (VIF/GVIF), intercorrelation screening, sparse cell detection |
| **B) Overlap** | `run_overlap_diagnostics()`, `check_covariate_overlap()`, `prepare_adjustment_set_for_overlap()` | Full overlap pipeline: univariate SMDs, propensity score AUC, common support, estimand feasibility tiers |
| **C) Balance** | `compute_balance_df()` | Post-weighting balance: unweighted vs. weighted SMDs for all covariates |
| **D) Visualization** | `plot_propensity_overlap()`, `save_overlap_diagnostics_summary()` | PS density plots, text report export |
| **E) Help** | `help()` | Prints all available methods with descriptions |

**Key overlap assessment tiers** (from `check_covariate_overlap()`):
1. **ATE Clean** — both groups well-covered, low AUC, no severe imbalance
2. **ATE with Caution** — decent overlap, some residual concerns
3. **ATT Feasible** — treated covered but controls lack overlap for ATE
4. **ATT with Trimming** — heavy extrapolation needed
5. **Causal Inference Questionable** — insufficient overlap for any estimand

### `CausalInferenceModel` (causal_inference_modelling.py)

Three complementary analysis approaches + reporting utilities:

| Method | Purpose |
|--------|---------|
| **`analyze_treatment_effect()`** | Full IPTW + covariate-adjusted GEE pipeline for survey outcomes |
| **`analyze_survival_effect()`** | Full IPTW + Cox PH pipeline with time interaction for time-to-event outcomes |
| **`dml_cluster_robust_ate()`** | DoubleML PLR with cluster-aware cross-fitting and cluster-robust inference |
| **`dml_estimate_treatment_effects()`** | Double Machine Learning via `econml` for CATE exploration |
| **`prepare_survival_data()`** | Convert departure dates → `days_observed` + `departed` + `departure_quarter` |
| **`plot_survival_curves()`** | IPTW-weighted Kaplan-Meier curves with risk table + HR annotation |
| **`build_summary_table()`** | Consolidates GEE results across outcomes with FDR correction |
| **`build_survival_summary_table()`** | Consolidates survival results (HR, snapshots, optional RMST) |
| **`compute_evalues_from_results()`** | E-value sensitivity analysis (auto-detects Cohen's d vs risk ratio) |
| **`compute_confounder_evalue_benchmarks()`** | Calibrates E-values against strongest measured confounders (SMD → RR → E-value) |
| **`compute_rmst_difference()`** | Restricted mean survival time difference with bootstrap CI |
| **`generate_gee_summary_report()`** | Markdown narrative report for survey (GEE) outcomes |
| **`generate_survival_summary_report()`** | Markdown narrative report for survival outcomes (HR, RMST, KM) |
| **`generate_comparison_table()`** | ATE vs ATT side-by-side Markdown comparison |

**Shared IPTW infrastructure:** Both `analyze_treatment_effect()` and `analyze_survival_effect()` delegate data prep, one-hot encoding, column sanitization, propensity score estimation, weight diagnostics, overlap/weight plotting, and balance checking to a shared private method `_prepare_iptw_data()` (~300 lines).

**Internal pipeline of `analyze_treatment_effect()`:**
1. Data prep → one-hot encode categoricals → clean column names
2. Propensity score estimation (GEE if clustered, GLM otherwise)
3. IPTW weight computation (ATE or ATT formula, stabilized + trimmed)
4. Propensity overlap + weight distribution plots
5. Post-weighting balance check via `CausalDiagnostics.compute_balance_df()`
6. IPTW-weighted GEE outcome model with covariate adjustment (Gaussian or Binomial auto-detected)
7. Effect size metrics (Cohen's d, % change)
8. Optional Excel export

**Internal pipeline of `analyze_survival_effect()`:**
1. Steps 1–5 identical to above (shared via `_prepare_iptw_data()`)
2. Cox PH with time interaction via `_fit_cox_model()`: `time_interaction="categorical"` expands data to person-period format and fits separate HRs per interval (e.g., quarterly); `time_interaction="continuous"` models treatment × time as a linear trend
3. Full-period KM still computed for survival curve plots
4. Returns per-interval HRs, survival snapshots, and balance diagnostics

---

## 5. The Synthetic Dataset

Generated by `data/generate_data.py` with `seed=42`. Key design decisions:

| Feature | Detail |
|---------|--------|
| **N** | 9,000 managers (~500 treated, ~8,500 control) |
| **Self-selection** | Driven by `organization` (R&D/Digital higher) and `performance_rating` (higher performers more likely) |
| **Hard blocks** | Below/Far Below performers are **never** treated |
| **Clustering** | Managers nested in within-organization teams (`team_id`, size 5–12) |
| **Built-in ground truth** | Known effect sizes for validation (see below) |

### Built-In Treatment Effects (Ground Truth)

| Outcome | True Effect | Notes |
|---------|-------------|-------|
| `manager_efficacy_index` | d = 0.20 | Digital +0.15 d; tenure ≤ median +0.10 d (stronger when tenure is lower) |
| `workload_index_mgr` | d = 0.2 | — |
| `stay_intention_index_mgr` | d = 0.15 | Higher score = more intention to stay |
| Retention (3/6/9/12 mo) | See `generate_data.py` | **DGP only** — timing encoded via `exit_date`; binary flags not in `manager_data.csv` |

### Variable Glossary

| Variable | Type | Role |
|----------|------|------|
| `treatment` | Binary 0/1 | Treatment indicator |
| `team_id` | Integer | Clustering variable |
| `organization` | Categorical (6 levels) | Confounder (drives selection) |
| `performance_rating` | Categorical (5 levels) | Confounder (drives selection) |
| `region`, `job_family`, `gender` | Categorical | Covariates |
| `age`, `tenure_months` | Continuous | Covariates |
| `num_direct_reports`, `tot_span_of_control` | Continuous | Covariates |
| `baseline_manager_efficacy` | Continuous (1–5 prior-year) | **Baseline** — outcome model only, excluded from PS |
| `baseline_workload` | Continuous | **Baseline** — outcome model only |
| `baseline_stay_intention` | Continuous | **Baseline** — outcome model only (higher = more intention to stay) |
| `propensity_score` | Continuous | True PS from data generation |
| `manager_efficacy_index` | Continuous 1–5 | **Outcome** (missing if left by 6 months) |
| `workload_index_mgr` | Continuous 1–5 | **Outcome** (missing if left by 6 months) |
| `stay_intention_index_mgr` | Continuous 1–5 | **Outcome** (missing if left by 6 months) |
| `exit_date` | Date (M/D/YYYY) | Departure date (blank if still employed) |
| `days_observed` | Integer | Days from study start to departure/censoring (created by `prepare_survival_data()`) |
| `departed` | Binary 0/1 | Event indicator: 1 = departed, 0 = censored (created by `prepare_survival_data()`) |
| `departure_quarter` | String | Quarter of departure (created by `prepare_survival_data()`) |

### Covariate Conventions in Code

The notebook and classes split covariates into three lists:

```python
categorical_vars = ['organization', 'region', 'job_family', 'performance_rating', 'gender']
binary_vars = []
continuous_vars = ['age', 'tenure_months', 'num_direct_reports', 'tot_span_of_control']
```

**Baseline variables** (prefixed `baseline_`) are included in the **GEE outcome model** for covariate adjustment but **excluded** from the **propensity score model**.

---

## 6. Notebook Walkthrough (causal_inference_workshop.ipynb)

The notebook has **45 cells** (27 code + 18 markdown) organized into a five-checkpoint learning flow:

| Checkpoint | What Happens |
|------------|--------------|
| **Checkpoint 1: Context & Overview** | Case study, timeline, data & outcomes, causal DAG, Colab setup, imports, class instantiation, load `manager_data.csv` |
| **Checkpoint 2: Diagnostics & Causal Identification** | Overlap diagnostics, VIF, intercorrelations, estimand feasibility assessment |
| **Checkpoint 3: Modeling** | IPTW + GEE (ATE/ATT) for survey outcomes, survival setup, Cox PH with time interaction, balance verification, E-values, summary reports, `generate_comparison_table()` |
| **Checkpoint 4: Key Takeaways for Stakeholders** | Technical summary, actionable recommendations for L&D team |
| **Checkpoint 5: Further Learning** | DoubleML ATE robustness check plus econml Causal Forest HTE exploration on `manager_efficacy_index` |

Data is loaded from `data/manager_data.csv`. The notebook drops `propensity_score` if present (generated by older data scripts).

### Typical Analysis Pattern (Per Outcome Family)

```python
# 1. Run analysis for each survey outcome
results = {}
for outcome in survey_outcomes:
    results[outcome] = causal_model.analyze_treatment_effect(
        data=data, outcome_var=outcome, treatment_var='treatment',
        categorical_vars=categorical_vars, binary_vars=binary_vars,
        continuous_vars=continuous_vars, cluster_var='team_id',
        estimand="ATE", baseline_var=baseline_vars.get(outcome),
    )

# 2. Build FDR-corrected summary table
summary = CausalInferenceModel.build_summary_table(results)

# 3. E-value sensitivity analysis
evalues = CausalInferenceModel.compute_evalues_from_results(results)

# 4. Confounder E-value benchmarks (calibration)
benchmarks = CausalInferenceModel.compute_confounder_evalue_benchmarks(results, evalue_df=evalues)

# 5. Generate markdown report
report = CausalInferenceModel.generate_gee_summary_report(summary, evalues, results, ...)
```

**Survival Analysis Pattern:**

```python
# 1. Prepare survival data
data = causal_model.prepare_survival_data(data, 'exit_date', 'treatment', ...)

# 2. Cox PH with time interaction
survival_results = {}
survival_results['retention'] = causal_model.analyze_survival_effect(
    data=data, time_var='days_observed', event_var='departed',
    treatment_var='treatment',
    time_interaction='categorical',
    period_breaks=[0, 90, 180, 270, 365],
    period_labels=['0-3mo', '3-6mo', '6-9mo', '9-12mo'], ...
)

# 3. KM curves + survival summary + E-values
fig = causal_model.plot_survival_curves(survival_results['retention'], ...)
summary = CausalInferenceModel.build_survival_summary_table(survival_results)
evalues = CausalInferenceModel.compute_evalues_from_results(survival_results, effect_type="risk_ratio")
benchmarks = CausalInferenceModel.compute_confounder_evalue_benchmarks(survival_results, evalue_df=evalues)
report = CausalInferenceModel.generate_survival_summary_report(summary, evalues, survival_results, survival_plot_fig=fig)
```

## 7. Common Gotchas

| Issue | Explanation |
|-------|-------------|
| `sys.path.append` import | No package install — classes are imported via path manipulation. Must run from repo root. |
| Column name cleaning | `_clean_column_name()` silently renames columns with special chars (e.g., `&` → `and`). Variable lists must use cleaned names downstream. |
| Binary outcome auto-detection | `analyze_treatment_effect()` and `dml_estimate_treatment_effects()` check if outcome values ⊆ {0, 1} and switch to Binomial family / `discrete_outcome=True` automatically. |
| Baseline vars excluded from PS | By design. If you add a new baseline variable, you must pass it as `baseline_var=` (not in `continuous_vars`) for it to be correctly routed. |
| `correction_method` in `analyze_treatment_effect()` | **Deprecated** — this parameter is ignored. FDR correction now happens in `build_summary_table()` across outcomes. |
| DML clustering | The econml-based CATE analysis (`dml_estimate_treatment_effects()`) does **not** handle clustering (`team_id`), so its standard errors may be anti-conservative. For cluster-robust ATE estimation, use `dml_cluster_robust_ate()` which provides native cluster support via `doubleml`. |
| New manager baseline = 0 | `baseline_manager_efficacy` is 0 (not NaN) for new managers. This is intentional — it represents "no prior manager-level data." |
| Survival analysis date format | `exit_date` uses M/D/YYYY without zero-padding. Pass `date_format='mixed'` to `prepare_survival_data()`. |
| No ATT for retention | The notebook only runs ATE for survival outcomes. ATT is supported but not demonstrated for retention. |
| KM confidence bands | IPTW-weighted KM confidence bands are **not valid** for inference (Greenwood variance ignores PS uncertainty). Use Cox HRs for statistical testing. |

---

## 8. Regenerating Data

```bash
python data/generate_data.py
```

Run from the repo root. This overwrites:
- `data/manager_data.csv`
- `data/data_descriptives.xlsx`

The script is deterministic (seed=42). Output should be identical across runs on the same platform. The script also prints extensive verification checks (treatment rates, SMDs, retention rates, statistical tests).

---

## 9. Glossary of Key Terms

| Term | Definition |
|------|-----------|
| **ATE** | Average Treatment Effect — average causal effect across the entire population |
| **ATT** | Average Treatment Effect on the Treated — average effect for those who received treatment |
| **IPTW** | Inverse Probability of Treatment Weighting — reweights observations to create pseudo-balance |
| **GEE** | Generalized Estimating Equations — regression that accounts for clustering |
| **Covariate Adjustment in Both Stages** | Uses PS weighting plus outcome-model covariates; under linearity, consistency holds if either model is correct |
| **DML** | Double Machine Learning — uses ML for nuisance functions, preserves valid inference |
| **CATE** | Conditional Average Treatment Effect — individualized treatment effects |
| **E-value** | Minimum confounding strength (as risk ratio) needed to explain away an observed effect |
| **SMD** | Standardized Mean Difference — measures covariate balance (target: |SMD| < 0.1) |
| **FDR** | False Discovery Rate — multiple testing correction applied across outcomes |
| **ESS** | Effective Sample Size — measures information loss from weighting: ESS = (Σw)² / Σw² |
| **Positivity** | Assumption that all covariate strata have non-zero probability of treatment |
| **Cox PH** | Cox Proportional Hazards — semi-parametric survival model: $h(t) = h_0(t) \exp(\beta X)$ |
| **HR** | Hazard Ratio — the ratio of departure rates; HR < 1 means treatment is protective |
| **KM** | Kaplan-Meier — non-parametric survival estimator showing retention probability over time |
| **RMST** | Restricted Mean Survival Time — area under survival curve up to a time horizon (available but unused in notebook) |
| **Confounder E-value benchmark** | E-value computed from the pre-weighting SMD of a measured covariate, used to calibrate treatment E-values — answers "how much stronger would an unmeasured confounder need to be vs. our strongest measured one?" |

---

*Last updated: 2026-03-19*
