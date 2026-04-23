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
│   ├── causal_diagnostics.py         # CausalDiagnostics class (~2,500 lines)
│   └── causal_inference_modelling.py # CausalInferenceModel class (~5,100 lines)
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
| `data/generate_data.py` | ~1,080 | Data generation + Excel reporting |
| `causal_diagnostics.py` | ~2,500 | All pre-modeling & balance diagnostics |
| `causal_inference_modelling.py` | ~5,100 | IPTW/GEE, Cox PH survival (with time interactions), RMST with bootstrap CI, DML (PLR + Causal Forest), E-values, summary tables |
| `causal_inference_workshop.ipynb` | 83 cells (44 code + 39 markdown) | Interactive walkthrough of the full pipeline |

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
| `nbconvert` | ≥7.16.0 | Export the notebook to `causal_inference_workshop_no_code.html` for workshop handouts (`python -m nbconvert --to html --no-input ...`) |

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
| **`analyze_treatment_effect()`** | Full IPTW + covariate-adjusted GEE pipeline for survey outcomes (ATE or ATT) |
| **`analyze_survival_effect()`** | End-to-end IPTW + Cox PH pipeline with time interaction for time-to-event outcomes (internal orchestrator; the notebook now drives the retention pipeline step-by-step via the building blocks below for teaching clarity) |
| **`dml_cluster_robust_ate()`** | DoubleML PLR with cluster-aware cross-fitting and cluster-robust inference (`doubleml`) — ATE robustness check |
| **`dml_estimate_treatment_effects()`** | Causal Forest DML via `econml` for CATE / heterogeneity exploration |
| **`prepare_survival_data()`** | Convert departure dates → `days_observed` + `departed` + `departure_quarter` |
| **`_fit_weighted_km_curves()`** | IPTW-weighted Kaplan–Meier survival curves (descriptive / exploratory visualization of the retention gap) |
| **`compute_rmst_difference()`** | Restricted Mean Survival Time difference with **bootstrap CI** — the **primary causal estimand** for retention |
| **`_fit_cox_model()`** | Cox PH fitter used for both the overall HR (+ Schoenfeld PH test) and the period-specific HRs via `time_interaction='categorical'` (inferential model when PH is violated) |
| **`plot_survival_curves()`** | IPTW-weighted KM curves with risk table + HR annotation (plotting wrapper) |
| **`build_summary_table()`** | Consolidates GEE results across outcomes with FDR correction |
| **`build_survival_summary_table()`** | Consolidates survival results (HR, snapshots, optional RMST) |
| **`compute_evalue()`** | Single-effect E-value (auto-detects Cohen's d vs risk ratio) |
| **`compute_evalues_from_results()`** | E-value sensitivity analysis across a dict of results |
| **`compute_confounder_evalue_benchmarks()`** | Calibrates E-values against strongest measured confounders (SMD → RR → E-value) |

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

**Retention pipeline (driven step-by-step from the notebook for pedagogical transparency):**
1. `prepare_survival_data()` → build `days_observed` + `departed` event indicator from `exit_date`
2. IPTW weights come from the **same propensity-score pipeline** used for the survey analysis (shared `_prepare_iptw_data()` / `estimate_propensity_weights()`) — retention does not re-estimate the PS
3. `_fit_weighted_km_curves()` → IPTW-weighted KM curves for **descriptive visualization** of the retention gap (snapshot differences at 3/6/9/12 months; no formal CIs attached to the snapshots)
4. `compute_rmst_difference()` → **primary causal estimand** — IPTW-weighted RMST difference with **bootstrap CI** (n=500). This is the stakeholder-friendly headline.
5. `_fit_cox_model()` (standard) → overall HR + **Schoenfeld PH test** to check for non-proportional hazards
6. `_fit_cox_model(time_interaction="categorical", period_breaks=[0,90,180,270,365])` → **inferential model**: period-specific HRs (0–3 / 3–6 / 6–9 / 9–12 months) that recover the strongly time-varying treatment effect built into the DGP
7. `compute_evalue()` on the overall HR + balance check via `compute_balance_df()`

**Why the notebook unpacks the orchestrator:** `analyze_survival_effect()` can run the whole survival pipeline in one call, but Checkpoint 4 in the notebook decomposes it into the five steps above so learners can see the KM / RMST / Cox distinction explicitly — KM is descriptive, RMST is the primary causal estimand, Cox is the inferential model.

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

See [`data/ground_truth.md`](data/ground_truth.md) for the full specification — this table is a summary.

| Outcome | Base True Effect | Heterogeneity (continuous gradients) |
|---------|------------------|--------------------------------------|
| `manager_efficacy_index` | **d = 0.33** | **Top HTE — `num_direct_reports`**: +0.15 × `nd_reports_scaled` extra *d* (larger teams → larger effect). **Second HTE — low tenure**: +0.05 × `low_tenure_scaled` extra *d* (newer managers → slightly larger effect). HTE gradients apply **only** to this outcome. |
| `workload_index_mgr` | **d = 0** | No treatment term — outcome is baseline + noise ("no harm" finding by design) |
| `stay_intention_index_mgr` | **d = 0.10** | Homogeneous (no HTE). Higher score = more intention to stay |
| Retention (3/6/9/12 mo) | See §7 of [`ground_truth.md`](data/ground_truth.md) | **Deliberately non-proportional**: strong 0–3 mo effect (target HR ≈ 0.27), attenuated 3–6 mo (target HR ≈ 0.64), null after 6 mo. Encoded via `exit_date`; intermediate binary retention flags are NOT in `manager_data.csv`. |

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

The notebook has **83 cells** (44 code + 39 markdown) organized into a five-checkpoint learning flow, plus a "Bonus: Further Learning" section. Several code cells render HTML slideshows with `from IPython.display import HTML; HTML(r"""…""")` — these visual aids are a hard exclusion when refreshing interpretation markdown (see the `update-markdown-interpretations` skill).

| Checkpoint | What Happens |
|------------|--------------|
| **Checkpoint 1: Context & Overview** | Case study, timeline, data & outcomes, causal DAG, Colab setup, imports, class instantiation, load `manager_data.csv` |
| **Checkpoint 2: Diagnostics & Causal Identification** | VIF / intercorrelations, overlap diagnostics, positivity violation (low performers hard-excluded from treated), estimand feasibility assessment |
| **Checkpoint 3: Survey Modeling** | IPTW + covariate-adjusted GEE for the three survey outcomes (manager efficacy, workload, stay intention); FDR-corrected summary table; **E-value sensitivity + confounder-benchmark calibration** |
| **Checkpoint 4: Retention Modeling** | Step-by-step IPTW-weighted survival analysis: KM curves (descriptive), **RMST difference with bootstrap CI (primary causal estimand)**, Cox PH + Schoenfeld PH test, **period-specific Cox HRs via time interaction** (inferential) + E-values |
| **Checkpoint 5: Key Takeaways for Stakeholders** | Technical-to-stakeholder translation; required bold-red "human review" notice in the What-We-Found cell; What-This-Means / Recommendations / Bottom Line sections |
| **Bonus: Further Learning** | (1) Cluster-robust DoubleML PLR as an alternative ATE identification strategy for manager efficacy; (2) **Causal Forest (econml) HTE** on manager efficacy restricted to actionable effect modifiers — recovers `num_direct_reports` and `tenure_months` as the top moderators built into the DGP |

Data is loaded from `data/manager_data.csv`. The notebook drops `propensity_score` if present (leaked from data generation).

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
```

**Retention / Survival Pattern (step-by-step, as taught in Checkpoint 4):**

```python
# 1. Prepare survival data
data = causal_model.prepare_survival_data(
    data, departure_date_col='exit_date', treatment_col='treatment',
    date_format='mixed',
)

# 2. IPTW-weighted KM curves — descriptive / exploratory visualization
km_results = causal_model._fit_weighted_km_curves(
    data=data, treatment_col='treatment',
    time_col='days_observed', event_col='departed', weights=iptw_weights,
)

# 3. RMST difference with bootstrap CI — PRIMARY causal estimand
rmst_result = CausalInferenceModel.compute_rmst_difference(
    data=data, time_col='days_observed', event_col='departed',
    treatment_col='treatment', weights=iptw_weights,
    tau=365, n_bootstrap=500,
)

# 4. Standard Cox PH — overall HR + Schoenfeld PH test
cox_standard = causal_model._fit_cox_model(
    data=data, time_col='days_observed', event_col='departed',
    treatment_col='treatment', weights=iptw_weights,
)

# 5. Time-interaction Cox — inferential period-specific HRs (PH violated)
cox_time_interaction = causal_model._fit_cox_model(
    data=data, time_col='days_observed', event_col='departed',
    treatment_col='treatment', weights=iptw_weights,
    time_interaction='categorical',
    period_breaks=[0, 90, 180, 270, 365],
    period_labels=['0-3mo', '3-6mo', '6-9mo', '9-12mo'],
)

# 6. Overall-HR E-value for sensitivity analysis
evalue_result = CausalInferenceModel.compute_evalue(
    effect=cox_standard['hr'], ci_lower=cox_standard['hr_ci_lower'],
    ci_upper=cox_standard['hr_ci_upper'], effect_type='risk_ratio',
)
```

The `analyze_survival_effect()` method wraps steps 2–5 into a single call and is still available; the notebook unpacks it so learners see each estimand separately. The retention technical summary in Cell 63 explicitly labels KM as descriptive, RMST as the primary causal estimand, and the time-interaction Cox as the inferential model.

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
| KM is descriptive, not inferential | IPTW-weighted KM curves are used in Checkpoint 4 purely for **visualization** of the retention gap; Greenwood-style confidence bands ignore PS uncertainty and are not valid for inference. The **primary causal estimand** for retention is the **RMST difference (with bootstrap CI)**; period-specific **Cox HRs** are the inferential model. |
| PH assumption is violated by design | The DGP deliberately builds in a front-loaded treatment effect (strong 0–3 mo → null after 6 mo). The Schoenfeld test rejects PH (p ≈ 0.0006 at seed 42), so the **single overall Cox HR** (≈0.65) is a time-averaged quantity and should **not** be the stakeholder headline — use RMST + period-specific HRs. |
| HTE effect modifiers are intentionally restricted | The Causal Forest is fit with the full covariate set as confounding controls `W`, but the **effect-modifier set `X` is restricted** to actionable variables (`organization`, `performance_rating`, `region`, `age`, `tenure_months`, `num_direct_reports`, `tot_span_of_control`). `gender` and `job_family` are intentionally excluded to avoid surfacing fairness-sensitive or high-cardinality noisy moderators. The forest correctly recovers `num_direct_reports` and `tenure_months` as the top moderators — matching the DGP. |
| HTE applies only to manager efficacy | By design, the DGP adds heterogeneous treatment effects **only** to `manager_efficacy_index`. `workload_index_mgr` and `stay_intention_index_mgr` are homogeneous. Running the Causal Forest on those outcomes would not produce meaningful moderator signal. |
| Report-generation helpers removed | Older snapshots of this repo exposed `generate_gee_summary_report()`, `generate_survival_summary_report()`, and `generate_comparison_table()` on `CausalInferenceModel`. Those methods have been **removed**; the notebook writes technical summaries manually in markdown cells (see Cells 42 and 63). |

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
| **PH assumption** | Proportional Hazards assumption — the HR is **constant over time**. When violated (as in this workshop), a single overall HR time-averages a changing effect and can mislead; remedy is a time-interaction Cox (period-specific HRs) or switch to RMST |
| **Schoenfeld residuals / Grambsch–Therneau test** | Statistical test for the PH assumption; a small p-value indicates PH is violated |
| **KM** | Kaplan–Meier — non-parametric survival estimator; in this workshop it is used **descriptively** (not as a formal causal estimand) to visualize the retention gap |
| **RMST** | Restricted Mean Survival Time — expected survival time up to a horizon τ (area under the survival curve from 0 to τ). In this workshop, the **IPTW-weighted RMST difference with a bootstrap CI** is the **primary causal estimand for retention**: the average additional days retained per trained manager within the study window |
| **DML / PLR** | Double Machine Learning / Partially Linear Regression — an alternative ATE identification strategy that residualizes the outcome and treatment against ML nuisance models before estimating the treatment coefficient. The workshop uses `doubleml.DoubleMLPLR` with cluster-aware cross-fitting for a like-for-like comparison against IPTW+GEE |
| **Causal Forest** | `econml.CausalForestDML` — ML estimator for individualized CATEs using the DML framework. Fit with the full covariate set as controls `W` but a restricted **actionable** set as effect modifiers `X` to surface interpretable heterogeneity |
| **Confounder E-value benchmark** | E-value computed from the pre-weighting SMD of a measured covariate, used to calibrate treatment E-values — answers "how much stronger would an unmeasured confounder need to be vs. our strongest measured one?" |

---

*Last updated: 2026-04-23*
