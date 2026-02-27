# TEAM_README — Reviewer Onboarding Guide

> **Purpose:** Get teammates up to speed quickly so they can review, critique, and provide feedback on this SIOP 2026 Master Tutorial codebase.

---

## 1. What This Project Does (30-Second Summary)

This is a **hands-on workshop** for teaching causal inference methods to I/O psychology practitioners at SIOP 2026. It evaluates whether a fictional **leadership development program** causally improves manager outcomes (efficacy, workload, turnover intention, retention) using **observational (non-randomized) data**.

The full pipeline:

```
Synthetic data generation → Pre-modeling diagnostics → IPTW + Doubly Robust GEE → DML (optional) → Sensitivity analysis
```

**Key teaching points:** IPTW weighting, doubly robust estimation, ATE vs ATT estimands, covariate balance, E-value sensitivity, and heterogeneous treatment effects via Double Machine Learning.

---

## 2. Repository Map

```
├── scenario2_workshop.ipynb          # Main teaching notebook (run this)
├── s2_generate_data.py               # Deterministic synthetic data generator (seed=42)
├── requirements.txt                  # Pinned Python dependencies
├── README.md                         # Public-facing README (Colab setup instructions)
├── TEAM_README.md                    # ← You are here
│
├── supp_functions/
│   ├── causal_diagnostics.py         # CausalDiagnostics class (2,393 lines)
│   └── causal_inference_modelling.py # IPTWGEEModel class (2,819 lines)
│
├── data/
│   ├── s2_manager_data.csv           # Pre-generated dataset (9,000 rows × 25 cols)
│   └── s2_data_descriptives.xlsx     # Formatted Excel descriptives (8 sheets)
│
├── pregenereated_results/s2/
│   └── s2_overlap_diagnostics_summary.txt  # Reference overlap diagnostics
│
└── results/                          # Workshop-generated output (created at runtime)
```

### File Size Reference

| File | Lines | Role |
|------|-------|------|
| `s2_generate_data.py` | ~1,038 | Data generation + Excel reporting |
| `causal_diagnostics.py` | ~2,393 | All pre-modeling & balance diagnostics |
| `causal_inference_modelling.py` | ~2,819 | IPTW/GEE, DML, summary tables, sensitivity, reports |
| `scenario2_workshop.ipynb` | 34 cells | Interactive walkthrough of the full pipeline |

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
python s2_generate_data.py

# 4. Open and run the workshop notebook
#    Run cells sequentially in VS Code or Jupyter
```

### Google Colab (Primary Target)

See the public [README.md](README.md) for Colab clone + setup instructions.

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `statsmodels` | 0.14.2 | GEE, propensity score models, VIF |
| `scikit-learn` | 1.5.1 | Random Forest (diagnostics + DML nuisance models) |
| `econml` | ≥0.15.0 | Double Machine Learning (Linear DML, Causal Forest) |
| `scipy` | 1.13.1 | Statistical tests |
| `matplotlib` / `seaborn` | 3.8.4 / 0.13.2 | Plotting |
| `openpyxl` | 3.1.5 | Excel export with formatting |

---

## 4. Architecture Deep Dive

### Two-Class Design

All reusable logic lives in two stateful classes, instantiated in the notebook:

```python
import sys
sys.path.append('./supp_functions')

from causal_diagnostics import CausalDiagnostics
from causal_inference_modelling import IPTWGEEModel

cd = CausalDiagnostics()
model = IPTWGEEModel()
```

> **Note:** There is no package install — imports rely on `sys.path.append`.

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

### `IPTWGEEModel` (causal_inference_modelling.py)

Two complementary analysis approaches + reporting utilities:

| Method | Purpose |
|--------|---------|
| **`analyze_treatment_effect()`** | Full IPTW + doubly robust GEE pipeline (single entry point) |
| **`dml_estimate_treatment_effects()`** | Double Machine Learning via `econml` (Linear DML + Causal Forest) |
| **`build_summary_table()`** | Consolidates results across outcomes with FDR correction |
| **`compute_evalues_from_results()`** | E-value sensitivity analysis for unmeasured confounding |
| **`generate_summary_report()`** | Markdown narrative report with tables, balance verification, E-values |
| **`generate_comparison_table()`** | ATE vs ATT side-by-side Markdown comparison |

**Internal pipeline of `analyze_treatment_effect()`:**
1. Data prep → one-hot encode categoricals → clean column names
2. Propensity score estimation (GEE if clustered, GLM otherwise)
3. IPTW weight computation (ATE or ATT formula, stabilized + trimmed)
4. Propensity overlap + weight distribution plots
5. Post-weighting balance check via `CausalDiagnostics.compute_balance_df()`
6. Doubly robust GEE outcome model (Gaussian or Binomial auto-detected)
7. Effect size metrics (Cohen's d, % change)
8. Optional Excel export

---

## 5. The Synthetic Dataset

Generated by `s2_generate_data.py` with `seed=42`. Key design decisions:

| Feature | Detail |
|---------|--------|
| **N** | 9,000 managers (~500 treated, ~8,500 control) |
| **Self-selection** | Driven by `organization` (R&D/Digital higher) and `performance_rating` (higher performers more likely) |
| **Hard blocks** | Below/Far Below performers are **never** treated |
| **New managers** | ~25% have `baseline_manager_efficacy = 0` (no prior manager data) |
| **Clustering** | Managers nested in within-organization teams (`team_id`, size 5–12) |
| **Built-in ground truth** | Known effect sizes for validation (see below) |

### Built-In Treatment Effects (Ground Truth)

| Outcome | True Effect | Notes |
|---------|-------------|-------|
| `manager_efficacy_index` | d = 0.50 | R&D gets extra +0.15 d |
| `workload_index_mgr` | d = 0.02 | Intentionally non-significant |
| `turnover_intention_index_mgr` | d = 0.25 | New managers get extra +0.20 d |
| `retention_3month` | OR = 2.0 | New managers get extra OR × 1.5 |
| `retention_6month` | Conditional on 3-month survival | Cumulative ~93% treated vs ~86% control |
| `retention_9month` | Conditional cascade | ~91% vs ~83% |
| `retention_12month` | Conditional cascade | ~89% vs ~80% |

### Variable Glossary

| Variable | Type | Role |
|----------|------|------|
| `treatment` | Binary 0/1 | Treatment indicator |
| `team_id` | Integer | Clustering variable |
| `organization` | Categorical (6 levels) | Confounder (drives selection) |
| `performance_rating` | Categorical (5 levels) | Confounder (drives selection) |
| `region`, `job_family`, `gender` | Categorical | Covariates |
| `age`, `tenure_months` | Continuous | Covariates |
| `is_new_manager` | Binary | Effect modifier |
| `num_direct_reports`, `tot_span_of_control` | Continuous | Covariates |
| `baseline_manager_efficacy` | Continuous (0 for new mgrs) | **Baseline** — outcome model only, excluded from PS |
| `baseline_workload` | Continuous | **Baseline** — outcome model only |
| `baseline_turnover_intention` | Continuous | **Baseline** — outcome model only |
| `propensity_score` | Continuous | True PS from data generation |
| `manager_efficacy_index` | Continuous 1–5 | **Outcome** |
| `workload_index_mgr` | Continuous 1–5 | **Outcome** |
| `turnover_intention_index_mgr` | Continuous 1–5 | **Outcome** |
| `retention_Xmonth` | Binary 0/1 | **Outcome** (3, 6, 9, 12 months) |

### Covariate Conventions in Code

The notebook and classes split covariates into three lists:

```python
categorical_vars = ['organization', 'region', 'job_family', 'performance_rating', 'gender']
binary_vars = ['is_new_manager']
continuous_vars = ['age', 'tenure_months', 'num_direct_reports', 'tot_span_of_control']
```

**Baseline variables** (prefixed `baseline_`) are included in the **GEE outcome model** for doubly robust adjustment but **excluded** from the **propensity score model**.

---

## 6. Notebook Walkthrough (scenario2_workshop.ipynb)

The notebook has **34 cells** organized into these sections:

| Section | Cells | What Happens |
|---------|-------|-------------|
| **Setup & Imports** | 1–3 | Install packages, import classes, load data |
| **Exploratory Data Analysis** | 4–7 | Demographics, treatment rates, crosstabs, distributions |
| **Pre-Modeling Diagnostics** | 8–10 | VIF check, intercorrelation check, overlap diagnostics |
| **IPTW + GEE Analysis (ATE)** | 11–13 | `analyze_treatment_effect()` loop over survey + retention outcomes |
| **Summary & Sensitivity (ATE)** | 14–15 | `build_summary_table()`, `compute_evalues_from_results()`, markdown report |
| **IPTW + GEE Analysis (ATT)** | 16–19 | Same pipeline with `estimand="ATT"` |
| **ATE vs ATT Comparison** | 20 | `generate_comparison_table()` |
| **DML Analysis** | 21–25 | `dml_estimate_treatment_effects()` for survey + retention outcomes |
| **DML Summary & Sensitivity** | 26–28 | Summary tables and E-values for DML |
| **Heterogeneous Effects** | 29–34 | CATE estimation, feature importance, tree interpreter |

### Typical Analysis Pattern (Per Outcome Family)

```python
# 1. Run analysis for each outcome
results = {}
for outcome in outcome_vars:
    results[outcome] = model.analyze_treatment_effect(
        data=data, outcome_var=outcome, treatment_var='treatment',
        categorical_vars=categorical_vars, binary_vars=binary_vars,
        continuous_vars=continuous_vars, cluster_var='team_id',
        estimand="ATE", baseline_var=baseline_vars.get(outcome),
    )

# 2. Build FDR-corrected summary table
summary = IPTWGEEModel.build_summary_table(results)

# 3. E-value sensitivity analysis
evalues = IPTWGEEModel.compute_evalues_from_results(results)

# 4. Generate markdown report
report = IPTWGEEModel.generate_summary_report(summary, evalues, results, ...)
```

---

## 7. Key Design Decisions to Review

These are the most impactful choices — feedback is especially valuable here:

### Statistical Methodology

1. **Baseline exclusion from PS model** — Baseline outcomes (`baseline_*`) are excluded from the propensity score model and only included in the GEE outcome model (doubly robust). This avoids outcome-specific PS models. *Is this the right call for this teaching context?*

2. **Stabilized + trimmed IPTW** — Weights are stabilized by marginal treatment probability and trimmed at the 99th percentile by default. *Are the defaults appropriate for the sample size (N=9,000, ~500 treated)?*

3. **GEE sandwich SEs** — Standard errors account for within-cluster correlation but do **not** propagate first-stage PS estimation uncertainty. The docstring acknowledges this. *Should we add a bootstrap option or is the caveat sufficient?*

4. **DML ATT derivation** — ATT from Causal Forest is derived by averaging CATE over treated observations, not a dedicated ATT estimator. *Is this adequately flagged for a teaching audience?*

5. **FDR correction** — Applied across outcomes in `build_summary_table()`, not within individual models. *Correct statistical practice, but is it explained clearly enough?*

6. **E-value computation** — Cohen's d → RR via VanderWeele (2017) formula `RR ≈ exp(0.91 * d)`. Binary outcomes use log-odds. *Are the auto-detection heuristics reliable?*

### Pedagogical Design

7. **Two-class architecture** — All logic in `CausalDiagnostics` + `IPTWGEEModel`, no package install, `sys.path.append`. *Clean for a workshop? Or should we make it pip-installable?*

8. **Overlap diagnostics depth** — `check_covariate_overlap()` produces extremely detailed output with interpretation guides, ASCII boxes, and tier-based estimand recommendations. *Is the verbosity appropriate for the audience (I/O psych practitioners)?*

9. **Ground-truth dataset** — Known effect sizes allow attendees to verify their analyses. The workload outcome is intentionally null. *Are the effect sizes realistic for this domain?*

10. **Column name sanitization** — `_clean_column_name()` replaces special characters for patsy formula compatibility. *Could this cause confusion if attendees bring their own data?*

### Code Quality

11. **No test suite** — Teaching repo, no unit tests. *Should we add at least smoke tests?*

12. **`generate_summary_report()`** — Auto-generates Markdown narratives with tables, balance checks, E-values, OR interpretation, and trend detection. *Is the auto-generated text accurate and helpful, or could it mislead?*

13. **Excel formatting helpers** — Defined inline in `s2_generate_data.py` (not importable). `IPTWGEEModel` has its own simpler export. *Should these be consolidated?*

---

## 8. How to Review This Project

### If You Have 15 Minutes

1. Read this document
2. Skim the notebook markdown cells (the teaching narrative between code cells)
3. Check the built-in ground truth table above against the analysis results

### If You Have 1 Hour

1. Run the notebook end-to-end (all 34 cells execute in ~2-3 minutes)
2. Review the overlap diagnostics output (cell 7) — is it understandable?
3. Compare ATE vs ATT results — does the narrative in the comparison table make sense?
4. Check E-value interpretations — are the robustness classifications reasonable?
5. Look at the DML CATE distribution — does heterogeneity match the data generation?

### If You Have a Half Day

1. Read `s2_generate_data.py` to understand the data generating process
2. Trace the full `analyze_treatment_effect()` pipeline in `causal_inference_modelling.py`
3. Review `check_covariate_overlap()` in `causal_diagnostics.py` — check the tier logic
4. Assess whether the teaching narrative flows logically through the notebook
5. Try modifying an outcome or variable list and re-running — does the code handle edge cases?

### Specific Feedback Requests

- [ ] Is the statistical methodology sound? Any errors in the IPTW/GEE implementation?
- [ ] Is the DML section (using `econml`) correctly implemented?
- [ ] Are the auto-generated narrative reports accurate and not misleading?
- [ ] Is the overall flow (diagnostics → IPTW → DML → sensitivity) pedagogically effective?
- [ ] Is the overlap diagnostics output overwhelming or just right for this audience?
- [ ] Are there any I/O psychology domain-specific issues with the framing?

---

## 9. Common Gotchas

| Issue | Explanation |
|-------|-------------|
| `sys.path.append` import | No package install — classes are imported via path manipulation. Must run from repo root. |
| Column name cleaning | `_clean_column_name()` silently renames columns with special chars (e.g., `&` → `and`). Variable lists must use cleaned names downstream. |
| Binary outcome auto-detection | `analyze_treatment_effect()` and `dml_estimate_treatment_effects()` check if outcome values ⊆ {0, 1} and switch to Binomial family / `discrete_outcome=True` automatically. |
| Baseline vars excluded from PS | By design. If you add a new baseline variable, you must pass it as `baseline_var=` (not in `continuous_vars`) for it to be correctly routed. |
| `correction_method` in `analyze_treatment_effect()` | **Deprecated** — this parameter is ignored. FDR correction now happens in `build_summary_table()` across outcomes. |
| DML clustering | DML does **not** handle clustering (`team_id`). Standard errors may be anti-conservative. Use IPTW/GEE for cluster-robust inference. |
| New manager baseline = 0 | `baseline_manager_efficacy` is 0 (not NaN) for new managers. This is intentional — it represents "no prior manager-level data." |

---

## 10. Regenerating Data

```bash
python s2_generate_data.py
```

This overwrites:
- `data/s2_manager_data.csv`
- `data/s2_data_descriptives.xlsx`

The script is deterministic (seed=42). Output should be identical across runs on the same platform. The script also prints extensive verification checks (treatment rates, SMDs, retention rates, statistical tests).

---

## 11. Glossary of Key Terms

| Term | Definition |
|------|-----------|
| **ATE** | Average Treatment Effect — average causal effect across the entire population |
| **ATT** | Average Treatment Effect on the Treated — average effect for those who received treatment |
| **IPTW** | Inverse Probability of Treatment Weighting — reweights observations to create pseudo-balance |
| **GEE** | Generalized Estimating Equations — regression that accounts for clustering |
| **Doubly Robust** | Uses both PS weighting AND outcome model covariates — consistent if either model is correct |
| **DML** | Double Machine Learning — uses ML for nuisance functions, preserves valid inference |
| **CATE** | Conditional Average Treatment Effect — individualized treatment effects |
| **E-value** | Minimum confounding strength (as risk ratio) needed to explain away an observed effect |
| **SMD** | Standardized Mean Difference — measures covariate balance (target: |SMD| < 0.1) |
| **FDR** | False Discovery Rate — multiple testing correction applied across outcomes |
| **ESS** | Effective Sample Size — measures information loss from weighting: ESS = (Σw)² / Σw² |
| **Positivity** | Assumption that all covariate strata have non-zero probability of treatment |

---

*Last updated: 2026-02-27*
