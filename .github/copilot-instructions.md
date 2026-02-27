# Copilot Instructions — SIOP Causal Inference Master Tutorial

## Project Overview

This is a **workshop/tutorial codebase** for teaching causal inference methods applied to HR/people analytics. It evaluates a leadership development program's causal impact on manager outcomes using observational (non-randomized) data. The primary audience is I/O psychology practitioners at SIOP 2026.

**Core workflow:** Generate synthetic data → Run diagnostics → Estimate treatment effects via IPTW + doubly robust GEE (and/or Double Machine Learning) → Sensitivity analysis.

## Architecture

| Component | Purpose |
|---|---|
| `s2_generate_data.py` | Deterministic synthetic data generator (seed=42). Produces manager-level CSV + Excel descriptives. Run standalone to regenerate `data/`. |
| `scenario2_workshop.ipynb` | Main teaching notebook. Walks through the full causal inference pipeline interactively. |
| `supp_functions/causal_diagnostics.py` | `CausalDiagnostics` class — pre-modeling checks (VIF, intercorrelations, overlap), covariate balance (SMD), propensity score visualization. |
| `supp_functions/causal_inference_modelling.py` | `IPTWGEEModel` class — propensity score estimation, IPTW weights, doubly robust GEE outcome models, Double Machine Learning (DML) via `econml` for ATE/ATT/CATE estimation, E-value sensitivity analysis, summary tables. |
| `data/s2_manager_data.csv` | Pre-generated manager-level dataset (9000 rows). |
| `pregenereated_results/s2/` | Reference diagnostic outputs for comparison. |
| `results/` | Output directory for workshop-generated results (Excel, plots). |

## Key Patterns & Conventions

### Two-class design
All reusable logic lives in two stateful classes instantiated in the notebook:
```python
from causal_diagnostics import CausalDiagnostics
from causal_inference_modelling import IPTWGEEModel
cd = CausalDiagnostics()
model = IPTWGEEModel()
```
Import path relies on `sys.path.append('./supp_functions')` — there is no package install.

### Analysis pipeline (per outcome)
1. **Pre-modeling diagnostics** — `cd.check_vif()`, `cd.check_high_intercorrelations()`, `cd.run_overlap_diagnostics()`
2. **IPTW + Outcome model** — `model.analyze_treatment_effect()` (single entry point that runs propensity scoring → weight estimation → balance check → doubly robust GEE)
3. **DML alternative** (optional) — `model.dml_estimate_treatment_effects()` estimates ATE/ATT via Linear DML and/or CATE via Causal Forest DML. Returns a dict compatible with `build_summary_table()` and `compute_evalues_from_results()`.
4. **Summary across outcomes** — `IPTWGEEModel.build_summary_table(results_dict)` applies FDR correction across outcome p-values
5. **Sensitivity analysis** — `IPTWGEEModel.compute_evalues_from_results(results_dict)` for E-values

### `CausalDiagnostics` method groups
The class is organized into five groups. Key signatures:

| Group | Method | Purpose |
|---|---|---|
| **A) Pre-Modeling** | `check_vif(df, controls, treatment=None, exclude_vars=None)` | VIF/GVIF multicollinearity check; returns DataFrame with severity ratings |
| | `check_high_intercorrelations(df, numerical_threshold=0.7, categorical_threshold=0.7)` | Pearson, Cramér's V, and Eta for all variable pairs |
| | `show_low_proportion_groups(df, treatment, ...)` | Flags sparse cells that threaten positivity |
| **B) Overlap** | `run_overlap_diagnostics(data, treatment_var, outcome_vars, ...)` | Full overlap pipeline per outcome: univariate SMDs, propensity AUC, common support. Returns recommendation dict. |
| | `prepare_adjustment_set_for_overlap(data, outcome_var, baseline_vars, ...)` | Builds the correct covariate set (excludes baseline from PS, keeps for outcome) |
| **C) Balance** | `compute_balance_df(data, controls, treatment, weights, already_encoded=True)` | Single-call unweighted + weighted SMDs. Used internally by `IPTWGEEModel` post-weighting. |
| **D) Visualization** | `plot_propensity_overlap(data, treatment_var, propensity_scores, outcome_var)` | Mirrored density plot of propensity scores by group |
| **E) Help** | `help()` | Prints all available methods with descriptions |

`run_overlap_diagnostics()` is the main pre-modeling entry point — it calls `check_covariate_overlap()` internally and produces a text summary saved to `results/`.

### Variable naming conventions
- Treatment is always binary `0/1` in a column named `treatment`
- Clustering variable: `team_id`
- Covariates split into three lists: `categorical_vars`, `binary_vars`, `continuous_vars`
- Baseline (prior-year) variables prefixed with `baseline_` — included in GEE outcome model (doubly robust) but **excluded** from propensity score model
- Outcome variables: `manager_efficacy_index`, `workload_index_mgr`, `turnover_intention_index_mgr`, `retention_Xmonth`

### Column name sanitization
`IPTWGEEModel._clean_column_name()` replaces special characters (e.g., `&` → `and`, spaces → `_`) for statsmodels formula compatibility. All variable references are remapped after one-hot encoding. When adding new variables, avoid characters that break patsy formulas.

### Estimand choice
Both ATE and ATT are supported across both analysis approaches:
- **IPTW/GEE:** The estimand changes the IPTW weight formula, not the GEE model itself. Pass `estimand="ATE"` or `estimand="ATT"` to `analyze_treatment_effect()`.
- **DML:** Pass `estimand="ATE"`, `"ATT"`, or `"both"` to `dml_estimate_treatment_effects()`. ATE is estimated directly; ATT is derived by averaging CATE estimates over treated observations (valid under unconfoundedness). When DML fits a constant effect (no X), ATE = ATT by construction.

### Binary outcome auto-detection
`analyze_treatment_effect()` and `dml_estimate_treatment_effects()` both auto-detect binary outcomes (retention). The IPTW method switches to Binomial family GEE; the DML method sets `discrete_outcome=True` and selects `RandomForestClassifier` for the outcome nuisance model. No manual override needed.

## Data Generation

`s2_generate_data.py` is a **single long script** (not importable). Key design decisions:
- Self-selection into treatment driven by `organization` and `performance_rating` (logistic model, bisection-calibrated)
- Below/Far Below performers are hard-blocked from treatment
- ~25% of managers are "new" (no prior manager-level baselines → `0` in `baseline_manager_efficacy`)
- Heterogeneous treatment effects: R&D gets extra effect on efficacy; new managers get extra effect on turnover intention
- Outputs: `data/s2_manager_data.csv`, `data/s2_data_descriptives.xlsx`

## Development Notes

- **Python deps:** `requirements.txt` pins versions — `statsmodels`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `openpyxl`, `econml`
- **No test suite** — this is a teaching repo, not a production package
- **Google Colab** is the primary execution target (see README clone instructions)
- The notebook has pre-cached outputs but cells have not been executed in the current session
- `CausalDiagnostics` uses `IPython.display` with a `print` fallback for non-notebook contexts
- Excel export uses `openpyxl` with custom formatting (conditional p-value coloring, alternating rows)

### Excel export conventions
Both `s2_generate_data.py` and `IPTWGEEModel.analyze_treatment_effect()` produce formatted `.xlsx` files via `openpyxl`. The data generator defines reusable formatting helpers that any new Excel export should follow:

- **`apply_header_format(ws, row, max_col)`** — dark-blue header row with white bold text
- **`apply_alternating_rows(ws, start_row, end_row, max_col)`** — striped rows; auto-bolds rows containing "Overall"/"Total"
- **`apply_pvalue_conditional(ws, col_letter, start_row, end_row)`** — green (`p < .05`), yellow (`.05–.10`), red (`≥ .10`)
- **`write_df(ws, df, start_row)`** — writes a DataFrame with header formatting + alternating rows; returns the last row written
- **`auto_fit_columns(ws)`** — auto-sizes column widths
- **`write_title(ws, row, title_text, max_col)`** — merged title cell with navy bold font

These helpers are defined inline in `s2_generate_data.py` (not importable). When adding new Excel sheets, copy the pattern: `write_title` → `write_df` → `apply_pvalue_conditional` on p-value columns → `auto_fit_columns`. Formatting constants (`HEADER_FILL`, `ALT_ROW_FILL`, `GREEN_FILL`, etc.) are at the top of the Excel section (~line 780).

### DML-specific conventions
- `dml_estimate_treatment_effects()` accepts the same triple-list covariate convention (`categorical_vars`, `binary_vars`, `continuous_vars`) or explicit `W_cols`/`X_cols` lists.
- The `estimate` parameter controls the statistical method: `"ATE"` (Linear DML), `"CATE"` (Causal Forest), or `"both"`. This is orthogonal to `estimand` (the causal quantity).
- DML does **not** natively handle clustering (`team_id`). For cluster-robust inference, use `analyze_treatment_effect()`.
- `model_y` and `model_t` default to `None` and are auto-instantiated inside the method to avoid mutable default arguments.
- Column sanitization via `_clean_column_name()` is applied identically to the IPTW pipeline.
- Return dict is structured for direct use with `build_summary_table()` and `compute_evalues_from_results()`.
- Call `model.dml_estimate_treatment_effects_help()` for detailed usage examples.

## Common Tasks

| Task | Command |
|---|---|
| Regenerate data | `python s2_generate_data.py` |
| Install deps | `pip install -r requirements.txt` |
| Run workshop | Open `scenario2_workshop.ipynb` and execute cells sequentially |
