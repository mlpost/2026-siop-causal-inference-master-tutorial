# Copilot Instructions — SIOP Causal Inference Master Tutorial

## Project Overview

This is a **workshop/tutorial codebase** for teaching causal inference methods applied to HR/people analytics. It evaluates a leadership development program's causal impact on manager outcomes using observational (non-randomized) data. The primary audience is I/O psychology practitioners at SIOP 2026.

**Core workflow:** Generate synthetic data → Run diagnostics → Estimate treatment effects via IPTW + doubly robust GEE (survey outcomes) and IPTW + Piecewise Cox PH (retention/survival outcomes) → Sensitivity analysis → Optional HTE exploration via Double Machine Learning.

## Architecture

| Component | Purpose |
|---|---|
| `s2_generate_data.py` | Deterministic synthetic data generator (seed=42). Produces manager-level CSV + Excel descriptives. Run standalone to regenerate `data/`. |
| `scenario2_workshop.ipynb` | Main teaching notebook. Walks through the full causal inference pipeline interactively. |
| `supp_functions/causal_diagnostics.py` | `CausalDiagnostics` class — pre-modeling checks (VIF, intercorrelations, overlap), covariate balance (SMD), propensity score visualization. |
| `supp_functions/causal_inference_modelling.py` | `CausalInferenceModel` class (aliased as `IPTWGEEModel` for backward compatibility) — three approaches: (1) IPTW + doubly robust GEE for continuous/binary outcomes, (2) IPTW + Cox PH with time interaction for time-to-event survival outcomes via `lifelines`, (3) Double Machine Learning (DML) via `econml` for ATE/ATT/CATE estimation. Also includes E-value sensitivity, RMST, Markdown report generation, and summary tables. Shared IPTW infrastructure is factored into `_prepare_iptw_data()`. Module has a comprehensive docstring cataloguing all public methods by category. |
| `data/s2_manager_data.csv` | Pre-generated manager-level dataset (9000 rows). |
| `pregenereated_results/s2/` | Reference diagnostic outputs for comparison. |
| `results/` | Output directory for workshop-generated results (Excel, plots). |

## Key Patterns & Conventions

### Two-class design
All reusable logic lives in two stateful classes instantiated in the notebook:
```python
from causal_diagnostics import CausalDiagnostics
from causal_inference_modelling import CausalInferenceModel
cd = CausalDiagnostics()
causal_model = CausalInferenceModel()
```
Import path relies on `sys.path.append('./supp_functions')` — there is no package install.

**Note:** The class was renamed from `IPTWGEEModel` to `CausalInferenceModel` to reflect the addition of survival analysis (Cox PH) alongside GEE.

### Analysis pipeline
The notebook runs two distinct outcome families, then optional HTE exploration:

**Survey outcomes (continuous):** `manager_efficacy_index`, `workload_index_mgr`, `turnover_intention_index_mgr`
1. **Pre-modeling diagnostics** — `cd.check_vif()`, `cd.check_high_intercorrelations()`, `cd.run_overlap_diagnostics()`
2. **IPTW + GEE** — `causal_model.analyze_treatment_effect()` (propensity scoring → weight estimation → balance check → doubly robust GEE)
3. **Summary** — `CausalInferenceModel.build_summary_table(results_dict)` applies FDR correction
4. **Balance verification** — notebook-defined `verify_balance()` for independent post-weighting checks
5. **Sensitivity** — `CausalInferenceModel.compute_evalues_from_results(results_dict, effect_type="cohens_d")`
6. **Report** — `CausalInferenceModel.generate_gee_summary_report()` renders Markdown technical summary

Both ATE and ATT are run for survey outcomes; results are compared via `CausalInferenceModel.generate_comparison_table()`.

**Retention (time-to-event):** single survival outcome keyed as `'retention'` — ATE only (no ATT for retention)
1. **Data prep** — `causal_model.prepare_survival_data()` converts `exit_date` → `days_observed` + `departed`
2. **IPTW + Cox Time Interaction** — `causal_model.analyze_survival_effect(time_interaction='categorical', period_breaks=[0, 90, 180, 270, 365], period_labels=['0-3mo', '3-6mo', '6-9mo', '9-12mo'])` fits separate HRs per quarterly interval
3. **Kaplan-Meier** — `causal_model.plot_survival_curves()` with snapshot overlays and risk table
4. **Summary** — `CausalInferenceModel.build_survival_summary_table()` (optionally includes RMST columns)
5. **Balance verification** — reuses `verify_balance()` (defined in survey section), excluding time/event columns
6. **Sensitivity** — `CausalInferenceModel.compute_evalues_from_results(survival_results, effect_type="risk_ratio")`
7. **Preserve** — results saved as `ate_survival_results`, `ate_survival_summary`, `ate_survival_evalues`
8. **Report** — `CausalInferenceModel.generate_survival_summary_report(survival_plot_fig=survival_fig)` renders Markdown technical summary with optional inline KM figure

Note: `CausalInferenceModel.compute_rmst_difference()` is available for business-friendly "additional days retained" but is not called in the current notebook pipeline.

**HTE exploration (optional):**
- `causal_model.dml_estimate_treatment_effects(estimand="ATE", estimate="both")` — runs both Linear DML (ATE validation) and Causal Forest DML (CATE exploration) on significant survey outcomes (currently only `manager_efficacy_index`)
- Baseline variable for the outcome (e.g., `baseline_manager_efficacy`) is appended to `continuous_vars` for HTE estimation

### `CausalDiagnostics` method groups
The class is organized into five groups. Key signatures:

| Group | Method | Purpose |
|---|---|---|
| **A) Pre-Modeling** | `check_vif(df, controls, treatment=None, exclude_vars=None)` | VIF/GVIF multicollinearity check; returns DataFrame with severity ratings |
| | `check_high_intercorrelations(df, numerical_threshold=0.7, categorical_threshold=0.7, verbose=False, exclude_vars=None)` | Pearson, Cramér's V, and Eta for all variable pairs |
| | `show_low_proportion_groups(df, treatment, ...)` | Flags sparse cells that threaten positivity |
| **B) Overlap** | `run_overlap_diagnostics(data, treatment_var, outcome_vars, ...)` | Full overlap pipeline per outcome: loops through outcomes calling `check_covariate_overlap()`. Returns recommendation dict. |
| | `check_covariate_overlap(data, treatment_var, categorical_vars, binary_vars, continuous_vars, baseline_vars, _show_guide, _quiet)` | Core univariate+multivariate overlap engine: SMDs, propensity AUC, common support. Called internally by `run_overlap_diagnostics()`. |
| | `prepare_adjustment_set_for_overlap(data, outcome_var, baseline_vars, ...)` | Builds the correct covariate set (excludes baseline from PS, keeps for outcome) |
| **C) Balance** | `compute_balance_df(data, controls, treatment, weights, already_encoded=False)` | Single-call unweighted + weighted SMDs. Used internally by `CausalInferenceModel` post-weighting. |
| **D) Visualization** | `plot_propensity_overlap(data, treatment_var, propensity_scores, outcome_var, title=None)` | Mirrored histogram of propensity scores by group with common-support shading |
| | `save_overlap_diagnostics_summary(overlap_results, save_path)` | Write plain-text overlap diagnostic report to file |
| **E) Help** | `help()` | Prints all available methods with descriptions |

`run_overlap_diagnostics()` is the main pre-modeling entry point — it calls `check_covariate_overlap()` for each outcome and produces a text summary saved to `results/`.

Configurable thresholds for SMD and AUC severity are stored in `self.overlap_thresholds` (set in `__init__`), making them tuneable without code changes.

### Variable naming conventions
- Treatment is always binary `0/1` in a column named `treatment`
- Clustering variable: `team_id`
- Covariates split into three lists:
  - `categorical_vars`: `organization`, `job_level`, `performance_rating`
  - `binary_vars`: `gender`, `is_people_manager`, `is_new_manager`
  - `continuous_vars`: `age`, `tenure_months`, `num_direct_reports`, `tot_span_of_control`
- Baseline (prior-year) variables prefixed with `baseline_` — included in GEE outcome model (doubly robust) but **excluded** from propensity score model
- Survey outcome variables: `manager_efficacy_index`, `workload_index_mgr`, `turnover_intention_index_mgr`
- Retention is analyzed via survival analysis using `exit_date` → `days_observed` + `departed` (not the binary `retention_Xmonth` flags)
- `outcome_descriptions` dict maps variable names to display-friendly labels
- `outcome_valence` dict marks higher-is-worse outcomes (e.g., `workload_index_mgr: 'negative'`)

### CausalInferenceModel internal architecture
The class uses a **shared IPTW pipeline** via `_prepare_iptw_data()` (~300 lines) that consolidates data prep, one-hot encoding, column sanitization, propensity score estimation, weight diagnostics, overlap/weight plotting, and balance checking. Both `analyze_treatment_effect()` and `analyze_survival_effect()` delegate Steps 0–2 to this shared method.

Key building-block methods (also usable standalone):
- `estimate_propensity_weights()` — fit PS model, compute stabilized IPTW weights
- `compute_weight_diagnostics()` — ESS computation + weight summary stats
- `fit_doubly_robust_model()` — IPTW-weighted GEE with covariate adjustment
- `plot_propensity_overlap()` — PS overlap density plot (delegates to `CausalDiagnostics`)
- `plot_weight_distribution()` — histogram of IPTW weights by treatment group
- `plot_survival_curves()` — IPTW-weighted KM curves with risk table + HR annotation
- `calculate_standardized_mean_difference()` — **deprecated**, prefer `CausalDiagnostics.compute_balance_df()`

Cox PH internals:
- `_fit_cox_model()` — fits IPTW-weighted Cox PH with time interaction; supports `time_interaction="categorical"` (person-period expansion with separate HRs per interval, e.g., quarterly) or `time_interaction="continuous"` (linear trend). Requires `period_breaks` and optional `period_labels` for categorical mode. There is no separate `_fit_piecewise_cox()` method — piecewise estimation is handled within `_fit_cox_model`.

### Column name sanitization
`CausalInferenceModel._clean_column_name()` replaces special characters (e.g., `&` → `and`, spaces → `_`) for statsmodels formula compatibility. All variable references are remapped after one-hot encoding. When adding new variables, avoid characters that break patsy formulas.

### Estimand choice
Both ATE and ATT are supported across both analysis approaches:
- **IPTW/GEE:** The estimand changes the IPTW weight formula, not the GEE model itself. Pass `estimand="ATE"` or `estimand="ATT"` to `analyze_treatment_effect()`.
- **DML:** Pass `estimand="ATE"`, `"ATT"`, or `"both"` to `dml_estimate_treatment_effects()`. ATE is estimated directly; ATT is derived by averaging CATE estimates over treated observations (valid under unconfoundedness). When DML fits a constant effect (no X), ATE = ATT by construction.

### Binary outcome auto-detection
`analyze_treatment_effect()` and `dml_estimate_treatment_effects()` both auto-detect binary outcomes. The IPTW method switches to Binomial family GEE; the DML method sets `discrete_outcome=True` and selects `RandomForestClassifier` for the outcome nuisance model. No manual override needed.

### Survival analysis conventions
- `prepare_survival_data()` converts a departure date column to `days_observed` (int) + `departed` (0/1) + `departure_quarter` (str)
- `analyze_survival_effect()` uses the same IPTW weighting infrastructure (`_prepare_iptw_data()`) as `analyze_treatment_effect()` but fits a Cox PH model instead of GEE
- **Time interaction Cox** — `analyze_survival_effect(time_interaction='categorical', period_breaks=[0, 90, 180, 270, 365], period_labels=['0-3mo', '3-6mo', '6-9mo', '9-12mo'])` fits separate hazard ratios per time interval via person-period expansion, preferred in the notebook over a single global Cox model
- `time_interaction='continuous'` is also available (models treatment × time as a linear trend) but not used in the current notebook
- `compute_rmst_difference()` provides a business-friendly metric: additional days retained within the study window (available but not called in the current notebook pipeline)
- `build_survival_summary_table()` is the survival analogue of `build_summary_table()` and optionally includes RMST columns
- `generate_survival_summary_report()` accepts an optional `survival_plot_fig` parameter for embedding the KM figure as an inline base64 image in the Markdown report

## Data Generation

`s2_generate_data.py` is a **single long script** (not importable). Key design decisions:
- Self-selection into treatment driven by `organization` and `performance_rating` (logistic model, bisection-calibrated)
- Below/Far Below performers are hard-blocked from treatment
- ~25% of managers are "new" (no prior manager-level baselines → `0` in `baseline_manager_efficacy`)
- Heterogeneous treatment effects: R&D gets extra effect on efficacy; new managers get extra effect on turnover intention
- Outputs: `data/s2_manager_data.csv`, `data/s2_data_descriptives.xlsx`

## Development Notes

- **Python deps:** `requirements.txt` pins versions — `statsmodels`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `openpyxl`, `lifelines`, `econml`
- **No test suite** — this is a teaching repo, not a production package
- **Google Colab** is the primary execution target (see README clone instructions)
- The notebook has pre-cached outputs but cells have not been executed in the current session
- `CausalDiagnostics` uses `IPython.display` with a `print` fallback for non-notebook contexts
- Excel export uses `openpyxl` with custom formatting (conditional p-value coloring, alternating rows)
- `correction_method` parameter in `analyze_treatment_effect()` is **deprecated** — FDR correction is now applied in `build_summary_table()`. Passing a non-default value emits `DeprecationWarning`.
- `compute_evalues_from_results()` defaults to `effect_type="auto"` — it auto-detects `cohens_d` vs `risk_ratio` from the result dict

### Excel export conventions
Both `s2_generate_data.py` and `CausalInferenceModel.analyze_treatment_effect()` produce formatted `.xlsx` files via `openpyxl`. The data generator defines reusable formatting helpers that any new Excel export should follow:

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
- In the notebook, DML is run with `estimate="both"` on `manager_efficacy_index` only — Linear DML validates the ATE from GEE, while Causal Forest explores heterogeneous treatment effects.
- DML does **not** natively handle clustering (`team_id`). For cluster-robust inference, use `analyze_treatment_effect()`.
- `model_y` and `model_t` default to `None` and are auto-instantiated inside the method to avoid mutable default arguments.
- Column sanitization via `_clean_column_name()` is applied identically to the IPTW pipeline.
- Return dict is structured for direct use with `build_summary_table()` and `compute_evalues_from_results()`.
- Call `causal_model.dml_estimate_treatment_effects_help()` for detailed usage examples.

### Markdown report generation
- `generate_gee_summary_report()` — renders a Markdown technical summary for survey (GEE) outcome families
- `generate_survival_summary_report()` — renders a Markdown technical summary for survival outcomes (hazard ratios, RMST, survival probabilities); accepts optional `survival_plot_fig` for inline base64 image embedding
- `generate_comparison_table()` — generates an ATE vs. ATT side-by-side comparison with effect sizes, E-values, and auto-generated observations
- All three are static methods that accept the output of `build_summary_table()` / `build_survival_summary_table()` and `compute_evalues_from_results()`
- Designed for use with `IPython.display.Markdown` in Jupyter notebooks
- Helper statics: `_format_pvalue()` (p-value → display string), `_significance_stars()` (p-value → `***`/`**`/`*`), `compute_evalue()` (single-effect E-value calculation per VanderWeele & Ding 2017)

## Notebook structure summary

The descriptive exploration cell (cell 8) includes:
- Demographic comparisons (continuous KDEs, categorical bar charts)
- Team size / new manager status plots
- **Retention over time line plot** — daily retention % by treatment group (Control vs. Treated) computed from `exit_date`

The notebook ends with three substantial markdown cells:
- **DML / HTE Learning Guide** — detailed explanation of Linear DML, Causal Forest, and CATE methodology
- **Global Technical Summary** — table consolidating all results across outcome families
- **Key Takeaways for Stakeholders** — plain-language recommendations for HR decision-makers

## Common Tasks

| Task | Command |
|---|---|
| Regenerate data | `python s2_generate_data.py` |
| Install deps | `pip install -r requirements.txt` |
| Run workshop | Open `scenario2_workshop.ipynb` and execute cells sequentially |
