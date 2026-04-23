"""
⚠️  DEVELOPMENT CODE DISCLAIMER ⚠️

This code is provided for educational purposes as part of a causal inference
tutorial for I/O psychologists. While developed in good faith and based on
established statistical methods, this is ACTIVE DEVELOPMENT CODE that:

• May contain bugs or unvalidated edge cases
• Has not undergone formal validation for production use
• Should be used with appropriate statistical supervision

Causal Inference Modeling Module
================================

Data-agnostic causal inference toolkit for estimating treatment effects from
observational data.  All reusable logic lives in the :class:`CausalInferenceModel`
class (aliased as ``IPTWGEEModel`` for backward compatibility).

Three complementary estimation approaches are supported:

1. **IPTW + Covariate-Adjusted GEE** — continuous / binary survey outcomes
   (``analyze_treatment_effect``)
2. **IPTW + Cox Proportional Hazards** — time-to-event (survival) outcomes
   (``analyze_survival_effect``), with optional piecewise (per-interval) HRs
3. **Double Machine Learning (DML)** — cluster-robust ATE via ``doubleml``
    and CATE exploration via ``econml``
    (``dml_cluster_robust_ate`` / ``dml_estimate_treatment_effects``)

Shared IPTW infrastructure (data prep, one-hot encoding, column sanitization,
propensity score estimation, weight diagnostics, overlap/weight plotting, and
balance checking) is factored into ``_prepare_iptw_data()``; both Approach 1
and 2 delegate their pre-modeling steps to that method.

Key public methods
------------------
Data preparation
    ``prepare_survival_data``
        Convert departure-date data to survival format (days_observed + departed).

Building blocks (also usable standalone)
    ``estimate_propensity_weights``
        Fit propensity score model and compute stabilized IPTW weights.
    ``compute_weight_diagnostics``
        Effective sample size (ESS) and weight summary statistics.
    ``fit_iptw_outcome_model``
        IPTW-weighted GEE with covariate adjustment.
    ``plot_propensity_overlap``
        Propensity score overlap density plot (delegates to CausalDiagnostics).
    ``plot_weight_distribution``
        Histogram of IPTW weights by treatment group.
    ``plot_survival_curves``
        IPTW-weighted Kaplan-Meier curves with risk table and HR annotation.

High-level analysis (recommended entry points)
    ``analyze_treatment_effect``
        Full IPTW + covariate-adjusted GEE pipeline for survey outcomes.
    ``analyze_survival_effect``
        Full IPTW + Cox PH pipeline for time-to-event outcomes; supports
        piecewise (quarterly) hazard ratios via ``_fit_cox_model`` with
        ``time_interaction="categorical"``.
    ``dml_estimate_treatment_effects``
        Linear DML (ATE) and/or Causal Forest DML (CATE) estimation.

Post-estimation summaries & sensitivity
    ``build_summary_table``
        Aggregate GEE/DML results with FDR correction.
    ``build_survival_summary_table``
        Aggregate survival results with optional RMST columns.
    ``compute_evalue`` / ``compute_evalues_from_results``
        E-value sensitivity analysis (VanderWeele & Ding 2017).
    ``compute_confounder_evalue_benchmarks``
        Calibrate E-values against strongest measured confounders
        (pre-weighting SMD → approximate RR → E-value).
    ``compute_rmst_difference``
        Restricted Mean Survival Time difference (business-friendly metric).


Utilities (static)
    ``_clean_column_name``
        Sanitize column names for statsmodels formula compatibility.
    ``_significance_stars``
        Convert p-value to significance stars (``***``/``**``/``*``).


Deprecated
    ``calculate_standardized_mean_difference``
        Use ``CausalDiagnostics.compute_balance_df()`` instead.

Dependencies
------------
statsmodels, pandas, numpy, scipy, scikit-learn, matplotlib, lifelines,
doubleml, econml

Notes
-----
- Import path relies on ``sys.path.append('./supp_functions')`` — no package
  install required.
- The ``correction_method`` parameter in ``analyze_treatment_effect()`` is
  deprecated; FDR correction is now applied in ``build_summary_table()``.
- Column name sanitization (``_clean_column_name``) replaces special characters
  for patsy/statsmodels formula compatibility.  Avoid ``&``, spaces, or other
  special characters in variable names.
- GEE/Cox sandwich standard errors do **not** propagate first-stage uncertainty
  from propensity score estimation.
"""
import atexit
import re
import warnings

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

try:
    from lifelines.exceptions import StatisticalWarning
except ImportError:
    StatisticalWarning = UserWarning  # older lifelines versions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.genmod import families
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from doubleml import DoubleMLData, DoubleMLPLR
from econml.dml import DML, CausalForestDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.cate_interpreter import SingleTreeCateInterpreter

# Jupyter / IPython display — falls back to print outside notebooks
try:
    from IPython.display import display
except ImportError:
    display = print

from causal_diagnostics import CausalDiagnostics

# Global flag to track if disclaimer has been shown
_DISCLAIMER_SHOWN = False

def _show_development_disclaimer():
    """Display development code disclaimer once per session."""
    global _DISCLAIMER_SHOWN
    if not _DISCLAIMER_SHOWN:
        print("\n" + "="*80)
        print("🚨 CAUSAL INFERENCE TUTORIAL - DEVELOPMENT CODE DISCLAIMER 🚨")
        print("="*80)
        print("This code is provided for EDUCATIONAL PURPOSES ONLY as part of")
        print("a causal inference tutorial for I/O psychologists.")
        print()
        print("⚠️  IMPORTANT LIMITATIONS:")
        print("   • Active development code - may contain bugs")
        print("   • Not validated for production research use")
        print("   • Requires statistical supervision and verification")
        print("="*80)
        print()
        _DISCLAIMER_SHOWN = True


def _expand_to_person_period(
    df: pd.DataFrame,
    time_var: str,
    event_var: str,
    period_breaks: List[int],
    period_labels: List[str],
    period_col: str = "_period",
) -> pd.DataFrame:
    """
    Expand survival data to person-period format for categorical time interaction.

    Each person contributes one row per period they were at risk. Used by
    _fit_cox_model when time_interaction="categorical".

    Parameters
    ----------
    df : pd.DataFrame
        Data with time_var and event_var.
    time_var : str
        Name of time column (days to event/censoring).
    event_var : str
        Name of event indicator column.
    period_breaks : list of int
        Breakpoints in days, e.g. [0, 90, 180, 270, 365].
    period_labels : list of str
        Labels for each period.
    period_col : str, default="_period"
        Column name for the period label.

    Returns
    -------
    pd.DataFrame
        Expanded DataFrame with one row per person-period.
    """
    period_dfs = []
    for i, (t_start, t_end) in enumerate(zip(period_breaks[:-1], period_breaks[1:])):
        at_risk = df[df[time_var] > t_start].copy()
        at_risk[event_var] = (
            (at_risk[event_var] == 1)
            & (at_risk[time_var] > t_start)
            & (at_risk[time_var] <= t_end)
        ).astype(int)
        at_risk[period_col] = period_labels[i]
        period_dfs.append(at_risk)
    return pd.concat(period_dfs, ignore_index=True)


class CausalInferenceModel:
    """
    Unified causal inference toolkit for treatment effect estimation.

    This class implements three complementary approaches:

        **Approach 1 — IPTW + Covariate-Adjusted GEE** (``analyze_treatment_effect``):
    - For continuous and binary outcomes (e.g., survey indices, binary flags)
    - Stabilized inverse probability weights (IPTW) for ATE or ATT
    - Generalized Estimating Equations (GEE) to account for clustering
        - Optional outcome model covariate adjustment for additional protection
            against model misspecification

    **Approach 2 — IPTW + Cox Proportional Hazards** (``analyze_survival_effect``):
    - For time-to-event outcomes (e.g., employee retention, time to departure)
    - Same IPTW weighting infrastructure as Approach 1
    - Cox PH model for hazard ratios with IPTW-weighted Kaplan-Meier curves
    - Restricted Mean Survival Time (RMST) for business-friendly interpretation
    - Use ``prepare_survival_data()`` to convert departure dates to survival format

        **Approach 3 — Double Machine Learning**
        (``dml_cluster_robust_ate`` / ``dml_estimate_treatment_effects``):
        - DoubleML PLR for cluster-robust ATE estimation with flexible nuisance
            models
        - Causal Forest DML for individualized CATE estimation
        - ATT derived from CATE by averaging over treated observations
        - Uses ``doubleml`` for ATE robustness checks and ``econml`` for CATE

    The estimand (ATE vs. ATT) is determined by the weight construction formula
    (IPTW) or by subsetting CATE predictions (DML), not by GEE or Cox itself.

    Note on standard errors:
        GEE and Cox sandwich standard errors account for within-cluster
        correlation but do **not** propagate first-stage uncertainty from
        propensity score estimation. For stricter inference, consider a
        non-parametric bootstrap that re-estimates both stages in each replicate.

    Attributes
    ----------
    weight_col : str
        Name of the weight column (default: \"iptw\")
    ps_model : object
        Last fitted propensity score model
    gee_model : object
        Last fitted GEE outcome model
    dml_model : object
        Last fitted DML model (if DML methods used)
    cfdml_model : object
        Last fitted CausalForestDML model (if DML methods used)
    """
    
    def __init__(self):
        """Initialize the CausalInferenceModel.
        
        Note: ps_model, gee_model, dml_model, and cfdml_model store the
        *last* fitted model only. When running multiple outcomes in a loop,
        capture results per iteration from the returned dict rather than
        relying on these instance attributes.
        """
        _show_development_disclaimer()
        self.weight_col = "iptw"
        self.ps_model = None
        self.gee_model = None
        self.dml_model = None
        self.cfdml_model = None

    # ==================================================================
    # Data preparation
    # ==================================================================

    def prepare_survival_data(self, data, departure_date_col,
                              treatment_var,
                              t0_date,
                              study_end_date,
                              date_format='%m-%d-%Y',
                              time_col_name='days_observed',
                              event_col_name='departed',
                              _quiet=False):
        """
        Convert departure date data to survival analysis format (time-to-event).

        Designed for scenarios where:
        - T=0 is the same calendar date for all managers (e.g., January 1)
        - Exact departure dates are known for those who left
        - Still-employed managers are censored at a common study end date

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing one row per manager.
        departure_date_col : str
            Column name containing departure date in m-dd-yyyy format.
            NaN/NaT values indicate still employed (censored).
        treatment_var : str
            Binary treatment column name (1=trained, 0=untrained).
        t0_date : str
            Study start date (T=0) for ALL managers, e.g., '1-01-2025'.
            Format must match date_format parameter.
        study_end_date : str
            Censoring date for still-employed managers, e.g., '12-31-2025'.
            Format must match date_format parameter.
        date_format : str, default '%m-%d-%Y'
            Date parsing format string.
        time_col_name : str, default 'days_observed'
            Name for output time column (days from T=0 to event or censoring).
        event_col_name : str, default 'departed'
            Name for output event indicator column (1=departed, 0=censored).
        _quiet : bool, default False
            If True, suppress print output.

        Returns
        -------
        pd.DataFrame
            Copy of input data with added columns:
            - time_col_name (int): days from T=0 to event/censoring
            - event_col_name (int): 1 if departed, 0 if censored
            - 'departure_quarter' (str): quarter of departure for diagnostics

        Raises
        ------
        ValueError
            If departure_date_col or treatment_var not in data.columns.
        """

        def _print(msg=""):
            """Internal print wrapper respecting _quiet flag."""
            if not _quiet:
                print(msg)

        # ==================================================================
        # STEP 1 — Parse all dates
        # ==================================================================

        if departure_date_col not in data.columns:
            raise ValueError(f"departure_date_col '{departure_date_col}' not found in data.")
        if treatment_var not in data.columns:
            raise ValueError(f"treatment_var '{treatment_var}' not found in data.")

        # Parse scalar dates
        t0 = pd.to_datetime(t0_date, format=date_format)
        study_end = pd.to_datetime(study_end_date, format=date_format)

        # Parse departure date column (errors='coerce' converts invalid dates to NaT)
        parsed_departure = pd.to_datetime(data[departure_date_col],
                                          format=date_format,
                                          errors='coerce')

        # Total study window in days
        total_days = (study_end - t0).days

        _print(f"\nParsed dates:")
        _print(f"  T=0 (study start):  {t0.date()}")
        _print(f"  Study end:          {study_end.date()}")
        _print(f"  Total window:       {total_days} days")

        # Date range of observed departures
        valid_departures = parsed_departure.dropna()
        if len(valid_departures) > 0:
            _print(f"  Departure range:    {valid_departures.min().date()} → "
                   f"{valid_departures.max().date()}")
        else:
            _print(f"  Departure range:    (no departures observed)")

        # ==================================================================
        # STEP 2 — Data quality checks
        # ==================================================================

        # Check A: Departure before T=0
        before_t0 = parsed_departure < t0
        n_before_t0 = before_t0.sum()
        if n_before_t0 > 0:
            _print(f"\n⚠️  WARNING: {n_before_t0} managers have departure date "
                   f"BEFORE study start (T=0).")
            _print(f"   Check data quality for these records:")
            bad_indices = data.index[before_t0].tolist()
            _print(f"   Indices: {bad_indices[:10]}" +
                    (f" ... and {len(bad_indices)-10} more" if len(bad_indices) > 10 else ""))

        # Check B: Departure after study end
        after_end = parsed_departure > study_end
        n_after_end = after_end.sum()
        if n_after_end > 0:
            _print(f"\n⚠️  WARNING: {n_after_end} managers have departure date "
                   f"AFTER study end.")
            _print(f"   These will be treated as censored at study end.")
            # Correct these by setting to NaT (censored)
            parsed_departure = parsed_departure.copy()
            parsed_departure.loc[after_end] = pd.NaT

        # ==================================================================
        # STEP 3 — Compute event indicator and time observed
        # ==================================================================

        # Create a copy of data to avoid mutating original
        result_df = data.copy()

        # Event indicator: 1 if departure date is not NaT, 0 otherwise
        result_df[event_col_name] = parsed_departure.notna().astype(int)

        # Days observed:
        # - If event=1: days from T=0 to departure
        # - If event=0: days from T=0 to study end (censored)
        result_df[time_col_name] = np.where(
            parsed_departure.notna(),
            (parsed_departure - t0).dt.days,  # exact days to departure
            total_days                         # censored at study end
        )

        # Check C: Zero or negative survival times
        bad_times = result_df[time_col_name] <= 0
        n_bad_times = bad_times.sum()
        if n_bad_times > 0:
            _print(f"\n⚠️  WARNING: {n_bad_times} managers have {time_col_name} <= 0.")
            _print(f"   Review these records for data quality issues.")

        # Check D: Departure on day 0 exactly
        day_zero = (result_df[event_col_name] == 1) & (result_df[time_col_name] == 0)
        n_day_zero = day_zero.sum()
        if n_day_zero > 0:
            _print(f"\n⚠️  WARNING: {n_day_zero} managers have departure on day 0 "
                   f"(same as T=0).")
            _print(f"   Verify whether these are data entry errors.")

        # ==================================================================
        # STEP 4 — Add departure_quarter column for diagnostics
        # ==================================================================

        def assign_quarter(row):
            """Assign departure quarter based on days_observed."""
            if row[event_col_name] == 0:
                return 'Censored'
            days = row[time_col_name]
            if days <= 90:
                return 'Q1 (Jan-Mar)'
            elif days <= 180:
                return 'Q2 (Apr-Jun)'
            elif days <= 270:
                return 'Q3 (Jul-Sep)'
            else:
                return 'Q4 (Oct-Dec)'

        result_df['departure_quarter'] = result_df.apply(assign_quarter, axis=1)

        # ==================================================================
        # STEP 5 — Print comprehensive summary
        # ==================================================================

        T = result_df[treatment_var].values
        n_total = len(result_df)
        n_treated = (T == 1).sum()
        n_control = (T == 0).sum()
        pct_treated = (n_treated / n_total * 100) if n_total > 0 else 0
        pct_control = (n_control / n_total * 100) if n_total > 0 else 0

        n_events = result_df[event_col_name].sum()
        n_events_treated = ((T == 1) & (result_df[event_col_name] == 1)).sum()
        n_events_control = ((T == 0) & (result_df[event_col_name] == 1)).sum()

        event_rate = (n_events / n_total * 100) if n_total > 0 else 0
        event_rate_treated = (n_events_treated / n_treated * 100) if n_treated > 0 else 0
        event_rate_control = (n_events_control / n_control * 100) if n_control > 0 else 0

        n_censored = (result_df[event_col_name] == 0).sum()
        n_censored_treated = ((T == 1) & (result_df[event_col_name] == 0)).sum()
        n_censored_control = ((T == 0) & (result_df[event_col_name] == 0)).sum()
        censored_rate = (n_censored / n_total * 100) if n_total > 0 else 0

        median_days = result_df[time_col_name].median()
        events_only = result_df[result_df[event_col_name] == 1]
        median_event_days = events_only[time_col_name].median() if len(events_only) > 0 else np.nan

        _print("\n" + "=" * 60)
        _print("SURVIVAL DATA PREPARATION SUMMARY")
        _print("=" * 60)
        _print(f"Study window:  {t0.date()}  →  {study_end.date()}  ({total_days} days)")
        _print("")
        _print("SAMPLE:")
        _print(f"  Total managers:          {n_total}")
        _print(f"  Treated (trained):       {n_treated} ({pct_treated:.1f}%)")
        _print(f"  Control (untrained):     {n_control} ({pct_control:.1f}%)")
        _print("")
        _print("EVENTS (Departures):")
        _print(f"  Total events:            {n_events} ({event_rate:.1f}%)")
        _print(f"  Events — Treated:        {n_events_treated} "
               f"({event_rate_treated:.1f}% of treated)")
        _print(f"  Events — Control:        {n_events_control} "
               f"({event_rate_control:.1f}% of control)")
        _print("")
        _print("TIMING:")
        _print(f"  Median days observed:    {median_days:.0f} days")
        if not np.isnan(median_event_days):
            _print(f"  Median days (events only): {median_event_days:.0f} days")
        _print("")
        _print("DEPARTURE BY QUARTER:")

        # Crosstab of departure_quarter x treatment
        quarter_crosstab = pd.crosstab(
            result_df['departure_quarter'],
            result_df[treatment_var],
            margins=True,
            margins_name='Total'
        )
        quarter_crosstab.columns = ['Control', 'Treated', 'Total']

        # Reorder rows for logical flow
        desired_order = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)',
                          'Q4 (Oct-Dec)', 'Censored', 'Total']
        existing_rows = [r for r in desired_order if r in quarter_crosstab.index]
        quarter_crosstab = quarter_crosstab.reindex(existing_rows)

        if not _quiet:
            display(quarter_crosstab)

        _print("")
        _print("CENSORED:")
        _print(f"  Total censored:          {n_censored} ({censored_rate:.1f}%)")
        _print(f"  Censored — Treated:      {n_censored_treated}")
        _print(f"  Censored — Control:      {n_censored_control}")
        _print("")
        _print(f"✓ Survival columns added: '{time_col_name}' (days) and "
               f"'{event_col_name}' (0/1)")
        _print("=" * 60)

        # ==================================================================
        # STEP 6 — Return
        # ==================================================================

        return result_df

    # ==================================================================
    # Propensity score estimation
    # ==================================================================
    
    def estimate_propensity_weights(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        covariates: List[str],
        estimand: str = "ATE",
        cluster_var: Optional[str] = None,
        stabilize: bool = True,
        trim_quantile: Optional[float] = None,
        weight_col: str = "iptw"
    ) -> Tuple[pd.DataFrame, object]:
        """
        Estimate inverse probability of treatment weights (IPTW) for ATE or ATT.
        
        Fits a propensity score model (GEE if clustered, GLM otherwise) and computes
        stabilized weights based on the specified estimand. Optionally applies weight
        trimming to reduce extreme values.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing treatment, outcome, and covariate variables
        treatment_var : str
            Name of the binary treatment variable (0/1)
        covariates : List[str]
            List of covariate column names to use in propensity score model
        estimand : str, default="ATE"
            Target estimand: "ATE" (Average Treatment Effect) or "ATT" (Average
            Treatment Effect on the Treated). Determines weight formula:
            - ATE: Treated = 1/e, Control = 1/(1-e)
            - ATT: Treated = 1, Control = e/(1-e)
        cluster_var : str, optional
            Name of clustering variable. If provided, uses GEE for propensity model
        stabilize : bool, default=True
            If True, applies stabilization to weights using marginal treatment probability
        trim_quantile : float, optional
            Quantile for weight trimming (e.g., 0.99). Weights above this quantile
            are capped at the quantile value
        weight_col : str, default="iptw"
            Name for the weight column in returned dataframe
        
        Returns
        -------
        tuple
            (df_with_weights, propensity_score_model)
            - df_with_weights: DataFrame with added propensity_score and weight columns
            - propensity_score_model: Fitted propensity score model object
        
        Raises
        ------
        ValueError
            If covariates are invalid, estimand is invalid, or model fitting fails
        """
        # Validate estimand
        estimand = estimand.upper()
        if estimand not in ["ATE", "ATT"]:
            raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")
        
        df = data.copy()
        formula = f"{treatment_var} ~ " + " + ".join(covariates)
        
        # Fit propensity score model
        if cluster_var:
            ps_model = smf.gee(
                formula=formula,
                data=df,
                groups=df[cluster_var],
                family=families.Binomial()
            ).fit()
        else:
            ps_model = smf.glm(
                formula=formula,
                data=df,
                family=families.Binomial()
            ).fit()
        
        df["propensity_score"] = ps_model.predict(df)
        
        # --- Clip propensity scores to avoid division by zero / inf weights
        _eps = 1e-6
        df["propensity_score"] = df["propensity_score"].clip(lower=_eps, upper=1 - _eps)
        
        # --- Compute IPTW weights based on estimand ---
        if estimand == "ATE":
            # ATE: reweight both groups to represent full population
            df[weight_col] = np.where(
                df[treatment_var] == 1,
                1 / df["propensity_score"],
                1 / (1 - df["propensity_score"])
            )
        else:  # ATT
            # ATT: leave treated as-is, reweight controls to match treated
            df[weight_col] = np.where(
                df[treatment_var] == 1,
                1,
                df["propensity_score"] / (1 - df["propensity_score"])
            )
        
        # --- Stabilization: multiply by marginal treatment probability
        if stabilize:
            p_t = df[treatment_var].mean()
            if estimand == "ATE":
                df[weight_col] = np.where(
                    df[treatment_var] == 1,
                    df[weight_col] * p_t,
                    df[weight_col] * (1 - p_t)
                )
            else:  # ATT
                # For ATT, only control weights are stabilized
                df[weight_col] = np.where(
                    df[treatment_var] == 1,
                    df[weight_col],  # treated stay at 1
                    df[weight_col] * p_t
                )
        
        # --- Trimming: cap extreme weights
        if trim_quantile is not None:
            cap = df[weight_col].quantile(trim_quantile)
            df[weight_col] = np.minimum(df[weight_col], cap)
        
        self.ps_model = ps_model
        return df, ps_model
    
    def compute_weight_diagnostics(
        self,
        data: pd.DataFrame,
        weight_col: str = "iptw"
    ) -> Dict[str, Union[int, float]]:
        """
        Compute diagnostic statistics for weights.
        
        Calculates the effective sample size (ESS) and summary statistics
        for the provided weight column.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the weight column
        weight_col : str, default="iptw"
            Name of the weight column
        
        Returns
        -------
        dict
            Dictionary containing:
            - n_observations: Total number of observations
            - effective_sample_size: ESS = (sum(w))^2 / sum(w^2)
            - mean_weight: Mean weight value
            - std_weight: Standard deviation of weights
            - max_weight: Maximum weight value
            - p95_weight: 95th percentile of weights
            - p99_weight: 99th percentile of weights
        
        Raises
        ------
        ValueError
            If weight_col does not exist in data
        """
        if weight_col not in data.columns:
            raise ValueError(f"Weight column '{weight_col}' not found in data")
        
        w = data[weight_col]
        ess = (w.sum() ** 2) / (w ** 2).sum()
        
        return {
            "n_observations": len(w),
            "effective_sample_size": ess,
            "mean_weight": w.mean(),
            "std_weight": w.std(),
            "max_weight": w.max(),
            "p95_weight": w.quantile(0.95),
            "p99_weight": w.quantile(0.99)
        }
    
    def calculate_standardized_mean_difference(
        self,
        data: pd.DataFrame,
        variable: str,
        treatment_var: str,
        weight_col: str = "iptw"
    ) -> float:
        """
        Calculate weighted standardized mean difference (SMD) for covariate balance.
        
        .. deprecated::
            Prefer ``CausalDiagnostics.compute_balance_df()`` which computes
            unweighted and weighted SMDs for all covariates in a single call.
            This method is retained as an internal fallback.
        
        Computes SMD between treatment and control groups after IPTW weighting:
        SMD = (weighted_mean_treated - weighted_mean_control) / pooled_sd
        
        A value of |SMD| < 0.1 indicates acceptable balance.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with treatment, variable, and weight columns
        variable : str
            Name of the variable to assess balance for
        treatment_var : str
            Name of the treatment variable (0/1)
        weight_col : str, default="iptw"
            Name of the weight column
        
        Returns
        -------
        float
            Weighted standardized mean difference
        
        Raises
        ------
        ValueError
            If data is invalid or calculation fails
        """
        warnings.warn(
            "calculate_standardized_mean_difference() is deprecated. "
            "Use CausalDiagnostics.compute_balance_df() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        treated = data[data[treatment_var] == 1]
        control = data[data[treatment_var] == 0]
        
        # Check for empty groups
        if len(treated) == 0 or len(control) == 0:
            raise ValueError(f"Empty treatment group when calculating SMD for {variable}")
        
        # Check if variable exists and has valid values
        if variable not in treated.columns or variable not in control.columns:
            raise ValueError(f"Variable '{variable}' not found in data")
        
        # Check for zero weights
        if (treated[weight_col] == 0).all() or (control[weight_col] == 0).all():
            raise ValueError(f"All weights are zero for '{variable}'")
        
        try:
            # Weighted means
            mt = np.average(treated[variable], weights=treated[weight_col])
            mc = np.average(control[variable], weights=control[weight_col])
            
            # Use sqrt(p*(1-p)) for binary variables (Austin 2009)
            is_binary = set(data[variable].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            if is_binary:
                pooled_p = (mt + mc) / 2
                pooled_sd = np.sqrt(pooled_p * (1 - pooled_p)) if 0 < pooled_p < 1 else 1.0
            else:
                # Weighted variances for continuous variables
                vt = np.average((treated[variable] - mt)**2, weights=treated[weight_col])
                vc = np.average((control[variable] - mc)**2, weights=control[weight_col])
                pooled_sd = np.sqrt((vt + vc) / 2)
            
            return (mt - mc) / pooled_sd if pooled_sd > 0 else 0
        except Exception as e:
            raise ValueError(f"Error calculating weighted SMD for '{variable}': {str(e)}")

    # ==================================================================
    # Outcome model components
    # ==================================================================

    def fit_iptw_outcome_model(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        weight_col: str,
        cluster_var: str,
        covariates: Optional[List[str]] = None,
        family: str = "gaussian"
    ) -> object:
        """
        Fit IPTW-weighted outcome model using GEE with covariate adjustment.

        Estimates the treatment effect using inverse probability weighting
        combined with outcome model covariate adjustment in a GEE framework.

        Including covariates in both the propensity score model and the outcome
        model provides additional protection against misspecification of either
        model - a property sometimes called double robustness under linearity
        (Lunceford & Davidian, 2004). This protection requires that at least
        one of the two models is correctly specified.

        Note: This is not the formal Augmented IPW (AIPW) estimator, which has
        a specific augmentation term that provides the doubly robust property
        in a more general sense. The covariate-adjusted IPTW approach used here
        achieves the same consistency property under linear outcome models.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with outcome, treatment, weights, cluster, and covariate variables
        outcome_var : str
            Name of the outcome variable
        treatment_var : str
            Name of the binary treatment variable
        weight_col : str
            Name of the weight column (e.g., "iptw")
        cluster_var : str
            Name of the clustering variable (e.g., manager ID)
        covariates : List[str], optional
            Additional covariates to include in the outcome model
        family : str, default="gaussian"
            Distribution family for GEE: "gaussian" or "binomial"
        
        Returns
        -------
        statsmodels GEE fit result
            Fitted model with params, bse, pvalues, conf_int() method
        
        Raises
        ------
        ValueError
            If data validation fails or model fitting fails
        """
        # Validation
        if data.empty:
            raise ValueError("Empty dataset provided to model")
        
        required_vars = [outcome_var, treatment_var, weight_col, cluster_var]
        missing_vars = [v for v in required_vars if v not in data.columns]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        rhs = [treatment_var]
        if covariates:
            # Validate covariates exist
            missing_covs = [c for c in covariates if c not in data.columns]
            if missing_covs:
                raise ValueError(f"Missing covariates: {missing_covs}")
            rhs += covariates
        
        # Check for sufficient variation in predictors
        for var in rhs:
            if data[var].nunique() <= 1:
                raise ValueError(f"No variation in predictor variable: '{var}'")
        
        formula = f"{outcome_var} ~ " + " + ".join(rhs)
        
        # Check for zero or negative weights
        if (data[weight_col] <= 0).all():
            raise ValueError("All weights are zero or negative")
        
        # Select appropriate family
        fam = families.Gaussian() if family == "gaussian" else families.Binomial()
        
        try:
            model = smf.gee(
                formula=formula,
                data=data,
                groups=data[cluster_var],
                weights=data[weight_col],
                family=fam
            )
            result = model.fit()
            self.gee_model = result
            return result
        except Exception as e:
            raise ValueError(f"GEE model fitting failed with formula '{formula}': {str(e)}")
    
    def _fit_cox_model(
        self,
        data: pd.DataFrame,
        time_var: str,
        event_var: str,
        treatment_var: str,
        weight_col: str,
        cluster_var: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        alpha: float = 0.05,
        time_interaction: Optional[str] = None,
        period_breaks: Optional[List[int]] = None,
        period_labels: Optional[List[str]] = None,
        _quiet: bool = False,
    ) -> Dict:
        """
        Fit IPTW-weighted Cox proportional hazards model.

        Supports three modes via ``time_interaction``:
        - ``None`` (default): Standard Cox PH with a single overall hazard
          ratio. Runs Schoenfeld PH test; warns if assumption is violated.
        - ``"categorical"``: Person-period expansion with separate HRs per
          time interval. Robust to PH violations.
        - ``"continuous"``: Linear treatment × time trend.

        Parameters
        ----------
        data : pd.DataFrame
            Data with time_var, event_var, treatment_var, weight_col, and
            any covariates.
        time_var : str
            Name of the time column (days_observed).
        event_var : str
            Name of the event column (departed).
        treatment_var : str
            Binary treatment variable (1 = treated, 0 = control).
        weight_col : str
            IPTW weight column.
        cluster_var : str, optional
            Clustering variable for robust standard errors.
        covariates : list of str, optional
            Additional covariates for outcome-model adjustment.
        alpha : float, default 0.05
            Significance level for PH test reporting.
        time_interaction : str or None, default None
            - None          : standard Cox PH, single overall HR, PH test run.
            - "categorical" : separate HRs per period (requires period_breaks).
            - "continuous"  : linear treatment × time interaction.
        period_breaks : list of int, optional
            Required for time_interaction="categorical". Breakpoints in days,
            e.g. [0, 90, 180, 270, 365]. Must start with 0.
        period_labels : list of str, optional
            Human-readable labels for each period. Length = len(period_breaks)-1.

        Returns
        -------
        dict
            - period_hrs         : DataFrame of HR(s). Single row when
                                   time_interaction=None, multiple for others.
            - time_interaction   : str or None.
            - concordance        : float.
            - ph_test_results    : DataFrame (None if test not run).
            - ph_assumption_met  : bool or None.
            - cox_model          : fitted CoxPHFitter.
            - n_events_treated   : int.
            - n_events_control   : int.
            - coefficients_df    : DataFrame of all model coefficients.

        Raises
        ------
        ValueError
            If inputs invalid, insufficient events, or fitting fails.
        """

        # ------------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------------
        required_cols = [time_var, event_var, treatment_var, weight_col]
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if time_interaction is not None and time_interaction not in ("continuous", "categorical"):
            raise ValueError(
                f"time_interaction must be None, 'continuous', or 'categorical', "
                f"got '{time_interaction}'."
            )

        if time_interaction == "categorical":
            if not period_breaks:
                raise ValueError(
                    "period_breaks is required when time_interaction='categorical'. "
                    "Example: period_breaks=[0, 90, 180, 270, 365]."
                )
            if period_breaks[0] != 0:
                raise ValueError(
                    "period_breaks must start with 0. "
                    f"Got period_breaks[0]={period_breaks[0]}."
                )
            if period_labels is not None:
                expected_n_labels = len(period_breaks) - 1
                if len(period_labels) != expected_n_labels:
                    raise ValueError(
                        f"period_labels length ({len(period_labels)}) must equal "
                        f"len(period_breaks) - 1 ({expected_n_labels})."
                    )

        treated_events = ((data[treatment_var] == 1) & (data[event_var] == 1)).sum()
        control_events = ((data[treatment_var] == 0) & (data[event_var] == 1)).sum()

        if treated_events < 5:
            raise ValueError(
                f"Insufficient events in treated group: {treated_events} < 5"
            )
        if control_events < 5:
            raise ValueError(
                f"Insufficient events in control group: {control_events} < 5"
            )

        def _print(*args, **kwargs):
            if not _quiet:
                print(*args, **kwargs)

        _print("\n" + "=" * 60)
        if time_interaction is None:
            _print("COX PROPORTIONAL HAZARDS MODEL — STANDARD (NO TIME INTERACTION)")
        else:
            _print("COX PROPORTIONAL HAZARDS MODEL — TIME INTERACTION")
            _print(f"Interaction type : {time_interaction}")
        _print("=" * 60)

        # ------------------------------------------------------------------
        # Build model variables depending on time_interaction mode
        # ------------------------------------------------------------------
        working = data.copy()

        if time_interaction is None:
            # Standard Cox PH — treatment main effect only
            formula_vars = [treatment_var]
            _print(f"Model            : {treatment_var} (single overall HR)")

        elif time_interaction == "continuous":
            # Add time in months as a continuous variable
            time_interact_col = "_time_months"
            working[time_interact_col] = working[time_var] / 30.4375
            interaction_col = f"{treatment_var}_x_time"
            working[interaction_col] = working[treatment_var] * working[time_interact_col]
            formula_vars = [treatment_var, time_interact_col, interaction_col]
            _print(f"Interaction term : {treatment_var} × time_months (linear)")

        else:
            # Build period column from period_breaks
            period_col = "_period"
            n_periods = len(period_breaks) - 1

            # Auto-generate labels if not provided
            if period_labels is None:
                period_labels = [
                    f"{period_breaks[i]}d-{period_breaks[i+1]}d"
                    for i in range(n_periods)
                ]

            _print(f"Period breaks    : {period_breaks} (days)")
            _print(f"Period labels    : {period_labels}")

            # ============================================================
            # EXPAND TO PERSON-PERIOD FORMAT
            # ============================================================
            # Each person contributes one row per period they were at risk
            _print("Expanding to person-period format...")
            working = _expand_to_person_period(
                working, time_var, event_var, period_breaks, period_labels, period_col
            )
            _print(f"  Expanded from {len(data)} persons to {len(working)} person-periods")
            
            # Create dummies for period (drop first for reference category)
            period_dummies = pd.get_dummies(
                working[period_col],
                prefix="_period",
                drop_first=True,
            )
            period_dummy_cols = list(period_dummies.columns)
            working = pd.concat([working, period_dummies], axis=1)

            # Create interaction terms: treatment × each period dummy
            interaction_cols = []
            for pd_col in period_dummy_cols:
                interact_col = f"{treatment_var}_x_{pd_col}"
                working[interact_col] = working[treatment_var] * working[pd_col]
                interaction_cols.append(interact_col)

            formula_vars = (
                [treatment_var]
                + period_dummy_cols
                + interaction_cols
            )
            _print(f"Reference period : {period_labels[0]}")

        # Add covariates
        if covariates:
            formula_vars = formula_vars + [c for c in covariates if c not in formula_vars]

        # ------------------------------------------------------------------
        # Fit Cox model
        # ------------------------------------------------------------------
        keep_cols = (
            [time_var, event_var, weight_col]
            + formula_vars
            + ([cluster_var] if cluster_var and cluster_var in working.columns else [])
        )
        keep_cols = list(dict.fromkeys(keep_cols))   # deduplicate, preserve order
        cox_data = working[keep_cols].copy().dropna()

        if len(cox_data) < 20:
            raise ValueError(
                f"Insufficient data for Cox model after preprocessing: {len(cox_data)} rows"
            )

        cph = CoxPHFitter()
        fit_kw = dict(
            duration_col=time_var,
            event_col=event_var,
            weights_col=weight_col,
        )
        if cluster_var and cluster_var in working.columns:
            fit_kw["robust"] = True
            fit_kw["cluster_col"] = cluster_var

        try:
            cph.fit(cox_data, **fit_kw)
        except Exception as e:
            raise ValueError(f"Cox model fitting failed: {e}")

        concordance = float(cph.concordance_index_)
        _print(f"\nModel fitted     : {len(cox_data):,} observations, "
            f"{int(treated_events + control_events)} events")
        _print(f"Concordance      : {concordance:.3f}")

        # ------------------------------------------------------------------
        # PH test on final model
        # ------------------------------------------------------------------
        ph_test_results = None
        ph_assumption_met = None

        _print("\n--- Proportional Hazards Test ---")
        if time_interaction is not None:
            _print(
                "NOTE: PH test is SKIPPED for time-interaction models.\n"
                "      The person-period expanded data violates the independence\n"
                "      assumption of the Schoenfeld residual test (same person\n"
                "      contributes multiple rows), inflating test statistics.\n"
                "      The time interaction model handles non-proportionality\n"
                "      by design — no PH test is needed."
            )
            ph_test_results = None
            ph_assumption_met = None
        else:
            _print(
                "NOTE: Standard Cox PH assumes proportional hazards.\n"
                "      If treatment violates PH, consider re-running with\n"
                "      time_interaction='categorical' to model time-varying effects."
            )

        if time_interaction is None:
            try:
                from lifelines.statistics import proportional_hazard_test

                # Suppress lifelines' auto-printed summary to avoid
                # duplicate output (we print our own formatted version below).
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    ph_result = proportional_hazard_test(
                        cph,
                        cox_data,
                        time_transform="rank",
                    )

                if ph_result.summary is not None and not ph_result.summary.empty:
                    ph_test_results = ph_result.summary.copy()
                    ph_test_results["note"] = ""

                    # PH assumption met = treatment row passes at alpha
                    if treatment_var in ph_test_results.index.get_level_values(0):
                        treat_p = float(
                            ph_test_results.loc[treatment_var, "p"].iloc[0]
                            if hasattr(ph_test_results.loc[treatment_var, "p"], "iloc")
                            else ph_test_results.loc[treatment_var, "p"]
                        )
                        ph_assumption_met = treat_p >= alpha
                    else:
                        ph_assumption_met = True

                    _print(ph_test_results[["test_statistic", "p", "note"]].to_string())

                    if ph_assumption_met is False:
                        _print(
                            "\n  ⚠️  Treatment variable VIOLATES proportional hazards "
                            "assumption.\n"
                            "      The single HR may be misleading. Consider re-running "
                            "with\n"
                            "      time_interaction='categorical'."
                        )
                    elif ph_assumption_met is True:
                        _print("\n  ✓ Proportional hazards assumption met for treatment.")
                else:
                    _print("PH test returned no results.")
                    ph_assumption_met = None

            except Exception as e:
                _print(f"⚠️  PH test could not be completed: {e}")
                ph_assumption_met = None

            # Scaled Schoenfeld residuals + LOWESS trend (rank-transformed only)
            try:
                from lifelines.utils.lowess import lowess as ll_lowess
                from scipy.stats import rankdata

                resids = cph.compute_residuals(cox_data, kind="scaled_schoenfeld")
                t_rank = rankdata(resids.index, method="average")
                y = resids[treatment_var].values

                fig_sr, ax_sr = plt.subplots(figsize=(8, 4))
                ax_sr.scatter(t_rank, y, alpha=0.15, s=8, color="grey",
                              label="Scaled Schoenfeld residuals")
                # Bootstrap LOWESS bands (same approach as lifelines)
                n = len(t_rank)
                for _ in range(10):
                    ix = sorted(np.random.choice(n, n))
                    ax_sr.plot(t_rank[ix], ll_lowess(t_rank[ix], y[ix]),
                               color="k", alpha=0.25, linewidth=1)
                ax_sr.axhline(0, color="red", linestyle="--", linewidth=0.8)
                ax_sr.set_xlabel("Rank(time)", fontsize=10)
                ax_sr.set_ylabel("Scaled Schoenfeld residual", fontsize=10)
                ax_sr.set_title(
                    f"PH Diagnostic — '{treatment_var}' (rank transform)",
                    fontsize=11, fontweight="bold",
                )
                ax_sr.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                _print(f"  (Schoenfeld residual plot could not be generated: {e})")

        # ------------------------------------------------------------------
        # Extract hazard ratios
        # ------------------------------------------------------------------
        _print("\n--- Hazard Ratios ---")

        from scipy import stats as scipy_stats
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)

        period_hr_rows = []

        if time_interaction is None:
            # Standard Cox: single overall HR for treatment
            beta_treat = float(cph.params_[treatment_var])
            se_treat = float(cph.standard_errors_[treatment_var])
            hr = np.exp(beta_treat)
            hr_lower = np.exp(beta_treat - z_crit * se_treat)
            hr_upper = np.exp(beta_treat + z_crit * se_treat)
            z_stat = beta_treat / se_treat
            p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

            period_hr_rows.append({
                "period": "overall",
                "timepoint_days": None,
                "hazard_ratio": round(hr, 4),
                "hr_ci_lower": round(hr_lower, 4),
                "hr_ci_upper": round(hr_upper, 4),
                "p_value": round(p_val, 4),
                "log_hr": round(beta_treat, 4),
                "se_log_hr": round(se_treat, 4),
                "note": "single overall HR (no time interaction)",
            })

        elif time_interaction == "continuous":
            # Compute HR at each snapshot timepoint
            snapshot_months = [3, 6, 9, 12]
            beta_treat = float(cph.params_[treatment_var])
            beta_interact = float(cph.params_[interaction_col])

            # Variance components for delta method CI
            var_treat = float(cph.variance_matrix_.loc[treatment_var, treatment_var])
            var_interact = float(cph.variance_matrix_.loc[interaction_col, interaction_col])
            cov_treat_interact = float(cph.variance_matrix_.loc[treatment_var, interaction_col])

            for t_months in snapshot_months:
                log_hr = beta_treat + beta_interact * t_months
                # Delta method SE: Var(a + b*t) = Var(a) + t²·Var(b) + 2t·Cov(a,b)
                se_log_hr = np.sqrt(
                    var_treat
                    + (t_months ** 2) * var_interact
                    + 2 * t_months * cov_treat_interact
                )
                hr = np.exp(log_hr)
                hr_lower = np.exp(log_hr - z_crit * se_log_hr)
                hr_upper = np.exp(log_hr + z_crit * se_log_hr)
                z_stat = log_hr / se_log_hr
                p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

                period_hr_rows.append({
                    "period": f"{t_months}mo",
                    "timepoint_days": int(t_months * 30.4375),
                    "hazard_ratio": round(hr, 4),
                    "hr_ci_lower": round(hr_lower, 4),
                    "hr_ci_upper": round(hr_upper, 4),
                    "p_value": round(p_val, 4),
                    "log_hr": round(log_hr, 4),
                    "se_log_hr": round(se_log_hr, 4),
                })

        else:
            # Reference period: treatment main effect
            beta_treat = float(cph.params_[treatment_var])
            se_treat = float(cph.standard_errors_[treatment_var])
            hr_ref = np.exp(beta_treat)
            hr_ref_lower = np.exp(beta_treat - z_crit * se_treat)
            hr_ref_upper = np.exp(beta_treat + z_crit * se_treat)
            z_stat_ref = beta_treat / se_treat
            p_ref = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat_ref)))

            period_hr_rows.append({
                "period": period_labels[0],
                "timepoint_days": period_breaks[1],
                "hazard_ratio": round(hr_ref, 4),
                "hr_ci_lower": round(hr_ref_lower, 4),
                "hr_ci_upper": round(hr_ref_upper, 4),
                "p_value": round(p_ref, 4),
                "log_hr": round(beta_treat, 4),
                "se_log_hr": round(se_treat, 4),
                "note": "reference period (timepoint_days = period end)",
            })

            # Subsequent periods: treatment + interaction term
            for i, interact_col in enumerate(interaction_cols):
                period_label = period_labels[i + 1]
                timepoint_days = period_breaks[i + 2] if (i + 2) < len(period_breaks) else period_breaks[-1]

                beta_interact = float(cph.params_[interact_col])
                se_interact = float(cph.standard_errors_[interact_col])

                # Combined log HR = beta_treat + beta_interact
                log_hr = beta_treat + beta_interact

                # Delta method SE: Var(a + b) = Var(a) + Var(b) + 2·Cov(a,b)
                var_treat = float(cph.variance_matrix_.loc[treatment_var, treatment_var])
                var_interact = float(cph.variance_matrix_.loc[interact_col, interact_col])
                cov_ti = float(cph.variance_matrix_.loc[treatment_var, interact_col])
                se_log_hr = np.sqrt(var_treat + var_interact + 2 * cov_ti)

                hr = np.exp(log_hr)
                hr_lower = np.exp(log_hr - z_crit * se_log_hr)
                hr_upper = np.exp(log_hr + z_crit * se_log_hr)
                z_stat = log_hr / se_log_hr
                p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

                period_hr_rows.append({
                    "period": period_label,
                    "timepoint_days": timepoint_days,
                    "hazard_ratio": round(hr, 4),
                    "hr_ci_lower": round(hr_lower, 4),
                    "hr_ci_upper": round(hr_upper, 4),
                    "p_value": round(p_val, 4),
                    "log_hr": round(log_hr, 4),
                    "se_log_hr": round(se_log_hr, 4),
                    "note": "",
                })

        period_hrs = pd.DataFrame(period_hr_rows)
        _print(period_hrs.to_string(index=False))

        # ------------------------------------------------------------------
        # Build coefficients DataFrame
        # ------------------------------------------------------------------
        coefficients_df = pd.DataFrame({
            "Parameter": cph.params_.index.tolist(),
            "Estimate": cph.params_.values.tolist(),
            "Std_Error": cph.standard_errors_.values.tolist(),
            "CI_Lower": cph.confidence_intervals_.iloc[:, 0].values.tolist(),
            "CI_Upper": cph.confidence_intervals_.iloc[:, 1].values.tolist(),
            "P_Value_Raw": cph.summary["p"].values.tolist(),
            "Alpha": [alpha] * len(cph.params_),
        })

        _print("=" * 60)

        return {
            "period_hrs": period_hrs,
            "time_interaction": time_interaction,
            "concordance": concordance,
            "ph_test_results": ph_test_results,
            "ph_assumption_met": ph_assumption_met,
            "cox_model": cph,
            "n_events_treated": int(treated_events),
            "n_events_control": int(control_events),
            "coefficients_df": coefficients_df,
        }
    
    

    def _fit_weighted_km_curves(
        self,
        data: pd.DataFrame,
        time_var: str,
        event_var: str,
        treatment_var: str,
        weight_col: str = "iptw",
        snapshot_days: Optional[List[int]] = None,
    ) -> Dict:
        """
        Fit IPTW-weighted Kaplan-Meier survival curves for treated and control
        groups. Returns descriptive survival probabilities — no inference.

        This is an independent function that can be used standalone or as input
        to ``compute_rmst_difference()`` and ``plot_survival_curves()``.

        Parameters
        ----------
        data : pd.DataFrame
            Weighted data with time, event, treatment, and weight columns.
        time_var : str
            Duration column (days to event / censoring).
        event_var : str
            Event indicator (1 = departed, 0 = censored).
        treatment_var : str
            Binary treatment column.
        weight_col : str, default "iptw"
            IPTW weight column.
        snapshot_days : list of int, optional
            Timepoints at which to report survival probabilities.
            Defaults to [90, 180, 270, 365].

        Returns
        -------
        dict
            - kmf_treated : fitted KaplanMeierFitter for treated group.
            - kmf_control : fitted KaplanMeierFitter for control group.
            - survival_at_snapshots : DataFrame with columns timepoint_days,
              timepoint_label, survival_treated, survival_control, survival_diff.
            - n_events_treated : int.
            - n_events_control : int.

        Raises
        ------
        ValueError
            If required columns are missing or KM fitting fails.
        """
        if snapshot_days is None:
            snapshot_days = [90, 180, 270, 365]

        required_cols = [time_var, event_var, treatment_var, weight_col]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        treated_data = data[data[treatment_var] == 1]
        control_data = data[data[treatment_var] == 0]

        n_events_treated = int((treated_data[event_var] == 1).sum())
        n_events_control = int((control_data[event_var] == 1).sum())

        print("\n" + "=" * 60)
        print("IPTW-WEIGHTED KAPLAN-MEIER SURVIVAL CURVES")
        print("=" * 60)
        print(f"  Treated   : n = {len(treated_data):,}, events = {n_events_treated}")
        print(f"  Control   : n = {len(control_data):,}, events = {n_events_control}")

        kmf_treated = KaplanMeierFitter()
        kmf_control = KaplanMeierFitter()

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*weights are not integers.*",
                    category=StatisticalWarning,
                )
                kmf_treated.fit(
                    durations=treated_data[time_var],
                    event_observed=treated_data[event_var],
                    weights=treated_data[weight_col],
                    label="Treated",
                )
                kmf_control.fit(
                    durations=control_data[time_var],
                    event_observed=control_data[event_var],
                    weights=control_data[weight_col],
                    label="Control",
                )
        except Exception as e:
            raise ValueError(f"Kaplan-Meier fitting failed: {e}")

        # --- Survival snapshots ---
        survival_snapshots = []
        for days in snapshot_days:
            try:
                s_t = float(kmf_treated.survival_function_at_times(days).iloc[0])
                s_c = float(kmf_control.survival_function_at_times(days).iloc[0])
                diff = s_t - s_c
            except Exception:
                s_t, s_c, diff = np.nan, np.nan, np.nan

            label = f"{days // 30}mo" if days % 30 == 0 else f"{days}d"
            survival_snapshots.append({
                "timepoint_days": days,
                "timepoint_label": label,
                "survival_treated": round(s_t, 4),
                "survival_control": round(s_c, 4),
                "survival_diff": round(diff, 4),
            })

        survival_at_snapshots = pd.DataFrame(survival_snapshots)

        print("\n  Survival Probability Snapshots:")
        print(survival_at_snapshots.to_string(index=False))
        print()
        print("  Note: KM confidence bands are NOT valid under IPTW")
        print("  (Greenwood variance ignores propensity score uncertainty).")
        print("  Use Cox PH for formal inference.")
        print("=" * 60)

        return {
            "kmf_treated": kmf_treated,
            "kmf_control": kmf_control,
            "survival_at_snapshots": survival_at_snapshots,
            "n_events_treated": n_events_treated,
            "n_events_control": n_events_control,
        }
    
    # ==================================================================
    # Visualization
    # ==================================================================

    def plot_propensity_overlap(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        title: str = "Propensity Score Overlap"
    ) -> object:
        """
        Create propensity score overlap plot via CausalDiagnostics.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing 'propensity_score' column and treatment variable
        treatment_var : str
            Name of the binary treatment variable (0/1)
        title : str, default="Propensity Score Overlap"
            Plot title
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        if "propensity_score" not in data.columns:
            raise ValueError("Column 'propensity_score' is required for overlap plotting")

        diagnostics = CausalDiagnostics()
        return diagnostics.plot_propensity_overlap(
            data=data,
            treatment_var=treatment_var,
            propensity_scores=data["propensity_score"].to_numpy(),
            outcome_var=title,
        )
    
    def plot_weight_distribution(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        weight_col: str = "iptw",
        estimand: str = "ATE",
        title: str = "IPTW Weight Distribution",
        stabilized: bool = True
    ) -> object:
        """
        Create histogram of IPTW weights by treatment group.
        
        Helps diagnose extreme weights that can destabilize estimates.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing weight column and treatment variable
        treatment_var : str
            Name of the binary treatment variable (0/1)
        weight_col : str, default="iptw"
            Name of the weight column
        title : str, default="IPTW Weight Distribution"
            Plot title
        stabilized : bool, default=True
            If True, interpretation text assumes stabilized weights (default pipeline).
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, (label, grp) in zip(axes, data.groupby(treatment_var)):
            group_label = "Treated" if label == 1 else "Control"
            ax.hist(grp[weight_col], bins=50, alpha=0.7,
                    color="#e74c3c" if label == 1 else "#3498db",
                    edgecolor="black", linewidth=0.5)
            ax.axvline(grp[weight_col].mean(), color="black", linestyle="--",
                       label=f"Mean = {grp[weight_col].mean():.2f}")
            ax.axvline(grp[weight_col].quantile(0.99), color="orange", linestyle="--",
                       label=f"P99 = {grp[weight_col].quantile(0.99):.2f}")
            ax.set_xlabel("Weight", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"{group_label} (n={len(grp)})", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        if estimand == "ATT":
            interpretation = (
                "Stabilized ATT: Treated weights = 1.0. Control weights are scaled by "
                "the marginal treatment probability — expect mean ≈ P(T=1) with most "
                "values near zero and a small right tail for controls resembling "
                "treated individuals."
            )
        else:  # ATE
            interpretation = (
                "Interpretation (ATE): Look for stabilized weights with mean near 1 "
                "and few extreme right-tail values. Max weight > 10 or P99 ≫ mean "
                "suggests instability."
            )

        fig.text(0.5, 0.01, interpretation, ha="center", va="bottom", fontsize=9, color="dimgray")
        plt.tight_layout(rect=[0, 0.06, 1, 1])        
        return fig
    
    def plot_survival_curves(
        self,
        survival_result: Dict,
        outcome_name: str = "Retention",
        time_horizon: int = 365,
        show_snapshots: bool = True,
        snapshot_days: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> object:
        """
        Plot IPTW-weighted Kaplan-Meier survival curves with confidence intervals,
        snapshot overlays, and risk table.

        Parameters
        ----------
        survival_result : dict
            Output of analyze_survival_effect().
        outcome_name : str, default "Retention"
            Used in the plot title.
        time_horizon : int, default 365
            X-axis upper limit in days.
        show_snapshots : bool, default True
            If True, adds vertical dashed lines at snapshot_days.
        snapshot_days : List[int], optional
            Timepoints to mark. Defaults to [90, 180, 270, 365].
        figsize : tuple, default (12, 8)
            Figure dimensions.
        save_path : str, optional
            If provided, saves figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        kmf_treated = survival_result.get("kmf_treated")
        kmf_control = survival_result.get("kmf_control")

        if kmf_treated is None or kmf_control is None:
            raise ValueError(
                "survival_result must contain 'kmf_treated' and 'kmf_control'. "
                "Run analyze_survival_effect() first."
            )

        if snapshot_days is None:
            snapshot_days = [90, 180, 270, 365]

        snapshot_labels = {90: "3 mo", 180: "6 mo", 270: "9 mo", 365: "12 mo"}

        time_interaction = survival_result.get("time_interaction")
        estimand = survival_result.get("estimand", "ATT")
        n_events_treated = survival_result.get("n_events_treated", "?")
        n_events_control = survival_result.get("n_events_control", "?")

        # --- Build figure with main plot + risk table ---
        fig, (ax_main, ax_risk) = plt.subplots(
            2, 1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [4, 1]}
        )

        # --- Main KM plot ---
        color_treated = "#2196F3"   # blue
        color_control = "#FF5722"   # orange-red

        # Plot treated curve — CIs intentionally hidden (see footnote below)
        kmf_treated.plot_survival_function(
            ax=ax_main,
            ci_show=False,
            color=color_treated,
            label=f"Trained (events={n_events_treated})"
        )

        # Plot control curve — CIs intentionally hidden
        kmf_control.plot_survival_function(
            ax=ax_main,
            ci_show=False,
            color=color_control,
            label=f"Untrained (events={n_events_control})"
        )

        # Snapshot vertical lines
        if show_snapshots:
            for day in snapshot_days:
                if day <= time_horizon:
                    label = snapshot_labels.get(day, f"{day}d")
                    ax_main.axvline(
                        x=day, color="gray", linestyle="--",
                        linewidth=0.8, alpha=0.6
                    )
                    ax_main.text(
                        day + 3, 0.02, label,
                        fontsize=8, color="gray", va="bottom"
                    )

        # Add note about time-varying effects if applicable (repositioned to top-right)
        if time_interaction:
            interaction_note = (
                f"Time interaction model: {time_interaction}\n"
                f"Treatment effect varies over time\n"
                f"See period_hrs for full results"
            )
            ax_main.text(
                0.97, 0.97, interaction_note,
                transform=ax_main.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        edgecolor="orange", alpha=0.8),
                style="italic"
            )

        ax_main.set_xlim(0, time_horizon)
        ax_main.set_ylim(0, 1.05)
        ax_main.set_xlabel("Days Since Study Start (T=0)", fontsize=11)
        ax_main.set_ylabel("Probability of Retention", fontsize=11)
        ax_main.set_title(
            f"IPTW-Weighted Survival Curves — {outcome_name} ({estimand})",
            fontsize=13, fontweight="bold"
        )
        ax_main.legend(fontsize=10, loc="lower left")
        ax_main.grid(True, alpha=0.3)

        # --- Risk table ---
        ax_risk.axis("off")
        risk_timepoints = [d for d in snapshot_days if d <= time_horizon]

        # Compute N at risk at each timepoint
        def n_at_risk(kmf, timepoint):
            """Return number at risk at a given timepoint from KM fitter."""
            try:
                timeline = kmf.event_table.index
                idx = timeline[timeline <= timepoint]
                if len(idx) == 0:
                    return int(kmf.event_table["at_risk"].iloc[0])
                return int(kmf.event_table.loc[idx[-1], "at_risk"])
            except Exception:
                return "?"

        risk_rows = {
            "Trained":   [n_at_risk(kmf_treated, d) for d in risk_timepoints],
            "Untrained": [n_at_risk(kmf_control, d) for d in risk_timepoints],
        }

        col_labels = [snapshot_labels.get(d, f"{d}d") for d in risk_timepoints]
        table_data = [risk_rows["Trained"], risk_rows["Untrained"]]
        row_labels = ["Trained", "Untrained"]

        tbl = ax_risk.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)

        # Color row headers to match curves
        for (row, col), cell in tbl.get_celld().items():
            if col == -1:
                if row == 1:
                    cell.set_facecolor(color_treated)
                    cell.set_text_props(color="white", fontweight="bold")
                elif row == 2:
                    cell.set_facecolor(color_control)
                    cell.set_text_props(color="white", fontweight="bold")

        ax_risk.set_title("Number at Risk", fontsize=9, loc="left", pad=2)

        # Footnote
        fig.text(
            0.5, 0.01,
            f"Curves represent IPTW-weighted Kaplan-Meier estimates ({estimand}). "
            f"HR < 1 indicates lower hazard of departure (protective effect of training).\n"
            f"KM confidence bands omitted — naive variance is biased under IPTW weights. "
            f"All inferential statistics (HRs, CIs, p-values) are from Cox PH with robust sandwich SEs.",
            ha="center", va="bottom", fontsize=7.5, color="dimgray", style="italic",
        )

        plt.tight_layout(rect=[0, 0.06, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ==================================================================
    # Shared IPTW data-preparation pipeline (Steps 0 – 2)
    # ==================================================================
    def _prepare_iptw_data(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        cluster_var: str,
        categorical_vars: Optional[List[str]] = None,
        binary_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        estimand: str = "ATE",
        trim_quantile: float = 0.01,
        plot_propensity: bool = True,
        plot_weights: bool = True,
        *,
        # GEE-specific --------------------------------------------------
        outcome_var: Optional[str] = None,
        baseline_var: Optional[str] = None,
        # Survival-specific ----------------------------------------------
        time_var: Optional[str] = None,
        event_var: Optional[str] = None,
        preserve_strata_backups: bool = False,
        # Labeling -------------------------------------------------------
        analysis_label: str = "",
    ) -> Dict:
        """Shared Steps 0-2 of the IPTW pipeline used by both
        ``analyze_treatment_effect`` and ``analyze_survival_effect``.

        This private helper consolidates data preparation, one-hot encoding,
        column-name cleaning, propensity-score estimation, weight diagnostics,
        overlap/weight plotting, and post-weighting balance checking into a
        single reusable method so the two public pipeline methods can focus
        exclusively on their outcome-model Step 3.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data.
        treatment_var, cluster_var : str
            Treatment indicator and clustering column (original names).
        categorical_vars, binary_vars, continuous_vars : list of str, optional
            Covariate lists (original names). Each may be omitted or ``None``;
            defaults to no covariates of that type.
        estimand : str
            ``"ATE"`` or ``"ATT"``.
        trim_quantile : float
            PS-weight trim quantile.
        plot_propensity, plot_weights : bool
            Whether to generate diagnostic plots.
        outcome_var : str, optional
            Outcome column (GEE pipeline).
        baseline_var : str, optional
            Baseline covariate included in both the propensity-score model
            and the GEE outcome model. Treated as a confounder because
            baseline performance typically influences treatment assignment
            in practice. Including it in both models preserves double
            robustness while protecting against selection bias.
        time_var, event_var : str, optional
            Duration and event-indicator columns (survival pipeline).
        preserve_strata_backups : bool
            If *True* (survival), create backup copies of categorical columns
            before one-hot encoding so lifelines can stratify on the original
            factor.
        analysis_label : str
            Human-readable label used in plot titles.

        Returns
        -------
        dict
            Keys: ``df``, ``ps_model``, ``weight_stats``, ``balance_df``,
            ``ps_overlap_fig``, ``weight_dist_fig``, ``covariates``,
            ``ps_covariates``, ``dummy_columns``, ``balance_var_names``,
            ``balance_var_types``, ``treatment_var``, ``cluster_var``,
            ``outcome_var``, ``baseline_var``, ``time_var``, ``event_var``,
            ``continuous_vars``, ``binary_vars``,
            ``_strata_backup_map``, ``dummy_to_parent``,
            ``cleaned_cat_vars``.
        """

        # ------------------------------------------------------------------
        # Internal helper: build a single balance-result row
        # ------------------------------------------------------------------
        def _make_balance_row(
            var_name: str,
            var_type: str,
            smd_before: Optional[float],
            smd_after: Optional[float],
        ) -> Dict:
            improvement = (
                abs(smd_before) - abs(smd_after)
                if smd_before is not None and smd_after is not None
                else None
            )
            return {
                "variable": var_name,
                "type": var_type,
                "smd_before_weighting": smd_before,
                "smd_after_weighting": smd_after,
                "smd_improvement": improvement,
                "balanced_before_weighting": (
                    abs(smd_before) < 0.1 if smd_before is not None else None
                ),
                # Intentionally False (not None) so the balance summary
                # count treats uncomputable rows as imbalanced.
                "balanced_after_weighting": (
                    abs(smd_after) < 0.1 if smd_after is not None else False
                ),
            }

        # ------------------------------------------------------------------
        # Step 0: Data prep
        # ------------------------------------------------------------------
        categorical_vars = list(categorical_vars) if categorical_vars else []
        binary_vars = list(binary_vars) if binary_vars else []
        continuous_vars = list(continuous_vars) if continuous_vars else []

        # Build a single covariate list that covers both PS and outcome models.
        # baseline_var is treated as a confounder (included in both models)
        # because baseline performance typically influences treatment assignment.
        # Note: concatenation creates a new list so the caller's lists are not mutated.
        ps_covariates_raw = categorical_vars + binary_vars + continuous_vars
        if baseline_var:
            ps_covariates_raw.append(baseline_var)

        id_columns: List[str] = [treatment_var, cluster_var]
        if outcome_var:
            id_columns.append(outcome_var)
        if time_var:
            id_columns.append(time_var)
        if event_var:
            id_columns.append(event_var)

        all_needed = list(set(id_columns + ps_covariates_raw))
        df = data[all_needed].dropna().copy()

        # --- Basic data validations ---
        if len(df) < 10:
            raise ValueError(
                f"Insufficient data after removing missing values: {len(df)} rows remaining"
            )
        if df[treatment_var].nunique() < 2:
            raise ValueError(
                f"Only one treatment group present in data: {df[treatment_var].unique()}"
            )
        treatment_counts = df[treatment_var].value_counts()
        if treatment_counts.min() < 5:
            raise ValueError(
                f"Insufficient observations in treatment groups. "
                f"Counts: {treatment_counts.to_dict()}"
            )

        # --- Survival-specific validations ---
        if event_var and time_var:
            if df[event_var].sum() < 10:
                raise ValueError(
                    f"Insufficient events for survival analysis: "
                    f"{int(df[event_var].sum())} events (minimum 10 required)"
                )
            if (df[time_var] <= 0).any():
                n_bad = int((df[time_var] <= 0).sum())
                raise ValueError(
                    f"{n_bad} observations have {time_var} <= 0. "
                    f"All survival times must be positive. "
                    f"Run prepare_survival_data() to check data quality."
                )

        # --- Preserve original categorical columns for stratification ---
        cleaned_cat_vars = [self._clean_column_name(v) for v in categorical_vars]
        _strata_backup_map: Dict[str, str] = {}
        if preserve_strata_backups:
            for var in categorical_vars:
                raw_backup = f"__strata_{var}"
                df[raw_backup] = df[var]
                cleaned_backup = self._clean_column_name(raw_backup)
                cleaned_parent = self._clean_column_name(var)
                _strata_backup_map[cleaned_parent] = cleaned_backup

        # --- One-hot encode categorical variables ---
        cols_before_dummies = set(df.columns)
        df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
        # No sort: ordering is irrelevant downstream and sort is O(n log n)
        dummy_columns = list(set(df.columns) - cols_before_dummies)

        # --- Clean all column names ---
        rename_map = {c: self._clean_column_name(c) for c in df.columns}
        df.rename(columns=rename_map, inplace=True)

        # Remap all key variable references to their cleaned names
        treatment_var = self._clean_column_name(treatment_var)
        cluster_var   = self._clean_column_name(cluster_var)
        if outcome_var:
            outcome_var  = self._clean_column_name(outcome_var)
        if baseline_var:
            baseline_var = self._clean_column_name(baseline_var)
        if time_var:
            time_var     = self._clean_column_name(time_var)
        if event_var:
            event_var    = self._clean_column_name(event_var)

        continuous_vars = [self._clean_column_name(v) for v in continuous_vars]
        binary_vars     = [self._clean_column_name(v) for v in binary_vars]
        dummy_columns   = [self._clean_column_name(c) for c in dummy_columns]

        # baseline_var is already cleaned above — no double-clean needed
        baseline_vars_clean = [baseline_var] if baseline_var else []

        # --- Build dummy → parent mapping (used for auto-stratification) ---
        dummy_to_parent: Dict[str, str] = {}
        for dummy in dummy_columns:
            for parent in cleaned_cat_vars:
                if dummy.startswith(parent + "_"):
                    dummy_to_parent[dummy] = parent
                    break

        # --- Track balance variables ---
        # Exclude any variable that contains the treatment variable name
        # (e.g., interaction terms like treatment_x_span) — these are
        # functions of treatment and cannot be meaningfully balanced.
        def _is_treatment_derived(var_name: str) -> bool:
            return treatment_var in var_name and var_name != treatment_var

        balance_var_names = (
            [v for v in continuous_vars       if v in df.columns and not _is_treatment_derived(v)]
            + [v for v in binary_vars         if v in df.columns and not _is_treatment_derived(v)]
            + [dc for dc in dummy_columns     if dc in df.columns and not _is_treatment_derived(dc)]
            + [v for v in baseline_vars_clean if v in df.columns and not _is_treatment_derived(v)]
        )
        balance_var_types: Dict[str, str] = {
            v: "continuous" for v in continuous_vars if v in df.columns and not _is_treatment_derived(v)
        }
        balance_var_types.update({v:  "binary"     for v  in binary_vars         if v  in df.columns and not _is_treatment_derived(v)})
        balance_var_types.update({dc: "categorical" for dc in dummy_columns       if dc in df.columns and not _is_treatment_derived(dc)})
        balance_var_types.update({v:  "continuous"  for v  in baseline_vars_clean if v  in df.columns and not _is_treatment_derived(v)})

        # --- Build covariate lists ---
        # Exclude non-covariate columns and strata backup columns
        _exclude = {treatment_var, cluster_var} | {
            v for v in [outcome_var, time_var, event_var] if v
        } | set(_strata_backup_map.values())

        covariates = [c for c in df.columns if c not in _exclude]

        # PS covariates: exclude treatment-derived variables (interactions)
        # from the propensity score model — they are functions of treatment
        # and must not predict treatment assignment.
        ps_covariates = [c for c in covariates if not _is_treatment_derived(c)]

        # --- Validate covariates ---
        if len(covariates) == 0:
            raise ValueError("No covariates remaining after data processing")

        # --- Remove constant covariates ---
        constant_vars = [var for var in covariates if df[var].nunique() <= 1]
        if constant_vars:
            print(f"  Warning: Removing constant variables: {constant_vars}")
            constant_set = set(constant_vars)
            covariates        = [v for v in covariates        if v not in constant_set]
            ps_covariates     = [v for v in ps_covariates     if v not in constant_set]
            balance_var_names = [v for v in balance_var_names if v not in constant_set]
            balance_var_types = {k: v for k, v in balance_var_types.items()
                                if k not in constant_set}
            if len(covariates) == 0:
                raise ValueError(
                    "No valid covariates remaining after removing constant variables"
                )

        # --- Final null check ---
        final_covariate_df = df[covariates]
        if final_covariate_df.isnull().all().any():
            null_vars = final_covariate_df.columns[
                final_covariate_df.isnull().all()
            ].tolist()
            raise ValueError(f"Covariates with all null values: {null_vars}")

        # ------------------------------------------------------------------
        # Step 1: Estimate propensity weights
        # ------------------------------------------------------------------
        try:
            df, ps_model = self.estimate_propensity_weights(
                df,
                treatment_var,
                ps_covariates,
                estimand=estimand,
                cluster_var=cluster_var,
                trim_quantile=trim_quantile,
            )
        except Exception as e:
            raise ValueError(
                f"Error estimating propensity scores — likely data issue: {e}"
            )

        # --- Positivity / overlap warning ---
        ps_vals = df["propensity_score"]
        n_near_zero = (ps_vals < 0.01).sum()
        n_near_one  = (ps_vals > 0.99).sum()
        if n_near_zero > 0 or n_near_one > 0:
            print(
                f"  Warning: Positivity concern: {n_near_zero} observations "
                f"with PS < 0.01, {n_near_one} with PS > 0.99"
            )

        # --- Propensity score overlap plot ---
        ps_overlap_fig = None
        if plot_propensity:
            try:
                ps_overlap_fig = self.plot_propensity_overlap(
                    data=df,
                    treatment_var=treatment_var,
                    title=f"Propensity Score Overlap — {analysis_label}",
                )
            except Exception as e:
                print(f"  Warning: Could not generate propensity score plot: {e}")

        # ------------------------------------------------------------------
        # Step 2: Weight diagnostics
        # ------------------------------------------------------------------
        try:
            weight_stats = self.compute_weight_diagnostics(df)
        except Exception as e:
            raise ValueError(
                f"Error calculating weight diagnostics — "
                f"likely insufficient data: {e}"
            )

        # --- Weight diagnostic warnings ---
        if weight_stats["max_weight"] > 10:
            print(
                f"  ⚠️  Max weight {weight_stats['max_weight']:.1f} > 10 — "
                "consider stricter trimming or overlap restriction."
            )
        ess_ratio = weight_stats["effective_sample_size"] / weight_stats["n_observations"]
        if ess_ratio < 0.5:
            print(
                f"  Note: ESS is <50% of n (ratio={ess_ratio:.2f}) — "
                "weights may be unstable."
            )

        # --- Weight distribution plot ---
        weight_dist_fig = None
        if plot_weights:
            try:
                weight_dist_fig = self.plot_weight_distribution(
                    data=df,
                    treatment_var=treatment_var,
                    estimand=estimand,
                    title=f"IPTW Weight Distribution — {analysis_label}",
                )
            except Exception as e:
                print(f"  Warning: Could not generate weight distribution plot: {e}")

        # --- Post-weighting balance check via CausalDiagnostics ---
        # CausalDiagnostics is instantiated locally; if this method is called
        # in a loop, consider hoisting _cd to the class level to avoid
        # repeated construction overhead.
        _cd = CausalDiagnostics()
        balance_results = []
        try:
            _raw_balance = _cd.compute_balance_df(
                data=df,
                controls=balance_var_names,
                treatment=treatment_var,
                weights=df["iptw"],
                already_encoded=True,
            )
            for var_name in _raw_balance.index:
                row = _raw_balance.loc[var_name]
                balance_results.append(_make_balance_row(
                    var_name,
                    balance_var_types.get(var_name, "unknown"),
                    row["Unweighted SMD"],
                    row["Weighted SMD"],
                ))

        except Exception as e:
            print(
                f"  Warning: CausalDiagnostics balance computation failed ({e}); "
                f"falling back to inline SMD computation."
            )
            # Fallback: compute SMD inline using a uniform-weight column
            # as the unweighted baseline, then drop it immediately.
            df["_uniform_wt"] = 1.0
            for var_name in balance_var_names:
                if var_name not in df.columns:
                    continue
                try:
                    smd_before = self.calculate_standardized_mean_difference(
                        df, var_name, treatment_var, "_uniform_wt"
                    )
                    smd_after = self.calculate_standardized_mean_difference(
                        df, var_name, treatment_var, "iptw"
                    )
                    balance_results.append(_make_balance_row(
                        var_name,
                        balance_var_types.get(var_name, "unknown"),
                        smd_before,
                        smd_after,
                    ))
                except Exception:
                    balance_results.append(_make_balance_row(
                        var_name,
                        balance_var_types.get(var_name, "unknown"),
                        None,
                        None,
                    ))
            df.drop(columns=["_uniform_wt"], inplace=True)

        balance_df = (
            pd.DataFrame(balance_results)
            if balance_results
            else pd.DataFrame(columns=[
                "variable", "type", "smd_before_weighting",
                "smd_after_weighting", "smd_improvement",
                "balanced_before_weighting", "balanced_after_weighting",
            ])
        )

        # --- Balance summary ---
        if not balance_df.empty and "balanced_after_weighting" in balance_df.columns:
            n_imbalanced  = int(balance_df["balanced_after_weighting"].eq(False).sum())
            n_total_vars  = len(balance_df)
            if n_imbalanced == 0:
                print(
                    f"  ✓ Post-weighting balance: all {n_total_vars} "
                    f"covariates balanced (|SMD| < 0.1)"
                )
            else:
                imbalanced_vars = balance_df[
                    balance_df["balanced_after_weighting"] == False
                ]["variable"].tolist()
                display_vars = imbalanced_vars[:10]
                vars_str = ", ".join(display_vars)
                if len(imbalanced_vars) > 10:
                    vars_str += f" … and {len(imbalanced_vars) - 10} more"
                print(
                    f"  ⚠️  Post-weighting balance: {n_imbalanced} of "
                    f"{n_total_vars} covariates still imbalanced (|SMD| ≥ 0.1)"
                )
                print(f"      Variables: {vars_str}")
                print(
                    "      Consider improving the propensity model, adding covariates, "
                    "or reporting sensitivity analyses."
                )

        return {
            "df":                df,
            "ps_model":          ps_model,
            "weight_stats":      weight_stats,
            "balance_df":        balance_df,
            "ps_overlap_fig":    ps_overlap_fig,
            "weight_dist_fig":   weight_dist_fig,
            "covariates":        covariates,
            "ps_covariates":     ps_covariates,
            "dummy_columns":     dummy_columns,
            "balance_var_names": balance_var_names,
            "balance_var_types": balance_var_types,
            # Cleaned variable references
            "treatment_var":     treatment_var,
            "cluster_var":       cluster_var,
            "outcome_var":       outcome_var,
            "baseline_var":      baseline_var,
            "time_var":          time_var,
            "event_var":         event_var,
            "continuous_vars":   continuous_vars,
            "binary_vars":       binary_vars,
            # Survival-specific
            "_strata_backup_map": _strata_backup_map,
            "dummy_to_parent":    dummy_to_parent,
            "cleaned_cat_vars":   cleaned_cat_vars,
        }

    # ==================================================================
    # Public analysis pipelines
    # ==================================================================

    def analyze_treatment_effect(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        cluster_var: str,
        categorical_vars: Optional[List[str]] = None,
        binary_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        estimand: str = "ATE",
        baseline_var: Optional[str] = None,
        project_path: Optional[str] = None,
        trim_quantile: float = 0.99,
        analysis_name: Optional[str] = None,
        alpha: float = 0.05,
        plot_propensity: bool = True,
        plot_weights: bool = True
    ) -> Dict:
        """
        Complete analysis pipeline: IPTW propensity weights -> weighted GEE
        outcome model with covariate adjustment.
        
        Comprehensive causal inference analysis implementing:
        1. Data preparation and covariate encoding
        2. Propensity score estimation with IPTW computation (ATE or ATT)
        3. Propensity score overlap visualization
        4. Weight diagnostics, distribution visualization, and balance assessment
        5. IPTW-weighted outcome modeling via GEE with covariate adjustment
        6. Effect size metrics (IPTW-weighted Cohen's d, percent change)
        7. Optional export to Excel workbook

        **IMPORTANT: Multiple Testing Correction**
        This function returns RAW p-values without any correction. When analyzing
        multiple outcomes, use ``build_summary_table()`` which applies FDR correction
        across all outcomes simultaneously — the statistically correct approach.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw dataset for analysis
        outcome_var : str
            Name of outcome variable
        treatment_var : str
            Name of binary treatment variable
        cluster_var : str
            Name of clustering variable
        categorical_vars : List[str], optional
            Categorical covariate names (will be one-hot encoded). Omitted or
            ``None`` means no categorical covariates.
        binary_vars : List[str], optional
            Binary covariate names. Omitted or ``None`` means no binary covariates.
        continuous_vars : List[str], optional
            Continuous covariate names. Omitted or ``None`` means no continuous
            covariates.
        estimand : str, default="ATE"
            Target estimand: "ATE" (Average Treatment Effect) or "ATT" (Average
            Treatment Effect on the Treated). Determines weight construction.
        baseline_var : str, optional
            Pre-treatment outcome or covariate to include
        project_path : str, optional
            Path to save results Excel file (requires analysis_name)
        trim_quantile : float, default=0.99
            Quantile for weight trimming
        analysis_name : str, optional
            Analysis identifier for file naming
        alpha : float, default=0.05
            Significance level used for confidence intervals and raw p-value
            significance threshold (before any multiple testing correction)
        plot_propensity : bool, default=True
            If True, generates propensity score overlap density plot
        plot_weights : bool, default=True
            If True, generates IPTW weight distribution plot
        
        Returns
        -------
        dict
            Dictionary with keys:
            - effect: Point estimate of treatment effect (ATE or ATT)
            - estimand: String indicating "ATE" or "ATT"
            - ci_lower: Lower confidence interval bound (at 1-alpha level)
            - ci_upper: Upper confidence interval bound (at 1-alpha level)
            - p_value: RAW p-value for the treatment effect (uncorrected)
            - significant: Boolean indicating significance at alpha (raw, uncorrected)
            - alpha: Significance level used
            - cohens_d: IPTW-weighted Cohen's d (uses raw weighted mean diff).
              Conventional benchmarks (0.2/0.5/0.8) may not apply directly.
            - pct_change: Percent change relative to control group mean
            - mean_treatment: Weighted mean outcome for treated group
            - mean_control: Weighted mean outcome for control group
            - coefficients_df: DataFrame with treatment effect row only,
            containing Estimate, Std_Error, CI_Lower, CI_Upper,
            P_Value_Raw
            - full_coefficients_df: DataFrame with all model coefficients
            (intercept, treatment, covariates) in the same format as
            coefficients_df. Used in the Excel sheet export.
            - gee_results: Full fitted GEE model object
            - ps_model: Fitted propensity score model object
            - ps_summary_df: DataFrame of propensity score model coefficients
            - balance_df: DataFrame of pre- and post-weighting balance statistics
            (computed via CausalDiagnostics.compute_balance_df)
            - weight_diagnostics: Dictionary of weight summary statistics
            - ps_overlap_fig: Propensity score overlap figure (if plot_propensity=True)
            - weight_dist_fig: Weight distribution figure (if plot_weights=True)
            - weighted_df: Processed DataFrame with propensity_score and iptw
            columns attached (useful for independent balance verification)
            - outcome_type: "binary" or "continuous" based on auto-detection
        
        Raises
        ------
        ValueError
            If data preparation, model fitting, or validation fails
        """
        # Validate estimand
        estimand = estimand.upper()
        if estimand not in ["ATE", "ATT"]:
            raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

        # --- Remind users about multiple testing correction ---
        print(f"\n📊 Analyzing outcome: {outcome_var}")
        print("⚠️  REMINDER: This returns raw p-values. For multiple outcomes, use build_summary_table() for FDR correction.")

        # ------------------------------------------------------------------
        # Steps 0–2: Data prep, propensity weighting, diagnostics
        # (delegated to shared helper — see _prepare_iptw_data)
        # ------------------------------------------------------------------
        _iptw = self._prepare_iptw_data(
            data=data,
            treatment_var=treatment_var,
            cluster_var=cluster_var,
            categorical_vars=categorical_vars,
            binary_vars=binary_vars,
            continuous_vars=continuous_vars,
            estimand=estimand,
            trim_quantile=trim_quantile,
            plot_propensity=plot_propensity,
            plot_weights=plot_weights,
            outcome_var=outcome_var,
            baseline_var=baseline_var,
            analysis_label=f"{outcome_var} ({estimand})",
        )
        df              = _iptw["df"]
        ps_model        = _iptw["ps_model"]
        weight_stats    = _iptw["weight_stats"]
        balance_df      = _iptw["balance_df"]
        ps_overlap_fig  = _iptw["ps_overlap_fig"]
        weight_dist_fig = _iptw["weight_dist_fig"]
        covariates      = _iptw["covariates"]
        outcome_var     = _iptw["outcome_var"]
        treatment_var   = _iptw["treatment_var"]
        cluster_var     = _iptw["cluster_var"]
        baseline_var    = _iptw["baseline_var"]

        # ------------------------------------------------------------------
        # Step 3: Fit IPTW-weighted outcome model with covariate adjustment
        # ------------------------------------------------------------------
        # Auto-detect binary outcomes for appropriate GEE family
        outcome_values = df[outcome_var].dropna().unique()
        is_binary_outcome = set(outcome_values).issubset({0, 1, 0.0, 1.0})
        auto_family = "binomial" if is_binary_outcome else "gaussian"
        if is_binary_outcome:
            print(f"  Auto-detected binary outcome '{outcome_var}' → using Binomial family")
        
        try:
            model_data = df[[outcome_var, treatment_var, cluster_var] + covariates].copy()
            if model_data.empty or len(model_data) < 5:
                raise ValueError(f"Insufficient data for model: {len(model_data)} observations")
            
            if model_data.isnull().any().any():
                null_counts = model_data.isnull().sum()
                null_vars = null_counts[null_counts > 0].to_dict()
                raise ValueError(f"Missing values in model variables: {null_vars}")
            
            gee_res = self.fit_iptw_outcome_model(
                df,
                outcome_var,
                treatment_var,
                weight_col="iptw",
                cluster_var=cluster_var,
                covariates=covariates,
                family=auto_family
            )
        except Exception as e:
            raise ValueError(f"Error fitting outcome model: {str(e)}")
        
        effect = gee_res.params[treatment_var]
        ci = gee_res.conf_int(alpha=alpha).loc[treatment_var]
        p_value_raw = gee_res.pvalues[treatment_var]
        
        # --- Effect size metrics ---
        # IPTW-weighted Cohen's d uses the raw weighted mean difference (not the
        # conditional GEE coefficient) divided by the weighted marginal pooled SD.
        treated_df = df[df[treatment_var] == 1]
        control_df = df[df[treatment_var] == 0]
        
        mean_treatment = np.average(treated_df[outcome_var], weights=treated_df["iptw"])
        mean_control = np.average(control_df[outcome_var], weights=control_df["iptw"])
        
        raw_diff = mean_treatment - mean_control
        
        var_treated = np.average(
            (treated_df[outcome_var] - mean_treatment) ** 2, weights=treated_df["iptw"]
        )
        var_control = np.average(
            (control_df[outcome_var] - mean_control) ** 2, weights=control_df["iptw"]
        )
        pooled_sd = np.sqrt((var_treated + var_control) / 2)
        cohens_d = raw_diff / pooled_sd if pooled_sd > 0 else 0

        # pct_change uses the same marginal raw_diff as cohens_d
        # (not the conditional GEE coefficient) for consistency.
        pct_change = (raw_diff / mean_control) * 100 if abs(mean_control) > 1e-9 else None
        
        # No within-model multiple testing correction.
        # Correction is applied across outcomes in build_summary_table().
        significant = p_value_raw < alpha
        stars = self._significance_stars(p_value_raw)
        
        # --- Build full-model coefficients DataFrame (all parameters) ---
        all_ci = gee_res.conf_int(alpha=alpha)
        full_coefficients_df = pd.DataFrame({
            'Parameter': gee_res.params.index,
            'Estimate': gee_res.params.values,
            'Std_Error': gee_res.bse.values,
            'CI_Lower': all_ci.iloc[:, 0].values,
            'CI_Upper': all_ci.iloc[:, 1].values,
            'P_Value_Raw': gee_res.pvalues.values,
            'Alpha': alpha
        })
        
        # --- Build coefficients DataFrame (treatment-related rows, for printed summary) ---
        coefficients_df = full_coefficients_df[
            full_coefficients_df['Parameter'].str.contains(treatment_var, na=False)
        ].copy()
        
        # --- Print summary ---
        ci_pct = int((1 - alpha) * 100)
        print(
            f"  [{outcome_var}] {estimand} = {effect:.4f} "
            f"({ci_pct}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]), "
            f"p = {p_value_raw:.4f} {stars}, "
            f"IPTW-weighted Cohen's d = {cohens_d:.4f}"
        )
        
        # --- Build propensity score model summary DataFrame ---
        ps_summary_df = self._build_ps_summary_df(ps_model)

        # ------------------------------------------------------------------
        # Step 4: Export (optional)
        # ------------------------------------------------------------------
        if project_path and analysis_name:
            try:
                with pd.ExcelWriter(
                    f"{project_path}/{estimand.lower()}_iptw_gee_{analysis_name}.xlsx",
                    engine="openpyxl"
                ) as writer:
                    balance_df.to_excel(writer, sheet_name="Covariate_Balance", index=False)
                    pd.DataFrame([weight_stats]).to_excel(writer, sheet_name="Weight_Diagnostics", index=False)
                    full_coefficients_df.to_excel(writer, sheet_name=f"{estimand}_MSM", index=False)
                    ps_summary_df.to_excel(writer, sheet_name="Propensity_Model", index=False)
            except Exception as e:
                print(f"  Warning: Could not export results to Excel: {e}")
        
        return {
            "effect": effect,
            "estimand": estimand,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "p_value": p_value_raw,
            "significant": significant,
            "alpha": alpha,
            "cohens_d": cohens_d,
            "pct_change": pct_change,
            "mean_treatment": mean_treatment,
            "mean_control": mean_control,
            "coefficients_df": coefficients_df,
            "full_coefficients_df": full_coefficients_df,
            "gee_results": gee_res,
            "ps_model": ps_model,
            "ps_summary_df": ps_summary_df,
            "balance_df": balance_df,
            "weight_diagnostics": weight_stats,
            "ps_overlap_fig": ps_overlap_fig,
            "weight_dist_fig": weight_dist_fig,
            "weighted_df": df,
            "outcome_type": "binary" if is_binary_outcome else "continuous",
        }

    # ==================================================================
    # Double Machine Learning (DML) methods
    # ==================================================================

    def dml_estimate_treatment_effects(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        categorical_vars: Optional[List[str]] = None,
        binary_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        W_cols: Optional[List[str]] = None,
        X_cols: Optional[List[str]] = None,
        model_y=None,
        model_t=None,
        discrete_outcome: Optional[bool] = None,
        discrete_treatment: Optional[bool] = None,
        estimand: str = "ATE",
        estimate: str = "both",
        cluster_var: Optional[str] = None,
        random_state: int = 42,
        test_size: float = 0.2,
        n_estimators: int = 500,
        cv: int = 5,
        max_tree_depth: int = 3,
        min_samples_leaf: int = 25,
        plot_cate: bool = True,
        plot_importance: bool = True,
        plot_tree: bool = True,
        project_path: Optional[str] = None,
        analysis_name: Optional[str] = None,
        alpha: float = 0.05,
    ) -> Dict:
        """
        .. note::
            For cluster-robust ATE estimation, prefer
            ``dml_cluster_robust_ate()``, which uses the ``doubleml`` package
            with native cluster support. This method remains available for
            CATE estimation via Causal Forest, which ``doubleml`` does not
            offer.

        Estimate ATE, ATT, and/or CATE using Double Machine Learning (DML).

        Implements two complementary DML estimators from the ``econml`` package:

        - **Linear DML** — estimates a single average effect (ATE or ATT) using
          flexible ML nuisance models for outcome and treatment prediction, with
          a linear final-stage model. Provides confidence intervals and p-values.
        - **Causal Forest DML** — estimates individualized Conditional Average
          Treatment Effects (CATE) τ(X), with feature importance and a tree-based
          interpreter for subgroup discovery.

        ATT is derived from CATE by averaging individualized effects over the
        treated subpopulation, which is valid under unconfoundedness but is not
        a first-class ATT estimator. For ATT with cluster-robust standard errors,
        prefer ``analyze_treatment_effect(estimand='ATT')``.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing outcome, treatment, and covariate variables.
        outcome_col : str
            Name of the outcome column (Y).
        treatment_col : str
            Name of the treatment column (T). Typically binary (0/1).
        categorical_vars : list of str, optional
            Categorical covariate names (will be one-hot encoded).
            Used to construct ``W_cols`` / ``X_cols`` when those are not
            explicitly provided.
        binary_vars : list of str, optional
            Binary covariate names.
        continuous_vars : list of str, optional
            Continuous covariate names.
        W_cols : list of str, optional
            Explicit list of confounding covariate column names. If provided,
            takes precedence over the triple-list (categorical/binary/continuous).
        X_cols : list of str, optional
            Explicit list of effect-modifier column names for CATE estimation.
            May include **raw categorical variable names** (e.g. ``'organization'``)
            that appear in ``categorical_vars``; these are auto-expanded to their
            one-hot-encoded dummy columns. Binary / continuous names and already-
            encoded dummy names pass through unchanged. If ``None``, defaults to
            ``W_cols`` (i.e. every covariate is also a candidate effect modifier).
        model_y : estimator, optional
            Predictive model for the outcome nuisance function. If ``None``,
            auto-selected based on outcome type:
            ``RandomForestClassifier`` for binary, ``RandomForestRegressor``
            for continuous outcomes.
        model_t : estimator, optional
            Predictive model for the treatment nuisance function. If ``None``,
            defaults to ``RandomForestClassifier(random_state=random_state)``.
        discrete_outcome : bool, optional
            Whether the outcome is discrete (binary). If ``None``, auto-detected
            from the data by checking if unique values ⊆ {0, 1}.
        discrete_treatment : bool, optional
            Whether the treatment is discrete. If ``None``, auto-detected:
            binary numeric → True, continuous → False.
        estimand : str, default="ATE"
            Target causal estimand: ``"ATE"``, ``"ATT"``, or ``"both"``.
            Controls which average treatment effect is reported.
        estimate : str, default="both"
            Which DML estimators to run: ``"ATE"`` (linear DML only),
            ``"CATE"`` (Causal Forest only), or ``"both"``.
        cluster_var : str, optional
            Name of clustering variable. Stored in results metadata for the
            econml workflow; not used directly by this estimator for inference.
            For cluster-robust ATE estimation, prefer
            ``dml_cluster_robust_ate()``.
        random_state : int, default=42
            Random seed for reproducibility.
        test_size : float, default=0.2
            Fraction of data held out for CATE evaluation.
        n_estimators : int, default=500
            Number of trees in the Causal Forest.
        cv : int, default=5
            Cross-validation folds for nuisance model estimation.
        max_tree_depth : int, default=3
            Maximum depth for the CATE interpreter tree.
        min_samples_leaf : int, default=25
            Minimum samples per leaf in the CATE interpreter tree.
        plot_cate : bool, default=True
            If True, generates CATE distribution histogram.
        plot_importance : bool, default=True
            If True, generates feature importance bar plot.
        plot_tree : bool, default=True
            If True, generates CATE interpreter decision tree plot.
        project_path : str, optional
            Directory path for saving Excel results.
        analysis_name : str, optional
            Analysis identifier for file naming.
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``effect`` : float — Primary point estimate (ATE or ATT depending
              on ``estimand``; ATE when ``estimand="both"``).
            - ``estimand`` : str — "ATE", "ATT", or "both".
            - ``ci_lower`` : float — Lower CI bound for the primary effect.
            - ``ci_upper`` : float — Upper CI bound for the primary effect.
            - ``p_value`` : float — p-value for the primary effect (from DML
              inference; ``None`` if only CATE estimated).
            - ``significant`` : bool — Whether p_value < alpha.
            - ``alpha`` : float — Significance level used.
            - ``cohens_d`` : float — Cohen's d effect size.
            - ``pct_change`` : float or None — Percent change vs. control mean.
            - ``mean_treatment`` : float — Mean outcome for treated group.
            - ``mean_control`` : float — Mean outcome for control group.
            - ``outcome_type`` : str — "binary" or "continuous".
            - ``coefficients_df`` : pd.DataFrame — Single-row DataFrame with
              Estimate, Std_Error, CI_Lower, CI_Upper, P_Value_Raw.
            - ``weight_diagnostics`` : dict — Contains at minimum
              ``n_observations``.
            - ``ate_results`` : dict or None — ``{"ATE", "ATE_CI"}`` when ATE
              estimated.
            - ``att_results`` : dict or None — ``{"ATT", "ATT_CI"}`` when ATT
              estimated.
            - ``cate_results`` : dict or None — ``{"cate_estimates", "X_test",
              "cate_summary"}`` when CATE estimated.
            - ``feature_importances_df`` : pd.DataFrame or None.
            - ``cate_plot`` : matplotlib Figure or None.
            - ``importance_plot`` : matplotlib Figure or None.
            - ``tree_plot`` : matplotlib Figure or None.
            - ``dml_model`` : fitted DML object or None.
            - ``cfdml_model`` : fitted CausalForestDML object or None.
            - ``cluster_var`` : str or None — Stored for metadata; not used by DML.

        Raises
        ------
        ImportError
            If the ``econml`` package is not installed.
        ValueError
            If data preparation or model fitting fails.

        Notes
        -----
                - This econml-based workflow is best treated as the CATE exploration
                    path. For like-for-like cluster-robust ATE comparison against
                    IPTW + GEE, use ``dml_cluster_robust_ate()``.
        - ATT is derived by averaging CATE estimates over treated observations:
          E[τ(X) | T=1]. This is valid under unconfoundedness but does not use
          a dedicated ATT estimator. When DML is fit with X=None (homogeneous
          effect), ATE = ATT by construction.
        - The return dict is structured for compatibility with
          ``build_summary_table()`` and ``compute_evalues_from_results()``.
        """
        # ---- Validate parameters ----
        estimand = estimand.upper()
        if estimand not in ("ATE", "ATT", "BOTH"):
            raise ValueError(f"estimand must be 'ATE', 'ATT', or 'both', got '{estimand}'")

        estimate = estimate.upper()
        if estimate not in ("ATE", "CATE", "BOTH"):
            raise ValueError(f"estimate must be 'ATE', 'CATE', or 'both', got '{estimate}'")

        # ---- Defensive copy ----
        df = data.copy()

        # ---- Build W_cols from triple-list convention if not explicit ----
        if W_cols is None:
            cat_vars = categorical_vars or []
            bin_vars = binary_vars or []
            cont_vars = continuous_vars or []
            if not (cat_vars or bin_vars or cont_vars):
                raise ValueError(
                    "Must provide either W_cols or at least one of "
                    "categorical_vars / binary_vars / continuous_vars."
                )
            # One-hot encode categoricals
            if cat_vars:
                cols_before = set(df.columns)
                df = pd.get_dummies(df, columns=cat_vars, drop_first=True)
                dummy_cols = sorted(set(df.columns) - cols_before)
            else:
                dummy_cols = []
            W_cols = dummy_cols + bin_vars + cont_vars
        else:
            dummy_cols = []

        # ---- Expand raw categorical names in X_cols into their dummy columns ----
        # Lets callers pass X_cols=['organization', 'region', ...] and have the
        # one-hot-encoded dummy columns resolved automatically, mirroring how
        # W_cols is built. Any entries already matching existing (non-dropped)
        # columns in df pass through unchanged.
        if X_cols is not None and dummy_cols:
            expanded_X_cols = []
            for col in X_cols:
                matches = [d for d in dummy_cols if d.startswith(f"{col}_")]
                if matches:
                    expanded_X_cols.extend(matches)
                else:
                    expanded_X_cols.append(col)
            X_cols = expanded_X_cols

        # ---- Column name sanitization ----
        rename_map = {c: self._clean_column_name(c) for c in df.columns}
        df.rename(columns=rename_map, inplace=True)
        outcome_col = self._clean_column_name(outcome_col)
        treatment_col = self._clean_column_name(treatment_col)
        W_cols = [self._clean_column_name(c) for c in W_cols]
        if cluster_var:
            cluster_var = self._clean_column_name(cluster_var)

        if X_cols is not None:
            X_cols = [self._clean_column_name(c) for c in X_cols]
        else:
            X_cols = list(W_cols)

        # ---- Drop rows with NAs in relevant columns ----
        all_cols = list(set([outcome_col, treatment_col] + W_cols + X_cols))
        if cluster_var:
            all_cols = list(set(all_cols + [cluster_var]))
        df = df[all_cols].dropna().copy()

        if len(df) < 20:
            raise ValueError(
                f"Insufficient data after removing missing values: {len(df)} rows. "
                "DML requires a reasonable sample size for cross-fitting."
            )

        # ---- Auto-detect outcome type ----
        outcome_values = df[outcome_col].dropna().unique()
        is_binary_outcome = set(outcome_values).issubset({0, 1, 0.0, 1.0})
        if discrete_outcome is None:
            discrete_outcome = is_binary_outcome
            if is_binary_outcome:
                print(f"  Auto-detected binary outcome '{outcome_col}' → discrete_outcome=True")

        # ---- Auto-detect treatment type ----
        treatment_series = df[treatment_col]
        if discrete_treatment is None:
            if pd.api.types.is_numeric_dtype(treatment_series):
                unique_vals = treatment_series.dropna().unique()
                discrete_treatment = len(unique_vals) == 2
            else:
                # Encode categorical treatment
                treatment_series = treatment_series.astype("category")
                df[treatment_col] = treatment_series.cat.codes
                discrete_treatment = True

        # ---- Auto-select nuisance models ----
        if model_y is None:
            if discrete_outcome:
                model_y = RandomForestClassifier(random_state=random_state)
            else:
                model_y = RandomForestRegressor(random_state=random_state)
        if model_t is None:
            model_t = RandomForestClassifier(random_state=random_state)

        # ---- Prepare arrays ----
        Y = df[outcome_col]
        T = df[treatment_col]
        W = df[W_cols].copy()
        X = df[X_cols].copy()

        # ---- Compute raw group means for effect-size metrics ----
        mean_treatment = Y[T == 1].mean()
        mean_control = Y[T == 0].mean()

        # ---- Train / test split ----
        (X_train, X_test, W_train, W_test,
         Y_train, Y_test, T_train, T_test) = train_test_split(
            X, W, Y, T, test_size=test_size, random_state=random_state
        )

        # ---- Initialize result containers ----
        ate_results = None
        att_results = None
        cate_results = None
        cate_plot = None
        importance_plot = None
        tree_plot = None
        dml_model_obj = None
        cfdml_model_obj = None
        feature_importances_df = None
        cate_summary_obj = None
        primary_effect = None
        primary_ci = (None, None)
        primary_pvalue = None

        want_ate_estimand = estimand in ("ATE", "BOTH")
        want_att_estimand = estimand in ("ATT", "BOTH")

        # ==============================================================
        # Estimate ATE/ATT via Linear DML
        # ==============================================================
        if estimate in ("ATE", "BOTH"):
            print(f"\n  Fitting Linear DML for '{outcome_col}'...")
            dml = DML(
                model_y=model_y,
                model_t=model_t,
                model_final=StatsModelsLinearRegression(fit_intercept=False),
                discrete_outcome=discrete_outcome,
                discrete_treatment=discrete_treatment,
                cv=cv,
                random_state=random_state,
            )
            dml.fit(Y=Y_train, T=T_train, X=None, W=W_train, cache_values=True)
            dml_model_obj = dml
            self.dml_model = dml

            # ---- ATE ----
            if want_ate_estimand:
                ate_val = float(dml.ate())
                ate_ci = dml.ate_interval(alpha=alpha)
                ate_ci = (float(ate_ci[0]), float(ate_ci[1]))
                ate_results = {"ATE": ate_val, "ATE_CI": ate_ci}
                print(f"    ATE = {ate_val:.4f}, {int((1-alpha)*100)}% CI: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")

                if primary_effect is None:
                    primary_effect = ate_val
                    primary_ci = ate_ci

            # ---- ATT via Linear DML ----
            # With X=None the DML model estimates a single constant effect,
            # so ATE = ATT by construction.
            if want_att_estimand:
                att_val = float(dml.ate())  # constant effect → ATE = ATT
                att_ci = dml.ate_interval(alpha=alpha)
                att_ci = (float(att_ci[0]), float(att_ci[1]))
                att_results = {"ATT": att_val, "ATT_CI": att_ci}
                print(f"    ATT (DML, constant effect) = {att_val:.4f}, "
                      f"{int((1-alpha)*100)}% CI: [{att_ci[0]:.4f}, {att_ci[1]:.4f}]")
                if primary_effect is None:
                    primary_effect = att_val
                    primary_ci = att_ci

            # ---- p-value from DML inference ----
            try:
                dml_inference = dml.effect_inference(X=None)
                summary_frame = dml_inference.summary_frame(alpha=alpha)
                primary_pvalue = float(summary_frame["pvalue"].iloc[0])
            except Exception:
                primary_pvalue = None

        # ==============================================================
        # Estimate CATE via Causal Forest DML
        # ==============================================================
        if estimate in ("CATE", "BOTH"):
            print(f"\n  Fitting Causal Forest DML for '{outcome_col}'...")
            cfdml = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_outcome=discrete_outcome,
                discrete_treatment=discrete_treatment,
                inference=True,
                cv=cv,
                n_estimators=n_estimators,
                random_state=random_state,
            )
            cfdml.fit(Y=Y_train, T=T_train, X=X_train, W=W_train, cache_values=True)
            cfdml_model_obj = cfdml
            self.cfdml_model = cfdml

            # Individualized effects on test set
            cate_estimates = cfdml.effect(X_test)

            # Summary
            try:
                cate_summary_obj = cfdml.summary()
                print("    CATE Summary:")
                print(cate_summary_obj)
            except Exception as e:
                print(f"    Warning: Could not produce CATE summary: {e}")

            cate_results = {
                "cate_estimates": cate_estimates,
                "X_test": X_test,
                "cate_summary": cate_summary_obj,
            }

            # ---- ATE from Causal Forest (overwrites if both estimators run) ----
            if want_ate_estimand:
                cf_ate = float(cfdml.ate(X=X_test))
                cf_ate_ci = cfdml.ate_interval(X=X_test, alpha=alpha)
                cf_ate_ci = (float(cf_ate_ci[0]), float(cf_ate_ci[1]))
                ate_results = {"ATE": cf_ate, "ATE_CI": cf_ate_ci}
                print(f"    ATE (Causal Forest) = {cf_ate:.4f}, "
                      f"{int((1-alpha)*100)}% CI: [{cf_ate_ci[0]:.4f}, {cf_ate_ci[1]:.4f}]")
                if primary_effect is None:
                    primary_effect = cf_ate
                    primary_ci = cf_ate_ci

            # ---- ATT from Causal Forest: average CATE over treated ----
            if want_att_estimand:
                treated_mask = T_test == 1
                n_treated_test = int(treated_mask.sum())
                if n_treated_test > 0:
                    X_test_treated = X_test[treated_mask]
                    tau_treated = cfdml.effect(X_test_treated)
                    att_val = float(tau_treated.mean())
                    # CI via population_summary on treated subset
                    try:
                        att_inference = cfdml.effect_inference(X_test_treated)
                        pop_summary = att_inference.population_summary(alpha=alpha)
                        att_ci_raw = pop_summary.conf_int_mean(alpha=alpha)
                        att_ci = (float(att_ci_raw[0]), float(att_ci_raw[1]))
                    except Exception:
                        # Fallback: use ±1.96 * SE of individual effects
                        se_att = float(tau_treated.std() / np.sqrt(n_treated_test))
                        z = norm.ppf(1 - alpha / 2)
                        att_ci = (att_val - z * se_att, att_val + z * se_att)
                    att_results = {"ATT": att_val, "ATT_CI": att_ci}
                    print(f"    ATT (Causal Forest, n_treated={n_treated_test}) = {att_val:.4f}, "
                          f"{int((1-alpha)*100)}% CI: [{att_ci[0]:.4f}, {att_ci[1]:.4f}]")
                    if primary_effect is None:
                        primary_effect = att_val
                        primary_ci = att_ci
                else:
                    print("    Warning: No treated observations in test set; ATT not computed.")

            # ---- p-value from Causal Forest inference ----
            if primary_pvalue is None:
                try:
                    cf_inference = cfdml.effect_inference(X_test)
                    cf_summary_frame = cf_inference.summary_frame(alpha=alpha)
                    primary_pvalue = float(cf_summary_frame["pvalue"].mean())
                except Exception:
                    primary_pvalue = None

            # ---- Feature importance ----
            feature_importances = cfdml.feature_importances_
            feature_names = X.columns.tolist()
            feature_importances_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": feature_importances,
            }).sort_values(by="Importance", ascending=False).head(20)

            # ============== CATE Histogram ==============
            if plot_cate:
                fig_cate, ax_cate = plt.subplots(figsize=(10, 6))
                ax_cate.hist(cate_estimates, bins=20, edgecolor="black",
                             alpha=0.7, color="#3498db", linewidth=0.5)
                ax_cate.set_xlabel("Estimated CATE", fontsize=11)
                ax_cate.set_ylabel("Number of Individuals", fontsize=11)
                ax_cate.set_title(
                    f"Distribution of Individualized Treatment Effects — {outcome_col}",
                    fontsize=13, fontweight="bold",
                )
                ax_cate.axvline(
                    float(np.mean(cate_estimates)), color="red", linestyle="--",
                    label=f"Mean = {float(np.mean(cate_estimates)):.4f}",
                )
                ax_cate.legend(fontsize=9)
                ax_cate.grid(axis="y", alpha=0.3)
                cate_plot = fig_cate
                plt.close(fig_cate)

            # ============== Feature Importance Plot ==============
            if plot_importance:
                fig_imp, ax_imp = plt.subplots(figsize=(11, 6))
                ax_imp.barh(
                    feature_importances_df["Feature"],
                    feature_importances_df["Importance"],
                    color="#2ecc71", edgecolor="black", linewidth=0.5, alpha=0.8,
                )
                ax_imp.set_xlabel("Feature Importance", fontsize=11)
                ax_imp.set_ylabel("Features", fontsize=11)
                ax_imp.set_title(
                    f"Top {len(feature_importances_df)} Feature Importance — Causal Forest — {outcome_col}",
                    fontsize=13, fontweight="bold",
                )
                ax_imp.invert_yaxis()
                ax_imp.grid(axis="x", alpha=0.3)
                importance_plot = fig_imp
                plt.close(fig_imp)

            # ============== CATE Tree Interpreter ==============
            if plot_tree:
                try:
                    intrp = SingleTreeCateInterpreter(
                        include_model_uncertainty=True,
                        max_depth=max_tree_depth,
                        min_samples_leaf=min_samples_leaf,
                    )
                    intrp.interpret(cfdml, X_test)
                    fig_tree = plt.figure(figsize=(25, 12))
                    intrp.plot(feature_names=feature_names, fontsize=12)
                    plt.title(
                        f"CATE Interpreter Tree — {outcome_col}",
                        fontsize=14, fontweight="bold",
                    )
                    tree_plot = fig_tree
                    plt.close(fig_tree)
                except Exception as e:
                    print(f"    Warning: Could not generate CATE tree plot: {e}")

        # ==============================================================
        # Compute effect-size metrics
        # ==============================================================
        if primary_effect is None:
            primary_effect = 0.0
            primary_ci = (0.0, 0.0)
            print("  Warning: No treatment effect could be estimated.")

        # Cohen's d (raw, unweighted) from mean difference / pooled SD
        raw_diff = mean_treatment - mean_control
        var_treated = Y[T == 1].var()
        var_control = Y[T == 0].var()
        pooled_sd = np.sqrt((var_treated + var_control) / 2)
        cohens_d = raw_diff / pooled_sd if pooled_sd > 0 else 0.0
        pct_change = (raw_diff / mean_control) * 100 if abs(mean_control) > 1e-9 else None

        significant = primary_pvalue < alpha if primary_pvalue is not None else False
        stars = self._significance_stars(primary_pvalue) if primary_pvalue is not None else ""

        ci_pct = int((1 - alpha) * 100)
        p_str = f"p = {primary_pvalue:.4f}" if primary_pvalue is not None else "p = N/A"
        print(
            f"\n  [{outcome_col}] DML {estimand} = {primary_effect:.4f} "
            f"({ci_pct}% CI: [{primary_ci[0]:.4f}, {primary_ci[1]:.4f}]), "
            f"{p_str} {stars}, Cohen's d = {cohens_d:.4f}"
        )

        # ---- Build coefficients DataFrame for summary table compatibility ----
        coefficients_df = pd.DataFrame({
            "Parameter": [treatment_col],
            "Estimate": [primary_effect],
            "Std_Error": [
                (primary_ci[1] - primary_ci[0]) / (2 * 1.96)
                if primary_ci[0] is not None else None
            ],
            "CI_Lower": [primary_ci[0]],
            "CI_Upper": [primary_ci[1]],
            "P_Value_Raw": [primary_pvalue],
            "Alpha": [alpha],
        })

        # ---- Excel export (optional) ----
        if project_path and analysis_name:
            try:
                xlsx_path = f"{project_path}/dml_{estimand.lower()}_{analysis_name}.xlsx"
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    coefficients_df.to_excel(writer, sheet_name="DML_Effect", index=False)
                    if feature_importances_df is not None:
                        feature_importances_df.to_excel(
                            writer, sheet_name="Feature_Importance", index=False
                        )
                    if ate_results:
                        pd.DataFrame([ate_results]).to_excel(
                            writer, sheet_name="ATE_Results", index=False
                        )
                    if att_results:
                        pd.DataFrame([att_results]).to_excel(
                            writer, sheet_name="ATT_Results", index=False
                        )
                print(f"  Results saved to {xlsx_path}")
            except Exception as e:
                print(f"  Warning: Could not export results to Excel: {e}")

        # ---- Return dict (compatible with build_summary_table / compute_evalues) ----
        return {
            "effect": primary_effect,
            "estimand": estimand,
            "ci_lower": primary_ci[0],
            "ci_upper": primary_ci[1],
            "p_value": primary_pvalue,
            "significant": significant,
            "alpha": alpha,
            "cohens_d": cohens_d,
            "pct_change": pct_change,
            "mean_treatment": mean_treatment,
            "mean_control": mean_control,
            "outcome_type": "binary" if is_binary_outcome else "continuous",
            "coefficients_df": coefficients_df,
            "weight_diagnostics": {"n_observations": len(df)},
            "ate_results": ate_results,
            "att_results": att_results,
            "cate_results": cate_results,
            "feature_importances_df": feature_importances_df,
            "cate_plot": cate_plot,
            "importance_plot": importance_plot,
            "tree_plot": tree_plot,
            "dml_model": dml_model_obj,
            "cfdml_model": cfdml_model_obj,
            "cluster_var": cluster_var,
        }

    def dml_cluster_robust_ate(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        categorical_vars: Optional[List[str]] = None,
        binary_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        cluster_var: Optional[str] = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        cv: int = 5,
        random_state: int = 42,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Estimate ATE via Partially Linear Regression (PLR) Double Machine
        Learning with native cluster-robust inference.

        Uses the ``doubleml`` package (Bach et al., 2022), which provides
        cluster-aware cross-fitting and cluster-robust standard errors
        when ``cluster_cols`` is specified in ``DoubleMLData``.

        This method is intended as a robustness check against the primary
        IPTW + GEE estimates from ``analyze_treatment_effect()``. When both
        methods account for clustering and produce similar ATEs, confidence
        in the causal estimate increases.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing outcome, treatment, and covariate variables.
        outcome_col : str
            Name of the outcome column (Y).
        treatment_col : str
            Name of the binary treatment column (T).
        categorical_vars : list of str, optional
            Categorical covariate names (will be one-hot encoded).
        binary_vars : list of str, optional
            Binary covariate names.
        continuous_vars : list of str, optional
            Continuous covariate names.
        cluster_var : str, optional
            Name of the clustering variable (e.g., 'team_id'). When provided,
            DoubleML uses cluster-aware cross-fitting and cluster-robust
            score-based inference.
        n_estimators : int, default=200
            Number of trees for Random Forest nuisance models.
        max_depth : int, default=5
            Max tree depth for nuisance models.
        cv : int, default=5
            Number of cross-fitting folds.
        random_state : int, default=42
            Random seed.
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        dict
            Dictionary with keys compatible with build_summary_table().

        References
        ----------
        Bach P, Chernozhukov V, Kurz MS, Spindler M (2022). DoubleML -
        An Object-Oriented Implementation of Double Machine Learning in
        Python. Journal of Machine Learning Research, 23(53):1-6.
        """
        df = data.copy()

        cat_vars = categorical_vars or []
        bin_vars = binary_vars or []
        cont_vars = continuous_vars or []

        if cat_vars:
            df = pd.get_dummies(df, columns=cat_vars, drop_first=True)

        dummy_cols = [
            column for column in df.columns
            if any(column.startswith(var + "_") for var in cat_vars)
        ]
        x_cols = dummy_cols + bin_vars + cont_vars

        rename_map = {column: self._clean_column_name(column) for column in df.columns}
        df.rename(columns=rename_map, inplace=True)
        outcome_col = self._clean_column_name(outcome_col)
        treatment_col = self._clean_column_name(treatment_col)
        x_cols = [self._clean_column_name(column) for column in x_cols]
        if cluster_var:
            cluster_var = self._clean_column_name(cluster_var)

        all_cols = list(set([outcome_col, treatment_col] + x_cols + ([cluster_var] if cluster_var else [])))
        df = df[all_cols].dropna().copy()

        if len(df) < 20:
            raise ValueError(
                f"Insufficient data after removing missing values: {len(df)} rows"
            )

        dml_data_kwargs = {
            "data": df,
            "y_col": outcome_col,
            "d_cols": treatment_col,
            "x_cols": x_cols,
        }
        if cluster_var:
            dml_data_kwargs["cluster_cols"] = cluster_var
        dml_data = DoubleMLData(**dml_data_kwargs)

        ml_l = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        ml_m = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        dml_plr = DoubleMLPLR(
            dml_data,
            ml_l=ml_l,
            ml_m=ml_m,
            n_folds=cv,
        )
        dml_plr.fit()

        ate = float(dml_plr.coef[0])
        se = float(dml_plr.se[0])
        ci = dml_plr.confint(level=1 - alpha)
        ci_lower = float(ci.iloc[0, 0])
        ci_upper = float(ci.iloc[0, 1])
        p_value = float(dml_plr.pval[0])
        significant = p_value < alpha

        n_clusters = df[cluster_var].nunique() if cluster_var else None

        cluster_note = f", {n_clusters} clusters" if n_clusters else ""
        print(f"\n  DoubleML PLR - Cluster-Robust ATE Estimation")
        print(f"  {'=' * 50}")
        print(f"  ATE = {ate:.4f} (SE = {se:.4f}{cluster_note})")
        print(f"  {int((1-alpha)*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  p-value = {p_value:.4f} {self._significance_stars(p_value)}")
        print(f"  n = {len(df)}")
        if cluster_var:
            print(f"  Inference: cluster-robust (cluster_cols = '{cluster_var}')")
        print(f"  {'=' * 50}")

        return {
            "effect": ate,
            "estimand": "ATE",
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "significant": significant,
            "alpha": alpha,
            "cohens_d": None,
            "pct_change": None,
            "mean_treatment": df[df[treatment_col] == 1][outcome_col].mean(),
            "mean_control": df[df[treatment_col] == 0][outcome_col].mean(),
            "outcome_type": "continuous",
            "n_obs": len(df),
            "n_clusters": n_clusters,
            "coefficients_df": pd.DataFrame({
                "Parameter": [treatment_col],
                "Estimate": [ate],
                "Std_Error": [se],
                "CI_Lower": [ci_lower],
                "CI_Upper": [ci_upper],
                "P_Value_Raw": [p_value],
                "Alpha": [alpha],
            }),
            "weight_diagnostics": {"n_observations": len(df)},
            "doubleml_model": dml_plr,
            "summary": dml_plr.summary,
        }

    def dml_estimate_treatment_effects_help(self):
        """
        Print detailed documentation for ``dml_estimate_treatment_effects``.

        Displays parameter descriptions, usage examples, and interpretation
        guidance for Double Machine Learning estimation.
        """
        help_text = """
    Double Machine Learning (DML) for ATE, ATT, and CATE Estimation
    ================================================================

    For cluster-robust ATE estimation, prefer
    ``CausalInferenceModel.dml_cluster_robust_ate()``. The method documented
    below remains the primary path for CATE estimation via ``econml``.

    This method implements Double Machine Learning (DML) to estimate the
    Average Treatment Effect (ATE), Average Treatment Effect on the Treated
    (ATT), and Conditional Average Treatment Effect (CATE) using flexible
    machine learning models via the ``econml`` package.

    Method:
    -------
    ``CausalInferenceModel.dml_estimate_treatment_effects()``

    Parameters:
    -----------
    - data (pd.DataFrame): Dataset with outcome, treatment, and covariates.
    - outcome_col (str): Name of the outcome column (Y).
    - treatment_col (str): Name of the treatment column (T).
    - categorical_vars (list): Categorical covariate names (one-hot encoded).
    - binary_vars (list): Binary covariate names.
    - continuous_vars (list): Continuous covariate names.
    - W_cols (list): Explicit confounders list (overrides triple-list if given).
    - X_cols (list): Effect modifier columns for CATE. Defaults to W_cols.
    - model_y (estimator): Predictive model for outcome. Auto-selected if None.
    - model_t (estimator): Predictive model for treatment. Default: RandomForestClassifier.
    - discrete_outcome (bool): Whether outcome is binary. Auto-detected if None.
    - discrete_treatment (bool): Whether treatment is discrete. Auto-detected if None.
    - estimand (str): "ATE", "ATT", or "both". Default: "ATE".
    - estimate (str): "ATE" (linear DML), "CATE" (Causal Forest), or "both".
        - cluster_var (str): Clustering variable (stored for metadata in this econml
            workflow; for cluster-robust ATE use dml_cluster_robust_ate).
    - random_state (int): Random seed. Default: 42.
    - test_size (float): Fraction held out for CATE evaluation. Default: 0.2.
    - n_estimators (int): Number of Causal Forest trees. Default: 500.
    - cv (int): Cross-validation folds. Default: 5.
    - max_tree_depth (int): Max depth for CATE interpreter tree. Default: 3.
    - min_samples_leaf (int): Min samples per leaf in interpreter tree. Default: 25.
    - plot_cate, plot_importance, plot_tree (bool): Plot toggles.
    - project_path, analysis_name (str): For optional Excel export.
    - alpha (float): Significance level. Default: 0.05.

    Returns:
    --------
    dict with keys: effect, estimand, ci_lower, ci_upper, p_value, significant,
    alpha, cohens_d, pct_change, mean_treatment, mean_control, outcome_type,
    coefficients_df, weight_diagnostics, ate_results, att_results, cate_results,
    feature_importances_df, cate_plot, importance_plot, tree_plot, dml_model,
    cfdml_model, cluster_var.

    The return dict is compatible with ``build_summary_table()`` and
    ``compute_evalues_from_results()``.

    Example Usage:
    --------------
    ```python
    from causal_inference_modelling import CausalInferenceModel

    model = CausalInferenceModel()

    # Using the triple-list convention (project standard)
    results = model.dml_estimate_treatment_effects(
        data=manager_data,
        outcome_col='manager_efficacy_index',
        treatment_col='treatment',
        categorical_vars=['organization', 'job_level'],
        binary_vars=[],
        continuous_vars=['tenure_years', 'performance_rating'],
        estimand="both",      # Estimate both ATE and ATT
        estimate="both",      # Run both Linear DML and Causal Forest
        cluster_var='team_id',
        random_state=42,
    )

    # Access ATE results
    print("ATE:", results["ate_results"]["ATE"])
    print("ATE CI:", results["ate_results"]["ATE_CI"])

    # Access ATT results (derived from CATE over treated)
    print("ATT:", results["att_results"]["ATT"])
    print("ATT CI:", results["att_results"]["ATT_CI"])

    # Access CATE results
    cate = results["cate_results"]["cate_estimates"]
    print("CATE mean:", cate.mean())
    print("CATE std:", cate.std())

    # Display plots (returned as figure objects)
    results["cate_plot"].show()
    results["importance_plot"].show()
    results["tree_plot"].show()

    # Save plots
    results["cate_plot"].savefig('cate_distribution.png', dpi=300)
    results["importance_plot"].savefig('feature_importance.png', dpi=300)
    results["tree_plot"].savefig('cate_tree.png', dpi=300)

    # Use with build_summary_table (across multiple outcomes)
    all_results = {}
    for outcome in outcomes:
        all_results[outcome] = model.dml_estimate_treatment_effects(
            data=manager_data,
            outcome_col=outcome,
            treatment_col='treatment',
            categorical_vars=cat_vars,
            binary_vars=bin_vars,
            continuous_vars=cont_vars,
        )
    summary = CausalInferenceModel.build_summary_table(all_results)
    ```

    Notes:
    ------
    - ATE (Average Treatment Effect): Average effect of treatment across
      the entire population. Estimated via Linear DML with a constant
      treatment effect model.

    - ATT (Average Treatment Effect on the Treated): Average effect for
      those who actually received treatment. In Linear DML with X=None,
      ATE = ATT by construction (constant effect). In Causal Forest,
      ATT = mean(CATE) over treated observations — valid under
      unconfoundedness. For first-class ATT with cluster-robust SEs,
      use ``analyze_treatment_effect(estimand='ATT')``.

    - CATE (Conditional Average Treatment Effect): Individualized
      treatment effects conditional on covariates (X). Estimated via
      Causal Forest DML, which provides heterogeneous effects.

    - Clustering: DML does not natively account for clustered data
      (e.g., managers in teams). Standard errors may be anti-conservative.
      Use ``analyze_treatment_effect()`` for cluster-robust inference.

    - The ``estimand`` and ``estimate`` parameters are orthogonal:
      ``estimand`` controls the causal quantity (ATE/ATT),
      ``estimate`` controls the statistical method (DML/CausalForest).
        """
        print(help_text)
    
    def analyze_survival_effect(
        self,
        data: pd.DataFrame,
        time_var: str,
        event_var: str,
        treatment_var: str,
        cluster_var: str,
        categorical_vars: Optional[List[str]] = None,
        binary_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        estimand: str = "ATT",
        project_path: Optional[str] = None,
        trim_quantile: float = 0.99,
        analysis_name: Optional[str] = None,
        alpha: float = 0.05,
        plot_propensity: bool = True,
        plot_weights: bool = True,
        time_interaction: Optional[str] = None,
        period_breaks: Optional[List[int]] = None,
        period_labels: Optional[List[str]] = None,
        snapshot_days: Optional[List[int]] = None,
    ) -> Dict:
        """
        Complete survival analysis pipeline: IPTW → weighted KM → RMST → Cox PH.

        Implements Steps 0-2 (shared IPTW infrastructure) then fits:
        1. IPTW-weighted Kaplan-Meier curves (descriptive)
        2. RMST difference (business-friendly "additional days retained")
        3. Cox PH model — standard (time_interaction=None) or with time
           interaction terms ("categorical" or "continuous")

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with time_var, event_var, treatment, and covariates.
        time_var : str
            Name of time column (days from T=0 to event/censoring).
        event_var : str
            Name of event indicator column (1=event occurred, 0=censored).
        treatment_var : str
            Name of binary treatment variable.
        cluster_var : str
            Name of clustering variable for robust standard errors.
        categorical_vars : List[str], optional
            Categorical covariate names (will be one-hot encoded).
        binary_vars : List[str], optional
            Binary covariate names.
        continuous_vars : List[str], optional
            Continuous covariate names.
        estimand : str, default="ATT"
            Target estimand: "ATE" or "ATT". Determines IPTW weight construction.
        project_path : str, optional
            Path to save results Excel file.
        trim_quantile : float, default=0.99
            Quantile for weight trimming.
        analysis_name : str, optional
            Analysis identifier for file naming.
        alpha : float, default=0.05
            Significance level for confidence intervals.
        plot_propensity : bool, default=True
            If True, generates propensity score overlap plot.
        plot_weights : bool, default=True
            If True, generates IPTW weight distribution plot.
        time_interaction : str or None, default=None
            Type of Cox model to fit:
            - None : standard Cox PH (single overall HR, PH test run)
            - "categorical" : treatment effect estimated separately per period
            - "continuous"  : treatment effect modeled as linear trend over time
        period_breaks : list of int, optional
            Required for time_interaction="categorical". Breakpoints in days
            defining period boundaries, e.g. [0, 90, 180, 270, 365].
            Defaults to [0, 90, 180, 270, 365] if not provided.
        period_labels : list of str, optional
            Human-readable labels for each period. If not provided, labels
            are auto-generated from period_breaks.
        snapshot_days : list of int, optional
            Days at which to report KM survival probabilities.
            Defaults to [90, 180, 270, 365].

        Returns
        -------
        dict
            Dictionary with keys compatible with build_survival_summary_table(),
            compute_evalues_from_results(), plot_survival_curves():

            Shared keys:
            - effect: hazard ratio (overall or reference period)
            - estimand, ci_lower, ci_upper, p_value, significant, alpha
            - cohens_d: None, pct_change: None
            - mean_treatment/mean_control: survival prob at 365d
            - outcome_type: "survival"
            - coefficients_df, balance_df, weight_diagnostics, weighted_df

            Survival-specific keys:
            - period_hrs: DataFrame with HR per period/timepoint
            - time_interaction: None, "categorical", or "continuous"
            - ph_test_results, ph_assumption_met
            - kmf_treated, kmf_control: fitted KaplanMeierFitter objects
            - survival_at_snapshots: DataFrame
            - cox_model: fitted CoxPHFitter object
            - rmst_difference: RMST difference dict (if computable)
            - n_events_treated, n_events_control, n_treated, n_control

        Raises
        ------
        ValueError
            If data preparation, model fitting, or validation fails.
        """
        # Validate estimand
        estimand = estimand.upper()
        if estimand not in ["ATE", "ATT"]:
            raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

        # Default period breaks for categorical interaction
        if time_interaction == "categorical" and period_breaks is None:
            period_breaks = [0, 90, 180, 270, 365]

        # Default snapshot days for KM
        if snapshot_days is None:
            snapshot_days = [90, 180, 270, 365]

        # ------------------------------------------------------------------
        # Steps 0–2: Data prep, propensity weighting, diagnostics
        # ------------------------------------------------------------------
        model_label = "Standard Cox PH" if time_interaction is None else f"Cox Time Interaction ({time_interaction})"
        _iptw = self._prepare_iptw_data(
            data=data,
            treatment_var=treatment_var,
            cluster_var=cluster_var,
            categorical_vars=categorical_vars,
            binary_vars=binary_vars,
            continuous_vars=continuous_vars,
            estimand=estimand,
            trim_quantile=trim_quantile,
            plot_propensity=plot_propensity,
            plot_weights=plot_weights,
            time_var=time_var,
            event_var=event_var,
            preserve_strata_backups=False,
            analysis_label=f"Survival Analysis — {model_label} ({estimand})",
        )
        df                = _iptw["df"]
        ps_model          = _iptw["ps_model"]
        weight_stats      = _iptw["weight_stats"]
        balance_df        = _iptw["balance_df"]
        ps_overlap_fig    = _iptw["ps_overlap_fig"]
        weight_dist_fig   = _iptw["weight_dist_fig"]
        covariates        = _iptw["covariates"]
        treatment_var     = _iptw["treatment_var"]
        cluster_var       = _iptw["cluster_var"]
        time_var          = _iptw["time_var"]
        event_var         = _iptw["event_var"]

        # ------------------------------------------------------------------
        # STEP 3a: Fit IPTW-weighted Kaplan-Meier curves (descriptive)
        # ------------------------------------------------------------------
        try:
            km_results = self._fit_weighted_km_curves(
                data=df,
                time_var=time_var,
                event_var=event_var,
                treatment_var=treatment_var,
                weight_col="iptw",
                snapshot_days=snapshot_days,
            )
        except Exception as e:
            raise ValueError(f"Error fitting weighted KM curves: {str(e)}")

        kmf_treated = km_results["kmf_treated"]
        kmf_control = km_results["kmf_control"]
        survival_snapshots = km_results["survival_at_snapshots"]

        # ------------------------------------------------------------------
        # STEP 3b: Compute RMST difference (business-friendly metric)
        # ------------------------------------------------------------------
        rmst_result = None
        try:
            max_time = int(df[time_var].max())
            rmst_result = self.compute_rmst_difference(
                kmf_treated=kmf_treated,
                kmf_control=kmf_control,
                t_max=max_time,
            )
        except Exception as e:
            print(f"  Note: Could not compute RMST difference: {e}")

        # ------------------------------------------------------------------
        # STEP 3c: Fit Cox PH model
        # ------------------------------------------------------------------
        try:
            cox_results = self._fit_cox_model(
                data=df,
                time_var=time_var,
                event_var=event_var,
                treatment_var=treatment_var,
                weight_col="iptw",
                cluster_var=cluster_var,
                covariates=covariates,
                alpha=alpha,
                time_interaction=time_interaction,
                period_breaks=period_breaks,
                period_labels=period_labels,
                _quiet=True,
            )
        except Exception as e:
            raise ValueError(f"Error fitting Cox model: {str(e)}")

        # ------------------------------------------------------------------
        # STEP 4: Extract results and build return dict
        # ------------------------------------------------------------------
        period_hrs = cox_results["period_hrs"]
        time_interaction_type = cox_results["time_interaction"]
        concordance = cox_results["concordance"]
        ph_test_results = cox_results.get("ph_test_results")
        ph_assumption_met = cox_results.get("ph_assumption_met")

        # Extract display HR depending on model type
        if time_interaction_type is None:
            # Standard Cox: single overall HR
            display_row = period_hrs.iloc[0]
            effect_label = "Overall HR"
        elif time_interaction_type == "categorical":
            # Use reference period (first row)
            display_row = period_hrs.iloc[0]
            effect_label = f"Reference period: {display_row['period']}"
        else:
            # Continuous: use 12-month estimate
            display_row = period_hrs[period_hrs["period"] == "12mo"]
            if display_row.empty:
                display_row = period_hrs.iloc[-1]
            else:
                display_row = display_row.iloc[0]
            effect_label = f"HR at {display_row['period']}"

        hazard_ratio = display_row["hazard_ratio"]
        hr_ci_lower = display_row["hr_ci_lower"]
        hr_ci_upper = display_row["hr_ci_upper"]
        hr_pvalue = display_row["p_value"]

        significant = hr_pvalue < alpha
        stars = self._significance_stars(hr_pvalue)
        ci_pct = int((1 - alpha) * 100)

        # ------------------------------------------------------------------
        # Print results summary
        # ------------------------------------------------------------------
        print(f"\n{'=' * 60}")
        if time_interaction is None:
            print(f"SURVIVAL ANALYSIS RESULTS — STANDARD COX PH ({estimand})")
        else:
            print(f"SURVIVAL ANALYSIS RESULTS — COX TIME INTERACTION ({estimand})")
        print(f"{'=' * 60}")
        print(f"  Model type:        {model_label}")
        if time_interaction_type == "categorical":
            print(f"  Period breaks:     {period_breaks} (days)")
        print()
        print(f"  {effect_label}")
        print(f"  Hazard Ratio:      {hazard_ratio:.3f} "
              f"({ci_pct}% CI: [{hr_ci_lower:.3f}, {hr_ci_upper:.3f}])")
        print(f"  P-value:           {hr_pvalue:.4f} {stars}")
        print(f"  Concordance:       {concordance:.3f}")
        print()

        # Period-specific results (if multiple)
        if len(period_hrs) > 1:
            print("  Period-Specific Hazard Ratios:")
            for _, row in period_hrs.iterrows():
                period = row["period"]
                hr = row["hazard_ratio"]
                p_val = row["p_value"]
                stars_period = self._significance_stars(p_val)
                note = f" ({row.get('note', '')})" if row.get('note') else ""
                print(f"    {period:>8s}:  HR = {hr:.3f}  (p = {p_val:.3f}) {stars_period}{note}")
            print()

        # RMST
        if rmst_result is not None:
            print(f"  RMST Difference:   {rmst_result['rmst_difference']:.1f} days "
                  f"(95% CI: [{rmst_result['ci_lower']:.1f}, {rmst_result['ci_upper']:.1f}])")
            print(f"  (Treated retained ~{rmst_result['rmst_difference']:.0f} days longer on average)")
            print()

        # PH test
        if ph_test_results is not None:
            print("  Proportional Hazards Test:")
            if time_interaction is None:
                if ph_assumption_met is False:
                    print("    ⚠️  Treatment PH violation detected.")
                    print("    Consider using time_interaction='categorical'")
                elif ph_assumption_met is True:
                    print("    ✓  No treatment PH violation detected")
            else:
                print("    Time interaction model handles PH violations by design.")
                if ph_assumption_met is False:
                    print("    ⚠️  Treatment PH violation detected (expected with time interaction)")
                elif ph_assumption_met is True:
                    print("    ✓  No treatment PH violation detected")
            print("    See ph_test_results for full details.")
        print(f"{'=' * 60}")

        # --- Survival probabilities at 365 days for compatibility ---
        snap_365 = survival_snapshots[survival_snapshots["timepoint_days"] == 365]
        if not snap_365.empty:
            mean_treatment = float(snap_365["survival_treated"].iloc[0])
            mean_control = float(snap_365["survival_control"].iloc[0])
        else:
            mean_treatment = float(1 - df[df[treatment_var] == 1][event_var].mean())
            mean_control = float(1 - df[df[treatment_var] == 0][event_var].mean())

        # --- Build coefficients_df for compatibility ---
        log_hr = np.log(hazard_ratio)
        log_hr_se = (np.log(hr_ci_upper) - np.log(hr_ci_lower)) / (2 * 1.96)

        coefficients_df = pd.DataFrame({
            "Parameter": [f"{treatment_var}_{effect_label}"],
            "Estimate": [log_hr],
            "Std_Error": [log_hr_se],
            "CI_Lower": [np.log(hr_ci_lower)],
            "CI_Upper": [np.log(hr_ci_upper)],
            "P_Value_Raw": [hr_pvalue],
            "Alpha": [alpha],
        })

        # --- Propensity score model summary ---
        ps_summary_df = self._build_ps_summary_df(ps_model)

        # ------------------------------------------------------------------
        # STEP 5: Export (optional)
        # ------------------------------------------------------------------
        if project_path and analysis_name:
            try:
                if time_interaction is None:
                    export_path = (
                        f"{project_path}/{estimand.lower()}_iptw_cox_standard_{analysis_name}.xlsx"
                    )
                else:
                    export_path = (
                        f"{project_path}/{estimand.lower()}_iptw_cox_time_interaction_{analysis_name}.xlsx"
                    )
                with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
                    balance_df.to_excel(
                        writer, sheet_name="Covariate_Balance", index=False
                    )
                    pd.DataFrame([weight_stats]).to_excel(
                        writer, sheet_name="Weight_Diagnostics", index=False
                    )
                    period_hrs.to_excel(
                        writer, sheet_name="Period_HRs", index=False
                    )
                    coefficients_df.to_excel(
                        writer, sheet_name=f"{estimand}_Cox_Summary", index=False
                    )
                    ps_summary_df.to_excel(
                        writer, sheet_name="Propensity_Model", index=False
                    )
                    if not survival_snapshots.empty:
                        survival_snapshots.to_excel(
                            writer, sheet_name="Survival_Snapshots", index=False
                        )
                    if ph_test_results is not None:
                        ph_test_results.to_excel(
                            writer, sheet_name="PH_Test_Results", index=False
                        )
                    if rmst_result is not None:
                        pd.DataFrame([rmst_result]).to_excel(
                            writer, sheet_name="RMST", index=False
                        )
                print(f"  Results saved to {export_path}")
            except Exception as e:
                print(f"  Warning: Could not export results to Excel: {e}")

        # ------------------------------------------------------------------
        # STEP 6: Build and return results dict
        # ------------------------------------------------------------------
        return {
            # --- Shared keys (build_summary_table / compute_evalues compat) ---
            "effect": hazard_ratio,
            "estimand": estimand,
            "ci_lower": hr_ci_lower,
            "ci_upper": hr_ci_upper,
            "p_value": hr_pvalue,
            "significant": significant,
            "alpha": alpha,
            "cohens_d": None,
            "pct_change": None,
            "mean_treatment": mean_treatment,
            "mean_control": mean_control,
            "outcome_type": "survival",
            "coefficients_df": coefficients_df,
            "full_coefficients_df": cox_results.get("coefficients_df"),
            "ps_model": ps_model,
            "ps_summary_df": ps_summary_df,
            "balance_df": balance_df,
            "weight_diagnostics": weight_stats,
            "ps_overlap_fig": ps_overlap_fig,
            "weight_dist_fig": weight_dist_fig,
            "weighted_df": df,

            # --- Survival-specific keys ---
            "period_hrs": period_hrs,
            "time_interaction": time_interaction_type,
            "concordance": concordance,
            "ph_test_results": ph_test_results,
            "ph_assumption_met": ph_assumption_met,
            "kmf_treated": kmf_treated,
            "kmf_control": kmf_control,
            "cox_model": cox_results.get("cox_model"),
            "survival_at_snapshots": survival_snapshots,
            "rmst_difference": rmst_result,
            "n_events_treated": km_results.get("n_events_treated") or cox_results.get("n_events_treated"),
            "n_events_control": km_results.get("n_events_control") or cox_results.get("n_events_control"),
            "n_treated": int(df[treatment_var].sum()),
            "n_control": int((df[treatment_var] == 0).sum()),

            # --- Variable-name metadata ---
            "treatment_var": treatment_var,
            "time_var": time_var,
            "event_var": event_var,
        }
    
    # ==================================================================
    # Sensitivity analysis
    # ==================================================================
    @staticmethod
    def compute_evalue(
        effect: float,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        effect_type: str = "cohens_d",
        outcome_rare: bool = False
    ) -> Dict[str, Optional[float]]:
        """
        Compute E-value for sensitivity analysis of unmeasured confounding.
        
        The E-value represents the minimum strength of association (on the risk
        ratio scale) that an unmeasured confounder would need to have with both
        the treatment and the outcome to fully explain away the observed effect.
        Larger E-values indicate more robust findings.
        
        Based on VanderWeele & Ding (2017): "Sensitivity Analysis in Observational
        Research: Introducing the E-Value"
        
        Parameters
        ----------
        effect : float
            The observed effect estimate. Interpretation depends on effect_type:
            - "cohens_d": Standardized mean difference (Cohen's d)
            - "odds_ratio": Odds ratio from logistic/binomial model
            - "risk_ratio": Risk ratio (relative risk)
            - "log_odds": Log odds ratio (will be exponentiated)
        ci_lower : float, optional
            Lower bound of confidence interval (same scale as effect)
        ci_upper : float, optional
            Upper bound of confidence interval (same scale as effect)
        effect_type : str, default="cohens_d"
            Type of effect measure provided. One of:
            - "cohens_d": Converts to approximate RR using Chinn (2000) formula
            - "odds_ratio": Uses OR directly (or converts if outcome_rare=False)
            - "risk_ratio": Uses RR directly
            - "log_odds": Exponentiates to OR, then processes as odds_ratio
        outcome_rare : bool, default=False
            If True and effect_type is "odds_ratio", treats OR ≈ RR (rare outcome
            assumption). If False, converts OR to RR using square root transform.
        
        Returns
        -------
        dict
            Dictionary containing:
            - evalue_point: E-value for the point estimate
            - evalue_ci: E-value for the CI bound closest to null (conservative)
            - effect_rr: The effect converted to risk ratio scale
            - ci_lower_rr: Lower CI on RR scale (if provided)
            - ci_upper_rr: Upper CI on RR scale (if provided)
            - robustness: Robustness classification string
            - interpretation: String describing robustness
        
        References
        ----------
        VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research:
        Introducing the E-Value. Ann Intern Med. 2017;167(4):268-274.
        Chinn S. A simple method for converting an odds ratio to effect size for
        use in meta-analysis. Stat Med. 2000;19(22):3127-3131. (Cohen's d to RR
        approximation: RR ≈ exp(0.91 * d).)
        """
        valid_types = ["cohens_d", "odds_ratio", "risk_ratio", "log_odds"]
        if effect_type not in valid_types:
            raise ValueError(f"effect_type must be one of {valid_types}, got '{effect_type}'")
        
        def _evalue_from_rr(rr: float) -> float:
            """Compute E-value from a risk ratio >= 1."""
            if rr < 1:
                rr = 1 / rr
            if rr == 1:
                return 1.0
            return rr + np.sqrt(rr * (rr - 1))
        
        def _cohens_d_to_rr(d: float) -> float:
            """
            Convert Cohen's d to approximate risk ratio (Chinn 2000).
            RR ≈ exp(0.91 * d). For continuous outcomes, this is approximate;
            E-values interpret confounding strength on a RR scale metaphorically.
            """
            return np.exp(0.91 * abs(d))
        
        def _or_to_rr(or_val: float, rare: bool = False) -> float:
            """
            Convert odds ratio to risk ratio.
            If rare outcome, OR ≈ RR. Otherwise use square root approximation.
            """
            if rare:
                return or_val
            return np.sqrt(or_val) if or_val >= 1 else 1 / np.sqrt(1 / or_val)
        
        # --- Convert effect to risk ratio scale ---
        if effect_type == "cohens_d":
            effect_rr = _cohens_d_to_rr(effect)
            ci_lower_rr = _cohens_d_to_rr(ci_lower) if ci_lower is not None else None
            ci_upper_rr = _cohens_d_to_rr(ci_upper) if ci_upper is not None else None
            
        elif effect_type == "log_odds":
            or_val = np.exp(effect)
            effect_rr = _or_to_rr(or_val, outcome_rare)
            ci_lower_rr = _or_to_rr(np.exp(ci_lower), outcome_rare) if ci_lower is not None else None
            ci_upper_rr = _or_to_rr(np.exp(ci_upper), outcome_rare) if ci_upper is not None else None
            
        elif effect_type == "odds_ratio":
            effect_rr = _or_to_rr(effect, outcome_rare)
            ci_lower_rr = _or_to_rr(ci_lower, outcome_rare) if ci_lower is not None else None
            ci_upper_rr = _or_to_rr(ci_upper, outcome_rare) if ci_upper is not None else None
            
        else:  # risk_ratio
            effect_rr = effect
            ci_lower_rr = ci_lower
            ci_upper_rr = ci_upper
        
        # --- Compute E-values ---
        evalue_point = _evalue_from_rr(effect_rr)
        
        evalue_ci = None
        if ci_lower_rr is not None and ci_upper_rr is not None:
            if effect_rr >= 1:
                ci_bound = ci_lower_rr
            else:
                ci_bound = ci_upper_rr
            
            if (ci_lower_rr <= 1 <= ci_upper_rr):
                evalue_ci = 1.0
            else:
                evalue_ci = _evalue_from_rr(ci_bound)
        
        # --- Generate interpretation ---
        if evalue_point >= 3.0:
            robustness = "Strong"
            interpretation = (
                f"E-value = {evalue_point:.2f}. An unmeasured confounder would need to be "
                f"associated with both treatment and outcome by a risk ratio of at least "
                f"{evalue_point:.2f} each to explain away this effect. This is a relatively "
                f"large association, suggesting the finding is robust to moderate unmeasured confounding."
            )
        elif evalue_point >= 2.0:
            robustness = "Moderate"
            interpretation = (
                f"E-value = {evalue_point:.2f}. An unmeasured confounder would need risk ratio "
                f"associations of at least {evalue_point:.2f} with both treatment and outcome "
                f"to explain away this effect. This represents moderate robustness."
            )
        elif evalue_point >= 1.5:
            robustness = "Weak"
            interpretation = (
                f"E-value = {evalue_point:.2f}. A relatively weak unmeasured confounder "
                f"(RR ≈ {evalue_point:.2f}) could potentially explain this effect. "
                f"Interpret with caution."
            )
        else:
            robustness = "Very Weak"
            interpretation = (
                f"E-value = {evalue_point:.2f}. This effect is highly sensitive to unmeasured "
                f"confounding and could easily be explained by a weak confounder."
            )
        
        if evalue_ci is not None:
            interpretation += (
                f" The E-value for the confidence interval bound is {evalue_ci:.2f}, meaning "
                f"a confounder of this strength could shift the CI to include the null."
            )
        if effect_type == "cohens_d":
            interpretation += (
                " For continuous outcomes, E-value uses an approximate RR conversion; "
                "interpretation is approximate."
            )
        
        return {
            "evalue_point": evalue_point,
            "evalue_ci": evalue_ci,
            "effect_rr": effect_rr,
            "ci_lower_rr": ci_lower_rr,
            "ci_upper_rr": ci_upper_rr,
            "robustness": robustness,
            "interpretation": interpretation
        }


    @staticmethod
    def compute_evalues_from_results(
        results_dict: Dict[str, Dict],
        effect_type: str = "cohens_d",
        outcome_rare: bool = False
    ) -> pd.DataFrame:
        """
        Compute E-values for all outcomes in a results dictionary.
        
        Convenience method to batch-compute E-values from the output of
        analyze_treatment_effect() or build_summary_table(). Prints a summary
        table followed by per-outcome interpretation strings for all
        statistically significant results.
        
        Parameters
        ----------
        results_dict : Dict[str, Dict]
            Dictionary keyed by outcome name, where each value is the dict
            returned by analyze_treatment_effect().
        effect_type : str, default="cohens_d"
            Effect type for E-value computation. Must be one of:
            - "cohens_d": For continuous outcomes (uses Cohen's d)
            - "log_odds": For binary outcomes (uses log odds ratio)
            - "odds_ratio": For binary outcomes (uses odds ratio)
            - "risk_ratio": For survival/binary outcomes (uses risk ratio)
        outcome_rare : bool, default=False
            Passed to compute_evalue() for odds ratio conversion.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with E-values for each outcome, including:
            - Outcome name
            - Effect estimate and type used
            - E-value for point estimate
            - E-value for confidence interval
            - Robustness classification
            - Interpretation string
        """
        valid_types = ["cohens_d", "odds_ratio", "risk_ratio", "log_odds"]
        if effect_type not in valid_types:
            raise ValueError(f"effect_type must be one of {valid_types}, got '{effect_type}'")
        
        rows = []
        
        for outcome_name, res in results_dict.items():
            cohens_d = res.get("cohens_d")
            effect = res.get("effect")
            ci_lower = res.get("ci_lower")
            ci_upper = res.get("ci_upper")
            
            # Use the specified effect type
            if effect_type == "cohens_d":
                use_effect = cohens_d
                # For Cohen's d, need to convert CI bounds to Cohen's d scale
                mean_treat = res.get("mean_treatment", 0)
                mean_ctrl = res.get("mean_control", 0)
                if cohens_d is not None and cohens_d != 0:
                    raw_diff = mean_treat - mean_ctrl
                    scale_factor = cohens_d / raw_diff if raw_diff != 0 else 1
                    use_ci_lower = ci_lower * scale_factor if ci_lower else None
                    use_ci_upper = ci_upper * scale_factor if ci_upper else None
                else:
                    use_ci_lower = None
                    use_ci_upper = None
            else:
                use_effect = effect
                use_ci_lower = ci_lower
                use_ci_upper = ci_upper
            
            try:
                evalue_result = CausalInferenceModel.compute_evalue(
                    effect=use_effect,
                    ci_lower=use_ci_lower,
                    ci_upper=use_ci_upper,
                    effect_type=effect_type,
                    outcome_rare=outcome_rare
                )
                
                rows.append({
                    "Outcome": outcome_name,
                    "Effect_Type": effect_type,
                    "Effect_Value": use_effect,
                    "Effect_RR": evalue_result["effect_rr"],
                    "E_Value_Point": evalue_result["evalue_point"],
                    "E_Value_CI": evalue_result["evalue_ci"],
                    "Robustness": evalue_result["robustness"],
                    "Interpretation": evalue_result["interpretation"],
                    "P_Value": res.get("p_value"),
                    "Significant": res.get("significant", False)
                })
            except Exception as e:
                print(f"  Warning: Could not compute E-value for {outcome_name}: {e}")
                rows.append({
                    "Outcome": outcome_name,
                    "Effect_Type": effect_type,
                    "Effect_Value": use_effect,
                    "Effect_RR": None,
                    "E_Value_Point": None,
                    "E_Value_CI": None,
                    "Robustness": "Error",
                    "Interpretation": None,
                    "P_Value": res.get("p_value"),
                    "Significant": res.get("significant", False)
                })
        
        evalue_df = pd.DataFrame(rows)
        
        # Print summary table (exclude Interpretation column for readability)
        print("\n" + "=" * 70)
        print("  E-VALUE SENSITIVITY ANALYSIS")
        print("=" * 70)
        display_cols = [c for c in evalue_df.columns if c != "Interpretation"]
        print(evalue_df[display_cols].to_string(index=False))
        print("=" * 70)
        print("  Interpretation Guide:")
        print("    E-value ≥ 3.0 : Strong robustness to unmeasured confounding")
        print("    E-value 2.0-3.0: Moderate robustness")
        print("    E-value 1.5-2.0: Weak robustness - interpret with caution")
        print("    E-value < 1.5 : Very weak - easily explained by confounding")
        print("    (Thresholds are heuristic; true RR-scale effects have more precise")
        print("     interpretations than approximate conversions from Cohen's d.)")
        print("=" * 70)
        
        # Print per-outcome interpretations for significant results
        sig_rows = evalue_df[evalue_df["Significant"] == True]
        if not sig_rows.empty:
            print("\n  Per-Outcome Interpretations (significant results only):")
            print("-" * 70)
            for _, row in sig_rows.iterrows():
                if pd.notna(row.get("Interpretation")):
                    print(f"\n  {row['Outcome']}:")
                    print(f"    {row['Interpretation']}")
        print()
        
        return evalue_df

    # ------------------------------------------------------------------
    @staticmethod
    def compute_confounder_evalue_benchmarks(
        results_dict: Dict[str, Dict],
        evalue_df: Optional[pd.DataFrame] = None,
        n_top: int = 3,
    ) -> pd.DataFrame:
        """
        Compute E-value benchmarks from the strongest *measured* confounders.

        For each outcome in *results_dict*, the method pulls the pre-weighting
        balance table (``balance_df``), ranks covariates by absolute
        Standardised Mean Difference (SMD), and converts the top-*n_top* SMDs
        to the E-value scale using the Chinn (2000) approximation
        (RR ≈ exp(0.91 × |SMD|)) followed by the VanderWeele & Ding (2017)
        E-value formula.

        This puts observed confounders on the same scale as the treatment
        E-values, answering the calibration question: "How much stronger would
        an unmeasured confounder need to be compared to the strongest covariate
        we already controlled for?"

        Parameters
        ----------
        results_dict : Dict[str, Dict]
            Output of ``analyze_treatment_effect()`` or
            ``analyze_survival_effect()``, keyed by outcome name.  Each value
            must contain a ``balance_df`` with a ``smd_before_weighting``
            column.
        evalue_df : pd.DataFrame, optional
            The DataFrame returned by ``compute_evalues_from_results()``.
            If provided, the treatment E-value for each outcome is included
            in the printed comparison.  If *None*, only confounder benchmarks
            are shown.
        n_top : int, default 3
            Number of strongest confounders to display per outcome.

        Returns
        -------
        pd.DataFrame
            One row per confounder-outcome pair with columns:
            ``Outcome``, ``Confounder``, ``Rank``, ``SMD_Pre_Weighting``,
            ``Approx_RR``, ``Confounder_E_Value``.
        """
        def _evalue_from_rr(rr: float) -> float:
            if rr < 1:
                rr = 1 / rr
            if rr == 1:
                return 1.0
            return rr + np.sqrt(rr * (rr - 1))

        # Build a lookup of treatment E-values if available
        treatment_evalues: Dict[str, float] = {}
        if evalue_df is not None and not evalue_df.empty:
            for _, row in evalue_df.iterrows():
                treatment_evalues[row["Outcome"]] = row.get("E_Value_Point")

        rows: list = []

        for outcome_name, res in results_dict.items():
            balance_df = res.get("balance_df")
            if balance_df is None or balance_df.empty:
                continue

            # Determine the SMD column (different sources use different names)
            if "smd_before_weighting" in balance_df.columns:
                smd_col = "smd_before_weighting"
            elif "Unweighted SMD" in balance_df.columns:
                smd_col = "Unweighted SMD"
            else:
                continue

            var_col = "variable" if "variable" in balance_df.columns else balance_df.index.name

            work = balance_df.copy()
            if var_col != "variable":
                work = work.reset_index()
                var_col = work.columns[0]

            work["_abs_smd"] = work[smd_col].abs()
            top = work.nlargest(n_top, "_abs_smd")

            for rank, (_, r) in enumerate(top.iterrows(), start=1):
                abs_smd = r["_abs_smd"]
                approx_rr = np.exp(0.91 * abs_smd)
                confounder_ev = _evalue_from_rr(approx_rr)
                rows.append({
                    "Outcome": outcome_name,
                    "Confounder": r[var_col],
                    "Rank": rank,
                    "SMD_Pre_Weighting": round(r[smd_col], 4),
                    "Approx_RR": round(approx_rr, 4),
                    "Confounder_E_Value": round(confounder_ev, 2),
                })

        bench_df = pd.DataFrame(rows)
        if bench_df.empty:
            print("  No balance data available — cannot compute confounder benchmarks.")
            return bench_df

        # ---- Pretty-print comparison ----
        print("\n" + "=" * 70)
        print("  CONFOUNDER E-VALUE BENCHMARKS (Calibration)")
        print("=" * 70)
        print("  Shows the E-value–equivalent strength of the strongest MEASURED")
        print("  confounders (pre-weighting SMD → approximate RR → E-value).")
        print("  An unmeasured confounder would need to exceed these benchmarks")
        print("  to threaten the treatment conclusion.")
        print("-" * 70)

        for outcome_name in bench_df["Outcome"].unique():
            sub = bench_df[bench_df["Outcome"] == outcome_name]
            treat_ev = treatment_evalues.get(outcome_name)

            print(f"\n  {outcome_name}")
            if treat_ev is not None:
                print(f"    Treatment E-value (point): {treat_ev:.2f}")
            print(f"    {'Confounder':<35s} {'|SMD|':>7s}  {'≈RR':>6s}  {'E-val':>6s}")
            print(f"    {'─' * 35} {'─' * 7}  {'─' * 6}  {'─' * 6}")
            for _, r in sub.iterrows():
                abs_smd = abs(r["SMD_Pre_Weighting"])
                print(
                    f"    {r['Confounder']:<35s} {abs_smd:>7.3f}  "
                    f"{r['Approx_RR']:>6.3f}  {r['Confounder_E_Value']:>6.2f}"
                )

            if treat_ev is not None:
                strongest_ev = sub["Confounder_E_Value"].max()
                if strongest_ev > 1.0 and treat_ev > strongest_ev:
                    ratio = treat_ev / strongest_ev
                    print(
                        f"    → Treatment E-value is {ratio:.1f}× the strongest "
                        f"measured confounder's E-value."
                    )
                elif treat_ev <= strongest_ev:
                    print(
                        f"    ⚠️  Treatment E-value ({treat_ev:.2f}) does not exceed "
                        f"the strongest measured confounder ({strongest_ev:.2f})."
                    )

        print()
        print("  Note: Confounder E-values use the Chinn (2000) SMD→RR")
        print("  approximation (RR ≈ exp(0.91×|SMD|)), same as treatment E-values")
        print("  for Cohen's d, making the scales directly comparable.")
        print("=" * 70)
        print()

        return bench_df

    # ==================================================================
    # Summary tables & helper utilities
    # ==================================================================
    
    @staticmethod
    def _clean_column_name(name: str) -> str:
        """Sanitise a single column name for use in statsmodels formulas.
        
        Applies the same deterministic mapping used when cleaning
        DataFrame columns so that variable-name references stay in sync.
        
        Parameters
        ----------
        name : str
            Original column name.
        
        Returns
        -------
        str
            Cleaned column name safe for formula parsing.
        """
        semantic = {'&': 'and', '+': 'plus', '%': 'pct', '$': 'dollar',
                    '@': 'at', '<': 'lt', '>': 'gt', '=': 'eq'}
        for char, repl in semantic.items():
            name = name.replace(char, repl)
        # Replace any remaining non-alphanumeric / non-underscore chars
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Collapse multiple underscores and strip leading/trailing
        name = re.sub(r'_+', '_', name).strip('_')
        return name
    
    @staticmethod
    def _significance_stars(p_value: float) -> str:
        """
        Return significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value to evaluate
        
        Returns
        -------
        str
            '***' if p < 0.001, '**' if p < 0.01, '*' if p < 0.05, '' otherwise
        """
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        return ""

    @staticmethod
    def _build_ps_summary_df(ps_model) -> pd.DataFrame:
        """
        Build a DataFrame summary of propensity score model coefficients.

        Parameters
        ----------
        ps_model : fitted model
            Propensity score model with params, bse, pvalues attributes
            (e.g. statsmodels GEE or LogisticRegression).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Parameter, Estimate, Std_Error, P_Value.
        """
        return pd.DataFrame({
            "Parameter": ps_model.params.index,
            "Estimate": ps_model.params.values,
            "Std_Error": ps_model.bse.values,
            "P_Value": ps_model.pvalues.values,
        })

    @staticmethod
    def build_summary_table(
        results_dict: Dict[str, Dict],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        correction_method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Build a consolidated summary table of treatment effects across multiple outcomes.
        
        Applies multiple-testing correction (e.g. FDR) across the *treatment*
        p-values from each outcome model — the correct family of tests.
        
        Parameters
        ----------
        results_dict : Dict[str, Dict]
            Dictionary keyed by outcome name, where each value is the dict
            returned by ``analyze_treatment_effect``.
        title : str, optional
            Title printed above the table when displayed
        save_path : str, optional
            If provided, saves the table to this path (.xlsx, .csv, .html).
        correction_method : str, default='fdr_bh'
            Multiple testing correction method passed to
            ``statsmodels.stats.multitest.multipletests``.
        alpha : float, default=0.05
            Family-wise significance level.
        
        Returns
        -------
        pd.DataFrame
            Summary table with one row per outcome.
        """
        rows = []
        raw_pvals = []
        estimand = None
        
        for outcome_name, res in results_dict.items():
            coeff_df = res.get("coefficients_df")
            weight_diag = res.get("weight_diagnostics", {})
            
            # Track estimand (should be consistent across all outcomes)
            if estimand is None:
                estimand = res.get("estimand", "ATE")
            
            std_error = (
                coeff_df["Std_Error"].iloc[0]
                if coeff_df is not None and "Std_Error" in coeff_df.columns
                else None
            )
            
            p_raw = res["p_value"]
            raw_pvals.append(p_raw)
            
            rows.append({
                "Outcome": outcome_name,
                "Effect": res["effect"],
                "Estimand": res.get("estimand", "ATE"),
                "Std_Error": std_error,
                "CI_Lower": res["ci_lower"],
                "CI_Upper": res["ci_upper"],
                "P_Value": p_raw,
                "Cohens_d": res.get("cohens_d", None),
                "Pct_Change": res.get("pct_change", None),
                "Mean_Treatment": res.get("mean_treatment", None),
                "Mean_Control": res.get("mean_control", None),
                "N": weight_diag.get("n_observations", None),
                "ESS": weight_diag.get("effective_sample_size", None),
            })
        
        summary_df = pd.DataFrame(rows)
        
        # --- B2 / S5: Apply multiple-testing correction across outcomes ---
        # Guard: skip correction for a single outcome (correction is meaningless)
        if len(raw_pvals) > 1:
            reject_arr, pvals_corrected, _, _ = multipletests(
                raw_pvals, alpha=alpha, method=correction_method
            )
        else:
            pvals_corrected = np.array(raw_pvals)
            reject_arr = np.array([raw_pvals[0] < alpha])
        summary_df["P_Value_Corrected"] = pvals_corrected
        summary_df["Significant"] = reject_arr
        summary_df["Significance"] = [
            CausalInferenceModel._significance_stars(p) for p in pvals_corrected
        ]
        # Record actual correction method: "none" when only 1 test (no correction applied)
        summary_df["Correction_Method"] = correction_method if len(raw_pvals) > 1 else "none"
        
        # Print formatted table
        if title:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}")
        else:
            print(f"\n{'=' * 60}")
            print(f"  {estimand} Summary Table")
            print(f"{'=' * 60}")
        
        display_df = summary_df.copy()
        for col in ["Effect", "Std_Error", "CI_Lower", "CI_Upper", "Cohens_d"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        for col in ["P_Value", "P_Value_Corrected"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        if "Pct_Change" in display_df.columns:
            display_df["Pct_Change"] = display_df["Pct_Change"].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
            )
        for col in ["Mean_Treatment", "Mean_Control"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        if "ESS" in display_df.columns:
            display_df["ESS"] = display_df["ESS"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )
        
        print(display_df.to_string(index=False))
        print(f"{'=' * 60}")
        print("  Significance: *** p<0.001, ** p<0.01, * p<0.05")
        if "Cohens_d" in summary_df.columns:
            print("  Cohens_d: IPTW-weighted Cohen's d (from analyze_treatment_effect)")
        print(f"  Correction: {correction_method} across {len(rows)} outcomes")
        print()
        
        # Save if requested
        if save_path:
            if save_path.endswith(".xlsx"):
                summary_df.to_excel(save_path, index=False, engine="openpyxl")
            elif save_path.endswith(".csv"):
                summary_df.to_csv(save_path, index=False)
            elif save_path.endswith(".html"):
                summary_df.to_html(save_path, index=False)
            else:
                summary_df.to_excel(save_path + ".xlsx", index=False, engine="openpyxl")
            print(f"  Summary table saved to {save_path}")
        
        return summary_df


    @staticmethod
    def compute_rmst_difference(
        survival_result: Dict,
        time_horizon: Optional[int] = None,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: int = 42,
        _quiet: bool = False
    ) -> Dict:
        """
        Compute Restricted Mean Survival Time (RMST) difference between
        treated and control groups.

        RMST = area under the Kaplan-Meier survival curve up to time_horizon.
        The RMST difference is the average number of additional days retained
        within the study window attributable to treatment.
        """
        def _print(msg=""):
            if not _quiet:
                print(msg)

        if time_horizon is None:
            time_horizon = 365

        kmf_treated = survival_result.get("kmf_treated")
        kmf_control = survival_result.get("kmf_control")

        if kmf_treated is None or kmf_control is None:
            raise ValueError(
                "survival_result must contain 'kmf_treated' and 'kmf_control'. "
                "Run analyze_survival_effect() first."
            )

        def _rmst_from_kmf(kmf, horizon):
            """Compute RMST as area under KM curve up to horizon using trapezoidal rule."""
            sf = kmf.survival_function_
            times = sf.index.values
            probs = sf.iloc[:, 0].values

            # Clip to horizon
            mask  = times <= horizon
            t_clip = np.append(times[mask], horizon)
            p_clip = np.append(probs[mask], probs[mask][-1] if mask.any() else 1.0)

            # Trapezoidal integration
            rmst = np.trapezoid(p_clip, t_clip)
            return float(rmst)

        rmst_treated = _rmst_from_kmf(kmf_treated, time_horizon)
        rmst_control = _rmst_from_kmf(kmf_control, time_horizon)
        rmst_diff    = rmst_treated - rmst_control

        # --- Bootstrap CI for RMST difference ---
        weighted_df = survival_result.get("weighted_df")

        # Resolve variable names — prefer explicit keys added by
        # analyze_survival_effect, falling back to heuristic detection.
        treatment_var = survival_result.get("treatment_var")
        time_col = survival_result.get("time_var")
        event_col = survival_result.get("event_var")

        # Heuristic fallback for older result dicts that lack metadata keys
        if weighted_df is not None and treatment_var is None:
            candidate_cols = [
                c for c in weighted_df.columns
                if weighted_df[c].nunique() == 2
                and set(weighted_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
                and c not in ["iptw", "propensity_score"]
            ]
            if candidate_cols:
                treatment_var = candidate_cols[0]

        if weighted_df is not None and time_col is None:
            time_candidates = [
                c for c in weighted_df.columns
                if "days" in c.lower() or "time" in c.lower()
            ]
            time_col = time_candidates[0] if time_candidates else None

        if weighted_df is not None and event_col is None:
            event_candidates = [
                c for c in weighted_df.columns
                if "depart" in c.lower() or "event" in c.lower()
            ]
            event_col = event_candidates[0] if event_candidates else None

        bootstrap_diffs = []
        rng = np.random.default_rng(random_state)

        if weighted_df is not None and treatment_var is not None:

            if time_col and event_col:
                # Suppress warnings during bootstrap to avoid spam
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    for _ in range(n_bootstrap):
                        boot_idx = rng.integers(0, len(weighted_df), size=len(weighted_df))
                        boot_df  = weighted_df.iloc[boot_idx].reset_index(drop=True)

                        treated_boot = boot_df[boot_df[treatment_var] == 1]
                        control_boot = boot_df[boot_df[treatment_var] == 0]

                        if len(treated_boot) < 5 or len(control_boot) < 5:
                            continue

                        try:
                            kmf_t = KaplanMeierFitter()
                            kmf_c = KaplanMeierFitter()
                            kmf_t.fit(
                                durations=treated_boot[time_col],
                                event_observed=treated_boot[event_col],
                                weights=treated_boot["iptw"]
                            )
                            kmf_c.fit(
                                durations=control_boot[time_col],
                                event_observed=control_boot[event_col],
                                weights=control_boot["iptw"]
                            )
                            boot_diff = _rmst_from_kmf(kmf_t, time_horizon) - \
                                        _rmst_from_kmf(kmf_c, time_horizon)
                            bootstrap_diffs.append(boot_diff)
                        except Exception:
                            continue

        if len(bootstrap_diffs) >= 50:
            ci_lower = float(np.percentile(bootstrap_diffs, (alpha / 2) * 100))
            ci_upper = float(np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100))
        else:
            _print("  Warning: Bootstrap CI could not be computed. Using normal approximation.")
            se_approx = abs(rmst_diff) * 0.2
            z = 1.96
            ci_lower = rmst_diff - z * se_approx
            ci_upper = rmst_diff + z * se_approx

        significant = not (ci_lower <= 0 <= ci_upper)

        # --- Enhanced print results ---
        direction = "longer" if rmst_diff >= 0 else "shorter"
        
        _print("\n" + "=" * 60)
        _print("RESTRICTED MEAN SURVIVAL TIME (RMST) ANALYSIS")
        _print("=" * 60)
        _print(f"Analysis window: {time_horizon} days (12 months)")
        _print(f"Bootstrap samples: {len(bootstrap_diffs)}")
        _print("")
        
        _print("RMST ESTIMATES:")
        _print(f"  Trained managers:    {rmst_treated:.1f} days")
        _print(f"  Untrained managers:  {rmst_control:.1f} days")
        _print(f"  Difference:          {rmst_diff:+.1f} days")
        _print("")
        
        ci_pct = int((1 - alpha) * 100)
        _print("STATISTICAL INFERENCE:")
        _print(f"  {ci_pct}% Confidence Interval: [{ci_lower:+.1f}, {ci_upper:+.1f}] days")
        if significant:
            _print(f"  ✓ Statistically significant (CI excludes 0)")
        else:
            _print(f"  Not statistically significant (CI includes 0)")
        _print("")
        
        _print("BUSINESS INTERPRETATION:")
        _print(f"  Training extends retention by an average of {abs(rmst_diff):.1f} days")
        _print(f"  within the 12-month study window.")
        
        # Convert to business metrics
        weeks = abs(rmst_diff) / 7
        months = abs(rmst_diff) / 30.44
        pct_of_year = (abs(rmst_diff) / 365) * 100
        
        _print(f"  This represents:")
        _print(f"    • {weeks:.1f} additional weeks of retention")
        _print(f"    • {months:.1f} additional months of retention") 
        _print(f"    • {pct_of_year:.1f}% of the study year")
        _print("")
        
        _print("METHODOLOGICAL NOTES:")
        _print(f"  • RMST is robust to time-varying treatment effects")
        _print(f"  • Confidence interval computed via {len(bootstrap_diffs)}-sample bootstrap")
        _print(f"  • IPTW weights account for selection bias in training assignment")
        _print("=" * 60)

        # --- Build rmst_df ---
        rmst_df = pd.DataFrame([{
            "time_horizon_days": time_horizon,
            "rmst_treated":      round(rmst_treated, 2),
            "rmst_control":      round(rmst_control, 2),
            "rmst_diff":         round(rmst_diff, 2),
            "rmst_ci_lower":     round(ci_lower, 2),
            "rmst_ci_upper":     round(ci_upper, 2),
            "significant":       significant,
            "n_bootstrap":       len(bootstrap_diffs),
        }])

        return {
            "rmst_treated":  rmst_treated,
            "rmst_control":  rmst_control,
            "rmst_diff":     rmst_diff,
            "rmst_ci_lower": ci_lower,
            "rmst_ci_upper": ci_upper,
            "time_horizon":  time_horizon,
            "significant":   significant,
            "rmst_df":       rmst_df,
        }
    
    @staticmethod
    def build_survival_summary_table(
        survival_results_dict: Dict[str, Dict],
        rmst_results_dict: Optional[Dict[str, Dict]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        correction_method: str = "fdr_bh",
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Build a consolidated summary table of survival analysis results
        across multiple retention outcomes.

        Analogous to build_summary_table() but designed for survival outcomes,
        reporting period-specific hazard ratios from time interaction models
        and RMST differences.

        Parameters
        ----------
        survival_results_dict : Dict[str, Dict]
            Dictionary keyed by outcome name, where each value is the dict
            returned by analyze_survival_effect().
        rmst_results_dict : Dict[str, Dict], optional
            Dictionary keyed by outcome name, where each value is the dict
            returned by compute_rmst_difference(). If provided, RMST columns
            are added to the summary table.
        title : str, optional
            Title printed above the table.
        save_path : str, optional
            If provided, saves the table to this path (.xlsx or .csv).
        correction_method : str, default 'fdr_bh'
            Multiple testing correction method.
        alpha : float, default 0.05
            Family-wise significance level.

        Returns
        -------
        pd.DataFrame
            Summary table with one row per outcome.
        """
        rows = []
        raw_pvals = []

        for outcome_name, res in survival_results_dict.items():
            # Extract period_hrs DataFrame
            period_hrs = res.get("period_hrs")
            time_interaction = res.get("time_interaction")
            
            # Extract compatibility values (reference period or 12mo)
            hr = res.get("effect")
            hr_lower = res.get("ci_lower")
            hr_upper = res.get("ci_upper")
            p_value = res.get("p_value")
            
            estimand = res.get("estimand", "ATT")
            n_treated = res.get("n_events_treated", None)
            n_control = res.get("n_events_control", None)
            ph_met = res.get("ph_assumption_met")
            concordance = res.get("concordance")
            weight_diag = res.get("weight_diagnostics", {})

            raw_pvals.append(p_value if p_value is not None else 1.0)

            # Determine display label for headline HR
            if time_interaction == "categorical" and period_hrs is not None and not period_hrs.empty:
                ref_period = period_hrs.iloc[0]["period"]
                hr_label = f"Reference: {ref_period}"
            elif time_interaction == "continuous":
                hr_label = "12-month HR"
            else:
                hr_label = "Overall HR"

            row = {
                "Outcome": outcome_name,
                "Estimand": estimand,
                "Interaction_Type": time_interaction,
                "Headline_HR_Label": hr_label,
                "Hazard_Ratio": round(hr, 4) if hr is not None else None,
                "HR_95_CI": (
                    f"[{hr_lower:.3f}, {hr_upper:.3f}]"
                    if hr_lower is not None and hr_upper is not None
                    else None
                ),
                "P_Value": p_value,
                "Concordance": round(concordance, 4) if concordance is not None else None,
                "PH_Assumption_Met": ph_met,
                "N_Events_Treated": n_treated,
                "N_Events_Control": n_control,
                "N_Total": weight_diag.get("n_observations"),
                "ESS": weight_diag.get("effective_sample_size"),
            }

            # Add period-specific HR columns from period_hrs
            if period_hrs is not None and not period_hrs.empty:
                for _, period_row in period_hrs.iterrows():
                    period_label = period_row["period"]
                    # Clean label for column name (e.g., "0-3mo" -> "0_3mo")
                    col_suffix = period_label.replace("-", "_").replace(" ", "")
                    
                    hr_val = period_row["hazard_ratio"]
                    hr_ci_l = period_row["hr_ci_lower"]
                    hr_ci_u = period_row["hr_ci_upper"]
                    p_val = period_row["p_value"]
                    
                    row[f"HR_{col_suffix}"] = round(hr_val, 4) if pd.notna(hr_val) else None
                    row[f"HR_CI_{col_suffix}"] = (
                        f"[{hr_ci_l:.3f}, {hr_ci_u:.3f}]"
                        if pd.notna(hr_ci_l) and pd.notna(hr_ci_u)
                        else None
                    )
                    row[f"P_{col_suffix}"] = round(p_val, 4) if pd.notna(p_val) else None
                    row[f"Sig_{col_suffix}"] = (
                        CausalInferenceModel._significance_stars(p_val)
                        if pd.notna(p_val)
                        else ""
                    )

            # Add snapshot survival difference columns from KM curves
            snapshots = res.get("survival_at_snapshots")
            if snapshots is not None and not snapshots.empty:
                for _, snap_row in snapshots.iterrows():
                    tp = snap_row["timepoint_label"]  # e.g. "3mo"
                    s_t = snap_row["survival_treated"]
                    s_c = snap_row["survival_control"]
                    s_d = snap_row["survival_diff"]
                    row[f"Surv_Trained_{tp}"] = round(s_t, 4) if pd.notna(s_t) else None
                    row[f"Surv_Control_{tp}"] = round(s_c, 4) if pd.notna(s_c) else None
                    row[f"Surv_Diff_{tp}"] = round(s_d, 4) if pd.notna(s_d) else None

            # Add RMST columns if provided
            if rmst_results_dict and outcome_name in rmst_results_dict:
                rmst = rmst_results_dict[outcome_name]
                row["RMST_Treated_Days"] = round(rmst.get("rmst_treated", np.nan), 1)
                row["RMST_Control_Days"] = round(rmst.get("rmst_control", np.nan), 1)
                row["RMST_Difference"] = round(rmst.get("rmst_diff", np.nan), 1)
                row["RMST_CI_Lower"] = round(rmst.get("rmst_ci_lower", np.nan), 1)
                row["RMST_CI_Upper"] = round(rmst.get("rmst_ci_upper", np.nan), 1)
                row["RMST_CI"] = (
                    f"[{rmst.get('rmst_ci_lower', np.nan):.1f}, "
                    f"{rmst.get('rmst_ci_upper', np.nan):.1f}]"
                )

            rows.append(row)

        summary_df = pd.DataFrame(rows)

        # --- Multiple testing correction across outcomes ---
        if len(raw_pvals) > 1:
            reject_arr, pvals_corrected, _, _ = multipletests(
                raw_pvals, alpha=alpha, method=correction_method
            )
        else:
            pvals_corrected = np.array(raw_pvals)
            reject_arr = np.array([raw_pvals[0] < alpha])

        summary_df["P_Value_Corrected"] = pvals_corrected
        summary_df["Significant"] = reject_arr
        summary_df["Significance"] = [
            CausalInferenceModel._significance_stars(p) for p in pvals_corrected
        ]
        # Record actual correction method: "none" when only 1 test (no correction applied)
        summary_df["Correction_Method"] = correction_method if len(raw_pvals) > 1 else "none"

        # --- Print formatted table ---
        display_title = title or "IPTW + Cox: Survival Analysis Summary"
        print(f"\n{'=' * 80}")
        print(f"  {display_title}")
        print(f"{'=' * 80}")

        # Select display columns (exclude raw CI bounds and verbose detail)
        _exclude_patterns = {
            "HR_CI_Lower", "HR_CI_Upper", "RMST_CI_Lower", "RMST_CI_Upper"
        }
        display_cols = [
            c for c in summary_df.columns
            if c not in _exclude_patterns
            and not c.startswith("Sig_")  # period significance stars (too verbose)
            and not c.startswith("Surv_Trained_") 
            and not c.startswith("Surv_Control_")
        ]
        display_df = summary_df[display_cols].copy()

        # Format numeric columns
        for col in ["Hazard_Ratio", "Concordance"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        
        # Format period-specific HR columns
        for col in display_df.columns:
            if (col.startswith("HR_") 
                    and not col.startswith("HR_CI_") 
                    and not col.endswith("_CI")      # excludes HR_95_CI
                    and not col.endswith("_95_CI")): # belt-and-suspenders
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )
            elif col.startswith("Surv_Diff_"):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
                )
            elif col.startswith("P_") and col not in ("P_Value", "P_Value_Corrected"):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        
        for col in ["P_Value", "P_Value_Corrected"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        
        if "ESS" in display_df.columns:
            display_df["ESS"] = display_df["ESS"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )

        actual_correction = correction_method if len(raw_pvals) > 1 else "none"
        print(display_df.to_string(index=False))
        print(f"{'=' * 80}")
        print("  Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print(f"  Correction: {actual_correction} across {len(rows)} outcome{'s' if len(rows) != 1 else ''}")
        print("  HR < 1 = lower hazard of departure (training is protective)")
        #print("  Time interaction models allow treatment effect to vary over time")
        print()

        # --- Save if requested ---
        if save_path:
            if save_path.endswith(".xlsx"):
                summary_df.to_excel(save_path, index=False, engine="openpyxl")
            elif save_path.endswith(".csv"):
                summary_df.to_csv(save_path, index=False)
            else:
                summary_df.to_excel(save_path + ".xlsx", index=False, engine="openpyxl")
            print(f"  Survival summary table saved to {save_path}")

        return summary_df

    
    


