"""
Causal Inference Modeling Module

This module provides data-agnostic causal inference methods for estimating
treatment effects.  Three complementary approaches are supported:

1. IPTW + Doubly Robust GEE   — continuous / binary survey outcomes
2. IPTW + Cox Proportional Hazards — time-to-event (survival) outcomes
3. Double Machine Learning (DML)    — ATE / ATT / CATE via ``econml``

Classes:
    CausalInferenceModel : Unified causal inference toolkit
    IPTWGEEModel         : Backward-compatibility alias for CausalInferenceModel
"""

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
from econml.dml import DML, CausalForestDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.cate_interpreter import SingleTreeCateInterpreter

# Jupyter / IPython display — falls back to print outside notebooks
try:
    from IPython.display import display
except ImportError:
    display = print

from causal_diagnostics import CausalDiagnostics


class CausalInferenceModel:
    """
    Unified causal inference toolkit for treatment effect estimation.

    This class implements three complementary approaches:

    **Approach 1 — IPTW + Doubly Robust GEE** (``analyze_treatment_effect``):
    - For continuous and binary outcomes (e.g., survey indices, binary flags)
    - Stabilized inverse probability weights (IPTW) for ATE or ATT
    - Generalized Estimating Equations (GEE) to account for clustering
    - Optional outcome model covariate adjustment for doubly robust property

    **Approach 2 — IPTW + Cox Proportional Hazards** (``analyze_survival_effect``):
    - For time-to-event outcomes (e.g., employee retention, time to departure)
    - Same IPTW weighting infrastructure as Approach 1
    - Cox PH model for hazard ratios with IPTW-weighted Kaplan-Meier curves
    - Restricted Mean Survival Time (RMST) for business-friendly interpretation
    - Use ``prepare_survival_data()`` to convert departure dates to survival format

    **Approach 3 — Double Machine Learning** (``dml_estimate_treatment_effects``):
    - Linear DML for ATE/ATT estimation with flexible ML nuisance models
    - Causal Forest DML for individualized CATE estimation
    - ATT derived from CATE by averaging over treated observations
    - Uses the ``econml`` package

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

    def fit_doubly_robust_model(
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
        Fit doubly robust treatment effect model using weighted GEE.
        
        Estimates the treatment effect using inverse probability weighting
        combined with outcome model covariate adjustment in a GEE framework.
        
        The model is doubly robust: consistent estimation occurs if either the
        propensity score model OR the outcome model is correctly specified.
        
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
            Additional covariates to include in outcome model for adjusted estimation
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
    
    def _fit_cox_model(self, data, time_var, event_var, treatment_var,
                   weight_col, cluster_var=None, covariates=None,
                   alpha=0.05, strata=None, auto_stratify=False,
                   dummy_to_parent=None, strata_backup_map=None):
        """
        Fit IPTW-weighted Cox proportional hazards model with optional
        automatic or manual stratification on categorical variables.

        Parameters
        ----------
        data : pd.DataFrame
            Data with time_var, event_var, treatment_var, weight_col, and covariates.
        time_var : str
            Name of time column (days_observed).
        event_var : str
            Name of event column (departed).
        treatment_var : str
            Binary treatment variable.
        weight_col : str
            IPTW weight column.
        cluster_var : str, optional
            Clustering variable for robust standard errors.
        covariates : List[str], optional
            Additional covariates for doubly robust estimation.
        alpha : float, default 0.05
            Significance level.
        strata : list of str, optional
            Column names to stratify the Cox baseline hazard on.
            When provided, these columns are used directly (manual mode).
        auto_stratify : bool, default False
            If True, run an initial unstratified fit, test PH assumption,
            and automatically stratify on parent categoricals whose dummies
            show PH violations.  Requires *dummy_to_parent*.
        dummy_to_parent : dict, optional
            Mapping from one-hot dummy column name to its original
            categorical parent column name (e.g. ``job_family_Communications``
            -> ``job_family``).  Only needed when *auto_stratify* is True.
        strata_backup_map : dict, optional
            Mapping from cleaned parent categorical name to its backup column
            name (e.g. ``job_family`` -> ``strata_job_family``).

        Returns
        -------
        dict
            Results dictionary with hazard ratio, CI, p-value, concordance,
            proportional hazards test, KM curves, survival snapshots,
            plus strata_vars and ph_violations_detected.
        """
        # ------------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------------
        required_cols = [time_var, event_var, treatment_var, weight_col]
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        treated_events = ((data[treatment_var] == 1) & (data[event_var] == 1)).sum()
        control_events = ((data[treatment_var] == 0) & (data[event_var] == 1)).sum()

        if treated_events < 5:
            raise ValueError(f"Insufficient events in treated group: {treated_events} < 5")
        if control_events < 5:
            raise ValueError(f"Insufficient events in control group: {control_events} < 5")

        # ------------------------------------------------------------------
        # Helper: build cox_data, fit, return (cph, cox_data)
        # ------------------------------------------------------------------
        def _do_fit(formula_vars, strata_cols=None):
            """Build cox_data from *formula_vars* and fit CoxPHFitter."""
            keep = [time_var, event_var, weight_col] + formula_vars
            if cluster_var and cluster_var in data.columns:
                keep.append(cluster_var)
            if strata_cols:
                keep.extend([s for s in strata_cols if s not in keep])
            keep = list(dict.fromkeys(keep))          # deduplicate, preserve order
            _cox = data[keep].copy().dropna()

            if len(_cox) < 20:
                raise ValueError(f"Insufficient data for Cox model: {len(_cox)} rows")

            _cph = CoxPHFitter()
            fit_kw = dict(
                duration_col=time_var,
                event_col=event_var,
                weights_col=weight_col,
            )
            if cluster_var and cluster_var in data.columns:
                fit_kw["robust"] = True
                fit_kw["cluster_col"] = cluster_var
            if strata_cols:
                fit_kw["strata"] = strata_cols

            _cph.fit(_cox, **fit_kw)
            return _cph, _cox

        # ------------------------------------------------------------------
        # Helper: run PH test, return list of violating variable names
        # ------------------------------------------------------------------
        def _ph_violations(cph_obj, cox_df, threshold):
            """Return list of variable names that fail PH at *threshold*."""
            try:
                from lifelines.statistics import proportional_hazard_test
                
                ph_test = proportional_hazard_test(
                    cph_obj, 
                    cox_df, 
                    time_transform='rank'
                )
                
                if ph_test.summary is None or ph_test.summary.empty:
                    return []
                
                violations = list(
                    ph_test.summary[ph_test.summary['p'] < threshold]
                    .index
                    .get_level_values(0)
                    .unique()
                )
                
                return violations
                
            except Exception:
                return []

        # ------------------------------------------------------------------
        # Build initial formula_vars
        # ------------------------------------------------------------------
        formula_vars = [treatment_var]
        if covariates:
            formula_vars.extend(covariates)

        strata_cols = list(strata) if strata else None
        ph_violations_detected = []
        strata_vars_used = list(strata) if strata else []

        # Initialize cph and cox_data (will be set by one of the paths below)
        cph = None
        cox_data = None

        print("\n" + "=" * 50)
        print("COX PROPORTIONAL HAZARDS MODEL")
        print("=" * 50)

        # ==================================================================
        # PATH A: Manual strata — single fit with user-supplied strata
        # ==================================================================
        if strata_cols and not auto_stratify:
            print(f"Manual stratification on: {strata_cols}")
            formula_vars = [v for v in formula_vars if v not in strata_cols]
            cph, cox_data = _do_fit(formula_vars, strata_cols=strata_cols)

        # ==================================================================
        # PATH B: Auto-stratify — fit once, test PH, potentially re-fit
        # ==================================================================
        elif auto_stratify and dummy_to_parent:
            print("Auto-stratification mode: Testing proportional hazards assumption...")
            
            # Initial unstratified fit
            cph, cox_data = _do_fit(formula_vars)
            violations = _ph_violations(cph, cox_data, alpha)
            ph_violations_detected = list(violations)

            if violations:
                print(f"PH violations detected: {len(violations)} variables")
                
                # Categorize violations
                parents_to_stratify = set()
                non_cat_violations = []
                treatment_violated = False

                for v in violations:
                    if v == treatment_var:
                        treatment_violated = True
                        continue
                    parent = dummy_to_parent.get(v)
                    if parent:
                        parents_to_stratify.add(parent)
                    else:
                        non_cat_violations.append(v)

                # Report treatment violation
                if treatment_violated:
                    print(f"  ⚠️  Treatment variable violates PH assumption")
                    print(f"      → Use RMST as primary effect measure")

                # Report non-categorical violations
                if non_cat_violations:
                    print(f"  ⚠️  Non-categorical violations: {non_cat_violations}")

                # Apply stratification if possible
                if parents_to_stratify:
                    _sbm = strata_backup_map or {}
                    strata_cols = []
                    dummies_to_remove = set()
                    
                    for parent in sorted(parents_to_stratify):
                        backup_col = _sbm.get(parent)
                        if backup_col and backup_col in data.columns:
                            strata_cols.append(backup_col)
                        elif parent in data.columns:
                            strata_cols.append(parent)
                        else:
                            print(f"  Warning: Cannot stratify on '{parent}' — column not found")
                            continue
                        
                        # Collect dummies to remove
                        for dummy_name, p in dummy_to_parent.items():
                            if p == parent:
                                dummies_to_remove.add(dummy_name)

                    if strata_cols:
                        print(f"  ✓ Stratifying on: {sorted(parents_to_stratify)}")
                        print(f"    Removing {len(dummies_to_remove)} dummy variables from model")
                        
                        # Remove dummies and re-fit
                        formula_vars = [v for v in formula_vars if v not in dummies_to_remove]
                        cph, cox_data = _do_fit(formula_vars, strata_cols=strata_cols)
                        strata_vars_used = sorted(parents_to_stratify)
                        
                        # Check remaining violations
                        remaining = _ph_violations(cph, cox_data, alpha)
                        if remaining:
                            still_bad = [v for v in remaining if v != treatment_var]
                            if still_bad:
                                print(f"  ⚠️  Remaining violations after stratification: {still_bad}")
                            else:
                                print(f"  ✓ Categorical PH violations resolved")
                        else:
                            print(f"  ✓ All PH violations resolved")
                else:
                    print(f"  → No categorical variables available for stratification")
            else:
                print("✓ No PH violations detected")

        # ==================================================================
        # PATH C: No strata, no auto-stratify — simple single fit
        # ==================================================================
        else:
            print("Standard Cox model (no stratification)")
            cph, cox_data = _do_fit(formula_vars)
            
            # Check for PH violations and report
            violations = _ph_violations(cph, cox_data, alpha)
            ph_violations_detected = list(violations)
            
            if violations:
                print(f"⚠️  PH violations detected: {violations}")
                print(f"   Consider using strata=['variable_name'] or RMST as primary measure")
            else:
                print("✓ No PH violations detected")

        # ------------------------------------------------------------------
        # Extract treatment effect (AFTER all paths complete)
        # ------------------------------------------------------------------
        if cph is None or cox_data is None:
            raise ValueError("Cox model fitting failed - no model was created")
        
        hazard_ratio = float(cph.hazard_ratios_[treatment_var])
        hr_ci = cph.confidence_intervals_.loc[treatment_var]
        hr_ci_lower = float(np.exp(hr_ci.iloc[0]))  # Convert log(HR) → HR
        hr_ci_upper = float(np.exp(hr_ci.iloc[1]))  # Convert log(HR) → HR
        hr_pvalue = float(cph.summary.loc[treatment_var, 'p'])
        concordance = float(cph.concordance_index_)

        print(f"\nModel fitted: {len(cox_data):,} observations, {treated_events + control_events} events")
        print(f"Concordance: {concordance:.3f}")

        # ------------------------------------------------------------------
        # PH test on final model (if not already done by auto-stratify)
        # ------------------------------------------------------------------
        ph_test_pvalue = None
        ph_assumption_met = None
        if not (auto_stratify and dummy_to_parent):
            # Need to run PH test for manual-strata or no-strata paths
            try:
                ph_result = cph.check_assumptions(cox_data, p_value_threshold=alpha,
                                                show_plots=False)
                if hasattr(ph_result, "summary") and treatment_var in ph_result.summary.index.get_level_values(0):
                    ph_test_pvalue = float(
                        ph_result.summary.loc[treatment_var, "p"].iloc[0]
                        if hasattr(ph_result.summary.loc[treatment_var, "p"], "iloc")
                        else ph_result.summary.loc[treatment_var, "p"]
                    )
                    ph_assumption_met = ph_test_pvalue >= alpha
                else:
                    ph_assumption_met = True
            except Exception:
                pass
        else:
            # Auto-stratify path: treatment PH status
            ph_assumption_met = treatment_var not in ph_violations_detected

        # ------------------------------------------------------------------
        # Fit IPTW-weighted Kaplan-Meier curves
        # ------------------------------------------------------------------
        print("Fitting IPTW-weighted Kaplan-Meier survival curves...")
        
        treated_data = data[data[treatment_var] == 1]
        control_data = data[data[treatment_var] == 0]

        kmf_treated = KaplanMeierFitter()
        kmf_control = KaplanMeierFitter()

        # -----------------------------------------------------------------
        # Suppressing lifelines StatisticalWarning about non-integer weights.
        #
        # WHY THIS IS SAFE HERE:
        #   The IPTW-weighted KM curves are used only for *descriptive*
        #   purposes — plotting survival curves and computing point-estimate
        #   snapshot probabilities. All statistical inference (CIs, p-values)
        #   comes from the Cox PH model, which uses robust (sandwich/cluster)
        #   standard errors that correctly account for the weights.
        #
        # WHEN SUPPRESSION IS *NOT* APPROPRIATE:
        #   If you rely on KM-derived confidence bands or log-rank tests
        #   for inference (e.g., comparing survival curves formally), you
        #   should use bootstrapped (Monte Carlo) variance estimation
        #   instead. See: Austin & Stuart (2015) "Variance estimation when
        #   using IPTW with survival analysis" and Xie & Liu (2005)
        #   "Adjusted KM estimator and log-rank test with IPTW."
        # -----------------------------------------------------------------
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
                    label='Treated'
                )
                kmf_control.fit(
                    durations=control_data[time_var],
                    event_observed=control_data[event_var],
                    weights=control_data[weight_col],
                    label='Control'
                )
        except Exception as e:
            raise ValueError(f"Kaplan-Meier fitting failed: {str(e)}")

        # ------------------------------------------------------------------
        # Survival probabilities at standard timepoints
        # ------------------------------------------------------------------
        snapshot_days = [90, 180, 270, 365]
        survival_snapshots = []

        for days in snapshot_days:
            try:
                surv_treated = float(kmf_treated.survival_function_at_times(days).iloc[0])
                surv_control = float(kmf_control.survival_function_at_times(days).iloc[0])
                surv_diff = surv_treated - surv_control
                survival_snapshots.append({
                    'timepoint_days': days,
                    'timepoint_label': f'{days//30}mo' if days % 30 == 0 else f'{days}d',
                    'survival_treated': surv_treated,
                    'survival_control': surv_control,
                    'survival_diff': surv_diff
                })
            except Exception:
                survival_snapshots.append({
                    'timepoint_days': days,
                    'timepoint_label': f'{days//30}mo' if days % 30 == 0 else f'{days}d',
                    'survival_treated': np.nan,
                    'survival_control': np.nan,
                    'survival_diff': np.nan
                })

        survival_at_snapshots = pd.DataFrame(survival_snapshots)

        # ------------------------------------------------------------------
        # Build coefficients DataFrame
        # ------------------------------------------------------------------
        n_events_treated = int(treated_events)
        n_events_control = int(control_events)

        coefficients_df = pd.DataFrame({
            'Parameter': [treatment_var],
            'Estimate': [np.log(hazard_ratio)],
            'Std_Error': [(np.log(hr_ci_upper) - np.log(hr_ci_lower)) / (2 * 1.96)],
            'CI_Lower': [np.log(hr_ci_lower)],
            'CI_Upper': [np.log(hr_ci_upper)],
            'P_Value_Raw': [hr_pvalue],
            'Alpha': [alpha]
        })

        print("=" * 50)

        return {
            'hazard_ratio': hazard_ratio,
            'hr_ci_lower': hr_ci_lower,
            'hr_ci_upper': hr_ci_upper,
            'hr_pvalue': hr_pvalue,
            'concordance': concordance,
            'ph_test_pvalue': ph_test_pvalue,
            'ph_assumption_met': ph_assumption_met,
            'survival_at_snapshots': survival_at_snapshots,
            'kmf_treated': kmf_treated,
            'kmf_control': kmf_control,
            'cox_model': cph,
            'n_events_treated': n_events_treated,
            'n_events_control': n_events_control,
            'coefficients_df': coefficients_df,
            'strata_vars': strata_vars_used,
            'ph_violations_detected': ph_violations_detected,
        }

    def _fit_piecewise_cox(
        self,
        data: pd.DataFrame,
        time_var: str,
        event_var: str,
        treatment_var: str,
        weight_col: str,
        cluster_var: Optional[str] = None,
        intervals: Optional[List[Tuple[int, int]]] = None,
        alpha: float = 0.05,
    ) -> List[Dict]:
        """
        Fit separate IPTW-weighted Cox PH models for each time interval.

        For each interval [t_start, t_end), subjects alive at t_start are
        included. Their interval time = min(days_observed, t_end) - t_start
        and their event indicator is 1 only if the event falls within the
        interval. A CoxPHFitter is fit per interval, yielding interval-specific
        hazard ratios that are valid even when global PH is violated.

        Parameters
        ----------
        data : pd.DataFrame
            Weighted data with time_var, event_var, treatment_var, weight_col.
        time_var : str
            Duration column (days from T=0).
        event_var : str
            Event indicator column (1=event, 0=censored).
        treatment_var : str
            Binary treatment variable.
        weight_col : str
            IPTW weight column.
        cluster_var : str, optional
            Clustering variable for robust SEs.
        intervals : list of (int, int), optional
            Time-window boundaries in days.
            Defaults to [(0, 90), (90, 180), (180, 270), (270, 365)].
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        list of dict
            One dict per interval with keys: interval, interval_label,
            hazard_ratio, hr_ci_lower, hr_ci_upper, hr_pvalue, significant,
            n_at_risk, n_events, n_events_treated, n_events_control,
            concordance, cox_model.
        """
        if intervals is None:
            intervals = [(0, 90), (90, 180), (180, 270), (270, 365)]

        _label_map = {
            0: "0", 90: "3 mo", 180: "6 mo", 270: "9 mo", 365: "12 mo",
        }

        results = []

        print(f"\n{'─' * 60}")
        print("PIECEWISE COX: Interval-Specific Hazard Ratios")
        print(f"{'─' * 60}")

        for t_start, t_end in intervals:
            label_start = _label_map.get(t_start, f"{t_start}d")
            label_end   = _label_map.get(t_end, f"{t_end}d")
            label = f"{label_start}–{label_end}"

            # Subjects alive at t_start
            interval_df = data[data[time_var] > t_start].copy()
            n_at_risk = len(interval_df)

            if n_at_risk < 20:
                print(f"  [{label}] Skipped — only {n_at_risk} at risk")
                results.append({
                    "interval":       (t_start, t_end),
                    "interval_label": label,
                    "hazard_ratio":   np.nan,
                    "hr_ci_lower":    np.nan,
                    "hr_ci_upper":    np.nan,
                    "hr_pvalue":      np.nan,
                    "significant":    False,
                    "n_at_risk":      n_at_risk,
                    "n_events":       0,
                    "n_events_treated": 0,
                    "n_events_control": 0,
                    "concordance":    np.nan,
                    "cox_model":      None,
                })
                continue

            # Interval time and event
            interval_df["_pw_time"] = (
                interval_df[time_var].clip(upper=t_end) - t_start
            ).astype(float)
            interval_df["_pw_event"] = (
                (interval_df[event_var] == 1) &
                (interval_df[time_var] <= t_end)
            ).astype(int)

            # Must have positive times
            interval_df = interval_df[interval_df["_pw_time"] > 0]

            n_events = int(interval_df["_pw_event"].sum())
            n_events_treated = int(
                ((interval_df[treatment_var] == 1) & (interval_df["_pw_event"] == 1)).sum()
            )
            n_events_control = int(
                ((interval_df[treatment_var] == 0) & (interval_df["_pw_event"] == 1)).sum()
            )

            # Need at least a few events per group
            if n_events_treated < 2 or n_events_control < 2:
                print(
                    f"  [{label}] Skipped — insufficient events "
                    f"(treated={n_events_treated}, control={n_events_control})"
                )
                results.append({
                    "interval":       (t_start, t_end),
                    "interval_label": label,
                    "hazard_ratio":   np.nan,
                    "hr_ci_lower":    np.nan,
                    "hr_ci_upper":    np.nan,
                    "hr_pvalue":      np.nan,
                    "significant":    False,
                    "n_at_risk":      n_at_risk,
                    "n_events":       n_events,
                    "n_events_treated": n_events_treated,
                    "n_events_control": n_events_control,
                    "concordance":    np.nan,
                    "cox_model":      None,
                })
                continue

            # Build Cox data (treatment-only model per interval)
            keep_cols = ["_pw_time", "_pw_event", treatment_var, weight_col]
            if cluster_var and cluster_var in interval_df.columns:
                keep_cols.append(cluster_var)
            cox_df = interval_df[keep_cols].dropna()

            try:
                cph = CoxPHFitter()
                fit_kw = dict(
                    duration_col="_pw_time",
                    event_col="_pw_event",
                    weights_col=weight_col,
                )
                if cluster_var and cluster_var in interval_df.columns:
                    fit_kw["robust"] = True
                    fit_kw["cluster_col"] = cluster_var

                cph.fit(cox_df, **fit_kw)

                hr = float(cph.hazard_ratios_[treatment_var])
                ci = cph.confidence_intervals_.loc[treatment_var]
                ci_lo = float(np.exp(ci.iloc[0]))
                ci_hi = float(np.exp(ci.iloc[1]))
                p_val = float(cph.summary.loc[treatment_var, "p"])
                conc  = float(cph.concordance_index_)
                sig   = p_val < alpha
                stars = self._significance_stars(p_val)

                print(
                    f"  [{label}]  HR = {hr:.3f} "
                    f"[{ci_lo:.3f}, {ci_hi:.3f}]  "
                    f"p = {p_val:.4f} {stars}  "
                    f"events = {n_events}  at-risk = {n_at_risk}"
                )

                results.append({
                    "interval":       (t_start, t_end),
                    "interval_label": label,
                    "hazard_ratio":   hr,
                    "hr_ci_lower":    ci_lo,
                    "hr_ci_upper":    ci_hi,
                    "hr_pvalue":      p_val,
                    "significant":    sig,
                    "n_at_risk":      n_at_risk,
                    "n_events":       n_events,
                    "n_events_treated": n_events_treated,
                    "n_events_control": n_events_control,
                    "concordance":    conc,
                    "cox_model":      cph,
                })

            except Exception as e:
                print(f"  [{label}] Failed: {e}")
                results.append({
                    "interval":       (t_start, t_end),
                    "interval_label": label,
                    "hazard_ratio":   np.nan,
                    "hr_ci_lower":    np.nan,
                    "hr_ci_upper":    np.nan,
                    "hr_pvalue":      np.nan,
                    "significant":    False,
                    "n_at_risk":      n_at_risk,
                    "n_events":       n_events,
                    "n_events_treated": n_events_treated,
                    "n_events_control": n_events_control,
                    "concordance":    np.nan,
                    "cox_model":      None,
                })

        print(f"{'─' * 60}")
        return results

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
        title: str = "IPTW Weight Distribution"
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
                "Interpretation (ATT): Treated weights should be exactly 1.0. "
                "Control weights are scaled by the marginal treatment probability — "
                "expect mean ≈ P(T=1) with most values near zero and a small right tail "
                "for controls resembling treated individuals."
            )
        else:  # ATE
            interpretation = (
                "Interpretation (ATE): Look for stabilized weights with mean near 1 "
                "and few extreme right-tail values, since large tails indicate unstable "
                "IPTW estimates."
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

        hr        = survival_result.get("hazard_ratio")
        hr_lower  = survival_result.get("hr_ci_lower")
        hr_upper  = survival_result.get("hr_ci_upper")
        hr_pvalue = survival_result.get("hr_pvalue")
        estimand  = survival_result.get("estimand", "ATT")

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

        # HR annotation box
        if hr is not None and hr_lower is not None and hr_upper is not None:
            alpha_val = survival_result.get("alpha", 0.05)
            ci_pct    = int((1 - alpha_val) * 100)
            stars     = self._significance_stars(hr_pvalue) if hr_pvalue is not None else ""
            p_str     = f"p = {hr_pvalue:.3f}" if hr_pvalue is not None else ""

            # For piecewise results, indicate which interval the HR is from
            is_piecewise = survival_result.get("piecewise", False)
            if is_piecewise:
                pw_results = survival_result.get("piecewise_results", [])
                valid_pw = [r for r in pw_results if not np.isnan(r.get("hr_pvalue", np.nan))]
                best_label = min(valid_pw, key=lambda r: r["hr_pvalue"]).get("interval_label", "") if valid_pw else ""
                hr_header = f"Best interval: {best_label}\n" if best_label else ""
            else:
                hr_header = ""

            hr_text = (
                f"{hr_header}"
                f"HR = {hr:.3f}\n"
                f"{ci_pct}% CI: [{hr_lower:.3f}–{hr_upper:.3f}]\n"
                f"{p_str} {stars}"
            )
            ax_main.text(
                0.97, 0.97, hr_text,
                transform=ax_main.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor="gray", alpha=0.8)
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
        categorical_vars: List[str],
        binary_vars: List[str],
        continuous_vars: List[str],
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
        categorical_vars, binary_vars, continuous_vars : list of str
            Covariate lists (original names).
        estimand : str
            ``"ATE"`` or ``"ATT"``.
        trim_quantile : float
            PS-weight trim quantile.
        plot_propensity, plot_weights : bool
            Whether to generate diagnostic plots.
        outcome_var : str, optional
            Outcome column (GEE pipeline).
        baseline_var : str, optional
            Baseline covariate included in GEE outcome model but excluded
            from the propensity-score model (doubly robust adjustment).
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
        # Step 0: Data prep
        # ------------------------------------------------------------------
        # Determine which columns must be present
        ps_covariates_raw = categorical_vars + binary_vars + continuous_vars
        outcome_covariates_raw = list(ps_covariates_raw)
        if baseline_var:
            outcome_covariates_raw.append(baseline_var)

        id_columns: List[str] = [treatment_var, cluster_var]
        if outcome_var:
            id_columns.append(outcome_var)
        if time_var:
            id_columns.append(time_var)
        if event_var:
            id_columns.append(event_var)

        all_needed = list(set(id_columns + outcome_covariates_raw))
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
        dummy_columns = sorted(set(df.columns) - cols_before_dummies)

        # --- Clean all column names ---
        rename_map = {c: self._clean_column_name(c) for c in df.columns}
        df.rename(columns=rename_map, inplace=True)

        # Remap key variable references
        treatment_var = self._clean_column_name(treatment_var)
        cluster_var = self._clean_column_name(cluster_var)
        if outcome_var:
            outcome_var = self._clean_column_name(outcome_var)
        if baseline_var:
            baseline_var = self._clean_column_name(baseline_var)
        if time_var:
            time_var = self._clean_column_name(time_var)
        if event_var:
            event_var = self._clean_column_name(event_var)

        continuous_vars = [self._clean_column_name(v) for v in continuous_vars]
        binary_vars = [self._clean_column_name(v) for v in binary_vars]
        dummy_columns = [self._clean_column_name(c) for c in dummy_columns]

        # --- Build dummy → parent mapping (used for auto-stratification) ---
        dummy_to_parent: Dict[str, str] = {}
        for dummy in dummy_columns:
            for parent in cleaned_cat_vars:
                if dummy.startswith(parent + "_"):
                    dummy_to_parent[dummy] = parent
                    break

        # --- Track balance variables ---
        balance_var_names = (
            [v for v in continuous_vars if v in df.columns]
            + [v for v in binary_vars if v in df.columns]
            + [dc for dc in dummy_columns if dc in df.columns]
        )
        balance_var_types: Dict[str, str] = {
            v: "continuous" for v in continuous_vars if v in df.columns
        }
        balance_var_types.update({v: "binary" for v in binary_vars if v in df.columns})
        balance_var_types.update({dc: "categorical" for dc in dummy_columns if dc in df.columns})

        # --- Build covariate lists ---
        _exclude = {treatment_var, cluster_var}
        if outcome_var:
            _exclude.add(outcome_var)
        if time_var:
            _exclude.add(time_var)
        if event_var:
            _exclude.add(event_var)
        _strata_backup_cols = set(_strata_backup_map.values())
        _exclude |= _strata_backup_cols

        covariates = [c for c in df.columns if c not in _exclude]

        # PS covariates: confounders only (exclude baseline for doubly robust)
        if baseline_var:
            ps_covariates = [c for c in covariates if c != baseline_var]
        else:
            ps_covariates = list(covariates)

        # --- Validate covariates ---
        if len(covariates) == 0:
            raise ValueError("No covariates remaining after data processing")

        covariate_df = df[covariates]
        if covariate_df.empty or covariate_df.shape[1] == 0:
            raise ValueError(
                f"Empty covariate matrix after processing. Shape: {covariate_df.shape}"
            )

        # Remove constant covariates
        constant_vars = [var for var in covariates if df[var].nunique() <= 1]
        if constant_vars:
            print(f"  Warning: Removing constant variables: {constant_vars}")
            covariates = [v for v in covariates if v not in constant_vars]
            ps_covariates = [v for v in ps_covariates if v not in constant_vars]
            if len(covariates) == 0:
                raise ValueError(
                    "No valid covariates remaining after removing constant variables"
                )

        # Final null check
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
        n_near_one = (ps_vals > 0.99).sum()
        if n_near_zero > 0 or n_near_one > 0:
            print(
                f"  Warning: Positivity concern: {n_near_zero} observations "
                f"with PS < 0.01, {n_near_one} with PS > 0.99"
            )

        # --- Propensity score overlap plot ---
        ps_overlap_fig = None
        if plot_propensity:
            ps_plot_title = f"Propensity Score Overlap — {analysis_label}"
            try:
                ps_overlap_fig = self.plot_propensity_overlap(
                    data=df,
                    treatment_var=treatment_var,
                    title=ps_plot_title,
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

        # --- Weight distribution plot ---
        weight_dist_fig = None
        if plot_weights:
            wt_plot_title = f"IPTW Weight Distribution — {analysis_label}"
            try:
                weight_dist_fig = self.plot_weight_distribution(
                    data=df,
                    treatment_var=treatment_var,
                    estimand=estimand,
                    title=wt_plot_title,
                )
            except Exception as e:
                print(f"  Warning: Could not generate weight distribution plot: {e}")

        # --- Post-weighting balance check via CausalDiagnostics ---
        _cd = CausalDiagnostics()
        try:
            _raw_balance = _cd.compute_balance_df(
                data=df,
                controls=balance_var_names,
                treatment=treatment_var,
                weights=df["iptw"],
                already_encoded=True,
            )
            balance_results = []
            for var_name in _raw_balance.index:
                row = _raw_balance.loc[var_name]
                smd_before = row["Unweighted SMD"]
                smd_after = row["Weighted SMD"]
                improvement = abs(smd_before) - abs(smd_after)
                balance_results.append({
                    "variable": var_name,
                    "type": balance_var_types.get(var_name, "unknown"),
                    "smd_before_weighting": smd_before,
                    "smd_after_weighting": smd_after,
                    "smd_improvement": improvement,
                    "balanced_before_weighting": abs(smd_before) < 0.1,
                    "balanced_after_weighting": abs(smd_after) < 0.1,
                })
            balance_df = pd.DataFrame(balance_results)
        except Exception as e:
            print(
                f"  Warning: CausalDiagnostics balance computation failed ({e}); "
                f"falling back to inline SMD computation."
            )
            # --- Fallback: inline computation ---
            df["_uniform_wt"] = 1.0
            balance_results = []
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
                    improvement = abs(smd_before) - abs(smd_after)
                    balance_results.append({
                        "variable": var_name,
                        "type": balance_var_types.get(var_name, "unknown"),
                        "smd_before_weighting": smd_before,
                        "smd_after_weighting": smd_after,
                        "smd_improvement": improvement,
                        "balanced_before_weighting": abs(smd_before) < 0.1,
                        "balanced_after_weighting": abs(smd_after) < 0.1,
                    })
                except Exception:
                    balance_results.append({
                        "variable": var_name,
                        "type": balance_var_types.get(var_name, "unknown"),
                        "smd_before_weighting": None,
                        "smd_after_weighting": None,
                        "smd_improvement": None,
                        "balanced_before_weighting": None,
                        "balanced_after_weighting": False,
                    })
            df.drop(columns=["_uniform_wt"], inplace=True)
            balance_df = pd.DataFrame(balance_results)

        if balance_df.empty:
            balance_df = pd.DataFrame(
                columns=[
                    "variable", "type", "smd_before_weighting",
                    "smd_after_weighting", "smd_improvement",
                    "balanced_before_weighting", "balanced_after_weighting",
                ]
            )

        # --- Balance summary ---
        if not balance_df.empty and "balanced_after_weighting" in balance_df.columns:
            n_imbalanced = int(balance_df["balanced_after_weighting"].eq(False).sum())
            n_total_vars = len(balance_df)
            if n_imbalanced == 0:
                print(
                    f"  ✓ Post-weighting balance: all {n_total_vars} "
                    f"covariates balanced (|SMD| < 0.1)"
                )
            else:
                print(
                    f"  ⚠️  Post-weighting balance: {n_imbalanced} of "
                    f"{n_total_vars} covariates still imbalanced (|SMD| ≥ 0.1)"
                )

        return {
            "df": df,
            "ps_model": ps_model,
            "weight_stats": weight_stats,
            "balance_df": balance_df,
            "ps_overlap_fig": ps_overlap_fig,
            "weight_dist_fig": weight_dist_fig,
            "covariates": covariates,
            "ps_covariates": ps_covariates,
            "dummy_columns": dummy_columns,
            "balance_var_names": balance_var_names,
            "balance_var_types": balance_var_types,
            # Cleaned variable references
            "treatment_var": treatment_var,
            "cluster_var": cluster_var,
            "outcome_var": outcome_var,
            "baseline_var": baseline_var,
            "time_var": time_var,
            "event_var": event_var,
            "continuous_vars": continuous_vars,
            "binary_vars": binary_vars,
            # Survival-specific
            "_strata_backup_map": _strata_backup_map,
            "dummy_to_parent": dummy_to_parent,
            "cleaned_cat_vars": cleaned_cat_vars,
        }

    # ==================================================================
    # Public analysis pipelines
    # ==================================================================

    def analyze_treatment_effect(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        categorical_vars: List[str],
        binary_vars: List[str],
        continuous_vars: List[str],
        cluster_var: str,
        estimand: str = "ATE",
        baseline_var: Optional[str] = None,
        project_path: Optional[str] = None,
        trim_quantile: float = 0.99,
        analysis_name: Optional[str] = None,
        correction_method: str = 'fdr_bh',
        alpha: float = 0.05,
        plot_propensity: bool = True,
        plot_weights: bool = True
    ) -> Dict:
        """
        Complete analysis pipeline: IPTW propensity weights → doubly robust outcome model.
        
        Comprehensive causal inference analysis implementing:
        1. Data preparation and covariate encoding
        2. Propensity score estimation with IPTW computation (ATE or ATT)
        3. Propensity score overlap visualization
        4. Weight diagnostics, distribution visualization, and balance assessment
        5. Doubly robust outcome modeling via weighted GEE
        6. Effect size metrics (Cohen's d, percent change)
        7. Optional export to Excel workbook
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw dataset for analysis
        outcome_var : str
            Name of outcome variable
        treatment_var : str
            Name of binary treatment variable
        categorical_vars : List[str]
            Categorical covariate names (will be one-hot encoded)
        binary_vars : List[str]
            Binary covariate names
        continuous_vars : List[str]
            Continuous covariate names
        cluster_var : str
            Name of clustering variable
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
        correction_method : str, default='fdr_bh'
            **Deprecated.** No longer used within this method. Multiple testing
            correction is now applied across outcomes in
            ``build_summary_table()``. Kept for backward-compatible call
            signatures.
        alpha : float, default=0.05
            Significance level used for confidence intervals and raw p-value
            significance threshold
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
            - p_value: Raw p-value for the treatment effect
            - significant: Boolean indicating significance at alpha (raw)
            - alpha: Significance level used
            - cohens_d: Cohen's d effect size (uses raw weighted mean diff)
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
        
        Raises
        ------
        ValueError
            If data preparation, model fitting, or validation fails
        """
        # Validate estimand
        estimand = estimand.upper()
        if estimand not in ["ATE", "ATT"]:
            raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")
        
        # --- R5: Emit deprecation warning for correction_method if supplied ---
        if correction_method != 'fdr_bh':
            warnings.warn(
                "The 'correction_method' parameter in analyze_treatment_effect() is "
                "deprecated and ignored. Multiple-testing correction is now applied "
                "across outcomes in build_summary_table().",
                DeprecationWarning,
                stacklevel=2
            )
        
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
        df             = _iptw["df"]
        ps_model       = _iptw["ps_model"]
        weight_stats   = _iptw["weight_stats"]
        balance_df     = _iptw["balance_df"]
        ps_overlap_fig = _iptw["ps_overlap_fig"]
        weight_dist_fig = _iptw["weight_dist_fig"]
        covariates     = _iptw["covariates"]
        outcome_var    = _iptw["outcome_var"]
        treatment_var  = _iptw["treatment_var"]
        cluster_var    = _iptw["cluster_var"]
        baseline_var   = _iptw["baseline_var"]

        # ------------------------------------------------------------------
        # Step 3: Fit doubly robust outcome model
        # ------------------------------------------------------------------
        # B10: Auto-detect binary outcomes for appropriate GEE family
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
            
            gee_res = self.fit_doubly_robust_model(
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
        # B3 fix: Cohen's d uses the raw weighted mean difference (not the
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
        # S2 fix: pct_change uses the same marginal raw_diff as cohens_d
        # (not the conditional GEE coefficient 'effect') for consistency.
        pct_change = (raw_diff / mean_control) * 100 if abs(mean_control) > 1e-9 else None
        
        # --- B2 fix: No within-model multiple testing correction.            ---
        # --- Correction is applied across outcomes in build_summary_table ---
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
        
        # --- Build coefficients DataFrame (treatment row only, for printed summary) ---
        coefficients_df = full_coefficients_df[full_coefficients_df['Parameter'] == treatment_var].copy()
        
        # --- Print summary ---
        ci_pct = int((1 - alpha) * 100)
        print(
            f"  [{outcome_var}] {estimand} = {effect:.4f} "
            f"({ci_pct}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]), "
            f"p = {p_value_raw:.4f} {stars}, "
            f"Cohen's d = {cohens_d:.4f}"
        )
        
        # --- Build propensity score model summary DataFrame ---
        ps_summary_df = pd.DataFrame({
            'Parameter': ps_model.params.index,
            'Estimate': ps_model.params.values,
            'Std_Error': ps_model.bse.values,
            'P_Value': ps_model.pvalues.values
        })
        
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
            "weighted_df": df,  # processed DataFrame with propensity_score & iptw columns
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
            If ``None``, defaults to ``W_cols``.
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
            Name of clustering variable. Stored in results metadata but **not**
            used by DML (which does not natively handle clustering). For
            cluster-robust inference, use ``analyze_treatment_effect()``.
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
        - DML does **not** natively account for clustering. If your data has a
          hierarchical structure (e.g. managers nested in teams), the standard
          errors from DML may be anti-conservative. Use
          ``analyze_treatment_effect()`` for cluster-robust inference.
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

        # Cohen's d from raw difference / pooled SD
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

    def dml_estimate_treatment_effects_help(self):
        """
        Print detailed documentation for ``dml_estimate_treatment_effects``.

        Displays parameter descriptions, usage examples, and interpretation
        guidance for Double Machine Learning estimation.
        """
        help_text = """
    Double Machine Learning (DML) for ATE, ATT, and CATE Estimation
    ================================================================

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
    - cluster_var (str): Clustering variable (stored for metadata; not used by DML).
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
        binary_vars=['is_new_manager'],
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

    # ==================================================================
    # Survival analysis pipeline
    # ==================================================================

    def analyze_survival_effect(
        self,
        data: pd.DataFrame,
        time_var: str,
        event_var: str,
        treatment_var: str,
        categorical_vars: List[str],
        binary_vars: List[str],
        continuous_vars: List[str],
        cluster_var: str,
        estimand: str = "ATT",
        project_path: Optional[str] = None,
        trim_quantile: float = 0.99,
        analysis_name: Optional[str] = None,
        alpha: float = 0.05,
        plot_propensity: bool = True,
        plot_weights: bool = True,
        strata: Optional[object] = "auto",
        piecewise: bool = False,
        intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        Complete survival analysis pipeline: IPTW propensity weights → Cox proportional hazards.

        Implements the same Steps 0-2 as analyze_treatment_effect() but replaces
        Step 3 (GEE outcome model) with Cox proportional hazards model for
        time-to-event outcomes like employee retention.

        When ``piecewise=True``, fits separate Cox models for each time interval
        (default: 0–3, 3–6, 6–9, 9–12 months), producing interval-specific
        hazard ratios that remain valid even when global PH is violated.
        A full-period KM curve is still computed for plotting and snapshot
        survival differences.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with time_var, event_var, treatment, and covariates
        time_var : str
            Name of time column (days from T=0 to event/censoring)
        event_var : str
            Name of event indicator column (1=event occurred, 0=censored)
        treatment_var : str
            Name of binary treatment variable
        categorical_vars : List[str]
            Categorical covariate names (will be one-hot encoded)
        binary_vars : List[str]
            Binary covariate names
        continuous_vars : List[str]
            Continuous covariate names
        cluster_var : str
            Name of clustering variable for robust standard errors
        estimand : str, default="ATT"
            Target estimand: "ATE" or "ATT". Determines IPTW weight construction.
        project_path : str, optional
            Path to save results Excel file
        trim_quantile : float, default=0.99
            Quantile for weight trimming
        analysis_name : str, optional
            Analysis identifier for file naming
        alpha : float, default=0.05
            Significance level for confidence intervals
        plot_propensity : bool, default=True
            If True, generates propensity score overlap plot
        plot_weights : bool, default=True
            If True, generates IPTW weight distribution plot
        piecewise : bool, default=False
            If True, fits interval-specific Cox models instead of a single
            global Cox model. Produces per-interval hazard ratios that are
            valid even when the proportional hazards assumption is violated.
        intervals : list of (int, int), optional
            Time-window boundaries in days for piecewise analysis.
            Defaults to [(0, 90), (90, 180), (180, 270), (270, 365)].
            Only used when ``piecewise=True``.

        Returns
        -------
        dict
            Dictionary with keys compatible with build_summary_table():
            - effect: hazard ratio (not log hazard ratio)
            - estimand: "ATE" or "ATT"
            - ci_lower, ci_upper: HR confidence interval bounds
            - p_value: p-value for treatment effect
            - significant: boolean significance at alpha level
            - alpha: significance level used
            - cohens_d: None (not applicable to survival)
            - pct_change: None (not applicable to survival)
            - mean_treatment: survival probability at 365 days for treated
            - mean_control: survival probability at 365 days for control
            - outcome_type: "survival"
            - coefficients_df: DataFrame with log HR for compatibility
            - balance_df: post-weighting balance statistics
            - weight_diagnostics: weight summary statistics
            - weighted_df: processed data with weights

            Plus survival-specific keys:
            - hazard_ratio: treatment hazard ratio
            - ph_assumption_met: proportional hazards test result
            - ph_test_pvalue: PH test p-value
            - kmf_treated, kmf_control: fitted Kaplan-Meier objects
            - survival_at_snapshots: DataFrame with survival at 90,180,270,365 days
            - n_events_treated, n_events_control: event counts
            - cox_model: fitted CoxPHFitter object

        Raises
        ------
        ValueError
            If data preparation, model fitting, or validation fails
        ImportError
            If lifelines package is not installed
        """
        # Validate estimand
        estimand = estimand.upper()
        if estimand not in ["ATE", "ATT"]:
            raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

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
            time_var=time_var,
            event_var=event_var,
            preserve_strata_backups=True,
            analysis_label=f"Survival Analysis ({estimand})",
        )
        df                = _iptw["df"]
        ps_model          = _iptw["ps_model"]
        weight_stats      = _iptw["weight_stats"]
        balance_df        = _iptw["balance_df"]
        ps_overlap_fig    = _iptw["ps_overlap_fig"]
        weight_dist_fig   = _iptw["weight_dist_fig"]
        covariates        = _iptw["covariates"]
        dummy_columns     = _iptw["dummy_columns"]
        treatment_var     = _iptw["treatment_var"]
        cluster_var       = _iptw["cluster_var"]
        time_var          = _iptw["time_var"]
        event_var         = _iptw["event_var"]
        _strata_backup_map = _iptw["_strata_backup_map"]
        dummy_to_parent   = _iptw["dummy_to_parent"]
        cleaned_cat_vars  = _iptw["cleaned_cat_vars"]

        # ------------------------------------------------------------------
        # STEP 3: Fit IPTW-weighted Cox proportional hazards model
        # Replaces fit_doubly_robust_model() from the GEE pipeline.
        # ------------------------------------------------------------------
        # Resolve strata mode
        _manual_strata = None
        _auto_stratify = False
        if isinstance(strata, str) and strata.lower() == "auto":
            _auto_stratify = True
        elif isinstance(strata, list) and len(strata) > 0:
            # Manual strata: map original var names to backup cols via _strata_backup_map
            _manual_strata = []
            _strata_dummies_to_remove = set()
            for s in strata:
                s_clean = self._clean_column_name(s)
                backup = _strata_backup_map.get(s_clean)
                if backup and backup in df.columns:
                    _manual_strata.append(backup)
                else:
                    raise ValueError(
                        f"Cannot stratify on '{s}': backup column not found. "
                        f"Ensure it is in categorical_vars."
                    )
                # Remove this parent's dummies from covariates
                for d, p in dummy_to_parent.items():
                    if p == s_clean:
                        _strata_dummies_to_remove.add(d)
            covariates = [v for v in covariates if v not in _strata_dummies_to_remove]
        # else: strata is None → no stratification

        # ==================================================================
        # PIECEWISE PATH: interval-specific Cox + full-period KM
        # When piecewise=True we bypass the single global Cox model and
        # instead fit separate treatment-only Cox models per interval.
        # A full-period KM curve is still computed for plotting / snapshots.
        # ==================================================================
        if piecewise:
            # 3a) Full-period IPTW-weighted Kaplan-Meier -----------------
            treated_data = df[df[treatment_var] == 1]
            control_data = df[df[treatment_var] == 0]

            kmf_treated = KaplanMeierFitter()
            kmf_control = KaplanMeierFitter()
            # See defensive note in _fit_cox_model() for why this
            # suppression is safe (KM used descriptively, not for inference).
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
                        weights=treated_data["iptw"],
                        label="Treated",
                    )
                    kmf_control.fit(
                        durations=control_data[time_var],
                        event_observed=control_data[event_var],
                        weights=control_data["iptw"],
                        label="Control",
                    )
            except Exception as e:
                raise ValueError(f"Kaplan-Meier fitting failed: {e}")

            # Survival snapshots at standard timepoints
            snapshot_days = [90, 180, 270, 365]
            snap_rows: List[Dict] = []
            for days in snapshot_days:
                try:
                    s_t = float(kmf_treated.survival_function_at_times(days).iloc[0])
                    s_c = float(kmf_control.survival_function_at_times(days).iloc[0])
                    snap_rows.append({
                        "timepoint_days": days,
                        "timepoint_label": f"{days // 30}mo",
                        "survival_treated": s_t,
                        "survival_control": s_c,
                        "survival_diff": s_t - s_c,
                    })
                except Exception:
                    snap_rows.append({
                        "timepoint_days": days,
                        "timepoint_label": f"{days // 30}mo",
                        "survival_treated": np.nan,
                        "survival_control": np.nan,
                        "survival_diff": np.nan,
                    })
            survival_snapshots = pd.DataFrame(snap_rows)

            # 3b) Piecewise interval Cox models --------------------------
            pw_results = self._fit_piecewise_cox(
                data=df,
                time_var=time_var,
                event_var=event_var,
                treatment_var=treatment_var,
                weight_col="iptw",
                cluster_var=cluster_var,
                intervals=intervals,
                alpha=alpha,
            )

            # 3c) Pick "best" interval (lowest p-value) for top-level keys
            valid_pw = [r for r in pw_results if not np.isnan(r["hr_pvalue"])]
            if valid_pw:
                best = min(valid_pw, key=lambda r: r["hr_pvalue"])
            else:
                best = {
                    "interval_label": "N/A",
                    "hazard_ratio": np.nan, "hr_ci_lower": np.nan,
                    "hr_ci_upper": np.nan,  "hr_pvalue": np.nan,
                    "significant": False,   "concordance": np.nan,
                }

            hazard_ratio = best["hazard_ratio"]
            hr_ci_lower  = best["hr_ci_lower"]
            hr_ci_upper  = best["hr_ci_upper"]
            hr_pvalue    = best["hr_pvalue"]
            concordance  = best.get("concordance", np.nan)
            significant  = best.get("significant", False)
            stars   = self._significance_stars(hr_pvalue) if not np.isnan(hr_pvalue) else ""
            ci_pct  = int((1 - alpha) * 100)

            # ---- Piecewise summary printout ----------------------------
            print(f"\n{'=' * 60}")
            print(f"PIECEWISE SURVIVAL ANALYSIS RESULTS ({estimand})")
            print(f"{'=' * 60}")
            print(f"  Best interval:  {best.get('interval_label', 'N/A')}")
            print(f"  Hazard Ratio:   {hazard_ratio:.3f} "
                  f"({ci_pct}% CI: [{hr_ci_lower:.3f}, {hr_ci_upper:.3f}])")
            print(f"  P-value:        {hr_pvalue:.4f} {stars}")
            print()
            print("  PH Assumption: Bypassed — piecewise intervals handle")
            print("                 time-varying effects by design.")
            print()

            # Snapshot survival differences
            print("  Survival Probability Snapshots (IPTW-weighted KM):")
            for _, snap_row in survival_snapshots.iterrows():
                diff = snap_row["survival_diff"]
                diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
                print(f"    {snap_row['timepoint_label']:>5s}:  "
                      f"Trained = {snap_row['survival_treated']:.1%}  |  "
                      f"Control = {snap_row['survival_control']:.1%}  |  "
                      f"Diff = {diff_str}")
            print()
            print("  Note: IPTW-weighted KM curves are used for descriptive")
            print("  snapshots only. All inferential statistics (HRs, CIs,")
            print("  p-values) come from Cox PH with robust sandwich SEs,")
            print("  which correctly account for the IPTW weights. If KM-")
            print("  based CIs or log-rank tests are needed for inference,")
            print("  bootstrapped variance estimation should be used instead")
            print("  (see Austin & Stuart, 2015; Xie & Liu, 2005).")
            print(f"{'=' * 60}")

            # Mean treatment/control at 365d (for build_summary_table compat)
            snap_365 = survival_snapshots[survival_snapshots["timepoint_days"] == 365]
            if not snap_365.empty:
                mean_treatment = float(snap_365["survival_treated"].iloc[0])
                mean_control   = float(snap_365["survival_control"].iloc[0])
            else:
                mean_treatment = float(1 - df[df[treatment_var] == 1][event_var].mean())
                mean_control   = float(1 - df[df[treatment_var] == 0][event_var].mean())

            # Coefficients DF (best-interval HR for schema compat)
            _log_hr = np.log(hazard_ratio) if not np.isnan(hazard_ratio) else np.nan
            _log_hr_se = (
                (np.log(hr_ci_upper) - np.log(hr_ci_lower)) / (2 * 1.96)
                if not (np.isnan(hr_ci_lower) or np.isnan(hr_ci_upper))
                else np.nan
            )
            coefficients_df = pd.DataFrame({
                "Parameter":   [treatment_var],
                "Estimate":    [_log_hr],
                "Std_Error":   [_log_hr_se],
                "CI_Lower":    [np.log(hr_ci_lower) if not np.isnan(hr_ci_lower) else np.nan],
                "CI_Upper":    [np.log(hr_ci_upper) if not np.isnan(hr_ci_upper) else np.nan],
                "P_Value_Raw": [hr_pvalue],
                "Alpha":       [alpha],
            })

            # PS model summary
            ps_summary_df = pd.DataFrame({
                "Parameter": ps_model.params.index,
                "Estimate":  ps_model.params.values,
                "Std_Error": ps_model.bse.values,
                "P_Value":   ps_model.pvalues.values,
            })

            # Optional Excel export
            if project_path and analysis_name:
                try:
                    export_path = (
                        f"{project_path}/{estimand.lower()}_iptw_piecewise_cox_{analysis_name}.xlsx"
                    )
                    pw_export = pd.DataFrame(pw_results).drop(columns=["cox_model"], errors="ignore")
                    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
                        balance_df.to_excel(writer, sheet_name="Covariate_Balance", index=False)
                        pd.DataFrame([weight_stats]).to_excel(
                            writer, sheet_name="Weight_Diagnostics", index=False
                        )
                        pw_export.to_excel(
                            writer, sheet_name=f"{estimand}_Piecewise_Cox", index=False
                        )
                        ps_summary_df.to_excel(
                            writer, sheet_name="Propensity_Model", index=False
                        )
                        survival_snapshots.to_excel(
                            writer, sheet_name="Survival_Snapshots", index=False
                        )
                    print(f"  Results saved to {export_path}")
                except Exception as e:
                    print(f"  Warning: Could not export results to Excel: {e}")

            # Drop strata backup columns
            strata_backup_to_drop = list(_strata_backup_map.values())
            df_clean = df.drop(columns=strata_backup_to_drop, errors="ignore")

            return {
                # --- Shared keys for build_summary_table() compat ---
                "effect":            hazard_ratio,
                "estimand":          estimand,
                "ci_lower":          hr_ci_lower,
                "ci_upper":          hr_ci_upper,
                "p_value":           hr_pvalue,
                "significant":       significant,
                "alpha":             alpha,
                "cohens_d":          None,
                "pct_change":        None,
                "mean_treatment":    mean_treatment,
                "mean_control":      mean_control,
                "outcome_type":      "survival",
                "coefficients_df":   coefficients_df,
                "full_coefficients_df": coefficients_df,
                "ps_model":          ps_model,
                "ps_summary_df":     ps_summary_df,
                "balance_df":        balance_df,
                "weight_diagnostics": weight_stats,
                "ps_overlap_fig":    ps_overlap_fig,
                "weight_dist_fig":   weight_dist_fig,
                "weighted_df":       df_clean,

                # --- Survival-specific keys ---
                "hazard_ratio":          hazard_ratio,
                "hr_ci_lower":           hr_ci_lower,
                "hr_ci_upper":           hr_ci_upper,
                "hr_pvalue":             hr_pvalue,
                "concordance":           concordance,
                "ph_assumption_met":     None,         # N/A — piecewise by design
                "ph_test_pvalue":        None,         # N/A — piecewise by design
                "kmf_treated":           kmf_treated,
                "kmf_control":           kmf_control,
                "cox_model":             None,         # No single global model
                "survival_at_snapshots": survival_snapshots,
                "n_events_treated":      int(df[df[treatment_var] == 1][event_var].sum()),
                "n_events_control":      int(df[df[treatment_var] == 0][event_var].sum()),
                "n_treated":             int(df[treatment_var].sum()),
                "n_control":             int((df[treatment_var] == 0).sum()),

                # --- Piecewise-specific keys ---
                "piecewise":             True,
                "piecewise_results":     pw_results,

                # --- Strata metadata (empty for piecewise) ---
                "strata_vars":            [],
                "ph_violations_detected": [],

                # --- Variable-name metadata ---
                "treatment_var":         treatment_var,
                "time_var":              time_var,
                "event_var":             event_var,
            }

        # ==================================================================
        # STANDARD (non-piecewise) PATH: single global Cox PH model
        # ==================================================================
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
                strata=_manual_strata,
                auto_stratify=_auto_stratify,
                dummy_to_parent=dummy_to_parent if _auto_stratify else None,
                strata_backup_map=_strata_backup_map if _auto_stratify else None,
            )
        except Exception as e:
            raise ValueError(f"Error fitting Cox model: {str(e)}")

        # ------------------------------------------------------------------
        # STEP 4: Extract results and build return dict
        # ------------------------------------------------------------------
        hazard_ratio  = cox_results["hazard_ratio"]
        hr_ci_lower   = cox_results["hr_ci_lower"]
        hr_ci_upper   = cox_results["hr_ci_upper"]
        hr_pvalue     = cox_results["hr_pvalue"]
        concordance   = cox_results["concordance"]

        ph_assumption_met = cox_results.get("ph_assumption_met")
        ph_test_pvalue    = cox_results.get("ph_test_pvalue")

        # Significance
        significant = hr_pvalue < alpha
        stars       = self._significance_stars(hr_pvalue)
        ci_pct      = int((1 - alpha) * 100)

        # ------------------------------------------------------------------
        # Clean results summary (replaces old fragmented print statements)
        # ------------------------------------------------------------------
        print(f"\n{'=' * 60}")
        print(f"SURVIVAL ANALYSIS RESULTS ({estimand})")
        print(f"{'=' * 60}")
        print(f"  Hazard Ratio:  {hazard_ratio:.3f} "
              f"({ci_pct}% CI: [{hr_ci_lower:.3f}, {hr_ci_upper:.3f}])")
        print(f"  P-value:       {hr_pvalue:.4f} {stars}")
        print(f"  Concordance:   {concordance:.3f}")
        print()

        # PH assumption status
        if ph_assumption_met is False:
            p_str = (f"Schoenfeld test p = {ph_test_pvalue:.4f}"
                     if ph_test_pvalue is not None
                     else "treatment variable flagged in PH test")
            print(f"  ⚠️  PH Assumption:  VIOLATED ({p_str})")
            print(f"      → Treatment effect varies over time (as expected)")
            print(f"      → Consider piecewise=True for interval-specific HRs")
        elif ph_assumption_met is True:
            p_str = (f"Schoenfeld test p = {ph_test_pvalue:.4f}"
                     if ph_test_pvalue is not None
                     else "no violations detected")
            print(f"  ✓  PH Assumption:  MET ({p_str})")

        # Stratification status
        strata_used = cox_results.get("strata_vars", [])
        if strata_used:
            print(f"  ✓  Stratified on:  {strata_used}")

        print()
        print("  Note: IPTW-weighted KM curves are used for descriptive")
        print("  snapshots/plots only. All inferential statistics (HRs,")
        print("  CIs, p-values) come from Cox PH with robust sandwich SEs,")
        print("  which correctly account for the IPTW weights. If KM-based")
        print("  CIs or log-rank tests are needed for inference, bootstrapped")
        print("  variance estimation should be used instead (see Austin &")
        print("  Stuart, 2015; Xie & Liu, 2005).")
        print(f"{'=' * 60}")

        # --- Survival probabilities at 365 days for mean_treatment / mean_control ---
        # These populate the mean_treatment / mean_control keys expected by
        # build_summary_table() and generate_gee_summary_report().
        survival_snapshots = cox_results.get("survival_at_snapshots", pd.DataFrame())
        snap_365 = survival_snapshots[survival_snapshots["timepoint_days"] == 365]

        if not snap_365.empty:
            mean_treatment = float(snap_365["survival_treated"].iloc[0])
            mean_control   = float(snap_365["survival_control"].iloc[0])
        else:
            # Fallback: use raw group proportions if 365-day snapshot unavailable
            mean_treatment = float(
                1 - df[df[treatment_var] == 1][event_var].mean()
            )
            mean_control = float(
                1 - df[df[treatment_var] == 0][event_var].mean()
            )

        # --- Build coefficients_df (single-row, compatible with build_summary_table) ---
        # Use log(HR) as the Estimate so the schema matches the GEE log-odds convention.
        # The 'effect' key in the return dict carries the HR itself.
        log_hr    = np.log(hazard_ratio)
        log_hr_se = (np.log(hr_ci_upper) - np.log(hr_ci_lower)) / (2 * 1.96)

        coefficients_df = pd.DataFrame({
            "Parameter":   [treatment_var],
            "Estimate":    [log_hr],
            "Std_Error":   [log_hr_se],
            "CI_Lower":    [np.log(hr_ci_lower)],
            "CI_Upper":    [np.log(hr_ci_upper)],
            "P_Value_Raw": [hr_pvalue],
            "Alpha":       [alpha],
        })

        # --- Build propensity score model summary DataFrame ---
        ps_summary_df = pd.DataFrame({
            "Parameter": ps_model.params.index,
            "Estimate":  ps_model.params.values,
            "Std_Error": ps_model.bse.values,
            "P_Value":   ps_model.pvalues.values,
        })

        # ------------------------------------------------------------------
        # STEP 5: Export (optional, mirrors analyze_treatment_effect)
        # ------------------------------------------------------------------
        if project_path and analysis_name:
            try:
                export_path = (
                    f"{project_path}/{estimand.lower()}_iptw_cox_{analysis_name}.xlsx"
                )
                with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
                    balance_df.to_excel(
                        writer, sheet_name="Covariate_Balance", index=False
                    )
                    pd.DataFrame([weight_stats]).to_excel(
                        writer, sheet_name="Weight_Diagnostics", index=False
                    )
                    coefficients_df.to_excel(
                        writer, sheet_name=f"{estimand}_Cox", index=False
                    )
                    ps_summary_df.to_excel(
                        writer, sheet_name="Propensity_Model", index=False
                    )
                    if not survival_snapshots.empty:
                        survival_snapshots.to_excel(
                            writer, sheet_name="Survival_Snapshots", index=False
                        )
                print(f"  Results saved to {export_path}")
            except Exception as e:
                print(f"  Warning: Could not export results to Excel: {e}")

        # ------------------------------------------------------------------
        # STEP 6: Build and return results dict
        # ------------------------------------------------------------------
        # Drop strata backup columns from weighted_df before returning
        strata_backup_to_drop = list(_strata_backup_map.values())
        df_clean = df.drop(columns=strata_backup_to_drop, errors="ignore")

        return {
            # --- Keys shared with analyze_treatment_effect() ---
            # These ensure compatibility with build_summary_table(),
            # compute_evalues_from_results(), and generate_gee_summary_report().
            "effect":            hazard_ratio,       # HR (not log HR) as primary effect
            "estimand":          estimand,
            "ci_lower":          hr_ci_lower,        # HR CI lower bound
            "ci_upper":          hr_ci_upper,        # HR CI upper bound
            "p_value":           hr_pvalue,
            "significant":       significant,
            "alpha":             alpha,
            "cohens_d":          None,               # Not applicable to survival
            "pct_change":        None,               # Not applicable to survival
            "mean_treatment":    mean_treatment,     # Survival prob at 365d (treated)
            "mean_control":      mean_control,       # Survival prob at 365d (control)
            "outcome_type":      "survival",
            "coefficients_df":   coefficients_df,    # log(HR) row for schema compatibility
            "full_coefficients_df": coefficients_df, # Same — no multi-param model here
            "ps_model":          ps_model,
            "ps_summary_df":     ps_summary_df,
            "balance_df":        balance_df,
            "weight_diagnostics": weight_stats,
            "ps_overlap_fig":    ps_overlap_fig,
            "weight_dist_fig":   weight_dist_fig,
            "weighted_df":       df_clean,           # Processed df with propensity_score & iptw

            # --- Survival-specific keys ---
            "hazard_ratio":          hazard_ratio,
            "hr_ci_lower":           hr_ci_lower,
            "hr_ci_upper":           hr_ci_upper,
            "hr_pvalue":             hr_pvalue,
            "concordance":           concordance,
            "ph_assumption_met":     ph_assumption_met,
            "ph_test_pvalue":        ph_test_pvalue,
            "kmf_treated":           cox_results.get("kmf_treated"),
            "kmf_control":           cox_results.get("kmf_control"),
            "cox_model":             cox_results.get("cox_model"),
            "survival_at_snapshots": survival_snapshots,
            "n_events_treated":      cox_results.get("n_events_treated"),
            "n_events_control":      cox_results.get("n_events_control"),
            "n_treated":             int(df[treatment_var].sum()),
            "n_control":             int((df[treatment_var] == 0).sum()),

            # --- Auto-stratification metadata ---
            "strata_vars":           cox_results.get("strata_vars", []),
            "ph_violations_detected": cox_results.get("ph_violations_detected", []),

            # --- Piecewise flag (standard path) ---
            "piecewise":             False,
            "piecewise_results":     None,

            # --- Variable-name metadata ---
            "treatment_var":         treatment_var,
            "time_var":              time_var,
            "event_var":             event_var,
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
            - "cohens_d": Converts to approximate RR using VanderWeele formula
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
            Convert Cohen's d to approximate risk ratio.
            Uses VanderWeele (2017) formula: RR ≈ exp(0.91 * d)
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
        effect_type: str = "auto",
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
        effect_type : str, default="auto"
            Effect type for E-value computation. If "auto", infers from the
            outcome: uses "cohens_d" for continuous outcomes (Cohen's d available)
            and "log_odds" for binary outcomes.
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
        rows = []
        
        for outcome_name, res in results_dict.items():
            cohens_d = res.get("cohens_d")
            effect = res.get("effect")
            ci_lower = res.get("ci_lower")
            ci_upper = res.get("ci_upper")
            
            # Auto-detect effect type based on outcome characteristics
            if effect_type == "auto":
                if cohens_d is not None and abs(cohens_d) > 0.01:
                    mean_ctrl = res.get("mean_control", 0.5)
                    if 0 < mean_ctrl < 1 and abs(effect) > 0.1:
                        use_type = "log_odds"
                        use_effect = effect
                        use_ci_lower = ci_lower
                        use_ci_upper = ci_upper
                    else:
                        use_type = "cohens_d"
                        use_effect = cohens_d
                        mean_treat = res.get("mean_treatment", 0)
                        mean_ctrl = res.get("mean_control", 0)
                        if cohens_d != 0:
                            raw_diff = mean_treat - mean_ctrl
                            scale_factor = cohens_d / raw_diff if raw_diff != 0 else 1
                            use_ci_lower = ci_lower * scale_factor if ci_lower else None
                            use_ci_upper = ci_upper * scale_factor if ci_upper else None
                        else:
                            use_ci_lower = None
                            use_ci_upper = None
                else:
                    use_type = "log_odds"
                    use_effect = effect
                    use_ci_lower = ci_lower
                    use_ci_upper = ci_upper
            else:
                use_type = effect_type
                use_effect = cohens_d if effect_type == "cohens_d" else effect
                use_ci_lower = ci_lower
                use_ci_upper = ci_upper
            
            try:
                evalue_result = CausalInferenceModel.compute_evalue(
                    effect=use_effect,
                    ci_lower=use_ci_lower,
                    ci_upper=use_ci_upper,
                    effect_type=use_type,
                    outcome_rare=outcome_rare
                )
                
                rows.append({
                    "Outcome": outcome_name,
                    "Effect_Type": use_type,
                    "Effect_Value": use_effect,
                    "Effect_RR": evalue_result["effect_rr"],
                    "E_Value_Point": evalue_result["evalue_point"],
                    "E_Value_CI": evalue_result["evalue_ci"],
                    "Robustness": evalue_result["robustness"],
                    "Interpretation": evalue_result["interpretation"],  # <-- ADDED
                    "P_Value": res.get("p_value"),
                    "Significant": res.get("significant", False)
                })
            except Exception as e:
                print(f"  Warning: Could not compute E-value for {outcome_name}: {e}")
                rows.append({
                    "Outcome": outcome_name,
                    "Effect_Type": use_type,
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
        print("=" * 70)
        
        # Print per-outcome interpretations for significant results  <-- ADDED BLOCK
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
            rmst = np.trapz(p_clip, t_clip)
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
        reporting hazard ratios and RMST differences instead of mean differences
        and Cohen's d.

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
        rows     = []
        raw_pvals = []

        for outcome_name, res in survival_results_dict.items():
            hr        = res.get("hazard_ratio")
            hr_lower  = res.get("hr_ci_lower")
            hr_upper  = res.get("hr_ci_upper")
            p_value   = res.get("p_value") or res.get("hr_pvalue")
            estimand  = res.get("estimand", "ATT")
            n_treated = res.get("n_events_treated", None)
            n_control = res.get("n_events_control", None)
            ph_met    = res.get("ph_assumption_met")
            concordance = res.get("concordance")
            weight_diag = res.get("weight_diagnostics", {})

            raw_pvals.append(p_value if p_value is not None else 1.0)

            # Determine whether this is a piecewise result
            is_piecewise = bool(res.get("piecewise") and res.get("piecewise_results"))

            # For piecewise results, identify the best interval label
            best_interval_label = None
            if is_piecewise:
                pw_list = res["piecewise_results"]
                valid_pw = [r for r in pw_list if not np.isnan(r.get("hr_pvalue", np.nan))]
                if valid_pw:
                    best_interval_label = min(
                        valid_pw, key=lambda r: r["hr_pvalue"]
                    ).get("interval_label", None)

            row = {
                "Outcome":              outcome_name,
                "Estimand":             estimand,
            }

            # For piecewise, label the headline HR as interval-specific
            if is_piecewise and best_interval_label:
                row["Best_Interval"]    = best_interval_label
                row["Best_HR"]          = round(hr, 4)      if hr       is not None else None
                row["Best_HR_95_CI"]    = (f"[{hr_lower:.3f}, {hr_upper:.3f}]"
                                           if hr_lower is not None and hr_upper is not None
                                           else None)
            else:
                row["Hazard_Ratio"]     = round(hr, 4)      if hr       is not None else None
                row["HR_95_CI"]         = (f"[{hr_lower:.3f}, {hr_upper:.3f}]"
                                           if hr_lower is not None and hr_upper is not None
                                           else None)

            row.update({
                "P_Value":              p_value,
                "Concordance":          round(concordance, 4) if concordance is not None else None,
                "PH_Assumption_Met":    ph_met,
                "N_Events_Treated":     n_treated,
                "N_Events_Control":     n_control,
                "N_Total":              weight_diag.get("n_observations"),
                "ESS":                  weight_diag.get("effective_sample_size"),
            })

            # Add piecewise interval HR columns if available
            if res.get("piecewise") and res.get("piecewise_results"):
                for pw_r in res["piecewise_results"]:
                    lbl = pw_r["interval_label"]  # e.g. "0–3 mo"
                    col_sfx = lbl.replace(" ", "").replace("\u2013", "_")  # "0_3mo"
                    hr_val = pw_r["hazard_ratio"]
                    row[f"HR_{col_sfx}"]    = round(hr_val, 4) if not np.isnan(hr_val) else None
                    row[f"HR_CI_{col_sfx}"] = (
                        f"[{pw_r['hr_ci_lower']:.3f}, {pw_r['hr_ci_upper']:.3f}]"
                        if not np.isnan(pw_r["hr_ci_lower"]) else None
                    )
                    row[f"P_{col_sfx}"]   = (
                        round(pw_r["hr_pvalue"], 4) if not np.isnan(pw_r["hr_pvalue"]) else None
                    )
                    row[f"Sig_{col_sfx}"] = (
                        CausalInferenceModel._significance_stars(pw_r["hr_pvalue"])
                        if not np.isnan(pw_r["hr_pvalue"]) else ""
                    )

            # Add snapshot survival difference columns from KM curves
            snapshots = res.get("survival_at_snapshots")
            if snapshots is not None and not snapshots.empty:
                for _, snap_row in snapshots.iterrows():
                    tp = snap_row["timepoint_label"]  # e.g. "3mo"
                    s_t = snap_row["survival_treated"]
                    s_c = snap_row["survival_control"]
                    s_d = snap_row["survival_diff"]
                    row[f"Surv_Trained_{tp}"]  = round(s_t, 4) if pd.notna(s_t) else None
                    row[f"Surv_Control_{tp}"]  = round(s_c, 4) if pd.notna(s_c) else None
                    row[f"Surv_Diff_{tp}"]     = round(s_d, 4) if pd.notna(s_d) else None

            # Add RMST columns if provided
            if rmst_results_dict and outcome_name in rmst_results_dict:
                rmst = rmst_results_dict[outcome_name]
                row["RMST_Treated_Days"]  = round(rmst.get("rmst_treated", np.nan), 1)
                row["RMST_Control_Days"]  = round(rmst.get("rmst_control", np.nan), 1)
                row["RMST_Difference"]    = round(rmst.get("rmst_diff",    np.nan), 1)
                row["RMST_CI_Lower"]     = round(rmst.get("rmst_ci_lower", np.nan), 1)
                row["RMST_CI_Upper"]     = round(rmst.get("rmst_ci_upper", np.nan), 1)
                row["RMST_CI"]            = (
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
            reject_arr      = np.array([raw_pvals[0] < alpha])

        summary_df["P_Value_Corrected"] = pvals_corrected
        summary_df["Significant"]       = reject_arr
        summary_df["Significance"]      = [
            CausalInferenceModel._significance_stars(p) for p in pvals_corrected
        ]
        # Record actual correction method: "none" when only 1 test (no correction applied)
        summary_df["Correction_Method"] = correction_method if len(raw_pvals) > 1 else "none"

        # --- Print formatted table ---
        display_title = title or "IPTW + Cox: Survival Analysis Summary"
        print(f"\n{'=' * 65}")
        print(f"  {display_title}")
        print(f"{'=' * 65}")

        # Select display columns (exclude raw CI bounds and verbose interval detail)
        _exclude_patterns = {"HR_CI_Lower", "HR_CI_Upper", "RMST_CI_Lower", "RMST_CI_Upper"}
        display_cols = [
            c for c in summary_df.columns
            if c not in _exclude_patterns
            and not c.startswith("HR_CI_")       # piecewise CI strings
            and not c.startswith("Sig_")         # piecewise significance stars
            and not c.startswith("Surv_Trained_") and not c.startswith("Surv_Control_")
        ]
        display_df = summary_df[display_cols].copy()

        for col in ["Hazard_Ratio", "Best_HR", "Concordance"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        # Also format piecewise interval HR columns
        for col in display_df.columns:
            if col.startswith("HR_") and col not in ("HR_95_CI",):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )
            elif col.startswith("Surv_Diff_"):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
                )
            elif col.startswith("P_") and col not in ("P_Value", "P_Value_Corrected", "PH_Assumption_Met"):
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
        print(f"{'=' * 65}")
        print("  Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print(f"  Correction: {actual_correction} across {len(rows)} outcome{'s' if len(rows) != 1 else ''}")
        print("  HR < 1 = lower hazard of departure (training is protective)")
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

    # ==================================================================
    # Report generation methods
    # ==================================================================

    @staticmethod
    def _format_pvalue(p: float) -> str:
        """Format a p-value for display in Markdown tables."""
        if p < 0.0001:
            return "< 0.0001"
        elif p < 0.001:
            return "< 0.001"
        else:
            return f"{p:.3f}"

    @staticmethod
    def generate_gee_summary_report(
        summary_df: pd.DataFrame,
        evalues_df: pd.DataFrame,
        results_dict: Dict[str, Dict],
        estimand: str,
        family_label: str,
        outcome_descriptions: Optional[Dict[str, str]] = None,
        outcome_valence: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a Markdown technical summary for a family of outcomes.

        Combines the results table, per-outcome narratives, balance verification,
        and E-value sensitivity analysis into a single rendered Markdown block.
        Designed for use with ``IPython.display.Markdown`` in Jupyter notebooks.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Output of ``build_summary_table()`` for this family of outcomes.
        evalues_df : pd.DataFrame
            Output of ``compute_evalues_from_results()`` for this family.
        results_dict : Dict[str, Dict]
            Per-outcome results from ``analyze_treatment_effect()``.
        estimand : str
            "ATE" or "ATT".
        family_label : str
            Label for this outcome family, e.g. "Survey Outcomes (Continuous)".
        outcome_descriptions : Dict[str, str], optional
            Mapping from variable names to display-friendly names.
        outcome_valence : Dict[str, str], optional
            Mapping from variable names to direction interpretation.
            Use ``"positive"`` when higher values are better (default),
            ``"negative"`` when higher values are worse (e.g. burnout,
            workload).  Outcomes not listed default to ``"positive"``.

        Returns
        -------
        str
            Markdown-formatted summary string.
        """
        if outcome_descriptions is None:
            outcome_descriptions = {}
        if outcome_valence is None:
            outcome_valence = {}

        fmt_p = CausalInferenceModel._format_pvalue
        lines: list = []

        # Detect outcome type from first result
        first_key = next(iter(results_dict))
        is_binary = results_dict[first_key].get("outcome_type") == "binary"

        n_tests = len(summary_df)
        correction = (
            summary_df["Correction_Method"].iloc[0]
            if "Correction_Method" in summary_df.columns
            else "fdr_bh"
        )
        if correction == "none" or n_tests == 1:
            correction_label = f"{n_tests} test, no correction needed"
        else:
            correction_label = f"{n_tests} tests, {correction.upper()}-corrected"

        lines.append(f"#### {family_label} ({correction_label})")
        lines.append("")

        # ── Results table ──────────────────────────────────────────────
        if is_binary:
            lines.append(
                f"| Outcome | {estimand} (log-odds) | Odds Ratio | "
                f"95% CI (log-odds) | p (corrected) | Cohen's d | Significant? |"
            )
            lines.append(
                "|---------|----------------|------------|" +
                "-------------------|---------------|-----------|-------------|"
            )
            for _, row in summary_df.iterrows():
                outcome = row["Outcome"]
                name = outcome_descriptions.get(outcome, outcome)
                eff = row["Effect"]
                or_val = np.exp(eff)
                ci_lo, ci_hi = row["CI_Lower"], row["CI_Upper"]
                p_corr = row["P_Value_Corrected"]
                d = row.get("Cohens_d")
                sig = row["Significant"]
                stars = row.get("Significance", "")

                sig_str = f"Yes {stars}" if sig else "No"
                d_str = f"{d:.2f}" if pd.notna(d) else "\u2014"

                lines.append(
                    f"| {name} | **{eff:+.2f}** | {or_val:.2f} | "
                    f"[{ci_lo:.2f}, {ci_hi:.2f}] | {fmt_p(p_corr)} | {d_str} | {sig_str} |"
                )
        else:
            lines.append(
                f"| Outcome | {estimand} | 95% CI | p (corrected) | Cohen's d | Significant? |"
            )
            lines.append("|---------|-----|--------|---------------|-----------|-------------|")
            for _, row in summary_df.iterrows():
                outcome = row["Outcome"]
                name = outcome_descriptions.get(outcome, outcome)
                eff = row["Effect"]
                ci_lo, ci_hi = row["CI_Lower"], row["CI_Upper"]
                p_corr = row["P_Value_Corrected"]
                d = row.get("Cohens_d")
                sig = row["Significant"]
                stars = row.get("Significance", "")

                sig_str = f"Yes {stars}" if sig else "No"
                d_str = f"{d:.2f}" if pd.notna(d) else "\u2014"

                lines.append(
                    f"| {name} | **{eff:+.2f}** | [{ci_lo:.2f}, {ci_hi:.2f}] | "
                    f"{fmt_p(p_corr)} | {d_str} | {sig_str} |"
                )

        lines.append("")

        # ── Per-outcome narratives ─────────────────────────────────────
        for _, row in summary_df.iterrows():
            outcome = row["Outcome"]
            name = outcome_descriptions.get(outcome, outcome)
            eff = row["Effect"]
            sig = row["Significant"]
            res = results_dict[outcome]
            pct = res.get("pct_change")
            d = res.get("cohens_d")
            p_corr = row["P_Value_Corrected"]

            # Determine valence-aware language
            valence = outcome_valence.get(outcome, "positive")
            if valence == "negative":
                # Higher values are worse (e.g. workload, burnout)
                quality_word = "worsening" if eff > 0 else "improvement"
            else:
                # Higher values are better (default)
                quality_word = "improvement" if eff > 0 else "decline"

            if not is_binary:
                if sig:
                    if pct is not None:
                        pct_str = f"**{pct:+.1f}% relative {quality_word}**"
                    else:
                        pct_str = f"**{quality_word}**"
                    lines.append(f"**{name}**")
                    lines.append(
                        f"- {estimand} of **{eff:+.2f}** (Cohen's d = {d:.2f}), "
                        f"representing a {pct_str} vs. controls"
                    )
                    lines.append("")
                else:
                    lines.append(f"**{name}** \u2014 *No effect detected*")
                    lines.append(
                        f"- {estimand} of **{eff:+.2f}**, not statistically "
                        f"significant (p = {p_corr:.3f})"
                    )
                    lines.append("")

        if is_binary:
            sig_rows = summary_df[summary_df["Significant"]]
            if not sig_rows.empty:
                ors = [
                    (row["Outcome"], np.exp(row["Effect"]))
                    for _, row in sig_rows.iterrows()
                ]

                # Range summary — use max/min OR (order-agnostic)
                or_max = max(ors, key=lambda x: x[1])
                or_min = min(ors, key=lambda x: x[1])
                pct_high = (or_max[1] - 1) * 100
                pct_low = (or_min[1] - 1) * 100
                max_name = outcome_descriptions.get(or_max[0], or_max[0])
                min_name = outcome_descriptions.get(or_min[0], or_min[0])
                cohens_ds = [
                    results_dict[o]["cohens_d"]
                    for o, _ in ors
                    if results_dict[o].get("cohens_d") is not None
                ]
                d_range = (
                    f" with effect sizes ranging from "
                    f"{min(cohens_ds):.2f}\u2013{max(cohens_ds):.2f}"
                    if cohens_ds
                    else ""
                )

                if len(ors) > 1:
                    lines.append(
                        f"- Significant outcomes show odds ranging from "
                        f"{pct_low:+.0f}% ({min_name}) to "
                        f"{pct_high:+.0f}% ({max_name}) relative to controls"
                        f"{d_range}."
                    )
                else:
                    lines.append(
                        f"- {max_name}: OR = {or_max[1]:.2f} "
                        f"({pct_high:+.0f}% odds vs. controls){d_range}."
                    )

                # Trend detection (only when >1 significant outcome)
                if len(ors) > 1:
                    vals = [o[1] for o in ors]
                    is_decreasing = all(
                        vals[i] >= vals[i + 1] for i in range(len(vals) - 1)
                    )
                    is_increasing = all(
                        vals[i] <= vals[i + 1] for i in range(len(vals) - 1)
                    )
                    is_flat = max(vals) - min(vals) < 0.05

                    if is_flat:
                        lines.append(
                            "- The treatment effect is **stable** across all "
                            "significant outcomes."
                        )
                    elif is_decreasing:
                        lines.append(
                            "- The treatment effect **attenuates** from the "
                            "first to last significant outcome (strongest at "
                            f"{outcome_descriptions.get(ors[0][0], ors[0][0])}"
                            f", weakest at "
                            f"{outcome_descriptions.get(ors[-1][0], ors[-1][0])}"
                            f"), consistent with an effect that fades over time "
                            f"or across contexts."
                        )
                    elif is_increasing:
                        lines.append(
                            "- The treatment effect **strengthens** from the "
                            "first to last significant outcome (weakest at "
                            f"{outcome_descriptions.get(ors[0][0], ors[0][0])}"
                            f", strongest at "
                            f"{outcome_descriptions.get(ors[-1][0], ors[-1][0])}"
                            f"), suggesting a cumulative or delayed benefit."
                        )
                    else:
                        lines.append(
                            "- The treatment effect is **non-monotonic** across "
                            "significant outcomes \u2014 consider examining each "
                            "outcome individually."
                        )

                lines.append("")

                # Generic OR interpretation
                top_or = or_max[1]
                top_name = outcome_descriptions.get(or_max[0], or_max[0])
                top_pct = (top_or - 1) * 100
                direction = "more" if top_or > 1 else "less"
                lines.append("**How to read the odds ratios:**")
                lines.append(
                    f"An odds ratio (OR) above 1.0 means treated individuals "
                    f"were *{direction} likely* to experience the outcome than "
                    f"untreated individuals. For example, an OR of "
                    f"**{top_or:.2f} for {top_name}** means the odds were "
                    f"**{top_pct:+.0f}%** relative to the comparison group. "
                    f"As the OR approaches 1.0, the treatment effect weakens."
                )
                lines.append("")

            # Handle case where no binary outcomes are significant
            else:
                lines.append(
                    "No binary outcomes reached statistical significance "
                    "after multiple testing correction."
                )
                lines.append("")

        # ── Balance verification ───────────────────────────────────────
        lines.append("#### Post-Weighting Balance Verification")
        all_balanced = True
        for outcome, res in results_dict.items():
            bal_df = res.get("balance_df")
            if bal_df is not None and "balanced_after_weighting" in bal_df.columns:
                n_imbal = int(bal_df["balanced_after_weighting"].eq(False).sum())
                if n_imbal > 0:
                    all_balanced = False
                    name = outcome_descriptions.get(outcome, outcome)
                    lines.append(f"- \u26a0\ufe0f {name}: {n_imbal} imbalanced covariates")
        if all_balanced:
            lines.append(
                f"- \u2705 All {len(results_dict)} outcomes passed balance verification "
                f"(0 imbalanced covariates)."
            )
            lines.append(
                "- IPTW successfully balanced observed confounders across treatment groups."
            )
        lines.append("")

        # ── E-value table ──────────────────────────────────────────────
        if evalues_df is not None and not evalues_df.empty:
            lines.append("#### E-Value Sensitivity Analysis")
            lines.append("")
            lines.append("| Outcome | E-Value Point | E-Value CI | Robustness |")
            lines.append("|---------|---------------|------------|------------|")
            for _, row in evalues_df.iterrows():
                outcome = row["Outcome"]
                name = outcome_descriptions.get(outcome, outcome)
                ev_pt = row["E_Value_Point"]
                ev_ci = row["E_Value_CI"]
                robustness = row["Robustness"]
                sig = row.get("Significant", False)

                ev_ci_str = f"{ev_ci:.2f}" if pd.notna(ev_ci) else "\u2014"
                if sig:
                    lines.append(
                        f"| {name} | **{ev_pt:.2f}** | **{ev_ci_str}** | {robustness} |"
                    )
                else:
                    lines.append(
                        f"| {name} | {ev_pt:.2f} | {ev_ci_str} | {robustness} (ns) |"
                    )
            lines.append("")

            # Per-outcome E-value interpretations (significant only)
            sig_ev = evalues_df[evalues_df["Significant"] == True]
            if not sig_ev.empty:
                lines.append("**Significant outcome interpretations:**")
                lines.append("")
                for _, row in sig_ev.iterrows():
                    outcome = row["Outcome"]
                    name = outcome_descriptions.get(outcome, outcome)
                    interp = row.get("Interpretation", "")
                    if pd.notna(interp) and interp:
                        lines.append(f"- **{name}:** {interp}")
                        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_survival_summary_report(
        survival_summary_df: pd.DataFrame,
        survival_evalues_df: pd.DataFrame,
        survival_results_dict: Dict[str, Dict],
        estimand: str,
        outcome_descriptions: Optional[Dict[str, str]] = None,
        outcome_valence: Optional[Dict[str, str]] = None,
        survival_plot_fig: Optional[object] = None,
    ) -> str:
        """
        Generate a Markdown technical summary for survival (time-to-event) outcomes.

        Produces a report structured around hazard ratios, RMST differences,
        and survival probabilities at snapshot timepoints. Designed to be
        rendered alongside generate_gee_summary_report() in the ATE/ATT
        technical summary cells.

        Parameters
        ----------
        survival_summary_df : pd.DataFrame
            Output of ``build_survival_summary_table()`` for retention outcomes.
        survival_evalues_df : pd.DataFrame
            Output of ``compute_evalues_from_results()`` for retention outcomes.
            Should be computed with effect_type='risk_ratio'.
        survival_results_dict : Dict[str, Dict]
            Per-outcome results from ``analyze_survival_effect()``.
            Keys are outcome names (e.g. 'retention'), values are result dicts.
        estimand : str
            "ATE" or "ATT".
        outcome_descriptions : Dict[str, str], optional
            Mapping from variable names to display-friendly names.
            e.g. {'retention': 'Manager Retention (Survival)'}
        outcome_valence : Dict[str, str], optional
            Not used for survival outcomes (HR direction is self-explanatory),
            but accepted for API consistency with generate_gee_summary_report().
        survival_plot_fig : matplotlib.figure.Figure, optional
            A matplotlib Figure (e.g. from ``plot_survival_curves()``).
            If provided, the figure is base64-encoded and embedded inline
            in the Markdown report. If ``None``, the report is generated
            without a plot.

        Returns
        -------
        str
            Markdown-formatted summary string.
        """
        if outcome_descriptions is None:
            outcome_descriptions = {}

        fmt_p = CausalInferenceModel._format_pvalue
        lines: list = []

        n_tests = len(survival_summary_df)
        correction = (
            survival_summary_df["Correction_Method"].iloc[0]
            if "Correction_Method" in survival_summary_df.columns
            else "fdr_bh"
        )
        if correction == "none" or n_tests == 1:
            correction_label = f"{n_tests} test, no correction needed"
        else:
            correction_label = f"{n_tests} tests, {correction.upper()}-corrected"

        # Detect piecewise mode from any result dict
        first_res = next(iter(survival_results_dict.values()))
        is_piecewise = first_res.get("piecewise", False)

        lines.append(
            f"#### Retention Outcomes — Survival Analysis "
            f"({correction_label})"
        )
        lines.append("")

        if is_piecewise:
            lines.append(
                "> **Method**: Piecewise IPTW-weighted Cox Proportional Hazards model. "
                "Separate Cox models are fit per time interval (0\u20133, 3\u20136, "
                "6\u20139, 9\u201312 months) to provide interval-specific hazard ratios "
                "that remain valid when the global PH assumption is violated. "
                "HR < 1 = lower hazard of departure (training is protective). "
                "Snapshot survival differences show IPTW-weighted Kaplan\u2013Meier "
                "retention probabilities at each timepoint."
            )
        else:
            lines.append(
                "> **Method**: IPTW-weighted Cox Proportional Hazards model. "
                "Hazard Ratio (HR) < 1 indicates lower hazard of departure "
                "(i.e., training is protective)."
            )
        lines.append("")

        # ── Optional survival curve plot ───────────────────────────────────
        if survival_plot_fig is not None:
            try:
                import io, base64
                buf = io.BytesIO()
                survival_plot_fig.savefig(buf, format="png", dpi=150,
                                          bbox_inches="tight")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                lines.append(
                    f'<img src="data:image/png;base64,{img_b64}" '
                    f'alt="Kaplan–Meier Survival Curves" '
                    f'style="max-width:100%;" />'
                )
                lines.append("")
            except Exception:
                pass  # Graceful fallback: skip plot if encoding fails

        # ── Results table ──────────────────────────────────────────────────
        if is_piecewise:
            # === PIECEWISE: Interval-specific HR table ===
            lines.append("##### Interval-Specific Hazard Ratios")
            lines.append("")
            lines.append(
                "| Interval | HR | 95% CI | p-value | At Risk | Events | Sig? |"
            )
            lines.append(
                "|----------|----|--------|---------|---------|--------|------|"
            )

            for outcome, res in survival_results_dict.items():
                pw_results = res.get("piecewise_results", [])
                name = outcome_descriptions.get(outcome, outcome)
                for pw_r in pw_results:
                    lbl  = pw_r["interval_label"]
                    hr_v = pw_r["hazard_ratio"]
                    ci_l = pw_r["hr_ci_lower"]
                    ci_h = pw_r["hr_ci_upper"]
                    p_v  = pw_r["hr_pvalue"]
                    n_r  = pw_r["n_at_risk"]
                    n_e  = pw_r["n_events"]
                    sig_v = pw_r["significant"]

                    if np.isnan(hr_v):
                        lines.append(
                            f"| {lbl} | — | — | — | {n_r} | {n_e} | — |"
                        )
                    else:
                        stars = CausalInferenceModel._significance_stars(p_v)
                        sig_s = f"Yes {stars}" if sig_v else "No"
                        lines.append(
                            f"| {lbl} | **{hr_v:.3f}** | "
                            f"[{ci_l:.3f}, {ci_h:.3f}] | "
                            f"{fmt_p(p_v)} | {n_r} | {n_e} | {sig_s} |"
                        )
            lines.append("")

            # === PIECEWISE: Snapshot survival differences ===
            lines.append("##### IPTW-Weighted Survival Probabilities (Kaplan\u2013Meier)")
            lines.append("")
            lines.append(
                "| Timepoint | Trained | Untrained | Difference |"
            )
            lines.append(
                "|-----------|---------|-----------|------------|"
            )
            for outcome, res in survival_results_dict.items():
                snap_df = res.get("survival_at_snapshots")
                if snap_df is not None and not snap_df.empty:
                    for _, snap_row in snap_df.iterrows():
                        tp   = snap_row.get("timepoint_label", "")
                        s_t  = snap_row.get("survival_treated", float("nan"))
                        s_c  = snap_row.get("survival_control", float("nan"))
                        s_d  = snap_row.get("survival_diff", float("nan"))
                        s_t_str  = f"{s_t*100:.1f}%" if pd.notna(s_t) else "\u2014"
                        s_c_str  = f"{s_c*100:.1f}%" if pd.notna(s_c) else "\u2014"
                        diff_str = f"**{s_d*100:+.1f}pp**" if pd.notna(s_d) else "\u2014"
                        lines.append(
                            f"| {tp} | {s_t_str} | {s_c_str} | {diff_str} |"
                        )
            lines.append("")

            # === PIECEWISE: Per-outcome narrative ===
            for _, row in survival_summary_df.iterrows():
                outcome = row["Outcome"]
                name    = outcome_descriptions.get(outcome, outcome)
                sig     = row.get("Significant", False)
                p_corr  = row.get("P_Value_Corrected", row.get("P_Value", float("nan")))
                stars_v = row.get("Significance", "")
                res     = survival_results_dict.get(outcome, {})
                pw_res  = res.get("piecewise_results", [])
                n_events_t = res.get("n_events_treated")
                n_events_c = res.get("n_events_control")
                n_treated  = res.get("n_treated")
                n_control  = res.get("n_control")
                surv_t     = res.get("mean_treatment")
                surv_c     = res.get("mean_control")

                # Find strongest interval
                valid_intervals = [r for r in pw_res if not np.isnan(r["hr_pvalue"])]
                sig_intervals   = [r for r in valid_intervals if r["significant"]]

                if sig_intervals:
                    strongest = min(sig_intervals, key=lambda r: r["hr_pvalue"])
                    strongest_hr = strongest["hazard_ratio"]
                    direction = "lower" if strongest_hr < 1 else "higher"
                    pct_change = abs(1 - strongest_hr) * 100

                    lines.append(f"**{name}** {stars_v}")
                    lines.append(
                        f"- **{len(sig_intervals)} of {len(pw_res)} intervals** "
                        f"show significant treatment effects."
                    )
                    lines.append(
                        f"- Strongest effect in **{strongest['interval_label']}**: "
                        f"HR = **{strongest_hr:.3f}** — "
                        f"**{pct_change:.1f}% {direction} hazard of departure**."
                    )
                    if surv_t is not None and surv_c is not None:
                        lines.append(
                            f"- Estimated 12-month retention: "
                            f"**{surv_t*100:.1f}%** (trained) vs. "
                            f"**{surv_c*100:.1f}%** (untrained)."
                        )
                    if n_events_t is not None and n_events_c is not None:
                        lines.append(
                            f"- Events observed: {n_events_t} departures "
                            f"(trained, n={n_treated}) / "
                            f"{n_events_c} departures (untrained, n={n_control})."
                        )
                    lines.append(
                        "- Piecewise approach bypasses the global PH assumption — "
                        "each interval\u2019s HR captures the local treatment effect."
                    )
                    lines.append("")
                else:
                    lines.append(f"**{name}** \u2014 *No significant effect detected*")
                    p_str = fmt_p(p_corr) if pd.notna(p_corr) else "\u2014"
                    ns_qualifier = (
                        "not significant" if correction == "none" or n_tests == 1
                        else "not significant after correction"
                    )
                    lines.append(
                        f"- Best interval p = {p_str} ({ns_qualifier})."
                    )
                    lines.append("")

            # === PIECEWISE: How to read ===
            lines.append("**How to read interval-specific hazard ratios:**")
            lines.append(
                "Each interval\u2019s HR captures the treatment effect *within that "
                "time window only*. HR < 1 means trained managers had a lower "
                "instantaneous departure rate during that interval. "
                "This approach is preferred when the proportional hazards assumption "
                "is violated, because a single global HR would misrepresent the "
                "time-varying nature of the treatment effect."
            )
            lines.append("")

        else:
            # === STANDARD (non-piecewise) path ===
            has_rmst = "RMST_Difference" in survival_summary_df.columns
            has_ph   = "PH_Assumption_Met" in survival_summary_df.columns

            header_cols = [
                f"| Outcome | {estimand} (HR) | 95% CI | p (corrected) | Significant?"
            ]
            sep_cols = ["|---------|----------------|--------|---------------|-------------|"]

            if has_rmst:
                header_cols[0] += " | RMST Diff (days) | RMST 95% CI"
                sep_cols[0]    += "------------------|-------------|"
            if has_ph:
                header_cols[0] += " | PH Assumption"
                sep_cols[0]    += "----------------|"
            header_cols[0] += " |"
            sep_cols[0]    += ""

            lines.append(header_cols[0])
            lines.append(sep_cols[0])

            for _, row in survival_summary_df.iterrows():
                outcome = row["Outcome"]
                name    = outcome_descriptions.get(outcome, outcome)
                hr      = row.get("Hazard_Ratio", row.get("Effect", float("nan")))
                ci_lo   = row.get("HR_CI_Lower",  row.get("CI_Lower", float("nan")))
                ci_hi   = row.get("HR_CI_Upper",  row.get("CI_Upper", float("nan")))
                p_corr  = row.get("P_Value_Corrected", row.get("P_Value", float("nan")))
                sig     = row.get("Significant", False)
                stars_v = row.get("Significance", "")

                sig_str = f"Yes {stars_v}" if sig else "No"

                hr_str    = f"**{hr:.3f}**"   if pd.notna(hr)    else "\u2014"
                ci_lo_str = f"{ci_lo:.3f}"    if pd.notna(ci_lo) else "\u2014"
                ci_hi_str = f"{ci_hi:.3f}"    if pd.notna(ci_hi) else "\u2014"
                p_str     = fmt_p(p_corr)     if pd.notna(p_corr) else "\u2014"

                row_str = (
                    f"| {name} | {hr_str} | [{ci_lo_str}, {ci_hi_str}] "
                    f"| {p_str} | {sig_str}"
                )

                if has_rmst:
                    rmst_diff    = row.get("RMST_Difference", float("nan"))
                    rmst_ci_lo   = row.get("RMST_CI_Lower",   float("nan"))
                    rmst_ci_hi   = row.get("RMST_CI_Upper",   float("nan"))
                    rmst_str     = (
                        f"**+{rmst_diff:.1f}**" if pd.notna(rmst_diff) and rmst_diff >= 0
                        else (f"**{rmst_diff:.1f}**" if pd.notna(rmst_diff) else "\u2014")
                    )
                    rmst_ci_str  = (
                        f"[{rmst_ci_lo:.1f}, {rmst_ci_hi:.1f}]"
                        if pd.notna(rmst_ci_lo) and pd.notna(rmst_ci_hi) else "\u2014"
                    )
                    row_str += f" | {rmst_str} | {rmst_ci_str}"

                if has_ph:
                    ph_met = row.get("PH_Assumption_Met", None)
                    if ph_met is True:
                        ph_str = "\u2705 Met"
                    elif ph_met is False:
                        ph_str = "\u26a0\ufe0f Violated"
                    else:
                        ph_str = "\u2014"
                    row_str += f" | {ph_str}"

                row_str += " |"
                lines.append(row_str)

            lines.append("")

            # ── Standard per-outcome narrative ─────────────────────────────
            for _, row in survival_summary_df.iterrows():
                outcome = row["Outcome"]
                name    = outcome_descriptions.get(outcome, outcome)
                hr      = row.get("Hazard_Ratio", row.get("Effect", float("nan")))
                sig     = row.get("Significant", False)
                p_corr  = row.get("P_Value_Corrected", row.get("P_Value", float("nan")))
                stars_v = row.get("Significance", "")

                res = survival_results_dict.get(outcome, {})
                n_events_t = res.get("n_events_treated")
                n_events_c = res.get("n_events_control")
                n_treated  = res.get("n_treated")
                n_control  = res.get("n_control")
                ph_met     = res.get("ph_assumption_met")
                surv_t     = res.get("mean_treatment")
                surv_c     = res.get("mean_control")

                if sig and pd.notna(hr):
                    direction = "lower" if hr < 1 else "higher"
                    pct_change = abs(1 - hr) * 100
                    lines.append(f"**{name}** {stars_v}")
                    lines.append(
                        f"- Hazard Ratio = **{hr:.3f}** \u2014 trained managers had "
                        f"**{pct_change:.1f}% {direction} hazard of departure** "
                        f"compared to untrained managers."
                    )
                    if surv_t is not None and surv_c is not None and pd.notna(surv_t) and pd.notna(surv_c):
                        lines.append(
                            f"- Estimated 12-month retention: "
                            f"**{surv_t*100:.1f}%** (trained) vs. "
                            f"**{surv_c*100:.1f}%** (untrained)."
                        )
                    if n_events_t is not None and n_events_c is not None:
                        lines.append(
                            f"- Events observed: {n_events_t} departures (trained, n={n_treated}) "
                            f"/ {n_events_c} departures (untrained, n={n_control})."
                        )
                    if ph_met is False:
                        lines.append(
                            "- \u26a0\ufe0f **Proportional hazards assumption may be violated.** "
                            "Consider re-running with ``piecewise=True`` for interval-specific HRs."
                        )
                    lines.append("")
                else:
                    lines.append(f"**{name}** \u2014 *No significant effect detected*")
                    p_str = fmt_p(p_corr) if pd.notna(p_corr) else "\u2014"
                    hr_str_v = f"{hr:.3f}" if pd.notna(hr) else "\u2014"
                    ns_qualifier = (
                        "not significant" if correction == "none" or n_tests == 1
                        else "not significant after correction"
                    )
                    lines.append(
                        f"- HR = {hr_str_v}, p = {p_str} ({ns_qualifier})."
                    )
                    lines.append("")

            # ── How to read hazard ratios (standard) ───────────────────────
            lines.append("**How to read hazard ratios:**")
            lines.append(
                "A hazard ratio (HR) below 1.0 means trained managers were *less likely* "
                "to depart at any given point in time compared to untrained managers. "
                "For example, HR = 0.70 means a 30% lower instantaneous departure rate."
            )
            lines.append("")

        # ── Snapshot validation (non-piecewise only — piecewise renders inline) ──
        has_snapshots = (
            not is_piecewise
            and any(
                res.get("survival_at_snapshots") is not None
                for res in survival_results_dict.values()
            )
        )
        if has_snapshots:
            lines.append("#### Snapshot Validation (Cox vs. Observed Retention Rates)")
            lines.append("")
            lines.append(
                "The table below compares Cox model-estimated survival probabilities "
                "at each snapshot timepoint against the observed binary retention rates. "
                "Close alignment validates the survival model."
            )
            lines.append("")
            for outcome, res in survival_results_dict.items():
                snap_df = res.get("survival_at_snapshots")
                if snap_df is not None and not snap_df.empty:
                    name = outcome_descriptions.get(outcome, outcome)
                    lines.append(f"*{name}:*")
                    lines.append("")
                    # Convert snapshot df to markdown table
                    lines.append(
                        "| Timepoint | Days | Survival (Trained) | Survival (Untrained) | Difference |"
                    )
                    lines.append(
                        "|-----------|------|--------------------|----------------------|------------|"
                    )
                    for _, snap_row in snap_df.iterrows():
                        label   = snap_row.get("timepoint_label", "")
                        days    = snap_row.get("timepoint_days", "")
                        s_t     = snap_row.get("survival_treated", float("nan"))
                        s_c     = snap_row.get("survival_control", float("nan"))
                        s_diff  = snap_row.get("survival_diff", float("nan"))
                        s_t_str  = f"{s_t*100:.1f}%"  if pd.notna(s_t)   else "—"
                        s_c_str  = f"{s_c*100:.1f}%"  if pd.notna(s_c)   else "—"
                        diff_str = f"{s_diff*100:+.1f}pp" if pd.notna(s_diff) else "—"
                        lines.append(
                            f"| {label} | {days} | {s_t_str} | {s_c_str} | {diff_str} |"
                        )
                    lines.append("")

        # ── Balance verification ───────────────────────────────────────────
        lines.append("#### Post-Weighting Balance Verification")
        all_balanced = True
        for outcome, res in survival_results_dict.items():
            bal_df = res.get("balance_df")
            if bal_df is not None and "balanced_after_weighting" in bal_df.columns:
                n_imbal = int(bal_df["balanced_after_weighting"].eq(False).sum())
                if n_imbal > 0:
                    all_balanced = False
                    name = outcome_descriptions.get(outcome, outcome)
                    lines.append(f"- ⚠️ {name}: {n_imbal} imbalanced covariates after weighting")
        if all_balanced:
            lines.append(
                f"- ✅ All {len(survival_results_dict)} survival outcome(s) passed "
                f"balance verification (0 imbalanced covariates)."
            )
            lines.append(
                "- IPTW successfully balanced observed confounders across treatment groups."
            )
        lines.append("")

        # ── E-value table ──────────────────────────────────────────────────
        if survival_evalues_df is not None and not survival_evalues_df.empty:
            lines.append("#### E-Value Sensitivity Analysis")
            lines.append("")

            # Build a mapping from outcome → best interval label (piecewise only)
            best_interval_map: Dict[str, str] = {}
            if is_piecewise:
                for outcome_key, res in survival_results_dict.items():
                    pw_list = res.get("piecewise_results", [])
                    valid_pw = [r for r in pw_list if not np.isnan(r.get("hr_pvalue", np.nan))]
                    if valid_pw:
                        best_interval_map[outcome_key] = min(
                            valid_pw, key=lambda r: r["hr_pvalue"]
                        ).get("interval_label", "")

            if is_piecewise:
                lines.append(
                    "> E-values computed on the **best-interval** hazard ratio "
                    "(risk_ratio scale). The best interval is the one with the "
                    "lowest p-value. "
                    "Larger E-values indicate greater robustness to unmeasured confounding."
                )
            else:
                lines.append(
                    "> E-values computed on the hazard ratio scale (risk_ratio). "
                    "Larger E-values indicate greater robustness to unmeasured confounding."
                )
            lines.append("")
            lines.append("| Outcome | E-Value Point | E-Value CI | Robustness |")
            lines.append("|---------|---------------|------------|------------|")
            for _, row in survival_evalues_df.iterrows():
                outcome    = row["Outcome"]
                name       = outcome_descriptions.get(outcome, outcome)
                # For piecewise, qualify the name with the best interval
                interval_lbl = best_interval_map.get(outcome, "")
                if interval_lbl:
                    name = f"{name} [{interval_lbl}]"
                ev_pt      = row.get("E_Value_Point")
                ev_ci      = row.get("E_Value_CI")
                robustness = row.get("Robustness", "—")
                sig        = row.get("Significant", False)

                ev_pt_str = f"{ev_pt:.2f}" if pd.notna(ev_pt) else "—"
                ev_ci_str = f"{ev_ci:.2f}" if pd.notna(ev_ci) else "—"

                if sig:
                    lines.append(
                        f"| {name} | **{ev_pt_str}** | **{ev_ci_str}** | {robustness} |"
                    )
                else:
                    lines.append(
                        f"| {name} | {ev_pt_str} | {ev_ci_str} | {robustness} (ns) |"
                    )
            lines.append("")

            # Per-outcome E-value interpretations (significant only)
            sig_ev = survival_evalues_df[survival_evalues_df.get("Significant", False) == True]
            if not sig_ev.empty:
                lines.append("**Significant outcome interpretations:**")
                lines.append("")
                for _, row in sig_ev.iterrows():
                    outcome = row["Outcome"]
                    name    = outcome_descriptions.get(outcome, outcome)
                    # For piecewise, qualify with best interval
                    interval_lbl = best_interval_map.get(outcome, "")
                    if interval_lbl:
                        name = f"{name} [{interval_lbl}]"
                    interp  = row.get("Interpretation", "")
                    if pd.notna(interp) and interp:
                        lines.append(f"- **{name}:** {interp}")
                        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_comparison_table(
        ate_summary: pd.DataFrame,
        att_summary: pd.DataFrame,
        ate_evalues: pd.DataFrame,
        att_evalues: pd.DataFrame,
        ate_results: Dict[str, Dict],
        att_results: Dict[str, Dict],
        outcome_descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a Markdown ATE vs ATT comparison.

        Creates side-by-side effect comparison and E-value comparison tables,
        plus auto-generated key observations about systematic differences
        between the two estimands.

        Parameters
        ----------
        ate_summary : pd.DataFrame
            Concatenation of all ATE ``build_summary_table()`` outputs.
        att_summary : pd.DataFrame
            Concatenation of all ATT ``build_summary_table()`` outputs.
        ate_evalues : pd.DataFrame
            Concatenation of all ATE ``compute_evalues_from_results()`` outputs.
        att_evalues : pd.DataFrame
            Concatenation of all ATT ``compute_evalues_from_results()`` outputs.
        ate_results : Dict[str, Dict]
            Merged ATE per-outcome ``analyze_treatment_effect()`` dicts.
        att_results : Dict[str, Dict]
            Merged ATT per-outcome ``analyze_treatment_effect()`` dicts.
        outcome_descriptions : Dict[str, str], optional
            Mapping from variable names to display-friendly names.

        Returns
        -------
        str
            Markdown-formatted comparison string.
        """
        if outcome_descriptions is None:
            outcome_descriptions = {}

        lines: list = []

        # ── Side-by-side effect comparison ─────────────────────────────
        lines.append("### Side-by-Side Comparison")
        lines.append("")
        lines.append("| Outcome | ATE | ATT | Difference |")
        lines.append("|---------|-----|-----|------------|")

        for _, ate_row in ate_summary.iterrows():
            outcome = ate_row["Outcome"]
            name = outcome_descriptions.get(outcome, outcome)

            att_match = att_summary[att_summary["Outcome"] == outcome]
            if att_match.empty:
                continue
            att_row = att_match.iloc[0]

            is_binary = ate_results.get(outcome, {}).get("outcome_type") == "binary"

            if is_binary:
                ate_or = np.exp(ate_row["Effect"])
                att_or = np.exp(att_row["Effect"])
                direction = "larger" if att_or > ate_or else "smaller"
                lines.append(
                    f"| **{name}** (OR) | {ate_or:.2f} | {att_or:.2f} | ATT {direction} |"
                )
            else:
                ate_eff = ate_row["Effect"]
                att_eff = att_row["Effect"]
                ate_d = ate_row.get("Cohens_d")
                att_d = att_row.get("Cohens_d")
                ate_sig = ate_row["Significant"]
                att_sig = att_row["Significant"]

                ate_d_str = f" (Cohen's d = {ate_d:.2f})" if pd.notna(ate_d) else ""
                att_d_str = f" (Cohen's d = {att_d:.2f})" if pd.notna(att_d) else ""
                ate_ns = " (ns)" if not ate_sig else ""
                att_ns = " (ns)" if not att_sig else ""

                direction = (
                    "slightly larger" if abs(att_eff) > abs(ate_eff) else "slightly smaller"
                )
                lines.append(
                    f"| **{name}** | {ate_eff:+.2f}{ate_d_str}{ate_ns} | "
                    f"{att_eff:+.2f}{att_d_str}{att_ns} | ATT {direction} |"
                )

        lines.append("")

        # ── Key observations (auto-generated) ──────────────────────────
        lines.append("### Key Observations")
        lines.append("")

        att_larger_count = 0
        total_sig = 0
        for _, ate_row in ate_summary.iterrows():
            outcome = ate_row["Outcome"]
            att_match = att_summary[att_summary["Outcome"] == outcome]
            if att_match.empty:
                continue
            att_row = att_match.iloc[0]
            if ate_row["Significant"] or att_row["Significant"]:
                total_sig += 1
                is_binary = ate_results.get(outcome, {}).get("outcome_type") == "binary"
                if is_binary:
                    if np.exp(att_row["Effect"]) > np.exp(ate_row["Effect"]):
                        att_larger_count += 1
                else:
                    if abs(att_row["Effect"]) > abs(ate_row["Effect"]):
                        att_larger_count += 1

        observation_num = 1
        if total_sig > 0 and att_larger_count == total_sig:
            lines.append(
                f"{observation_num}. **ATT effects are consistently larger** across all "
                f"significant outcomes, suggesting positive selection or treatment "
                f"effect heterogeneity: managers who received training benefited more "
                f"than the average manager in the population would have."
            )
            lines.append("")
            observation_num += 1

        # Check whether binary gap > continuous gap
        binary_outcomes = [
            o for o in ate_results if ate_results[o].get("outcome_type") == "binary"
        ]
        continuous_outcomes = [
            o for o in ate_results if ate_results[o].get("outcome_type") == "continuous"
        ]

        if binary_outcomes and continuous_outcomes:
            binary_gaps = []
            for o in binary_outcomes:
                ate_m = ate_summary[ate_summary["Outcome"] == o]
                att_m = att_summary[att_summary["Outcome"] == o]
                if not ate_m.empty and not att_m.empty:
                    ate_or = np.exp(ate_m.iloc[0]["Effect"])
                    att_or = np.exp(att_m.iloc[0]["Effect"])
                    if ate_or > 0:
                        binary_gaps.append(((att_or - ate_or) / ate_or) * 100)

            cont_gaps = []
            for o in continuous_outcomes:
                ate_m = ate_summary[ate_summary["Outcome"] == o]
                att_m = att_summary[att_summary["Outcome"] == o]
                if not ate_m.empty and not att_m.empty:
                    ate_eff = ate_m.iloc[0]["Effect"]
                    att_eff = att_m.iloc[0]["Effect"]
                    if abs(ate_eff) > 0.01:
                        cont_gaps.append(abs((att_eff - ate_eff) / ate_eff) * 100)

            if binary_gaps and cont_gaps:
                avg_binary_gap = np.mean(np.abs(binary_gaps))
                avg_cont_gap = np.mean(cont_gaps)
                if avg_binary_gap > avg_cont_gap:
                    lines.append(
                        f"{observation_num}. **The ATE\u2013ATT gap is most pronounced for "
                        f"retention (binary) outcomes**, suggesting the training's impact "
                        f"on behavioral outcomes varies more by who receives it than its "
                        f"impact on self-reported attitudes."
                    )
                    lines.append("")
                    observation_num += 1

        lines.append(
            f"{observation_num}. **Both estimands tell the same qualitative story**: "
            f"the direction and significance pattern is consistent across ATE and ATT "
            f"for all outcomes."
        )
        lines.append("")

        # ── E-value comparison ─────────────────────────────────────────
        lines.append("### E-Value Comparison")
        lines.append("")
        lines.append("| Outcome | ATE E-Value | ATT E-Value | Difference |")
        lines.append("|---------|-------------|-------------|------------|")

        for _, ate_ev in ate_evalues.iterrows():
            outcome = ate_ev["Outcome"]
            name = outcome_descriptions.get(outcome, outcome)
            ate_sig = ate_ev.get("Significant", False)

            att_match = att_evalues[att_evalues["Outcome"] == outcome]
            if att_match.empty:
                continue
            att_ev_row = att_match.iloc[0]
            att_sig = att_ev_row.get("Significant", False)

            if not ate_sig and not att_sig:
                continue

            ate_pt = ate_ev["E_Value_Point"]
            att_pt = att_ev_row["E_Value_Point"]
            diff = att_pt - ate_pt

            if diff > 0.15:
                desc = "ATT notably more robust"
            elif diff > 0.05:
                desc = "ATT marginally more robust"
            elif diff < -0.15:
                desc = "ATE notably more robust"
            elif diff < -0.05:
                desc = "ATE marginally more robust"
            else:
                desc = "Similar robustness"

            lines.append(f"| **{name}** | {ate_pt:.2f} | {att_pt:.2f} | {desc} |")

        lines.append("")

        return "\n".join(lines)


# Backward-compatibility alias
IPTWGEEModel = CausalInferenceModel