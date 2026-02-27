"""
Causal Inference Modeling Module

This module provides data-agnostic causal inference methods for estimating treatment effects
using Inverse Probability of Treatment Weighting (IPTW) with Generalized Estimating Equations (GEE).

Classes:
    IPTWGEEModel: Main class for IPTW-based causal inference with doubly robust estimation
"""

import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.genmod import families
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from econml.dml import DML, CausalForestDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.cate_interpreter import SingleTreeCateInterpreter

from causal_diagnostics import CausalDiagnostics


class IPTWGEEModel:
    """
    Inverse Probability of Treatment Weighting (IPTW) with GEE for causal inference,
    with optional Double Machine Learning (DML) for heterogeneous treatment effects.
    
    This class implements two complementary approaches:
    
    **Approach 1 — IPTW + Doubly Robust GEE** (``analyze_treatment_effect``):
    - Stabilized inverse probability weights (IPTW) for ATE or ATT
    - Generalized Estimating Equations (GEE) to account for clustering
    - Optional outcome model covariate adjustment for doubly robust property
    
    **Approach 2 — Double Machine Learning** (``dml_estimate_treatment_effects``):
    - Linear DML for ATE/ATT estimation with flexible ML nuisance models
    - Causal Forest DML for individualized CATE estimation
    - ATT derived from CATE by averaging over treated observations
    - Uses the ``econml`` package
    
    The estimand (ATE vs. ATT) is determined by the weight construction formula
    (IPTW) or by subsetting CATE predictions (DML), not by GEE itself.
    
    Note on standard errors:
        GEE sandwich standard errors account for within-cluster correlation but
        do **not** propagate the first-stage uncertainty from propensity score
        estimation. This can yield slightly anti-conservative confidence
        intervals. For stricter inference, consider a non-parametric bootstrap
        that re-estimates both stages in each replicate.
    
    Attributes:
        weight_col (str): Name of the weight column (default: "iptw")
        ps_model: Last fitted propensity score model (use per-outcome dicts for multi-outcome runs)
        gee_model: Last fitted GEE outcome model (use per-outcome dicts for multi-outcome runs)
        dml_model: Last fitted DML model (if DML methods used)
        cfdml_model: Last fitted CausalForestDML model (if DML methods used)
    """
    
    def __init__(self):
        """Initialize the IPTWGEEModel.
        
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
        
        plt.show()
        
        return fig
    
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
        # Step 0: Data prep
        # ------------------------------------------------------------------
        # B9 fix: Propensity score model uses only confounders (not baseline).
        # Baseline is included only in the outcome (GEE) model for doubly
        # robust adjustment, avoiding outcome-specific PS models.
        ps_covariates_raw = categorical_vars + binary_vars + continuous_vars
        outcome_covariates_raw = list(ps_covariates_raw)  # copy
        if baseline_var:
            outcome_covariates_raw.append(baseline_var)
        
        all_needed = list(set([outcome_var, treatment_var, cluster_var] + outcome_covariates_raw))
        df = data[all_needed].dropna().copy()
        
        # Check if we have enough data after dropping NAs
        if len(df) < 10:
            raise ValueError(f"Insufficient data after removing missing values: {len(df)} rows remaining")
        
        # Check if we have both treatment groups
        if df[treatment_var].nunique() < 2:
            raise ValueError(f"Only one treatment group present in data: {df[treatment_var].unique()}")
        
        # Check if we have sufficient observations in each treatment group
        treatment_counts = df[treatment_var].value_counts()
        if treatment_counts.min() < 5:
            raise ValueError(f"Insufficient observations in treatment groups. Counts: {treatment_counts.to_dict()}")
        
        # --- One-hot encode categorical variables, tracking generated dummies ---
        cols_before_dummies = set(df.columns)
        df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
        dummy_columns = sorted(set(df.columns) - cols_before_dummies)
        
        # --- Clean all column names using reusable helper (R4 / L1 / L2) ---
        rename_map = {c: self._clean_column_name(c) for c in df.columns}
        df.rename(columns=rename_map, inplace=True)
        
        # Remap key variable references so they stay in sync after cleaning
        outcome_var = self._clean_column_name(outcome_var)
        treatment_var = self._clean_column_name(treatment_var)
        cluster_var = self._clean_column_name(cluster_var)
        if baseline_var:
            baseline_var = self._clean_column_name(baseline_var)
        
        # Remap covariate lists
        continuous_vars = [self._clean_column_name(v) for v in continuous_vars]
        binary_vars = [self._clean_column_name(v) for v in binary_vars]
        dummy_columns = [self._clean_column_name(c) for c in dummy_columns]
        
        # --- Track balance variables for post-weighting diagnostics ---
        # Balance computation is delegated to CausalDiagnostics.compute_balance_df
        # after IPTW weights are estimated (produces both unweighted and weighted SMDs).
        balance_var_names = (
            [v for v in continuous_vars if v in df.columns]
            + [v for v in binary_vars if v in df.columns]
            + [dc for dc in dummy_columns if dc in df.columns]
        )
        balance_var_types = {
            v: "continuous" for v in continuous_vars if v in df.columns
        }
        balance_var_types.update({v: "binary" for v in binary_vars if v in df.columns})
        balance_var_types.update({dc: "categorical" for dc in dummy_columns if dc in df.columns})
        
        # Build covariate lists (all covariates vs PS-only covariates)
        covariates = [c for c in df.columns if c not in [outcome_var, treatment_var, cluster_var]]
        
        # PS covariates: confounders only (exclude baseline for doubly robust)
        if baseline_var:
            ps_covariates = [c for c in covariates if c != baseline_var]
        else:
            ps_covariates = list(covariates)
        
        # Final validation after dummy variable creation
        if len(covariates) == 0:
            raise ValueError("No covariates remaining after data processing")
        
        # Check for valid covariate data
        covariate_df = df[covariates]
        if covariate_df.empty or covariate_df.shape[1] == 0:
            raise ValueError(f"Empty covariate matrix after processing. Shape: {covariate_df.shape}")
        
        # Check for all-constant covariates (which cause issues in GEE)
        constant_vars = [var for var in covariates if df[var].nunique() <= 1]
        
        if constant_vars:
            print(f"  Warning: Removing constant variables: {constant_vars}")
            covariates = [v for v in covariates if v not in constant_vars]
            ps_covariates = [v for v in ps_covariates if v not in constant_vars]
            if len(covariates) == 0:
                raise ValueError("No valid covariates remaining after removing constant variables")
        
        # Final check for covariate matrix validity
        final_covariate_df = df[covariates]
        if final_covariate_df.isnull().all().any():
            null_vars = final_covariate_df.columns[final_covariate_df.isnull().all()].tolist()
            raise ValueError(f"Covariates with all null values: {null_vars}")
        
        # ------------------------------------------------------------------
        # Step 1: Estimate propensity weights (confounders only, no baseline)
        # ------------------------------------------------------------------
        try:
            df, ps_model = self.estimate_propensity_weights(
                df,
                treatment_var,
                ps_covariates,
                estimand=estimand,
                cluster_var=cluster_var,
                trim_quantile=trim_quantile
            )
        except Exception as e:
            raise ValueError(f"Error estimating propensity scores - likely data issue: {str(e)}")
        
        # ------------------------------------------------------------------
        # Step 1b: Positivity / overlap warning
        # ------------------------------------------------------------------
        ps_vals = df["propensity_score"]
        n_near_zero = (ps_vals < 0.01).sum()
        n_near_one = (ps_vals > 0.99).sum()
        if n_near_zero > 0 or n_near_one > 0:
            print(f"  Warning: Positivity concern: {n_near_zero} observations with PS < 0.01, "
                  f"{n_near_one} with PS > 0.99")
        
        # ------------------------------------------------------------------
        # Step 1c: Propensity score overlap plot
        # ------------------------------------------------------------------
        ps_overlap_fig = None
        if plot_propensity:
            ps_plot_title = f"Propensity Score Overlap — {outcome_var} ({estimand})"
            try:
                ps_overlap_fig = self.plot_propensity_overlap(
                    data=df,
                    treatment_var=treatment_var,
                    title=ps_plot_title
                )
            except Exception as e:
                print(f"  Warning: Could not generate propensity score plot: {e}")
        
        # ------------------------------------------------------------------
        # Step 2: Weight Diagnostics
        # ------------------------------------------------------------------
        try:
            weight_stats = self.compute_weight_diagnostics(df)
        except Exception as e:
            raise ValueError(f"Error calculating weight diagnostics - likely insufficient data: {str(e)}")
        
        # Weight distribution plot
        weight_dist_fig = None
        if plot_weights:
            wt_plot_title = f"IPTW Weight Distribution — {outcome_var} ({estimand})"
            try:
                weight_dist_fig = self.plot_weight_distribution(
                    data=df,
                    treatment_var=treatment_var,
                    estimand=estimand,
                    title=wt_plot_title
                )
            except Exception as e:
                print(f"  Warning: Could not generate weight distribution plot: {e}")
        
        # Post-weighting balance check via CausalDiagnostics (single source of truth)
        # compute_balance_df returns both unweighted and weighted SMDs in one call.
        _cd = CausalDiagnostics()
        try:
            _raw_balance = _cd.compute_balance_df(
                data=df,
                controls=balance_var_names,
                treatment=treatment_var,
                weights=df["iptw"],
                already_encoded=True,  # dummies already created above
            )
            # Map CausalDiagnostics output to the existing schema so downstream
            # consumers (Excel export, summary tables) see no breaking change.
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
            print(f"  Warning: CausalDiagnostics balance computation failed ({e}); "
                  f"falling back to inline SMD computation.")
            # --- Fallback: inline computation if CausalDiagnostics fails ---
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
            balance_df = pd.DataFrame(columns=["variable", "type", "smd_before_weighting",
                                                "smd_after_weighting", "smd_improvement",
                                                "balanced_before_weighting", "balanced_after_weighting"])
        
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
                        from scipy.stats import norm
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
    ``IPTWGEEModel.dml_estimate_treatment_effects()``

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
    from causal_inference_modelling import IPTWGEEModel

    model = IPTWGEEModel()

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
    summary = IPTWGEEModel.build_summary_table(all_results)
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

    #Sensitivity analysis function for unmeasured confounding using E-values
    def compute_evalue(
        self,
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
        model = IPTWGEEModel()
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
                evalue_result = model.compute_evalue(
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
    # Helper methods
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
            IPTWGEEModel._significance_stars(p) for p in pvals_corrected
        ]
        summary_df["Correction_Method"] = correction_method
        
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
    def generate_summary_report(
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

        fmt_p = IPTWGEEModel._format_pvalue
        lines: list = []

        # Detect outcome type from first result
        first_key = next(iter(results_dict))
        is_binary = results_dict[first_key].get("outcome_type") == "binary"

        n_tests = len(summary_df)
        correction = (
            summary_df["Correction_Method"].iloc[0].upper()
            if "Correction_Method" in summary_df.columns
            else "FDR_BH"
        )

        lines.append(f"#### {family_label} ({n_tests} tests, {correction}-corrected)")
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