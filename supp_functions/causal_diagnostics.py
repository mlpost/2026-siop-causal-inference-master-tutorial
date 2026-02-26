"""
CausalDiagnostics — unified causal-inference diagnostic toolkit.

Usage:
    from supp_functions.causal_diagnostics import CausalDiagnostics
    cd = CausalDiagnostics()
    cd.help()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

# Jupyter/IPython display — fall back to print if unavailable
try:
    from IPython.display import display
except ImportError:
    display = print


class CausalDiagnostics:
    """
    Comprehensive diagnostic toolkit for causal inference analyses.

    Organised into five method groups:

    A) Pre-Modeling Diagnostics
       - check_high_intercorrelations()
       - check_vif()
       - show_low_proportion_groups()

    B) Overlap / Common-Support Diagnostics
       - check_covariate_overlap()
       - prepare_adjustment_set_for_overlap()
       - run_overlap_diagnostics()

    C) Post-Estimation Balance
       - compute_balance_df()

    D) Visualisation & I/O
       - plot_propensity_overlap()
       - save_overlap_diagnostics_summary()

    E) Help
       - help()
    """

    def __init__(self):
        self.overlap_thresholds = {
            'smd_minor': 0.1,
            'smd_moderate': 0.25,
            'smd_severe': 0.5,
            'auc_caution': 0.7,
            'auc_warning': 0.8,
            'auc_severe': 0.9,
        }

    @staticmethod
    def _validate_binary_treatment(data, treatment_var):
        """Validate treatment is strictly binary (0/1) and return clean integer series."""
        if treatment_var not in data.columns:
            raise ValueError(f"Treatment variable '{treatment_var}' not found in data.")

        treatment = data[treatment_var]
        if treatment.isna().any():
            raise ValueError(
                f"Treatment variable '{treatment_var}' contains missing values. "
                "Please impute/drop missing treatment values before diagnostics."
            )

        unique_vals = set(pd.Series(treatment).unique().tolist())
        if unique_vals != {0, 1}:
            raise ValueError(
                f"Treatment variable '{treatment_var}' must be strictly coded as 0/1. "
                f"Found values: {sorted(unique_vals)}"
            )

        n_treated = int((treatment == 1).sum())
        n_control = int((treatment == 0).sum())
        if n_treated == 0 or n_control == 0:
            raise ValueError(
                f"Treatment variable '{treatment_var}' must contain both groups. "
                f"Found treated={n_treated}, control={n_control}."
            )

        return treatment.astype(int), n_treated, n_control

    @staticmethod
    def _safe_pct(numerator, denominator):
        """Safe percentage helper."""
        return (numerator / denominator * 100) if denominator else np.nan

    @staticmethod
    def _continuous_smd(treated_vals, control_vals):
        """Compute SMD for continuous variables."""
        mean_t = treated_vals.mean()
        mean_c = control_vals.mean()
        std_t = treated_vals.std()
        std_c = control_vals.std()

        pooled_std = np.sqrt((std_t ** 2 + std_c ** 2) / 2)
        if pooled_std == 0 or np.isnan(pooled_std):
            pooled_std = 1.0

        smd = (mean_t - mean_c) / pooled_std
        return smd, mean_t, mean_c

    @staticmethod
    def _binary_smd_from_props(prop_t, prop_c):
        """Compute SMD for binary variables from treated/control proportions."""
        pooled_prop = (prop_t + prop_c) / 2
        if pooled_prop <= 0 or pooled_prop >= 1:
            return 0.0
        pooled_std = np.sqrt(pooled_prop * (1 - pooled_prop))
        return (prop_t - prop_c) / pooled_std

    @staticmethod
    def _safe_weighted_stats(values, weights):
        """Return weighted mean/variance with NaN-safe finite, non-negative weights."""
        mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
        if mask.sum() == 0:
            return np.nan, np.nan

        vals = values[mask]
        w = weights[mask]
        w_sum = w.sum()
        if w_sum <= 0:
            return np.nan, np.nan

        w_mean = np.average(vals, weights=w)
        w_var = np.average((vals - w_mean) ** 2, weights=w)
        return float(w_mean), float(w_var)

    @staticmethod
    def _cramers_v_bias_corrected(x, y):
        """Bias-corrected Cramér's V (Bergsma correction)."""
        confusion_matrix = pd.crosstab(x, y)
        if confusion_matrix.empty:
            return 0.0

        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        if n <= 1:
            return 0.0

        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2corr / denom))

    # ============================================================================
    # GROUP A — PRE-MODELING DIAGNOSTICS
    # ============================================================================

    def check_high_intercorrelations(self, df, numerical_threshold=0.7,
                                     categorical_threshold=0.7, verbose=False,
                                     exclude_vars=None):
        """
        Checks for highly correlated pairs of variables in a DataFrame.

        Uses Pearson correlation (num–num), Cramér's V (cat–cat), and
        Correlation Ratio / Eta (num–cat).

        Args:
            df: pandas DataFrame.
            numerical_threshold: Absolute correlation threshold for numerical variables (0-1).
            categorical_threshold: Cramer's V threshold for categorical variables (0-1).
            verbose: If True, print detailed messages.
            exclude_vars: A list of variable names to exclude from the analysis.

        Returns:
            dict with keys 'numerical_pairs', 'categorical_pairs', 'mixed_pairs',
            'all_high_correlation_pairs'. Each value is a list of (var_a, var_b, statistic).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if not (0 <= numerical_threshold <= 1 and 0 <= categorical_threshold <= 1):
            raise ValueError("Thresholds must be between 0 and 1.")

        if exclude_vars is None:
            exclude_vars = []
        elif not isinstance(exclude_vars, list):
            raise TypeError("exclude_vars must be a list or None.")
        elif not all(isinstance(var, str) for var in exclude_vars):
            raise ValueError("All elements in exclude_vars must be strings.")
        elif not all(var in df.columns for var in exclude_vars):
            raise ValueError("All variables in exclude_vars must be present in the DataFrame.")

        filtered_df = df.drop(columns=exclude_vars, errors='ignore')

        numerical_cols = filtered_df.select_dtypes(include=np.number).columns
        categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns

        numerical_pairs = []
        categorical_pairs = []
        mixed_pairs = []
        all_high_correlation_pairs = []

        # 1. Numerical–Numerical (Pearson)
        corr_matrix = filtered_df[numerical_cols].corr()
        for i, j in combinations(numerical_cols, 2):
            correlation = corr_matrix.loc[i, j]
            if abs(correlation) >= numerical_threshold:
                numerical_pairs.append((i, j, correlation))
                all_high_correlation_pairs.append((i, j, correlation))
                if verbose:
                    print(f"Numerical: {i} and {j} have a correlation of {correlation:.2f}")

        # 2. Categorical–Categorical (Cramér's V)
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            if min((kcorr - 1), (rcorr - 1)) == 0:
                return 0
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        for i, j in combinations(categorical_cols, 2):
            v = cramers_v(filtered_df[i], filtered_df[j])
            if v >= categorical_threshold:
                categorical_pairs.append((i, j, v))
                all_high_correlation_pairs.append((i, j, v))
                if verbose:
                    print(f"Categorical: {i} and {j} have a Cramer's V of {v:.2f}")

        # 3. Numerical–Categorical (Correlation Ratio / Eta)
        def correlation_ratio(categories, values):
            fcat = pd.Categorical(categories)
            cat_num = fcat.codes
            y_avg_array = np.zeros(len(fcat.categories))
            n_array = np.zeros(len(fcat.categories))
            for idx in range(len(fcat.categories)):
                cat_filter = (cat_num == idx)
                y_cat = values[cat_filter]
                y_avg_array[idx] = y_cat.mean()
                n_array[idx] = len(y_cat)
            y_overall_mean = values.mean()
            numerator = np.sum(n_array * ((y_avg_array - y_overall_mean) ** 2))
            denominator = np.sum((values - y_overall_mean) ** 2)
            if denominator == 0:
                return 0
            return np.sqrt(numerator / denominator)

        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                eta = correlation_ratio(filtered_df[cat_col], filtered_df[num_col])
                if eta >= numerical_threshold:
                    mixed_pairs.append((cat_col, num_col, eta))
                    all_high_correlation_pairs.append((cat_col, num_col, eta))
                    if verbose:
                        print(f"Mixed (Correlation Ratio): {cat_col} and {num_col} have an eta of {eta:.2f}")

        return {
            'numerical_pairs': numerical_pairs,
            'categorical_pairs': categorical_pairs,
            'mixed_pairs': mixed_pairs,
            'all_high_correlation_pairs': all_high_correlation_pairs
        }

    # ------------------------------------------------------------------------

    def check_vif(self, df, controls: list, treatment=None, exclude_vars=None):
        """
        Check multicollinearity using Variance Inflation Factor (VIF) for continuous
        variables and Generalized VIF (GVIF) for categorical variables.

        GVIF^(1/(2*df)) is used for categorical variables to make them comparable
        to continuous VIF values. Apply the same thresholds (5, 10) to this value.

        Args:
            df (pd.DataFrame): Dataset.
            controls (list): Control variable names.
            treatment (str, optional): Treatment variable name.
            exclude_vars (list, optional): Variables to exclude.

        Returns:
            pd.DataFrame with columns: Variable, Type, VIF/GVIF, Adjusted (GVIF^1/2df), 
            Shared Variance, Severity.
        """
        # --- Input validation ---
        if exclude_vars is None:
            exclude_vars = []
        elif not isinstance(exclude_vars, list):
            raise TypeError("exclude_vars must be a list or None.")
        elif not all(isinstance(var, str) for var in exclude_vars):
            raise ValueError("All elements in exclude_vars must be strings.")
        elif not all(var in df.columns for var in exclude_vars):
            raise ValueError("All variables in exclude_vars must be present in the DataFrame.")

        # --- Select predictors ---
        if treatment is None:
            vif_predictors = df[controls].copy()
        else:
            vif_predictors = df[[treatment] + controls].copy()

        vif_predictors = vif_predictors.drop(columns=exclude_vars, errors='ignore')

        # --- Identify categorical vs continuous variables BEFORE encoding ---
        categorical_vars = [
            col for col in vif_predictors.columns
            if vif_predictors[col].dtype == 'object' or str(vif_predictors[col].dtype) == 'category'
        ]
        continuous_vars = [col for col in vif_predictors.columns if col not in categorical_vars]

        # --- Encode and prepare matrix ---
        vif_encoded = pd.get_dummies(vif_predictors, drop_first=True)
        vif_encoded = sm.add_constant(vif_encoded)
        vif_encoded = vif_encoded.astype(float)
        X = vif_encoded.values
        col_names = vif_encoded.columns.tolist()

        # --- Compute raw VIF for all columns (excluding const) ---
        raw_vif = {
            col_names[i]: variance_inflation_factor(X, i)
            for i in range(len(col_names))
            if col_names[i] != 'const'
        }

        # --- Compute GVIF for categorical variables ---
        def compute_gvif(X, col_names, dummy_cols):
            """
            Compute GVIF for a group of dummy columns representing one categorical variable.

            GVIF = det(R_jj) * det(R_(-j)(-j)) / det(R)
            where R is the full correlation matrix, R_jj is the sub-matrix for the
            group's columns, and R_(-j)(-j) is the sub-matrix for all other columns.
            """
            all_idx = [i for i, c in enumerate(col_names) if c != 'const']
            group_idx = [i for i, c in enumerate(col_names) if c in dummy_cols]
            other_idx = [i for i in all_idx if i not in group_idx]

            if len(all_idx) < 2 or len(group_idx) == 0:
                return np.nan

            X_sub = X[:, all_idx]
            R = np.corrcoef(X_sub, rowvar=False)
            R = np.atleast_2d(R)

            if not np.isfinite(R).all():
                return np.nan

            # Re-index within the sub-matrix
            sub_col_names = [col_names[i] for i in all_idx]
            group_sub_idx = [sub_col_names.index(col_names[i]) for i in group_idx]
            other_sub_idx = [sub_col_names.index(col_names[i]) for i in other_idx]

            R_jj = R[np.ix_(group_sub_idx, group_sub_idx)]
            R_ii = R[np.ix_(other_sub_idx, other_sub_idx)] if other_sub_idx else np.array([[1.0]])

            sign_R, logdet_R = np.linalg.slogdet(R)
            sign_Rjj, logdet_Rjj = np.linalg.slogdet(R_jj)
            sign_Rii, logdet_Rii = np.linalg.slogdet(R_ii)

            if sign_R <= 0 or sign_Rjj <= 0 or sign_Rii <= 0:
                return np.nan  # Singular matrix — perfect multicollinearity

            gvif = np.exp(logdet_Rjj + logdet_Rii - logdet_R)
            return gvif

        # --- Interpret the comparable VIF/GVIF^(1/2df) value ---
        def interpret_vif(vif):
            """
            Interpret VIF value and return shared variance description and severity.
            
            Returns:
                tuple: (shared_variance_description, severity_indicator)
            """
            if vif < 1:
                return "Invalid VIF (must be ≥ 1)", "❌ Invalid"
            elif vif < 1.33:
                return "<25% of variance is shared", "✅ None"
            elif vif < 2:
                return "25%-50% of variance is shared", "✅ None"
            elif vif < 5:
                return "50%-80% of variance is shared", "⚠️ Moderate"
            elif vif < 10:
                return "80%-90% of variance is shared", "🔴 High"
            else:
                return ">90% of variance is shared", "🚨 Severe"

        # --- Build results ---
        results = []

        # Continuous variables — standard VIF
        for var in continuous_vars:
            if var in raw_vif:
                vif_val = raw_vif[var]
                shared_variance, severity = interpret_vif(vif_val)
                results.append({
                    'Variable': var,
                    'Type': 'Continuous',
                    'VIF / GVIF': round(vif_val, 4),
                    'Adjusted (GVIF^1/2df)': round(vif_val, 4),
                    'Shared Variance': shared_variance,
                    'Severity': severity
                })

        # Categorical variables — GVIF
        for var in categorical_vars:
            dummy_cols = [c for c in col_names if c.startswith(var + '_') or c == var]
            df_var = len(dummy_cols)  # degrees of freedom = number of dummies

            if df_var == 0:
                continue

            gvif = compute_gvif(X, col_names, dummy_cols)

            if gvif is None or np.isnan(gvif):
                adjusted = np.nan
                shared_variance = "Singular matrix — check for perfect multicollinearity"
                severity = "❌ Invalid"
            else:
                adjusted = round(gvif ** (1 / (2 * df_var)), 4)
                shared_variance, severity = interpret_vif(adjusted)

            results.append({
                'Variable': var,
                'Type': f'Categorical ({df_var} dummies)',
                'VIF / GVIF': round(gvif, 4) if not np.isnan(gvif) else np.nan,
                'Adjusted (GVIF^1/2df)': adjusted,
                'Shared Variance': shared_variance,
                'Severity': severity
            })

        results_df = pd.DataFrame(results)
        return results_df.sort_values(by='Adjusted (GVIF^1/2df)', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------------

    def show_low_proportion_groups(self, df, treatment, treatment_type='categorical',
                                   threshold=0.05, exclude_vars=None, bins=5,
                                   custom_bins=None):
        """
        Flag covariate subgroups where the proportion relative to the treatment
        variable falls below *threshold*.

        Args:
            df (pd.DataFrame): Modelling dataframe.
            treatment (str): Treatment column name.
            treatment_type (str): 'categorical' or 'numeric'.
            threshold (float): Proportion below which subgroups are flagged (0-1).
            exclude_vars (list, optional): Variables to exclude.
            bins (int): Bins for numeric treatment variables (default 5).
            custom_bins (list, optional): Custom bin edges.

        Returns:
            list of dicts {covariate_name: DataFrame_with_flagged_rows}.
        """
        covariates = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if exclude_vars:
            covariates = [var for var in covariates if var not in exclude_vars]

        mydata = df.copy()

        # Handle treatment type
        if treatment_type.lower() == 'numeric':
            print(f"Treating '{treatment}' as a numeric variable. Creating bins for analysis.")
            if custom_bins is not None:
                mydata[f'{treatment}_binned'] = pd.cut(mydata[treatment], bins=custom_bins)
                print(f"Using custom bins: {custom_bins}")
            else:
                try:
                    mydata[f'{treatment}_binned'] = pd.qcut(mydata[treatment], q=bins, duplicates='drop')
                    bin_counts = mydata[f'{treatment}_binned'].value_counts()
                    print(f"Created {len(bin_counts)} bins using equal frequency (qcut):")
                    for bin_name, count in bin_counts.items():
                        print(f"  {bin_name}: {count} observations")
                except ValueError:
                    mydata[f'{treatment}_binned'] = pd.cut(mydata[treatment], bins=bins)
                    bin_counts = mydata[f'{treatment}_binned'].value_counts()
                    print(f"Created {len(bin_counts)} bins using equal width (cut):")
                    for bin_name, count in bin_counts.items():
                        print(f"  {bin_name}: {count} observations")
            treatment_for_analysis = f'{treatment}_binned'
        else:
            mydata[treatment] = mydata[treatment].astype('str')
            treatment_for_analysis = treatment

        valid_covariates = []
        for var in covariates:
            if (pd.api.types.is_categorical_dtype(mydata[var])
                    or pd.api.types.is_object_dtype(mydata[var])
                    or pd.api.types.is_bool_dtype(mydata[var])
                    or mydata[var].nunique() == 2):
                valid_covariates.append(var)
            else:
                print(f"Warning: Variable '{var}' is not categorical or binary. Skipping.")

        output = []
        for c in valid_covariates:
            try:
                if not pd.api.types.is_categorical_dtype(mydata[c]):
                    mydata[c] = mydata[c].astype(str)

                mydata_comp = mydata.groupby([c, treatment_for_analysis], observed=True).size().unstack().fillna(0)
                mydata_comp_prop = mydata_comp.div(mydata_comp.sum(axis=1), axis=0)

                for col in mydata_comp_prop.columns:
                    col_name = str(col)
                    mydata_comp[f'{col_name}_prop'] = mydata_comp_prop[col]

                prop_cols = [col for col in mydata_comp.columns if '_prop' in str(col)]
                threshold_value = float(threshold)

                filtered_rows = []
                for idx, row in mydata_comp.iterrows():
                    for prop_col in prop_cols:
                        if pd.to_numeric(row[prop_col], errors='coerce') < threshold_value:
                            filtered_rows.append(idx)
                            break

                filtered_mydata = mydata_comp.loc[filtered_rows] if filtered_rows else mydata_comp.iloc[0:0]
                output.append({c: filtered_mydata})
            except Exception as e:
                print(f"Error processing variable '{c}': {str(e)}")

        for dic in output:
            if list(dic.values())[0].shape[0] > 0:
                print(f"\n{list(dic.keys())[0]}")
                display(list(dic.values())[0])
            else:
                print(f"{list(dic.keys())[0]}: No problematic values found")

        if treatment_type.lower() == 'numeric' and f'{treatment}_binned' in mydata.columns:
            mydata = mydata.drop(f'{treatment}_binned', axis=1)

        return output

    # ============================================================================
    # GROUP B — OVERLAP / COMMON-SUPPORT DIAGNOSTICS
    # ============================================================================

    def check_covariate_overlap(self, data, treatment_var, categorical_vars=None,
                                binary_vars=None, continuous_vars=None,
                                baseline_vars=None, _show_guide=True, _quiet=False):
        """
        Check covariate overlap between treatment and control groups BEFORE analysis.

        Determines if causal inference is feasible with the data by assessing whether
        the two groups occupy comparable covariate spaces.

        Handles continuous, binary, categorical, and baseline variables:
        - Continuous: SMD based on means and pooled SD
        - Binary: SMD based on proportions
        - Categorical: Chi-square test + proportion differences per category
        - Baseline: SMD for pre-treatment outcome levels (CHECK 1D)

        Parameters
        ----------
        data : pd.DataFrame
        treatment_var : str
            Name of treatment variable (1=treated, 0=control).
        categorical_vars : list, optional
        binary_vars : list, optional
        continuous_vars : list, optional
        baseline_vars : list, optional
            Baseline outcome variables (e.g., ['growth_2024']).
        _show_guide : bool, default True
            Whether to print the interpretation guide at the end.
        _quiet : bool, default False
            If True, suppress all print output (for batch processing).

        Returns
        -------
        dict   Diagnostic results including overlap metrics and recommendations.
        """

        def _print(msg=""):
            """Internal print wrapper that respects _quiet flag."""
            if not _quiet:
                print(msg)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")

        T_series, n_treated, n_control = self._validate_binary_treatment(data, treatment_var)
        T = T_series.values

        # --- Retrieve all thresholds from config ---
        smd_minor    = self.overlap_thresholds['smd_minor']
        smd_moderate = self.overlap_thresholds['smd_moderate']
        smd_severe   = self.overlap_thresholds['smd_severe']
        auc_caution  = self.overlap_thresholds['auc_caution']
        auc_warning  = self.overlap_thresholds['auc_warning']
        auc_severe   = self.overlap_thresholds['auc_severe']

        # Configurable thresholds for categorical variables
        cramers_v_minor    = self.overlap_thresholds.get('cramers_v_minor',    0.1)
        cramers_v_moderate = self.overlap_thresholds.get('cramers_v_moderate', 0.2)
        cramers_v_severe   = self.overlap_thresholds.get('cramers_v_severe',   0.3)
        cat_diff_moderate  = self.overlap_thresholds.get('cat_diff_moderate',  10.0)
        cat_diff_severe    = self.overlap_thresholds.get('cat_diff_severe',    20.0)

        # Configurable thresholds for multivariate overlap assessment
        mv_pct_treated_good     = self.overlap_thresholds.get('mv_pct_treated_good',     80.0)
        mv_pct_treated_moderate = self.overlap_thresholds.get('mv_pct_treated_moderate', 50.0)
        mv_pct_controls_good    = self.overlap_thresholds.get('mv_pct_controls_good',    50.0)

        # Configurable thresholds for estimand feasibility
        ate_pct_treated_min          = self.overlap_thresholds.get('ate_pct_treated_min',          85.0)
        ate_pct_controls_min         = self.overlap_thresholds.get('ate_pct_controls_min',         80.0)
        ate_with_caution_treated_min = self.overlap_thresholds.get('ate_with_caution_treated_min', 75.0)
        ate_with_caution_controls_min= self.overlap_thresholds.get('ate_with_caution_controls_min',70.0)
        att_pct_treated_min          = self.overlap_thresholds.get('att_pct_treated_min',          80.0)
        att_trimming_pct_treated_min = self.overlap_thresholds.get('att_trimming_pct_treated_min', 50.0)

        _print("\n" + "=" * 80)
        _print("COVARIATE OVERLAP DIAGNOSTIC")
        _print("=" * 80)
        _print("\nThis check determines if treatment and control groups are comparable.")
        _print("If groups occupy different covariate spaces, causal inference is problematic.")
        treatment_rate = self._safe_pct(n_treated, len(T))
        _print(f"\nSample: {n_treated} treated, {n_control} control "
            f"({treatment_rate:.2f}% treatment rate)")

        results = {
            'n_treated':      n_treated,
            'n_control':      n_control,
            'treatment_rate': treatment_rate,
        }

        all_smd         = []
        all_var_names   = []
        imbalance_details = []

        # ----------------------------------------------------------------
        # CHECK 1A: Continuous Variables — SMD
        # ----------------------------------------------------------------
        if continuous_vars:
            _print("\n" + "-" * 80)
            _print("CHECK 1A: Continuous Variables (Standardized Mean Difference)")
            _print("-" * 80)

            for var in continuous_vars:
                if var not in data.columns:
                    _print(f"  ⚠️  {var} not found in DataFrame — skipping")
                    continue

                values    = data[var].values
                val_t     = values[T == 1]
                val_c     = values[T == 0]
                val_t     = val_t[~np.isnan(val_t)]
                val_c     = val_c[~np.isnan(val_c)]

                if len(val_t) == 0 or len(val_c) == 0:
                    _print(f"  ⚠️  {var} has insufficient non-NaN data — skipping")
                    continue

                smd, mean_t, mean_c = self._continuous_smd(val_t, val_c)
                abs_smd = abs(smd)
                all_smd.append(abs_smd)
                all_var_names.append(var)

                if abs_smd >= smd_severe:
                    flag = "🚨 SEVERE"
                    imbalance_details.append(f"{var}: SMD={smd:.3f} (severe)")
                elif abs_smd >= smd_moderate:
                    flag = "⚠️  MODERATE"
                    imbalance_details.append(f"{var}: SMD={smd:.3f} (moderate)")
                elif abs_smd >= smd_minor:
                    flag = "⚡ MINOR"
                else:
                    flag = "✓"

                _print(f"  {var:<30} SMD={smd:>7.3f}  "
                    f"Treated: {mean_t:>8.2f}  Control: {mean_c:>8.2f}  {flag}")

        # ----------------------------------------------------------------
        # CHECK 1B: Binary Variables — SMD (proportion difference)
        # ----------------------------------------------------------------
        if binary_vars:
            _print("\n" + "-" * 80)
            _print("CHECK 1B: Binary Variables (Proportion Difference)")
            _print("-" * 80)

            for var in binary_vars:
                if var not in data.columns:
                    _print(f"  ⚠️  {var} not found in DataFrame — skipping")
                    continue

                try:
                    values = data[var].values.astype(float)
                except (ValueError, TypeError):
                    _print(f"  ⚠️  {var} could not be converted to float — skipping")
                    continue

                clean_vals = values[~np.isnan(values)]
                if len(clean_vals) > 0 and not set(np.unique(clean_vals)).issubset({0.0, 1.0}):
                    _print(f"  ⚠️  {var}: Skipping — expected binary coding 0/1 for binary SMD.")
                    continue

                val_t = values[T == 1]
                val_c = values[T == 0]
                val_t = val_t[~np.isnan(val_t)]
                val_c = val_c[~np.isnan(val_c)]

                if len(val_t) == 0 or len(val_c) == 0:
                    _print(f"  ⚠️  {var} has insufficient non-NaN data — skipping")
                    continue

                prop_t = val_t.mean()
                prop_c = val_c.mean()
                smd    = self._binary_smd_from_props(prop_t, prop_c)
                abs_smd = abs(smd)
                all_smd.append(abs_smd)
                all_var_names.append(var)

                if abs_smd >= smd_severe:
                    flag = "🚨 SEVERE"
                    imbalance_details.append(
                        f"{var}: SMD={smd:.3f}, {prop_t*100:.1f}% vs {prop_c*100:.1f}%")
                elif abs_smd >= smd_moderate:
                    flag = "⚠️  MODERATE"
                    imbalance_details.append(
                        f"{var}: SMD={smd:.3f}, {prop_t*100:.1f}% vs {prop_c*100:.1f}%")
                elif abs_smd >= smd_minor:
                    flag = "⚡ MINOR"
                else:
                    flag = "✓"

                _print(f"  {var:<30} SMD={smd:>7.3f}  "
                    f"Treated: {prop_t*100:>6.1f}%  Control: {prop_c*100:>6.1f}%  {flag}")

        # ----------------------------------------------------------------
        # CHECK 1C: Categorical Variables — Chi-square + category-level
        # ----------------------------------------------------------------
        if categorical_vars:
            _print("\n" + "-" * 80)
            _print("CHECK 1C: Categorical Variables (Distribution Comparison)")
            _print("-" * 80)

            for var in categorical_vars:
                if var not in data.columns:
                    _print(f"  ⚠️  {var} not found in DataFrame — skipping")
                    continue
                try:
                    contingency = pd.crosstab(data[var], data[treatment_var])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    cramers_v = self._cramers_v_bias_corrected(data[var], data[treatment_var])

                    if cramers_v >= cramers_v_severe:
                        flag = "🚨 SEVERE"
                        imbalance_details.append(
                            f"{var}: Cramér's V={cramers_v:.3f} (large association)")
                    elif cramers_v >= cramers_v_moderate:
                        flag = "⚠️  MODERATE"
                        imbalance_details.append(
                            f"{var}: Cramér's V={cramers_v:.3f} (medium association)")
                    elif cramers_v >= cramers_v_minor:
                        flag = "⚡ MINOR"
                    else:
                        flag = "✓"

                    _print(f"\n  {var}:")
                    _print(f"    Chi-square p-value: {p_value:.4f}, "
                        f"Cramér's V: {cramers_v:.3f}  {flag}")

                    cat_smds = []
                    for cat in sorted(data[var].unique(), key=str):
                        mask_cat = data[var] == cat
                        prop_t   = ((mask_cat & (T == 1)).sum() / n_treated
                                    if n_treated > 0 else 0)
                        prop_c   = ((mask_cat & (T == 0)).sum() / n_control
                                    if n_control > 0 else 0)
                        diff     = (prop_t - prop_c) * 100
                        diff_flag = ""
                        if abs(diff) > cat_diff_severe:
                            diff_flag = "🚨"
                        elif abs(diff) > cat_diff_moderate:
                            diff_flag = "⚠️"
                        _print(f"      {str(cat):<25} Treated: {prop_t*100:>5.1f}%  "
                            f"Control: {prop_c*100:>5.1f}%  (diff: {diff:>+6.1f}%) {diff_flag}")
                        cat_smds.append(abs(self._binary_smd_from_props(prop_t, prop_c)))

                    if cat_smds:
                        max_cat_smd = max(cat_smds)
                        all_smd.append(max_cat_smd)
                        all_var_names.append(var)
                    _print(f"    Max per-category SMD: {max(cat_smds):.3f}" if cat_smds else "")

                except Exception as e:
                    _print(f"  ⚠️  {var}: Could not analyze - {e}")

        # ----------------------------------------------------------------
        # CHECK 1D: Baseline Outcome Variables — SMD
        # ----------------------------------------------------------------
        if baseline_vars and len(baseline_vars) > 0:
            _print("\n" + "-" * 80)
            _print("CHECK 1D: Baseline Outcome Variables (Pre-treatment Levels)")
            _print("-" * 80)
            _print("  ⚠️  Imbalance on baseline outcomes is particularly concerning!")
            _print("     It suggests selection on trajectory "
                "(people who were already changing).\n")

            for var in baseline_vars:
                if var not in data.columns:
                    _print(f"  {var:<30} NOT FOUND in dataset")
                    continue

                values = data[var].values
                val_t  = values[T == 1]
                val_c  = values[T == 0]
                val_t  = val_t[~np.isnan(val_t)]
                val_c  = val_c[~np.isnan(val_c)]

                if len(val_t) == 0 or len(val_c) == 0:
                    _print(f"  {var:<30} Insufficient data (all NaN)")
                    continue

                smd, mean_t, mean_c = self._continuous_smd(val_t, val_c)
                abs_smd = abs(smd)
                all_smd.append(abs_smd)
                all_var_names.append(var)

                if abs_smd >= smd_severe:
                    flag          = "🚨 SEVERE"
                    interpretation = "← Strong selection on pre-treatment trajectory!"
                    imbalance_details.append(
                        f"{var} (BASELINE): SMD={smd:.3f} (severe - selection on trajectory)")
                elif abs_smd >= smd_moderate:
                    flag          = "⚠️  MODERATE"
                    interpretation = "← Moderate selection on pre-treatment level"
                    imbalance_details.append(f"{var} (BASELINE): SMD={smd:.3f} (moderate)")
                elif abs_smd >= smd_minor:
                    flag          = "⚡ MINOR"
                    interpretation = "← Minor baseline difference"
                else:
                    flag          = "✓"
                    interpretation = "← Good balance on pre-treatment level"

                _print(f"  {var:<30} SMD={smd:>7.3f}  "
                    f"Treated: {mean_t:>8.2f}  Control: {mean_c:>8.2f}  "
                    f"{flag} {interpretation}")

        # ----------------------------------------------------------------
        # SUMMARY: SMD Distribution
        # ----------------------------------------------------------------
        _print("\n" + "-" * 80)
        _print("SUMMARY: Overall Covariate Balance")
        _print("-" * 80)

        if all_smd:
            all_smd_arr  = np.array(all_smd)
            n_small      = (all_smd_arr < smd_minor).sum()
            n_medium     = ((all_smd_arr >= smd_minor)    & (all_smd_arr < smd_moderate)).sum()
            n_large      = ((all_smd_arr >= smd_moderate) & (all_smd_arr < smd_severe)).sum()
            n_very_large = (all_smd_arr >= smd_severe).sum()

            _print(f"\n  Total variables checked: {len(all_smd_arr)}")
            _print(f"  Good balance    (SMD < {smd_minor}):          "
                f"{n_small:3d} ({n_small/len(all_smd_arr)*100:5.1f}%)")
            _print(f"  Minor imbalance ({smd_minor}–{smd_moderate}): "
                f"{n_medium:3d} ({n_medium/len(all_smd_arr)*100:5.1f}%)")
            _print(f"  Moderate imbal. ({smd_moderate}–{smd_severe}): "
                f"{n_large:3d} ({n_large/len(all_smd_arr)*100:5.1f}%)")
            _print(f"  Severe imbalance (>= {smd_severe}):           "
                f"{n_very_large:3d} ({n_very_large/len(all_smd_arr)*100:5.1f}%)")

            results['n_variables']          = len(all_smd_arr)
            results['n_severe_imbalance']   = int(n_very_large)
            results['pct_severe_imbalance'] = float(n_very_large / len(all_smd_arr) * 100)
            results['mean_abs_smd']         = float(all_smd_arr.mean())
            results['max_abs_smd']          = float(all_smd_arr.max())
            results['var_smd_pairs']        = list(zip(all_var_names, all_smd_arr.tolist()))

            # Carry forward for estimand logic
            all_smd = all_smd_arr

        # ----------------------------------------------------------------
        # CHECK 2: Multivariate overlap via Propensity Score
        # ----------------------------------------------------------------
        _print("\n" + "-" * 80)
        _print("CHECK 2: MULTIVARIATE OVERLAP (via Propensity Score)")
        _print("-" * 80)
        _print("""
    WHY THIS CHECK MATTERS:
    - Univariate SMDs check each variable separately
    - But poor overlap can exist in COMBINATIONS of variables
    - Example: Age OK separately, tenure OK separately, but NO controls have
        BOTH high age AND high tenure (the combination treated people have)

    The propensity score collapses ALL covariates into one dimension.
    Checking PS overlap = checking overlap in the FULL multivariate space.

    NOTE: Baseline outcome variables are INCLUDED in the PS model to capture
            selection on pre-treatment levels and trajectories.
        """)

        X_list = []
        if continuous_vars:
            for var in continuous_vars:
                if var in data.columns:
                    X_list.append(data[[var]].fillna(data[var].median()).astype(float))
        if binary_vars:
            for var in binary_vars:
                if var in data.columns:
                    X_list.append(data[[var]].fillna(0).astype(float))
        if categorical_vars:
            for var in categorical_vars:
                if var in data.columns:
                    dummies = pd.get_dummies(data[var], prefix=var,
                                            drop_first=True, dtype=float)
                    X_list.append(dummies)
        if baseline_vars and len(baseline_vars) > 0:
            for var in baseline_vars:
                if var in data.columns:
                    X_list.append(data[[var]].fillna(data[var].median()).astype(float))
                    _print(f"  Including baseline variable '{var}' in PS model")
                else:
                    _print(f"  WARNING: Baseline variable '{var}' not found, skipping")

        if X_list:
            X = pd.concat(X_list, axis=1).values
            baseline_count = len([v for v in (baseline_vars or []) if v in data.columns])
            _print(f"\n  Total features in PS model: {X.shape[1]}")
            _print(f"    - Static covariates: {X.shape[1] - baseline_count}")
            _print(f"    - Baseline outcomes: {baseline_count}")

            try:
                rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                            random_state=42, n_jobs=-1)
                min_class_count = min(n_treated, n_control)
                cv_folds = min(5, min_class_count)
                if cv_folds < 2:
                    raise ValueError(
                        "Not enough observations per treatment arm for cross-validated "
                        "propensity diagnostics. Need at least 2 in each group."
                    )

                proba = cross_val_predict(
                    rf, X, T, cv=cv_folds, method='predict_proba')[:, 1]
                auc = roc_auc_score(T, proba)

                _print(f"\n  Treatment prediction AUC: {auc:.3f}")
                _print(f"  (0.5 = random/groups identical, 1.0 = perfectly separable)")
                _print(f"\n  NOTE: This propensity score uses a Random Forest for")
                _print(f"  separability diagnostics. IPTW modeling uses logistic")
                _print(f"  regression. Overlap conclusions here are conservative")
                _print(f"  (RF captures non-linearities that logistic PS may not).")
                _print(f"  Cross-validation folds used: {cv_folds}")

                if auc > auc_severe:
                    _print(f"\n  🚨 SEVERE: AUC > {auc_severe} — groups almost perfectly separable!")
                elif auc > auc_warning:
                    _print(f"\n  ⚠️  WARNING: AUC > {auc_warning} — substantial group differences.")
                elif auc > auc_caution:
                    _print(f"\n  ⚡ CAUTION: AUC > {auc_caution} — moderate group differences.")
                else:
                    _print(f"\n  ✓ Good: AUC < {auc_caution} — reasonable multivariate overlap.")

                results['separability_auc'] = auc

                ps_treated = proba[T == 1]
                ps_control = proba[T == 0]

                _print(f"\n  Propensity score distribution:")
                _print(f"    Treated: mean={ps_treated.mean():.3f}, "
                    f"median={np.median(ps_treated):.3f}, "
                    f"range=[{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
                _print(f"    Control: mean={ps_control.mean():.3f}, "
                    f"median={np.median(ps_control):.3f}, "
                    f"range=[{ps_control.min():.3f}, {ps_control.max():.3f}]")

                # Percentile-based common support (robust to outliers)
                overlap_min = max(np.percentile(ps_treated, 2.5),
                                np.percentile(ps_control,  2.5))
                overlap_max = min(np.percentile(ps_treated, 97.5),
                                np.percentile(ps_control,  97.5))
                overlap_width = max(0, overlap_max - overlap_min)

                _print(f"\n  COMMON SUPPORT REGION (2.5th–97.5th percentile bounds):")
                _print(f"    PS range: [{overlap_min:.3f}, {overlap_max:.3f}]")
                _print(f"    Width: {overlap_width:.3f} (out of possible 1.0)")
                _print(f"    Note: Narrow width is expected with low treatment rates —")
                _print(f"    percentage-based metrics below are more informative.")

                treated_in_overlap  = int(
                    ((ps_treated >= overlap_min) & (ps_treated <= overlap_max)).sum())
                controls_in_overlap = int(
                    ((ps_control >= overlap_min) & (ps_control <= overlap_max)).sum())

                pct_treated_overlap  = self._safe_pct(treated_in_overlap,  n_treated)
                pct_controls_overlap = self._safe_pct(controls_in_overlap, n_control)

                _print(f"\n  Observations within common support:")
                _print(f"    Treated: {treated_in_overlap} of {n_treated} "
                    f"({pct_treated_overlap:.1f}%)")
                _print(f"    Control: {controls_in_overlap} of {n_control} "
                    f"({pct_controls_overlap:.1f}%)")

                treated_outside  = n_treated - treated_in_overlap
                controls_outside = n_control - controls_in_overlap

                if treated_outside > 0:
                    _print(f"\n  ⚠️  {treated_outside} treated "
                        f"({100 - pct_treated_overlap:.1f}%) are OUTSIDE common support!")
                    _print("      These individuals have NO comparable controls.")

                # Overlap assessment — percentage-based (width is misleading at low
                # treatment rates where PS scores are naturally compressed)
                _print(f"\n  MULTIVARIATE OVERLAP ASSESSMENT:")
                if (pct_treated_overlap > mv_pct_treated_good
                        and pct_controls_overlap > mv_pct_controls_good):
                    _print("    ✓ GOOD: Most observations in both groups are within "
                        "common support")
                elif pct_treated_overlap > mv_pct_treated_moderate:
                    _print("    ⚠️  MODERATE: Treated mostly covered, some "
                        "extrapolation needed")
                else:
                    _print("    🚨 POOR: Limited common support, heavy extrapolation "
                        "required")

                results['ps_overlap_width']          = overlap_width
                results['pct_treated_in_overlap']    = pct_treated_overlap
                results['pct_controls_in_overlap']   = pct_controls_overlap
                results['n_treated_outside_support'] = treated_outside
                results['n_controls_outside_support']= controls_outside
                results['propensity_scores']         = proba

            except Exception as e:
                _print(f"\n  Could not compute separability: {e}")
                results['separability_auc']          = None
                results['pct_treated_in_overlap']    = None
                results['pct_controls_in_overlap']   = None

        # ----------------------------------------------------------------
        # Estimand feasibility — based on directional overlap metrics
        # ----------------------------------------------------------------
        pct_treated_in  = results.get('pct_treated_in_overlap')
        pct_controls_in = results.get('pct_controls_in_overlap')
        auc_val         = results.get('separability_auc')
        n_severe        = results.get('n_severe_imbalance', 0)
        mean_smd        = results.get('mean_abs_smd', 0)

        if pct_treated_in is not None and pct_controls_in is not None:

            # ── Tier 1: Clean ATE ───────────────────────────────────────────
            # Both directions well-covered, low separability, no severe imbalance
            ate_clean = (
                pct_treated_in  > ate_pct_treated_min           # e.g. 85%
                and pct_controls_in > ate_pct_controls_min      # e.g. 80%
                and (auc_val is None or auc_val < auc_warning)  # AUC < 0.8
                and n_severe == 0                                # no severe SMD vars
            )

            # ── Tier 2: ATE with caution ────────────────────────────────────
            # Decent overlap in both directions but some residual concerns
            # (e.g. moderate SMD, borderline AUC, or slightly below clean thresholds)
            ate_with_caution = (
                not ate_clean
                and pct_treated_in  > ate_with_caution_treated_min   # e.g. 75%
                and pct_controls_in > ate_with_caution_controls_min  # e.g. 70%
                and (auc_val is None or auc_val < auc_warning)       # AUC < 0.8
            )

            # ── Tier 3: ATT feasible ────────────────────────────────────────
            att_feasible = (
                not ate_clean
                and not ate_with_caution
                and pct_treated_in > att_pct_treated_min  # e.g. 80%
            )

            # ── Tier 4: ATT with trimming ───────────────────────────────────
            att_with_trimming = (
                not ate_clean
                and not ate_with_caution
                and not att_feasible
                and pct_treated_in > att_trimming_pct_treated_min  # e.g. 50%
            )

            # ── Tier 5: Causal inference questionable ───────────────────────
            causal_questionable = (
                not ate_clean
                and not ate_with_caution
                and not att_feasible
                and not att_with_trimming
            )

        else:
            ate_clean = ate_with_caution = att_feasible = None
            att_with_trimming = causal_questionable = None

        results['estimand_feasibility'] = {
            'ate_clean':           ate_clean,
            'ate_with_caution':    ate_with_caution,
            'att_feasible':        att_feasible,
            'att_with_trimming':   att_with_trimming,
            'causal_questionable': causal_questionable,
            'pct_treated_in_overlap':  pct_treated_in,
            'pct_controls_in_overlap': pct_controls_in,
        }

        # ----------------------------------------------------------------
        # OVERALL ASSESSMENT
        # ----------------------------------------------------------------
        _print("\n" + "=" * 80)
        _print("OVERALL ASSESSMENT")
        _print("=" * 80)

        problems = []
        if results.get('pct_severe_imbalance', 0) > 20:
            problems.append(
                f"Many variables with severe imbalance "
                f"({results['pct_severe_imbalance']:.0f}% have SMD > {smd_severe})")
        auc_val = results.get('separability_auc')
        if auc_val is not None and auc_val > auc_severe:
            problems.append(
                f"Groups are highly separable (AUC = {auc_val:.2f} > {auc_severe})")
        if (results.get('pct_controls_in_overlap') is not None
                and results['pct_controls_in_overlap'] < 50):
            problems.append(
                f"Only {results['pct_controls_in_overlap']:.0f}% of controls "
                f"in overlap region")
        if (results.get('pct_treated_in_overlap') is not None
                and results['pct_treated_in_overlap'] < 80):
            problems.append(
                f"Only {results['pct_treated_in_overlap']:.0f}% of treated "
                f"in overlap region")
        if results.get('mean_abs_smd', 0) > smd_moderate:
            problems.append(
                f"High average imbalance (mean |SMD| = {results['mean_abs_smd']:.2f})")

        if imbalance_details:
            _print("\n  Variables with notable imbalance:")
            for detail in imbalance_details[:10]:
                _print(f"    • {detail}")
            if len(imbalance_details) > 10:
                _print(f"    ... and {len(imbalance_details) - 10} more")

        ef = results['estimand_feasibility']

        # ── Tier 1: Clean ATE ───────────────────────────────────────────────
        if ef.get('ate_clean'):
            _print("""
    ✓ GOOD OVERLAP: Bidirectional overlap is strong.

    RECOMMENDED ESTIMAND: ATE (Average Treatment Effect)
        Both groups are well-represented across the covariate space.
        ATE is estimable — you can generalise to the full analytic population.

    ATT is also valid if your research question focuses on participants only.
            """)
            results['recommendation'] = 'PROCEED'
            results['confidence']     = 'HIGH'

        # ── Tier 2: ATE with caution ────────────────────────────────────────
        elif ef.get('ate_with_caution'):
            _print(f"""
    ⚡ REASONABLE OVERLAP: ATE is defensible with appropriate methods and care.

    RECOMMENDED ESTIMAND: ATE (with caution)
        Overlap is sufficient for ATE estimation, but residual imbalance means
        your estimates will rely partly on model-based adjustment.
        The quality of your ATE estimate depends on how well you handle this.
            """)
            if problems:
                _print("  Residual concerns:")
                for p in problems:
                    _print(f"    • {p}")
            _print(f"""
    IF PROCEEDING WITH ATE, consider the following:

    1. USE STABILIZED, TRIMMED IPTW WEIGHTS
        • Stabilized weights:  w = P(T) / PS  for treated,
                                    P(T=0) / (1-PS)  for controls
        • Trim at 99th percentile to prevent extreme weights destabilising
            your estimates
        • Check: max weight should ideally be < 10x the mean weight

    2. VERIFY EFFECTIVE SAMPLE SIZE (ESS) AFTER WEIGHTING
        • ESS = (Σw)² / Σw²  separately for treated and controls
        • Rule of thumb: ESS > 50% of actual N in each group
        • If ESS drops below 50%, weights are too concentrated —
            fall back to ATT

    3. CHECK POST-WEIGHTING BALANCE
        • All weighted SMDs should be < 0.1 after IPTW
        • If any remain > 0.1, your PS model needs improvement
            (add interactions, splines, or use a more flexible model)

    4. USE DOUBLY ROBUST ESTIMATION (AIPW)
        • Combines IPTW with an outcome model
        • Consistent if EITHER the PS model OR the outcome model is correct
        • Provides protection against PS model misspecification

    5. RUN SENSITIVITY ANALYSES
        • Compare ATE vs ATT estimates — if they diverge substantially,
            the extrapolation required for ATE is doing real work
        • Try different weight trimming thresholds (95th, 99th percentile)
        • Report both trimmed and untrimmed results

    6. BE TRANSPARENT IN REPORTING
        • State the analytic population explicitly
            (e.g. "ATE for avg+ performing managers")
        • Report ESS alongside N
        • Acknowledge that ATE relies on model-based extrapolation
            for covariate combinations underrepresented in one group
            """)
            results['recommendation'] = 'ATE_WITH_CAUTION'
            results['confidence']     = 'MEDIUM'

        # ── Tier 3: ATT feasible ────────────────────────────────────────────
        elif ef.get('att_feasible'):
            _print(f"\n  ✓ ATT FEASIBLE: Treated units are well-covered by controls.\n")
            if problems:
                _print("  Notes:")
                for p in problems:
                    _print(f"    • {p}")
            _print(f"""
    INTERPRETATION:
        • ATT (effect on the treated) is well-supported
        • ATE requires extrapolation — controls extend beyond treated space
        • This is NORMAL with imbalanced treatment rates

    RECOMMENDATIONS:
        • Target ATT as your primary estimand
        • Use matching or ATT-targeted weighting
        • If you want ATE, consider whether structural non-overlap can be
        removed (e.g. trimming covariate categories with 0% treatment)
        and re-run diagnostics on the restricted sample
            """)
            results['recommendation'] = 'ATT_FEASIBLE'
            results['confidence']     = 'HIGH'

        # ── Tier 4: ATT with trimming ────────────────────────────────────────
        elif ef.get('att_with_trimming'):
            _print(f"\n  ⚠️  ATT FEASIBLE WITH TRIMMING: Moderate treated coverage.\n")
            _print("  Issues found:")
            for p in problems:
                _print(f"    • {p}")
            _print(f"""
    INTERPRETATION:
        • Only {ef['pct_treated_in_overlap']:.0f}% of treated have comparable controls
        • ATT is feasible if you RESTRICT to the common support region
        • Trimmed ATT estimates the effect for the subset of treated
        with good matches

    RECOMMENDATIONS:
        • Restrict analysis to common support region
        • Report the proportion of treated excluded and characterise them
        • Be clear that results apply to "matchable" treated only
        • Consider whether excluded treated differ systematically
            """)
            results['recommendation'] = 'ATT_WITH_TRIMMING'
            results['confidence']     = 'MEDIUM'

        # ── Tier 5: Causal inference questionable ───────────────────────────
        elif ef.get('causal_questionable'):
            _print(f"\n  🚨 SERIOUS OVERLAP PROBLEMS: Most treated lack comparable "
                f"controls.\n")
            _print("  Issues found:")
            for p in problems:
                _print(f"    • {p}")
            _print(f"""
    INTERPRETATION:
        • Only {ef['pct_treated_in_overlap']:.0f}% of treated have comparable controls
        • Even ATT requires heavy extrapolation or severe sample restriction
        • Causal inference may not be appropriate for this data

    IMPLICATIONS:
        • Matching will exclude most treated units
        • Weighting will produce extreme / unstable weights
        • Any estimate relies heavily on modelling assumptions

    RECOMMENDATIONS:
        • Consider whether causal inference is appropriate
        • Report results as EXPLORATORY, not causal
        • Consider alternative designs (RCT, RDD, DiD, IV)
        • If proceeding, restrict to common support and acknowledge limitations
            """)
            results['recommendation'] = 'SERIOUS_CONCERNS'
            results['confidence']     = 'LOW'

        # ── Fallback (overlap metrics unavailable) ───────────────────────────
        else:
            if len(problems) == 0:
                _print("""
    ✓ GOOD BALANCE: Covariate balance appears acceptable.
        (Multivariate overlap could not be assessed — interpret with care)
                """)
                results['recommendation'] = 'PROCEED'
                results['confidence']     = 'MEDIUM'
            elif len(problems) <= 2:
                _print(f"\n  ⚠️  MODERATE CONCERNS: Some balance issues detected.\n")
                _print("  Issues found:")
                for p in problems:
                    _print(f"    • {p}")
                _print("""
    RECOMMENDATIONS:
        • Target ATT rather than ATE
        • Use doubly robust methods (AIPW) for protection against
        model misspecification
        • Report effect for COMPARABLE individuals only
        • Acknowledge extrapolation limitations
                """)
                results['recommendation'] = 'PROCEED_WITH_CAUTION'
                results['confidence']     = 'MEDIUM'
            else:
                _print(f"\n  🚨 SERIOUS CONCERNS: Multiple balance issues detected.\n")
                _print("  Issues found:")
                for p in problems:
                    _print(f"    • {p}")
                _print("""
    RECOMMENDATIONS:
        • Consider whether causal inference is appropriate
        • Report results as EXPLORATORY, not causal
        • Focus on matched/comparable subpopulation only
        • Consider alternative study designs (RCT, RDD, DiD)
                """)
                results['recommendation'] = 'SERIOUS_CONCERNS'
                results['confidence']     = 'LOW'

        results['problems']           = problems
        results['imbalance_details']  = imbalance_details

        # ----------------------------------------------------------------
        # INTERPRETATION GUIDE
        # ----------------------------------------------------------------
        if _show_guide and not _quiet:
            _print("\n" + "=" * 80)
            _print("INTERPRETATION GUIDE: WHAT THIS MEANS FOR YOUR ANALYSIS")
            _print("=" * 80)
            self._print_interpretation_guide()

        return results

    # ------------------------------------------------------------------------

    def prepare_adjustment_set_for_overlap(self, data, outcome_var, baseline_vars,
                                        categorical_vars, binary_vars,
                                        continuous_vars):
        """
        Prepare the full adjustment set including baseline variables.

        Returns baseline variables as a SEPARATE list for CHECK 1D.

        Parameters
        ----------
        data : pd.DataFrame
        outcome_var : str
            e.g. 'growth_2025'
        baseline_vars : dict
            {outcome: baseline} mapping.
        categorical_vars, binary_vars, continuous_vars : list

        Returns
        -------
        tuple  (categorical_list, binary_list, continuous_list, baseline_list)
        """
        cat_adj  = categorical_vars.copy() if categorical_vars else []
        bin_adj  = binary_vars.copy()      if binary_vars      else []
        cont_adj = continuous_vars.copy()  if continuous_vars  else []

        if baseline_vars is None:
            baseline_vars = {}
        if not isinstance(baseline_vars, dict):
            raise TypeError(
                "baseline_vars must be a dict mapping outcome -> baseline variable.")

        baseline_var  = baseline_vars.get(outcome_var)
        baseline_list = []
        if baseline_var and baseline_var in data.columns:
            baseline_list.append(baseline_var)
            print(f"  Baseline variable '{baseline_var}' will be checked separately "
                f"(CHECK 1D)")
        elif baseline_var:
            print(f"  WARNING: Baseline variable '{baseline_var}' not found in "
                f"data columns!")

        return cat_adj, bin_adj, cont_adj, baseline_list

    # ------------------------------------------------------------------------
    def run_overlap_diagnostics(self, data, treatment_var, outcome_vars,
                            baseline_vars, categorical_vars, binary_vars,
                            continuous_vars):
        """
        Run Step 2: Overlap / Common-Support Diagnostics for each outcome.

        Loops over outcome_vars, prepares per-outcome adjustment sets
        (including baseline), runs check_covariate_overlap, and recommends
        ATT vs ATE based on directional overlap metrics.

        Returns
        -------
        dict  keyed by outcome variable name + 'summary'.
        """
        if not outcome_vars or not isinstance(outcome_vars, (list, tuple)):
            raise ValueError(
                "outcome_vars must be a non-empty list/tuple of outcome names.")

        if baseline_vars is None:
            baseline_vars = {}
        elif not isinstance(baseline_vars, dict):
            raise TypeError(
                "baseline_vars must be a dict mapping outcome -> baseline variable.")

        all_results = {}

        # ── Header ────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("STEP 2: OVERLAP / COMMON SUPPORT DIAGNOSTICS")
        print("=" * 80)
        print("""
    This step determines WHICH ESTIMAND (ATT vs ATE) is credible in your data.

    KEY CONCEPTS:
    - ATT (Average Treatment Effect on the Treated): Effect for those who got treatment
    → Requires: Good overlap for TREATED units (each treated has comparable controls)

    - ATE (Average Treatment Effect): Effect if we treated everyone
    → Requires: Good overlap BOTH directions (treated exist across control range too)

    With imbalanced samples, ATT is typically the credible estimand because
    many controls sit in regions where no treated units exist.

    We'll check overlap for each outcome (since adjustment sets include different baselines).
        """)

        # ── First outcome: full verbose output ────────────────────────────
        first_outcome = outcome_vars[0]

        print("=" * 80)
        print("DETAILED DIAGNOSTICS")
        print("=" * 80)
        print("Checks 1A–1C (continuous, binary, categorical SMDs) are identical")
        print("across outcomes. Only baseline check (1D) and propensity-score model")
        print("(Check 2) vary by outcome due to outcome-specific baseline variables.")
        print(f"Showing full detail for first outcome: {first_outcome}\n")

        cat_adj, bin_adj, cont_adj, baseline_adj = (
            self.prepare_adjustment_set_for_overlap(
                data, first_outcome, baseline_vars,
                categorical_vars, binary_vars, continuous_vars))

        print(f"\nAdjustment set for {first_outcome}:")
        print(f"  Categorical ({len(cat_adj)}):  {cat_adj}")
        print(f"  Binary      ({len(bin_adj)}):  {bin_adj}")
        print(f"  Continuous  ({len(cont_adj)}): {cont_adj}")
        print(f"  Baseline    ({len(baseline_adj)}): {baseline_adj}")

        first_result = self.check_covariate_overlap(
            data=data,
            treatment_var=treatment_var,
            categorical_vars=cat_adj,
            binary_vars=bin_adj,
            continuous_vars=cont_adj,
            baseline_vars=baseline_adj,
            _show_guide=False,
            _quiet=False,
        )
        all_results[first_outcome] = first_result

        # ── Remaining outcomes: quiet computation ─────────────────────────
        for outcome_var in outcome_vars[1:]:
            cat_adj, bin_adj, cont_adj, baseline_adj = (
                self.prepare_adjustment_set_for_overlap(
                    data, outcome_var, baseline_vars,
                    categorical_vars, binary_vars, continuous_vars))

            outcome_result = self.check_covariate_overlap(
                data=data,
                treatment_var=treatment_var,
                categorical_vars=cat_adj,
                binary_vars=bin_adj,
                continuous_vars=cont_adj,
                baseline_vars=baseline_adj,
                _show_guide=False,
                _quiet=True,
            )
            all_results[outcome_var] = outcome_result

        # ── Per-outcome compact results ───────────────────────────────────
        print("\n" + "=" * 80)
        print("PER-OUTCOME RESULTS")
        print("=" * 80)

        for outcome_var in outcome_vars:
            result              = all_results[outcome_var]
            baseline_var        = baseline_vars.get(outcome_var, None)
            auc                 = result.get('separability_auc')
            overlap_pct_treated = result.get('pct_treated_in_overlap')
            overlap_pct_controls= result.get('pct_controls_in_overlap')
            max_smd             = result.get('max_abs_smd')
            rec                 = result.get('recommendation', 'UNKNOWN')
            conf                = result.get('confidence',     'UNKNOWN')

            auc_str       = f"{auc:.3f}"              if auc                 is not None else "N/A"
            overlap_t_str = f"{overlap_pct_treated:.1f}%"  if overlap_pct_treated  is not None else "N/A"
            overlap_c_str = f"{overlap_pct_controls:.1f}%" if overlap_pct_controls is not None else "N/A"
            max_smd_str   = f"{max_smd:.3f}"          if max_smd             is not None else "N/A"

            print(f"\n  {outcome_var}")
            print(f"    Baseline var : {baseline_var or '(none)'}")
            print(f"    AUC: {auc_str}  |  Treated in overlap: {overlap_t_str}"
                f"  |  Controls in overlap: {overlap_c_str}"
                f"  |  Max |SMD|: {max_smd_str}")
            print(f"    Assessment   : {rec} (confidence: {conf})")

            if overlap_pct_treated is not None and overlap_pct_treated < 80:
                n_out = result.get('n_treated_outside_support', 0)
                print(f"    ⚠️  {n_out} treated ({100 - overlap_pct_treated:.1f}%) "
                    f"outside common support")

        # ── Summary table ─────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("SUMMARY TABLE: ESTIMAND RECOMMENDATIONS BY OUTCOME")
        print("=" * 80)

        smd_minor    = self.overlap_thresholds['smd_minor']
        smd_moderate = self.overlap_thresholds['smd_moderate']
        smd_severe   = self.overlap_thresholds['smd_severe']

        summary_rows = []
        for outcome_var in outcome_vars:
            result       = all_results[outcome_var]
            auc          = result.get('separability_auc')
            pct_treated  = result.get('pct_treated_in_overlap')
            pct_controls = result.get('pct_controls_in_overlap')
            max_smd      = result.get('max_abs_smd')
            ef           = result.get('estimand_feasibility', {})

            # Imbalanced variables (|SMD| >= 0.1)
            var_smd_pairs  = result.get('var_smd_pairs', [])
            concerned      = sorted([(v, s) for v, s in var_smd_pairs
                                    if s >= smd_minor],
                                    key=lambda x: x[1], reverse=True)
            
            # Truncate long lists for readability
            if len(concerned) > 5:
                concerned_str = ('; '.join(f"{v} ({s:.2f})" for v, s in concerned[:5]) 
                                + f"; ... +{len(concerned)-5} more")
            elif concerned:
                concerned_str = '; '.join(f"{v} ({s:.2f})" for v, s in concerned)
            else:
                concerned_str = '(none)'

            # Max SMD source
            if var_smd_pairs:
                max_var, max_val = max(var_smd_pairs, key=lambda x: x[1])
                max_smd_source = f"{max_val:.3f} ({max_var})"
            else:
                max_smd_source = "N/A"

            # Baseline balance
            baseline_var      = baseline_vars.get(outcome_var)
            baseline_smd_val  = None
            if baseline_var and var_smd_pairs:
                baseline_smd_val = next((s for v, s in var_smd_pairs
                                        if v == baseline_var), None)
            
            if baseline_var is None:
                baseline_str = 'N/A (no baseline)'
            elif baseline_smd_val is None:
                baseline_str = 'N/A'
            elif baseline_smd_val < smd_minor:
                baseline_str = f'✓ Good ({baseline_smd_val:.3f})'
            elif baseline_smd_val < smd_moderate:
                baseline_str = f'⚡ Minor ({baseline_smd_val:.3f})'
            elif baseline_smd_val < smd_severe:
                baseline_str = f'⚠️ Moderate ({baseline_smd_val:.3f})'
            else:
                baseline_str = f'🚨 Severe ({baseline_smd_val:.3f})'

            # Estimand recommendation — use new tiers
            if ef.get('ate_clean'):
                estimand = 'ATE'
                reason   = 'Strong bidirectional overlap; ATE fully defensible'
            elif ef.get('ate_with_caution'):
                estimand = 'ATE (with caution)'
                reason   = ('Reasonable overlap; ATE defensible with stabilized '
                            'IPTW, ESS check, and post-weighting balance verification')
            elif ef.get('att_feasible'):
                estimand = 'ATT (recommended)'
                reason   = ('Treated well-covered by controls; ATE requires '
                            'extrapolation — consider removing structural '
                            'non-overlap if ATE is needed')
            elif ef.get('att_with_trimming'):
                estimand = 'ATT (with trimming)'
                reason   = (f"Only {pct_treated:.0f}% treated in common support; "
                            f"restrict sample" if pct_treated is not None 
                            else "Limited treated coverage; restrict sample")
            elif ef.get('causal_questionable'):
                estimand = 'Causal inference questionable'
                reason   = (f"Only {pct_treated:.0f}% treated have comparable "
                            f"controls" if pct_treated is not None
                            else "Very limited treated coverage")
            else:
                # Fallback based on recommendation string
                rec = result.get('recommendation', 'UNKNOWN')
                if rec in ('PROCEED', 'ATE_WITH_CAUTION'):
                    estimand = 'ATE (with caution)'
                    reason   = 'Reasonable balance (overlap metrics unavailable)'
                elif rec in ('ATT_FEASIBLE', 'PROCEED_WITH_CAUTION'):
                    estimand = 'ATT (recommended)'
                    reason   = 'Moderate imbalance; ATE requires extrapolation'
                else:
                    estimand = 'ATT only (with caution)'
                    reason   = 'Poor overlap; interpret results carefully'

            # Append quantitative details to rationale
            details = []
            if auc is not None:
                details.append(f"AUC={auc:.2f}")
            if pct_treated is not None:
                details.append(f"treated overlap={pct_treated:.0f}%")
            if pct_controls is not None and pct_controls < 80:
                details.append(f"control overlap={pct_controls:.0f}%")
            if max_smd is not None and max_smd > smd_moderate:
                details.append(f"max SMD={max_smd:.2f}")
            
            if details:
                reason += f" ({', '.join(details)})"

            summary_rows.append({
                'Outcome':                    outcome_var,
                'AUC':                        f"{auc:.3f}"         if auc          is not None else "N/A",
                'Treated Overlap %':          f"{pct_treated:.1f}" if pct_treated  is not None else "N/A",
                'Control Overlap %':          f"{pct_controls:.1f}"if pct_controls is not None else "N/A",
                'Max |SMD| (Source)':         max_smd_source,
                'Baseline Balance':           baseline_str,
                'Imbalanced Vars':            concerned_str,
                'Estimand':                   estimand,
                'Rationale':                  reason,
            })

        summary_df = pd.DataFrame(summary_rows)
        display(summary_df)

        # ── Interpretation guide ──────────────────────────────────────────
        print("\n" + "=" * 80)
        print("INTERPRETATION GUIDE: WHAT THIS MEANS FOR YOUR ANALYSIS")
        print("=" * 80)
        self._print_interpretation_guide()

        # ── Overall estimand recommendation ──────────────────────────────
        print("\n" + "=" * 80)
        print("OVERALL STEP 2 SUMMARY: ESTIMAND SELECTION GUIDANCE")
        print("=" * 80)

        all_ate_clean = all(
            r.get('estimand_feasibility', {}).get('ate_clean', False)
            for r in all_results.values())

        all_ate_feasible = all(
            r.get('estimand_feasibility', {}).get('ate_clean', False)
            or r.get('estimand_feasibility', {}).get('ate_with_caution', False)
            for r in all_results.values())

        all_att_feasible = all(
            r.get('estimand_feasibility', {}).get('ate_clean', False)
            or r.get('estimand_feasibility', {}).get('ate_with_caution', False)
            or r.get('estimand_feasibility', {}).get('att_feasible', False)
            for r in all_results.values())

        any_att_with_trimming = any(
            r.get('estimand_feasibility', {}).get('att_with_trimming', False)
            for r in all_results.values())

        any_causal_questionable = any(
            r.get('estimand_feasibility', {}).get('causal_questionable', False)
            for r in all_results.values())

        avg_treated_overlap = (
            np.mean([r.get('pct_treated_in_overlap', 100)
                    for r in all_results.values()
                    if r.get('pct_treated_in_overlap') is not None])
            if any(r.get('pct_treated_in_overlap') is not None
                for r in all_results.values()) else None)

        avg_controls_overlap = (
            np.mean([r.get('pct_controls_in_overlap', 100)
                    for r in all_results.values()
                    if r.get('pct_controls_in_overlap') is not None])
            if any(r.get('pct_controls_in_overlap') is not None
                for r in all_results.values()) else None)

        # ── Print overall guidance ────────────────────────────────────────
        if all_ate_clean:
            recommended_estimand = 'ATE'
            next_step = 'ATE via IPTW or doubly robust estimation'
            print("""
    ✓ EXCELLENT OVERLAP: Strong bidirectional overlap across all outcomes.

    RECOMMENDED ESTIMAND: ATE (Average Treatment Effect)
        Both groups are well-represented — ATE is fully defensible.

    NEXT STEPS:
        • IPTW with stabilized weights is appropriate
        • Doubly robust (AIPW) adds further protection
        • Compare ATE and ATT estimates as a robustness check
            """)

        elif all_ate_feasible and not any_causal_questionable:
            recommended_estimand = 'ATE (with caution)'
            next_step = 'Stabilized IPTW + ESS check + post-weighting balance'

            overlap_note = ""
            if avg_treated_overlap is not None and avg_controls_overlap is not None:
                overlap_note = (
                    f"\n  OVERLAP SUMMARY:\n"
                    f"    • Average treated in common support:  {avg_treated_overlap:.1f}%\n"
                    f"    • Average controls in common support: {avg_controls_overlap:.1f}%\n"
                )

            print(f"""
    ⚡ REASONABLE OVERLAP: ATE is defensible across all outcomes with care.
    {overlap_note}
    RECOMMENDED ESTIMAND: ATE (with caution)

    RATIONALE:
        • Both groups have reasonable coverage of the covariate space
        • Residual imbalance means estimates rely partly on model adjustment
        • ATE is appropriate if your question is about the broader population

    REQUIRED SAFEGUARDS FOR ATE:
        1. Stabilized, trimmed IPTW weights
        (trim at 99th percentile; check max weight < 10x mean)
        2. Effective sample size (ESS) check — must be > 50% of N in each group
        3. Post-weighting balance — all weighted SMDs must be < 0.1
        4. Doubly robust estimation (AIPW) recommended for added protection
        5. Sensitivity analysis — compare ATE vs ATT; report both if they diverge

    IF ANY SAFEGUARD FAILS → fall back to ATT

    NEXT STEPS:
        • Fit PS model (logistic regression recommended for IPTW)
        • Compute stabilized weights and trim
        • Run cd.compute_balance_df() to verify post-weighting balance
        • Proceed with IPTW GEE using robust standard errors
            """)

        elif all_att_feasible and not any_causal_questionable:
            recommended_estimand = 'ATT'
            next_step = 'ATT-targeting (matching or ATT weights)'
            print(f"""
    ✓ ATT IS WELL-SUPPORTED across all outcomes.

    RECOMMENDED ESTIMAND: ATT (Average Treatment Effect on the Treated)

    RATIONALE:
        • Each treated unit can find comparable controls (ATT feasible)
        • Controls extend into regions with no treated (ATE requires extrapolation)
        • This is normal and expected with imbalanced treatment rates

    TO PURSUE ATE INSTEAD:
        • Identify covariate categories with 0% treatment and consider removing them
        • Re-run overlap diagnostics on the restricted sample
        • If overlap improves to ATE_WITH_CAUTION tier, proceed with safeguards above

    NEXT STEPS:
        • Use ATT-targeting methods (match controls to treated, or ATT weights)
        • Your estimate represents the effect for people like those who received treatment
            """)

        elif any_att_with_trimming and not any_causal_questionable:
            recommended_estimand = 'ATT (with sample restriction)'
            next_step = 'ATT-targeting with common support trimming'
            print(f"""
    ⚠️  ATT FEASIBLE WITH TRIMMING: Some outcomes require sample restriction.

    RECOMMENDED APPROACH:
        • Target ATT with restriction to common support region
        • Some treated units will be excluded — report the proportion and
        characterise whether they differ systematically from included treated

    NEXT STEPS:
        • Apply common support restriction before matching/weighting
        • Use sensitivity analysis for trimming threshold
            """)

        elif any_causal_questionable:
            recommended_estimand = 'Causal inference may not be appropriate'
            next_step = 'Consider alternative designs or exploratory analysis'
            problematic = [
                ov for ov, r in all_results.items()
                if r.get('estimand_feasibility', {}).get('causal_questionable', False)
            ]
            print(f"""
    🚨 SERIOUS OVERLAP CONCERNS for: {', '.join(problematic)}

    IMPLICATIONS:
        • Many treated units have no comparable controls
        • Even ATT requires heavy extrapolation or severe sample restriction

    OPTIONS:
        1. EXPLORATORY: Proceed but report as exploratory, not causal
        2. RESTRICT SAMPLE: Analyse only the common support region
        3. ALTERNATIVE DESIGNS: RCT, RDD, DiD, or IV
        4. DESCRIPTIVE: Focus on descriptive rather than causal inference
            """)

        else:
            recommended_estimand = 'ATT (mixed results across outcomes)'
            next_step = 'ATT-targeting with outcome-specific diagnostics'
            print("""
    ℹ️  MIXED RESULTS: Overlap quality varies across outcomes.

    RECOMMENDED APPROACH:
        • Target ATT for all outcomes for consistency
        • Report outcome-specific overlap quality
        • Acknowledge that confidence varies by outcome
            """)

        all_results['summary'] = {
            'all_ate_clean':          all_ate_clean,
            'all_ate_feasible':       all_ate_feasible,
            'all_att_feasible':       all_att_feasible,
            'any_att_with_trimming':  any_att_with_trimming,
            'any_causal_questionable':any_causal_questionable,
            'avg_treated_overlap':    avg_treated_overlap,
            'avg_controls_overlap':   avg_controls_overlap,
            'recommended_estimand':   recommended_estimand,
            'next_step_method':       next_step,
        }

        return all_results

    # ============================================================================
    # GROUP C — POST-ESTIMATION BALANCE
    # ============================================================================

    def compute_balance_df(self, data, controls, treatment, weights,
                           already_encoded=False):
        """
        Compute covariate balance DataFrame (unweighted vs weighted).

        Used after Inverse Probability Weighting to verify that reweighting
        has improved covariate balance between treatment groups.

        For binary/dummy variables the denominator uses sqrt(p*(1-p))
        (Austin 2009); for continuous variables it uses the pooled SD
        from individual-level data.

        Args:
            data (pd.DataFrame): Dataset.
            controls (list): Control variable names.
            treatment (str): Binary treatment column name.
            weights (pd.Series): IPW weights for each observation.
            already_encoded (bool): If True, skip pd.get_dummies encoding
                (use when controls are already one-hot encoded).

        Returns:
            pd.DataFrame with columns: Unweighted Treated Mean,
            Unweighted Control Mean, Weighted Treated Mean,
            Weighted Control Mean, Unweighted SMD, Weighted SMD.
        """
        T, _, _ = self._validate_binary_treatment(data, treatment)
        if already_encoded:
            X = data[controls].copy()
        else:
            X = pd.get_dummies(data[controls], drop_first=True)
        weights = pd.Series(weights).reindex(data.index)

        invalid_weights = (~np.isfinite(weights)) | (weights < 0)
        if invalid_weights.any():
            raise ValueError(
                "Weights must be finite and non-negative for all rows in data index."
            )

        unweighted_means_treated = X.loc[T == 1].mean()
        unweighted_means_control = X.loc[T == 0].mean()

        weighted_means_treated = pd.Series(index=X.columns, dtype=float)
        weighted_means_control = pd.Series(index=X.columns, dtype=float)

        smd_unweighted = pd.Series(index=X.columns, dtype=float)
        smd_weighted = pd.Series(index=X.columns, dtype=float)

        for col in X.columns:
            treated_vals = X.loc[T == 1, col]
            control_vals = X.loc[T == 0, col]
            is_binary = set(X[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})

            # --- Unweighted means (already computed above) ---
            mean_t = unweighted_means_treated[col]
            mean_c = unweighted_means_control[col]

            # --- Unweighted SMD (per-variable, individual-level data) ---
            if is_binary:
                pooled_p = (mean_t + mean_c) / 2
                denom_uw = np.sqrt(pooled_p * (1 - pooled_p)) if 0 < pooled_p < 1 else 1.0
            else:
                # Population variance (ddof=0) to match weighted-variance
                # convention and Austin 2009 pooled-SD definition.
                var_t = np.mean((treated_vals.to_numpy(dtype=float) - mean_t) ** 2)
                var_c = np.mean((control_vals.to_numpy(dtype=float) - mean_c) ** 2)
                denom_uw = np.sqrt((var_t + var_c) / 2)
                if denom_uw == 0:
                    denom_uw = 1.0
            smd_unweighted[col] = (mean_t - mean_c) / denom_uw

            # --- Weighted means ---
            w_mean_t = np.nan
            w_mean_c = np.nan
            tw = weights.loc[T == 1].to_numpy(dtype=float)
            cw = weights.loc[T == 0].to_numpy(dtype=float)
            treated_array = treated_vals.to_numpy(dtype=float)
            control_array = control_vals.to_numpy(dtype=float)

            w_mean_t, wvar_t = self._safe_weighted_stats(treated_array, tw)
            w_mean_c, wvar_c = self._safe_weighted_stats(control_array, cw)
            weighted_means_treated[col] = w_mean_t
            weighted_means_control[col] = w_mean_c

            # --- Weighted SMD (per-variable, individual-level data) ---
            if is_binary:
                pooled_p_w = (w_mean_t + w_mean_c) / 2
                denom_w = np.sqrt(pooled_p_w * (1 - pooled_p_w)) if 0 < pooled_p_w < 1 else 1.0
            else:
                wvar_t = 0 if np.isnan(wvar_t) else wvar_t
                wvar_c = 0 if np.isnan(wvar_c) else wvar_c
                denom_w = np.sqrt((wvar_t + wvar_c) / 2)
                if denom_w == 0:
                    denom_w = 1.0
            smd_weighted[col] = (w_mean_t - w_mean_c) / denom_w

        balance_df = pd.DataFrame({
            'Unweighted Treated Mean': unweighted_means_treated,
            'Unweighted Control Mean': unweighted_means_control,
            'Weighted Treated Mean': weighted_means_treated,
            'Weighted Control Mean': weighted_means_control,
            'Unweighted SMD': smd_unweighted,
            'Weighted SMD': smd_weighted
        })

        return balance_df.round(3)

    # ============================================================================
    # GROUP D — VISUALISATION & I/O
    # ============================================================================

    def plot_propensity_overlap(self, data, treatment_var, propensity_scores,
                                outcome_var, save_path=None, title=None):
        """
        Create propensity-score overlap visualisation.

        Shows density of PS for treated vs control, common-support region,
        and regions requiring extrapolation.

        Parameters
        ----------
        data : pd.DataFrame
        treatment_var : str
        propensity_scores : np.array
        outcome_var : str   (used in plot title)
        save_path : str, optional
            Deprecated and ignored. Plot saving is disabled.
        title : str, optional
            Full plot title. If not provided, defaults to
            'Propensity Score Overlap - {outcome_var}'.

        Returns
        -------
        matplotlib.figure.Figure
        """
        T = data[treatment_var].values
        ps_treated = propensity_scores[T == 1]
        ps_control = propensity_scores[T == 0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.hist(ps_control, bins=50, alpha=0.5, label='Control',
                density=True, color='blue', edgecolor='black')
        ax.hist(ps_treated, bins=30, alpha=0.5, label='Treated',
                density=True, color='red', edgecolor='black')

        overlap_min = max(ps_treated.min(), ps_control.min())
        overlap_max = min(ps_treated.max(), ps_control.max())

        ax.axvline(overlap_min, color='green', linestyle='--', linewidth=2,
                label=f'Common support: [{overlap_min:.3f}, {overlap_max:.3f}]')
        ax.axvline(overlap_max, color='green', linestyle='--', linewidth=2)
        ax.axvspan(overlap_min, overlap_max, alpha=0.1, color='green')

        ax.set_xlabel('Propensity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        plot_title = title if title is not None else f'Propensity Score Overlap - {outcome_var}'
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.text(
            0.5,
            0.01,
            'Interpretation: better overlap between treated and control score distributions indicates more credible causal comparison.',
            ha='center',
            va='bottom',
            fontsize=9,
            color='dimgray'
        )

        plt.tight_layout(rect=[0, 0.06, 1, 1])

        plt.show()
        return fig

    # ------------------------------------------------------------------------

    def save_overlap_diagnostics_summary(self, overlap_results, save_path):
        """
        Save a concise summary of overlap diagnostics to a text file.

        Parameters
        ----------
        overlap_results : dict
            Results from ``run_overlap_diagnostics()``.
        save_path : str
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OVERLAP / COMMON SUPPORT DIAGNOSTICS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            summary = overlap_results.get('summary', {})
            f.write("OVERALL RECOMMENDATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Recommended Estimand: "
                    f"{summary.get('recommended_estimand', 'N/A')}\n")
            f.write(f"Next Step Method: "
                    f"{summary.get('next_step_method', 'N/A')}\n\n")

            for outcome_var in overlap_results.keys():
                if outcome_var == 'summary':
                    continue

                result = overlap_results[outcome_var]

                f.write("=" * 80 + "\n")
                f.write(f"OUTCOME: {outcome_var}\n")
                f.write("=" * 80 + "\n\n")

                # Sample sizes
                f.write("SAMPLE COMPOSITION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Treated:   {result.get('n_treated', 'N/A'):>6}\n")
                f.write(f"  Control:   {result.get('n_control', 'N/A'):>6}\n")
                f.write(f"  Treatment rate: "
                        f"{result.get('treatment_rate', 0):.2f}%\n\n")

                # Univariate balance
                f.write("UNIVARIATE BALANCE (SMD):\n")
                f.write("-" * 80 + "\n")
                n_vars = result.get('n_variables', 0)
                if n_vars > 0:
                    pct_severe = result.get('pct_severe_imbalance', 0)
                    f.write(f"  Total variables:          {n_vars}\n")
                    f.write(f"  Severe imbalance (>0.5):  "
                            f"{result.get('n_severe_imbalance', 0)} "
                            f"({pct_severe:.1f}%)\n")
                    f.write(f"  Mean |SMD|:               "
                            f"{result.get('mean_abs_smd', 0):.3f}\n")
                    f.write(f"  Max |SMD|:                "
                            f"{result.get('max_abs_smd', 0):.3f}\n\n")

                    if result.get('imbalance_details'):
                        f.write("  Notable imbalances:\n")
                        for detail in result['imbalance_details'][:5]:
                            f.write(f"    • {detail}\n")
                        if len(result['imbalance_details']) > 5:
                            f.write(f"    ... and "
                                    f"{len(result['imbalance_details']) - 5} more\n")
                        f.write("\n")

                # Multivariate overlap
                f.write("MULTIVARIATE OVERLAP (Propensity Score):\n")
                f.write("-" * 80 + "\n")
                auc = result.get('separability_auc')
                if auc is not None:
                    f.write(f"  Separability AUC:         {auc:.3f}\n")
                    if auc > 0.9:
                        f.write("    → SEVERE: Groups almost perfectly separable\n")
                    elif auc > 0.8:
                        f.write("    → WARNING: Substantial group differences\n")
                    elif auc > 0.7:
                        f.write("    → CAUTION: Moderate group differences\n")
                    else:
                        f.write("    → Good: Reasonable multivariate overlap\n")
                    f.write("\n")

                    f.write(f"  Common support width:     "
                            f"{result.get('ps_overlap_width', 0):.3f}\n")
                    f.write(f"  Treated in overlap:       "
                            f"{result.get('pct_treated_in_overlap', 0):.1f}%\n")
                    f.write(f"  Controls in overlap:      "
                            f"{result.get('pct_controls_in_overlap', 0):.1f}%\n")

                    n_outside = result.get('n_treated_outside_support', 0)
                    if n_outside > 0:
                        f.write(f"\n  ⚠️  {n_outside} treated units outside "
                                f"common support!\n")
                        f.write("     These require EXTRAPOLATION for causal "
                                "inference.\n")
                    f.write("\n")

                # Assessment
                f.write("ASSESSMENT:\n")
                f.write("-" * 80 + "\n")
                recommendation = result.get('recommendation', 'UNKNOWN')
                confidence = result.get('confidence', 'UNKNOWN')
                f.write(f"  Recommendation: {recommendation}\n")
                f.write(f"  Confidence:     {confidence}\n\n")

                if result.get('problems'):
                    f.write("  Issues identified:\n")
                    for problem in result['problems']:
                        f.write(f"    • {problem}\n")
                    f.write("\n")

                # Estimand guidance
                f.write("ESTIMAND GUIDANCE:\n")
                f.write("-" * 80 + "\n")
                if recommendation == 'PROCEED' and confidence == 'HIGH':
                    f.write("  ✓ Both ATT and ATE may be feasible\n")
                    f.write("    Choice depends on research question\n\n")
                elif recommendation in ['PROCEED_WITH_CAUTION', 'MODERATE CONCERNS']:
                    f.write("  ⚠️  ATT is likely feasible, ATE is questionable\n")
                    f.write("    Recommend: Target ATT (effect on the treated)\n\n")
                else:
                    f.write("  🚨 Only ATT on matched subset may be credible\n")
                    f.write("    ATE is generally not recommended\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Diagnostics summary saved to: {save_path}")


    # ============================================================================
    # PRIVATE HELPERS
    # ============================================================================

    @staticmethod
    def _print_interpretation_guide():
        """Print the detailed interpretation guide for overlap diagnostics."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  WHAT IS COVARIATE OVERLAP AND WHY DOES IT MATTER?                           ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Causal inference requires COMPARABLE treatment and control groups.          ║
    ║  "Comparable" means: for each treated person, there exist control people     ║
    ║  with similar characteristics (covariates) who can serve as counterfactuals. ║
    ║                                                                              ║
    ║  If groups occupy DIFFERENT covariate spaces, we cannot observe what would   ║
    ║  have happened to treated people without treatment — we must EXTRAPOLATE.    ║
    ║                                                                              ║
    ║  Example of POOR overlap:                                                    ║
    ║    Treated: Senior managers, age 45-60, high performers                      ║
    ║    Control: Junior staff, age 25-35, mixed performers                        ║
    ║    → No controls "look like" the treated → must extrapolate → unreliable     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  WHAT IS EXTRAPOLATION AND WHY IS IT DANGEROUS?                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  EXTRAPOLATION = Predicting outcomes for people UNLIKE anyone in the data.   ║
    ║                                                                              ║
    ║  When we estimate "what would treated person X have scored without           ║
    ║  treatment?", we use control group data. But if no controls are similar      ║
    ║  to person X, we're guessing based on dissimilar people.                     ║
    ║                                                                              ║
    ║  Visual example:                                                             ║
    ║                                                                              ║
    ║  Control data:     [●●●●●●●●]                                                ║
    ║  Treated person:                    [X]                                      ║
    ║                    ↑_______________↑                                         ║
    ║                    We have data    No data here — must extrapolate!          ║
    ║                                                                              ║
    ║  The further X is from observed controls, the more uncertain our estimate.   ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  HOW DIFFERENT METHODS HANDLE EXTRAPOLATION                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  METHOD              HOW IT EXTRAPOLATES           QUALITY OF EXTRAPOLATION  ║
    ║  ─────────────────────────────────────────────────────────────────────────── ║
    ║  OLS Regression      Extends linear trend          May be reasonable IF      ║
    ║                      mathematically                relationship is linear    ║
    ║                                                                              ║
    ║  ML (Random Forest,  Predicts CONSTANT value       POOR — predicts average   ║
    ║  Gradient Boosting)  (average of nearest points)   of nearest training data  ║
    ║                                                                              ║
    ║  IPW/DR              Reweights, but weights        POOR — weights explode,   ║
    ║                      explode for extreme cases     estimates unstable        ║
    ║                                                                              ║
    ║  PSM                 REFUSES to extrapolate        HONEST — drops cases      ║
    ║                      (drops unmatched cases)       that can't be matched     ║
    ║                                                                              ║
    ║  KEY INSIGHT: ML is often WORSE at extrapolation than simple regression!     ║
    ║  Tree-based models predict the same value for all points outside the         ║
    ║  training data range, which can severely bias treatment effect estimates.    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  WHAT THE METRICS MEAN                                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  STANDARDIZED MEAN DIFFERENCE (SMD):                                         ║
    ║    How different are group means, in standard deviation units?               ║
    ║    < 0.10 = negligible    0.10-0.25 = small    0.25-0.50 = medium            ║
    ║          > 0.50 = large                                                      ║
    ║    Rule: SMD > 0.25 on important confounders → residual bias likely          ║
    ║                                                                              ║
    ║  SEPARABILITY AUC:                                                           ║
    ║    How well can we predict treatment from covariates?                        ║
    ║    0.5 = random (groups identical)    1.0 = perfect separation               ║
    ║    Rule: AUC > 0.85 → groups are so different that overlap is minimal        ║
    ║                                                                              ║
    ║  CONTROLS IN OVERLAP:                                                        ║
    ║    What % of controls have propensity scores in the treated range?           ║
    ║    Rule: < 50% → most controls are not comparable to any treated person      ║
    ║                                                                              ║
    ║  CRAMÉR'S V (for categorical variables):                                     ║
    ║    Strength of association between category and treatment                    ║
    ║    < 0.1 = weak    0.1-0.3 = moderate    > 0.3 = strong                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  DECISION GUIDE: WHAT TO DO BASED ON OVERLAP RESULTS                         ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  GOOD OVERLAP (AUC < 0.7, most SMD < 0.25):                                  ║
    ║    → Proceed with confidence                                                 ║
    ║    → All methods (OLS, ML, PSM, IPW) should give similar results             ║
    ║    → Report ATE or ATT as appropriate                                        ║
    ║                                                                              ║
    ║  MODERATE CONCERNS (AUC 0.7-0.85, some SMD > 0.25):                          ║
    ║    → Proceed with caution                                                    ║
    ║    → Prefer OLS over ML (better extrapolation)                               ║
    ║    → Compare OLS with PSM — if they differ, report range                     ║
    ║    → Acknowledge uncertainty in the report                                   ║
    ║                                                                              ║
    ║  SERIOUS CONCERNS (AUC > 0.85, many SMD > 0.5):                              ║
    ║    → Full sample causal inference may not be appropriate                     ║
    ║    → RECOMMENDED: Estimate ATT on matched subset only                        ║
    ║       • Use PSM to find matchable treated units (may be 30-70% of sample)    ║
    ║       • Report "effect for matchable participants" not "overall effect"      ║
    ║       • This is honest and credible (no extrapolation required)              ║
    ║    → AVOID: Full sample OLS/ML (extrapolation-dependent, unreliable)         ║
    ║    → If reporting unmatched estimates, label as EXPLORATORY/DESCRIPTIVE      ║
    ║    → Consider: Is the research question answerable with this data?           ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  HONEST REPORTING WHEN OVERLAP IS POOR                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  If overlap is poor, do NOT claim causal effects. Instead:                   ║
    ║                                                                              ║
    ║  ✗ WRONG: "The program increased engagement by 0.5 points"                  ║
    ║                                                                              ║
    ║  ✓ RIGHT: "Among the subset of participants who could be matched to          ║
    ║           comparable non-participants (N=56 of 147), we observed 0.3 points  ║
    ║           higher engagement. This estimate may not generalize to all         ║
    ║           participants, particularly those in senior roles who had no        ║
    ║           comparable controls."                                              ║
    ║                                                                              ║
    ║  ✓ ALSO RIGHT: "Due to substantial differences between participant and       ║
    ║                non-participant groups on key characteristics (e.g., 85%      ║
    ║                vs 42% managers), we cannot reliably estimate the causal      ║
    ║                effect of the program. Descriptive comparisons suggest        ║
    ║                participants scored higher, but this may reflect selection    ║
    ║                rather than program impact."                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
        """)


    # ============================================================================
    # GROUP E — HELP
    # ============================================================================

    def help(self):
        """Display a summary of the CausalDiagnostics class and all its methods."""
        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  CausalDiagnostics — Unified Causal Inference Diagnostic Toolkit            ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        Usage:
            from supp_functions.causal_diagnostics import CausalDiagnostics
            cd = CausalDiagnostics()

        ────────────────────────────────────────────────────────────────────────────────
        GROUP A: PRE-MODELING DIAGNOSTICS
        ────────────────────────────────────────────────────────────────────────────────

        1. check_high_intercorrelations(df, numerical_threshold=0.7,
                                        categorical_threshold=0.7, verbose=False,
                                        exclude_vars=None)
        Detect highly correlated variable pairs using:
            • Pearson correlation     (numerical–numerical)
            • Cramér's V              (categorical–categorical)
            • Correlation Ratio / Eta (numerical–categorical)
        Returns dict with 'numerical_pairs', 'categorical_pairs', 'mixed_pairs',
        'all_high_correlation_pairs'.

        2. check_vif(df, controls, treatment=None, exclude_vars=None)
        Variance Inflation Factor for multicollinearity detection.
        Returns pd.DataFrame with Variable, VIF, Shared Variance.

        3. show_low_proportion_groups(df, treatment, treatment_type='categorical',
                                    threshold=0.05, exclude_vars=None,
                                    bins=5, custom_bins=None)
        Flag covariate subgroups where the proportion relative to treatment
        falls below a threshold. Supports categorical & numeric treatments.

        ────────────────────────────────────────────────────────────────────────────────
        GROUP B: OVERLAP / COMMON-SUPPORT DIAGNOSTICS
        ────────────────────────────────────────────────────────────────────────────────

        4. check_covariate_overlap(data, treatment_var, categorical_vars=None,
                                binary_vars=None, continuous_vars=None,
                                baseline_vars=None)
        Full overlap diagnostic:
            CHECK 1A: Continuous variables   — SMD (means & pooled SD)
            CHECK 1B: Binary variables       — SMD (proportions)
            CHECK 1C: Categorical variables  — Chi-square + Cramér's V
            CHECK 1D: Baseline outcomes      — SMD (pre-treatment levels)
            CHECK 2:  Multivariate overlap   — Propensity score via Random Forest
        Returns dict with metrics, recommendations, propensity scores.

        5. prepare_adjustment_set_for_overlap(data, outcome_var, baseline_vars,
                                            categorical_vars, binary_vars,
                                            continuous_vars)
        Build per-outcome adjustment sets, separating baseline variables.
        Returns (cat_list, bin_list, cont_list, baseline_list).

        6. run_overlap_diagnostics(data, treatment_var, outcome_vars, baseline_vars,
                                categorical_vars, binary_vars, continuous_vars)
        Loop over outcomes, run check_covariate_overlap for each, and
        recommend ATT vs ATE estimand. Returns dict keyed by outcome + 'summary'.

        ────────────────────────────────────────────────────────────────────────────────
        GROUP C: POST-ESTIMATION BALANCE
        ────────────────────────────────────────────────────────────────────────────────

        7. compute_balance_df(data, controls, treatment, weights)
        IPW covariate balance: unweighted & weighted means + SMDs.
        Returns pd.DataFrame.

        ────────────────────────────────────────────────────────────────────────────────
        GROUP D: VISUALISATION & I/O
        ────────────────────────────────────────────────────────────────────────────────

        8. plot_propensity_overlap(data, treatment_var, propensity_scores,
                                outcome_var, save_path=None)
        Histogram of propensity scores for treated vs control with
        common-support shading. Returns matplotlib Figure.

        9. save_overlap_diagnostics_summary(overlap_results, save_path)
        Write plain-text summary report of overlap diagnostics.

        ────────────────────────────────────────────────────────────────────────────────
        TYPICAL WORKFLOW
        ────────────────────────────────────────────────────────────────────────────────

        cd = CausalDiagnostics()

        # Step 1 — Pre-modeling checks
        cd.check_high_intercorrelations(df)
        cd.check_vif(df, controls=my_controls, treatment='T')
        cd.show_low_proportion_groups(df, treatment='T')

        # Step 2 — Overlap / common support
        overlap = cd.run_overlap_diagnostics(
            df, 'T', outcome_vars, baseline_vars,
            categorical_vars, binary_vars, continuous_vars)
        cd.save_overlap_diagnostics_summary(overlap, 'overlap_report.txt')

        # Step 3 — After IPW / matching
        cd.compute_balance_df(df, controls, 'T', weights)
                """)