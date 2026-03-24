# Causal Inference Workshop Notebook — Change Specification

## Purpose

This document specifies all adjustments to be made to the SIOP 2026 Causal Inference Workshop notebook (`causal_inference_workshop.ipynb`) and its supporting modules (`supp_functions/causal_inference_modelling.py` and `supp_functions/causal_diagnostics.py`). Changes aim to improve methodological accuracy, tighten the presentation for a 60+20 minute session, and fix several analytical issues identified during review.

## Files in scope

| File | Role |
|------|------|
| `causal_inference_workshop.ipynb` | Main workshop notebook (45 cells) |
| `supp_functions/causal_inference_modelling.py` | `CausalInferenceModel` class (~4500 lines) |
| `supp_functions/causal_diagnostics.py` | `CausalDiagnostics` class (~2460 lines) |

## General instructions for the AI agent

- Preserve all existing functionality unless explicitly told to remove or modify it.
- When inserting markdown cells, match the existing notebook style: use `##` and `###` headings, use markdown tables, use bold for emphasis.
- When modifying Python code, maintain the existing code style (docstring format, variable naming conventions, print statement patterns).
- After all changes, verify the notebook runs end-to-end without errors. The notebook clones a GitHub repo for data — ensure no references to removed functions break.
- All markdown text provided below is final copy unless marked `[AGENT: compute and insert]`. For those placeholders, run the relevant code cell and substitute the actual computed values.

---

## CHANGE 1: Reframe the low-performer exclusion

### Problem
Cell 10 silently drops all "Far Below" and "Below" performers with a one-line comment. This hides a positivity violation and silently changes the estimand.

### What to do

**Step 1:** Do NOT remove the exclusion. Keep it, but move it AFTER the initial diagnostics so participants can see the positivity violation before it's fixed.

**Step 2:** Replace the current Cell 10 markdown:

```
!! We observe that no low performers completed training! We will remove them from the analysis. !!
```

With this expanded markdown cell:

```markdown
### Positivity Violation: No Low Performers in the Treated Group

We observe that **zero managers rated "Far Below" or "Below" completed the training program.** This is a **positivity violation** — for this subgroup, the probability of treatment is exactly zero: $P(T=1 \mid \text{performance} \in \{\text{Far Below, Below}\}) = 0$.

This means we **cannot estimate the treatment effect for low performers** because there are no treated low performers to learn from. No statistical method can fix this — it is a fundamental data limitation.

**Our decision:** We exclude low performers from the analysis. This is a defensible response to positivity violations, but it has an important consequence: **our ATE now applies only to average-and-above performers.** If L&D scales this program to include low performers, our estimates do not cover that subgroup.

This is an **estimand-scoping decision**, not a data cleaning step. It should be documented in any report and flagged as a limitation.
```

**Step 3:** Keep the code cell (Cell 11) that performs the exclusion unchanged. Keep the re-run of descriptives after exclusion, but refactor the duplicate descriptive code into a function to avoid the ~80-line copy-paste. Define the function once before Cell 9:

```python
def run_descriptive_comparison(data, treatment_col="treatment"):
    """Run demographic comparison between treated and control groups."""
    # [move the body of the descriptive exploration code here]
    ...
```

Then Cell 9 becomes `run_descriptive_comparison(data)` and Cell 11 becomes:

```python
# Exclude low performers (positivity violation — see discussion above)
data = data[~data["performance_rating"].isin(["Far Below", "Below"])].copy().reset_index(drop=True)
print(f"After exclusion: {data.shape[0]} managers remain")
print(data.groupby('treatment').size())

# Re-run descriptives on restricted sample
run_descriptive_comparison(data)
```

---

## CHANGE 2: Rename `fit_doubly_robust_model` and soften DR claims

### Problem
The method name and markdown text claim "doubly robust" estimation, but the implementation is IPTW-weighted covariate-adjusted GEE — not the formal AIPW estimator.

### What to do

**Step 1: In `causal_inference_modelling.py`**

Rename the method `fit_doubly_robust_model` → `fit_iptw_outcome_model`.

Update the docstring to:

```python
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
    model — a property sometimes called double robustness under linearity
    (Lunceford & Davidian, 2004). This protection requires that at least
    one of the two models is correctly specified.

    Note: This is not the formal Augmented IPW (AIPW) estimator, which has
    a specific augmentation term that provides the doubly robust property
    in a more general sense. The covariate-adjusted IPTW approach used here
    achieves the same consistency property under linear outcome models.
    ...
    """
```

Update ALL internal references to this method. Search for `fit_doubly_robust_model` throughout the file and replace with `fit_iptw_outcome_model`. There is one call site in `analyze_treatment_effect`:

```python
gee_res = self.fit_iptw_outcome_model(
    df,
    outcome_var,
    treatment_var,
    weight_col="iptw",
    cluster_var=cluster_var,
    covariates=covariates,
    family=auto_family
)
```

**Step 2: In the notebook**

In Cell 21 markdown, replace every occurrence of "doubly robust" with the softer formulation. Specifically:

Find: `"Covariate adjustment in the outcome model for **doubly robust estimation**"`
Replace with: `"Covariate adjustment in the outcome model for **additional protection against model misspecification**"`

Find: `"## Doubly Robust Interpretation"` and its content block.
Replace the heading and first paragraph with:

```markdown
## Protection from Model Misspecification

Including covariates in both the propensity score model (via weights) and the outcome model (via regression adjustment) provides additional protection: the treatment effect estimate is consistent if **at least one** of the following holds:

- The propensity score model $P(Treatment_i = 1 \mid X_i)$ is correctly specified, **or**
- The outcome model (including $X_i$) is correctly specified

This property, sometimes called double robustness under linearity, gives you two chances instead of one. However, both models here use the same covariates with main-effects-only functional forms — so misspecification of the functional form (e.g., missing interaction terms) could affect both simultaneously. This is a safeguard, not a guarantee.
```

Also find and replace in the design choices summary table:
Find: `**Doubly Robust Estimation**`
Replace with: `**Covariate Adjustment in Both Stages**`

Find the description: `"Includes covariates in both the propensity score model (via weights) and the outcome model (via regression adjustment). ATE estimate is consistent if **either** model is correct — you get two chances instead of one."`
Replace with: `"Includes covariates in both the propensity score model (via weights) and the outcome model (via regression adjustment). Under linearity, the ATE estimate is consistent if **either** model is correctly specified. Both models use the same covariates with main-effects-only forms, so misspecification of functional form could affect both."`

Apply the same softening in Cell 26 markdown (ATT section) and Cell 31 markdown (survival section) — search for "doubly robust" and replace with the equivalent softer language.

---

## CHANGE 3: Add GEE justification

### Problem
The notebook never explains why GEE is used instead of the simpler OLS/WLS with cluster-robust SEs.

### What to do

Insert a new markdown cell immediately BEFORE Cell 22 (the ATE survey analysis code). Content:

```markdown
### Why GEE Instead of OLS with Cluster-Robust Standard Errors?

For our continuous survey outcomes with one observation per manager and team-level clustering, **GEE with an exchangeable working correlation and sandwich standard errors produces results that are very similar to weighted least squares (WLS) with cluster-robust covariance.** Either approach would be defensible here. We use GEE for three reasons:

**1. It generalizes naturally to non-continuous outcomes.** GEE handles Gaussian, Binomial, and other families through the same interface. If your outcomes include binary flags alongside continuous scales — common in applied program evaluation — GEE lets you use a single modeling framework rather than switching between WLS and logistic regression.

**2. It models within-cluster correlation explicitly.** GEE specifies a working correlation structure (here, exchangeable — assuming constant pairwise correlation among teammates). When this structure is approximately correct, GEE can be more efficient than OLS, producing tighter confidence intervals. The efficiency gain is typically small for cross-sectional data with moderate cluster sizes, but it becomes substantial in longitudinal settings with repeated measures on the same individual — a common extension of this type of analysis.

**3. It is the conventional choice in the causal inference literature.** The Robins-Hernán tradition of marginal structural models pairs IPTW with GEE as the standard estimation approach. Using GEE aligns this workshop with the broader methodological literature, making it easier for participants to connect what they learn here to published research and textbooks.

**Important caveat:** With sandwich (robust) standard errors, GEE coefficient estimates are consistent regardless of whether the working correlation is correctly specified. The working correlation affects efficiency but not validity. This means GEE's advantage over WLS with cluster-robust SEs is an efficiency argument, not a consistency argument — and the efficiency gain is modest when you have a single cross-sectional observation per person (as we do here) rather than repeated measures.

> **If you are adapting this analysis for your own work** and your outcomes are all continuous with simple team-level clustering, `statsmodels.formula.api.wls(...).fit(cov_type="cluster", cov_kwds={"groups": cluster_col})` is a simpler alternative that produces nearly identical results.
```

---

## CHANGE 4: Name unmeasured confounders in the DAG discussion

### Problem
The DAG section asks "Could there be unobserved confounders?" but never names specific candidates.

### What to do

In Cell 5 markdown, find the line:
```
- Could there be unobserved confounders?
```

Replace with:
```markdown
- **Unmeasured confounders are likely present.** Three plausible candidates:
  - **Manager motivation/ambition:** Managers who proactively seek development are both more likely to enroll and more likely to have better outcomes regardless of training.
  - **Relationship with their own leader:** Managers with supportive bosses may be encouraged to participate and may also receive other forms of support that improve outcomes.
  - **Pre-existing career trajectory:** Managers already on an upward path may self-select into training and also show improved efficacy and lower turnover independent of the program.
  
  We will quantify how strong such confounders would need to be to explain our findings using E-value sensitivity analysis.
```

---

## CHANGE 5: Add self-report bias acknowledgment after survey results

### Problem
All three survey outcomes are self-report from unblinded participants. The notebook never discusses this.

### What to do

In Cell 25 markdown (ATE Technical Summary: Survey Outcomes), after the results table and before the "Post-Weighting Balance Verification" subsection, insert:

```markdown
#### Self-Report Bias Caveat

These are self-reported outcomes from managers who know they were trained. Demand characteristics (responding in the way they believe the organization expects), cognitive dissonance reduction (justifying time invested in the program), and social desirability bias could all inflate these numbers. A Cohen's d of 0.47 for self-reported efficacy after a voluntary program is consistent with a real effect, but also consistent with a meaningful placebo component. The retention analysis, which uses an objective behavioral outcome, provides our stronger evidence.
```

---

## CHANGE 6: Condense the ATT analysis

### Problem
The full ATT analysis (Cells 26–30) takes 10+ minutes to run identical code with one parameter changed. In a 60-minute session this is not viable.

### What to do

**Step 1:** Convert the ATT code cells (Cells 27, 28) to markdown cells that display pre-computed results, or comment them out with a header note:

```python
# =============================================================================
# ATT ANALYSIS — OPTIONAL (uncomment to run)
# =============================================================================
# The ATT analysis uses the same pipeline as ATE above with one change:
# estimand='ATT' in the analyze_treatment_effect() call.
# This changes only the weight formula — everything else is identical.
# Pre-computed results are shown in the comparison table below.
#
# To run this yourself, uncomment the code below:
# ...
```

**Step 2:** Replace the Cell 26 (ATT methodology explanation) and Cell 29–30 (ATT results and comparison) with a single condensed markdown cell:

```markdown
---

## ATE vs. ATT: What Changes?

The ATT (Average Treatment Effect on the Treated) answers a different question: *"Did the program work for those who actually participated?"* — rather than *"What would happen if we scaled this to everyone?"*

**Mechanically, only the weight formula changes:**

| Group | ATE Weights | ATT Weights |
|-------|-------------|-------------|
| **Treated** | $1 / \hat{e}_i$ | $1$ (unweighted) |
| **Control** | $1 / (1 - \hat{e}_i)$ | $\hat{e}_i / (1 - \hat{e}_i)$ |

ATE reweights both groups to represent the full population. ATT leaves treated individuals at their natural weight and reweights only controls to resemble the treated group. Everything else — propensity score model, GEE specification, covariates, clustering — stays identical.

### Pre-Computed Comparison

| Outcome | ATE | ATT | Difference |
|---------|-----|-----|------------|
| Manager Efficacy Index | +0.40 (d = 0.47) | +0.41 (d = 0.50) | ATT slightly larger |
| Workload Index | -0.04 (ns) | -0.05 (ns) | Both null |
| Turnover Intention Index | +0.23 (d = 0.27) | +0.26 (d = 0.30) | ATT slightly larger |

ATT effects are consistently slightly larger, consistent with **positive selection**: managers who chose to participate benefit somewhat more than the average manager would. Both estimands tell the same qualitative story.

**Which to use?** For scaling decisions (should we roll this out?), **lead with ATE** — it generalizes to the full (restricted) population. ATT is useful for understanding whether the program worked for participants, but it doesn't answer the policy question.

> **Note:** Both estimates come from the same propensity score model. If that model is misspecified, both are biased in correlated ways. Agreement between ATE and ATT indicates the two estimands are close — it does **not** validate that either is correct. The DML analysis below provides a more independent robustness check.
```

**Step 3:** Remove Cell 24 (the `ate_survey_results` preservation cell). It's no longer needed since we're not running ATT code. If any downstream cells reference `ate_survey_results`, update them to reference `survey_results` directly.

---

## CHANGE 7: Add HR survivor selection caveat to survival section

### Problem
The notebook presents the rising HR pattern (0.587 → 0.948) purely as treatment effect decay, without noting that built-in survivor selection also pushes later HRs toward the null.

### What to do

In Cell 31 markdown, after the "Hazard Ratio Interpretation" subsection (after the line about HR > 1), insert a new subsection:

```markdown
### Causal Interpretation Caveat: Built-In Survivor Selection

The hazard ratio is a **conditional-on-survival** quantity: at each time point, it compares departure rates among individuals who have not yet departed. When treatment affects survival (which is the effect we are estimating), the composition of the surviving risk set differs between treated and control groups at later time points.

Concretely: if training prevents early departures, the control group at month 6 has already lost its most departure-prone members, while the treated group retained them. The HR at month 6 compares a **selected** control group against a less-selected treated group — biasing later HRs toward the null even if the true treatment effect is constant (Hernán, 2010).

**Implication:** The rising HR pattern (0.587 → 0.948) may reflect genuine effect decay, survivor selection, or both. We cannot distinguish these from the data.

**What we do about it:** We report two additional estimands that avoid this problem:

1. **Survival probability differences** at fixed timepoints from IPTW-weighted Kaplan-Meier curves — these have a clean causal interpretation under IPTW assumptions.
2. **RMST difference** (Restricted Mean Survival Time) — the average additional days of retention attributable to treatment. This integrates over the full survival curve, avoiding instantaneous-rate conditioning.

We use HRs to characterize *when* the effect is strongest, and survival differences + RMST for the primary causal claims and stakeholder communication.
```

Also add one row to the "Summary: Model Design Choices" table in Cell 31:

```markdown
| **RMST Difference** | HRs are conditional-on-survival and subject to built-in selection bias at later time points. Stakeholders need an intuitive, unconditional metric. | Computes the area between IPTW-weighted KM curves up to the time horizon. Reports the average additional days of retention attributable to treatment. Business-friendly: "training retains managers an average of X additional days." |
```

---

## CHANGE 8: Add RMST computation

### Problem
`compute_rmst_difference` is implemented in `causal_inference_modelling.py` but never called in the notebook.

### What to do

Insert a new code cell between the current Cell 36 (survival summary table) and Cell 37 (balance verification):

```python
# =============================================================================
# RMST DIFFERENCE — CAUSAL ESTIMAND WITHOUT SURVIVOR SELECTION BIAS
# =============================================================================
# RMST answers: "How many additional days of retention does training buy
# within the 12-month study window?" It avoids the built-in survivor
# selection problem of hazard ratios.

rmst_results = {}
rmst_results[retention_outcome_name] = CausalInferenceModel.compute_rmst_difference(
    survival_result=survival_results[retention_outcome_name],
    time_horizon=365,
    alpha=0.05,
    n_bootstrap=500,
    random_state=SEED,
)
```

---

## CHANGE 9: Restructure retention results to lead with survival differences and RMST

### Problem
Cell 40 (ATE Technical Summary — Retention) leads with hazard ratios. The survival probability differences and RMST are either absent or labeled "descriptive only."

### What to do

Replace Cell 40 markdown with:

```markdown
### ATE Technical Summary — Retention (Survival)
Note: Results do not generalize to poor performers.

> **Method**: IPTW-weighted survival analysis combining three complementary estimands: (1) survival probability differences from IPTW-weighted KM curves (primary causal estimand), (2) RMST difference (business-friendly metric), and (3) Cox PH with categorical time interaction (characterizes time-varying pattern). Survival differences and RMST are preferred for causal claims; period-specific HRs characterize *when* the effect occurs but are subject to built-in survivor selection at later time points.

##### Primary Estimand: Survival Probability Differences (IPTW-Weighted KM)

| Timepoint | Trained | Untrained | Difference | Note |
|-----------|---------|-----------|------------|------|
| 3 months | 93.5% | 89.4% | **+4.1pp** | Strongest and most reliable effect |
| 6 months | 90.0% | 85.5% | +4.5pp | Early gains maintained |
| 9 months | 87.2% | 82.5% | +4.7pp | Plateau — no new gains accumulating |
| 12 months | 84.4% | 79.7% | +4.7pp | Reflects month 0–3 effect persisting |

These differences have a direct causal interpretation under IPTW assumptions: they estimate the population-level difference in retention if everyone were trained vs. no one.

##### RMST Difference

[AGENT: run the RMST cell and insert the computed values below]

Training retains managers an average of approximately **X additional days** within the 12-month study window (95% CI: [X, X] days, bootstrap n=500). This translates to roughly X additional weeks per trained manager.

##### Supporting Evidence: Period-Specific Hazard Ratios

The Cox time interaction model reveals *when* the treatment effect is strongest. Later-period HRs are subject to built-in survivor selection (see caveat above) and should be interpreted as pattern characterization, not pure causal effects.

| Period | HR | 95% CI | p-value | Sig? |
|--------|-----|--------|---------|------|
| 0–3mo *(ref)* | **0.587** | [0.386, 0.892] | 0.013 | Yes * |
| 3–6mo | 0.846 | [0.510, 1.405] | 0.518 | No |
| 6–9mo | 0.896 | [0.509, 1.576] | 0.703 | No |
| 9–12mo | 0.948 | [0.540, 1.667] | 0.854 | No |

**Interpreting the rising HR pattern:** The HR increases from 0.587 to 0.948. This is consistent with treatment effect decay, but also with survivor selection — the control group's most departure-prone members have already left by later periods, mechanically pushing the HR toward 1. The survival probability differences (which do not suffer from this selection) show the gap stabilizing at ~4.7pp rather than growing, supporting an early effect that persists rather than ongoing active protection.

#### E-Value Sensitivity Analysis

| Outcome | E-Value Point | E-Value CI | Robustness |
|---------|---------------|------------|------------|
| Manager Retention | **2.80** | **1.49** | Moderate |

The E-value CI bound of 1.49 means a confounder like **pre-existing intent to stay** — which plausibly has an RR of 1.5+ with both treatment participation and departure risk — could explain the retention finding. This represents a genuine threat to the causal interpretation and should be acknowledged when presenting results.

#### Post-Weighting Balance Verification
- ✅ Balance verification passed (0 imbalanced covariates).
```

---

## CHANGE 10: Update the Global Technical Summary (Cell 41)

### What to do

In Cell 41, update the retention section to lead with RMST and survival differences:

```markdown
### Retention Outcomes

**Primary estimand — RMST difference:** Training retains managers an average of **X additional days** within the 12-month study window (95% CI: [X, X]).

**Survival probability difference at 12 months:** +4.7pp (84.4% vs 79.7%), driven primarily by a significant early effect in months 0–3.

**Time-varying pattern (Cox PH, for characterization — subject to survivor selection at later periods):**

| Time Period | HR | 95% CI | p-value | Interpretation |
|-------------|-----|--------|---------|----------------|
| 0–3 months | 0.587 | [0.386, 0.892] | 0.013 | Significant * (41% lower hazard) |
| 3–6 months | 0.846 | [0.491, 1.454] | 0.542 | Not significant |
| 6–9 months | 0.899 | [0.495, 1.632] | 0.725 | Not significant |
| 9–12 months | 0.944 | [0.523, 1.703] | 0.848 | Not significant |
```

---

## CHANGE 11: Update stakeholder section (Cell 42) with RMST-grounded claims

### What to do

In Cell 42, replace the final "Impact at scale" paragraph with:

```markdown
**Impact at scale:** Based on the RMST analysis, training 1,000 managers would yield approximately [AGENT: compute 1000 × RMST_diff and insert] additional person-days of retention within the first year (95% CI: [AGENT: insert scaled CI]). The 12-month survival gap of 4.7 percentage points — approximately 47 additional managers retained per 1,000 trained — reflects early gains (months 0–3) that persist through the year, not new gains accumulating in later quarters. The strongest and most statistically reliable effects are concentrated in the first quarter.
```

---

## CHANGE 12: Add E-value note for survival outcomes

### What to do

Insert a new markdown cell immediately after Cell 38 (E-value computation for retention):

```markdown
**Note on E-values for survival outcomes:** The E-value is computed on the reference-period hazard ratio (HR = 0.587). Because HRs at later time points are subject to built-in survivor selection, the E-value applies most directly to the early-period finding. A confounder like pre-existing intent to stay — which plausibly has an RR of 1.5+ with both treatment and departure — could explain the retention finding (E-value CI bound = 1.49). This should be transparently communicated to stakeholders.
```

---

## CHANGE 13: Switch ATE robustness check from `econml` to `doubleml` with native cluster support

### Problem
The current DML implementation uses `econml.dml.DML`, which accepts a `cluster_var` parameter but ignores it entirely. Standard errors are computed assuming independence, making the DML robustness check invalid as a comparison against the cluster-robust GEE estimates. Our previous approach was to manually bolt cluster-robust SEs onto `econml` with ~40 lines of hand-rolled sandwich estimation. A better solution exists.

### Solution
The `doubleml` package (`DoubleML`) provides **native cluster-robust inference** out of the box. When you pass `cluster_cols` to `DoubleMLData`, the package automatically handles:

1. **Cluster-aware cross-fitting** — all members of a cluster are assigned to the same fold during nuisance estimation (no information leakage).
2. **Cluster-robust score-based inference** — standard errors, CIs, and p-values account for within-cluster correlation using the clustered influence function.

No manual sandwich estimator, no custom `GroupKFold` wrapper, no extracting residuals. This is cleaner, better tested, and maintained by the package authors.

**Tradeoff:** The notebook will now use two DML packages: `doubleml` for the Linear DML ATE robustness check (with cluster support), and `econml` for Causal Forest CATE estimation (which `doubleml` doesn't offer). This is a minor pedagogical cost — acknowledge it with one sentence — but the benefit is correct, clean cluster-robust inference.

### What to do

**Step 1: Add `doubleml` to `requirements.txt`**

Add the following line:

```
doubleml
```

**Step 2: Add import in `causal_inference_modelling.py`**

At the top of the file, alongside the existing `econml` imports, add:

```python
from doubleml import DoubleMLData, DoubleMLPLR
```

**Step 3: Add a new method to `CausalInferenceModel`**

Add the following method to the class. Place it after the existing `dml_estimate_treatment_effects` method:

```python
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

    This method is intended as a **robustness check** against the primary
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
        Dictionary with keys compatible with build_summary_table():
        - effect: ATE point estimate
        - se: standard error (cluster-robust if cluster_var provided)
        - ci_lower, ci_upper: confidence interval bounds
        - p_value: p-value for H0: ATE = 0
        - significant: bool
        - alpha: significance level
        - n_clusters: number of clusters (if clustered)
        - n_obs: number of observations
        - doubleml_model: fitted DoubleMLPLR object
        - summary: model summary DataFrame

    References
    ----------
    Bach P, Chernozhukov V, Kurz MS, Spindler M (2022). DoubleML —
    An Object-Oriented Implementation of Double Machine Learning in
    Python. Journal of Machine Learning Research, 23(53):1-6.
    """
    df = data.copy()

    # One-hot encode categoricals
    cat_vars = categorical_vars or []
    bin_vars = binary_vars or []
    cont_vars = continuous_vars or []

    if cat_vars:
        df = pd.get_dummies(df, columns=cat_vars, drop_first=True)

    # Build covariate list
    dummy_cols = [c for c in df.columns
                  if any(c.startswith(v + "_") for v in cat_vars)]
    x_cols = dummy_cols + bin_vars + cont_vars

    # Clean column names for formula compatibility
    rename_map = {c: self._clean_column_name(c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)
    outcome_col = self._clean_column_name(outcome_col)
    treatment_col = self._clean_column_name(treatment_col)
    x_cols = [self._clean_column_name(c) for c in x_cols]
    if cluster_var:
        cluster_var = self._clean_column_name(cluster_var)

    # Drop NAs
    all_cols = list(set([outcome_col, treatment_col] + x_cols +
                        ([cluster_var] if cluster_var else [])))
    df = df[all_cols].dropna().copy()

    if len(df) < 20:
        raise ValueError(
            f"Insufficient data after removing missing values: {len(df)} rows"
        )

    # Build DoubleMLData object
    dml_data = DoubleMLData(
        df,
        y_col=outcome_col,
        d_cols=treatment_col,
        x_cols=x_cols,
        cluster_cols=cluster_var if cluster_var else None,
    )

    # Nuisance models
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

    # Fit PLR model
    dml_plr = DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=cv,
    )
    dml_plr.fit()

    # Extract results
    ate = float(dml_plr.coef[0])
    se = float(dml_plr.se[0])
    ci = dml_plr.confint(level=1 - alpha)
    ci_lower = float(ci.iloc[0, 0])
    ci_upper = float(ci.iloc[0, 1])
    p_value = float(dml_plr.pval[0])
    significant = p_value < alpha

    n_clusters = df[cluster_var].nunique() if cluster_var else None

    # Print summary
    cluster_note = f", {n_clusters} clusters" if n_clusters else ""
    print(f"\n  DoubleML PLR — Cluster-Robust ATE Estimation")
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
        "cohens_d": None,  # Not computed here — use IPTW-weighted d from primary analysis
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
```

**Step 4: Update the notebook Cell 43 markdown**

In the `Linear DML vs. IPTW + GEE` comparison table, update the Clustering row:

From: `| **Clustering** | ✅ Cluster-robust sandwich SEs | ❌ Assumes independence |`
To: `| **Clustering** | ✅ Cluster-robust sandwich SEs | ✅ Cluster-aware cross-fitting + cluster-robust SEs (via DoubleML) |`

Add a note after the table explaining the two-package choice:

```markdown
> **Note on DML packages:** We use two complementary DML implementations. For the **ATE robustness check**, we use the `doubleml` package (Bach et al., 2022), which provides native cluster-robust inference — cluster-aware cross-fitting and cluster-robust standard errors are handled automatically when `cluster_cols` is specified. For **heterogeneous treatment effects (CATE)**, we use the `econml` package, which provides Causal Forest DML with feature importance and tree-based subgroup discovery — functionality not available in `doubleml`. This ensures the ATE comparison with IPTW+GEE is on equal footing (both cluster-robust), while still enabling CATE exploration.
```

Also remove or update any text that says DML doesn't account for clustering. Search Cell 43 for phrases like "assumes independence" or "does not natively handle clustering" and remove/update them.

**Step 5: Update the notebook Cell 44 code**

Replace the current Linear DML section of Cell 44 with a call to the new method. The cell currently runs `dml_estimate_treatment_effects` with `estimate="both"`, which fits both Linear DML (ATE) and Causal Forest (CATE). Split this into two separate calls:

```python
# ====================================================================================
#  PART 1: ATE Robustness Check via DoubleML (cluster-robust)
# ====================================================================================
# Uses the doubleml package for native cluster-robust inference.
# This ensures the DML vs IPTW+GEE comparison is apples-to-apples:
# both methods now account for team-level clustering.

sig_survey_outcomes = ['manager_efficacy_index']

dml_ate_results = {}

for o in sig_survey_outcomes:
    print("\n" + "=" * 60)
    print(f"CLUSTER-ROBUST DML — ATE ESTIMATION: {o}")
    print("=" * 60)

    baseline = baseline_vars.get(o)
    dml_continuous = list(continuous_vars)
    if baseline is not None and baseline not in dml_continuous:
        dml_continuous.append(baseline)

    dml_ate_results[o] = causal_model.dml_cluster_robust_ate(
        data=data,
        outcome_col=o,
        treatment_col=treatment,
        categorical_vars=categorical_vars,
        binary_vars=binary_vars,
        continuous_vars=dml_continuous,
        cluster_var='team_id',
        random_state=42,
        alpha=0.05,
    )

# Compare with IPTW+GEE ATE
for o in sig_survey_outcomes:
    iptw_ate = survey_results[o]['effect']
    iptw_ci = (survey_results[o]['ci_lower'], survey_results[o]['ci_upper'])
    dml_ate = dml_ate_results[o]['effect']
    dml_ci = (dml_ate_results[o]['ci_lower'], dml_ate_results[o]['ci_upper'])

    print(f"\n  [{o}] Method Comparison:")
    print(f"    IPTW+GEE (cluster-robust):  ATE = {iptw_ate:.4f}  CI: [{iptw_ci[0]:.4f}, {iptw_ci[1]:.4f}]")
    print(f"    DoubleML (cluster-robust):   ATE = {dml_ate:.4f}  CI: [{dml_ci[0]:.4f}, {dml_ci[1]:.4f}]")
    print(f"    Difference in point estimates: {abs(iptw_ate - dml_ate):.4f}")
```

Then keep the existing Causal Forest CATE section (using `econml`) as a separate code block:

```python
# ====================================================================================
#  PART 2: Heterogeneous Treatment Effects via Causal Forest (econml)
# ====================================================================================
# Uses econml's CausalForestDML for individualized CATE estimation.
# Note: econml does not natively support cluster-robust inference.
# The CATE analysis is exploratory — point estimates for subgroup
# discovery, not inferential claims about specific subgroup effects.

hte_results = {}

for o in sig_survey_outcomes:
    print("\n" + "=" * 60)
    print(f"HTE — CAUSAL FOREST CATE: {o}")
    print("=" * 60)

    baseline = baseline_vars.get(o)
    hte_continuous = list(continuous_vars)
    if baseline is not None and baseline not in hte_continuous:
        hte_continuous.append(baseline)

    hte_results[o] = causal_model.dml_estimate_treatment_effects(
        data=data,
        outcome_col=o,
        treatment_col=treatment,
        categorical_vars=categorical_vars,
        binary_vars=binary_vars,
        continuous_vars=hte_continuous,
        estimand="ATE",
        estimate="CATE",       # Only Causal Forest, skip Linear DML
        cluster_var='team_id',
        project_path=str(base_dir / "results"),
        analysis_name=f"HTE_{o}",
        random_state=42,
        alpha=0.05,
    )

    res = hte_results[o]
    if res.get("cate_plot") is not None:
        display(res["cate_plot"])
    if res.get("importance_plot") is not None:
        display(res["importance_plot"])
    if res.get("tree_plot") is not None:
        display(res["tree_plot"])
```

**Step 6: Update the existing `dml_estimate_treatment_effects` docstring**

Add a note at the top of the docstring recommending the new method for cluster-robust ATE:

```python
"""
...
.. note::
    For cluster-robust ATE estimation, prefer ``dml_cluster_robust_ate()``,
    which uses the ``doubleml`` package with native cluster support. This
    method (``dml_estimate_treatment_effects``) remains available for CATE
    estimation via Causal Forest, which ``doubleml`` does not offer.
...
"""
```

**Step 7: Add a markdown comparison cell after the DML code**

Insert a new markdown cell after the DML code cells with a three-way comparison:

```markdown
### Method Comparison: ATE Estimates Across Three Approaches

| Method | ATE | SE | 95% CI | Clustering? |
|--------|-----|-----|--------|-------------|
| IPTW + GEE | [AGENT: insert] | [AGENT: insert] | [AGENT: insert] | ✅ Cluster-robust sandwich SEs |
| DoubleML PLR | [AGENT: insert] | [AGENT: insert] | [AGENT: insert] | ✅ Cluster-aware CV + cluster-robust SEs |

**Interpretation:** Point estimates converge across two fundamentally different estimation strategies — propensity-score weighting (IPTW+GEE) and residualization-based machine learning (DoubleML). This agreement increases confidence that the observed effect reflects a genuine treatment impact rather than an artifact of a single modeling approach. Both methods account for team-level clustering, making the comparison like-for-like.

Note: The DoubleML CI may be slightly different in width from the GEE CI. This reflects different variance estimation approaches (score-based vs. sandwich), not a meaningful methodological disagreement.
```

---

## CHANGE 14: Add "Threats to Validity" cell

### Problem
The notebook never consolidates the limitations in one place.

### What to do

Insert a new markdown cell at the end of Cell 41 (Global Technical Summary), before Checkpoint 4:

```markdown
### Threats to Validity

| Threat | Impact | Mitigation | Residual Risk |
|--------|--------|------------|---------------|
| **Unmeasured confounding** (motivation, leader relationship, career trajectory) | Biases ATE away from true effect, most likely upward | IPTW balances observed confounders; E-values quantify required confounder strength | E-value CI bound of 1.49 for retention — a plausible confounder could explain the finding |
| **Self-report bias** on survey outcomes | Inflates efficacy and turnover intention effects (demand characteristics, cognitive dissonance) | Retention analysis uses objective behavioral outcome | Survey effect sizes should be interpreted as upper bounds |
| **Restricted population** (no low performers) | ATE applies only to average-and-above performers | Positivity violation makes inclusion impossible; exclusion is transparent | Cannot extrapolate to low performers if program is scaled to them |
| **No pre-training outcome measurement** | Cannot do difference-in-differences; baseline variables may not fully capture pre-treatment state | Baseline covariates included as confounders in both PS and outcome models | Residual bias possible if baselines are poor proxies for pre-treatment trajectory |
| **Survivor selection in hazard ratios** | Later-period HRs biased toward null | Report survival probability differences and RMST as primary causal estimands | HR time-trend pattern is descriptive, not purely causal |
```

---

## Summary of all changes

| # | File(s) | Type | Description |
|---|---------|------|-------------|
| 1 | Notebook | Reframe | Low-performer exclusion: visible positivity violation → explicit estimand scoping |
| 2 | Notebook + .py | Rename + soften | `fit_doubly_robust_model` → `fit_iptw_outcome_model`; soften all DR claims |
| 3 | Notebook | New markdown | GEE vs OLS justification |
| 4 | Notebook | Edit markdown | Name specific unmeasured confounders in DAG section |
| 5 | Notebook | New markdown | Self-report bias caveat after survey results |
| 6 | Notebook | Condense | ATT analysis: full code → comparison table with pre-computed results |
| 7 | Notebook | New markdown | HR survivor selection caveat in survival section |
| 8 | Notebook | New code cell | RMST computation (calls existing function) |
| 9 | Notebook | Rewrite markdown | Retention results lead with survival diffs + RMST, HRs become supporting |
| 10 | Notebook | Edit markdown | Global summary restructured for retention |
| 11 | Notebook | Edit markdown | Stakeholder "impact at scale" grounded in RMST |
| 12 | Notebook | New markdown | E-value scope note for survival |
| 13 | Notebook + .py | Code change | DML ATE: switch to `doubleml` with native cluster-robust inference; keep `econml` for CATE only |
| 14 | Notebook | New markdown | Consolidated threats-to-validity table |
