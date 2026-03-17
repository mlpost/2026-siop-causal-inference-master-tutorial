---
name: refactor-code
description: Refactor methods, functions, or code chunks in the SIOP 2026 Master Tutorial causal inference codebase while preserving statistical correctness, method signatures, and the two-class architecture.
---

# Refactor Code

This skill safely refactors code in the causal inference codebase, improving readability, maintainability, and performance while preserving statistical correctness and backward compatibility.

## When to Use

- Use this skill when methods exceed 200 lines and need extraction into helper methods
- This skill is helpful for improving code readability without changing functionality
- Use when duplicate code patterns appear across methods
- This skill is helpful for optimizing performance bottlenecks in IPTW or GEE calculations
- Use when method complexity makes testing or debugging difficult
- This skill is helpful for consolidating similar logic across CausalDiagnostics and CausalInferenceModel
- Use when adding new features requires refactoring existing code first
- This skill is helpful for improving error handling patterns

## Instructions

### Step 1: Understand the Refactoring Context

Before making any changes:
- Identify the specific method, function, or code chunk to refactor
- Understand its role in the causal inference pipeline (diagnostics, IPTW, GEE, survival, DML)
- Review all callers of the method to understand usage patterns
- Check if the method is part of the public API (called from notebooks) or internal
- Use the ask questions tool if the refactoring scope or goals are unclear

### Step 2: Preserve Statistical Correctness

**CRITICAL: Do not break causal inference methodology**

- **IPTW weight calculations** must remain mathematically identical:
  - ATE formula: `1/PS` for treated, `1/(1-PS)` for control
  - ATT formula: `1` for treated, `PS/(1-PS)` for control
  - Stabilization and trimming logic must be preserved exactly
  
- **Propensity score estimation** must maintain:
  - Model family (GEE for clustered data, GLM otherwise)
  - Covariate specifications (one-hot encoding, column sanitization)
  - Convergence criteria and warnings
  
- **GEE outcome models** must preserve:
  - Family auto-detection (Gaussian vs Binomial)
  - Doubly robust specification (weights + covariates)
  - Cluster-robust standard errors
  - Baseline variable handling (included in outcome model, excluded from PS)
  
- **Cox PH survival models** must maintain:
  - Time interaction specifications (categorical vs continuous)
  - Person-period data expansion for categorical interactions
  - Hazard ratio calculations and interpretations
  
- **Balance diagnostics** must preserve:
  - SMD calculation formulas (weighted vs unweighted)
  - Variance estimation for continuous and binary variables
  - Threshold interpretations (|SMD| < 0.1)

### Step 3: Maintain Method Signatures and Return Formats

**Preserve backward compatibility:**

- **Do not change** public method signatures without explicit approval
- **Do not change** return value structures (dictionaries, DataFrames, tuples)
- **Do not change** parameter names or default values
- If signature changes are necessary:
  - Add new parameters with defaults (don't remove old ones)
  - Maintain backward-compatible aliases
  - Update all callers in the notebook
  - Document the change in TEAM_README.md Section 7 (Gotchas)

**Key methods to be especially careful with:**
- `CausalDiagnostics.run_overlap_diagnostics()`
- `CausalDiagnostics.compute_balance_df()`
- `CausalInferenceModel.analyze_treatment_effect()`
- `CausalInferenceModel.analyze_survival_effect()`
- `CausalInferenceModel.dml_estimate_treatment_effects()`
- `CausalInferenceModel.build_summary_table()`
- `CausalInferenceModel.build_survival_summary_table()`

### Step 4: Follow Docstring Conventions

**Maintain existing documentation patterns:**

```python
def method_name(self, param1, param2, param3=None):
    """
    Brief one-line description.
    
    Longer description explaining purpose, methodology, and key decisions.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    param3 : type, optional
        Description of param3 (default: None)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'key1': description
        - 'key2': description
    
    Notes
    -----
    - Important implementation detail 1
    - Important implementation detail 2
    
    Examples
    --------
    >>> result = obj.method_name(data, 'treatment')
    >>> print(result['key1'])
    """
```

**Update docstrings when refactoring:**
- If extracting helper methods, add docstrings to new methods
- If changing internal logic, update the "Notes" section
- If adding parameters, document them fully
- Keep examples current and working

### Step 5: Preserve Error Handling Patterns

**Maintain existing error handling:**

- **Input validation** at method entry:
  ```python
  if outcome_var not in data.columns:
      raise ValueError(f"Outcome variable '{outcome_var}' not found in data")
  ```

- **Statistical warnings** for diagnostics:
  ```python
  if max_vif > 10:
      print(f"⚠️ WARNING: High VIF detected (max={max_vif:.2f})")
  ```

- **Graceful degradation** for optional features:
  ```python
  try:
      # Attempt advanced feature
  except Exception as e:
      print(f"⚠️ Optional feature failed: {e}")
      # Continue with basic functionality
  ```

- **Informative error messages** that guide users:
  ```python
  raise ValueError(
      f"Insufficient overlap for {estimand}. "
      f"Consider ATT or trimming. See overlap diagnostics."
  )
  ```

### Step 6: Respect the Two-Class Architecture

**CausalDiagnostics vs CausalInferenceModel separation:**

- **CausalDiagnostics** handles:
  - Pre-modeling checks (VIF, intercorrelations, sparse cells)
  - Overlap assessment (univariate SMDs, PS AUC, common support)
  - Balance checking (post-weighting SMDs)
  - Diagnostic visualizations
  
- **CausalInferenceModel** handles:
  - IPTW weight estimation and application
  - Outcome modeling (GEE, Cox PH, DML)
  - Effect estimation and inference
  - Summary tables and reports
  
**Do not mix responsibilities:**
- Don't add modeling logic to CausalDiagnostics
- Don't add diagnostic logic to CausalInferenceModel
- Shared utilities can go in private methods within each class
- If cross-class functionality is needed, use composition (one class calls the other)

### Step 7: Common Refactoring Patterns

**Extracting helper methods from large methods:**

```python
# Before: 300-line method with embedded logic
def analyze_treatment_effect(self, data, outcome_var, ...):
    # 50 lines of data prep
    # 100 lines of PS estimation
    # 50 lines of weight calculation
    # 100 lines of outcome modeling
    
# After: Main method delegates to helpers
def analyze_treatment_effect(self, data, outcome_var, ...):
    prepared_data = self._prepare_iptw_data(data, ...)
    outcome_results = self._fit_outcome_model(prepared_data, ...)
    return self._format_results(outcome_results, ...)

def _prepare_iptw_data(self, data, ...):
    # 150 lines of data prep + PS + weights
    
def _fit_outcome_model(self, prepared_data, ...):
    # 100 lines of GEE modeling
    
def _format_results(self, outcome_results, ...):
    # 50 lines of result formatting
```

**Consolidating duplicate code:**

```python
# Before: Similar logic in multiple methods
def analyze_treatment_effect(self, ...):
    # Column cleaning logic
    data.columns = [col.replace('&', 'and') for col in data.columns]
    
def analyze_survival_effect(self, ...):
    # Same column cleaning logic
    data.columns = [col.replace('&', 'and') for col in data.columns]

# After: Shared helper method
def _clean_column_names(self, data):
    """Sanitize column names for statsmodels compatibility."""
    data = data.copy()
    data.columns = [col.replace('&', 'and').replace(' ', '_') for col in data.columns]
    return data

def analyze_treatment_effect(self, ...):
    data = self._clean_column_names(data)
    
def analyze_survival_effect(self, ...):
    data = self._clean_column_names(data)
```

**Improving readability with intermediate variables:**

```python
# Before: Nested, hard-to-read
result = self._compute_effect(
    self._fit_model(
        self._prepare_data(data, vars), 
        outcome
    ), 
    estimand
)

# After: Clear intermediate steps
prepared_data = self._prepare_data(data, vars)
fitted_model = self._fit_model(prepared_data, outcome)
result = self._compute_effect(fitted_model, estimand)
```

### Step 8: Maintain Data Pipeline Integrity

**Critical data transformations to preserve:**

- **One-hot encoding** of categorical variables:
  - Must happen before PS estimation
  - Must use `drop_first=True` to avoid multicollinearity
  - Column names must be sanitized after encoding
  
- **Baseline variable routing**:
  - Baseline vars excluded from PS model
  - Baseline vars included in outcome model
  - Must maintain this separation in refactored code
  
- **Survival data preparation**:
  - `exit_date` → `days_observed` + `departed` conversion
  - Date format handling (`mixed` format for M/D/YYYY)
  - Censoring logic must be preserved
  
- **Weight calculations**:
  - Stabilization (numerator/denominator)
  - Trimming (percentile-based or absolute)
  - ESS computation for diagnostics

### Step 9: Update Related Documentation

After refactoring:

- **Update TEAM_README.md** using the `update-team-readme` skill:
  - Section 4: Update method tables and descriptions
  - Section 7: Add new gotchas if discovered
  - Update line counts in Section 2 if file sizes change significantly
  
- **Update inline comments** in the code:
  - Explain why refactoring was done
  - Document any subtle behavior changes
  - Add references to related methods
  
- **Update notebook** if public API changed:
  - Modify cell code to use new signatures
  - Update markdown explanations if workflow changed
  - Test all cells to ensure they still run

### Step 10: Test Refactored Code

**Validation checklist:**

- [ ] All notebook cells run without errors
- [ ] Results match pre-refactoring output (within numerical precision)
- [ ] Balance diagnostics show identical SMDs
- [ ] Effect estimates match (ATEs, HRs, p-values)
- [ ] Plots render correctly
- [ ] Excel exports work
- [ ] Error messages still trigger appropriately
- [ ] Edge cases still handled (e.g., new managers with baseline=0)
- [ ] Performance is maintained or improved

**Numerical precision checks:**

```python
# Compare pre- and post-refactoring results
import numpy as np

# ATEs should match within floating-point precision
assert np.allclose(old_ate, new_ate, rtol=1e-10)

# SMDs should match exactly
assert np.allclose(old_smds, new_smds, atol=1e-12)

# P-values should match within rounding
assert np.allclose(old_pvals, new_pvals, rtol=1e-6)
```

### Step 11: Performance Considerations

**When optimizing performance:**

- **Profile first** — don't optimize without measuring:
  ```python
  import time
  start = time.time()
  result = method()
  print(f"Execution time: {time.time() - start:.2f}s")
  ```

- **Common bottlenecks** in this codebase:
  - GEE model fitting (can't optimize much — statsmodels limitation)
  - Propensity score estimation (same)
  - Balance computation across many covariates (vectorize if possible)
  - Excel export with formatting (consider caching)
  
- **Safe optimizations**:
  - Vectorize pandas operations instead of loops
  - Cache expensive computations that don't change
  - Use `inplace=False` and explicit copies to avoid side effects
  - Avoid redundant data copies
  
- **Unsafe optimizations** (avoid):
  - Changing statistical formulas for speed
  - Reducing precision of numerical calculations
  - Skipping validation checks
  - Removing warnings

### Best Practices

- **Refactor incrementally** — one method at a time, test after each change
- **Preserve git history** — commit working code before refactoring
- **Use descriptive commit messages**: "Refactor: Extract _prepare_iptw_data() helper from analyze_treatment_effect()"
- **Keep refactoring separate from feature additions** — don't mix in one commit
- **Test with the actual workshop data** (`s2_manager_data.csv`) not toy examples
- **Run the full notebook** end-to-end after refactoring
- **Check for deprecation warnings** — statsmodels and scikit-learn evolve
- **Maintain the teaching focus** — code should be readable for workshop participants
- **Use the ask questions tool** if you're unsure whether a refactoring preserves correctness

### Domain-Specific Cautions

**Things that will break the causal inference methodology:**

- ❌ Changing IPTW weight formulas
- ❌ Including baseline variables in propensity score models
- ❌ Removing cluster-robust standard errors from GEE
- ❌ Changing SMD calculation formulas
- ❌ Modifying survival data preparation logic
- ❌ Altering FDR correction implementation
- ❌ Changing E-value calculation formulas
- ❌ Removing positivity/overlap checks

**Things that are safe to refactor:**

- ✅ Extracting helper methods for data prep
- ✅ Consolidating duplicate plotting code
- ✅ Improving variable naming for clarity
- ✅ Adding intermediate variables for readability
- ✅ Reorganizing method order within classes
- ✅ Improving error messages
- ✅ Adding type hints
- ✅ Optimizing pandas operations (if results identical)