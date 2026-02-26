# 2026-siop-causal-inference-master-tutorial

# No Experiment, No Problem? Causal Inference in Applied Quasi-Experimental Settings

A hands-on course on estimating causal treatment effects from observational HR data in quasi-experimental setting.

## What You'll Learn

This course walks through a real-world people analytics use case: **evaluating whether a manager training program causally improves leadership outcomes and team retention.** You'll learn to:

- Estimate **Average Treatment Effects (ATE)** and **Average Treatment Effects on the Treated (ATT)** from observational data
- Build and diagnose **inverse probability of treatment weights** (stabilized, trimmed)
- Fit **doubly robust outcome models** using weighted GEE with cluster-robust standard errors
- Assess covariate balance before and after weighting
- Conduct **E-value sensitivity analyses** to evaluate robustness to unmeasured confounding
- Interpret and communicate causal findings to technical and non-technical audiences
- Understand when and why ATE vs. ATT estimands differ
---

## Prerequisites

- Working knowledge of Python (pandas, numpy)
- Familiarity with regression concepts (OLS, logistic regression)
- Basic understanding of causal inference concepts (confounding, selection bias) is helpful but not required — the course materials cover these

---

## Getting Started with Google Colab

No local installation required. Everything runs in your browser.

### Step 1: Navigate to Google Colab

https://colab.research.google.com/

### Step 2: Clone this Repo

Open a notebook and run this cell: <br>
!! Be sure to insert your own username in the URL below !!

```python
import os

# Clone the repo
if not os.path.exists('causal-inference-people-analytics'):
    !git clone [https://github.com/YOUR_USERNAME/causal-inference-people-analytics.git](https://github.com/mlpost/2026-siop-causal-inference-master-tutorial.git)

# Install pinned dependencies
!pip install -q -r causal-inference-people-analytics/requirements.txt

# Add source code to path
import sys
sys.path.insert(0, 'causal-inference-people-analytics/src')
```
