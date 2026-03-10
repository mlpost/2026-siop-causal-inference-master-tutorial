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

- Familiarity with regression concepts (OLS, logistic regression)
- Basic understanding of causal inference concepts (confounding, selection bias) is helpful but not required — the course materials cover these
- Working knowledge of Python (pandas, numpy) or similar language (R) also helpful but not required

---

## Getting Started with Google Colab

No local installation required. Everything runs in your browser.

### Step 1: Open the Main Notebook in Colab

<a href="https://colab.research.google.com/github/mlpost/2026-siop-causal-inference-master-tutorial/blob/main/scenario2_workshop.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Step 2: Clone the Repo

Uncomment and run the very first cell in the notebook. This is crucial to: 
- ensure you get access to the data and supplementary py files
- install required libraries

```python
# Clone the repo
import os
if not os.path.exists('2026-siop-causal-inference-master-tutorial'):
  !git clone https://github.com/mlpost/2026-siop-causal-inference-master-tutorial.git

!pip install -q -r 2026-siop-causal-inference-master-tutorial/requirements.txt

# Add source code to path
import sys
sys.path.insert(0, '2026-siop-causal-inference-master-tutorial')

# If needed: delete the repo
# !rm -rf 2026-siop-causal-inference-master-tutorial
```
