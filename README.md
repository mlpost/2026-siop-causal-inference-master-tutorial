# 2026-siop-causal-inference-master-tutorial

# No Experiment, No Problem? Causal Inference in Applied Quasi-Experimental Settings

A hands-on course on estimating causal treatment effects from observational HR data in quasi-experimental setting.

## What You'll Learn

This course walks through a real-world people analytics use case: **evaluating whether a manager training program causally improves leadership outcomes and team retention.** You'll learn to:

- **Frame the causal question** — choose the right estimand for the question (ATE vs. ATT) and understand when and why they diverge
- **Diagnose before modeling** — multicollinearity (VIF/GVIF), covariate overlap, positivity, and estimand feasibility
- **Estimate with IPTW** — build, stabilize, trim, and diagnose inverse probability of treatment weights, then verify covariate balance pre- vs. post-weighting
- **Model survey outcomes** — IPTW + covariate-adjusted GEE with cluster-robust standard errors
- **Model retention (time-to-event)** — IPTW-weighted Kaplan–Meier (descriptive), **RMST** (primary causal estimand), and **Cox PH with time interactions** (inferential, when proportional hazards is violated)
- **Check robustness** — cluster-robust **DoubleML (PLR)** as an alternative identification strategy, and **E-value sensitivity analysis** for unmeasured confounding
- **Explore heterogeneity** — use a **Causal Forest (DML)** to surface conditional average treatment effects (CATE) along actionable moderators — who benefits most
- **Communicate results** — translate technical findings into stakeholder-ready recommendations with transparent caveats
---

## Prerequisites

- Familiarity with regression concepts (OLS, logistic regression)
- Basic understanding of causal inference concepts (confounding, selection bias) is helpful but not required — the course materials cover these
- Working knowledge of Python (pandas, numpy) or similar language (R) also helpful but not required

---

## Get Started in Google Colab

No local installation required. Everything runs in your browser.

<a href="https://colab.research.google.com/github/mlpost/2026-siop-causal-inference-master-tutorial/blob/main/causal_inference_workshop.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
