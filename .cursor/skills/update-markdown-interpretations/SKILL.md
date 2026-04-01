---
name: update-markdown-interpretations
description: >-
  Resync manually written interpretation markdown in causal_inference_workshop.ipynb
  with fresh analysis outputs; skip HTML slideshow code cells; apply Checkpoint 5
  stakeholder rules including a bold red human-review notice.
---

# Update Markdown Interpretations (Workshop Notebook)

This skill tells you how to **scan analysis outputs** in [`causal_inference_workshop.ipynb`](../../../causal_inference_workshop.ipynb) and **update markdown cells that interpret those analyses**—without changing overall structure, flow, or HTML slideshow cells.

## When to Use

- After **re-running** the notebook or analysis sections
- After **regenerating [`data/generate_data.py`](../../../data/generate_data.py)** output or changing [`supp_functions/causal_inference_modelling.py`](../../../supp_functions/causal_inference_modelling.py) in ways that shift printed estimates
- When the user asks to **align stakeholder or technical text** with new numbers (ATEs, CIs, HRs, E-values, RMST, survival percentages, counts, etc.)

## Hard Exclusions (do not edit)

1. **Code cells that render HTML slideshows**  
   - Any **code cell** whose source contains `HTML(` (typically `from IPython.display import HTML` and `HTML(r"""...""")`).  
   - Do **not** change these cells—even if they duplicate numbers or narrative. There are many such cells in the workshop notebook.

2. **Notebook output blobs**  
   - Do **not** edit stored `outputs` on code cells to “fix” interpretation. This skill updates **markdown interpretation** to match **outputs** (or exported results), not the reverse.

## Source of Truth (order of preference)

1. **Upstream code cells’ saved outputs** in the notebook: `stdout`, `display(...)` / printed DataFrames, `text/plain` / `text/html` where estimates appear.
2. If outputs are **missing or stale** relative to the code: rerun **Run All** or the relevant section **before** editing markdown.
3. **Optional cross-check**: [`results/`](../../../results/) `*.xlsx` files (e.g. `ate_summary_survey.xlsx`, `ate_iptw_*`, DML exports) when they mirror the same tables as the notebook.

## In-Scope Markdown (anchors)

Treat these as the primary **interpretation** cells (by heading); also update any other markdown in the same notebook that **restates numeric results** from analyses.

| Section | Typical heading / role |
| --- | --- |
| Survey | `#### ATE Technical Summary: Survey Outcomes` — tables, per-outcome bullets, E-value table, relative % lines |
| Survival | Short markdown **Note on E-values for survival outcomes** (before retention summary) |
| Retention | `#### ATE Technical Summary — Retention (Survival)` — KM % table, RMST paragraph, HR-by-period table, E-value table, balance bullets |
| Stakeholders | `## Checkpoint 5: Key Takeaways for Stakeholders` — see **Checkpoint 5 rules** below |
| Diagnostics / scoping | Conceptual cells that still cite **numbers** (e.g. positivity / sample counts) — update counts so they match stdout |

Do **not** rewrite cells that are purely **teaching prose** with no statistics unless an upstream number change forces a correction.

## Editing Rules

- **Preserve structure**: Keep headings, horizontal rules (`---`), **table column headers and row order**, list nesting, and section order.  
- **Tables**: Change only **cell values** (numbers, significance flags, stars, “Yes/No” where applicable). Rename columns or merge/split rows **only** if the upstream output format actually changed—then make the **minimal** markdown change to match.  
- **Wording**: Prefer **minimal** edits—only **stat-bearing** phrases and clauses. Do not rewrite for style or “clarity” unless the user asks.  
- **Consistency**: Percentages, HRs, CIs, p-values, E-values, RMST days, bootstrap `n`, and person-day scaling must match the **same** source outputs you used for that section.

## Checkpoint 5: Key Takeaways for Stakeholders — Special Rules

Applies to the markdown cell under `## Checkpoint 5: Key Takeaways for Stakeholders` (and its subsections: **What We Found**, **Robustness Check**, **What This Means**, **Recommendations**, **Bottom Line**, etc.).

1. **Update all statistics** everywhere they appear (including scaled “per 1,000 managers” / person-days / RMST-style claims if present).  
2. **Do not** change the **overall narrative** or **action recommendations** (what to do, ordering of ideas, tone)—**except** where a number must be corrected (e.g. a “41% less likely” line must match the reported HR).  
3. **Human-review notice (bold red)**  
   - Add **one** short notice in **bold red** using HTML (Jupyter renders this in markdown cells).  
   - Place it **after** the Checkpoint 5 title line (`## Checkpoint 5: …`) or **immediately under** `### What We Found`—pick one placement and keep it consistent.  
   - Example patterns (adapt text; keep it concise):

```html
<p style="color:#c00;font-weight:bold;">Human review required: Statistical figures and automated wording below must be verified by a subject-matter expert before stakeholder distribution.</p>
```

   - **Idempotency**: If a paragraph with this intent (red, bold, human review) **already exists**, **do not** add a second copy.

## Validation Checklist

Before finishing:

- [ ] **No** edits were made to any code cell containing `HTML(`.
- [ ] Interpretation markdown numbers match the **notebook outputs** (or agreed `results/` exports) used for that run.
- [ ] **Tables** still have the same columns and row order; only values changed.
- [ ] **Checkpoint 5**: stats updated; narrative/recommendations unchanged except for necessary numeric corrections; **single** red bold human-review notice present (or already present, not duplicated).
- [ ] Spot-check at least one **survey** line, one **retention** line (HR or survival %), and one **Checkpoint 5** statistic against the source output.

## Optional

- A long mapping of “which markdown cell follows which code cell” can live in `reference.md` in this folder if the workflow grows; keep **`SKILL.md`** short and skimmable.

## Related

- For documentation updates about repo structure or dependencies, use the separate **`update-repo-onboarding`** skill—do not expand this skill into REPO_ONBOARDING edits unless the user explicitly asks.
