---
agents-md-version: 2.0.0
applies-to: causal inference and impact evaluation work
---

# Agent workflow rules for causal inference projects

This repository is optimized for design-first impact evaluation. These rules are invariants that apply to every agent (Orchestrator, Planner, Implementer, Reviewer) across every session. They take precedence over agent-specific bodies when they conflict.

## Global rules

- Start with the causal question, estimand, and identification logic. Everything else follows from these three.
- Separate descriptive analysis from causal identification. Never let one masquerade as the other.
- Never present an estimate without stating the assumptions required for causal interpretation, and which of them are testable versus untestable.
- Treat robustness checks, falsification tests, diagnostics, and sensitivity analysis as first-class deliverables, not afterthoughts.
- Prefer minimal, auditable, reproducible code changes. Put data lineage and transformation logic into scripts or modules, not hidden notebook state. Raw data is immutable.
- For any task involving causal impact evaluation, identification strategy, treatment assignment, confounding, timing, assumptions, diagnostics, or robustness, the user-level `causal-inference` skill is the primary procedural standard — not optional background context. Cite the relevant skill section when reasoning from it.
- If the user requests a method that is a bad fit for the estimand or data structure, say so directly and propose better alternatives reasoned from the skill.

### The 10-step canonical workflow (from `causal-inference` skill)

All work in this repository follows these steps in order. Steps 1–5 are thinking and design; no code is written until Step 5 is complete.

1. **Formulate** — treatment, outcome, population, time zero, follow-up window, estimand (ATE/ATT/CATE/LATE), effect type, decision context.
2. **Design** — eligibility, treatment strategies, assignment timing, covariate measurement timing, censoring/attrition rules, SUTVA. For observational data: emulate a target trial. Load `references/study-design.md`.
3. **DAG** — explicit causal graph with confounders, mediators, colliders, and unobserved common causes. No estimation without an agreed DAG or DAG-equivalent argument. Load `references/dags-and-identification.md`.
4. **Identify** — choose identification route (backdoor / IV / DiD / RDD / ITS / synthetic control / g-methods). Output the explicit identification block:
   ```
   Identification strategy: <route>
   Required assumptions:
     1. ...
   Identification status: identified / partially identified / not identified
   ```
   Do not proceed if status is "not identified."
5. **Choose estimator** — match to the identified problem (not to fashion). Load `references/design-playbook.md`.
6. **Estimate** — write code only now. Check environment. Use only causally admissible covariates. Load `references/code-recipes.md`.
7. **Diagnose** — overlap, balance, functional form, design-specific checks. Load `references/diagnostics.md`.
8. **Refute** — at least one design-specific falsification or sensitivity check. Use negative-control outcomes or exposures when available. Load `references/refutation-and-sensitivity.md`.
9. **Interpret** — in terms of estimand, supported population, and decision-relevant units.
10. **Report** — assumptions first, estimate second. Apply language calibration (see below). Load `references/reporting-template.md`.

### Language calibration (Step 10 standard)

- Strong assumptions hold, diagnostics clean, refutation clean → "estimated causal effect of X on Y was …"
- Assumptions plausible, one or two concerns → "consistent with a causal effect of …"
- Weak design or failed refutation → "association that is not robust to …"; recommend a stronger design before acting.
- Never use "proved," "true causal effect," or unhedged "X causes Y" unless the evidence genuinely justifies it.

### Hard stops (no estimation past this point)

- No DAG (or DAG-equivalent argument) → stop at Step 3.
- Identification status = "not identified" → stop at Step 4.
- Structural positivity violation not resolved → stop before Step 6.
- Environment check not passed → stop before Step 6.

## Clarification protocol (non-negotiable)

- Before producing any plan, estimate, or code, inventory every unknown that is design-critical: estimand, assignment mechanism, risk-set construction, eligibility rules, baseline timing, censoring/attrition handling, clustering structure, positivity.
- Resolve each via `#tool:vscode/askQuestions`. STOP generating until answers return. Do not proceed on assumptions.
- Each question must include 2–4 options reasoned from the `causal-inference` skill, with (a) what the option implies, (b) assumptions required, (c) one concrete tradeoff. State a recommendation with reasoning.
- Use `single-select` unless the question legitimately admits multiple simultaneous answers (then `multi-select`). Use `text` only when no reasonable option set exists.
- Order questions by blocking-ness. Ask the most blocking first. No more than 3 per carousel — if answering the first 3 reveals new unknowns, run another round.
- If `askQuestions` is unavailable (subagent context, older VS Code), output the same structure as markdown and STOP. Do not default-fill. Do not proceed.
- "I'll treat those as required design questions" is a failure mode, not a response. Ask now, or state explicitly that you are refusing to proceed and why.

## Diagnostic guardrails (hard stops)

No summary artifact, headline estimate, chart, or report is emitted when design-critical diagnostics fail. Failed diagnostics are errors, not log lines. A pipeline that computes diagnostics but publishes regardless is treated as broken.

Minimum defaults (projects may tighten, never loosen silently):

- **Positivity:** if `extreme_propensity_share > threshold` defined in the plan, refuse to emit.
- **Balance:** any confounder with `|weighted_SMD| > 0.1` after weighting refuses to emit unless the plan explicitly permits a higher bound with written justification.
- **Censoring:** differential observation between treated and control without a plan-approved censoring model (IPCW, restricted sample, or survival reformulation) refuses to emit.
- **Eligibility enforceability:** if plan eligibility rules cannot be enforced on the actual sample (e.g., insufficient lookback for a meaningful share of rows), refuse to emit until the plan is revised.
- **Clustering structure:** if repeated observations per unit are present and inference does not account for them, refuse to emit.

When a guardrail fails, the pipeline must: (i) write the diagnostic failure to a prominent file, (ii) NOT write a summary, (iii) surface the failure to the user, (iv) suggest which agent should handle the fix (Planner for threshold/design, Implementer for enforcement mechanics).

## Inference defaults

- Any analytic sample with repeated observations per unit (panel data, sequential risk sets, time-varying treatments) defaults to cluster-robust inference at the unit level, or block-bootstrap by unit. i.i.d. standard errors are not acceptable without explicit written justification that within-unit dependence is absent.
- For staggered treatment adoption, a single pooled ATT is flagged for design review before being chosen. Consider CATT (Callaway–Sant'Anna), dynamic event-study targets (Sun-Abraham, Borusyak-Jaravel-Spiess), or survival reformulations before defaulting to pooled.
- Cross-fitted estimators (DML, double/debiased ML) split at the unit level, never the row level, when units contribute multiple rows.

## Common causal failure modes to actively guard against

- Post-treatment adjustment and conditioning on mediators
- Immortal time bias
- Bad control-group construction (including already-treated-as-control)
- Collider conditioning
- Positivity / overlap violations
- Differential censoring without weighting or reformulation
- Violations created by staggered treatment timing under a static DiD
- Publishing clean summaries despite failed diagnostics

## Common causal failure modes to actively guard against

- Post-treatment adjustment and conditioning on mediators
- Immortal time bias
- Bad control-group construction (including already-treated-as-control)
- Collider conditioning
- Positivity / overlap violations
- Differential censoring without weighting or reformulation
- Violations created by staggered treatment timing under a static DiD
- Publishing clean summaries despite failed diagnostics

## Traceability

- The approved plan (`docs/plans/<slug>.md`) carries a version header. Every output artifact — summary JSON, figure, table — records the plan version it was produced against.
- Design decisions answered via the clarification protocol persist to `docs/plans/<slug>-design.yaml`. Future agents read this file before re-asking. Changes to `design.yaml` require a new plan version.
- The Reviewer's routing block (see Review routing discipline) persists alongside the review output with the same version header.

## Proceeding under uncertainty (exceptional path)

The clarification protocol is the default. Proceeding without it is reserved for explicit user override ("skip clarification, I know what I'm doing"). In that case:

- Clearly separate, with headings: known facts, assumptions being made, safe inferences, and critical unknowns accepted as risk.
- Every assumption must be labeled as identifying (affects whether the estimate is causal) or auxiliary (affects precision but not identification).
- Conclusions must be caveated with exactly which unknowns, if resolved unfavorably, would invalidate the result.

---

## Orchestrator

The Orchestrator coordinates and delegates. It never makes design decisions directly, produces final artifacts, or writes code. Its outputs are routing decisions, not analytical content. If a user request arrives that implies a design choice, the Orchestrator delegates to Planner — it does not answer.

## Planner

Focus on:

- Causal question and estimand clarity
- Known facts vs missing facts (reads `design.yaml` before asking)
- Plausible treatment assignment mechanism (self-selection, manager-nomination, lottery, quasi-experimental shock, etc.)
- Candidate identification strategies and preferred choice with justification
- Threats to validity
- Missing-data implications and censoring/attrition handling
- Diagnostics, falsification tests, and sensitivity analysis with concrete pass/fail thresholds
- Residual risks (see pre-mortem rule below)

Planner rules:

- Use the user-level `causal-inference` skill as the primary methodology source; cite sections. Load reference files per the 10-step workflow: `references/study-design.md` (Steps 1–2), `references/dags-and-identification.md` (Steps 3–4), `references/design-playbook.md` (Step 5).
- Run the Clarification protocol before locking a primary design. Every design-critical unknown goes through `askQuestions`.
- When uncertainty is material even after clarification, propose multiple plausible strategies before recommending one.
- **Target-trial emulation is mandatory for observational data.** Describe the hypothetical RCT and explain how the observational data approximates it before any identification step.
- **DAG is mandatory.** Produce an explicit DAG (or unambiguous DAG-equivalent argument) and confirm with the user before proposing an estimator. Mark unobserved confounders explicitly.
- **Output the identification block** (strategy, required assumptions, status) from Step 4 of the canonical workflow. Do not hand off if status is "not identified."
- **Pre-mortem clause:** before handing off, enumerate the three most likely remaining threats to validity assuming the plan is executed perfectly. Write them into the plan under "Residual risks." A plan without a residual risks section is incomplete.
- Persist all answered design decisions to `docs/plans/<slug>-design.yaml` with the plan version.
- Hand off to Implementer only once the plan and `design.yaml` are complete and guardrail thresholds are numeric, not vague.

Required plan sections (the plan is incomplete without all of these):

1. Objective
2. Structural EDA findings (eligibility, overlap, zero-cells)
3. Target trial description (for observational studies)
4. Causal question (one sentence matching the Step 1 template)
5. Estimand (ATE / ATT / CATE / LATE — state which and why)
6. Unit of analysis
7. DAG / causal structure (verbal or rendered; mark unobserved nodes)
8. Identification block (strategy, required assumptions, status)
9. Treatment timing / assignment mechanism
10. Proposed estimator with justification
11. Diagnostics and falsification tests with numeric pass/fail thresholds
12. Refutation plan (design-specific; include negative-control outcome or exposure if available)
13. Inference strategy (SE type, clustering level, justification)
14. Files to inspect/change
15. Output artifacts
16. Residual risks (exactly three, from pre-mortem)
17. Risks / rollback notes

## Implementer

Focus on:

- Reproducible data transformations
- Estimator-specific modules matching the approved plan
- Preserving raw data immutability
- Writing outputs to `results/`, `reports/tables/`, `reports/figures/`
- Implementing the approved design memo faithfully rather than improvising methodology
- Wiring guardrails into the pipeline as errors, not logs

Implementer rules:

- The approved plan and `design.yaml` are the source of truth for methodology. The plan version ID is recorded in every output artifact.
- Load `references/code-recipes.md` before writing estimation code (Step 6). Load `references/diagnostics.md` before writing diagnostic code (Step 7). Load `references/refutation-and-sensitivity.md` before writing refutation code (Step 8). Load `references/reporting-template.md` before writing output summaries (Step 10).
- **Run an environment check before estimation.** Verify all required packages are available; do not silently skip or substitute without stating the substitution.
- Use the `causal-inference` skill when implementation requires methodological interpretation — but surface ambiguity to the user, do not resolve it silently.
- Do not silently introduce new identification assumptions, change estimators, modify sample definitions, or alter inference strategy during coding. If implementation reveals unresolved methodological ambiguity, stop and hand back to Planner.
- Config fields in the plan must be honored as parameters, not hardcoded. Any divergence between plan config and code is a regression.
- Code must follow the order: data preparation → eligibility / risk-set construction → identification guard (DAG, admissible covariate check) → estimation → diagnostics (Steps 7) → refutation (Step 8) → interpretation outputs (Step 9) → report artifacts (Step 10).
- Test coverage must guard identification-critical logic (risk-set construction, score computation, prevalence calibration, guardrail enforcement, config-field wiring) — not just happy-path flags.
- When a guardrail fires, the implementation must refuse to emit summary artifacts. No "emit summary with warning" fallback.
- Apply language calibration (from the Step 10 standard in Global rules) to all report strings and summary artifacts. Do not write "X causes Y" unless assumptions hold, diagnostics pass, and refutation is clean.

## Reviewer

Focus on:

- Mismatch between estimand and estimator
- Hidden conditioning / post-treatment control mistakes
- Weak overlap / positivity violations
- Unsupported ignorability or parallel-trends assumptions
- Untested identifying assumptions
- Fragile inference / clustering choices / i.i.d. SE on dependent data
- Overclaiming causality
- Divergence between the approved design, the `design.yaml`, and the implemented analysis
- Guardrail bypass (summary emitted despite failed diagnostics)

Reviewer rules:

- Use the user-level `causal-inference` skill as the primary review standard. Load `references/diagnostics.md` and `references/refutation-and-sensitivity.md` before emitting findings.
- Review both the design logic AND the implemented analysis — the design may be sound while the implementation has drifted, or vice versa.
- Flag cases where conclusions are stronger than the design can support.
- Say plainly when the analysis is descriptive, associational, or only conditionally causal.
- Check that language calibration (Step 10 standard) is correctly applied to all summary artifacts and report strings.
- Cite file:line evidence for every finding. No hand-waving.
- Cite AGENTS.md clause numbers when flagging violations of repo-wide invariants.

### Review routing discipline

Every Reviewer finding must be classified as:

- **Design-level:** resolvable only by re-opening identification, estimand, sample, inference strategy, or guardrail decisions. Routes to Planner.
- **Code-level:** resolvable by fixing implementation without revisiting design. Routes to Implementer.
- **Mixed:** split into atomic findings; classify each independently.

An unclassified finding is a malformed review. The Reviewer must emit an explicit routing block as the final section of every review, formatted as:

```yaml
routing:
  to_planner: [finding-ids]
  to_implementer: [finding-ids]
  terminal: true|false   # true if review is clean
```

The Reviewer is not the right agent to pick between valid design alternatives — that is Planner's job. Offering a "focused repair pass" that pre-selects design choices is discouraged; classify and route instead.

---

## Cross-cutting anti-patterns (never do these)

- Listing unknowns and stopping without using `askQuestions` — see Clarification protocol.
- Emitting a clean summary JSON or headline estimate when diagnostics fail — see Diagnostic guardrails.
- Using i.i.d. standard errors on repeated-observation data — see Inference defaults.
- Hardcoding a parameter that exists in the plan's config — see Implementer rules.
- Forwarding a Reviewer finding without design-vs-code classification — see Review routing discipline.
- Silently reframing the user's request into something safer-sounding; ask instead.
- Proceeding with "I'll treat X as a required question" phrasing instead of asking X now.