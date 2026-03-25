# Workshop pacing — Causal inference notebook (80 minutes)

Pacing guide for [causal_inference_workshop.ipynb](causal_inference_workshop.ipynb). **Body content is tables only.** Fill the **Lead** column with team member names as you assign ownership.

## Document header


| Field          | Value                                                                     |
| -------------- | ------------------------------------------------------------------------- |
| Title          | SIOP causal inference workshop — live pacing                              |
| Notebook       | `causal_inference_workshop.ipynb`                                         |
| Session length | 80 minutes (default allocation below; ~5–8 min flex for Q&A or slow runs) |
| Format         | Live walkthrough; students run cells while instructor narrates            |
| Environment    | Colab (setup cell) or local clone + `SET-UP` code path                    |


## Teaching map


| Checkpoint | Notebook anchor / heading                                                                                                    | Notes                                     | Lead |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ---- |
| Intro.     | PPT Slides 1-7                                                                                                               |                                           |      |
| Environ.   | (PPT Slide 8) Colab instructions + clone/install cell; local `SET-UP`                                                        | Run first; idempotent setup               |      |
| **CP1**    | `## Checkpoint 1: Context & Overview` + case study, timeline, `## The Data & Outcomes`, DAG                                  | Story + measurement + causal picture      |      |
| **CP2**    | `## Checkpoint 2: Diagnostics & Causal Identification Strategy` + diagnostics code (`CausalDiagnostics`, VIF, overlap)       | ATE vs ATT; credibility before estimation |      |
| **CP3**    | `checkpoint3.png` then `## Estimating Average Treatment Effect using IPTW + GEE...` through `## Global Technical Summary...` | No `## Checkpoint 3` heading in notebook  |      |
| **CP4**    | `## Checkpoint 4: Key Takeaways for Stakeholders`                                                                            | Stakeholder narrative                     |      |
| **CP5**    | `## Checkpoint 5: Further Learning` (DML, Causal Forest, method comparison)                                                  | Often async or pointer-only in 80 min     |      |


## Checkpoint schedule


| Block                                                  | Minutes  | Purpose                                             | Lead |
| ------------------------------------------------------ | -------- | --------------------------------------------------- | ---- |
| Intro + Environment                                    | **10**   | Learning goals; Colab vs repo; run setup once       |      |
| **CP1** Context & overview (pair with PPT slides 9-14) | **10**   | Story; estimand intuition; DAG / threats            |      |
| **CP2** Diagnostics & identification                   | **20**   | Why ATE fits; overlap and diagnostics               |      |
| **CP3** Estimation & results                           | **30**   | IPTW+GEE; survey results; retention; global summary |      |
| **CP4** Stakeholders                                   | **5**    | HR/L&D narrative; recommendations                   |      |
| **CP5** Further learning                               | **2**    | Orient to DML/HTE; pointer or async                 |      |
| Buffer                                                 | **~5–8** | Transitions; questions; overtime on CP3             |      |


## Opening + environment (~10 min total)


| Sub-topic   | Minutes | Notebook focus                                                                                           | Live teaching emphasis                                           | Lead |
| ----------- | ------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---- |
| Intro.      | ~5      | Slides 1-7                                                                                               | Intro to session, causal inference, and teaching goals           | MP   |
| Environment | ~5      | Slide 8 + Colab setup cell vs local `SET-UP`; paths; imports `CausalDiagnostics`, `CausalInferenceModel` | Students run setup + load data; instructor narrates over outputs | MP   |


## CP1 — Context & overview (~10 min total)


| Sub-topic             | Minutes | Notebook focus                         | Live teaching emphasis                                                    | Lead |
| --------------------- | ------- | -------------------------------------- | ------------------------------------------------------------------------- | ---- |
| Case study & timeline | ~2.5    | Intro markdown through timeline figure | Voluntary treatment; uneven promotion; mid-year review stakes             | MP   |
| Data & outcomes       | ~2.4    | `## The Data & Outcomes`               | Sample sizes; survey scales vs retention / survival measurement           | MP   |
| DAG & confounding     | ~5      | `### Causal DAG...`                    | `X→T`, `X→Y`, org→promotion→`T`; why later methods; E-values foreshadowed | MP   |


## CP2 — Diagnostics & identification (~20 min total)


| Sub-topic                 | Minutes | Notebook focus                                                             | Live teaching emphasis                                       | Lead |
| ------------------------- | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------ | ---- |
| ATE vs ATT                | ~3      | CP2 opening table                                                          | Tie rows to scale vs participants; justify **ATE**           | LS   |
| Positivity / restriction  | ~5      | `### Positivity Violation: No Low Performers...` + restricted descriptives | Logic of restriction; concepts over re-deriving every number | LS   |
| Diagnostic prep           | ~1      | Variable lists; `CausalDiagnostics`                                        | What the class does (high level)                             | LS   |
| Multicollinearity screens | ~5      | Intercorrelations; VIF                                                     | Concept and thresholds; skip line-by-line if tight           | LS   |
| Overlap / propensity      | ~5      | Pre-modeling overlap diagnostics                                           | Core credibility question (non-optional heart of CP2)        | LS   |
| Bridge to estimation      | ~1      | Narrative before `checkpoint3.png`                                         | Estimand + overlap → proceed to IPTW+GEE                     | LS   |


## CP3 — Estimation & results (~30 min total)


| Sub-topic                | Minutes | Notebook focus                                                                         | Live teaching emphasis                                                                                           | Lead |
| ------------------------ | ------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---- |
| IPTW + GEE rationale     | ~5      | `## Estimating Average Treatment Effect...` through `## Summary: Model Design Choices` | Weights + clustered SEs + covariate adjustment; pointer-only optional: continuous/binary subsections, GEE vs OLS | MP   |
| Survey ATE execution     | ~8      | ATE loop + summary table                                                               | Cohen’s d; FDR from printed summary                                                                              | MP   |
| Interpretation blocks    | ~4      | Self-report bias caveat; balance; E-values                                             | Interpretation, not full sensitivity lecture                                                                     | LS   |
| ATE vs ATT               | ~3      | `## ATE vs. ATT: What Changes?`                                                        | Weight intuition; stakeholder choice; optional code stays commented                                              | LS   |
| Retention / survival     | ~8      | Retention markdown + Cox / KM outputs                                                  | Why not four binary GEEs; time interaction; HR intuition; survivor caveat; KM quick or pointer                   | NL   |
| Global technical summary | ~2      | `## Global Technical Summary...`                                                       | One-slide consolidation before CP4                                                                               | NL   |


## CP4 — Stakeholders (~5 min total)


| Sub-topic           | Minutes | Notebook focus                                                            | Live teaching emphasis                                | Lead |
| ------------------- | ------- | ------------------------------------------------------------------------- | ----------------------------------------------------- | ---- |
| Narrative & actions | ~5      | `### What We Found`; fade story; `### Recommendations`; `### Bottom Line` | Decisions and caveats; minimize formula re-derivation | IU   |


## CP5 — Further learning (~2 min live)


| Sub-topic   | Minutes | Notebook focus                                                               | Live teaching emphasis                                                        | Lead |
| ----------- | ------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---- |
| Orientation | ~2      | `### Why Explore Alternative Methods...`; Part 1 / Part 2; method comparison | Pointer to DML + Causal Forest; full math / HTE plots async unless time added | NL   |


## Optional / pointer-only segments

Use to save time; say “the notebook spells this out — focus on [topic].” **Est. time saved** is indicative only.


| Area                              | Notebook location                                | Suggested shortcut                    | Est. time saved | Lead |
| --------------------------------- | ------------------------------------------------ | ------------------------------------- | --------------- | ---- |
| Colab edge cases                  | Cleanup cell; Drive copy notes                   | Mention idempotent setup only         | ~1–2 min        |      |
| Full data hygiene walkthrough     | `# DATA HYGIENE CHECK`                           | Headline only                         | ~3–5 min        |      |
| Every descriptive plot/table      | `# DESCRIPTIVE EXPLORATION`                      | One or two examples                   | ~5–10 min       |      |
| Check 1 / Check 2 detail          | Intercorrelations; VIF cells                     | One threshold + concept               | ~3–5 min        |      |
| Binary outcome / logit subsection | Under IPTW + GEE                                 | Skip if continuous-only session       | ~2–3 min        |      |
| Deep GEE vs OLS                   | `### Why GEE Instead of OLS...`                  | One sentence: clustering + GLM        | ~2–4 min        |      |
| ATT rerun cells                   | Commented `ATT ANALYSIS — OPTIONAL`              | Use pre-computed markdown table       | ~5–10 min       |      |
| Cox internals                     | Categorical vs continuous time interaction essay | Period HRs only; skip derivation      | ~3–5 min        |      |
| Kaplan-Meier                      | `### Kaplan-Meier Survival Curves`               | Flash figure or “see notebook”        | ~2–3 min        |      |
| Retention inference notes         | `### Inference Notes`                            | Reading assignment                    | ~2–3 min        |      |
| Entire CP5 live                   | DML + Causal Forest + method comparison          | Async / advanced track                | ~15–25 min      |      |
| Global summary subsubsections     | Methodological notes; threats                    | Rely on CP4 + global doc as reference | ~2–5 min        |      |


## Discussion and pause prompts


| When | Prompt                                                                | Related notebook anchor                           | Lead |
| ---- | --------------------------------------------------------------------- | ------------------------------------------------- | ---- |
| CP2  | Why is **ATE** the right estimand for “should we scale this program?” | CP2 opening ATE vs ATT table                      |      |
| CP2  | After restriction: what **positivity** problem were we avoiding?      | Positivity violation markdown + restricted sample |      |
| CP2  | Do treated and controls have **enough overlap** to trust comparison?  | Propensity / overlap diagnostics                  |      |
| CP3  | How much should we trust **self-reported** efficacy vs **retention**? | Self-report bias caveat; retention section        |      |
| CP3  | Does the **ATE vs ATT** story change the scaling recommendation?      | `## ATE vs. ATT: What Changes?`                   |      |
| CP4  | What would you tell HR in **one minute**?                             | `### Bottom Line`                                 |      |


