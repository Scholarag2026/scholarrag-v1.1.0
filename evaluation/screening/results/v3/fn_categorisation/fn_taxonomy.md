# False-negative error taxonomy (screening evaluation, spec E1)

Scope: the 923 false negatives recorded in `Nagtegaal_2019_false_negatives.md` (446), `van_de_Schoot_2017_false_negatives.md` (464) and `Smid_2020_false_negatives.md` (13). A false negative is a record the human screeners labelled as included (protocol `label_field`) that the LLM screener excluded. Each record carries the screener's own stated reason, and this taxonomy classifies that stated reason, not the record itself.

The five categories below are exhaustive over the 923 reasons and are made mutually exclusive by the precedence rule. Codes are intended for the `category` column of `results/<dataset>_fn_categories.csv` (columns `dataset,run,record_id,title,reason,category,notes`).

## Precedence rule

A reason often mentions more than one ground. Apply the codes top-down and take the first match, so that every reason receives exactly one code:

1. `FN-ABS` if the reason states that the abstract was missing or that the judgement rested on the title alone.
2. `FN-POP` if the operative ground is who or what the study is about (target group, population, clinical construct).
3. `FN-DESIGN` if the operative ground is the study design or the publication type.
4. `FN-DEF` if the operative ground is that the intervention or the analytic method does not match the protocol's technical definition.
5. `FN-OTHER` otherwise, including the entries where no reason was recorded.

## Categories

### FN-ABS. Missing or truncated abstract

**Definition.** The screener excluded the record because no abstract was available, so the decision was taken on the title alone and the screener declined to infer eligibility.

**Cue words and phrases.** "No abstract", "No abstract available", "no abstract confirming", "cannot confirm", "insufficient evidence", "title suggests", "title about", "title indicates", "title does not indicate".

**Examples.**

- "No abstract available; cannot confirm relevance to nudges or experimental design." (Nagtegaal_2019, run A, record 642)
- "No abstract available; cannot confirm clustering method or trajectory analysis of PTSD." (van_de_Schoot_2017, runs A and B, record 4835)

### FN-POP. Population, target group or clinical construct judged out of scope

**Definition.** The screener accepted that the record reports an eligible kind of study but excluded it because the people it studies, the group the intervention targets, or the clinical construct it measures were judged to fall outside the protocol's stated population.

**Cue words and phrases.** "targets patients", "not healthcare professionals", "targets service users", "general population", "general public", "primary school students", "not PTSD", "not PTSD symptoms", "depression/anxiety trajectories, not PTSD", "not following an event fulfilling criterion A1", "not human".

**Examples.**

- "Computerized reminders target women patients for mammography, not healthcare professionals' behaviour." (Nagtegaal_2019, runs A and B, record 1474)
- "Trajectories of anxiety/depression symptoms in maltreated children, not PTSD symptoms." (van_de_Schoot_2017, runs A and B, record 3133)

### FN-DESIGN. Study design or publication type judged ineligible

**Definition.** The screener excluded the record because it read the design or the document type as one the protocol rules out, for example a review, a commentary, a protocol, a purely cross-sectional study, or a study with too few measurement waves.

**Cue words and phrases.** "systematic review", "review article", "overview article", "Cochrane review", "narrative review", "literature review", "commentary", "conceptual paper", "theoretical discussion", "trial protocol", "development of a complex intervention", "economic evaluation", "cost-effectiveness analysis", "not an experiment", "no comparison of two or more interventions", "observational", "retrospective", "qualitative", "grounded theory", "erratum", "dissertation", "cross-sectional", "fewer than three waves", "only two time points", "no simulation study".

**Examples.**

- "Systematic review of interventions to improve asthma guideline adherence; systematic reviews are excluded." (Nagtegaal_2019, run A, record 1114)
- "Only two time points measured; fewer than three waves required for trajectory modelling." (van_de_Schoot_2017, run B, record 2410)

### FN-DEF. Eligibility definition applied too strictly

**Definition.** The record is on topic, in the right population and of an eligible design, but the screener excluded it because the intervention or the analytic method was not named in the exact technical terms of the protocol's definition, so a paraphrase or a near synonym was read as a mismatch.

**Cue words and phrases.** "not a nudge per taxonomy", "decision support, not a nudge", "CDSS is not a nudge", "not soft steering", "feedback is not classified as a nudge", "no evidence of nudge framing", "unclear if nudge taxonomy applies", "no clustering method (LGMM/LCGA) mentioned", "not LGMM/LCGA/hierarchical cluster analysis", "uses SEM and hierarchical regressions, not", "not SEM as defined", "no explicit small-sample comparison", "not framed as", "not described as", "not specifically".

**Examples.**

- "Computerized decision support intervention; CDSS is not a nudge per taxonomy." (Nagtegaal_2019, runs A and B, record 436)
- "Longitudinal PTSD study but no clustering method (LGMM/LCGA) mentioned in abstract." (van_de_Schoot_2017, run A, record 4055)

### FN-OTHER. Off-topic dismissal or no reason recorded

**Definition.** Residual category for reasons that dismiss the record as belonging to an entirely different subject area, and for the entries where the screener returned no usable reason string.

**Cue words and phrases.** "unrelated to", "a study from another field", "(no reason recorded)", plus one-off subject dismissals such as pharmacokinetics, cluster munitions and animal behaviour.

**Examples.**

- "Pharmacokinetic study of vasopressin antagonist; unrelated to PTSD trajectory clustering." (van_de_Schoot_2017, run B, record 1691)
- "(no reason recorded)" (Nagtegaal_2019, run A, record 700; 27 of the 923 reasons are of this form)

## Indicative distribution

The table below is an automated cue-word pass over all 923 reasons using the precedence rule above. It is a sanity check that every category is populated and that the five codes cover the whole set. It is not the final coding: the final `category` column is assigned by `results/categorise_fn.py`'s ordered cue-word rules (903 of 923 rows) plus 20 overrides recorded in `results/fn_overrides.json`. A Claude Code agent, not the authors of this software, made those 20 overrides, read and confirmed or overrode the 372 rows the rule flagged low-confidence or could not match, and read a seed-20260905 10 % audit of the high-confidence matches (55 rows); a second, independent Claude Code agent session then ran a blind check on a further 73 rows, shipped at `results/fn_blind_check/` (see `evaluation/README.md` for the full disclosure and the agreement and kappa this file does not repeat). None of these reads has been checked by the authors; they must be re-read before any number here is cited in the manuscript. The final category is recorded per record in `results/<dataset>_fn_categories.csv`.

| Code | Nagtegaal_2019 | van_de_Schoot_2017 | Smid_2020 | Total | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| FN-DEF | 259 | 190 | 7 | 456 | 49.4 % |
| FN-DESIGN | 119 | 148 | 1 | 268 | 29.0 % |
| FN-POP | 40 | 90 | 0 | 130 | 14.1 % |
| FN-ABS | 11 | 23 | 5 | 39 | 4.2 % |
| FN-OTHER | 17 | 13 | 0 | 30 | 3.3 % |
| Total | 446 | 464 | 13 | 923 | 100 % |

Two notes on reading the table. First, 48 of the 923 false negatives carry `abstract: no`, that is they were screened on the title alone, but only 39 stated reasons name the missing abstract as the ground; the remaining 9 give a substantive ground and are coded accordingly. Second, the same record can appear once in run A and once in run B, so the 923 rows are decision-level, not record-level.

## Coding notes

- Code the reason as written. Do not re-read the title or the abstract to decide whether the screener was right; that judgement belongs to the spot checks, not to this taxonomy.
- `FN-DESIGN` and `FN-DEF` are the pair most at risk of being confused. The test is whether the reason disqualifies the document itself ("systematic review", "cross-sectional", "not an experiment") or only the label attached to an otherwise eligible study ("cluster RCT of tailored guideline implementation, but not a nudge intervention per taxonomy"). The first is `FN-DESIGN`, the second is `FN-DEF`.
- `FN-POP` outranks `FN-DESIGN` and `FN-DEF` because reasons that name the wrong target group usually add a second clause restating the intervention criterion, and the target group is the more specific claim.
- Record any disagreement or any reason that resists all five codes in the `notes` column of the CSV rather than inventing a sixth code.
