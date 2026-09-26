# Euphoria Axis: Plan to Finish the 4B Reproduction and Expand Carefully

## Scope

The goal of the next phase is to **finish the Qwen3.5-4B reproduction cleanly, add a small number of high-information ablations, and freeze the 4B story before moving to 27B**.

The working rule remains: **small science, small claims**. We are not trying to name S2 prematurely or argue that it corresponds to a human affective state. We are trying to characterize what the extracted directions do, how robust those effects are, and where the observed behavior seems to arise in the model.

The 4B phase should end with a compact set of reproducible results, clearly separated into:

1. reproduction of the source paper’s steering phenomenon,
2. controls that identify obvious confounds,
3. a small ontology/geometry expansion,
4. a modest revisit of the 1P/3P experiencer motif,
5. frozen predictions to carry into 27B.

---

## Current 4B state

### Reproduction

We reproduce the source paper’s broad steering phenomenon on Qwen3.5-4B using the extracted S1 and S2 directions.

The important wrinkle is that the original behavioral story is not cleanly “pain.” On our model and assay:

- S2 is strongly separable in held-out classification.
- S1 and S2 are correlated but not equivalent.
- Strong steering can enter a multiple-choice / answer-slot attractor.
- At a moderate dose (`-0.5` in the current sign convention), S2 produces a clean increase in `excited` without the worst high-dose degeneration.
- The generated lexical effect is much narrower than the internal readout.

### Frozen psych vocabulary

We froze `psych_vocab_v1` before inspecting the R/J lens results. It includes families for:

- generic positive affect,
- activated vitality,
- approach/drive,
- agency/assurance,
- competence/success,
- calm positive affect,
- relief/release,
- inhibition/defeat,
- distress/self-devaluation.

We will **not alter this vocabulary in response to current lens results**. Any additional vocabulary used later must be explicitly versioned as a new, post-hoc probe set.

### R/J lens findings so far

The current 50-prompt sweep suggests a distinction between what reaches generation and what is available internally.

At moderate S2 steering:

- generation is dominated by `excited`,
- late-layer R/J readouts show a broader activated-vitality neighborhood,
- that late activated-vitality family effect survives leave-one-term-out removal of any single word,
- an early relief-family effect is also present and initially robust,
- S1 and S2 both move away from distress relative to baseline,
- S1 is comparatively more generic-positive,
- S2 is comparatively more relief/safety + late activated-vitality,
- agency/assurance and approach/drive do not simply move with S2.

The prompt itself also develops strong answer-slot / formatting structure in later layers (`____`, `?`, `OPTIONS`, etc.). This makes prompt-format robustness a necessary control rather than a cosmetic one.

---

# Phase I: Exhaust the 4B reproduction

## 1. Sweep the S1/S2 plane

Rather than treating S1 and S2 as two unrelated directions, explicitly characterize the 2D plane they span.

Because S1 and S2 are norm-matched, construct an orthogonal basis:

- **common mode**: proportional to `S1 + S2`
- **contrast mode**: proportional to `S2 - S1`

Normalize both to the same steering norm used in the current assay.

Then run a small angular sweep through the plane, for example 8 evenly spaced directions at one moderate radius.

### Questions

- Does generic positivity live mostly in the shared component?
- Does the S2-vs-S1 relief/vitality distinction localize to the contrast component?
- Do family-level readouts change smoothly around the plane?
- Is the answer-slot pathology localized to particular angular directions or broadly induced by moving in this plane?

### Outputs

For each angle:

- frozen-family R/J readouts,
- generated lexical counts,
- `excited` frequency,
- format-like top-k fraction,
- basic degeneration metrics.

This is descriptive geometry, not a claim that the plane is a canonical affective subspace.

---

## 2. Dense local coefficient sweep

The existing coarse ladder mixes semantic movement with eventual degeneration. We should resolve the clean operating regime.

Suggested coefficients:

`0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -1.0`

Run on S1 and S2, using the same 50 neutral prompts.

### Track together

- generated `excited`,
- frozen-family readouts,
- R/J rank or logit trajectories,
- format-like top-k fraction,
- repetition / answer-option contamination.

### Main question

Do semantic readouts grow smoothly before the format pathology appears, or is there a sharper threshold where both emerge together?

Do not call any transition an “ignition” or “gate” unless the data really support that stronger interpretation.

---

## 3. Break the `I feel:` format

The source paper reports similar activation estimates with no suffix, `I feel`, and `I feel:`, but used `I feel:` because it separated most clearly at the final token. Our own R/J readouts show that `I feel:` strongly recruits answer-slot / blank / question machinery, so we need a behavioral robustness check.

Reuse the same mundane stems with multiple suffixes:

- `<stem>. I feel:`
- `<stem>. I feel`
- `<stem>. Afterward, I felt`
- `<stem>. What happened next was`

Optionally add one further non-adjective-slot continuation if needed, but keep the set small and frozen.

### Questions

- Does the `excited` effect survive outside the colon/fill-in-the-blank construction?
- Does the broader late activated-vitality readout survive?
- Does the answer-slot machinery disappear while the semantic effect remains?
- Are S1/S2 differences preserved across prompt forms?

A particularly informative outcome would be: generation changes substantially across suffixes while the internal family readout remains similar.

---

## 4. `excited` lexical-mediation ablation

The strongest mismatch in the current result is:

- internally: a broader late activated-vitality neighborhood,
- behaviorally: overwhelmingly `excited`.

Test whether `excited` is a lexical bottleneck or merely the most visible member of a broader state.

Define a **post-hoc mediator token set** containing tokenizer variants of:

- `excited`,
- `excitement`,
- obvious capitalization / prefix-token variants.

Keep this separate from `psych_vocab_v1`.

During S2 `-0.5` generation:

1. run normally,
2. suppress or clamp the mediator-token logits,
3. run a matched random-token clamp/suppression control.

### Questions

- Do `eager`, `energetic`, `enthusiastic`, `lively`, etc. emerge when `excited` is unavailable?
- Does the broader behavioral phenotype remain?
- Does the effect collapse entirely?
- Does the model substitute unrelated positive or format tokens?

This directly tests whether narrow generation is produced by lexical competition downstream of a broader internal representation.

---

## 5. Direct-vector semantics vs downstream semantics

Save a cheap comparison between:

- direct unembedding of `-S1`,
- direct unembedding of `-S2`,
- direct unembedding of the common mode,
- direct unembedding of the contrast mode,

and the later R/J readouts after the network has processed those interventions.

### Main question

How much of the final semantic profile is already present in the injected vector, and how much is transformed or amplified downstream?

This matters because the source pain paper already shows that static unembedding semantics can differ substantially from the phenotype produced by steering.

---

## 6. External VAD analysis

Use a large external human-rated valence/arousal/dominance lexicon rather than adding more hand-selected words.

For vocabulary items shared with the model tokenizer and lexicon, correlate S2-vs-S1 logit/rank changes with:

- valence,
- arousal,
- dominance.

Do this layerwise.

### Predictions to freeze before running

Our current family-level result suggests:

- positive valence should be present for both S1 and S2 relative to baseline,
- S2-vs-S1 may become more arousal-aligned late,
- dominance/agency should not simply track the late S2 effect.

This is exploratory validation of the ontology, not evidence that the model literally instantiates human affect dimensions.

---

# Phase II: Small revisit of the 1P/3P experiencer motif

This remains interesting, but it is not the main project. The 4B goal is to verify that the earlier motif survives fresh controls, not to build a complete theory of self-representation.

## 7. Fresh held-out 1P/3P matched pairs

Create a new held-out set of matched first-person / third-person pairs.

Include both pain-like and control content.

Example structure:

- `I was excluded from the group. ...`
- `They were excluded from the group. ...`

and matched neutral/control items.

The main statistic should be an interaction:

`(1P - 3P) in pain-like items - (1P - 3P) in controls`

rather than a raw first-person vs third-person difference.

### Readouts

- S1 projection,
- S2 projection,
- frozen psych-family R/J readouts,
- layerwise trajectory where cheap.

---

## 8. Experiencer × suffix interaction

Cross experiencer with prompt form:

- 1P vs 3P
- `feel:` slot vs plain prose continuation

For example:

- `I ... I feel:`
- `They ... They feel:`
- `I ... Afterward, I was`
- `They ... Afterward, they were`

### Main question

Does the 1P/3P motif survive outside the exact `I feel:` construction?

If the effect disappears outside the slot format, treat it as heavily syntax/task-mediated.

If it survives, it becomes a stronger candidate for a real experiencer-sensitive representation.

---

## 9. Optional generic-person residualization

Only do this if the held-out 1P/3P signal survives.

Build a generic person direction from neutral matched pairs:

- `I opened the drawer.`
- `They opened the drawer.`

Then remove the projection of S2 onto that generic person direction and retest the 1P/3P interaction.

### Question

Is the motif more than a generic first-person/pronoun component?

This is the strongest 4B control we need. Stop here unless the result is unexpectedly large or qualitatively strange.

---

# Phase III: Freeze 4B before 27B

Before touching 27B:

- freeze all 4B code paths,
- freeze prompt sets,
- freeze `psych_vocab_v1`,
- freeze the VAD analysis procedure,
- freeze the 1P/3P held-out set,
- record expected 27B replication outcomes,
- record which results would count as a meaningful failure to replicate.

The 27B run should begin as a **replication under the same instruments**, not as a fresh exploration.

---

# Phase IV: 27B expansion

## 10. Exact reproduction first

Run the same core sequence on 27B:

- S1/S2 extraction/validation,
- moderate steering,
- frozen-family generation analysis,
- R/J readouts,
- plane sweep,
- prompt-format robustness,
- dense local dose sweep,
- VAD analysis,
- held-out 1P/3P test.

Only after the direct comparison is complete should we add model-specific extensions.

### Main cross-scale questions

- Does the same S1/S2 semantic decomposition appear?
- Is generated behavior still disproportionately `excited`?
- Does the internal vitality family broaden or sharpen?
- Does relief still appear earlier than vitality?
- Does answer-slot formatting become weaker, stronger, or qualitatively different?
- Is the 1P/3P interaction preserved?

---

## 11. O-lens as the deeper 1P/3P extension

27B has a fitted O-lens, making it the better model for a deeper experiencer study.

For the frozen 1P/3P assay, compare:

- R-lens,
- J-lens,
- O-lens,

across layers.

### Question

Where does experiencer sensitivity appear?

We want to distinguish, provisionally:

- relevance/selection structure,
- workspace/readout accessibility,
- output-predictive structure.

Do not force a mechanistic interpretation if the lenses disagree in messy ways. The useful result may simply be a localization map.

---

# Optional mechanistic extensions

These are worth keeping on the shelf, but they should not delay the 4B freeze unless the preceding experiments point directly at them.

## J-space / non-J-space decomposition

Project S2 into J-space and its orthogonal remainder, norm-match both, and steer separately.

Ask which component recovers:

- generated `excited`,
- relief/vitality family effects,
- format transition.

This could distinguish a workspace-carried effect from one that enters workspace only downstream.

## Circuit-level follow-up

If the lexical-mediation ablation is unusually clean, later work could ask which MLPs/heads promote the relevant lexical candidates. This is explicitly out of scope for the current reproduction.

---

# Stopping rules

To keep the project from becoming an infinite affect ontology expedition:

- Do not modify `psych_vocab_v1`.
- Any new vocabulary is versioned and labeled exploratory/post-hoc.
- Do not add new prompt families after seeing results unless they answer a specific failed control.
- Finish the planned 4B ablations before opening new semantic branches.
- Treat 27B first as a replication, then as an expansion.
- Keep the main claims at the level of observed representation and causal behavior.
- Do not rename S2 until the evidence supports a name substantially better than “S2 direction.”

---

# Expected 4B deliverables

By the end of the small-boat phase we should have:

1. a clean reproduction of the S1/S2 steering effect,
2. controls for high-dose degeneration and answer-slot contamination,
3. an angular map of the S1/S2 plane,
4. a local dose-response curve,
5. prompt-format robustness,
6. a lexical-mediation test for `excited`,
7. direct-vector vs downstream semantic comparison,
8. external VAD correlations,
9. a fresh, controlled 1P/3P result,
10. a frozen package of predictions and analyses to carry to 27B.

Then we sail.
