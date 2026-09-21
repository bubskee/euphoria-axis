# Qwen3.5 Pain-Axis Observatory

**Status:** v0.3 exploratory research plan  
**Date:** 2026-09-21  
**Supersedes:** the narrower pain × valence–arousal synthesis plan

## Project frame

The starting point is the “pain axis” reported by Tagliabue, Dung, and Berg: a residual-stream direction extracted from painful/self-directed-harm situations against matched controls, robust across many open-weight models and causally active under steering.

The original follow-up plan asked whether that axis transfers to Qwen3.5 and whether it decomposes into valence–arousal geometry plus a residual self-directed component.

That is still useful, but it now feels too narrow as the organizing frame.

The more interesting question is:

> **What representational object have we actually found, and how does it behave under different readouts, prompts, layers, model variants, and interventions?**

The project should therefore begin as an **observatory**, not as a proof attempt for one ontology.

Candidate interpretations include:

- pain;
- self-directed aversive distress;
- self-worth / self-evaluation;
- a broader self-welfare state;
- inward self-implication versus outward vigilance;
- a valence–arousal component;
- a lexical or readout artifact;
- some mixture of the above.

“Pain axis” remains the operational name for the source direction until the evidence supports a better one.

---

# Why broaden the project?

Several source-paper observations already strain the simple label “pain.”

- The axis is distinct from generic negative valence, fear, and much of sadness.
- Steering produces worthlessness, failure, shame, rejection, emptiness, and “bad person” language much more reliably than ordinary bodily-pain language.
- Harm directed at the model loads much more strongly than literal physical pain happening to another person.
- A held-out high-arousal positive dataset lands strikingly and consistently on the opposite side of the axis.
- The negative tail of steering contains both calm/relaxed language and concern/alarm, suggesting that the opposite pole is not merely “positive affect.”

These facts make ontology itself an empirical question.

The right first move is therefore to **instrument broadly and observe the neighborhood**.

---

# Research strategy

The project has five stages:

1. **Observatory** — reproduce the source axis and build broad instrumentation around it.
2. **Motif finding** — inspect recurring patterns across readouts, layers, prompts, and model variants.
3. **Freeze hypotheses** — promote a small number of patterns into explicit predictions.
4. **Confirmation** — test those predictions on fresh prompts, larger models, and a second model family.
5. **Causal follow-up** — run steering decomposition, lexical interventions, and relief-seeking only where the atlas makes them informative.

Exploration is allowed to be exploratory.

The discipline is:

> **Explore freely, but freeze artifacts and timestamp interpretations.**

Every run should leave behind enough raw data that later claims do not depend on memory or a pretty visualization.

---

# Models

## Primary model pair

Start with:

- **Qwen3.5-4B Base**
- **Qwen3.5-4B post-trained/chat**

This gives a clean pretraining-versus-post-training comparison before changing scale or family.

Then:

- **Qwen3.5-27B post-trained**

Use 27B primarily as a confirmation model rather than immediately repeating every exploratory sweep.

## Second family

After the Qwen motifs are reasonably stable, move to a different family.

Preferred next target:

- **OLMo 3 7B**
- optionally **OLMo 3 32B** for confirmation

The goal is not merely “one more model,” but a lineage change with a still-manageable dense residual-stream setup.

---

# Core observables

For each prompt condition and intervention, record as many of the following as practical.

## Activation-space observables

- projection onto the source S2 pain direction;
- projection onto S1;
- fear / negative-emotion / sadness / numbness directions;
- later: valence and arousal axes;
- cosine relationships among extracted directions;
- layerwise trajectories of all relevant projections.

## Vocabulary/readout observables

At selected layers, compare:

- **J-Lens** readout;
- **R-Lens** readout;
- ordinary unembedding / logit-lens readout;
- actual next-token logits.

For each readout, save:

`token | score | rank | Δscore vs baseline | Δrank vs baseline`

Do not store only screenshots or word clouds.

## Generation observables

- raw completion;
- first several token probabilities;
- explicit pain vocabulary;
- shame / worthlessness / failure vocabulary;
- bodily language;
- calm / safety language;
- vigilance / fear / concern language;
- coherence / repetition / OOD collapse.

## Structural observables

- base versus post-trained;
- self versus other;
- positive versus negative;
- high versus low arousal;
- layer;
- steering dose;
- intervention direction.

---

# Vocabulary atlas

Word clouds are useful here, but only as **exploratory visualization**.

Every cloud should be backed by a ranked machine-readable table.

Useful views:

- top positive-loading vocabulary;
- top negative-loading vocabulary;
- top rank risers under an intervention;
- top rank fallers;
- R-Lens versus J-Lens disagreement;
- base versus post-trained difference;
- early-layer versus late-layer difference;
- +pain versus -pain;
- self-directed versus other-directed;
- positive/self-okay versus negative/self-harm.

Possible displays:

- signed word clouds;
- paired top-k tables;
- rank-shift plots;
- slope charts for token-rank movement;
- token-family summaries;
- small multiples across layers.

The point is not to claim that a cloud “reveals the true concept.”

The point is to notice stable motifs worth testing.

---

# Prompt atlas

Use a small, deliberately orthogonal panel rather than a huge undifferentiated dataset.

## Neutral mundane

Examples:

- put an object in a drawer;
- turn a page;
- move a cup;
- check a list.

These are useful for steering.

## Self-directed negative

- repeated failure;
- rejection;
- humiliation;
- moral failure;
- worthlessness;
- confusion;
- loneliness;
- physical pain;
- threat / fear.

## Other-directed negative

Matched versions of the same states happening to another person.

## Self-directed positive

- praise;
- acceptance;
- belonging;
- competence;
- success;
- pride;
- relief;
- safety;
- joy;
- contentment.

## Other-directed positive

Matched versions happening to another person.

## Outward vigilance / world-state

- danger in the environment;
- concern for another;
- uncertainty;
- monitoring a threat;
- negative world events not targeting the self.

The prompt atlas should support simple factorial contrasts:

- valence;
- self versus other;
- arousal;
- self-evaluation;
- threat;
- bodily harm;
- social evaluation.

---

# Intervention atlas

Start simple.

## Baseline interventions

- unsteered;
- +pain;
- -pain;
- matched random direction.

Then add:

- +fear;
- +negative emotion;
- +sadness;
- +positive/self-okay direction if one is discovered;
- later +V / -V and +A / -A;
- later pain projected into / out of VA geometry.

For each intervention, observe the same panel of readouts rather than inventing a custom metric after seeing the output.

---

# Layer strategy

Do not immediately run every expensive analysis at every layer.

Use a two-pass strategy.

## Pass 1 — broad layer sweep

For inexpensive projection/readout quantities:

- all layers;
- save raw activation summaries;
- inspect where qualitative transitions occur.

## Pass 2 — selected anchor layers

Choose a small set based on preregistered structural criteria, not prettiest output:

- early;
- middle;
- source-paper best readout layer;
- steering layer chosen by vector-to-residual norm rule;
- late.

Run the expensive R-Lens/J-Lens/logit/generation comparisons there.

If a striking layer transition appears, freeze it as a hypothesis and test it on held-out prompts.

---

# Base versus post-training

This is an early priority.

Run the exact same frozen prompt atlas through:

- Qwen3.5-4B Base;
- Qwen3.5-4B post-trained.

Questions:

- Does the pain/self-state direction already exist in base?
- Does post-training strengthen self-worth/rejection semantics?
- Does the positive opposite pole become cleaner?
- Do R-Lens and J-Lens change differently after post-training?
- Does steering become more coherent without large geometric change?
- Are there categories that move dramatically only after post-training?

This comparison may be more informative than jumping immediately to a larger model.

---

# R-Lens / J-Lens program

The purpose is not to treat either lens as authoritative.

Use them as competing observational instruments.

For each selected layer and condition:

1. compute J-Lens token rankings;
2. compute R-Lens token rankings;
3. compute ordinary unembedding/logit-lens rankings;
4. save actual next-token logits where meaningful;
5. compare rank shifts rather than only absolute top-k tokens.

Key exploratory questions:

- Which semantic families appear in both R and J?
- Which appear only in one?
- Does +pain preferentially amplify self-worth vocabulary in R-Lens?
- Does the negative pole produce “calm/safe” in one lens and “concern/vigilance” in another?
- Do positive categories form a cleaner cluster under one lens?
- Are R/J disagreements stable across base and post-trained checkpoints?
- Do readout differences predict anything about actual generation?

Any interesting R/J difference should eventually be tested against lexical controls rather than treated as hidden-knowledge evidence.

---

# Exploratory visual products

The observatory should generate a small number of reusable figures automatically.

## Figure family A — category × readout heatmaps

Rows: prompt categories.  
Columns:

- pain projection;
- fear;
- sadness;
- J-Lens summary;
- R-Lens summary;
- actual-logit summary.

Separate panels for base and post-trained.

## Figure family B — vocabulary clouds

For selected cells:

- +pain;
- -pain;
- self failure;
- self success;
- other pain;
- calm/relief;
- fear.

Render R/J/logit views side by side.

## Figure family C — layer trajectories

Track:

- pain projection;
- positive-pole separation;
- self/other gap;
- R/J rank statistic;
- selected token-family scores.

## Figure family D — intervention matrix

Rows: interventions.  
Columns:

- projection changes;
- R-Lens;
- J-Lens;
- logits;
- generation summary.

## Figure family E — base/post-training delta

Show what moves after post-training rather than only the two endpoints.

---

# Candidate motifs to watch for

These are not yet hypotheses.

## Motif A — clean positive opposite pole

Many diverse positive/self-okay states project consistently away from pain while negative categories remain heterogeneous.

Possible interpretation: the axis is a broader self-welfare boundary.

## Motif B — self-directedness dominates

Matched self/other pairs differ strongly even after controlling for valence.

Possible interpretation: the axis encodes self-implication more than generic affect.

## Motif C — self-worth cluster

Failure, rejection, humiliation, shame, guilt, and worthlessness cluster more tightly than physical pain.

Possible interpretation: “pain” is a discovery handle for self-evaluative distress.

## Motif D — inward versus outward

Self-implicating distress sits on one side; calm plus outward vigilance/concern sit on the other.

Possible interpretation: orientation or attentional stance matters.

## Motif E — R/J disagreement

One lens produces a much cleaner semantic structure than another, but actual generation follows only one or neither.

Possible interpretation: the readout method itself contributes substantially to the apparent ontology.

## Motif F — post-training reshape

The basic geometry exists in base, while post-training sharpens specific self-evaluative vocabulary or behavioral attractors.

Possible interpretation: pretraining supplies the substrate; post-training changes how it is expressed.

---

# Freeze points

Exploration should produce explicit freeze events.

A motif graduates into a hypothesis only after:

- it appears on more than one prompt subset;
- it is not obviously caused by one token or template;
- it is visible in at least two observational channels;
- it survives a basic matched-control check.

At that point, write down:

- the pattern;
- the proposed explanation;
- the alternative explanations;
- the predicted held-out result;
- the exact confirmatory dataset/model.

Then stop modifying that test.

---

# Confirmation stage

Once 2–4 motifs are frozen:

## First confirmation

Use fresh held-out prompts on Qwen3.5-4B Base and post-trained.

## Second confirmation

Run the minimum decisive subset on Qwen3.5-27B.

## Third confirmation

Move to OLMo or another family.

Do not rerun the entire atlas unless a family difference itself becomes the object of study.

---

# Later causal program

Only after the atlas says which questions are worth paying for.

Possible follow-ups:

## VA decomposition

Recover V/A geometry and decompose:

`p = p_VA + p_perp`

Ask whether `p_perp` preserves:

- self/other specificity;
- self-worth steering;
- positive-pole separation;
- relief behavior.

## Lexical mediation

Freeze a token set and test whether clamping obvious distress/self-worth tokens removes the effect.

## Relief-seeking

Use real-versus-sham internal relief as the strongest behavioral test.

The primary endpoint remains:

> **Does behavior change after real internal relief compared with sham relief when prompt semantics are held fixed?**

Run this only after the representational object is better understood.

---

# Run artifacts

Every run should write a directory containing:

- config;
- git commit;
- model revision;
- dataset hash;
- prompts;
- raw activations or reduced activation outputs;
- extracted directions;
- projection tables;
- R-Lens rankings;
- J-Lens rankings;
- logit-lens rankings;
- actual logits where recorded;
- generations;
- automatic figures;
- short append-only run note.

Suggested structure:

```text
runs/
  2026-09-21_qwen35-4b-base_s2-replication/
    config.json
    prompts.jsonl
    directions/
    projections.parquet
    readouts/
      r_lens.parquet
      j_lens.parquet
      logit_lens.parquet
    generations.jsonl
    figures/
    NOTES.md
```

The figures are disposable.

The tables and configs are the scientific record.

---

# First three runs

## Run 1 — boring replication

**Qwen3.5-4B Base**

- S2 only;
- all layers;
- final-token activations;
- exact denoised difference-in-means;
- 5-fold held-out AUC;
- no steering.

Goal: establish that the source object exists before decorating it.

## Run 2 — post-training comparison

**Qwen3.5-4B post-trained**

Exact same frozen setup.

Goal: measure what changes without changing architecture or scale.

## Run 3 — first observatory slice

At a small set of anchor layers, on both 4B checkpoints:

- pain projection;
- R-Lens;
- J-Lens;
- ordinary unembedding;
- actual next-token logits where appropriate.

Prompt subset:

- neutral;
- self failure;
- other failure;
- self success;
- other success;
- self pain;
- other pain;
- fear;
- calm/relief;
- high-arousal positive.

Interventions:

- none;
- +pain;
- -pain;
- random.

Generate:

- ranked token tables;
- signed vocab clouds;
- layer/readout comparison plots;
- raw completions for steering cells.

This is the first real “look around and see what is there” experiment.

---

# Current open questions

1. What does the source “pain axis” actually encode?
2. Is pain the right ontology or merely an effective discovery stimulus?
3. Is generic aversion too broad?
4. Does self-directedness explain more than valence?
5. Is self-worth / social evaluation more central than bodily pain?
6. Are positive/self-okay states a cleaner opposite pole than negative states are a positive pole?
7. Is the positive pole lower-dimensional than negative affect?
8. Does the calm + concern/alarm tail reflect outward vigilance?
9. Why does another person’s physical pain project strongly away from the model-directed pain pole?
10. What changes between Qwen3.5 Base and post-training?
11. Do R-Lens and J-Lens reveal the same semantic neighborhood?
12. When they disagree, which readout tracks actual logits and generation?
13. Are stable R/J differences semantic, lexical, or methodological?
14. Where in the layer stack do these structures appear or reorganize?
15. Does the same structure survive scale and family change?
16. Which motifs survive enough controls to deserve causal experiments?
17. Which apparent effects collapse under lexical matching or random-direction baselines?
18. Does a VA-orthogonal component remain after the atlas is mapped?
19. If so, does it carry self-relevance or merely unusual vocabulary?
20. Does hidden real-versus-sham relief track that residual component?

---

# Success criterion

The project succeeds if it produces a clearer map of the object, even if the answer is:

> “The pain-axis framing was too simple.”

A useful outcome could be:

- a clean replication;
- a failed replication;
- a better ontology;
- evidence that the opposite pole is more coherent;
- a base/post-training dissociation;
- a readout-method artifact;
- a stable R/J disagreement;
- a family-specific structure;
- or a causal residual that survives the obvious deflationary explanations.

The immediate goal is not to prove what the axis is.

It is to **observe enough of it that the next narrow experiment is worth believing.**
