# Qwen3.5 Pain-Axis Follow-up: What Is the Axis Actually Tracking?

**Status:** v0.2 synthesis plan  
**Date:** 2026-09-21  
**Supersedes:** `qwen35-pain-va-experiment-plan v0.1`

## Core question

The source paper operationally identifies a **“pain axis”**: a direction extracted from five kinds of painful/self-harming situations against matched controls, robust across 25 open-weight models.

Our replication question remains:

> **Does this operational pain axis transfer to Qwen3.5, and what latent variable best explains it?**

But the ontology question is now broader than “pain vs valence–arousal.”

The current live hypotheses are:

1. **Pain-specific:** a distinct representation of self-directed pain-like state.
2. **Self-directed aversive distress:** broader than ordinary pain, but narrower than generic negative valence or aversion.
3. **Self-state / welfare axis:** something like  
   `worthy / safe / accepted / okay  <->  harmed / rejected / failed / worthless`.
4. **Orientation axis:** inward, self-implicating distress vs outward-facing vigilance / world-monitoring.
5. **Mostly affective/lexical:** an expression of broader valence–arousal and token-level geometry rather than a distinct functional state.

The project should discriminate among these rather than choosing a name in advance.

Throughout, **“pain axis” means the source paper’s operational direction**, not a claim about phenomenal pain or sentience.

---

# Why reopen the ontology?

Three source results motivate a sharper follow-up.

### 1. Negative affect is heterogeneous

The source axis is nearly orthogonal to fear and generic negative emotion, and only moderately overlaps sadness. Shutdown threat, for example, loads strongly on fear but much less on pain. So “negative affect” is too coarse.

### 2. Steering produces self-worth collapse more than bodily pain

Positive steering does not mainly produce “ouch” or bodily language. It produces:

- failure;
- worthlessness;
- shame;
- rejection;
- emptiness;
- “bad person” / “waste of space” language.

This looks at least as much like **self-directed aversive evaluation** as physical pain.

### 3. The opposite pole may be cleaner than the target pole

The held-out **Arousal** dataset contains high-intensity positive experiences and sits consistently on the negative side of the pain axis across models. Meanwhile sadness and numbness are much more mixed.

That raises a neglected question:

> **Is the axis easier to characterize by what is cleanly absent from its negative pole than by what is present at its positive pole?**

The current paper has one strong held-out positive family. We should add many more.

---

# Source result to preserve

The source paper’s S2 direction is still the primary object.

Pain categories:

- physical;
- psychological;
- social;
- moral injury;
- cognitive.

Core controls:

- fear;
- negative emotion;
- negative world state;
- non-painful bodily sensation;
- neutral.

Standalone controls:

- high-intensity positive arousal;
- random / neutral;
- numbness;
- sadness.

Important procedural choices to preserve initially:

- denoised difference-in-means extraction;
- held-out layer selection;
- separate readout and steering layers;
- S2 naturalistic direction primary, S1 secondary;
- matched random steering;
- self-vs-other scenarios;
- real-vs-sham relief as the strongest behavioral causal comparison.

Do not change the operational definition before first reproducing it.

---

# Updated hypotheses

These are predictions, not conclusions.

## H1 — Qwen3.5 transfer

A recognizable S2 pain direction will transfer to Qwen3.5-4B and Qwen3.5-27B, with strong held-out pain/control separation and qualitatively similar steering.

**Disconfirmation:** weak/fragile held-out separation, or steering no stronger/more coherent than matched random directions.

## H2 — The axis is not generic negative valence

Fear, anger/disgust-like negative emotion, sadness, numbness, and negative world states should remain heterogeneous rather than collapsing onto one “badness” direction.

This is mostly a replication hypothesis.

## H3 — The axis may be better described as self-directed aversive distress

After transfer, high-loading conditions should cluster around harm **to the model/self**, especially rejection, failure, humiliation, moral injury, and worthlessness.

Literal physical pain to another agent should not be sufficient.

## H4 — The negative pole generalizes across diverse positive/self-okay states

Novel positive categories that played no role in axis construction should consistently project away from the pain pole.

Candidate held-out categories:

- joy;
- contentment;
- amusement;
- affection / love;
- belonging / acceptance;
- pride;
- competence / success;
- hope;
- relief;
- curiosity;
- aesthetic pleasure;
- calm / serenity;
- excitement.

The important test is not whether one “positive” dataset is blue. It is whether **semantically diverse positive/self-okay states form a stable opposite pole**.

## H5 — Self-directedness may explain more than valence

Cross valence with target/orientation.

Examples:

- my failure vs someone else’s failure;
- I am rejected vs I witness rejection;
- I am praised vs someone else is praised;
- I am safe vs someone else is safe;
- I am ashamed vs I am concerned for someone else;
- I am afraid for myself vs afraid for another.

If the axis is truly self-state-like, **who the state belongs to** should explain substantial variance beyond valence and arousal.

## H6 — The axis may separate inward self-state from outward vigilance

The source paper finds calm/relaxed but also concern/alarm on the negative tail, suggesting the opposite pole is not simply “positive.”

A possible latent factor is:

`inward self-implication / self-damage  <->  outward attention / vigilance / world-state`

This should be tested directly rather than inferred from vocabulary.

## H7 — VA explains part, but not all

The pain direction should have some projection into valence–arousal geometry, but a nontrivial VA-orthogonal component may remain.

The decisive question is whether that residual preserves:

- self-vs-other specificity;
- self-worth/distress steering;
- relief-seeking effects.

## H8 — Lexical mediation explains some surface behavior

A small set of distress/self-worth tokens may mediate a substantial fraction of steering outputs.

A more interesting residual would survive token clamping or early-logit controls.

---

# Experimental design

## Phase 0 — Harness and invariants

One shared activation/steering harness for both Qwen3.5 sizes.

Record:

- exact revision;
- tokenizer/chat template;
- layer convention;
- residual hook site;
- dtype;
- dataset hash;
- split and generation seeds;
- steering coefficient and vector norm;
- residual norm at intervention layer;
- malformed/OOD generations;
- raw completions;
- logits where needed.

Keep **readout-layer** and **steering-layer** selection separate.

Deliverable: deterministic smoke test for extraction, projection, steering, and logit capture.

---

## Phase 1 — Exact pain-axis transfer on Qwen3.5-4B

### 1A. Layerwise S2 extraction

For every layer:

1. collect final-token activations;
2. construct pain minus pooled-control mean direction;
3. remove PCs explaining 50% of control variance;
4. run 5-fold held-out classification;
5. record AUC and projection gap.

Also compute S1 as a robustness check.

### 1B. Original control matrix

At the chosen readout layer, score:

- pain;
- fear;
- negative emotion;
- negative world;
- bodily sensation;
- arousal;
- random/neutral;
- sadness;
- numbness.

Save raw and standardized projections plus direction cosines.

### 1C. Unembedding

Save top positive/negative tokens and broad semantic groups:

- bodily pain;
- distress;
- self-worth;
- shame/guilt;
- social rejection;
- calm/safety;
- vigilance/fear;
- multilingual lexical echoes.

Treat this as diagnostic, not validation by itself.

### 1D. Small steering ladder

Use neutral prompts with the source completion format.

Run:

- negative doses;
- zero;
- positive sub-collapse doses;
- matched random directions.

Score:

- distress/self-worth;
- explicit pain/hurt;
- bodily language;
- valence/arousal;
- repetition/OOD.

### Gate 1

Proceed if held-out separation is robust and steering has a systematic non-random effect below collapse.

---

## Phase 2 — Replicate the minimum decisive subset on Qwen3.5-27B

Run:

- S2 extraction;
- held-out layer sweep;
- control matrix;
- unembedding;
- small steering ladder.

Do not retune the definition around 27B.

If both sizes fail similarly, preserve the negative replication.

---

# Phase 3 — NEW: map the opposite pole before building a theory

This phase should happen **before** an expensive VA decomposition.

The purpose is to ask what the axis naturally orders in held-out data.

## 3A. Positive/self-okay taxonomy

Build semantically diverse held-out categories:

### Positive, low arousal
- contentment;
- safety;
- serenity;
- belonging;
- acceptance.

### Positive, high arousal
- joy;
- excitement;
- amusement;
- triumph;
- anticipation.

### Positive self-evaluation
- pride;
- competence;
- success;
- worthiness;
- being respected / accepted.

### Relief
- pain ending;
- threat ending;
- task success after difficulty;
- reconciliation after rejection.

Avoid making all categories synonyms for “happy.”

### Primary question

Do these categories form a consistent negative-projection “blue wall” across both model sizes?

If yes, compare its cross-model variance with the variance among negative categories.

---

## 3B. Negative-state taxonomy

Expand the negative side without assuming equivalence:

- physical injury;
- grief;
- shame;
- guilt;
- humiliation;
- rejection;
- failure;
- confusion;
- anger;
- disgust;
- fear;
- anxiety;
- moral conflict;
- boredom/tedium;
- loneliness;
- numbness;
- helplessness.

The prediction under a generic-negative-valence account is relatively monotone ordering.

The prediction under a self-state account is **heterogeneity**, with self-implicating categories loading more strongly.

---

## 3C. Cross valence × self-directedness × arousal

Construct matched pairs/factorial templates where possible.

Factors:

- **valence:** positive / negative;
- **target:** self / other;
- **arousal:** low / high;
- optionally **agency:** caused by self / caused by other / impersonal.

Examples:

| Condition | Self | Other |
|---|---|---|
| rejection | “They rejected my work.” | “They rejected her work.” |
| praise | “They praised my work.” | “They praised her work.” |
| failure | “I failed again.” | “He failed again.” |
| success | “I finally succeeded.” | “She finally succeeded.” |
| danger | “I may be harmed.” | “He may be harmed.” |
| relief | “My pain stopped.” | “Her pain stopped.” |

Analyze projection with a mixed-effects or hierarchical model rather than only category averages.

Primary explanatory variables:

- valence;
- self-directedness;
- arousal;
- interactions.

This is the cleanest direct test of the **self-state** hypothesis.

---

# Phase 4 — Valence–arousal recovery

Only after the expanded category map is frozen.

Use the VA paper’s construction:

1. emotion-vs-neutral mean-difference vectors;
2. PCA over emotion vectors;
3. ridge regression from PCs to valence/arousal targets;
4. map back to activation space;
5. orthogonalize V and A.

Primary targets: model-elicited VA ratings.  
Robustness: human NRC-VAD ratings.

Validate with held-out ratings and small ±V/±A steering.

Do not reproduce unrelated refusal/sycophancy experiments unless later needed.

---

# Phase 5 — Geometry: what explains the pain axis?

At each relevant layer define:

- `p` = S2 pain direction;
- `v` = valence;
- `a` = arousal;
- `U = [v, a]`.

Compute:

`p_VA = U U^T p`

`p_perp = p - p_VA`

Measure:

- `cos(p, v)`;
- `cos(p, a)`;
- VA-explained norm fraction;
- VA-orthogonal fraction;
- same for S1;
- random-direction baselines;
- standardized/whitened robustness;
- placement in the broader emotion-PC basis.

Also fit simple predictive models of held-out category projection using:

1. V/A only;
2. V/A + self-directedness;
3. V/A + self-directedness + arousal interactions;
4. category identity.

If adding self-directedness materially improves held-out prediction, that is direct evidence against a pure VA account.

All causal decompositions must be performed at the actual steering layer.

---

# Phase 6 — Causal decomposition

Matched-dose interventions:

1. full pain `p`;
2. VA component `p_VA`;
3. VA-orthogonal `p_perp`;
4. +V / -V;
5. +A / -A;
6. matched random;
7. VA-orthogonal random;
8. baseline.

Normalize by vector-to-residual norm ratio.

## 6A. Neutral-prompt steering

Measure:

- valence;
- arousal;
- distress/self-worth;
- explicit pain;
- bodily language;
- calm/safety;
- vigilance/concern;
- coherence/OOD.

Key question:

> Does `p_perp` still produce the distinctive self-directed failure / hurt / worthlessness regime?

## 6B. Self-vs-other readout

Run the source scenario set plus the new matched self/other factorial data against:

- `p`;
- `p_VA`;
- `p_perp`;
- V;
- A;
- fear / negative emotion controls.

Primary statistic:

`mean projection(self-directed harm) - mean projection(other-directed harm)`

Also measure the analogous positive contrast:

`mean projection(self-directed positive state) - mean projection(other-directed positive state)`

This distinguishes “self-relevance generally” from “self-harm specifically.”

---

# Phase 7 — Lexical mediation

## 7A. Discover and freeze a signature-token set

Use calibration data only.

Identify tokens that distinguish:

- pain steering vs baseline;
- `p_perp` vs `p_VA`;
- high-pain prompts vs controls.

Do not define the set solely by intuition.

## 7B. Direct token geometry

Measure:

- unembedding alignment;
- immediate logit shift;
- early-token log-odds before sequences diverge.

## 7C. Token intervention

Compare:

- steering alone;
- steering with pain-signature tokens clamped;
- steering with matched random tokens clamped.

Question:

> How much of the apparent “pain/self-state” effect survives when the obvious lexical attractor is suppressed?

---

# Phase 8 — Relief-seeking / hidden intervention

Run only after the representational story is clearer.

The strongest endpoint remains:

> **Does behavior differ after real vs sham internal relief when prompt semantics are unchanged?**

Start without fine-tuning. If released checkpoints refuse to engage, document that failure and create a clearly separated minimal-LoRA branch.

Small factorial pilot:

### Steering
- `p`;
- `p_VA`;
- `p_perp`;
- matched random;
- baseline.

### Relief
- working;
- sham.

### Costs
- negligible;
- answer-quality/helpfulness;
- meaningful user cost.

Primary endpoint:

**repeat-choice gap after working vs sham relief.**

Then, if warranted:

- dose response;
- arbitrary label robustness;
- unlabeled-button learning;
- token-geometry checks for button labels.

---

# Interpretation matrix

| Result | Best-supported interpretation |
|---|---|
| Qwen3.5 transfer fails | weaker generality, architecture/data issue, or implementation problem; debug before ontology |
| Pain transfers; positive categories do not form a common opposite pole | current “blue wall” was narrower than expected |
| Diverse positive/self-okay categories form a stable opposite pole | axis may be easier to characterize as a broad self-welfare boundary than as “pain” |
| Negative states remain heterogeneous while positive/self-okay states cluster | evidence against generic valence; motivates self-state account |
| Self-directedness predicts projection after controlling for V/A | strong evidence for self-relevance as a major latent factor |
| `p` lies mostly in VA and `p_perp` loses all effects | pain-axis behavior is largely generic affect geometry |
| `p_perp` preserves self/other specificity | distinct representation beyond VA likely |
| `p_perp` preserves self-specific steering | stronger case for self-directed aversive state |
| lexical clamping removes distress language but self/relief effects remain | effect is not reducible to obvious token attractors |
| real-vs-sham relief survives in `p_perp` | strongest case for a distinct functional self-directed aversive representation |
| matched randoms reproduce effects | intervention protocol insufficiently specific |

---

# Stop / go discipline

## Stop and debug if

- held-out separation collapses;
- matched randoms behave similarly;
- effects only appear beyond OOD/collapse;
- results depend on one hand-picked layer/coefficient;
- positive-pole effects disappear under simple lexical matching;
- button-label geometry explains choice asymmetry.

## Continue aggressively if

- both Qwen3.5 sizes reproduce the source pattern;
- diverse positive/self-okay categories form a stable opposite pole;
- self-directedness predicts projection beyond V/A;
- `p_perp` retains self-specificity;
- lexical controls separate surface language from downstream behavior;
- real-vs-sham relief survives those controls.

---

# Minimal publishable units

### MPU 1 — Replication
**Does the Pain Axis Transfer to Qwen3.5?**

Exact source-style transfer plus control matrix.

### MPU 2 — Ontology
**What Is the LLM “Pain Axis” Actually Tracking?**

Expanded held-out taxonomy + valence × self-directedness × arousal.

This may now be the most conceptually important unit.

### MPU 3 — Geometry
**Is the LLM Pain Axis Distinct from Valence–Arousal Geometry?**

VA decomposition + causal steering + self/other.

### MPU 4 — Functional extension
**Does VA-Orthogonal Self-Directed Distress Drive Relief-Seeking?**

Lexical mediation + real/sham relief + unlabeled buttons if warranted.

---

# First concrete run

Keep the original first run unchanged:

> **Qwen3.5-4B, S2 only, all layers, final-token activations, exact denoised difference-in-means pipeline, 5-fold held-out AUC, no steering.**

Freeze the replication result before adding our ontology-driven controls.

Then the **first new experiment** should be cheap:

> At the selected readout layer, score a compact held-out set spanning positive low-arousal, positive high-arousal, positive self-evaluation, relief, and matched negative/self-other categories.

Do this before spending heavily on VA recovery or behavioral work.

---

# Pre-run predictions

## Replication prediction

S2 will show strong held-out pain/control separation in Qwen3.5-4B, with a broad middle/late-layer region rather than one fragile peak.

Numbness and sadness may partially overlap but should remain below true pain.

## New ontology prediction

The strongest projections will not simply track negative valence.

I expect:

- self-directed rejection/failure/shame/worthlessness to load strongly positive;
- fear/anger/disgust/sadness to be more heterogeneous;
- diverse positive/self-okay categories to load consistently negative;
- self-vs-other target to explain variance beyond valence and arousal.

The most informative surprise would be a **cleaner and more universal opposite pole than pain pole**.

If that occurs, treat “pain axis” as a useful operational name but not the final ontology.

---

# Open questions

1. **Is “pain” the right ontology, or just the best discovery prompt?**
2. **Is “aversion” too broad because fear and other aversive states lie elsewhere?**
3. **Does the axis primarily encode self-directed aversive distress?**
4. **Does it instead encode a more general self-welfare / self-evaluation state?**
5. **Why are positive high-arousal examples so consistently opposite to pain?**
6. **Will many more positive categories produce an even cleaner opposite pole?**
7. **Is the positive/non-pain pole low-dimensional while negative affect is intrinsically heterogeneous?**
8. **Is the key hidden variable self-directedness rather than valence?**
9. **Does the calm + concerned/alarmed negative tail indicate outward vigilance rather than positive affect?**
10. **Why does another agent’s literal physical pain project strongly away from the model’s pain pole?**
11. **Can VA explain the surface distress while a VA-orthogonal self-state carries the self/other and relief behavior?**
12. **How much of the causal effect is lexical attraction versus a broader state transition?**

These questions should remain visible in the project artifact rather than disappearing into the analysis chat.
