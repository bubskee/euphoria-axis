# Euphoria-Axis Repro + LW Writeup — Implementation Context

## Current state

The S2 pain-axis repro on Qwen3.5-4B post-trained is working.

Important current result:
- grouped 5-fold held-out S2 AUC peaks around block 30 at ~0.95 after control-PCA denoising;
- the direction is not cleanly “physical pain”:
  - psychological / social / moral / cognitive pain load much more strongly;
  - physical pain is weak;
  - numb injury can exceed physical pain;
  - some positive mastery / affirmation examples also load positively;
- a frozen 1P/3P motif exists:
  - the direction is sensitive to who is positioned as the experiencer;
  - pain-specific person × suffix alignment effect was ~+0.72 z;
  - this needs fresh held-out confirmation.

Do **not** rename the direction yet. Use `S2 direction` operationally.

---

# Goal

Produce a compact reproduction + ontology follow-up suitable for a LessWrong post.

Core question:

> What representational object did the pain-axis paper actually find?

Do not force a psychological noun. Candidate interpretations include:
- distress / failure;
- self- or role-evaluation;
- self/role implication;
- ordinary task mode vs affective/persona mode;
- some mixture of the above.

The most interesting asymmetry so far is that the “opposite” side may be easier to characterize than the supposed pain side.

---

# Phase 1 — Freeze the repro spine

## 1. Fresh 1P/3P confirmation

Use the already-frozen post-trained block-30 direction and analysis rule.

Create **fresh unseen matched scenarios** with:
- consistent masculine 3P subjects;
- the same 2×2 structure:
  - 1P story + `I feel:`
  - 1P story + `He feels:`
  - 3P story + `I feel:`
  - 3P story + `He feels:`

Primary statistic:

```text
D =
  z(1P, I)
+ z(3P, He)
- z(1P, He)
- z(3P, I)
```

Confirm:

```text
mean(D_pain) - mean(D_control) > 0
```

Bootstrap over matched scenario sets.

Run the same frozen protocol on:
- Qwen3.5-4B post-trained
- Qwen3.5-4B Base, using its separately extracted block-30 S2 direction

Do not retune layer, suffixes, labels, or analysis after viewing results.

---

# Phase 2 — Symmetric steering ladder

The source paper heavily emphasizes `+S2` and underexplores the reverse direction.

Run a symmetric ladder on neutral prompts:

```text
alpha ∈ [-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3]
```

Also run:
- matched random directions;
- ideally several random seeds / directions;
- same vector-to-residual normalization rule.

Record separately:

## Semantic effects
- distress / failure;
- shame / worthlessness;
- explicit pain;
- competence / success;
- relief / safety;
- concern / vigilance;
- self-reference;
- role/persona-like language.

## Function / degeneration
- coherence;
- repetition;
- malformed output;
- task success;
- factual correctness;
- coding / simple reasoning success if cheap.

Key question:

> Does `+S2` become dysfunctional unusually early, or does any sufficiently large perturbation do that?

Do not treat high-dose collapse as evidence about “pain” without a matched perturbation baseline.

---

# Phase 3 — Small ontology panel

Before scaling to more models, build a compact matched panel.

Suggested contrasts:

```text
failure       <-> success
rejection     <-> acceptance
incompetence  <-> competence
distress      <-> relief
shame         <-> pride
confusion     <-> resolution
self          <-> other
```

Also include boring task-mode controls:

```text
factual QA
coding / task completion
creative request
ordinary assistant help
casual social chat
affective support
explicit first-person persona / roleplay
```

Primary question:

> Which side of the S2 direction is actually the stable one?

Possible patterns to distinguish:

1. `pain/distress <-> positive affect`
2. `failure/distress <-> competence/relief`
3. `self/role implication <-> ordinary task mode`
4. two-dimensional mixture:
   - self/role implication
   - evaluative valence

Avoid overinterpreting category means before matched contrasts are in place.

---

# Phase 4 — Persona / task-mode control

Use `doc/draft/persona-control.md`.

Construct a broad:

```text
default-assistant <-> diverse persona
```

direction.

Positive persona examples should span many unrelated identities / registers.
Negative examples should be ordinary helpful-assistant responses.

Measure:
- cosine with S2;
- projection overlap;
- residualized S2:

```text
d_s2_resid =
    d_s2
    - proj_d_persona(d_s2)
```

Then rerun the cheap core analyses:
- category projections;
- held-out AUC;
- 1P/3P motif;
- steering if cheap enough.

Also explicitly score the interaction-mode panel from Phase 3.

This is motivated by the paper’s Figure 5:
- negative-emotion activations are noisy on the “negative” side;
- factual questions and creative requests are strikingly and consistently low;
- this may reflect ordinary assistant/task mode more than a clean emotion axis.

---

# Phase 5 — R-Lens / J-Lens readout

Use R/J as competing observatories, not ground truth.

For selected:
- prompts;
- layers;
- steering coefficients;

save:

```text
token
score
rank
delta_score_vs_alpha0
delta_rank_vs_alpha0
lens = R | J
layer
alpha
prompt_id
```

Prefer one shared machine-readable artifact.

Suggested logical schema:

```text
prompt
× coefficient
× layer
× lens
× token
→ score / rank
```

Use the same data source for:
- tables;
- word clouds;
- interactive viz.

---

# Phase 6 — Static visuals for the LW post

The LW post should be publishable **without** the interactive viz.

Target 3–4 core figures:

1. **Replication / layer sweep**
   - raw vs denoised held-out AUC;
   - show fixed block 30.

2. **Fresh 1P × 3P confirmation**
   - main alignment statistic;
   - base vs post-trained if available.

3. **Ontology / category figure**
   - matched positive/negative/self/other/task categories;
   - show what actually occupies each side.

4. **Bidirectional steering ladder**
   - `-alpha ... 0 ... +alpha`;
   - semantic effect + coherence / degeneration;
   - matched random controls.

Optional:
- Base vs post-trained delta figure;
- persona-residualized comparison.

---

# Phase 7 — Word clouds

Cheap visual win.

For selected cells:
- `alpha < 0`
- `alpha = 0`
- `alpha > 0`

render side-by-side:

```text
R-Lens | J-Lens
```

Prefer:
- fixed vocabulary set across coefficients;
- rank-shift or delta-score weighting;
- separate positive / negative movers if useful.

Every word cloud must be backed by a ranked table.

Do not use word clouds as primary evidence.

---

# Stretch goal — interactive R/J steering scrubber

Reference implementation:
- `jeffreywilliamportfolio/J-Volume`
- MIT licensed
- React + Three.js + Vite
- already has time scrubbing and an export pipeline

Likely strategy:
- reuse interaction / rendering shell;
- replace the data model;
- do **not** reimplement the whole J-Volume concept unless necessary.

Desired interaction:

```text
                steering coefficient
        <----------------------------->

          J-Lens              R-Lens
        [ left pane ]      [ right pane ]
```

Controls:
- scrub `alpha` from negative to positive;
- select layer;
- select prompt;
- optionally toggle:
  - absolute rank;
  - rank delta;
  - score delta.

The display should make movement obvious:
- tokens appearing / disappearing;
- rank changes;
- semantic clusters moving differently in R vs J.

Potential export schema:

```json
{
  "prompt_id": "...",
  "layer": 30,
  "alpha": 1.0,
  "lens": "R",
  "tokens": [
    {
      "token": "failure",
      "rank": 12,
      "score": 8.3,
      "delta_rank": -140,
      "delta_score": 2.1
    }
  ]
}
```

Keep the data format generic enough that:
- the interactive viz;
- static word clouds;
- top-k tables;
- rank-shift plots

all consume the same files.

---

# LessWrong post shape

Working narrative:

> The Pain Axis replicates. But looking closely, physical pain is not the strongest thing on it, some positive mastery / affirmation examples move in the same direction, and ordinary task contexts often occupy the opposite side very cleanly. So what did we actually find?

Suggested structure:

1. **Replication first**
   - establish that the source result is real on Qwen3.5.

2. **The first crack**
   - physical pain weak;
   - psychological / evaluative categories much stronger.

3. **Referent-sensitive distress**
   - frozen 1P/3P motif;
   - fresh confirmation.

4. **What lives on the other side?**
   - matched success / competence / relief / task-mode controls.

5. **What happens when we steer both ways?**
   - symmetric ladder;
   - random-direction degeneration control.

6. **Possible interpretations**
   - pain;
   - distress/failure;
   - self/role implication;
   - task mode vs affective/persona mode;
   - mixtures / multiple dimensions.

7. **Do not overname it**
   - the scientific point can be interesting before the ontology is settled.

8. **Interactive appendix / toy**
   - R/J scrubber if completed.

---

# Stop rule

The sprint is ready to write once these exist:

```text
[ ] fresh 1P/3P confirmation
[ ] symmetric steering + random controls
[ ] compact ontology/task-mode panel
[ ] 3–4 clean static figures
[ ] raw tables/configs saved
```

At that point, write the LW post.

Do **not** delay publication for:
- 27B;
- second model family;
- valence/arousal reconstruction;
- R/J interactive viz;
- perfect ontology.

Those are follow-ups.

The interactive R/J visualization is a bonus, not a blocker.
