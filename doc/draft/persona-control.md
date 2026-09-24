# Persona / Roleplay Control

## Motivation

A plausible confound for the pain axis is that it partly tracks a **non-default-assistant / suffering-persona state** rather than pain specifically.

Relevant result: `mild-rgb/mech-interp-on-randomly-emergent-personas` finds a shared linear component separating default-assistant behavior from many distinct personas. Steering this component can causally induce roleplay, and raw vs debiased persona directions produce qualitatively different behavior.

## Minimal control

On the same model used for the pain-axis repro:

1. Construct a **default-assistant ↔ persona** direction from diverse examples.

   * Positive: distinct first-person identities/registers across many persona families.
   * Negative: ordinary helpful-assistant responses.
   * Avoid making this merely “sad persona vs assistant.”

2. Measure cosine / projection between:

   * `d_pain`
   * `d_persona`

3. Residualize pain against persona:

```text
d_pain_resid =
    d_pain
    - proj_d_persona(d_pain)
```

Normalize afterward if existing analyses assume unit vectors.

4. Re-run the existing pain-axis analyses with `d_pain_resid`:

   * category projections / means
   * S2 classification / AUC
   * category ordering
   * steering, if cheap enough

## Interpretation

* **Pain results mostly survive:** generic persona/roleplay is unlikely to explain the axis.
* **Social/moral pain collapses but physical pain behaves differently:** original axis likely mixed pain with a suffering-character / role-state feature.
* **Most results collapse:** strong evidence the original “pain” direction was substantially a default-assistant ↔ persona direction.

Do not treat cosine alone as decisive. The important test is whether removing the persona component changes the behavioral and classification results.
