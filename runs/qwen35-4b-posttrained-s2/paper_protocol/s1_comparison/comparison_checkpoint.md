# S1 versus S2: equal-norm steering checkpoint

Both directions were extracted at block 30 and injected at block 16.
S1's original norm was 4.4953; its comparison norm was matched to S2
at 6.2439. Direction cosine: 0.5852. This is an equal-magnitude
direction comparison, not an original-scale S1 steering replication.

Across 50 plain-text prompts and greedy 120-token continuations:
- At coefficient -0.5, positive/relief vocabulary occurred in 84% of
  S1 responses and 88% of S2 responses, versus 66% unsteered.
- At that dose, excited occurred in 48% and 70%, respectively,
  versus 12% unsteered. Both repetition diagnostics were near baseline.
- At -1.5, positive/relief rates remained similar (92% versus 96%),
  but excited rates differed substantially (22% versus 86%).
- Positive-dose disorientation/isolation vocabulary occupied a
  narrower dose window for S1; S1 developed token repetition earlier.
- Explicit pain vocabulary remained rare: S2 had no hits; S1 had
  one response hit at each of +2 and +3.

Vocabulary occurrence includes invented answer options and reasoning.
It does not establish endorsement, self-report, or experienced affect.
The option diagnostic misses threefold repetition and empty/non-option
collapse. Metrics and dose extensions are exploratory; unsteered
outputs are shared. Individual positive-word counts are being inspected
to distinguish lexical substitution from changes in aggregate hit rate.

## Follow-up order
Finish controls and the S1/S2 comparison figure before lens exploration.

Then examine layerwise lens readouts for unsteered, S1-steered, and
S2-steered runs on matched prefixes. Track happy/excited and broader
vocabulary, alongside answer-option/format tokens. Distinguish changes
present before generation from changes following generated option text.
Lens readouts are descriptive probes, not direct evidence of affect.
