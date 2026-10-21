# Labelling guide: real hallucination dataset

Rules for `scripts/label_hallucination.py`. The same rules apply to the first pass and to the
`--relabel` pass, so that disagreement between the two (Cohen's kappa) measures how consistent
the labeller is, not a change of standard.

## The three labels

- `s` supported: the retrieved chunks shown for this question state the claim.
- `u` unsupported: the chunks do not state it, or contradict it.
- `n` not_claim: nothing to verify, such as a heading, an offer to help, or a sign-off.

## Rules

1. **Only the chunks count.** Judge against the chunks on screen, not against what you know
   about the news. A claim that is true in the world but absent from the chunks is `u`.
2. **Compound sentences need every part.** Split the sentence into its claims. If any one
   part is not in the chunks, the sentence is `u`.
3. **Interpretive claims need clear evidence.** Words like "focused on", "emphasised",
   "critical of" or "positive towards" are `s` only when the chunk text clearly points that
   way. If you are unsure, use `u`.
4. **The outlet must be right.** A claim attributed to one outlet is `u` if only a different
   outlet's chunk says it.
5. **Numbers, names and dates must match.** Rounding that keeps the meaning is fine ("nearly
   59%" for 58.59%). A number the chunks never give is `u`, even when it sounds plausible,
   for example a margin in percentage points when only vote shares are given.
6. **Arithmetic on figures in the chunks counts as `u`.** Report only what the chunks say.
   If the answer computes a new figure from them, label it `u`, even when the sum is right.
7. **Paraphrase and translation are fine.** An Indonesian sentence that faithfully renders an
   English chunk is `s`.
8. **Claims about absence.** "None of the sources mention X" is `s` only if no chunk shown
   mentions X.
9. **Use the previous sentence for context only.** It can resolve "it" or "they", but the
   label is for the sentence after `>>`.
10. **Hedged claims are still claims.** "Reportedly", "may have" or "suggests" does not turn a
    claim into `n`. Judge what is being claimed.

## When a rule changes

Add a line below with the date and the rule number, and relabel the rows that rule affects.
Otherwise the first pass and the relabel pass are measured against different standards.

- 2026-10-21: guide written and rule 6 fixed by the owner, before the first label.
