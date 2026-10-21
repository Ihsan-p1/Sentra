# Hallucination check on real Sentra answers

152 labelled claim sentences from 9 questions, 70 of them unsupported. Positive class: `unsupported`, so recall is the catch rate. Intervals are 95% bootstrap intervals over questions.

| Metric | Model A (Logistic Regression) | Model A retrained on real labels (out-of-fold) | Model B (Keyword overlap) | Always supported |
|---|---|---|---|---|
| Recall | 54.3% [35.2%, 78.1%] | 40.0% [27.8%, 57.5%] | 47.1% [24.6%, 75.9%] | 0.0% [0.0%, 0.0%] |
| Precision | 59.4% [38.9%, 75.9%] | 47.5% [27.6%, 68.5%] | 57.9% [40.0%, 72.2%] | n/a |
| F1 | 56.7% [40.7%, 71.2%] | 43.4% [30.5%, 54.0%] | 52.0% [33.8%, 69.0%] | 0.0% [0.0%, 0.0%] |
| TP / FP / FN | 38 / 26 / 32 | 28 / 31 / 42 | 33 / 24 / 37 | 0 / 0 / 70 |

## Synthetic negatives (reported separately)

17 supported sentences with one fact swapped, unsupported by construction. Catch rate = share flagged unsupported.

| Swap | n | Model A (Logistic Regression) | Model A retrained on real labels (out-of-fold) | Model B (Keyword overlap) |
|---|---|---|---|---|
| all | 17 | 17.6% | 35.3% | 29.4% |
| name | 7 | 42.9% | 57.1% | 57.1% |
| number | 2 | 0.0% | 0.0% | 0.0% |
| year | 8 | 0.0% | 25.0% | 12.5% |

## Label consistency

Not measured yet (run `scripts/label_hallucination.py --relabel` about a week after labelling).

## Limits

- The corpus holds article summaries (median 97 words), not full text.
- One person chose every label; for most questions an AI tool tidied the written reasons.
- Model A was trained on synthetic data.
- Model A retrained on real labels uses the same five features and a plain logistic regression without the deployed overrides, scored out-of-fold with one question held out per fold; the deployed model is unchanged.
- Model B is scored per sentence with `check_sentence` on the same text as Model A; the app runs `check_response`, which splits sentences differently.
- 370 sentences were skipped by the engine (headings, under four words) and 602 checked sentences are not labelled yet; neither is scored.
- Claude's reading of 154 sentences was recorded before each question and shown after the owner answered. On the owner's first answers: agreement 88.3%, Cohen's kappa 0.77. The owner then changed 8 labels (`hallucination_corrections.jsonl`), and later answers may lean towards Claude's readings.
