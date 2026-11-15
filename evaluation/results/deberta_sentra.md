# neutral's DeBERTa on Sentra sentences

Model `deberta-babe-f0s3` served by `neutral`'s `POST /analyze` (onnx, threshold 0.5), scored on 2026-11-13 over corpus `indonesia-election-2024-fulltext@69a10833e719`. Labels are the repository owner's, given without seeing the model's output. Built by `scripts/fulltext/measure_deberta.py`.

| Domain | n | Macro F1 | 95% CI | Best constant |
|---|---|---|---|---|
| BABE fold 0 test, US news, in-domain | 751 | 0.8765 | not reported here | 0.3598 |
| BASIL test, US news, other dataset, no adaptation | 5780 | 0.5656 | not reported here | 0.4868 |
| Sentra full-text corpus | 145 | 0.5167 | [0.4545, 0.6359] | 0.4649 (all neutral) |

Sample: 150 sentences drawn, 145 scored, 5 skipped, 0 not labelled.

| Outlet | n | Owner: biased | Model: biased | Macro F1 | Best constant |
|---|---|---|---|---|---|
| antaranews | 75 | 3 | 0 | 0.4898 | 0.4898 (all neutral) |
| tempo | 70 | 16 | 1 | 0.4978 | 0.4355 (all neutral) |

| | Model: biased | Model: neutral |
|---|---|---|
| Owner: biased | 1 | 18 |
| Owner: neutral | 0 | 126 |

A second reading by Claude, written after the owner finished, matches the owner on 149 of 150 sentences (biased, neutral or skip). It is not used as gold.

## Limits

- One annotator and one pass. No second annotator, so no inter-annotator agreement.
- 145 sentences. The interval resamples whole articles (2,000 draws), because sentences of one article are not independent.
- The sample is stratified: the same number of sentences per outlet, split evenly over each outlet's events. It does not follow the corpus's own proportions.
- Macro F1 counts both classes equally. With few sentences labelled biased, a handful of them moves the score; read it next to the best constant.
- Sentences are the units `/analyze` produced (pysbd). BASIL's figure uses BASIL's own segmentation.
- The owner labelled from a summary of BABE's annotation manual (`evaluation/datasets/DEBERTA_LABELING_GUIDE.md`), not as a trained BABE annotator.
- Loaded wording inside quoted or reported speech counts as biased, as in BABE's manual. A biased label means the sentence prints loaded language, not that an outlet is biased.
