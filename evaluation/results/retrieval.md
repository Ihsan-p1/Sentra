# Retrieval evaluation

49 questions scored against owner-judged gold articles. Recall@5 is the share of a question's relevant articles in the top 5; MRR@10 is the reciprocal rank of the first relevant article in the top 10. Intervals are 95% bootstrap intervals over questions.

| System | Recall@5 | MRR@10 | Questions |
|---|---|---|---|
| Dense (current, cosine over chunk embeddings) | 55.6% [46.4%, 64.9%] | 80.5% [71.1%, 89.4%] | 49 |
| TF-IDF baseline | 65.3% [55.3%, 74.8%] | 79.3% [69.0%, 88.6%] | 49 |
| Hybrid RRF (dense + TF-IDF, no model) | 68.3% [59.2%, 76.9%] | 82.9% [73.6%, 91.1%] | 49 |
| Cross-encoder rerank of dense candidates | 74.3% [66.8%, 81.6%] | 92.3% [85.2%, 98.0%] | 49 |
| Cross-encoder rerank of dense + TF-IDF candidates | 72.9% [64.8%, 80.6%] | 90.6% [83.2%, 96.9%] | 49 |
| Dense minus TF-IDF (paired) | -9.7 pts [-21.1, +1.7] | +1.2 pts [-8.3, +11.5] | 49 |
| Hybrid RRF (dense + TF-IDF, no model) minus dense (paired) | +12.7 pts [+3.1, +22.5] | +2.4 pts [-6.9, +11.6] | 49 |
| Cross-encoder rerank of dense candidates minus dense (paired) | +18.7 pts [+10.1, +28.1] | +11.8 pts [+3.3, +20.9] | 49 |
| Cross-encoder rerank of dense + TF-IDF candidates minus dense (paired) | +17.2 pts [+9.2, +26.1] | +10.1 pts [+2.3, +18.4] | 49 |

Best possible Recall@5 with this gold: 88.6%, because some questions have more than 5 relevant articles.

No candidate was judged relevant for c21; those questions are left out of both averages.

## Refusal gate (dense only; TF-IDF has no gate)

A question is refused when its best chunk scores below the threshold.

| Group | Refused at 0.35 (default) | Refused at 0.42 (strict) |
|---|---|---|
| In-corpus, calibration set (the threshold was tuned on these; refusal is a false refusal) | 2/25 | 2/25 |
| In-corpus, held out (refusal is a false refusal) | 0/25 | 0/25 |
| Out-of-corpus, calibration set (the threshold was tuned on these) | 8/8 | 8/8 |
| Out-of-corpus, held out | 4/12 | 4/12 |
| of which near-domain (elections outside Indonesia) | 0/6 | 0/6 |

## Reranker score as refusal signal

The best reranker score per question takes the place of the best cosine. Each threshold was picked on the calibration groups only (fewest errors, ties to fewer leaks); the held-out groups test it. Reranker scores are raw logits.

- Cross-encoder rerank of dense candidates: threshold -3.709, picked on the calibration groups only.
- Cross-encoder rerank of dense + TF-IDF candidates: threshold -3.850, picked on the calibration groups only.

| Group | Dense, cosine at 0.35 | Cross-encoder rerank of dense candidates | Cross-encoder rerank of dense + TF-IDF candidates |
|---|---|---|---|
| In-corpus, calibration set (the threshold was tuned on these; refusal is a false refusal) | 2/25 | 2/25 | 1/25 |
| In-corpus, held out (refusal is a false refusal) | 0/25 | 1/25 | 1/25 |
| Out-of-corpus, calibration set (the threshold was tuned on these) | 8/8 | 8/8 | 8/8 |
| Out-of-corpus, held out | 4/12 | 7/12 | 5/12 |
| of which near-domain (elections outside Indonesia) | 0/6 | 1/6 | 1/6 |

## Latency per retrieval call

| System | p50 | p95 | Calls |
|---|---|---|---|
| Dense (current, cosine over chunk embeddings) | 76 ms | 169 ms | 210 |
| TF-IDF baseline | 3 ms | 4 ms | 210 |
| Hybrid RRF (dense + TF-IDF, no model) | 78 ms | 172 ms | 210 |
| Cross-encoder rerank of dense candidates | 1416 ms | 5246 ms | 210 |
| Cross-encoder rerank of dense + TF-IDF candidates | 1469 ms | 6798 ms | 210 |

## Limits

- The corpus holds 154 article summaries (median 97 words), not full text.
- Gold comes from a single judge: the owner judged every candidate, with an AI tool helping to read or summarise articles.
- Gold is pooled from the top 10 of both systems; a relevant article neither system ranked in its top 10 was never judged and counts as not relevant.
- 109 articles first ranked by the phase 3 systems were judged later by the same owner; every system is scored on the same extended gold.
- The calibration questions, in-corpus and out-of-corpus, are the data the refusal threshold was tuned on; only the held-out groups test it.
- Articles in a top 10 that were never judged: dense 0, tfidf 0, rrf 0, rerank 0, rerank_hybrid 0. They count as not relevant.
- Latency was measured on one laptop against a local PostgreSQL; the dense figure includes embedding the query and fetching every chunk from the database.

## Setup

Commit `a518378`, corpus `indonesia-election-2024@cecce290d0ff`, embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k 8 per outlet, generated 2026-10-22T14:56:12.
