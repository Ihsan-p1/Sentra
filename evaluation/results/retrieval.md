# Retrieval evaluation

48 questions scored against owner-judged gold articles. Recall@5 is the share of a question's relevant articles in the top 5; MRR@10 is the reciprocal rank of the first relevant article in the top 10. Intervals are 95% bootstrap intervals over questions.

| System | Recall@5 | MRR@10 | Questions |
|---|---|---|---|
| Dense (current, cosine over chunk embeddings) | 59.2% [50.5%, 68.1%] | 82.2% [73.2%, 90.5%] | 48 |
| TF-IDF baseline | 68.5% [58.9%, 77.8%] | 80.9% [71.4%, 89.8%] | 48 |

No candidate was judged relevant for c21, n25; those questions are left out of both averages.

## Refusal gate (dense only; TF-IDF has no gate)

A question is refused when its best chunk scores below the threshold.

| Group | Refused at 0.35 (default) | Refused at 0.42 (strict) |
|---|---|---|
| In-corpus questions (refusal here is a false refusal) | 2/50 | 2/50 |
| Out-of-corpus, calibration set (the threshold was tuned on these) | 8/8 | 8/8 |
| Out-of-corpus, held out | 4/12 | 4/12 |
| of which near-domain (elections outside Indonesia) | 0/6 | 0/6 |

## Latency per retrieval call

| System | p50 | p95 | Calls |
|---|---|---|---|
| Dense (current, cosine over chunk embeddings) | 127 ms | 160 ms | 210 |
| TF-IDF baseline | 3 ms | 4 ms | 210 |

## Limits

- The corpus holds 154 article summaries (median 97 words), not full text.
- Gold comes from a single judge: the owner judged every candidate, with an AI tool helping to read or summarise articles.
- Gold is pooled from the top 10 of both systems; a relevant article neither system ranked in its top 10 was never judged and counts as not relevant.
- The calibration out-of-corpus questions are the data the refusal threshold was tuned on; only the held-out group tests it.
- Latency was measured on one laptop against a local PostgreSQL; the dense figure includes embedding the query and fetching every chunk from the database.

## Setup

Commit `a781466`, corpus `indonesia-election-2024@cecce290d0ff`, embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k 8 per outlet, generated 2026-10-21T23:55:02.
