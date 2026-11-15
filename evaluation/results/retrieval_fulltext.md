# Retrieval evaluation

12 questions scored against gold built from event membership (see Limits). Recall@5 is the share of a question's relevant articles in the top 5; MRR@10 is the reciprocal rank of the first relevant article in the top 10. Intervals are 95% bootstrap intervals over questions.

| System | Recall@5 | MRR@10 | Questions |
|---|---|---|---|
| Dense (current, cosine over chunk embeddings) | 39.4% [32.5%, 46.6%] | 95.8% [87.5%, 100.0%] | 12 |
| TF-IDF baseline | 39.2% [33.6%, 44.7%] | 100.0% [100.0%, 100.0%] | 12 |
| Hybrid RRF (dense + TF-IDF, no model) | 39.4% [33.9%, 45.1%] | 100.0% [100.0%, 100.0%] | 12 |
| Cross-encoder rerank of dense candidates | 40.3% [35.0%, 46.3%] | 100.0% [100.0%, 100.0%] | 12 |
| Cross-encoder rerank of dense + TF-IDF candidates | 40.2% [34.8%, 46.5%] | 100.0% [100.0%, 100.0%] | 12 |
| Dense minus TF-IDF (paired) | +0.3 pts [-6.3, +7.8] | -4.2 pts [-12.5, +0.0] | 12 |
| Hybrid RRF (dense + TF-IDF, no model) minus dense (paired) | +0.0 pts [-4.2, +3.5] | +4.2 pts [+0.0, +12.5] | 12 |
| Cross-encoder rerank of dense candidates minus dense (paired) | +0.9 pts [-5.2, +6.9] | +4.2 pts [+0.0, +12.5] | 12 |
| Cross-encoder rerank of dense + TF-IDF candidates minus dense (paired) | +0.7 pts [-5.5, +7.1] | +4.2 pts [+0.0, +12.5] | 12 |

Best possible Recall@5 with this gold: 48.8%, because some questions have more than 5 relevant articles.

## Refusal gate (dense only; TF-IDF has no gate)

A question is refused when its best chunk scores below the threshold.

| Group | Refused at 0.48 (default) | Refused at 0.55 (strict) |
|---|---|---|
| In-corpus, held out (refusal is a false refusal) | 0/12 | 0/12 |
| Out-of-corpus, calibration set (the threshold was tuned on these) | 8/8 | 8/8 |
| Out-of-corpus, held out | 7/12 | 9/12 |
| of which near-domain (elections outside Indonesia) | 1/6 | 3/6 |

## Keep rule (pre-registered)

A system is kept when its paired Recall@5 gain over dense is at least 5 pts, its MRR@10 falls by no more than 2 pts and its p95 is at most 1000 ms. It is called better than dense only when the Recall@5 interval excludes 0. A gate is useful when it refuses at least 4 near-domain held-out questions with at most 1 in-corpus held-out false refusal.

- Hybrid RRF (dense + TF-IDF, no model): drop (Recall@5 gain under 5 pts). Recall@5 +0.0 pts [-4.2, +3.5], MRR@10 +4.2 pts [+0.0, +12.5], p95 220 ms; not shown better than dense.
- Cross-encoder rerank of dense candidates: drop (Recall@5 gain under 5 pts; p95 over 1000 ms). Recall@5 +0.9 pts [-5.2, +6.9], MRR@10 +4.2 pts [+0.0, +12.5], p95 10389 ms; not shown better than dense.
- Cross-encoder rerank of dense + TF-IDF candidates: drop (Recall@5 gain under 5 pts; p95 over 1000 ms). Recall@5 +0.7 pts [-5.5, +7.1], MRR@10 +4.2 pts [+0.0, +12.5], p95 14869 ms; not shown better than dense.

## Latency per retrieval call

| System | p50 | p95 | Calls |
|---|---|---|---|
| Dense (current, cosine over chunk embeddings) | 164 ms | 216 ms | 96 |
| TF-IDF baseline | 4 ms | 5 ms | 96 |
| Hybrid RRF (dense + TF-IDF, no model) | 169 ms | 220 ms | 96 |
| Cross-encoder rerank of dense candidates | 8116 ms | 10389 ms | 96 |
| Cross-encoder rerank of dense + TF-IDF candidates | 10781 ms | 14869 ms | 96 |

## Limits

- The corpus holds 63 full-text news articles (median 309 words).
- Gold is constructed, not judged: a question about an event counts every news article of that event as relevant and every other article as not relevant. It can be too broad for a narrow question and misses articles from another event that answer it.
- The owner checked 10 (question, article) pairs (5 pairs the gold calls relevant and 5 it calls not relevant, drawn at random with seed 7, judged blind on 2026-11-06) and agreed with the gold on 7. Disagreements: f02 https://en.antaranews.com/news/311403/mk-dismisses-challenges-to-prabowo-gibrans-election-win (gold not relevant, owner relevant); f05 https://en.antaranews.com/news/305952/ganjar-mahfud-pair-wins-at-polling-station-where-ganjar-voted (gold relevant, owner not relevant); f06 https://en.antaranews.com/news/305883/former-president-megawati-casts-her-vote-at-tps-no-53-in-jakarta (gold relevant, owner not relevant).
- The calibration questions, in-corpus and out-of-corpus, are the data the refusal threshold was tuned on; only the held-out groups test it.
- Articles in a top 10 that were never judged: dense 0, tfidf 0, rrf 0, rerank 0, rerank_hybrid 0. They count as not relevant.
- Latency was measured on one laptop against a local PostgreSQL; the dense figure includes embedding the query and fetching every chunk from the database.

## Setup

Commit `82c294a`, corpus `indonesia-election-2024-fulltext@69a10833e719`, embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k 8 per outlet, generated 2026-11-06T21:31:29.
