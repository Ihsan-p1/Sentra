# Sentra Evaluation Report

Generated: 2026-07-26T20:26:06  
Corpus: 154 articles - {'abc_news': 31, 'antaranews': 89, 'tempo': 34}

## Hallucination detection

16 labelled claim/source pairs.

| Metric | Model A (Logistic Regression) | Model B (Keyword overlap) |
|---|---|---|
| Accuracy | 75.00% | 56.25% |
| Precision | 66.67% | 53.33% |
| Recall | 100.00% | 100.00% |
| F1 | 80.00% | 69.57% |

> Trained on synthetic data; treat as proof-of-concept, not a benchmark.

## Confidence scoring

| Metric | Model A (Random Forest) | Model B (Heuristic) |
|---|---|---|
| In-range accuracy | 88.89% | 44.44% |
| Mean absolute error | 0.089 | 0.180 |

## Framing keyword extractors (top-15)

| Metric | TF-IDF | WordCount baseline |
|---|---|---|
| Distinctiveness (lower better) | 17.92% | 26.81% |
| Generic-term rate (lower better) | 76.67% | 86.67% |
| Source attribution accuracy | 51.35% | 45.95% |
| Source attribution macro-F1 | 46.74% | 43.22% |
| Majority-class baseline | 59.46% | 59.46% |

> TF-IDF produces more distinctive, less generic keyword sets, but neither extractor beats the majority-class baseline at source attribution. Keyword frequency alone does not identify the outlet.
