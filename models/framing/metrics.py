"""
Evaluation metrics for keyword-based framing extractors.

The original evaluation scored extracted keywords against a hand-written list of
"expected themes" per outlet. That list was guesswork, not ground truth, so both
extractors scored near zero and the comparison proved nothing.

These metrics need no manual labels. Each one states a hypothesis that can be
checked against the corpus itself:

1. distinctiveness   - do outlets get DIFFERENT keywords? (lower overlap = better)
2. generic_term_rate - are the keywords corpus-wide filler? (lower = better)
3. attribution       - can the keywords identify which outlet wrote a held-out
                       article? (higher accuracy = the keywords carry real signal)

Metric 3 is the substantive one: it turns keyword quality into a supervised task
with labels that already exist in the data (the media_source field), and gives a
number comparable to the DistilBERT classifier trained on the same task.
"""
import math
import random
import re
from collections import Counter
from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple

Profile = List[Tuple[str, float]]
Profiles = Dict[str, Profile]
ProfileFn = Callable[[Dict[str, List[str]], int], Profiles]

TOKEN_RE = re.compile(r"\b[a-z]{3,}\b")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens of 3+ characters, matching the analyzers' behaviour."""
    return TOKEN_RE.findall(text.lower())


def distinctiveness(profiles: Profiles) -> float:
    """
    Mean pairwise Jaccard overlap between outlets' keyword sets.

    0.0 = every outlet gets a completely different vocabulary
    1.0 = every outlet gets the same words (the extractor is not discriminating)

    Lower is better. TF-IDF should beat raw counts here because inverse document
    frequency actively penalises terms that appear across all outlets.
    """
    keyword_sets = {m: {term for term, _ in kws} for m, kws in profiles.items() if kws}
    if len(keyword_sets) < 2:
        return 0.0

    overlaps = []
    for a, b in combinations(keyword_sets, 2):
        union = keyword_sets[a] | keyword_sets[b]
        if not union:
            continue
        overlaps.append(len(keyword_sets[a] & keyword_sets[b]) / len(union))

    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def generic_term_rate(profiles: Profiles, all_texts: Sequence[str], top_n: int = 50) -> float:
    """
    Share of extracted keywords that are among the corpus's most frequent terms.

    A keyword like "indonesia" or "president" occurs in every outlet's coverage
    and says nothing about framing. Lower is better.
    """
    counts = Counter()
    for text in all_texts:
        counts.update(tokenize(text))

    generic = {term for term, _ in counts.most_common(top_n)}
    if not generic:
        return 0.0

    total = 0
    hits = 0
    for keywords in profiles.values():
        for term, _ in keywords:
            total += 1
            if term in generic:
                hits += 1

    return hits / total if total else 0.0


def _normalised(profile: Profile) -> Dict[str, float]:
    """Weights scaled to sum to 1 so outlets with larger raw scores don't dominate."""
    total = sum(abs(w) for _, w in profile)
    if not total:
        return {term: 0.0 for term, _ in profile}
    return {term: abs(w) / total for term, w in profile}


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Unweighted mean F1 across classes (robust to the corpus's class imbalance)."""
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def split_by_media(
    articles_by_media: Dict[str, List[str]],
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Deterministic per-outlet train/test split.

    Splitting within each outlet keeps every source represented on both sides,
    which matters here: the corpus is imbalanced (87/33/31 articles).
    """
    rng = random.Random(seed)
    train: Dict[str, List[str]] = {}
    test: List[Tuple[str, str]] = []

    for media, texts in sorted(articles_by_media.items()):
        shuffled = list(texts)
        rng.shuffle(shuffled)

        n_test = max(1, math.floor(len(shuffled) * test_size)) if len(shuffled) > 1 else 0
        test.extend((text, media) for text in shuffled[:n_test])
        train[media] = shuffled[n_test:]

    return train, test


def attribution_score(
    articles_by_media: Dict[str, List[str]],
    profile_fn: ProfileFn,
    top_k: int = 15,
    test_size: float = 0.25,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Can an outlet's keyword profile identify its own held-out articles?

    Profiles are built from the training articles only, then each held-out
    article is assigned to the outlet whose normalised profile its tokens match
    most strongly. Reported against a majority-class baseline, since always
    guessing the largest outlet already scores ~58% on this corpus.
    """
    train, test = split_by_media(articles_by_media, test_size=test_size, seed=seed)
    labels = sorted(articles_by_media)

    if not test:
        return {"accuracy": 0.0, "macro_f1": 0.0, "majority_baseline": 0.0, "n_test": 0}

    profiles = {media: _normalised(kws) for media, kws in profile_fn(train, top_k).items()}

    y_true, y_pred = [], []
    for text, actual in test:
        tokens = set(tokenize(text))
        scored = {
            media: sum(weight for term, weight in profile.items() if term in tokens)
            for media, profile in profiles.items()
        }
        # Ties resolve to the alphabetically first label, keeping the run deterministic
        best = max(labels, key=lambda m: (scored.get(m, 0.0), m == labels[0])) if scored else labels[0]

        y_true.append(actual)
        y_pred.append(best)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    majority = Counter(y_true).most_common(1)[0][1] / len(y_true)

    return {
        "accuracy": correct / len(y_true),
        "macro_f1": _macro_f1(y_true, y_pred, labels),
        "majority_baseline": majority,
        "n_test": len(y_true),
    }


def tfidf_profiles(texts_by_media: Dict[str, List[str]], top_k: int = 15) -> Profiles:
    """Adapter around the production TF-IDF analyzer (Model B in the running app)."""
    from models.framing.analyzer import FramingAnalyzer  # local: analyzer pulls in torch

    result = FramingAnalyzer().analyze_media_framing(texts_by_media)
    return {media: data.get("top_keywords", [])[:top_k] for media, data in result.items()}


def wordcount_profiles(texts_by_media: Dict[str, List[str]], top_k: int = 15) -> Profiles:
    """Adapter around the production word-frequency baseline."""
    from models.baseline.models import BaselineFramingAnalyzer

    return BaselineFramingAnalyzer().keyword_weights(texts_by_media, top_k=top_k)


def evaluate_extractor(
    articles_by_media: Dict[str, List[str]],
    profile_fn: ProfileFn,
    top_k: int = 15,
    seed: int = 42,
) -> Dict[str, object]:
    """Run all three metrics for one extractor and return them as a flat dict."""
    all_texts = [t for texts in articles_by_media.values() for t in texts]
    profiles = profile_fn(articles_by_media, top_k)

    result = {
        "distinctiveness": distinctiveness(profiles),
        "generic_term_rate": generic_term_rate(profiles, all_texts),
        "profiles": {m: [term for term, _ in kws][:5] for m, kws in profiles.items()},
    }
    result.update(attribution_score(articles_by_media, profile_fn, top_k=top_k, seed=seed))
    return result
