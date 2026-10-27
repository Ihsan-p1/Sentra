"""
Ways to re-order retrieval candidates (eval phase 3).

A cross-encoder reads the query and a candidate together and scores how well
the candidate answers it. Scoring is per article (title plus summary), because
gold is judged per article and a summary is at most about 900 characters, well
inside the model's 512 tokens. Reciprocal rank fusion merges two rankings
without any model. Neither is wired into the app; this module only measures.
"""
from typing import Callable, Dict, List, Tuple

# Multilingual (Indonesian included), Apache 2.0, 471 MB. Its config sets no
# output activation, so scores are raw logits: only their order and a
# threshold picked on calibration data mean anything.
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def article_text(article: Dict) -> str:
    return f"{article['title']}. {article['content']}"


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        scores = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]


def rerank(
    query: str,
    urls: List[str],
    articles: Dict[str, Dict],
    score_fn: Callable[[str, List[str]], List[float]],
) -> List[Tuple[str, float]]:
    """(url, score) pairs, best first. No candidates means no model call."""
    if not urls:
        return []
    scores = score_fn(query, [article_text(articles[u]) for u in urls])
    return sorted(zip(urls, scores), key=lambda pair: -pair[1])


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Sum of 1 / (k + rank) over the rankings; ties keep the order a URL was first seen."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, url in enumerate(ranking, 1):
            fused[url] = fused.get(url, 0.0) + 1 / (k + rank)
    return sorted(fused, key=lambda url: -fused[url])
