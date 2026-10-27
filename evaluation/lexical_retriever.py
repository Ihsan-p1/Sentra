"""
TF-IDF retrieval baseline (eval phase 2).

The comparison system for the dense retriever, under Sentra's A/B rule, and
the second source of candidates when the gold pool is built, so the pool does
not lean only towards the system being measured. Ranks whole articles
(title + content) and reads the corpus file, so it needs no database.
"""
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, articles: List[Dict]):
        self.urls = [a["url"] for a in articles]
        self.vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            [f"{a['title']} {a['content']}" for a in articles]
        )

    def rank(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """(url, score) pairs, best first; articles sharing no term with the query are dropped."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(self.urls[i], float(scores[i])) for i in order if scores[i] > 0]
