"""
Model A framing (DistilBERT) behind the corpus's "unevaluated_models" list.

DistilBERT was trained and measured on the old summary corpus only. On a corpus
that lists "framing_bert" as unevaluated, it returns None per outlet instead of a
score, so the UI cannot show an unmeasured number as a style match.
"""
from typing import Dict, List, Optional

MODEL_NAME = "framing_bert"


def neural_framing(
    texts_by_media: Dict[str, List[str]], bert, unevaluated: List[str]
) -> Dict[str, Optional[float]]:
    """Per-outlet DistilBERT style score, or None when the model is unevaluated here."""
    result: Dict[str, Optional[float]] = {}
    for media, texts in texts_by_media.items():
        if not texts:
            continue
        if MODEL_NAME in unevaluated:
            result[media] = None
            continue
        prediction = bert.predict_source_style(" ".join(texts)[:512])
        result[media] = prediction.get(media, 0.0)
    return result


def style_match_label(score: Optional[float]) -> str:
    """Sidebar label for one outlet's DistilBERT score."""
    if score is None:
        return "Not evaluated on this corpus"
    return f"Style Match: {score:.1%}"
