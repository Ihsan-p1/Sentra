"""
Embedding Generation Module
Uses sentence-transformers for generating text embeddings.
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('..')
from config.settings import settings


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.

    Configured model: settings.EMBEDDING_MODEL (384-dim multilingual MiniLM).

    This is a singleton so the weights load once. That made it silently ignore
    its own argument: `EmbeddingGenerator("some/other-model")` returned the
    already-loaded instance and handed back vectors from a *different* model than
    the caller asked for. That is precisely the failure that put mismatched
    vectors in the database, so requesting a different model now raises instead.
    """

    _instance = None
    _model = None

    def __new__(cls, model_name: str = None):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        requested = model_name or settings.EMBEDDING_MODEL

        if self._model is not None:
            if requested != self.model_name:
                raise ValueError(
                    f"EmbeddingGenerator is already loaded with {self.model_name!r}; "
                    f"cannot serve {requested!r} from the same process. Vectors from "
                    f"two different models are not comparable - change "
                    f"EMBEDDING_MODEL and re-ingest instead."
                )
            return

        self.model_name = requested
        print(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        self._dimension = self._read_dimension()
        print(f"Embedding model loaded (dim={self._dimension})")

    def _read_dimension(self) -> int:
        """
        Embedding width, via whichever accessor this sentence-transformers exposes.

        `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension`;
        calling the old name emits a FutureWarning. The property below is read once
        per chunk during verification, so the old call produced a wall of warnings
        on every startup and will break outright on the next major release.
        """
        for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            accessor = getattr(self._model, attr, None)
            if callable(accessor):
                return int(accessor())
        # Last resort: measure it.
        return int(self._model.encode("dimension probe", convert_to_numpy=True).shape[0])

    @property
    def model(self):
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding width, resolved once at load time."""
        return self._dimension
    
    def generate(
        self, 
        texts: List[str], 
        show_progress: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            batch_size: Batch size for encoding
        
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
        
        Returns:
            numpy array of shape (dimension,)
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def compute_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.
        """
        from sentence_transformers import util
        
        emb1 = self.generate_single(text1)
        emb2 = self.generate_single(text2)
        
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity
    
    def compute_similarities(
        self, 
        query: str, 
        documents: List[str]
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple documents.
        
        Returns:
            numpy array of similarity scores
        """
        from sentence_transformers import util
        
        query_emb = self.generate_single(query)
        doc_embs = self.generate(documents, show_progress=False)
        
        similarities = util.cos_sim(query_emb, doc_embs)[0].numpy()
        return similarities


# Global instance for convenience
_embedder = None

def get_embedder() -> EmbeddingGenerator:
    """Get or create global embedder instance"""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingGenerator()
    return _embedder
