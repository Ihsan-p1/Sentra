"""
RAG Retrieval Module (NumPy implementation)
Performs client-side vector search since pgvector is unavailable.
"""
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('..')
from config.settings import settings
from database.connection import db_manager
from pipeline.embeddings import get_embedder

# Query text is embedded as-is.
#
# A "News about: " prefix was tried here and removed. It appeared to help a lot -
# it lifted in-corpus similarity by ~0.06 and turned overlapping distributions
# into separated ones - but that measurement was taken while the stored
# embeddings came from a DIFFERENT model than the query embeddings (see
# ArticleIngestor.verify_embeddings_match_model). Once the corpus was re-ingested
# with the configured model, the prefix produced identical error rates and
# slightly WORSE distribution separation, so it was pure noise compensating for a
# broken comparison.
#
# Lesson worth keeping: a fix validated against a broken baseline measures the
# breakage, not the fix.
QUERY_PREFIX = ""

# Chunks scoring below this are dropped from the per-source result list before the
# refusal gate ever sees them. It is NOT the refusal threshold - that is
# settings.RETRIEVAL_THRESHOLD (0.35), applied by the engine to the best-scoring
# chunk overall. This is only a floor on what is worth carrying around: at
# top_k=8 per source, an outlet with nothing relevant would otherwise contribute
# eight near-random chunks to the context budget.
MIN_CHUNK_SIMILARITY = 0.15


def expand_query(query: str) -> str:
    """
    Text used for embedding the query.

    Currently identity (see QUERY_PREFIX above). Kept as a seam so any future
    query rewriting has one obvious place to live, and so it can be tested.
    """
    return f"{QUERY_PREFIX}{query.strip()}"


class RAGRetriever:
    """
    Retrieves chunks using NumPy for cosine similarity.
    Scalable enough for <100k chunks.
    """

    def __init__(self, db=None, embedder=None):
        self.db = db or db_manager
        self.embedder = embedder or get_embedder()


    async def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        media_filter: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Main retrieval method.
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        media_sources = media_filter or settings.SUPPORTED_MEDIA

        # 1. Generate query embedding (from the expanded form - see QUERY_PREFIX)
        query_embedding = self.embedder.generate_single(expand_query(query)).reshape(1, -1)
        
        results = {}
        for media in media_sources:
            # 2. Fetch ALL chunks for this media (ID + Embedding + Metadata)
            chunks_data = await self._fetch_media_chunks(media)
            
            if not chunks_data:
                results[media] = []
                continue
            
            # 3. Compute Similarity in Vector Space
            embeddings_matrix = np.array([c['embedding'] for c in chunks_data])
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            
            # 4. Get Top-K Indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # 5. Construct Result List
            media_results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score < MIN_CHUNK_SIMILARITY:
                    continue

                chunk = chunks_data[idx]
                chunk['similarity'] = score
                # Remove raw embedding from output to save bandwidth
                del chunk['embedding'] 
                media_results.append(chunk)
                
            results[media] = media_results
            
        return results
    
    async def _fetch_media_chunks(self, media_source: str) -> List[Dict]:
        """Fetch all chunks for a media source"""
        query = """
            SELECT 
                ac.chunk_text,
                ac.embedding,
                ac.metadata,
                a.title,
                a.url,
                a.published_date
            FROM article_chunks ac
            JOIN articles a ON ac.article_id = a.id
            WHERE LOWER(a.media_source) = LOWER($1)
        """
        
        rows = await self.db.fetch(query, media_source)
        
        # Convert to list of dicts
        return [
            {
                'text': row['chunk_text'],
                'embedding': row['embedding'], # This is now a float list, not vector type
                'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                'title': row['title'],
                'url': row['url'],
                'date': str(row['published_date']),
                'media': media_source
            }
            for row in rows
        ]

    def flatten_results(self, grouped_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Flatten grouped results"""
        all_chunks = []
        for media, chunks in grouped_results.items():
            all_chunks.extend(chunks)
        return sorted(all_chunks, key=lambda x: x['similarity'], reverse=True)
