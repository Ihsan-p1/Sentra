-- Sentra Database Schema (No pgvector)
-- Using float8[] arrays for embeddings

-- =====================================================
-- MAIN TABLES
-- =====================================================

-- News articles from different media sources
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    media_source VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT,
    author VARCHAR(255),
    published_date TIMESTAMP,
    category VARCHAR(100),
    -- Fingerprint of the corpus file this article was ingested from, e.g.
    -- "indonesia-election-2024@a1b2c3d4e5f6". Without it there is no way to tell
    -- which version of the corpus a populated database actually contains.
    corpus_version VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Article chunks with embeddings
CREATE TABLE IF NOT EXISTS article_chunks (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding FLOAT8[],  -- Changed from vector(384) to float8[]
    -- Which model produced `embedding`. A vector is only comparable to another
    -- vector from the same model, and that used to be untracked: EMBEDDING_MODEL
    -- was changed from an English-only model to a multilingual one without
    -- re-ingesting. Both emit 384 dimensions, so nothing failed - retrieval just
    -- silently compared vectors across two embedding spaces, which compressed
    -- every similarity score and broke cross-lingual matching. Recording the
    -- model name makes that mismatch detectable with one query.
    embedding_model VARCHAR(255),
    embedding_dim INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- MIGRATIONS
-- CREATE TABLE IF NOT EXISTS will not add columns to a table that already
-- exists, so new columns need an explicit ALTER for databases created before
-- they were introduced. These are idempotent and safe to re-run.
-- =====================================================

ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(255);
ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_dim INTEGER;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS corpus_version VARCHAR(64);

CREATE INDEX IF NOT EXISTS article_chunks_embedding_model_idx
    ON article_chunks (embedding_model);

-- =====================================================
-- LABELED DATA TABLES
-- =====================================================

CREATE TABLE IF NOT EXISTS hallucination_labels (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL,
    source_chunks TEXT[] NOT NULL,
    is_supported BOOLEAN NOT NULL,
    max_similarity FLOAT,
    labeler VARCHAR(100) DEFAULT 'synthetic',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS confidence_labels (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    retrieved_chunk_ids INTEGER[] NOT NULL,
    confidence_score FLOAT NOT NULL,
    labeler VARCHAR(100) DEFAULT 'synthetic',
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS articles_updated_at ON articles;
CREATE TRIGGER articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
