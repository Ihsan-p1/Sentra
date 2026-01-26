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
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

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
