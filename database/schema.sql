-- ==============================================================================
-- Schema for MyInvestmentBanker Vector & Relational Database (Supabase Postgres)
-- Run this script inside the Supabase SQL Editor.
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Portfolio Holdings Table
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    symbol VARCHAR(12) PRIMARY KEY,
    name VARCHAR(255),
    quantity NUMERIC(15, 4) NOT NULL DEFAULT 0,
    cost_basis NUMERIC(15, 4) NOT NULL DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_portfolio_holdings_modtime ON portfolio_holdings;
CREATE TRIGGER update_portfolio_holdings_modtime
    BEFORE UPDATE ON portfolio_holdings
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();


-- 2. User Investment Thesis Table
CREATE TABLE IF NOT EXISTS investment_thesis (
    symbol VARCHAR(12) PRIMARY KEY REFERENCES portfolio_holdings(symbol) ON DELETE CASCADE,
    thesis_text TEXT NOT NULL,
    thesis_vector vector(768),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

DROP TRIGGER IF EXISTS update_investment_thesis_modtime ON investment_thesis;
CREATE TRIGGER update_investment_thesis_modtime
    BEFORE UPDATE ON investment_thesis
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();


-- 3. Corporate Analyst Memos
CREATE TABLE IF NOT EXISTS corporate_analyst_memos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(12) NOT NULL REFERENCES portfolio_holdings(symbol) ON DELETE CASCADE,
    period VARCHAR(20) NOT NULL,
    memo_text TEXT NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memos_symbol ON corporate_analyst_memos(symbol);


-- 4. Event Cache / News Digests
CREATE TABLE IF NOT EXISTS news_digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(24),
    entity_type VARCHAR(50) NOT NULL DEFAULT 'symbol',
    entity_key VARCHAR(255) NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    article_vector vector(768),
    url TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_digests(symbol);
CREATE INDEX IF NOT EXISTS idx_news_entity ON news_digests(entity_type, entity_key);
CREATE INDEX IF NOT EXISTS idx_news_vector ON news_digests USING hnsw (article_vector vector_cosine_ops);


-- 5. User Profile & Preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);


-- 6. Short-Term Memory: Secure Chat Logs & Checkpoints
CREATE TABLE IF NOT EXISTS chat_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_logs(user_id);


-- 7. Discovery Run History
CREATE TABLE IF NOT EXISTS discovery_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    policy_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    themes JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary_text TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_discovery_runs_type ON discovery_runs(run_type, created_at DESC);


-- 8. Discovery Recommendations / Candidate Memory
CREATE TABLE IF NOT EXISTS discovery_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES discovery_runs(id) ON DELETE SET NULL,
    theme_key VARCHAR(100) NOT NULL,
    symbol VARCHAR(24) NOT NULL,
    source_etf VARCHAR(24),
    recommendation_type VARCHAR(50),
    status VARCHAR(50) NOT NULL DEFAULT 'recommended',
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    rationale TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_discovery_candidates_theme_symbol ON discovery_candidates(theme_key, symbol, created_at DESC);
