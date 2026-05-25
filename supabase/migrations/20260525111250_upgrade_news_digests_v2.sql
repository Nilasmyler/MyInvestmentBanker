-- Migration: Upgrade news_digests table to support structured entities and metadata.
-- Date: 2026-05-25
-- Description: Adds entity_type, entity_key, and metadata columns to news_digests table for theme-led discovery sweeps, with safe migration of existing rows and index generation.

-- 1. Add entity_type column with a default of 'symbol'
ALTER TABLE news_digests ADD COLUMN IF NOT EXISTS entity_type VARCHAR(50) NOT NULL DEFAULT 'symbol';

-- 2. Add entity_key column (temporarily nullable so we can migrate existing data safely)
ALTER TABLE news_digests ADD COLUMN IF NOT EXISTS entity_key VARCHAR(255);

-- 3. Populate entity_key with existing symbol data where it is currently null
UPDATE news_digests SET entity_key = symbol WHERE entity_key IS NULL;

-- 4. Set entity_key to NOT NULL now that existing rows have been populated
ALTER TABLE news_digests ALTER COLUMN entity_key SET NOT NULL;

-- 5. Add metadata column with an empty jsonb object default
ALTER TABLE news_digests ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

-- 6. Create indexes to speed up entity-based filtering and news lookup
CREATE INDEX IF NOT EXISTS idx_news_entity ON news_digests(entity_type, entity_key);
CREATE INDEX IF NOT EXISTS idx_news_entity_url ON news_digests(entity_type, entity_key, url);
