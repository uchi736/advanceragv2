-- ============================================
-- Advanced RAG System - Complete Database Schema
-- ============================================
-- Description: Unified schema for the entire RAG system
-- Version: 1.0
-- Date: 2025-12-11
--
-- Prerequisites:
--   - PostgreSQL 12+
--   - pgvector extension (requires superuser or rds_superuser role)
--   - uuid-ossp extension (requires superuser or rds_superuser role)
--
-- Usage:
--   psql -U your_user -d your_database -f database_schema.sql
--   or
--   python setup_database.py
-- ============================================

-- ============================================
-- 1. Extensions
-- ============================================
-- Note: On managed databases (AWS RDS, Azure PostgreSQL, etc.),
-- these extensions must be enabled by a superuser beforehand.
-- If you don't have permission, ask your DBA to run:
--   CREATE EXTENSION vector;
--   CREATE EXTENSION "uuid-ossp";

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 2. Jargon Dictionary Table
-- ============================================
-- Stores extracted specialized terms and their definitions
-- Used by: term extraction, HDBSCAN synonym detection, UI

CREATE TABLE IF NOT EXISTS jargon_dictionary (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(255) NOT NULL DEFAULT 'documents',
    term TEXT NOT NULL,
    definition TEXT NOT NULL,
    domain TEXT,                    -- 分野 (例: 機械工学、電気工学)
    aliases TEXT[],                 -- 類義語リスト
    related_terms TEXT[],           -- 関連用語リスト
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_name, term)
);

-- Indexes for jargon_dictionary
CREATE INDEX IF NOT EXISTS idx_jargon_collection ON jargon_dictionary(collection_name);
CREATE INDEX IF NOT EXISTS idx_jargon_term ON jargon_dictionary(term);
CREATE INDEX IF NOT EXISTS idx_jargon_domain ON jargon_dictionary(domain) WHERE domain IS NOT NULL;

-- Updated timestamp trigger for jargon_dictionary
CREATE OR REPLACE FUNCTION update_jargon_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_jargon_updated
BEFORE UPDATE ON jargon_dictionary
FOR EACH ROW EXECUTE FUNCTION update_jargon_updated_at();

-- ============================================
-- 3. Knowledge Graph - Node Table
-- ============================================
-- Stores nodes in the knowledge graph (terms, categories, domains)
-- Used by: knowledge graph builder, graph visualizer, query expander

DROP TABLE IF EXISTS knowledge_edges CASCADE;
DROP TABLE IF EXISTS knowledge_nodes CASCADE;

CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type VARCHAR(50) NOT NULL CHECK (
        node_type IN ('Term', 'Category', 'Domain', 'Component', 'System')
    ),
    term VARCHAR(255),
    definition TEXT,
    properties JSONB DEFAULT '{}',
    embedding vector(1536),          -- Note: 1536 dimensions for text-embedding-3-small
                                     -- If changing embedding model, adjust this dimension
                                     -- text-embedding-3-large: vector(3072)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Partial unique index for Term nodes only
CREATE UNIQUE INDEX uniq_term_only_term_nodes
ON knowledge_nodes (term)
WHERE node_type = 'Term';

-- HNSW index for vector similarity search
CREATE INDEX idx_nodes_embedding_hnsw
ON knowledge_nodes
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Other indexes
CREATE INDEX idx_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX idx_nodes_properties ON knowledge_nodes USING GIN(properties);

-- ============================================
-- 4. Knowledge Graph - Edge Table
-- ============================================
-- Stores relationships between nodes in the knowledge graph

CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL CHECK (
        edge_type IN (
            -- Hierarchical relations
            'IS_A',           -- 〜の一種である (specific→general)
            'HAS_SUBTYPE',    -- 下位分類を持つ (general→specific)
            'BELONGS_TO',     -- カテゴリに属する (term→category)

            -- Compositional relations
            'PART_OF',        -- 〜の一部である (part→whole)
            'HAS_COMPONENT',  -- 構成要素を持つ (whole→part)
            'INCLUDES',       -- 含む/包含する (container→contained)

            -- Functional relations
            'USED_FOR',       -- 〜に使用される (tool→purpose)
            'PERFORMS',       -- 〜を実行する (agent→action)
            'CONTROLS',       -- 〜を制御する (controller→controlled)
            'MEASURES',       -- 〜を測定する (instrument→measured)

            -- Association relations
            'RELATED_TO',     -- 関連する (bidirectional)
            'SIMILAR_TO',     -- 類似する (bidirectional)
            'SYNONYM',        -- 同義語 (bidirectional)
            'CO_OCCURS_WITH', -- 共起する (bidirectional)
            'DEPENDS_ON',     -- 依存する (dependent→dependency)

            -- Process relations
            'CAUSES',         -- 引き起こす (cause→effect)
            'PREVENTS',       -- 防止する (preventer→prevented)
            'PROCESSES',      -- 処理する (processor→processed)
            'GENERATES'       -- 生成する (generator→generated)
        )
    ),
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT no_self_loop CHECK (source_id != target_id),
    CONSTRAINT unique_edge UNIQUE(source_id, target_id, edge_type)
);

-- Edge indexes
CREATE INDEX idx_edges_source ON knowledge_edges(source_id);
CREATE INDEX idx_edges_target ON knowledge_edges(target_id);
CREATE INDEX idx_edges_type ON knowledge_edges(edge_type);
CREATE INDEX idx_edges_weight ON knowledge_edges(weight DESC);
CREATE INDEX idx_edges_confidence ON knowledge_edges(confidence DESC);
CREATE INDEX idx_edges_properties ON knowledge_edges USING GIN(properties);

-- ============================================
-- 5. Triggers for Knowledge Graph
-- ============================================

-- Auto-update updated_at timestamp for knowledge_nodes
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_nodes_updated
BEFORE UPDATE ON knowledge_nodes
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================
-- 6. Helper Functions for Knowledge Graph
-- ============================================

-- Get node by term
CREATE OR REPLACE FUNCTION get_node_by_term(p_term VARCHAR)
RETURNS TABLE(
    id UUID,
    node_type VARCHAR,
    term VARCHAR,
    definition TEXT,
    properties JSONB,
    embedding vector,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT n.*
    FROM knowledge_nodes n
    WHERE n.term = p_term
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Get edges for a node
CREATE OR REPLACE FUNCTION get_edges_for_node(p_node_id UUID)
RETURNS TABLE(
    edge_id UUID,
    source_id UUID,
    target_id UUID,
    edge_type VARCHAR,
    weight FLOAT,
    confidence FLOAT,
    properties JSONB,
    direction VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id as edge_id,
        e.source_id,
        e.target_id,
        e.edge_type,
        e.weight,
        e.confidence,
        e.properties,
        'outgoing'::VARCHAR as direction
    FROM knowledge_edges e
    WHERE e.source_id = p_node_id

    UNION ALL

    SELECT
        e.id as edge_id,
        e.source_id,
        e.target_id,
        e.edge_type,
        e.weight,
        e.confidence,
        e.properties,
        'incoming'::VARCHAR as direction
    FROM knowledge_edges e
    WHERE e.target_id = p_node_id;
END;
$$ LANGUAGE plpgsql;

-- Get subgraph around a node (BFS)
CREATE OR REPLACE FUNCTION get_subgraph(
    p_center_id UUID,
    p_max_depth INTEGER DEFAULT 2
)
RETURNS TABLE(
    node_id UUID,
    depth INTEGER
) AS $$
DECLARE
    current_depth INTEGER := 0;
BEGIN
    -- Create temp table for BFS
    CREATE TEMP TABLE IF NOT EXISTS bfs_nodes (
        node_id UUID PRIMARY KEY,
        depth INTEGER
    ) ON COMMIT DROP;

    -- Start with center node
    INSERT INTO bfs_nodes VALUES (p_center_id, 0);

    -- BFS traversal
    WHILE current_depth < p_max_depth LOOP
        INSERT INTO bfs_nodes
        SELECT DISTINCT
            CASE
                WHEN e.source_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth)
                THEN e.target_id
                ELSE e.source_id
            END,
            current_depth + 1
        FROM knowledge_edges e
        WHERE (
            e.source_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth)
            OR e.target_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth)
        )
        AND NOT EXISTS (
            SELECT 1 FROM bfs_nodes bn
            WHERE bn.node_id = CASE
                WHEN e.source_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth)
                THEN e.target_id
                ELSE e.source_id
            END
        );

        current_depth := current_depth + 1;
    END LOOP;

    RETURN QUERY SELECT * FROM bfs_nodes;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 7. Views
-- ============================================

-- View for term relationships
CREATE OR REPLACE VIEW v_term_relationships AS
SELECT
    n1.term as source_term,
    e.edge_type,
    n2.term as target_term,
    e.weight,
    e.confidence,
    e.properties->>'provenance' as provenance
FROM knowledge_edges e
JOIN knowledge_nodes n1 ON e.source_id = n1.id
JOIN knowledge_nodes n2 ON e.target_id = n2.id
WHERE n1.node_type = 'Term' AND n2.node_type = 'Term';

-- View for graph statistics
CREATE OR REPLACE VIEW v_graph_statistics AS
SELECT
    (SELECT COUNT(*) FROM knowledge_nodes) as total_nodes,
    (SELECT COUNT(*) FROM knowledge_nodes WHERE node_type = 'Term') as term_nodes,
    (SELECT COUNT(*) FROM knowledge_nodes WHERE node_type = 'Category') as category_nodes,
    (SELECT COUNT(*) FROM knowledge_edges) as total_edges,
    (SELECT COUNT(DISTINCT edge_type) FROM knowledge_edges) as edge_types,
    (SELECT AVG(weight) FROM knowledge_edges) as avg_edge_weight,
    (SELECT AVG(confidence) FROM knowledge_edges) as avg_edge_confidence;

-- View for jargon dictionary statistics
CREATE OR REPLACE VIEW v_jargon_statistics AS
SELECT
    collection_name,
    COUNT(*) as total_terms,
    COUNT(domain) as terms_with_domain,
    COUNT(*) FILTER (WHERE array_length(aliases, 1) > 0) as terms_with_aliases,
    AVG(array_length(aliases, 1)) as avg_aliases_per_term,
    COUNT(*) FILTER (WHERE array_length(related_terms, 1) > 0) as terms_with_related_terms
FROM jargon_dictionary
GROUP BY collection_name;

-- ============================================
-- 8. Maintenance
-- ============================================

-- Analyze tables for query optimization
ANALYZE jargon_dictionary;
ANALYZE knowledge_nodes;
ANALYZE knowledge_edges;

-- ============================================
-- 9. Setup Complete Message
-- ============================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Database schema setup completed successfully!';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Created tables:';
    RAISE NOTICE '  - jargon_dictionary';
    RAISE NOTICE '  - knowledge_nodes';
    RAISE NOTICE '  - knowledge_edges';
    RAISE NOTICE '';
    RAISE NOTICE 'Created views:';
    RAISE NOTICE '  - v_term_relationships';
    RAISE NOTICE '  - v_graph_statistics';
    RAISE NOTICE '  - v_jargon_statistics';
    RAISE NOTICE '';
    RAISE NOTICE 'You can verify the tables with:';
    RAISE NOTICE '  SELECT * FROM v_graph_statistics;';
    RAISE NOTICE '  SELECT * FROM v_jargon_statistics;';
    RAISE NOTICE '============================================';
END $$;
