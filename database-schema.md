# AlignAI Database Schema

## Overview

AlignAI uses Neon PostgreSQL with pgvector extension for storing policies, analysis logs, and other application data. The schema is designed to support multi-tenancy, comprehensive logging for compliance purposes, and efficient vector search.

## Tables

### 1. `policies`

Stores chunks of the EU AI Act and other regulatory texts with vector embeddings for semantic search.

```sql
CREATE TABLE policies (
    id SERIAL PRIMARY KEY,
    act_title TEXT NOT NULL,                   -- e.g., "EU AI Act"
    document_type TEXT NOT NULL,               -- e.g., "Regulation", "Guideline", "Annex"
    chapter_identifier TEXT,                   -- e.g., "Chapter 2"
    chapter_title TEXT,                        -- e.g., "High-Risk AI Systems" 
    article_number TEXT,                       -- e.g., "Article 10"
    section_identifier TEXT,                   -- e.g., "Section 2"
    paragraph_identifier TEXT,                 -- e.g., "3"
    point_identifier TEXT,                     -- e.g., "a"
    policy_text TEXT NOT NULL,                 -- The actual text chunk
    category TEXT[],                           -- e.g., ["Bias", "PII", "Transparency"]
    embedding vector(384) NOT NULL,            -- Vector embedding for semantic search
    publication_date DATE,                     -- When the policy was published
    version TEXT,                              -- Version of the regulation
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create HNSW index for efficient vector search
CREATE INDEX ON policies USING hnsw (embedding vector_cosine_ops);
```

### 2. `organizations`

Stores information about organizations using AlignAI.

```sql
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    subscription_tier TEXT NOT NULL DEFAULT 'free',
    settings JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 3. `users`

Managed by Supabase Auth with custom fields.

```sql
-- This table is managed by Supabase Auth
-- Additional custom fields can be added as needed
```

### 4. `prompts_log`

Logs all analyzed prompts with detailed risk assessments for auditing and reporting.

```sql
CREATE TABLE prompts_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    user_id UUID NOT NULL,
    prompt_text TEXT NOT NULL,
    analysis_results JSONB NOT NULL,           -- Stores risk scores, flagged policies, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- JSON structure for analysis_results:
    -- {
    --   "overall_risk_score": 0.75,
    --   "risk_categories": {
    --     "bias": 0.8,
    --     "pii": 0.6,
    --     "policy_violation": 0.7
    --   },
    --   "flagged_policies": [
    --     {
    --       "policy_id": 123,
    --       "similarity_score": 0.85,
    --       "article": "Article 10",
    --       "text": "..."
    --     }
    --   ],
    --   "token_level_risks": [
    --     {
    --       "start_pos": 10,
    --       "end_pos": 15,
    --       "risk_score": 0.9,
    --       "risk_type": "bias",
    --       "explanation": "Gender bias detected"
    --     }
    --   ],
    --   "pii_detected": [
    --     {
    --       "type": "EMAIL",
    --       "start_pos": 20,
    --       "end_pos": 35
    --     }
    --   ]
    -- }
);

-- Index for faster queries by organization
CREATE INDEX prompts_log_org_idx ON prompts_log (organization_id);
```

### 5. `llm_outputs_log`

Similar to prompts_log but for tracking LLM-generated outputs and their compliance analysis.

```sql
CREATE TABLE llm_outputs_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    user_id UUID NOT NULL,
    prompt_id UUID REFERENCES prompts_log(id),  -- Optional reference to the prompt
    output_text TEXT NOT NULL,
    analysis_results JSONB NOT NULL,            -- Same structure as prompts_log
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX llm_outputs_log_org_idx ON llm_outputs_log (organization_id);
```

### 6. `risk_definitions`

Configurable rules and definitions for bias, PII, and other compliance concerns.

```sql
CREATE TABLE risk_definitions (
    id SERIAL PRIMARY KEY,
    risk_type TEXT NOT NULL,                 -- e.g., "bias", "pii", "prohibited_practice"
    subtype TEXT NOT NULL,                   -- e.g., "gender_bias", "email" (for PII)
    detection_rule JSONB,                    -- Rule definition (regex, keywords, etc.)
    severity TEXT NOT NULL,                  -- "high", "medium", "low"
    description TEXT NOT NULL,               -- Human-readable description
    recommendation TEXT,                     -- Suggested remediation
    related_policies TEXT[],                 -- IDs of related policies
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## Row Level Security Policies

To ensure proper data isolation in a multi-tenant environment, Row Level Security (RLS) policies are implemented:

```sql
-- Organizations table RLS
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;

CREATE POLICY organization_members_select ON organizations
    FOR SELECT
    USING (
        auth.uid() IN (
            SELECT user_id FROM organization_members WHERE organization_id = organizations.id
        )
    );

-- Prompts log RLS
ALTER TABLE prompts_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY prompts_log_select ON prompts_log
    FOR SELECT
    USING (
        auth.uid() IN (
            SELECT user_id FROM organization_members WHERE organization_id = prompts_log.organization_id
        )
    );
```

## Indexes

In addition to the HNSW index for vector search, other indexes are added for query performance:

```sql
-- Index for policy filtering
CREATE INDEX policies_category_idx ON policies USING GIN (category);

-- Full text search index for policy text
CREATE INDEX policies_text_idx ON policies USING GIN (to_tsvector('english', policy_text));
```
