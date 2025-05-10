# AlignAI Project Structure

## Overview
AlignAI is structured as a monorepo containing both frontend (Next.js) and backend (FastAPI) code. This allows for shared types and easier development.

```
/AlignAI
├── frontend/                      # Next.js frontend application
│   ├── app/                       # App Router structure
│   │   ├── api/                   # API routes (for client-side endpoints)
│   │   ├── analyzer/              # Prompt analysis interface
│   │   ├── dashboard/             # User dashboard
│   │   ├── policies/              # Policy library browser
│   │   ├── reports/               # Compliance reports
│   │   ├── sandbox/               # Public compliance sandbox
│   │   ├── layout.tsx             # Root layout
│   │   └── page.tsx               # Landing page
│   ├── components/                # Reusable React components
│   │   ├── analyzer/              # Analyzer-specific components
│   │   ├── dashboard/             # Dashboard-specific components 
│   │   ├── layout/                # Layout components
│   │   └── ui/                    # UI components (buttons, inputs, etc.)
│   ├── lib/                       # Frontend utilities
│   │   ├── api.ts                 # API client
│   │   └── websocket.ts           # WebSocket client
│   ├── public/                    # Static assets
│   ├── styles/                    # Global styles
│   ├── types/                     # TypeScript type definitions
│   └── utils/                     # Utility functions
│
├── backend/                       # FastAPI backend application
│   ├── api/                       # API routes
│   │   ├── analyze.py             # Prompt analysis endpoints
│   │   ├── auth.py                # Authentication endpoints
│   │   ├── policies.py            # Policy retrieval endpoints
│   │   └── reports.py             # Report generation endpoints
│   ├── core/                      # Core application code
│   │   ├── bias_detection/        # Bias detection logic
│   │   ├── guardrails/            # LangChain Guardrails setup
│   │   ├── pii_detection/         # PII detection logic
│   │   └── semantic_search/       # Vector search implementation
│   ├── database/                  # Database models and connections
│   │   ├── migrations/            # Database migrations
│   │   └── models/                # SQLAlchemy models
│   ├── ingest/                    # Data ingestion scripts
│   │   ├── eu_ai_act.py           # EU AI Act ingestion
│   │   └── vectorize.py           # Text vectorization
│   ├── schemas/                   # Pydantic schemas
│   └── utils/                     # Utility functions
│
├── shared/                        # Shared code between frontend and backend
│   └── types/                     # TypeScript/Python shared types
│
├── tests/                         # Tests
│   ├── frontend/                  # Frontend tests
│   └── backend/                   # Backend tests
│
├── .env.example                   # Example environment variables
├── .gitignore                     # Git ignore file
├── docker-compose.yml             # Docker Compose for local development
├── package.json                   # Root package.json for monorepo commands
├── README.md                      # Project README
└── tsconfig.json                  # TypeScript configuration
```
