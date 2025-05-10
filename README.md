# AlignAI

AlignAI is an AI alignment and ethical guardrail app ensuring responsible, transparent, and safe interactions with AI. Built to help organizations comply with the EU AI Act, AlignAI provides real-time analysis of AI prompts and outputs, identifying potential risks related to bias, PII exposure, and regulatory violations.

## Features

- **Real-time Prompt Analysis**: Token-level analysis of prompts with visual heatmap highlighting potential compliance issues
- **Policy-to-Prompt Semantic Matching**: Maps input text against relevant EU AI Act regulations using vector similarity
- **Bias Detection**: Identifies potentially biased language in prompts and AI-generated content
- **PII Detection**: Flags personally identifiable information to prevent data leakage
- **Compliance Dashboard**: Visualizes compliance trends and risk factors across your organization
- **Detailed Reports**: Generates audit-ready compliance reports for regulatory purposes
- **Public Sandbox**: Demonstrates AlignAI's capabilities with sub-200ms analysis response time

## Architecture

AlignAI uses a modern, serverless-first architecture:

- **Frontend**: Next.js with App Router deployed on Vercel Edge
- **Backend**: FastAPI with WebSocket support for real-time analysis
- **Database**: Neon PostgreSQL with pgvector for semantic search
- **Authentication**: Supabase Auth for user and organization management
- **AI Components**:
  - LangChain Guardrails for compliance orchestration
  - Sentence transformer models for text embeddings
  - Specialized PII and bias detection models

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Vercel account (for deployment)
- Neon PostgreSQL database with pgvector extension
- Supabase project for authentication

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alignai.git
cd alignai

# Install dependencies
npm install            # Root monorepo dependencies
cd frontend && npm install  # Frontend dependencies
cd ../backend && pip install -r requirements.txt  # Backend dependencies
```

### Environment Setup

Copy the example environment files and fill in your specific values:

```bash
cp .env.example .env
```

### Running Locally

```bash
# Start the development server
npm run dev
```

Visit `http://localhost:3000` to access the application.

## Implementation Roadmap

AlignAI is being developed in phases:

1. **Foundation & Regulatory Knowledge Ingestion**
   - Core infrastructure setup
   - EU AI Act text processing and vectorization
   - Initial policy library UI

2. **Core Analysis Engine**
   - FastAPI setup with authentication
   - Policy-to-Prompt semantic scoring
   - Baseline PII and bias detection
   - LangChain Guardrails integration

3. **Prompt Analyzer UI & Core User Workflow**
   - Rich text editor with real-time heatmap
   - Risk summary panel with policy references
   - Comprehensive logging and analysis

4. **Dashboard, Reporting & Public Sandbox**
   - Compliance dashboard with visualizations
   - Basic reporting functionality
   - Public demonstration sandbox

## Documentation

Detailed documentation is available in the `/docs` directory, including:

- API documentation
- Component architecture
- Database schema
- Deployment guide
