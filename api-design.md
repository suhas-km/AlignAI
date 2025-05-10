# AlignAI API Design

## Overview

The AlignAI API is built with FastAPI and provides endpoints for prompt analysis, policy retrieval, user management, and reporting. The API includes both REST endpoints and WebSocket connections for real-time analysis.

## Base URL

- Development: `http://localhost:8000/api`
- Production: `https://api.alignai.com/api`

## Authentication

All API endpoints (except public sandbox endpoints) require authentication using JWT tokens provided by Supabase Auth. The token should be included in the `Authorization` header as a Bearer token:

```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### Authentication

#### `POST /auth/register`

Register a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "John Doe"
}
```

**Response:**
```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "name": "John Doe"
}
```

#### `POST /auth/login`

Authenticate a user and return a JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "access_token": "jwt_token",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Prompt Analysis

#### `WebSocket /analyze/prompt`

Real-time analysis of prompt text with token-level feedback.

**Connection Parameters:**
- `token`: JWT authentication token

**Client Message:**
```json
{
  "text": "Generate an algorithm to determine credit worthiness based on user data including name, address, and social security number.",
  "options": {
    "analyze_bias": true,
    "analyze_pii": true,
    "analyze_policy": true
  }
}
```

**Server Messages (stream):**
```json
{
  "token_risks": [
    {
      "start": 45,
      "end": 54,
      "risk_score": 0.85,
      "risk_type": "pii",
      "explanation": "Mentions use of personal data"
    },
    {
      "start": 78,
      "end": 97,
      "risk_score": 0.95,
      "risk_type": "pii",
      "explanation": "Mentions social security number (sensitive PII)"
    }
  ],
  "policy_matches": [
    {
      "policy_id": 123,
      "article": "Article 10",
      "similarity_score": 0.87,
      "text_snippet": "..."
    }
  ],
  "overall_risk": {
    "score": 0.75,
    "categories": {
      "bias": 0.6,
      "pii": 0.9,
      "policy_violation": 0.7
    }
  }
}
```

#### `WebSocket /analyze/output`

Similar to the prompt analysis endpoint, but for analyzing LLM-generated outputs.

#### `POST /api/prompts`

Log a completed prompt analysis to the database.

**Request:**
```json
{
  "prompt_text": "...",
  "analysis_results": {
    "overall_risk_score": 0.75,
    "risk_categories": {
      "bias": 0.8,
      "pii": 0.6,
      "policy_violation": 0.7
    },
    "flagged_policies": [...],
    "token_level_risks": [...],
    "pii_detected": [...]
  }
}
```

**Response:**
```json
{
  "prompt_id": "uuid",
  "created_at": "2025-05-09T23:45:12Z"
}
```

### Policy Retrieval

#### `GET /policies`

Retrieve a paginated list of policies.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20)
- `act`: Filter by act title (e.g., "EU AI Act")
- `article`: Filter by article number
- `category`: Filter by category (e.g., "Bias", "PII")
- `search`: Full-text search query

**Response:**
```json
{
  "total": 245,
  "page": 1,
  "limit": 20,
  "policies": [
    {
      "id": 123,
      "act_title": "EU AI Act",
      "document_type": "Regulation",
      "chapter_identifier": "Chapter 2",
      "chapter_title": "Requirements for High-Risk AI Systems",
      "article_number": "Article 10",
      "section_identifier": null,
      "paragraph_identifier": "1",
      "point_identifier": null,
      "policy_text": "...",
      "category": ["Bias", "Data Governance"]
    }
  ]
}
```

#### `GET /policies/{policy_id}`

Retrieve a specific policy by ID.

**Response:**
```json
{
  "id": 123,
  "act_title": "EU AI Act",
  "document_type": "Regulation",
  "chapter_identifier": "Chapter 2",
  "chapter_title": "Requirements for High-Risk AI Systems",
  "article_number": "Article 10",
  "section_identifier": null,
  "paragraph_identifier": "1",
  "point_identifier": null,
  "policy_text": "...",
  "category": ["Bias", "Data Governance"]
}
```

#### `POST /policies/search`

Semantic search for policies based on text query.

**Request:**
```json
{
  "query": "How should I handle user data?",
  "limit": 5,
  "categories": ["Data Governance", "PII"]
}
```

**Response:**
```json
{
  "results": [
    {
      "policy_id": 123,
      "similarity_score": 0.87,
      "act_title": "EU AI Act",
      "article_number": "Article 10",
      "policy_text": "..."
    }
  ]
}
```

### Dashboard & Reports

#### `GET /dashboard/summary`

Get summary statistics for the organization's dashboard.

**Response:**
```json
{
  "total_prompts_analyzed": 1254,
  "average_risk_score": 0.42,
  "risk_distribution": {
    "high": 124,
    "medium": 356,
    "low": 774
  },
  "top_risk_categories": [
    {"category": "pii", "count": 356},
    {"category": "bias", "count": 243},
    {"category": "prohibited_practice", "count": 87}
  ],
  "trend_data": {
    "dates": ["2025-05-01", "2025-05-02", "..."],
    "risk_scores": [0.45, 0.43, "..."]
  }
}
```

#### `GET /reports`

Get a list of available reports.

**Response:**
```json
{
  "reports": [
    {
      "id": "uuid",
      "name": "Monthly Compliance Report - May 2025",
      "created_at": "2025-05-09T12:00:00Z",
      "type": "monthly"
    }
  ]
}
```

#### `POST /reports/generate`

Generate a new compliance report.

**Request:**
```json
{
  "report_type": "monthly",
  "start_date": "2025-05-01",
  "end_date": "2025-05-31",
  "name": "Monthly Compliance Report - May 2025"
}
```

**Response:**
```json
{
  "report_id": "uuid",
  "status": "processing",
  "estimated_completion_time": "2025-05-09T23:50:00Z"
}
```

#### `GET /reports/{report_id}`

Get a specific report.

**Response:**
```json
{
  "id": "uuid",
  "name": "Monthly Compliance Report - May 2025",
  "created_at": "2025-05-09T12:00:00Z",
  "type": "monthly",
  "status": "completed",
  "data": {
    "summary": {...},
    "detailed_findings": [...],
    "recommendations": [...]
  }
}
```

### Public Sandbox

#### `WebSocket /sandbox/analyze`

Public endpoint for the sandbox demo (no authentication required).

**Client Message:**
```json
{
  "text": "Generate a profile of this person based on their browsing history and social media activity."
}
```

**Server Messages (stream):**
```json
{
  "token_risks": [...],
  "overall_risk": {...}
}
```

## Error Responses

All API endpoints return appropriate HTTP status codes and error messages in case of failures:

```json
{
  "error": {
    "code": "authentication_failed",
    "message": "Invalid authentication credentials"
  }
}
```

Common error codes:
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Rate Limiting

The API implements rate limiting based on subscription tier:
- Free tier: 100 requests per hour
- Pro tier: 1,000 requests per hour
- Enterprise tier: Customizable limits

Rate limit headers are included in API responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```
