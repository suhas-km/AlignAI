from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class TokenRisk(BaseModel):
    """Represents risk information for a specific text token/span."""
    start: int
    end: int
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_type: str
    explanation: str

class PolicyMatch(BaseModel):
    """Represents a matched policy from semantic search."""
    policy_id: int
    article: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    text_snippet: str

class OverallRisk(BaseModel):
    """Represents overall risk assessment."""
    score: float = Field(ge=0.0, le=1.0)
    categories: Dict[str, float]

class PromptAnalyzeRequest(BaseModel):
    """Request for prompt analysis."""
    text: str
    options: Optional[Dict[str, bool]] = Field(
        default={
            "analyze_bias": True,
            "analyze_pii": True,
            "analyze_policy": True
        }
    )

class PromptAnalysisResponse(BaseModel):
    """Response from prompt analysis."""
    token_risks: List[TokenRisk]
    policy_matches: List[PolicyMatch]
    overall_risk: OverallRisk
    recommendations: List[str] = Field(default_factory=list)

class PromptLogCreate(BaseModel):
    """Schema for logging a prompt analysis."""
    organization_id: UUID
    user_id: UUID
    prompt_text: str
    analysis_results: Dict[str, Any]

class PromptLogResponse(BaseModel):
    """Response schema for prompt log."""
    id: UUID
    organization_id: UUID
    user_id: UUID
    prompt_text: str
    analysis_results: Dict[str, Any]
    created_at: datetime
    
    class Config:
        orm_mode = True
