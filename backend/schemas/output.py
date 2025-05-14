from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from schemas.prompt import TokenRisk, PolicyMatch, OverallRisk

class OutputAnalyzeRequest(BaseModel):
    """Request for LLM output analysis."""
    text: str
    prompt_id: Optional[UUID] = None
    options: Optional[Dict[str, bool]] = Field(
        default={
            "analyze_bias": True,
            "analyze_pii": True,
            "analyze_policy": True
        }
    )

class OutputAnalysisResponse(BaseModel):
    """Response from LLM output analysis."""
    token_risks: List[TokenRisk]
    policy_matches: List[PolicyMatch]
    overall_risk: OverallRisk
    recommendations: List[str] = Field(default_factory=list)

class OutputLogCreate(BaseModel):
    """Schema for logging an LLM output analysis."""
    organization_id: UUID
    user_id: UUID
    prompt_id: Optional[UUID] = None
    output_text: str
    analysis_results: Dict[str, Any]

class OutputLogResponse(BaseModel):
    """Response schema for output log."""
    id: UUID
    organization_id: UUID
    user_id: UUID
    prompt_id: Optional[UUID]
    output_text: str
    analysis_results: Dict[str, Any]
    created_at: datetime
    
    class Config:
        orm_mode = True
