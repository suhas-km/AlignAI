from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class RiskDefinitionBase(BaseModel):
    """Base schema for risk definitions."""
    risk_type: str
    subtype: str
    detection_rule: Optional[Dict[str, Any]] = None
    severity: str
    description: str
    recommendation: Optional[str] = None
    related_policies: Optional[List[str]] = None

class RiskDefinitionCreate(RiskDefinitionBase):
    """Schema for creating a risk definition."""
    pass

class RiskDefinitionUpdate(BaseModel):
    """Schema for updating a risk definition."""
    risk_type: Optional[str] = None
    subtype: Optional[str] = None
    detection_rule: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    recommendation: Optional[str] = None
    related_policies: Optional[List[str]] = None

class RiskDefinitionResponse(RiskDefinitionBase):
    """Response schema for risk definitions."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True
