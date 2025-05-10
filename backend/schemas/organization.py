from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel

class OrganizationBase(BaseModel):
    name: str
    subscription_tier: str = "free"
    settings: Optional[Dict[str, Any]] = None

class OrganizationCreate(OrganizationBase):
    pass

class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    subscription_tier: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

class OrganizationResponse(OrganizationBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class OrganizationMemberBase(BaseModel):
    organization_id: UUID
    user_id: UUID
    role: str = "member"

class OrganizationMemberCreate(OrganizationMemberBase):
    pass

class OrganizationMemberResponse(OrganizationMemberBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True
