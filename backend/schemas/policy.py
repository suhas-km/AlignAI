from typing import List, Optional
from datetime import date
from pydantic import BaseModel, Field

class PolicyBase(BaseModel):
    act_title: str
    document_type: str
    chapter_identifier: Optional[str] = None
    chapter_title: Optional[str] = None
    article_number: Optional[str] = None
    section_identifier: Optional[str] = None
    paragraph_identifier: Optional[str] = None
    point_identifier: Optional[str] = None
    policy_text: str
    category: Optional[List[str]] = None
    publication_date: Optional[date] = None
    version: Optional[str] = None

class PolicyCreate(PolicyBase):
    embedding: List[float]

class PolicyUpdate(BaseModel):
    act_title: Optional[str] = None
    document_type: Optional[str] = None
    chapter_identifier: Optional[str] = None
    chapter_title: Optional[str] = None
    article_number: Optional[str] = None
    section_identifier: Optional[str] = None
    paragraph_identifier: Optional[str] = None
    point_identifier: Optional[str] = None
    policy_text: Optional[str] = None
    category: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    publication_date: Optional[date] = None
    version: Optional[str] = None

class PolicyResponse(PolicyBase):
    id: int
    created_at: date
    updated_at: date
    
    class Config:
        orm_mode = True

class PolicySearchQuery(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)
    categories: Optional[List[str]] = None
