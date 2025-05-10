import uuid
from sqlalchemy import Column, Text, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

from database.session import Base

class PromptLog(Base):
    __tablename__ = "prompts_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )
    user_id = Column(UUID(as_uuid=True), nullable=False)
    prompt_text = Column(Text, nullable=False)
    analysis_results = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=func.now())
