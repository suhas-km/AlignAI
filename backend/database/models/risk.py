from sqlalchemy import Column, Integer, String, Text, ARRAY, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB

from database.session import Base

class RiskDefinition(Base):
    __tablename__ = "risk_definitions"

    id = Column(Integer, primary_key=True, index=True)
    risk_type = Column(String, nullable=False)
    subtype = Column(String, nullable=False)
    detection_rule = Column(JSONB)
    severity = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    recommendation = Column(Text)
    related_policies = Column(ARRAY(String))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
