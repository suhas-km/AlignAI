import datetime
from sqlalchemy import Column, Integer, String, Text, ARRAY, Date, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from database.session import Base

class Policy(Base):
    __tablename__ = "policies"

    id = Column(Integer, primary_key=True, index=True)
    act_title = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    chapter_identifier = Column(String)
    chapter_title = Column(String)
    article_number = Column(String)
    section_identifier = Column(String)
    paragraph_identifier = Column(String)
    point_identifier = Column(String)
    policy_text = Column(Text, nullable=False)
    category = Column(ARRAY(String))
    embedding = Column(Vector(384), nullable=False)
    publication_date = Column(Date)
    version = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
