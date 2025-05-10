from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Table, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

# Association table for many-to-many relationships
prompt_policy_association = Table(
    'prompt_policy_association',
    Base.metadata,
    Column('prompt_id', Integer, ForeignKey('prompts.id'), primary_key=True),
    Column('policy_id', Integer, ForeignKey('policies.id'), primary_key=True)
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    prompts = relationship("Prompt", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"

class Policy(Base):
    __tablename__ = 'policies'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    article = Column(String(50))
    category = Column(String(100))
    content = Column(Text, nullable=False)
    summary = Column(Text)
    risk_level = Column(String(20))  # 'low', 'medium', 'high'
    source_url = Column(String(512))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    prompts = relationship("Prompt", secondary=prompt_policy_association, back_populates="policies")
    
    def __repr__(self):
        return f"<Policy {self.title}>"

class Prompt(Base):
    __tablename__ = 'prompts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text, nullable=False)
    risk_score = Column(Float)
    analysis_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="prompts")
    policies = relationship("Policy", secondary=prompt_policy_association, back_populates="prompts")
    token_risks = relationship("TokenRisk", back_populates="prompt")
    
    def __repr__(self):
        return f"<Prompt {self.id}>"

class TokenRisk(Base):
    __tablename__ = 'token_risks'
    
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'))
    start_pos = Column(Integer, nullable=False)
    end_pos = Column(Integer, nullable=False)
    risk_type = Column(String(50), nullable=False)  # 'bias', 'pii', 'policy_violation'
    risk_score = Column(Float, nullable=False)
    explanation = Column(Text)
    
    # Relationships
    prompt = relationship("Prompt", back_populates="token_risks")
    
    def __repr__(self):
        return f"<TokenRisk {self.id}>"

class ApiKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(100))
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<ApiKey {self.name}>"
