from typing import List
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AlignAI"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://localhost:3000", "http://localhost:3001", "https://localhost:3001", "*"]
    
    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Rate limiting (disabled for now)
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 1000
    
    class Config:
        case_sensitive = True

# Create settings instance
settings = Settings()
