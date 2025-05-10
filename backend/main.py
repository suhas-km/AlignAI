from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routers import analyze_simple, policies, reports
from core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("alignai")

# Create FastAPI app
app = FastAPI(
    title="AlignAI API",
    description="AI alignment and ethical guardrail API ensuring responsible, transparent, and safe interactions with AI",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze_simple.router)
app.include_router(policies.router)
app.include_router(reports.router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AlignAI API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint with more details."""
    return {
        "status": "healthy",
        "api_version": app.version,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
