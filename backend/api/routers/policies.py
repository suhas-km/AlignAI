from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from database.session import get_db
from schemas.policy import PolicyCreate, PolicyResponse, PolicySearchQuery
from database.models.policy import Policy

router = APIRouter(
    prefix="/api/v1/policies",
    tags=["policies"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/", response_model=List[PolicyResponse])
async def get_policies(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 20,
    act: Optional[str] = None,
    article: Optional[str] = None,
    category: Optional[str] = None,
):
    """Get a paginated list of policies with optional filters."""
    try:
        query = db.query(Policy)
        
        # Apply filters if provided
        if act:
            query = query.filter(Policy.act_title == act)
        if article:
            query = query.filter(Policy.article_number == article)
        if category:
            query = query.filter(Policy.category.contains([category]))
        
        # Apply pagination
        policies = query.offset(skip).limit(limit).all()
        
        return policies
        
    except Exception as e:
        logger.error(f"Error getting policies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving policies: {str(e)}")

@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(
    policy_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific policy by ID."""
    try:
        policy = db.query(Policy).filter(Policy.id == policy_id).first()
        if policy is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")
        
        return policy
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting policy {policy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving policy: {str(e)}")

@router.post("/search", response_model=List[PolicyResponse])
async def search_policies(
    query: PolicySearchQuery,
    db: Session = Depends(get_db),
):
    """Semantic search for policies based on text query."""
    try:
        from core.embeddings.text_embedding import TextEmbeddingService
        
        # Initialize embeddings service
        embedding_service = TextEmbeddingService()
        
        # Generate embedding for the query
        query_embedding = embedding_service.embed_text(query.query)
        
        # In a real implementation, this would use pgvector to perform semantic search
        # Here we're using a placeholder that would be replaced with actual vector search
        
        # Placeholder implementation
        policies = db.query(Policy).limit(query.limit).all()
        
        return policies
        
    except Exception as e:
        logger.error(f"Error searching policies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching policies: {str(e)}")

@router.post("/", response_model=PolicyResponse)
async def create_policy(
    policy: PolicyCreate,
    db: Session = Depends(get_db),
):
    """Create a new policy."""
    try:
        db_policy = Policy(**policy.dict())
        db.add(db_policy)
        db.commit()
        db.refresh(db_policy)
        
        return db_policy
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating policy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating policy: {str(e)}")

@router.delete("/{policy_id}", status_code=204)
async def delete_policy(
    policy_id: int,
    db: Session = Depends(get_db),
):
    """Delete a policy."""
    try:
        policy = db.query(Policy).filter(Policy.id == policy_id).first()
        if policy is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")
        
        db.delete(policy)
        db.commit()
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting policy {policy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting policy: {str(e)}")
