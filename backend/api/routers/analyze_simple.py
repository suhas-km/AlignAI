from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json
import logging
import random

from database.session import get_db
from schemas.prompt import PromptAnalyzeRequest, PromptAnalysisResponse
from schemas.output import OutputAnalyzeRequest, OutputAnalysisResponse

router = APIRouter(
    prefix="/api/v1/analyze",
    tags=["analysis"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.websocket("/prompt")
async def analyze_prompt_ws(websocket: WebSocket, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time prompt analysis."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Convert to Pydantic model for validation
            request = PromptAnalyzeRequest(**request_data)
            
            # Process the prompt (simplified for testing)
            text = request.text
            
            # Create mock analysis results
            token_risks = []
            policy_matches = []
            
            # Add some mock risks if certain keywords are present
            if "personal" in text.lower() or "email" in text.lower() or "phone" in text.lower():
                token_risks.append({
                    "start": text.lower().find("personal"),
                    "end": text.lower().find("personal") + 8 if text.lower().find("personal") >= 0 else 0,
                    "risk_score": 0.8,
                    "risk_type": "pii",
                    "explanation": "Potential PII reference detected"
                })
            
            if "all" in text.lower() and any(term in text.lower() for term in ["woman", "women", "man", "men"]):
                token_risks.append({
                    "start": text.lower().find("all"),
                    "end": text.lower().find("all") + 3,
                    "risk_score": 0.7,
                    "risk_type": "bias",
                    "explanation": "Potential gender bias detected"
                })
            
            # Add mock policy match
            if len(text) > 20:
                # Get a random policy from the database
                policy = db.execute("SELECT id, article, title, summary FROM policies ORDER BY RANDOM() LIMIT 1").fetchone()
                
                if policy:
                    policy_matches.append({
                        "policy_id": policy[0],
                        "article": policy[1],
                        "similarity_score": random.uniform(0.6, 0.9),
                        "text_snippet": policy[3][:100] + "..." if len(policy[3]) > 100 else policy[3]
                    })
            
            # Calculate overall risk
            risk_scores = [risk["risk_score"] for risk in token_risks]
            policy_scores = [match["similarity_score"] for match in policy_matches]
            
            all_scores = risk_scores + policy_scores
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.1
            
            categories = {}
            if any(risk["risk_type"] == "pii" for risk in token_risks):
                categories["pii"] = max([risk["risk_score"] for risk in token_risks if risk["risk_type"] == "pii"])
            
            if any(risk["risk_type"] == "bias" for risk in token_risks):
                categories["bias"] = max([risk["risk_score"] for risk in token_risks if risk["risk_type"] == "bias"])
            
            if policy_matches:
                categories["policy_violation"] = max([match["similarity_score"] for match in policy_matches])
            
            overall_risk = {
                "score": overall_score,
                "categories": categories
            }
            
            # Prepare response
            response = {
                "token_risks": token_risks,
                "policy_matches": policy_matches,
                "overall_risk": overall_risk
            }
            
            # Send response back through WebSocket
            await websocket.send_json(response)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}")
        await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")

@router.post("/prompt", response_model=PromptAnalysisResponse)
async def analyze_prompt(
    request: PromptAnalyzeRequest, 
    db: Session = Depends(get_db)
):
    """HTTP endpoint for prompt analysis (non-WebSocket version)."""
    try:
        # Process the prompt (simplified for testing)
        text = request.text
        logger.info(f"Analyzing prompt via HTTP: {text[:50]}...")
        
        # Create mock analysis results
        token_risks = []
        policy_matches = []
        
        # Add some mock risks if certain keywords are present
        if "personal" in text.lower() or "email" in text.lower() or "phone" in text.lower():
            token_risks.append({
                "start": text.lower().find("personal"),
                "end": text.lower().find("personal") + 8 if text.lower().find("personal") >= 0 else 0,
                "risk_score": 0.8,
                "risk_type": "pii",
                "explanation": "Potential PII reference detected"
            })
        
        if "all" in text.lower() and any(term in text.lower() for term in ["woman", "women", "man", "men"]):
            token_risks.append({
                "start": text.lower().find("all"),
                "end": text.lower().find("all") + 3,
                "risk_score": 0.7,
                "risk_type": "bias",
                "explanation": "Potential gender bias detected"
            })
        
        # Add mock policy match
        if len(text) > 20:
            # Get a random policy from the database
            try:
                policy = db.execute("SELECT id, article, title, summary FROM policies ORDER BY RANDOM() LIMIT 1").fetchone()
                
                if policy:
                    policy_matches.append({
                        "policy_id": policy[0],
                        "article": policy[1],
                        "similarity_score": random.uniform(0.6, 0.9),
                        "text_snippet": policy[3][:100] + "..." if len(policy[3]) > 100 else policy[3]
                    })
            except Exception as db_error:
                logger.warning(f"Database query failed: {str(db_error)}")
                # Add a fallback policy match for testing
                policy_matches.append({
                    "policy_id": 1,
                    "article": "Article 10",
                    "similarity_score": 0.75,
                    "text_snippet": "Data quality and transparency requirements..."
                })
        
        # Calculate overall risk
        risk_scores = [risk["risk_score"] for risk in token_risks]
        policy_scores = [match["similarity_score"] for match in policy_matches]
        
        all_scores = risk_scores + policy_scores
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.1
        
        categories = {}
        if any(risk["risk_type"] == "pii" for risk in token_risks):
            categories["pii"] = max([risk["risk_score"] for risk in token_risks if risk["risk_type"] == "pii"])
        
        if any(risk["risk_type"] == "bias" for risk in token_risks):
            categories["bias"] = max([risk["risk_score"] for risk in token_risks if risk["risk_type"] == "bias"])
        
        if policy_matches:
            categories["policy_violation"] = max([match["similarity_score"] for match in policy_matches])
        
        overall_risk = {
            "score": overall_score,
            "categories": categories
        }
        
        # Prepare response
        response = {
            "token_risks": token_risks,
            "policy_matches": policy_matches,
            "overall_risk": overall_risk
        }
        
        logger.info("HTTP analysis complete")
        return response
    
    except Exception as e:
        logger.error(f"Error in analyze_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
