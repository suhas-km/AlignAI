from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json
import logging

from database.session import get_db
from schemas.prompt import PromptAnalyzeRequest, PromptAnalysisResponse
from schemas.output import OutputAnalyzeRequest, OutputAnalysisResponse
from core.embeddings.text_embedding import TextEmbeddingService
from core.bias_detection.bias_detector import BiasDetectionEngine
from core.pii_detection.pii_detector import PIIDetectionEngine
from core.guardrails.compliance_guard import ComplianceGuard

router = APIRouter(
    prefix="/api/v1/analyze",
    tags=["analysis"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Initialize services
text_embedding_service = TextEmbeddingService()
bias_detection_engine = BiasDetectionEngine()
pii_detection_engine = PIIDetectionEngine()
compliance_guard = ComplianceGuard()

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
            
            # Process the prompt
            token_risks = []
            policy_matches = []
            
            # Analyze based on options
            if request.options.get("analyze_policy", True):
                # Convert text to embeddings
                embedding = text_embedding_service.embed_text(request.text)
                
                # Find similar policies
                # In a real implementation, this would query the database using pgvector
                # Here we're using a placeholder for demonstration
                policy_matches = []  # Replace with actual vector search
            
            if request.options.get("analyze_bias", True):
                # Detect bias
                bias_results = bias_detection_engine.detect_bias(request.text)
                token_risks.extend(bias_results)
            
            if request.options.get("analyze_pii", True):
                # Detect PII
                pii_results = pii_detection_engine.detect_pii(request.text)
                token_risks.extend(pii_results)
            
            # Use compliance guard to aggregate results
            overall_risk = compliance_guard.calculate_overall_risk(token_risks, policy_matches)
            
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

@router.websocket("/output")
async def analyze_output_ws(websocket: WebSocket, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time LLM output analysis."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Convert to Pydantic model for validation
            request = OutputAnalyzeRequest(**request_data)
            
            # Similar processing to prompt analysis
            # (Implementation would be similar to analyze_prompt_ws)
            
            # Placeholder response
            response = {
                "token_risks": [],
                "policy_matches": [],
                "overall_risk": {
                    "score": 0.0,
                    "categories": {}
                }
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
        # Similar logic to WebSocket version but returns a single response
        token_risks = []
        policy_matches = []
        
        # Implementation would be similar to WebSocket version
        
        overall_risk = {
            "score": 0.0,
            "categories": {}
        }
        
        return {
            "token_risks": token_risks,
            "policy_matches": policy_matches,
            "overall_risk": overall_risk
        }
    
    except Exception as e:
        logger.error(f"Error in analyze_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/output", response_model=OutputAnalysisResponse)
async def analyze_output(
    request: OutputAnalyzeRequest, 
    db: Session = Depends(get_db)
):
    """HTTP endpoint for LLM output analysis (non-WebSocket version)."""
    # Implementation would be similar to analyze_prompt
    return {
        "token_risks": [],
        "policy_matches": [],
        "overall_risk": {
            "score": 0.0,
            "categories": {}
        }
    }
