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
# Use ML-based detectors by default
compliance_guard = ComplianceGuard(use_ml=True)

@router.websocket("/prompt")
async def analyze_prompt_ws(websocket: WebSocket, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time prompt analysis."""
    await websocket.accept()
    try:
        while True:
            # Receive data from the client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Convert to Pydantic model for validation
            request = PromptAnalyzeRequest(**request_data)
            
            # Map frontend options to backend options
            analysis_options = {
                "analyzeBias": request.options.get("analyze_bias", True),
                "analyzePII": request.options.get("analyze_pii", True),
                "analyzePolicy": request.options.get("analyze_policy", True)
            }
            
            # Use the compliance guard to validate the prompt
            logger.info(f"Analyzing prompt with options: {analysis_options}")
            validation_results = compliance_guard.validate_prompt(request.text, analysis_options)
            
            # Convert the results to a format compatible with the API
            response = {
                "token_risks": validation_results.get("token_risks", []),
                "policy_matches": validation_results.get("policy_matches", []),
                "overall_risk": validation_results.get("overall_risk", {
                    "score": 0.0,
                    "categories": {}
                }),
                "recommendations": validation_results.get("recommendations", [])
            }
            
            logger.info(f"Analysis completed with risk score: {response['overall_risk']['score']}")
            
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
            # Receive data from the client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Convert to Pydantic model for validation
            request = OutputAnalyzeRequest(**request_data)
            
            # Map frontend options to backend options
            analysis_options = {
                "analyzeBias": request.options.get("analyze_bias", True),
                "analyzePII": request.options.get("analyze_pii", True),
                "analyzePolicy": request.options.get("analyze_policy", True)
            }
            
            # Use the compliance guard to validate the LLM output
            # This uses the same method as for prompts, as the analysis logic is identical
            logger.info(f"Analyzing LLM output with options: {analysis_options}")
            validation_results = compliance_guard.validate_prompt(request.text, analysis_options)
            
            # Convert the results to a format compatible with the API
            response = {
                "token_risks": validation_results.get("token_risks", []),
                "policy_matches": validation_results.get("policy_matches", []),
                "overall_risk": validation_results.get("overall_risk", {
                    "score": 0.0,
                    "categories": {}
                }),
                "recommendations": validation_results.get("recommendations", [])
            }
            
            logger.info(f"Output analysis completed with risk score: {response['overall_risk']['score']}")
            
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
        # Map frontend options to backend options
        analysis_options = {
            "analyzeBias": request.options.get("analyze_bias", True),
            "analyzePII": request.options.get("analyze_pii", True),
            "analyzePolicy": request.options.get("analyze_policy", True)
        }
        
        # Use the compliance guard to validate the prompt
        logger.info(f"Analyzing prompt via HTTP with options: {analysis_options}")
        validation_results = compliance_guard.validate_prompt(request.text, analysis_options)
        
        # Convert the results to a format compatible with the API
        response = {
            "token_risks": validation_results.get("token_risks", []),
            "policy_matches": validation_results.get("policy_matches", []),
            "overall_risk": validation_results.get("overall_risk", {
                "score": 0.0,
                "categories": {}
            }),
            "recommendations": validation_results.get("recommendations", [])
        }
        
        logger.info(f"HTTP analysis completed with risk score: {response['overall_risk']['score']}")
        return response
    
    except Exception as e:
        logger.error(f"Error in analyze_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/output", response_model=OutputAnalysisResponse)
async def analyze_output(
    request: OutputAnalyzeRequest, 
    db: Session = Depends(get_db)
):
    """HTTP endpoint for LLM output analysis (non-WebSocket version)."""
    try:
        # Map frontend options to backend options
        analysis_options = {
            "analyzeBias": request.options.get("analyze_bias", True),
            "analyzePII": request.options.get("analyze_pii", True),
            "analyzePolicy": request.options.get("analyze_policy", True)
        }
        
        # Use the compliance guard to validate the LLM output
        logger.info(f"Analyzing LLM output via HTTP with options: {analysis_options}")
        validation_results = compliance_guard.validate_prompt(request.text, analysis_options)
        
        # Convert the results to a format compatible with the API
        response = {
            "token_risks": validation_results.get("token_risks", []),
            "policy_matches": validation_results.get("policy_matches", []),
            "overall_risk": validation_results.get("overall_risk", {
                "score": 0.0,
                "categories": {}
            }),
            "recommendations": validation_results.get("recommendations", [])
        }
        
        logger.info(f"HTTP output analysis completed with risk score: {response['overall_risk']['score']}")
        return response
    
    except Exception as e:
        logger.error(f"Error in analyze_output: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
