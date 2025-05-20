from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field

from core.bias_detection import detect_bias
from core.pii_detection import detect_pii
from core.policy_detection import check_policy_violations

router = APIRouter(
    prefix="/analyze",
    tags=["analysis"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Request and Response Models
class AnalysisOptions(BaseModel):
    analyze_bias: bool = True
    analyze_pii: bool = True
    analyze_policy: bool = True
    language: Optional[str] = "en"
    threshold: float = Field(0.7, ge=0.0, le=1.0)

class AnalysisRequest(BaseModel):
    text: str = Field(..., description="The text to analyze")
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)

class AnalysisResponse(BaseModel):
    text: str = Field(..., description="The analyzed text")
    analysis: Dict[str, Any] = Field(..., description="Detailed analysis results")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    is_safe: bool = Field(True, description="Whether the text is considered safe")

@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(
    text: str = Body(..., embed=True, description="Text to analyze"),
    options: AnalysisOptions = Body(default_factory=AnalysisOptions)
) -> Dict[str, Any]:
    """
    Analyze text for bias, PII, and policy violations.
    
    This endpoint analyzes the provided text based on the specified options and returns
    detailed analysis results including any detected issues.
    
    Args:
        text: The text to analyze
        options: Analysis configuration options
        
    Returns:
        Analysis results including any warnings and safety status
    """
    try:
        analysis = {}
        warnings = []
        is_safe = True
        
        # Check for bias if enabled
        if options.analyze_bias:
            bias_result = detect_bias(text, threshold=options.threshold)
            analysis["bias"] = bias_result
            if bias_result.get("has_bias", False):
                warnings.append("Potential bias detected in text")
                is_safe = False
        
        # Check for PII if enabled
        if options.analyze_pii:
            pii_result = detect_pii(text, language=options.language)
            analysis["pii"] = pii_result
            if pii_result.get("has_pii", False):
                warnings.append("PII detected in text")
                is_safe = False
        
        # Check for policy violations if enabled
        if options.analyze_policy:
            policy_result = check_policy_violations(text)
            analysis["policy"] = policy_result
            if policy_result.get("has_violation", False):
                warnings.append("Policy violation detected")
                is_safe = False
        
        return AnalysisResponse(
            text=text,
            analysis=analysis,
            warnings=warnings,
            is_safe=is_safe
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing text: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for the analysis service.
    
    Returns:
        A simple status message indicating the service is running
    """
    return {"status": "ok"}
