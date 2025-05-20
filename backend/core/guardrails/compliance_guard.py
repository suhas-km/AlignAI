from typing import List, Dict, Any, Optional
import logging
import json
from statistics import mean
from pathlib import Path

# Import ML-based detection engines
from ..bias_detection.bias_detector import BiasDetectionEngine
from ..pii_detection.pii_detector import PIIDetectionEngine
from ..policy_detection.policy_detector import PolicyDetectionEngine

logger = logging.getLogger(__name__)

class ComplianceGuard:
    """Orchestrates the compliance checking process by integrating results from various detectors."""
    
    def __init__(self, use_ml: bool = True):
        """Initialize the compliance guard with optional ML-based detection.
        
        Args:
            use_ml: Whether to use ML-based detectors when available.
        """
        self.use_ml = use_ml
        
        # Initialize detectors
        models_dir = Path(__file__).parent.parent.parent / "models"
        
        # Initialize bias detection engine
        self.bias_detector = BiasDetectionEngine(use_ml=use_ml, 
                                              model_dir=str(models_dir / "bias_detection") if models_dir.exists() else None)
        
        # Initialize PII detection engine
        self.pii_detector = PIIDetectionEngine(use_ml=use_ml,
                                             model_dir=str(models_dir / "pii_detection") if models_dir.exists() else None)
        
        # Initialize policy detection engine
        self.policy_detector = PolicyDetectionEngine(use_ml=use_ml,
                                                  model_dir=str(models_dir / "policy_detection") if models_dir.exists() else None)
        
        logger.info("ComplianceGuard initialized with ML-based detection: %s", use_ml)
    
    def validate_prompt(self, prompt: str, options: Dict[str, bool] = None) -> Dict[str, Any]:
        """Validate prompt against compliance rules.
        
        This comprehensive method:
        1. Runs bias detection to identify potential biases in text
        2. Performs PII detection to identify personal information
        3. Conducts policy compliance checking against EU AI Act requirements
        4. Aggregates and scores all results
        
        Args:
            prompt: The prompt text to validate
            options: Analysis options (analyzeBias, analyzePII, analyzePolicy)
            
        Returns:
            Dict containing validation results including:
            - valid: bool indicating if prompt meets compliance requirements
            - token_risks: List of detected risks in the prompt
            - policy_matches: List of matched policies
            - overall_risk: Risk assessment with score and categories
            - recommendations: List of improvement suggestions
        """
        logger.info("=== Starting prompt validation ===")
        logger.info(f"Options received: {options}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        if not prompt or not prompt.strip():
            logger.warning("Empty or whitespace-only prompt provided")
            return {
                "valid": True,
                "token_risks": [],
                "policy_matches": [],
                "overall_risk": {
                    "score": 0.0,
                    "categories": {}
                },
                "recommendations": []
            }
        
        if options is None:
            options = {
                "analyzeBias": True,
                "analyzePII": True,
                "analyzePolicy": True
            }
            logger.info("Using default analysis options")
        
        token_risks = []
        policy_matches = []
        
        try:
            # Detect bias if enabled
            if options.get("analyzeBias", True):
                logger.info("Starting bias detection...")
                try:
                    bias_results = self.bias_detector.detect_bias(prompt)
                    logger.info(f"Bias detection completed. Found {len(bias_results)} potential issues")
                    if logger.isEnabledFor(logging.DEBUG):
                        for i, issue in enumerate(bias_results[:5]):  # Log first 5 issues to avoid log spam
                            logger.debug(f"Bias issue {i+1}: {issue}")
                    token_risks.extend(bias_results)
                except Exception as e:
                    logger.error(f"Error in bias detection: {str(e)}", exc_info=True)
                    raise
            else:
                logger.info("Bias detection skipped as per options")
            
            # Detect PII if enabled
            if options.get("analyzePII", True):
                logger.info("Starting PII detection...")
                try:
                    pii_results = self.pii_detector.detect_pii(prompt)
                    logger.info(f"PII detection completed. Found {len(pii_results)} potential PII instances")
                    if logger.isEnabledFor(logging.DEBUG):
                        for i, pii in enumerate(pii_results[:5]):  # Log first 5 PII instances
                            logger.debug(f"PII detected {i+1}: {pii}")
                    token_risks.extend(pii_results)
                except Exception as e:
                    logger.error(f"Error in PII detection: {str(e)}", exc_info=True)
                    raise
            else:
                logger.info("PII detection skipped as per options")
            
            # Check policy compliance if enabled
            if options.get("analyzePolicy", True):
                logger.info("Starting policy compliance check...")
                try:
                    policy_results = self.policy_detector.detect_policy_violations(prompt)
                    logger.info(f"Policy check completed. Found {len(policy_results)} potential policy violations")
                    if logger.isEnabledFor(logging.DEBUG) and policy_results:
                        for i, violation in enumerate(policy_results[:3]):  # Log first 3 violations
                            logger.debug(f"Policy violation {i+1}: {violation.get('policy_id', 'N/A')} - {violation.get('text_snippet', '')[:100]}...")
                    policy_matches.extend(policy_results)
                except Exception as e:
                    logger.error(f"Error in policy compliance check: {str(e)}", exc_info=True)
                    raise
            else:
                logger.info("Policy compliance check skipped as per options")
            
            # Calculate overall risk
            logger.info("Calculating overall risk...")
            overall_risk = self.calculate_overall_risk(token_risks, policy_matches)
            logger.info(f"Overall risk score: {overall_risk.get('score', 0.0):.2f}")
            
            # Generate recommendations
            logger.info("Generating recommendations...")
            recommendations = self.generate_recommendations(token_risks, policy_matches)
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            # Determine if valid based on risk threshold
            is_valid = overall_risk["score"] < 0.8  # High threshold for now
            logger.info(f"Validation result: {'VALID' if is_valid else 'INVALID'}")
            
            result = {
                "valid": is_valid,
                "token_risks": token_risks,
                "policy_matches": policy_matches,
                "overall_risk": overall_risk,
                "recommendations": recommendations
            }
            return result
            
        except Exception as e:
            logger.error(f"Error validating prompt: {str(e)}")
            return {
                "valid": False,
                "token_risks": [],
                "policy_matches": [],
                "overall_risk": {
                    "score": 0.0,
                    "categories": {}
                },
                "recommendations": ["An error occurred during validation. Please try again."]
            }
    
    def calculate_overall_risk(
        self, 
        token_risks: List[Dict[str, Any]], 
        policy_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall risk score from individual risk components.
        
        Args:
            token_risks: List of token-level risks from bias/PII detection
            policy_matches: List of matched policies from semantic search
        
        Returns:
            Dict containing overall risk assessment
        """
        try:
            # Group risks by type
            risk_by_type = {}
            for risk in token_risks:
                risk_type = risk.get("risk_type", "unknown")
                if risk_type not in risk_by_type:
                    risk_by_type[risk_type] = []
                risk_by_type[risk_type].append(risk.get("risk_score", 0.0))
            
            # Calculate average risk score by type
            category_scores = {}
            for risk_type, scores in risk_by_type.items():
                if scores:
                    category_scores[risk_type] = mean(scores)
            
            # Add policy match risk (if any)
            if policy_matches:
                policy_scores = [match.get("similarity_score", 0.0) for match in policy_matches]
                if policy_scores:
                    category_scores["policy_violation"] = mean(policy_scores)
            
            # Calculate overall risk score (average of category scores)
            overall_score = mean(category_scores.values()) if category_scores else 0.0
            
            return {
                "score": overall_score,
                "categories": category_scores
            }
        except Exception as e:
            logger.error(f"Error calculating overall risk: {str(e)}")
            return {
                "score": 0.0,
                "categories": {}
            }
    
    def generate_recommendations(
        self, 
        token_risks: List[Dict[str, Any]], 
        policy_matches: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on detected risks.
        
        Args:
            token_risks: List of token-level risks from bias/PII detection
            policy_matches: List of matched policies from semantic search
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            # Group risks by type
            bias_risks = [r for r in token_risks if r.get("risk_type") == "bias"]
            pii_risks = [r for r in token_risks if r.get("risk_type") == "pii"]
            
            # Generate bias recommendations
            if bias_risks:
                bias_types = set(r.get("bias_type", "unknown") for r in bias_risks)
                for bias_type in bias_types:
                    if bias_type == "gender_bias":
                        recommendations.append("Consider using gender-neutral language to avoid stereotypes.")
                    elif bias_type == "racial_bias":
                        recommendations.append("Avoid generalizations based on race or ethnicity.")
                    elif bias_type == "age_bias":
                        recommendations.append("Avoid stereotyping based on age.")
                    else:
                        recommendations.append(f"Consider revising text to address potential {bias_type}.")
            
            # Generate PII recommendations
            if pii_risks:
                pii_types = set(r.get("pii_type", "unknown") for r in pii_risks)
                pii_rec = "Remove or anonymize personally identifiable information"
                if len(pii_types) == 1:
                    pii_rec += f" ({list(pii_types)[0].lower()})"
                elif len(pii_types) > 1:
                    pii_rec += f" ({', '.join(t.lower() for t in list(pii_types)[:3])})"
                    if len(pii_types) > 3:
                        pii_rec += ", and others"
                pii_rec += "."
                recommendations.append(pii_rec)
            
            # Generate policy recommendations
            if policy_matches:
                articles = set(match.get("article", "") for match in policy_matches)
                if articles:
                    articles_str = ", ".join(sorted(articles))
                    recommendations.append(f"Review compliance with EU AI Act {articles_str} to ensure alignment with regulations.")
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review the prompt for potential compliance issues."]
