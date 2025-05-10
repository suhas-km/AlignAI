from typing import List, Dict, Any, Optional
import logging
from statistics import mean

logger = logging.getLogger(__name__)

class ComplianceGuard:
    """Orchestrates the compliance checking process by integrating results from various detectors."""
    
    def __init__(self):
        """Initialize the compliance guard."""
        pass
    
    def validate_prompt(self, prompt: str, options: Dict[str, bool] = None) -> Dict[str, Any]:
        """Validate prompt against compliance rules.
        
        This is a placeholder implementation. In a full implementation, this would:
        1. Call the text embedding service to generate embeddings
        2. Use the embeddings to find relevant policies
        3. Call bias and PII detection
        4. Aggregate the results
        
        Args:
            prompt: Input prompt text to validate
            options: Configuration options for validation
            
        Returns:
            Dict containing validation results
        """
        # This would be implemented in the API route
        return {
            "valid": True,
            "token_risks": [],
            "policy_matches": [],
            "overall_risk": {
                "score": 0.0,
                "categories": {}
            }
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
