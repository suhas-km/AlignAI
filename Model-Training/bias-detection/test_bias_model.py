#!/usr/bin/env python3
"""
Test Script for Bias Detection Model
-----------------------------------
This script tests the trained bias detection models on a few example prompts.
"""

from bias_predictor import BiasPredictor

def main():
    # Create a bias predictor instance
    print("Loading bias detection models...")
    predictor = BiasPredictor()
    predictor.load_models()
    
    # Example prompts to test
    test_prompts = [
        "Women will be invited only if they are young",
        "I need someone who is a native English speaker for this job",
        "We're looking for a programmer who is detail-oriented and logical",
        "Older workers tend to be less productive and adaptable",
        "You would expect a person from that religion to be more closed-minded",
        "People with disabilities require too many accommodations to be efficient workers",
        "Men are typically better at leadership roles"
    ]
    
    # Test each prompt
    print("\n===== BIAS DETECTION RESULTS =====\n")
    for i, prompt in enumerate(test_prompts):
        print(f"PROMPT {i+1}: \"{prompt}\"")
        
        # Get detailed analysis
        result = predictor.analyze_text(prompt)
        
        # Print results
        print(f"Bias Detected: {'Yes' if result['contains_bias'] else 'No'}")
        print(f"Bias Score: {result['bias_score']}%")
        
        if result["primary_bias_type"]:
            print(f"Primary Bias Type: {result['primary_bias_type']['category_name']}")
        
        if result["bias_categories"]:
            print("Bias Categories:")
            for category in result["bias_categories"]:
                print(f"  - {category['category_name']} ({category['score']}%)")
        
        if result["recommendations"]:
            print("Recommendations:")
            for rec in result["recommendations"]:
                print(f"  - {rec}")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
