#!/usr/bin/env python3
"""
Test Script for PII Detection Model
-----------------------------
This script tests the trained PII detection model on a variety of example texts.
"""

from pii_predictor import PIIPredictor

def main():
    # Create a PII predictor instance
    print("Loading PII detection model...")
    predictor = PIIPredictor()
    predictor.load_model()
    
    # Example texts to test
    test_texts = [
        "My name is John Smith and I live in New York City.",
        "Please contact me at john.smith@example.com or call 555-123-4567.",
        "My social security number is 123-45-6789 and my credit card is 4111-1111-1111-1111.",
        "The company Amazon is based in Seattle, Washington and was founded by Jeff Bezos.",
        "I was born on January 15, 1985 and my IP address is 192.168.1.1.",
        "Apple Inc. reported strong quarterly earnings with revenue of $89.5 billion.",
        "The event will take place at 123 Main Street, Boston, MA 02108 next Monday."
    ]
    
    # Test each text
    print("\n===== PII DETECTION RESULTS =====\n")
    for i, text in enumerate(test_texts):
        print(f"TEXT {i+1}: \"{text}\"")
        
        # Get comprehensive analysis
        results = predictor.analyze_text(text)
        
        # Print results
        print(f"PII Detected: {'Yes' if results['pii_detected'] else 'No'}")
        
        if results['pii_detected']:
            print(f"Number of PII entities: {results['pii_count']}")
            print("Detected entities:")
            for entity in results['entities']:
                print(f"  - '{entity['text']}' - Type: {entity['type']}")
            
            print(f"\nMasked text: {results['masked_text']}")
        
        if results["recommendations"]:
            print("\nRecommendations:")
            for rec in results["recommendations"]:
                print(f"  - {rec}")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
