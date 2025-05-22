#!/usr/bin/env python3
"""
Enhanced Policy Data Generator Script

Generates a comprehensive, high-quality dataset for policy violation detection
using Ollama's Qwen3:8b model. The dataset includes a balanced distribution of
policy categories with detailed metadata for better model training.
"""

import json
import random
import time
import re
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "data2"
RANDOM_SEED = 42
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"  # Default model, can be changed at runtime
NUM_EXAMPLES_PER_CATEGORY = 20  # Number of examples to generate per category
TOTAL_NON_VIOLATIONS = 100      # Number of non-violation examples

# EU AI Act policy categories with article references
POLICY_CATEGORIES = {
    "prohibited_practices": {
        "name": "Prohibited AI Practices",
        "articles": ["Article 5"],
        "description": "Systems that manipulate, exploit vulnerabilities, social scoring, or remote biometric identification"
    },
    "high_risk_systems": {
        "name": "High-Risk AI Systems",
        "articles": ["Article 6", "Article 7"],
        "description": "Systems posing significant risks to health, safety, or fundamental rights"
    },
    "transparency": {
        "name": "Transparency Requirements",
        "articles": ["Article 13", "Article 52"],
        "description": "Disclosure of AI nature, capabilities, limitations, and intended purpose"
    },
    "human_oversight": {
        "name": "Human Oversight",
        "articles": ["Article 14", "Article 29"],
        "description": "Human monitoring and ability to override AI decisions"
    },
    "data_governance": {
        "name": "Data and Data Governance",
        "articles": ["Article 10"],
        "description": "Data quality, relevance, representativeness, and processing requirements"
    },
    "risk_management": {
        "name": "Risk Management System",
        "articles": ["Article 9"],
        "description": "Process to identify, analyze, and mitigate risks"
    },
    "technical_robustness": {
        "name": "Technical Robustness and Safety",
        "articles": ["Article 15"],
        "description": "Accuracy, resilience to errors, cybersecurity, and backup plans"
    },
    "record_keeping": {
        "name": "Record-Keeping",
        "articles": ["Article 12"],
        "description": "Automatic recording of events and human oversight measures"
    },
    "accuracy_robustness": {
        "name": "Accuracy, Robustness, and Cybersecurity",
        "articles": ["Article 15"],
        "description": "Appropriate level of accuracy, resilience to errors, and security"
    }
}

# Severity levels with descriptions
SEVERITY_LEVELS = {
    "critical": "Severe violation that could cause significant harm or is explicitly prohibited",
    "high": "Major violation with potential for harm or serious non-compliance",
    "medium": "Moderate violation requiring substantial remediation",
    "low": "Minor violation requiring remediation but with limited impact"
}

# Industry contexts for varied examples
INDUSTRY_CONTEXTS = [
    "healthcare", "finance", "education", "hiring", "law enforcement",
    "social media", "e-commerce", "manufacturing", "transportation",
    "smart city", "insurance", "legal services", "customer service"
]

def generate_prompt(category: str, is_violation: bool, context: str) -> str:
    """Generate prompt for Ollama to create policy examples."""
    category_info = POLICY_CATEGORIES.get(category, {})
    articles = category_info.get("articles", [])
    article_str = ", ".join(articles) if articles else "relevant articles"
    
    violation_type = "violation" if is_violation else "compliant implementation"
    severity_options = ", ".join(SEVERITY_LEVELS.keys())
    
    # Modified prompt with clearer instructions and formatting
    prompt = f"""You are assisting with EU AI Act compliance. Your task is to generate a realistic {violation_type} example.

Context: {context}
Category: {category_info.get('name', category)}
Relevant Articles: {article_str}

Create a specific example of an AI system or practice that {'violates' if is_violation else 'complies with'} the EU AI Act regulations.

Provide your response in this exact JSON format without any additional text:

{{
  "statement": "A detailed description of the AI system or practice",
  "explanation": "Explanation of why this {'violates' if is_violation else 'complies with'} the EU AI Act",
  "articles": "{article_str}",
  "severity": "{severity_options if is_violation else 'none'}",
  "context": "{context}"
}}

Remember: Only return the JSON with no other text."""
    
    return prompt

def call_ollama(prompt: str, temp: float = 0.7, max_retries: int = 3) -> Optional[str]:
    """Call Ollama API with retries and error handling."""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL,  # Use the global model name
        "prompt": prompt,
        "stream": False,  # Get complete response, not streaming
        "options": {
            "temperature": temp,
            "num_predict": 1024  # Get a longer response to ensure complete JSON
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API, headers=headers, json=data)
            if response.status_code == 200:
                # Extract the response content from the JSON
                try:
                    resp_json = json.loads(response.text)
                    # Ollama API returns the generated text in the 'response' field
                    if 'response' in resp_json:
                        # Remove <think> tags if present
                        response_text = resp_json['response']
                        response_text = re.sub(r'</?think>', '', response_text)
                        return response_text
                    else:
                        logger.warning(f"Unexpected response format: {response.text[:100]}...")
                        return response.text
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse API response as JSON: {response.text[:100]}...")
                    return response.text
            else:
                logger.warning(f"Attempt {attempt+1} failed with status {response.status_code}")
                logger.warning(f"Response: {response.text[:200]}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
        
        # Exponential backoff
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    logger.error(f"Failed to get response after {max_retries} attempts")
    return None

def parse_ollama_response(response_text: str) -> Optional[Dict]:
    """Parse Ollama response to extract JSON data."""
    if not response_text:
        return None
    
    # Log the raw response for debugging
    logger.debug(f"Raw response: {response_text[:500]}...")
    
    # Method 1: Try to find complete JSON block using regex
    json_patterns = [
        r'\{\s*"statement".*?\}',  # Original pattern
        r'```json\s*(\{.*?\})\s*```',  # JSON in code block
        r'```\s*(\{.*?\})\s*```',  # JSON in generic code block
        r'\{\s*["\']statement["\'].*?\}'  # More flexible pattern
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                if '(' in pattern and ')' in pattern:  # Pattern has capture group
                    json_str = match.group(1)
                else:
                    json_str = match.group(0)
                
                # Clean up any markdown formatting
                json_str = re.sub(r'```json|```', '', json_str)
                parsed = json.loads(json_str)
                logger.info("Successfully parsed JSON using regex pattern")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON match: {e}")
    
    # Method 2: Look for lines that might contain JSON and try to parse them
    lines = response_text.split('\n')
    for i, line in enumerate(lines):
        if '{' in line and '}' in line and 'statement' in line.lower():
            # Try to find JSON in this line and surrounding lines
            for window_size in range(1, 20):  # Try increasingly larger windows
                start = max(0, i - window_size)
                end = min(len(lines), i + window_size + 1)
                candidate = '\n'.join(lines[start:end])
                
                # Extract potential JSON content
                start_idx = candidate.find('{')
                end_idx = candidate.rfind('}')
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_candidate = candidate[start_idx:end_idx+1]
                    try:
                        parsed = json.loads(json_candidate)
                        if 'statement' in parsed:
                            logger.info("Successfully parsed JSON using line extraction method")
                            return parsed
                    except json.JSONDecodeError:
                        pass  # Try next window size
    
    # Method 3: Last resort - build a JSON object from key-value patterns in the text
    if '"statement"' in response_text or "'statement'" in response_text:
        try:
            # Create a new JSON object manually from extracted fields
            statement_match = re.search(r'["\']statement["\']\s*:\s*["\']([^"\']*)["\'\n]', response_text, re.DOTALL)
            explanation_match = re.search(r'["\']explanation["\']\s*:\s*["\']([^"\']*)["\'\n]', response_text, re.DOTALL)
            articles_match = re.search(r'["\']articles["\']\s*:\s*["\']([^"\']*)["\'\n]', response_text, re.DOTALL)
            severity_match = re.search(r'["\']severity["\']\s*:\s*["\']([^"\']*)["\'\n]', response_text, re.DOTALL)
            context_match = re.search(r'["\']context["\']\s*:\s*["\']([^"\']*)["\'\n]', response_text, re.DOTALL)
            
            result = {}
            if statement_match:
                result["statement"] = statement_match.group(1).strip()
            if explanation_match:
                result["explanation"] = explanation_match.group(1).strip()
            if articles_match:
                result["articles"] = articles_match.group(1).strip()
            if severity_match:
                result["severity"] = severity_match.group(1).strip()
            if context_match:
                result["context"] = context_match.group(1).strip()
            
            if "statement" in result:
                logger.info("Successfully parsed JSON using key-value extraction method")
                return result
        except Exception as e:
            logger.warning(f"Failed to extract fields: {e}")
    
    logger.warning("No JSON found in response after trying multiple methods")
    return None

def generate_example(category: str, is_violation: bool) -> Optional[Dict]:
    """Generate a single example using Ollama."""
    context = random.choice(INDUSTRY_CONTEXTS)
    prompt = generate_prompt(category, is_violation, context)
    
    response_text = call_ollama(prompt)
    if not response_text:
        return None
    
    parsed_data = parse_ollama_response(response_text)
    if not parsed_data:
        return None
    
    # Ensure articles is always a list
    articles = parsed_data.get("articles", [])
    if isinstance(articles, str):
        # Convert comma-separated string to list
        articles = [art.strip() for art in articles.split(',')]
    
    # Format the result
    result = {
        "text": parsed_data.get("statement", ""),
        "violation": is_violation,
        "category": category,  # Keep category for both violations and non-violations
        "severity": parsed_data.get("severity", "none") if is_violation else "none",
        "articles": articles,
        "explanation": parsed_data.get("explanation", ""),
        "context": parsed_data.get("context", context)
    }
    
    return result

def save_interim_results(dataset: List[Dict[str, Any]], suffix: str = ""):
    """Save interim results to temporary files."""
    if not dataset:
        return
        
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save all current examples to an interim file
    timestamp = time.strftime("%H%M%S")
    interim_file = OUTPUT_DIR / f"interim{suffix}_{timestamp}.jsonl"
    
    with open(interim_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved {len(dataset)} interim examples to {interim_file}")
    
    # Also update the main files with the current data
    # This is a simple split without stratification for interim results
    n = len(dataset)
    train_end = int(n * 0.7)
    dev_end = train_end + int(n * 0.15)
    
    splits = {
        "train": dataset[:train_end],
        "dev": dataset[train_end:dev_end],
        "test": dataset[dev_end:]
    }
    
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        output_file = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Updated {split_name}.jsonl with {len(split_data)} examples")

def generate_dataset() -> List[Dict[str, Any]]:
    """Generate a comprehensive dataset of policy examples."""
    dataset = []
    target_counts = {}
    last_save_time = time.time()
    save_interval = 300  # Save every 5 minutes
    
    # Initialize target counts for each category
    for category in POLICY_CATEGORIES.keys():
        target_counts[category] = NUM_EXAMPLES_PER_CATEGORY
    
    # Generate violation examples for each category
    for category_idx, (category, target_count) in enumerate(target_counts.items()):
        logger.info(f"Generating {target_count} examples for category: {category} ({category_idx+1}/{len(target_counts)})")
        count = 0
        attempts = 0
        max_attempts = target_count * 2  # Allow for some failures
        
        while count < target_count and attempts < max_attempts:
            example = generate_example(category, is_violation=True)
            if example and len(example["text"]) > 20:  # Ensure meaningful content
                dataset.append(example)
                count += 1
                logger.info(f"Generated example {count}/{target_count} for {category}")
                
                # Check if it's time to save interim results
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    save_interim_results(dataset, f"_cat{category_idx+1}of{len(target_counts)}")
                    last_save_time = current_time
            attempts += 1
            # Small delay to avoid overloading Ollama
            time.sleep(0.5)
        
        # Save after each category is completed
        save_interim_results(dataset, f"_completed_{category}")
    
    # Generate non-violation examples with balanced categories
    logger.info(f"Generating {TOTAL_NON_VIOLATIONS} non-violation examples")
    count = 0
    attempts = 0
    max_attempts = TOTAL_NON_VIOLATIONS * 3
    
    # Distribute non-violations evenly across categories
    non_violations_per_category = TOTAL_NON_VIOLATIONS // len(POLICY_CATEGORIES)
    remaining = TOTAL_NON_VIOLATIONS % len(POLICY_CATEGORIES)
    
    for category_idx, category in enumerate(POLICY_CATEGORIES.keys()):
        target_count = non_violations_per_category + (1 if remaining > 0 else 0)
        if remaining > 0:
            remaining -= 1
            
        cat_count = 0
        cat_attempts = 0
        max_cat_attempts = target_count * 3
        
        logger.info(f"Generating {target_count} non-violation examples for category: {category}")
        while cat_count < target_count and cat_attempts < max_cat_attempts and count < TOTAL_NON_VIOLATIONS:
            example = generate_example(category, is_violation=False)
            if example and len(example["text"]) > 20:  # Ensure meaningful content
                # Keep the category information for non-violations
                example["category"] = category  # Important: Keep the category, just mark as non-violation
                dataset.append(example)
                cat_count += 1
                count += 1
                logger.info(f"Generated non-violation example for {category}: {cat_count}/{target_count}")
                
                # Check if it's time to save interim results
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    save_interim_results(dataset, f"_nonviol{count}of{TOTAL_NON_VIOLATIONS}")
                    last_save_time = current_time
            cat_attempts += 1
            attempts += 1
            # Small delay to avoid overloading Ollama
            time.sleep(0.5)
            
        if count >= TOTAL_NON_VIOLATIONS:
            break
            
        # Save after each category of non-violations is completed
        save_interim_results(dataset, f"_nonviol_{category}")
    
    return dataset

def deduplicate_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate examples from the dataset based on text and category."""
    unique_examples = {}
    duplicate_count = 0
    
    for example in dataset:
        # Create a hash key using text and category to identify duplicates
        # Normalize text by removing whitespace and lowercasing
        text = re.sub(r'\s+', ' ', example["text"]).strip().lower()
        key = f"{text}_{example['category']}_{example['violation']}"
        
        if key not in unique_examples:
            unique_examples[key] = example
        else:
            duplicate_count += 1
    
    logger.info(f"Removed {duplicate_count} duplicate examples")
    return list(unique_examples.values())

def validate_example(example: Dict[str, Any]) -> bool:
    """Validate that an example has all required fields and proper format."""
    required_fields = ["text", "violation", "category", "severity", "articles", "explanation"]
    
    # Check for required fields
    for field in required_fields:
        if field not in example:
            logger.warning(f"Example missing required field: {field}")
            return False
    
    # Validate field types
    if not isinstance(example["text"], str) or len(example["text"]) < 10:
        logger.warning(f"Invalid text field: {example['text'][:30]}...")
        return False
    
    # Allow both boolean and string "borderline" for violation field
    if not (isinstance(example["violation"], bool) or example["violation"] == "borderline"):
        logger.warning(f"Invalid violation field: {example['violation']}")
        return False
    
    if not isinstance(example["category"], str) or not example["category"]:
        logger.warning(f"Invalid category field: {example['category']}")
        return False
    
    # More validation could be added here
    
    return True

def generate_borderline_examples(n: int = 20) -> List[Dict[str, Any]]:
    """Generate borderline/ambiguous examples that fall into gray areas."""
    logger.info(f"Generating {n} borderline/ambiguous examples")
    borderline_examples = []
    attempts = 0
    max_attempts = n * 3
    last_save_time = time.time()
    save_interval = 300  # Save every 5 minutes
    
    # Prompt template for borderline cases
    borderline_prompt = """You are assisting with EU AI Act compliance. Your task is to generate a realistic ambiguous example.

Context: {context}
Category: {category_name}

Create a specific example of an AI system or practice that falls into a gray area - where experts might disagree about whether it violates the EU AI Act regulations.

Provide your response in this exact JSON format without any additional text:

{{
  "statement": "A detailed description of the ambiguous AI system or practice",
  "explanation": "Explanation of why this case is ambiguous under the EU AI Act",
  "articles": "Relevant EU AI Act articles",
  "severity": "borderline",
  "context": "{context}"
}}

Remember: Only return the JSON with no other text."""
    
    while len(borderline_examples) < n and attempts < max_attempts:
        # Select a random category and context
        category = random.choice(list(POLICY_CATEGORIES.keys()))
        category_info = POLICY_CATEGORIES[category]
        context = random.choice(INDUSTRY_CONTEXTS)
        
        prompt = borderline_prompt.format(
            category_name=category_info["name"],
            context=context
        )
        
        response_text = call_ollama(prompt, temp=0.8)  # Higher temperature for more creative examples
        if not response_text:
            attempts += 1
            continue
        
        parsed_data = parse_ollama_response(response_text)
        if not parsed_data:
            attempts += 1
            continue
        
        example = {
            "text": parsed_data.get("statement", ""),
            "violation": "borderline",  # Special value for borderline cases
            "category": category,
            "severity": "borderline",
            "articles": parsed_data.get("articles", []),
            "explanation": parsed_data.get("explanation", ""),
            "context": parsed_data.get("context", context),
            "ambiguity": True  # Flag to identify borderline examples
        }
        
        if validate_example(example) and len(example["text"]) > 20:
            borderline_examples.append(example)
            logger.info(f"Generated borderline example {len(borderline_examples)}/{n}")
            
            # Check if it's time to save interim results
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                # Just save the borderline examples collected so far
                save_interim_results(borderline_examples, f"_borderline{len(borderline_examples)}of{n}")
                last_save_time = current_time
        
        attempts += 1
        time.sleep(0.5)
    
    # Save the final borderline examples
    save_interim_results(borderline_examples, f"_borderline{n}")
    return borderline_examples

def save_dataset(dataset: List[Dict[str, Any]], split_ratios: tuple = (0.7, 0.15, 0.15)):
    """Save dataset to train, dev, and test files with stratified sampling."""
    # Validate and deduplicate the dataset
    logger.info("Validating examples...")
    valid_dataset = [example for example in dataset if validate_example(example)]
    logger.info(f"Removed {len(dataset) - len(valid_dataset)} invalid examples")
    
    logger.info("Deduplicating dataset...")
    deduplicated_dataset = deduplicate_dataset(valid_dataset)
    logger.info(f"Final dataset size after validation and deduplication: {len(deduplicated_dataset)}")
    
    # Group examples by category for stratified sampling
    categorized = {}
    for item in deduplicated_dataset:
        cat = item["category"]
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(item)
    
    # Create stratified splits
    splits = {"train": [], "dev": [], "test": []}
    
    for cat, items in categorized.items():
        random.shuffle(items)
        n = len(items)
        train_end = int(n * split_ratios[0])
        dev_end = train_end + int(n * split_ratios[1])
        
        splits["train"].extend(items[:train_end])
        splits["dev"].extend(items[train_end:dev_end])
        splits["test"].extend(items[dev_end:])
    
    # Shuffle again to mix categories
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    # Save to files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(split_data)} examples to {output_file}")

def try_ollama_model(model_name: str) -> bool:
    """Test if a specific Ollama model is available and working."""
    # Test if a specific Ollama model is available and working
    test_prompt = "Return this exact text: {\"test\": \"successful\"}. Only return this JSON."
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": test_prompt,
        "stream": False,
        "options": {"num_predict": 100}  # Short response is enough for test
    }
    
    try:
        response = requests.post(OLLAMA_API, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            logger.info(f"Model {model_name} is available")
            return True
        else:
            logger.warning(f"Model {model_name} returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Failed to test model {model_name}: {e}")
        return False

def generate_backup_dataset() -> List[Dict[str, Any]]:
    """Generate a small backup dataset in case Ollama is not available."""
    logger.warning("Using backup dataset generation method")
    dataset = []
    
    # Sample violation examples for different categories
    violations = [
        {
            "text": "Our facial recognition system is deployed in public spaces without notifying individuals.",
            "category": "prohibited_practices",
            "severity": "critical",
            "articles": ["Article 5"],
            "explanation": "Remote biometric identification systems in publicly accessible spaces without proper notification violate Article 5.",
            "context": "smart city"
        },
        {
            "text": "Our AI automatically rejects loan applications from specific neighborhoods without human review.",
            "category": "human_oversight",
            "severity": "high",
            "articles": ["Article 14", "Article 29"],
            "explanation": "Lack of human oversight in high-risk financial decisions violates Article 14.",
            "context": "finance"
        },
        {
            "text": "We use unaudited, biased datasets for training our hiring algorithm.",
            "category": "data_governance",
            "severity": "high",
            "articles": ["Article 10"],
            "explanation": "Using poor quality, biased data violates data governance requirements in Article 10.",
            "context": "hiring"
        },
        {
            "text": "Our medical diagnostic AI doesn't log its decision-making process for later review.",
            "category": "record_keeping",
            "severity": "medium",
            "articles": ["Article 12"],
            "explanation": "Failure to maintain proper logs of AI decisions violates record-keeping requirements in Article 12.",
            "context": "healthcare"
        },
        {
            "text": "Our AI system provides no explanation for why it flagged a transaction as fraudulent.",
            "category": "transparency",
            "severity": "medium",
            "articles": ["Article 13"],
            "explanation": "Lack of transparency about decision-making violates Article 13 requirements.",
            "context": "finance"
        }
    ]
    
    # Add each violation and generate variations
    for violation in violations:
        dataset.append(violation)
        
        # Add variations with the same structure but slight differences
        base_text = violation["text"]
        category = violation["category"]
        context = violation["context"]
        
        variations = [
            f"In our {context} system, {base_text.lower()}",
            f"The {context} AI we deployed {base_text.lower().replace('our', 'the')}",
            f"Despite regulations, {base_text.lower().replace('our', 'the')}"
        ]
        
        for var_text in variations:
            var = violation.copy()
            var["text"] = var_text
            dataset.append(var)
    
    # Add non-violation examples
    non_violations = [
        "Our AI system includes complete documentation explaining its capabilities and limitations.",
        "All decisions from our high-risk AI system require human review and confirmation.",
        "We thoroughly test our data for bias before training our models.",
        "Our system maintains comprehensive logs of all decisions for audit purposes.",
        "Users are clearly informed when they are interacting with an AI system."
    ]
    
    for text in non_violations:
        context = random.choice(INDUSTRY_CONTEXTS)
        dataset.append({
            "text": text,
            "violation": False,
            "category": "none",
            "severity": "none",
            "articles": [],
            "explanation": "This implementation complies with EU AI Act requirements.",
            "context": context
        })
    
    return dataset

def main():
    """Main function to generate and save the dataset."""
    random.seed(RANDOM_SEED)
    logger.info("Starting dataset generation process")
    
    try:
        # Check if Ollama is available
        logger.info("Testing Ollama connection...")
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            # Test if the specified model is available
            model_available = try_ollama_model("qwen3:8b")
            
            if model_available:
                logger.info("Qwen3:8b model is available. Generating dataset.")
                dataset = generate_dataset()
                
                # Add borderline examples if Ollama is available
                logger.info("Generating borderline examples...")
                borderline_examples = generate_borderline_examples(20)  # Generate 20 borderline examples
                if borderline_examples:
                    dataset.extend(borderline_examples)
                    logger.info(f"Added {len(borderline_examples)} borderline examples")
            else:
                # Try to find other available models
                logger.warning("Qwen3:8b model not responding correctly. Checking for other models...")
                alternative_models = ["llama3", "mistral", "openhermes", "gemma"]
                for model in alternative_models:
                    if try_ollama_model(model):
                        logger.info(f"Using alternative model: {model}")
                        # Update the global model name
                        global OLLAMA_MODEL
                        OLLAMA_MODEL = model
                        dataset = generate_dataset()
                        break
                else:  # No models worked
                    logger.warning("No working Ollama models found. Using backup generation method.")
                    dataset = generate_backup_dataset()
        else:
            logger.warning("Ollama API returned non-200 status. Using backup generation method.")
            dataset = generate_backup_dataset()
    except requests.exceptions.RequestException:
        logger.warning("Could not connect to Ollama. Using backup generation method.")
        dataset = generate_backup_dataset()
    
    if dataset:
        save_dataset(dataset)
        logger.info(f"Generated dataset with {len(dataset)} total examples")
        
        # Print category distribution
        categories = {}
        violations = {"true": 0, "false": 0, "borderline": 0}
        
        for item in dataset:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count by violation type
            if item['violation'] == "borderline":
                violations["borderline"] += 1
            elif item['violation']:
                violations["true"] += 1
            else:
                violations["false"] += 1
        
        logger.info("Category distribution:")
        for cat, count in sorted(categories.items()):
            logger.info(f"- {cat}: {count} examples")
            
        logger.info("\nViolation type distribution:")
        for vtype, count in violations.items():
            logger.info(f"- {vtype}: {count} examples")
    else:
        logger.error("Failed to generate dataset")

if __name__ == "__main__":
    main()