#!/usr/bin/env python3
"""
Enhanced Q&A pair generator for the EU AI Act
---------------------------------------------
This improved script generates high-quality Q&A pairs from the EU AI Act 
using a two-stage approach and sophisticated chunking for better context retention.

Requirements:
  * Ollama server running locally (https://ollama.com/)
  * Python packages: ollama, tqdm, markdown, nltk

Input:
  euaiact.md - markdown version of the EU AI Act
Output:
  enhanced_qa.jsonl - JSON lines with question-answer pairs and metadata
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from markdown import markdown
from ollama import chat
from tqdm import tqdm

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration
INPUT_FILE = Path("euaiact.md")
OUTPUT_FILE = Path("output/enhanced_qa.jsonl")
PROGRESS_LOG = Path("enhanced_progress.log")
MODEL = "mistral:7b"  # Adjust based on available models
SLEEP_TIME = 1.0  # Seconds between API calls
MAX_ATTEMPTS = 3   # Maximum retry attempts for API calls

# Create and configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("eu_ai_act_qa")

# Create output directories
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# Question type templates to enhance diversity
QUESTION_TEMPLATES = {
    "definitional": [
        "What is defined as '{concept}' according to the EU AI Act?",
        "How does the EU AI Act define '{concept}'?",
        "What constitutes '{concept}' under the provisions of the EU AI Act?"
    ],
    "compliance": [
        "What obligations do organizations have regarding '{topic}' under the EU AI Act?",
        "How can entities ensure compliance with the EU AI Act's provisions on '{topic}'?",
        "What measures should be implemented to comply with requirements related to '{topic}'?"
    ],
    "prohibition": [
        "What restrictions does the EU AI Act place on '{practice}'?",
        "Under what circumstances is '{practice}' prohibited according to the EU AI Act?",
        "What limitations apply to '{practice}' as specified in the EU AI Act?"
    ],
    "comparative": [
        "What is the difference between '{concept1}' and '{concept2}' as defined in the EU AI Act?",
        "How does the EU AI Act distinguish between '{concept1}' and '{concept2}'?",
        "What are the key distinctions between '{concept1}' and '{concept2}' under the EU AI Act?"
    ],
    "scenario": [
        "How would the EU AI Act apply to a scenario where {scenario}?",
        "What provisions of the EU AI Act would be relevant if {scenario}?",
        "Under the EU AI Act, what requirements would apply if {scenario}?"
    ],
    "exceptions": [
        "What are the exceptions to rules regarding '{topic}' in the EU AI Act?",
        "Under what conditions might requirements for '{topic}' be waived according to the EU AI Act?",
        "What derogations exist for '{topic}' as provided in the EU AI Act?"
    ],
    "authority": [
        "What role do '{authority}' play in enforcing the EU AI Act?",
        "What powers are granted to '{authority}' under the EU AI Act?",
        "What responsibilities do '{authority}' have according to the EU AI Act?"
    ]
}

def clean_markdown(text: str) -> str:
    """Remove markdown formatting and clean the text."""
    # Convert markdown to HTML, then strip HTML tags
    html = markdown(text)
    text = re.sub('<[^<]+?>', '', html)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_section_hierarchy(md_text: str) -> List[Dict[str, Any]]:
    """
    Extract sections based on markdown headings to preserve hierarchical structure.
    This creates a more logical breakdown of the document.
    """
    # Regex for different heading levels
    heading_pattern = r'^(#{1,6})\s+(.+)$'
    
    lines = md_text.strip().split('\n')
    sections = []
    current_section = None
    current_level = 0
    section_text = []
    
    for line in lines:
        heading_match = re.match(heading_pattern, line)
        
        if heading_match:
            # Save previous section if it exists
            if current_section is not None and section_text:
                sections.append({
                    'title': current_section,
                    'level': current_level,
                    'text': '\n'.join(section_text).strip()
                })
            
            # Start new section
            current_level = len(heading_match.group(1))  # Number of # symbols
            current_section = heading_match.group(2)
            section_text = []
        else:
            if current_section is not None:
                section_text.append(line)
    
    # Add the last section
    if current_section is not None and section_text:
        sections.append({
            'title': current_section,
            'level': current_level,
            'text': '\n'.join(section_text).strip()
        })
    
    # Add section IDs and parent references to capture hierarchical relationships
    for i, section in enumerate(sections):
        section['id'] = i
        
        # Find parent section (closest previous section with lower level)
        parent_id = None
        for j in range(i-1, -1, -1):
            if sections[j]['level'] < section['level']:
                parent_id = sections[j]['id']
                break
        
        section['parent_id'] = parent_id
    
    return sections

def extract_important_concepts(section_text: str, num_concepts: int = 5) -> List[str]:
    """
    Extract important concepts from section text using a simple method.
    In a production environment, this would use more sophisticated NLP.
    """
    # Extract capitalized terms as potential concepts
    capitalized_terms = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', section_text)
    
    # Extract terms in quotes
    quoted_terms = re.findall(r'[\'"]([^\'"]+)[\'"]', section_text)
    
    # Extract terms after "such as" or "including"
    examples = []
    for phrase in re.finditer(r'(?:such as|including|e\.g\.,?|for example,?)\s+([^,.;]+)', section_text):
        examples.append(phrase.group(1).strip())
    
    # Combine all concepts and take the most frequently mentioned ones
    all_concepts = capitalized_terms + quoted_terms + examples
    
    # Count frequency
    concept_counts = {}
    for concept in all_concepts:
        concept = concept.strip()
        if len(concept) > 3:  # Ignore very short terms
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    # Sort by frequency
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top concepts
    return [c[0] for c in sorted_concepts[:num_concepts]]

def extract_key_provisions(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Stage 1: Extract key provisions and concepts from a section.
    This function identifies important elements to ask questions about.
    """
    text = section['text']
    title = section['title']
    
    # Prepare prompt for extracting key provisions
    prompt = f"""
    You are a legal expert analyzing the EU AI Act. I'll provide a section from the Act.
    Extract the 3-5 most important regulatory provisions or concepts from this section.
    For each one, provide:
    1. A concise name for the provision/concept
    2. A brief explanation of what it entails
    3. What entities it applies to
    4. Any notable exceptions or conditions
    
    Format your response as a valid JSON array of objects with the structure:
    [
      {{
        "name": "Name of provision/concept",
        "explanation": "Brief explanation",
        "applies_to": "Relevant entities",
        "conditions": "Notable exceptions or conditions"
      }}
    ]
    
    SECTION TITLE: {title}
    
    SECTION TEXT:
    {text}
    """
    
    try:
        # Call Ollama API
        response = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        content = response['message']['content']
        
        # Extract JSON
        json_match = re.search(r'```json\s*(.+?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # Try to find content between [ and ]
            json_match = re.search(r'(\[.+\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        
        provisions = json.loads(content)
        return provisions
    except Exception as e:
        logger.error(f"Error extracting provisions from section '{title}': {str(e)}")
        logger.debug(f"Response content: {response['message']['content'][:200]}...")
        
        # Fallback: Create basic provisions based on concepts
        concepts = extract_important_concepts(text)
        return [
            {
                "name": concept,
                "explanation": f"Concept related to {title}",
                "applies_to": "Entities under EU AI Act",
                "conditions": "Standard conditions apply"
            }
            for concept in concepts[:3]
        ]

def generate_diverse_questions(provision: Dict[str, Any], section: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Stage 2: Generate diverse questions based on the extracted provisions.
    This creates a variety of question types to enhance the training dataset.
    """
    section_title = section['title']
    provision_name = provision['name']
    provision_explanation = provision['explanation']
    
    questions = []
    
    # Select question types based on the provision content
    question_types = []
    
    # Always include definitional questions
    question_types.append("definitional")
    
    # Add other question types based on content
    if "prohibited" in provision_explanation.lower() or "restriction" in provision_explanation.lower():
        question_types.append("prohibition")
    
    if "compliance" in provision_explanation.lower() or "requirement" in provision_explanation.lower():
        question_types.append("compliance")
    
    if "exception" in provision['conditions'].lower() or "derogation" in provision['conditions'].lower():
        question_types.append("exceptions")
    
    if "authority" in provision_explanation.lower() or "regulator" in provision_explanation.lower():
        question_types.append("authority")
    
    # Add scenario questions if we have enough context
    if len(provision_explanation) > 100:
        question_types.append("scenario")
    
    # If we have multiple sections to compare, add comparative questions
    if "versus" in provision_name or "compared" in provision_name:
        question_types.append("comparative")
    
    # Ensure we have at least 3 question types
    if len(question_types) < 3:
        remaining_types = [t for t in QUESTION_TEMPLATES.keys() if t not in question_types]
        question_types.extend(remaining_types[:3-len(question_types)])
    
    # Pick a subset to keep diversity but not too many questions per provision
    selected_types = question_types[:3]
    
    # Prepare prompt for generating diverse questions
    prompt = f"""
    You are a legal expert on the EU AI Act. Generate specialized questions about the following provision/concept:
    
    SECTION: {section_title}
    PROVISION: {provision_name}
    EXPLANATION: {provision_explanation}
    APPLIES TO: {provision['applies_to']}
    CONDITIONS: {provision['conditions']}
    
    For each question type below, create 1 detailed question and its comprehensive answer:
    {", ".join(selected_types)}
    
    Ensure the answer:
    1. Is based strictly on the provided information
    2. Uses precise legal terminology from the EU AI Act
    3. Includes specific details from the provision
    4. Explains any conditions or exceptions
    
    Format your response as a valid JSON array:
    [
      {{
        "question_type": "type name",
        "question": "The question text",
        "answer": "The comprehensive answer"
      }}
    ]
    """
    
    try:
        # Call Ollama API
        response = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        content = response['message']['content']
        
        # Extract JSON
        json_match = re.search(r'```json\s*(.+?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # Try to find content between [ and ]
            json_match = re.search(r'(\[.+\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        
        qa_pairs = json.loads(content)
        return qa_pairs
    except Exception as e:
        logger.error(f"Error generating questions for provision '{provision_name}': {str(e)}")
        logger.debug(f"Response content: {response['message']['content'][:200]}...")
        
        # Fallback: Create a basic question using templates
        question_type = selected_types[0]
        template = QUESTION_TEMPLATES[question_type][0]
        question = template.format(concept=provision_name, topic=provision_name, practice=provision_name)
        
        return [{
            "question_type": question_type,
            "question": question,
            "answer": f"{provision_explanation}. This applies to {provision['applies_to']} under conditions: {provision['conditions']}."
        }]

def resume_from_log() -> int:
    """Get the last processed section index from the progress log."""
    if PROGRESS_LOG.exists():
        return int(PROGRESS_LOG.read_text().strip())
    return -1

def save_progress(idx: int):
    """Save the current progress to the log file."""
    PROGRESS_LOG.write_text(str(idx))

def append_jsonl(record: dict):
    """Append a record to the output JSONL file."""
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def perform_api_call_with_retry(func, *args, max_attempts=MAX_ATTEMPTS, **kwargs):
    """Perform an API call with retry logic for robustness."""
    attempts = 0
    last_error = None
    
    while attempts < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempts += 1
            last_error = e
            wait_time = SLEEP_TIME * (2 ** attempts)  # Exponential backoff
            logger.warning(f"API call failed (attempt {attempts}/{max_attempts}): {str(e)}")
            logger.info(f"Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    # If we get here, all attempts failed
    logger.error(f"All {max_attempts} attempts failed. Last error: {str(last_error)}")
    raise last_error

def main(start_from_section=None):
    """Main function to process the EU AI Act and generate enhanced Q&A pairs."""
    # Check if input file exists
    if not INPUT_FILE.exists():
        logger.error(f"Error: Input file {INPUT_FILE} not found.")
        return
    
    # Read the markdown file
    try:
        logger.info(f"Reading input file: {INPUT_FILE}")
        md_text = INPUT_FILE.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    # Extract section hierarchy
    logger.info("Extracting section hierarchy...")
    sections = extract_section_hierarchy(md_text)
    logger.info(f"Extracted {len(sections)} sections from the document")
    
    # Determine starting section
    if start_from_section is not None:
        start_idx = start_from_section
        logger.info(f"Starting from manually specified section #{start_idx}")
        # Always append to existing file when starting from specific section
    else:
        # Use log file if no specific section provided
        start_idx = resume_from_log() + 1
        if start_idx == 0 and OUTPUT_FILE.exists():
            # Starting from beginning but file exists - ask what to do
            logger.warning(f"Output file {OUTPUT_FILE} already exists.")
            response = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if response == 'y':
                OUTPUT_FILE.unlink()
            else:
                logger.info("Aborting operation.")
                return
    
    # Check if we've already processed all sections
    if start_idx >= len(sections):
        logger.info("Nothing to do – all sections processed.")
        return
    
    # Process each section
    logger.info(f"Processing sections: {len(sections)} total, starting at #{start_idx}")
    
    for idx in tqdm(range(start_idx, len(sections)), desc="Processing sections"):
        section = sections[idx]
        
        # Skip sections with very little content
        if len(section['text'].strip()) < 50:
            logger.info(f"Skipping section {idx}: '{section['title']}' (too short)")
            save_progress(idx)
            continue
        
        try:
            # Stage 1: Extract key provisions
            logger.info(f"Extracting key provisions from section {idx}: '{section['title']}'")
            provisions = perform_api_call_with_retry(
                extract_key_provisions, 
                section
            )
            
            if not provisions:
                logger.warning(f"No provisions extracted from section {idx}")
                continue
                
            logger.info(f"Extracted {len(provisions)} provisions from section {idx}")
            
            # Stage 2: Generate diverse questions for each provision
            total_qa_pairs = 0
            
            for provision in provisions:
                logger.info(f"Generating questions for provision: '{provision['name']}'")
                
                # Allow time between API calls
                time.sleep(SLEEP_TIME)
                
                qa_pairs = perform_api_call_with_retry(
                    generate_diverse_questions, 
                    provision, 
                    section
                )
                
                # Save each QA pair with metadata
                for qa in qa_pairs:
                    qa_record = {
                        "section_id": section['id'],
                        "section_title": section['title'],
                        "section_level": section['level'],
                        "provision": provision['name'],
                        "question_type": qa["question_type"],
                        "question": qa["question"],
                        "answer": qa["answer"]
                    }
                    append_jsonl(qa_record)
                    total_qa_pairs += 1
            
            logger.info(f"Generated {total_qa_pairs} Q&A pairs for section {idx}")
            
            # Update progress
            save_progress(idx)
            
            # Allow time between sections
            time.sleep(SLEEP_TIME)
            
        except Exception as e:
            logger.error(f"Error processing section {idx}: {str(e)}")
            # Continue with next section
    
    logger.info(f"Processing complete. Q&A pairs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Continue processing from section 60, appending to existing file
    main(start_from_section=61)
