#!/usr/bin/env python3
"""
EU AI Act Q&A Generator

This script processes the EU AI Act document to generate high-quality Q&A pairs
using Ollama's Qwen3:8B model. The generated Q&A pairs are saved in JSONL format.

Requirements:
- ollama (https://ollama.ai/)
- qwen3:8b model installed in Ollama
- python packages: requests, tqdm, python-dotenv, jsonschema

Usage:
1. Place the EU AI Act document in the data/ directory
2. Run: python eu_ai_act.py
"""

import os
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import jsonschema
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eu_ai_act_qa_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_BASE = "http://localhost:11434/api"
MODEL_NAME = "qwen3:8b"
CHUNK_SIZE = 2000  # Increased chunk size for better context
MAX_QUESTIONS_PER_CHUNK = 5  # Increased max questions per chunk
OUTPUT_FILE = "data/qa.jsonl"
MAX_RETRIES = 3  # Max retries for API calls
REQUEST_TIMEOUT = 120  # seconds

# JSON Schema for Q&A validation
QA_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 10},
        "answer": {"type": "string", "minLength": 10},
        "context": {"type": "string"},
        "source": {"type": "string"},
        "category": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "generated_at": {"type": "string", "format": "date-time"},
                "model": {"type": "string"}
            },
            "required": ["chunk_id", "generated_at", "model"]
        }
    },
    "required": ["question", "answer", "context", "source", "metadata"]
}

@dataclass
class QAPair:
    """Data class to hold a single Q&A pair with metadata."""
    question: str
    answer: str
    context: str
    source: str = "EU AI Act"
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert QAPair to dictionary with proper formatting."""
        data = asdict(self)
        # Ensure all string fields are properly stripped
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.strip()
        return data
    
    @classmethod
    def validate(cls, data: Dict[str, Any]) -> bool:
        """Validate Q&A pair against schema."""
        try:
            jsonschema.validate(instance=data, schema=QA_SCHEMA)
            return True
        except jsonschema.ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return False

class EUAIActProcessor:
    """Processes the EU AI Act document to generate Q&A pairs."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the processor with the specified model."""
        self.model_name = model_name
        self.ollama_url = f"{OLLAMA_API_BASE}/generate"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama service is running and accessible."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Ollama service: {e}")
            return False
    
    def _ensure_model_available(self) -> bool:
        """Ensure the required model is available, pull if needed."""
        try:
            # First check if the model is available locally
            response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=30)
            response.raise_for_status()
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Check if the exact model name is in the list
            if self.model_name in model_names:
                logger.info(f"Model '{self.model_name}' is available locally")
                return True
                
            # If not found, try pulling the model
            logger.info(f"Model '{self.model_name}' not found locally. Attempting to pull...")
            try:
                response = requests.post(
                    f"{OLLAMA_API_BASE}/pull",
                    json={"name": self.model_name},
                    timeout=600  # 10 minutes for model download
                )
                response.raise_for_status()
                
                # Stream the pull progress
                for line in response.iter_lines():
                    if line:
                        status = json.loads(line)
                        if 'status' in status:
                            logger.info(f"Download status: {status['status']}")
                            
                logger.info("Model pulled successfully!")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to pull model: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response: {e.response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            logger.error("Please make sure Ollama is running and accessible.")
            logger.error("Try running 'ollama serve' in a separate terminal.")
            return False

    def _call_ollama(self, prompt: str, max_tokens: int = 500) -> str:
        """Call the Ollama API with the given prompt."""
        logger.info("Checking Ollama connection...")
        if not self._check_ollama_connection():
            logger.error("Ollama service is not running. Please start it with 'ollama serve'")
            return ""
            
        logger.info("Ensuring model is available...")
        if not self._ensure_model_available():
            logger.error(f"Failed to ensure model '{self.model_name}' is available")
            return ""
        
        # Truncate prompt if too long for logging
        log_prompt = (prompt[:200] + '...') if len(prompt) > 200 else prompt
        logger.info(f"Sending request to Ollama API with model: {self.model_name}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        logger.debug(f"Prompt preview: {log_prompt}")
            
        try:
            # Split the request into connection and read timeouts
            session = requests.Session()
            
            # First, try a simple test request
            test_response = session.get(f"{OLLAMA_API_BASE}/version", timeout=5)
            test_response.raise_for_status()
            logger.info(f"Ollama version: {test_response.json().get('version', 'unknown')}")
            
            # Prepare the request data
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,  # Use streaming to get progress updates
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "keep_alive": "5m"
            }
            
            logger.info("Sending request to generate response...")
            
            # Use a session with a longer timeout
            with session.post(
                self.ollama_url,
                json=request_data,
                stream=True,
                timeout=60  # 1 minute timeout for the connection
            ) as response:
                response.raise_for_status()
                
                # Process the streamed response
                full_response = []
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                full_response.append(chunk['response'])
                            logger.debug(f"Received chunk: {chunk}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse chunk: {line}")
                
                if not full_response:
                    logger.error("No response received from Ollama API")
                    return ""
                    
                return ''.join(full_response).strip()
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request to Ollama API timed out: {e}")
            logger.error("The model might be taking too long to respond. Try reducing the chunk size or max tokens.")
            return ""
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response headers: {dict(e.response.headers)}")
                    logger.error(f"Response content: {e.response.text[:500]}...")
                except Exception as err:
                    logger.error(f"Failed to parse error response: {err}")
            return ""
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Ollama API response: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return ""
            
        except Exception as e:
            logger.error(f"Unexpected error in _call_ollama: {str(e)}")
            logger.exception("Full traceback:")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text before processing."""
        # Remove multiple spaces, newlines, etc.
        text = ' '.join(text.split())
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        return text.strip()
    
    def _extract_qa_pairs(self, text_chunk: str, chunk_num: int) -> List[QAPair]:
        """Extract Q&A pairs from a text chunk using the LLM with retries."""
        chunk_id = f"chunk_{chunk_num:04d}"
        cleaned_text = self._clean_text(text_chunk)
        
        prompt = f"""Generate exactly {MAX_QUESTIONS_PER_CHUNK} clear, specific Q&A pairs from this text about the EU AI Act.

RULES:
- Be concise and factual
- Each answer must be directly from the text
- Focus on key points, definitions, and requirements
- Format as JSON array with 'question' and 'answer' fields
- Keep answers under 100 words
- No explanations, just the JSON array

Text:
{cleaned_text}

Example output:
[
  {{
    "question": "What is the definition of an AI system according to the EU AI Act?",
    "answer": "According to Article 3(1), an AI system is software that can, for a given set of human-defined objectives, generate outputs such as content, predictions, recommendations, or decisions influencing the environments they interact with.",
    "category": "Definitions"
  }}
]
"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self._call_ollama(prompt)
                if not response:
                    continue
                    
                # Try to find and extract JSON array
                # Find the first [ and last ] in the response
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                
                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    logger.warning(f"No valid JSON array found in response (attempt {attempt + 1})")
                    continue
                    
                try:
                    json_str = response[start_idx:end_idx+1]
                    qa_list = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON (attempt {attempt + 1}): {e}")
                    continue
                if not isinstance(qa_list, list):
                    logger.warning(f"Expected JSON array, got {type(qa_list).__name__}")
                    continue
                    
                valid_pairs = []
                for i, qa in enumerate(qa_list[:MAX_QUESTIONS_PER_CHUNK]):
                    if not isinstance(qa, dict):
                        continue
                        
                    # Create QAPair with metadata
                    qa_pair = QAPair(
                        question=str(qa.get('question', '')).strip(),
                        answer=str(qa.get('answer', '')).strip(),
                        context=cleaned_text[:1000] + ("..." if len(cleaned_text) > 1000 else ""),
                        category=qa.get('category', 'General'),
                        source="EU AI Act",
                        metadata={
                            "chunk_id": f"{chunk_id}_qa{i}",
                            "generated_at": datetime.utcnow().isoformat(),
                            "model": self.model_name,
                            "version": "1.0"
                        }
                    )
                    
                    # Validate the Q&A pair
                    if QAPair.validate(qa_pair.to_dict()):
                        valid_pairs.append(qa_pair)
                    
                if valid_pairs:
                    return valid_pairs
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Error processing response (attempt {attempt + 1}): {e}")
                time.sleep(2)  # Backoff before retry
                
        logger.warning(f"Failed to extract valid Q&A pairs after {MAX_RETRIES} attempts for chunk {chunk_num}")
        return []
    
    def process_document(self, file_path: str) -> None:
        """Process the EU AI Act document and generate Q&A pairs."""
        logger.info(f"Processing document: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return
        
        # Verify Ollama is accessible before proceeding
        if not self._check_ollama_connection():
            logger.error("Cannot connect to Ollama service. Please make sure it's running with 'ollama serve'")
            return
            
        if not self._ensure_model_available():
            logger.error(f"Failed to ensure model '{self.model_name}' is available")
            return
        
        # Split content into chunks
        chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Process each chunk and collect Q&A pairs
        qa_pairs = []
        successful_chunks = 0
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            if not chunk.strip():
                continue
                
            try:
                # Get Q&A pairs for this chunk
                chunk_qa = self._extract_qa_pairs(chunk, i + 1)
                if chunk_qa:  # Only count chunks that produced Q&A pairs
                    successful_chunks += 1
                    qa_pairs.extend(chunk_qa)
                    
                    # Save after each successful chunk to prevent data loss
                    self._save_qa_pairs(qa_pairs)
                    
                    # Be nice to the API
                    time.sleep(1)
                    
                    # Log progress every 10 chunks
                    if successful_chunks % 10 == 0:
                        logger.info(f"Processed {successful_chunks}/{len(chunks)} chunks, generated {len(qa_pairs)} Q&A pairs")
                        
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue  # Continue with next chunk on error
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs in total")
    
    def _save_qa_pairs(self, qa_pairs: List[QAPair]) -> None:
        """Save Q&A pairs to the output file in JSONL format with proper formatting."""
        if not qa_pairs:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            
            # Write to a temporary file first to prevent data loss
            temp_file = f"{OUTPUT_FILE}.tmp"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    qa_dict = qa.to_dict()
                    # Ensure proper JSON serialization
                    json.dump(qa_dict, f, ensure_ascii=False, indent=2)
                    f.write('\n')
            
            # Atomically replace the old file
            if os.path.exists(OUTPUT_FILE):
                os.replace(temp_file, OUTPUT_FILE)
            else:
                os.rename(temp_file, OUTPUT_FILE)
                
            logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {OUTPUT_FILE}")
            
        except Exception as e:
            logger.error(f"Error saving Q&A pairs: {e}")
            # Clean up temporary file if it exists
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    """Main function to run the Q&A generation process."""
    # Load environment variables
    load_dotenv()
    
    # Get the EU AI Act document path
    document_dir = os.path.dirname(__file__)  # Look in the same directory as the script
    document_path = os.path.join(document_dir, 'euaiact.md')  # Updated filename
    
    if not os.path.exists(document_path):
        logger.error(f"EU AI Act document not found at {document_path}")
        logger.info("Please ensure the EU AI Act markdown file (euaiact.md) is in the same directory as this script.")
        return
    
    # Initialize and run the processor
    processor = EUAIActProcessor()
    processor.process_document(document_path)
    logger.info(f"Q&A pairs saved to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
