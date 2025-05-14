# #!/usr/bin/env python3
# """
# EU AI Act Q&A Generator

# This script processes the EU AI Act document to generate high-quality Q&A pairs
# using Ollama's Qwen3:8B model. The generated Q&A pairs are saved in JSONL format.

# Requirements:
# - ollama (https://ollama.ai/)
# - qwen3:8b model installed in Ollama
# - python packages: requests, tqdm, python-dotenv, jsonschema, tiktoken, backoff

# Usage:
#   python eu_ai_act.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--workers NUM_WORKERS]

# Example:
#   python eu_ai_act.py --input data/euaiact.md --output data/qa.jsonl --workers 4
# """

# import os
# import json
# import time
# import re
# import logging
# import argparse
# import math
# import concurrent.futures
# from pathlib import Path
# from typing import List, Dict, Optional, Any, Tuple, Generator
# from dataclasses import dataclass, asdict, field
# import requests
# from tqdm import tqdm
# from dotenv import load_dotenv
# import jsonschema
# from datetime import datetime
# import tiktoken
# import backoff
# from functools import partial

# # Configure logging with more detailed format
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
#     handlers=[
#         logging.FileHandler('eu_ai_act_qa_generator.log', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger('EUAIActProcessor')

# # Set higher log level for requests/urllib3 to avoid excessive logging
# logging.getLogger('urllib3').setLevel(logging.WARNING)
# logging.getLogger('requests').setLevel(logging.WARNING)

# # Constants
# OLLAMA_API_BASE = "http://localhost:11434/api"
# MODEL_NAME = "qwen3:8b"
# DEFAULT_OUTPUT_FILE = "data/qa.jsonl"
# DEFAULT_INPUT_FILE = "euaiact.md"

# # Model and token limits
# MAX_TOKENS = 8000  # Qwen's context window size
# RESERVED_TOKENS = 1000  # Reserve for prompt template, response, and safety margin
# MAX_QUESTIONS_PER_CHUNK = 5  # Reduced to improve quality
# MIN_ANSWER_LENGTH = 10  # Minimum answer length in words
# MAX_ANSWER_LENGTH = 150  # Maximum answer length in words
# MAX_RETRIES = 3  # Max retries for API calls
# REQUEST_TIMEOUT = 300  # Increased timeout for Qwen 8B
# MAX_WORKERS = 4  # Default number of worker threads
# BATCH_SIZE = 10  # Number of chunks to process before saving

# # Initialize tiktoken for accurate token counting
# try:
#     enc = tiktoken.get_encoding("cl100k_base")
#     logger.info("Using cl100k_base encoding for token counting")
# except Exception as e:
#     logger.warning(f"Failed to load cl100k_base encoding: {e}. Using fallback token counter.")
#     enc = None

# def count_tokens(text: str) -> int:
#     """Count tokens in text using tiktoken if available, fallback to word-based estimation."""
#     if enc:
#         return len(enc.encode(text))
#     # Fallback: rough estimation (1 token ≈ 4 characters for English)
#     return len(text) // 4

# # JSON Schema for Q&A validation
# QA_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "question": {"type": "string", "minLength": 10},
#         "answer": {"type": "string", "minLength": 10},
#         "context": {"type": "string"},
#         "source": {"type": "string"},
#         "category": {"type": "string"},
#         "metadata": {
#             "type": "object",
#             "properties": {
#                 "chunk_id": {"type": "string"},
#                 "generated_at": {"type": "string", "format": "date-time"},
#                 "model": {"type": "string"}
#             },
#             "required": ["chunk_id", "generated_at", "model"]
#         }
#     },
#     "required": ["question", "answer", "context", "source", "metadata"]
# }

# @dataclass
# class QAPair:
#     """Data class to hold a single Q&A pair with metadata."""
#     question: str
#     answer: str
#     context: str
#     source: str = "EU AI Act"
#     category: Optional[str] = None
#     metadata: Dict[str, Any] = field(default_factory=dict)
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert QAPair to dictionary with proper formatting."""
#         data = asdict(self)
#         # Ensure all string fields are properly stripped
#         for key, value in data.items():
#             if isinstance(value, str):
#                 data[key] = value.strip()
#         return data
    
#     @classmethod
#     def validate(cls, data: Dict[str, Any]) -> bool:
#         """Validate Q&A pair against schema."""
#         try:
#             jsonschema.validate(instance=data, schema=QA_SCHEMA)
#             return True
#         except jsonschema.ValidationError as e:
#             logger.warning(f"Validation error: {e}")
#             return False

# class EUAIActProcessor:
#     """Processes the EU AI Act document to generate Q&A pairs with parallel processing."""
    
#     def __init__(self, model_name: str = MODEL_NAME, output_file: str = DEFAULT_OUTPUT_FILE, 
#                  max_workers: int = MAX_WORKERS, batch_size: int = BATCH_SIZE):
#         """Initialize the processor with configuration.
        
#         Args:
#             model_name: Name of the Ollama model to use
#             output_file: Path to save the output JSONL file
#             max_workers: Maximum number of parallel workers for processing chunks
#             batch_size: Number of chunks to process before saving to disk
#         """
#         self.model_name = model_name
#         self.output_file = os.path.abspath(output_file)
#         self.max_workers = max_workers
#         self.batch_size = batch_size
#         self.ollama_url = f"{OLLAMA_API_BASE}/generate"
        
#         # Create output directory if it doesn't exist
#         os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
#         logger.info(f"Output will be saved to: {self.output_file}")
        
#         # Initialize session for connection pooling
#         self.session = requests.Session()
#         self.session.headers.update({
#             'Content-Type': 'application/json',
#             'Accept': 'application/json'
#         })
    
#     def _check_ollama_connection(self) -> bool:
#         """Check if Ollama service is running and accessible."""
#         try:
#             response = self.session.get(
#                 f"{OLLAMA_API_BASE}/tags",
#                 timeout=10,
#                 allow_redirects=True
#             )
#             response.raise_for_status()
#             return True
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Cannot connect to Ollama service: {e}")
#             if hasattr(e, 'response') and e.response is not None:
#                 logger.error(f"Response: {e.response.status_code} - {e.response.text}")
#             return False
    
#     def _ensure_model_available(self) -> bool:
#         """Ensure the required model is available, pull if needed."""
#         try:
#             # First check if the model is available locally
#             response = self.session.get(
#                 f"{OLLAMA_API_BASE}/tags",
#                 timeout=30
#             )
#             response.raise_for_status()
            
#             models = response.json().get('models', [])
#             model_names = [m.get('name', '') for m in models if 'name' in m]
            
#             # Check if the exact model name is in the list
#             if self.model_name in model_names:
#                 logger.info(f"Model '{self.model_name}' is available locally")
#                 return True
                
#             # If not found, try pulling the model
#             logger.warning(f"Model '{self.model_name}' not found locally. Attempting to pull...")
#             try:
#                 with self.session.post(
#                     f"{OLLAMA_API_BASE}/pull",
#                     json={"name": self.model_name},
#                     stream=True,
#                     timeout=600  # 10 minutes for model download
#                 ) as response:
#                     response.raise_for_status()
                    
#                     # Stream the pull progress
#                     for line in response.iter_lines():
#                         if line:
#                             try:
#                                 status = json.loads(line)
#                                 if 'status' in status:
#                                     logger.info(f"Download status: {status['status']}")
#                                 elif 'error' in status:
#                                     logger.error(f"Pull error: {status['error']}")
#                                     return False
#                             except json.JSONDecodeError:
#                                 logger.warning(f"Could not parse response: {line}")
                
#                 logger.info("Model pulled successfully!")
#                 return True
                
#             except requests.exceptions.RequestException as e:
#                 logger.error(f"Failed to pull model: {e}")
#                 if hasattr(e, 'response') and e.response is not None:
#                     logger.error(f"Response: {e.response.status_code} - {e.response.text}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"Error checking model availability: {e}", exc_info=True)
#             logger.error("Please make sure Ollama is running and accessible.")
#             logger.error("Try running 'ollama serve' in a separate terminal.")
#             return False

#     @backoff.on_exception(backoff.expo, 
#                          (requests.exceptions.RequestException, 
#                           json.JSONDecodeError, 
#                           KeyError, 
#                           ValueError),
#                          max_tries=MAX_RETRIES,
#                          jitter=backoff.full_jitter(60))
#     def _call_ollama(self, prompt: str, max_tokens: int = 1000) -> str:
#         """
#         Call the Ollama API with the given prompt using the configured model.
        
#         Args:
#             prompt: The prompt to send to the model
#             max_tokens: Maximum number of tokens to generate
            
#         Returns:
#             The generated text response from the model
#         """
#         # Verify Ollama is accessible before proceeding
#         if not self._check_ollama_connection():
#             raise ConnectionError("Ollama service is not running. Please start it with 'ollama serve'")
            
#         if not self._ensure_model_available():
#             raise ValueError(f"Model '{self.model_name}' is not available")
        
#         # Calculate token count accurately
#         prompt_tokens = count_tokens(prompt)
        
#         # Calculate safe max_tokens based on model's context window
#         safe_max_tokens = min(max_tokens, MAX_TOKENS - prompt_tokens - 100)  # 100 token buffer
#         if safe_max_tokens < 100:  # If we can't even get 100 tokens, it's not worth it
#             raise ValueError(f"Prompt too long: {prompt_tokens} tokens, max is {MAX_TOKENS}")
            
#         logger.debug(f"Calling {self.model_name} with {prompt_tokens} prompt tokens, "
#                    f"max {safe_max_tokens} response tokens")
        
#         # Prepare the request data
#         request_data = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "stream": False,  # Disable streaming for simplicity
#             "options": {
#                 "num_ctx": MAX_TOKENS,
#                 "temperature": 0.3,
#                 "top_p": 0.9,
#                 "repeat_penalty": 1.1,
#                 "stop": ["\n###"]
#             },
#             "keep_alive": "5m"
#         }
        
#         try:
#             # Make the API call
#             response = self.session.post(
#                 self.ollama_url,
#                 json=request_data,
#                 timeout=REQUEST_TIMEOUT
#             )
#             response.raise_for_status()
            
#             # Parse the response
#             result = response.json()
#             if 'response' not in result:
#                 raise ValueError("No 'response' field in Ollama API response")
                
#             return result['response'].strip()
            
#         except requests.exceptions.RequestException as e:
#             error_msg = f"Request to Ollama API failed: {str(e)}"
#             if hasattr(e, 'response') and e.response is not None:
#                 try:
#                     error_details = e.response.json()
#                     error_msg += f" - {error_details.get('error', 'Unknown error')}"
#                 except:
#                     error_msg += f" - Status: {e.response.status_code}"
#             logger.error(error_msg)
#             raise
            
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode Ollama API response: {e}")
#             if hasattr(e, 'response') and e.response is not None:
#                 logger.error(f"Response content: {e.response.text[:500]}")
#             raise
    
#     def _clean_text(self, text: str) -> str:
#         """Clean and normalize text before processing."""
#         if not text or not isinstance(text, str):
#             return ""
            
#         # Remove non-printable characters except newlines and tabs
#         text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
        
#         # Normalize whitespace but preserve paragraph breaks
#         text = re.sub(r'[\r\n]+', '\n', text)  # Normalize line endings
#         text = re.sub(r'[\t ]+', ' ', text)     # Normalize spaces
#         text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        
#         return text.strip()
        
#     def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
#         """Truncate text to a maximum number of tokens."""
#         if not text or max_tokens <= 0:
#             return ""
            
#         if enc:
#             # Use tiktoken for accurate token counting if available
#             tokens = enc.encode(text)
#             if len(tokens) <= max_tokens:
#                 return text
#             return enc.decode(tokens[:max_tokens])
#         else:
#             # Fallback: approximate by characters (1 token ≈ 4 chars for English)
#             approx_max_chars = max_tokens * 4
#             return text[:approx_max_chars]
    
#     def _extract_json_from_response(self, response: str) -> List[Dict]:
#         """Extract JSON array from model response with improved error handling."""
#         try:
#             # Try to find JSON array in the response
#             start_idx = response.find('[')
#             end_idx = response.rfind(']') + 1
            
#             if start_idx == -1 or end_idx == 0 or start_idx >= end_idx:
#                 logger.warning("No JSON array found in response")
#                 return []
                
#             json_str = response[start_idx:end_idx].strip()
            
#             # Try to parse the JSON
#             try:
#                 data = json.loads(json_str)
#                 if not isinstance(data, list):
#                     logger.warning("JSON is not an array")
#                     return []
#                 return data
#             except json.JSONDecodeError as e:
#                 logger.warning(f"Failed to parse JSON: {e}")
#                 # Try to fix common JSON issues
#                 try:
#                     # Try to fix unescaped quotes and other common issues
#                     json_str = json_str.replace('\n', ' ').replace('\r', ' ')
#                     json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas
#                     json_str = re.sub(r',\s*}', '}', json_str)    # Remove trailing commas in objects
#                     return json.loads(json_str)
#                 except json.JSONDecodeError as e2:
#                     logger.warning(f"Failed to fix JSON: {e2}")
#                     return []
                    
#         except Exception as e:
#             logger.error(f"Error extracting JSON from response: {e}")
#             return []
    
#     def _process_qa_pairs(self, qa_list: List[Dict], context: str, chunk_id: str) -> List[QAPair]:
#         """Process raw Q&A pairs into validated QAPair objects."""
#         valid_pairs = []
        
#         for i, qa in enumerate(qa_list[:MAX_QUESTIONS_PER_CHUNK]):
#             if not isinstance(qa, dict):
#                 continue
                
#             try:
#                 # Extract and validate fields
#                 question = str(qa.get('question', '')).strip()
#                 answer = str(qa.get('answer', '')).strip()
#                 category = str(qa.get('category', 'Other')).strip()
                
#                 # Validate required fields
#                 if not question or not answer:
#                     logger.debug(f"Skipping Q&A pair {i+1}: missing question or answer")
#                     continue
                    
#                 # Validate answer length
#                 answer_words = len(answer.split())
#                 if not (MIN_ANSWER_LENGTH <= answer_words <= MAX_ANSWER_LENGTH):
#                     logger.debug(f"Skipping Q&A pair {i+1}: answer length {answer_words} words is out of range")
#                     continue
                
#                 # Validate category
#                 valid_categories = [
#                     'Definitions', 'Requirements', 'Prohibitions', 'Obligations',
#                     'Enforcement', 'Risk Management', 'Compliance', 'Other'
#                 ]
#                 if category not in valid_categories:
#                     category = 'Other'
                
#                 # Create QAPair with metadata
#                 qa_pair = QAPair(
#                     question=question,
#                     answer=answer,
#                     context=context[:1200] + ("..." if len(context) > 1200 else ""),
#                     category=category,
#                     source="EU AI Act",
#                     metadata={
#                         "chunk_id": f"{chunk_id}_qa{i}",
#                         "generated_at": datetime.utcnow().isoformat(),
#                         "model": self.model_name,
#                         "version": "1.2",
#                         "answer_length": answer_words,
#                         "article_references": re.findall(r'[Aa]rticle\s+\d+', answer)
#                     }
#                 )
                
#                 # Validate the Q&A pair
#                 if QAPair.validate(qa_pair.to_dict()):
#                     valid_pairs.append(qa_pair)
#                 else:
#                     logger.debug(f"Skipping invalid Q&A pair {i+1}: validation failed")
                    
#             except Exception as e:
#                 logger.warning(f"Error processing Q&A pair {i+1}: {e}", exc_info=True)
#                 continue
                
#         return valid_pairs
    
#     def _extract_qa_pairs(self, text_chunk: str, chunk_num: int) -> List[QAPair]:
#         """
#         Extract Q&A pairs from a text chunk using the configured model with retries.
        
#         Args:
#             text_chunk: The text to extract Q&A pairs from
#             chunk_num: The chunk number for logging and identification
            
#         Returns:
#             List of valid QAPair objects
#         """
#         chunk_id = f"chunk_{chunk_num:04d}"
#         cleaned_text = self._clean_text(text_chunk)
        
#         # Ensure we don't exceed the token limit
#         max_chunk_tokens = MAX_TOKENS - RESERVED_TOKENS - 1000  # Reserve space for prompt and response
#         chunk_tokens = count_tokens(cleaned_text)
        
#         if chunk_tokens > max_chunk_tokens:
#             # If chunk is too large, split it intelligently
#             logger.warning(f"Chunk {chunk_num} is too large ({chunk_tokens} tokens), truncating...")
#             cleaned_text = self._truncate_text_by_tokens(cleaned_text, max_chunk_tokens)
#             logger.info(f"Truncated chunk {chunk_num} to {count_tokens(cleaned_text)} tokens")
        
#         # Prepare the prompt with clear instructions
#         prompt = f"""You are an expert in the EU AI Act. Your task is to generate exactly {MAX_QUESTIONS_PER_CHUNK} high-quality question-answer pairs from the provided text.

# INSTRUCTIONS:
# 1. Generate diverse question types (what, why, how, when, who, etc.)
# 2. Focus on key concepts, requirements, definitions, obligations, and prohibitions
# 3. Include article and paragraph references when available
# 4. Each answer should be self-contained and understandable without the question
# 5. Be specific and avoid vague or generic questions

# TEXT TO ANALYZE:
# {cleaned_text}

# OUTPUT FORMAT (must be valid JSON array):
# [
#   {{
#     "question": "A clear, specific question about the text",
#     "answer": "A concise, accurate answer directly from the text. Include relevant article numbers and paragraphs.",
#     "category": "One of: Definitions, Requirements, Prohibitions, Obligations, Enforcement, Risk Management, Compliance, or Other"
#   }}
# ]

# GUIDELINES:
# - Questions must be answerable from the provided text only
# - Answers should be between {MIN_ANSWER_LENGTH} and {MAX_ANSWER_LENGTH} words
# - Include article numbers and paragraphs in answers when possible
# - Each question should cover a distinct aspect of the text
# - Avoid yes/no questions unless they include a 'why' or 'how' component
# - Ensure factual accuracy and completeness of answers
# - Maintain consistency with the EU AI Act terminology
# - Do not include any explanations, just the JSON array

# EXAMPLE:
# [
#   {{
#     "question": "What are the key requirements for high-risk AI systems under Article 14?",
#     "answer": "According to Article 14 of the EU AI Act, high-risk AI systems must meet several key requirements including: (a) establishment of a risk management system, (b) use of high-quality training data, (c) detailed technical documentation, (d) record-keeping, (e) transparency and provision of information to users, (f) human oversight, and (g) robustness, accuracy, and cybersecurity.",
#     "category": "Requirements"
#   }}
# ]"""
        
#         # Add retry logic with exponential backoff
#         for attempt in range(MAX_RETRIES):
#             try:
#                 # Call the model
#                 response = self._call_ollama(prompt, max_tokens=2000)
#                 if not response:
#                     logger.warning(f"Empty response from model on attempt {attempt + 1}")
#                     continue
                
#                 # Try to extract JSON from the response
#                 qa_list = self._extract_json_from_response(response)
#                 if not qa_list:
#                     logger.warning(f"No valid Q&A pairs found in response on attempt {attempt + 1}")
#                     continue
                
#                 # Process the Q&A pairs
#                 valid_pairs = self._process_qa_pairs(qa_list, cleaned_text, chunk_id)
#                 if valid_pairs:
#                     logger.info(f"Successfully extracted {len(valid_pairs)} Q&A pairs from chunk {chunk_num}")
#                     return valid_pairs
                
#                 logger.warning(f"No valid Q&A pairs extracted on attempt {attempt + 1}")
                
#             except Exception as e:
#                 logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
#                 if attempt == MAX_RETRIES - 1:  # Last attempt
#                     logger.error(f"Failed to extract Q&A pairs after {MAX_RETRIES} attempts: {e}")
#                 time.sleep(2 ** attempt)  # Exponential backoff
#                 continue
        
#         return []  # Return empty list if all attempts failed
    
#     def process_document(self, file_path: str) -> None:
#         """Process the EU AI Act document and generate high-quality Q&A pairs."""
#         start_time = time.time()
#         logger.info(f"Starting document processing: {file_path}")
        
#         try:
#             # Read and validate the document
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
                
#             if not content.strip():
#                 logger.error("Document is empty")
#                 return
                
#             doc_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
#             logger.info(f"Document loaded successfully. Size: {doc_size_mb:.2f} MB")
                
#         except Exception as e:
#             logger.error(f"Error reading file {file_path}: {e}")
#             logger.exception("Full traceback:")
#             return
        
#         # Verify Ollama is accessible before proceeding
#         logger.info("Verifying Ollama connection...")
#         if not self._check_ollama_connection():
#             logger.error("Cannot connect to Ollama service. Please make sure it's running with 'ollama serve'")
#             return
            
#         logger.info("Checking model availability...")
#         if not self._ensure_model_available():
#             logger.error(f"Failed to ensure model '{self.model_name}' is available")
#             return
        
#         # Pre-process content
#         logger.info("Pre-processing document content...")
#         cleaned_content = self._clean_text(content)
        
#         # Split content into meaningful chunks (trying to keep paragraphs together)
#         paragraphs = [p for p in cleaned_content.split('\n\n') if p.strip()]
#         chunks = []
#         current_chunk = []
#         current_length = 0
        
#         for para in paragraphs:
#             para = para.strip()
#             if not para:
#                 continue
                
#             para_length = len(para)
            
#             # If adding this paragraph would exceed chunk size, finalize current chunk
#             if current_length + para_length > CHUNK_SIZE and current_chunk:
#                 chunks.append('\n\n'.join(current_chunk))
#                 current_chunk = []
#                 current_length = 0
                
#             current_chunk.append(para)
#             current_length += para_length
            
#             # If current paragraph is very long, split it
#             if para_length > CHUNK_SIZE:
#                 # Split long paragraph into sentences
#                 sentences = re.split(r'(?<=[.!?])\s+', para)
#                 current_para = []
#                 current_para_length = 0
                
#                 for sentence in sentences:
#                     sentence = sentence.strip()
#                     if not sentence:
#                         continue
                        
#                     if current_para_length + len(sentence) > CHUNK_SIZE and current_para:
#                         chunks.append('\n\n'.join(current_para))
#                         current_para = []
#                         current_para_length = 0
                        
#                     current_para.append(sentence)
#                     current_para_length += len(sentence)
                    
#                 if current_para:
#                     current_chunk = current_para  # Replace with the processed sentences
#                     current_length = current_para_length
        
#         # Add the last chunk if not empty
#         if current_chunk:
#             chunks.append('\n\n'.join(current_chunk))
        
#         logger.info(f"Split document into {len(chunks)} meaningful chunks")
        
#         # Process each chunk and collect Q&A pairs
#         qa_pairs = []
#         successful_chunks = 0
#         total_qa_pairs = 0
        
#         try:
#             with tqdm(chunks, desc="Processing chunks") as pbar:
#                 for i, chunk in enumerate(pbar):
#                     if not chunk.strip():
#                         continue
                        
#                     try:
#                         # Update progress bar description
#                         pbar.set_description(f"Processing chunk {i+1}/{len(chunks)}")
                        
#                         # Get Q&A pairs for this chunk
#                         chunk_qa = self._extract_qa_pairs(chunk, i + 1)
                        
#                         if chunk_qa:  # Only count chunks that produced Q&A pairs
#                             successful_chunks += 1
#                             qa_pairs.extend(chunk_qa)
#                             total_qa_pairs = len(qa_pairs)
                            
#                             # Save after each successful chunk to prevent data loss
#                             self._save_qa_pairs(qa_pairs)
                            
#                             # Update progress bar with current stats
#                             pbar.set_postfix({
#                                 'success_rate': f"{successful_chunks/(i+1):.0%}",
#                                 'qa_pairs': total_qa_pairs
#                             })
                            
#                             # Be nice to the API
#                             time.sleep(1.5)  # Slightly increased delay
                            
#                             # Log progress every 5 chunks
#                             if successful_chunks % 5 == 0:
#                                 logger.info(
#                                     f"Processed {successful_chunks}/{len(chunks)} chunks, "
#                                     f"generated {total_qa_pairs} Q&A pairs, "
#                                     f"success rate: {successful_chunks/(i+1):.0%}"
#                                 )
                                    
#                     except Exception as e:
#                         logger.error(f"Error processing chunk {i+1}: {e}")
#                         logger.exception("Chunk processing error:")
#                         continue  # Continue with next chunk on error
            
#             # Final save to ensure all pairs are written
#             self._save_qa_pairs(qa_pairs)
            
#             # Log completion stats
#             elapsed_time = time.time() - start_time
#             avg_time_per_chunk = elapsed_time / len(chunks) if chunks else 0
#             qa_per_minute = (total_qa_pairs / (elapsed_time / 60)) if elapsed_time > 0 else 0
            
#             logger.info("\n" + "="*50)
#             logger.info("DOCUMENT PROCESSING COMPLETE")
#             logger.info("="*50)
            
#             # Log document statistics
#             stats = f"""
# Document Statistics:
# - Total chunks processed: {len(chunks)}
# - Successful chunks: {successful_chunks} ({successful_chunks/len(chunks):.1%})
# - Total Q&A pairs generated: {total_qa_pairs}
# - Processing time: {elapsed_time/60:.1f} minutes
# - Average time per chunk: {avg_time_per_chunk:.1f} seconds
# - Q&A pairs per minute: {qa_per_minute:.1f}
# """
#             logger.info(stats)
            
#         except KeyboardInterrupt:
#             logger.warning("\nProcess interrupted by user. Saving progress...")
#             if qa_pairs:
#                 self._save_qa_pairs(qa_pairs)
#                 logger.info(f"Saved {len(qa_pairs)} Q&A pairs before interruption")
#             return
#         except Exception as e:
#             logger.error(f"Unexpected error during document processing: {e}")
#             logger.exception("Full traceback:")
#             if qa_pairs:
#                 self._save_qa_pairs(qa_pairs)
#                 logger.info(f"Saved {len(qa_pairs)} Q&A pairs before error")
#             return
    
#     def _save_qa_pairs(self, qa_pairs: List[QAPair]) -> None:
#         """Save Q&A pairs to the output file in JSONL format with proper formatting and deduplication."""
#         if not qa_pairs:
#             return
            
#         try:
#             # Create directory if it doesn't exist
#             os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            
#             # Read existing Q&A pairs to avoid duplicates
#             existing_hashes = set()
#             existing_pairs = []
#             if os.path.exists(OUTPUT_FILE):
#                 with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
#                     for line in f:
#                         try:
#                             pair = json.loads(line.strip())
#                             # Create a unique hash based on question and answer
#                             pair_hash = hash((pair['question'].lower(), pair['answer'].lower()))
#                             existing_hashes.add(pair_hash)
#                             existing_pairs.append(pair)
#                         except (json.JSONDecodeError, KeyError) as e:
#                             logger.warning(f"Error reading existing Q&A pair: {e}")
            
#             # Prepare new pairs, skipping duplicates
#             new_pairs = []
#             duplicates = 0
#             for qa in qa_pairs:
#                 qa_dict = qa.to_dict()
#                 pair_hash = hash((qa_dict['question'].lower(), qa_dict['answer'].lower()))
#                 if pair_hash not in existing_hashes:
#                     new_pairs.append(qa_dict)
#                     existing_hashes.add(pair_hash)
#                 else:
#                     duplicates += 1
            
#             if duplicates > 0:
#                 logger.info(f"Skipped {duplicates} duplicate Q&A pairs")
            
#             if not new_pairs:
#                 logger.info("No new Q&A pairs to save")
#                 return
            
#             # Write to a temporary file first to prevent data loss
#             temp_file = f"{OUTPUT_FILE}.tmp"
            
#             try:
#                 with open(temp_file, 'w', encoding='utf-8') as f:
#                     # Write existing pairs first
#                     for pair in existing_pairs:
#                         json.dump(pair, f, ensure_ascii=False, separators=(',', ':'))
#                         f.write('\n')
                    
#                     # Append new pairs
#                     for pair in new_pairs:
#                         json.dump(pair, f, ensure_ascii=False, separators=(',', ':'))
#                         f.write('\n')
                
#                 # Atomically replace the old file
#                 if os.path.exists(OUTPUT_FILE):
#                     os.replace(temp_file, OUTPUT_FILE)
#                 else:
#                     os.rename(temp_file, OUTPUT_FILE)
                    
#                 logger.info(f"Saved {len(new_pairs)} new Q&A pairs to {OUTPUT_FILE} (total: {len(existing_pairs) + len(new_pairs)})")
                
#             except Exception as e:
#                 logger.error(f"Error writing Q&A pairs: {e}")
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
#                 raise
                
#         except Exception as e:
#             logger.error(f"Error in _save_qa_pairs: {e}")
#             logger.exception("Full traceback:")
#             # Clean up temporary file if it exists
#             if 'temp_file' in locals() and os.path.exists(temp_file):
#                 try:
#                     os.remove(temp_file)
#                 except Exception as cleanup_error:
#                     logger.error(f"Error cleaning up temp file: {cleanup_error}")

# def main():
#     """Main function to run the Q&A generation process."""
#     # Load environment variables
#     load_dotenv()
    
#     # Get the EU AI Act document path
#     document_dir = os.path.dirname(__file__)  # Look in the same directory as the script
#     document_path = os.path.join(document_dir, 'euaiact.md')  # Updated filename
    
#     if not os.path.exists(document_path):
#         logger.error(f"EU AI Act document not found at {document_path}")
#         logger.info("Please ensure the EU AI Act markdown file (euaiact.md) is in the same directory as this script.")
#         return
    
#     # Initialize and run the processor
#     processor = EUAIActProcessor()
#     processor.process_document(document_path)
#     logger.info(f"Q&A pairs saved to {os.path.abspath(OUTPUT_FILE)}")

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
EU AI Act Q&A Generator + Train/Dev/Test Splitter

This script:
 1. Generates Q&A pairs from the EU AI Act via Ollama's Qwen3:8B.
 2. Appends each QA (with UUID & confidence=1.0) into a JSONL (flushed per line).
 3. At the end, splits that JSONL into train/dev/test by chunk_id groups.

Usage:
  python eu_ai_act.py --input data/euaiact.md --output data/qa.jsonl --workers 4
"""

import os
import json
import time
import re
import logging
import argparse
import uuid
import random

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field
import requests
from tqdm import tqdm
import jsonschema
from datetime import datetime
import tiktoken
import backoff
from concurrent.futures import ThreadPoolExecutor

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('eu_ai_act.log'), logging.StreamHandler()]
)
logger = logging.getLogger('EUAIActProcessor')

# ─── Configuration ──────────────────────────────────────────────────────────
OLLAMA_API_BASE    = "http://localhost:11434/api"
MODEL_NAME         = "qwen3:8b"
MAX_TOKENS         = 8000
RESERVED_TOKENS    = 1000
MAX_Q_PER_CHUNK    = 5
MIN_A_WORDS        = 10
MAX_A_WORDS        = 150
MAX_RETRIES        = 3
REQUEST_TIMEOUT    = 300
CHUNK_CHAR_LIMIT   = 12000

TRAIN_RATIO, DEV_RATIO = 0.8, 0.1  # test = 1 - (train+dev)

# ─── Tokenizer ─────────────────────────────────────────────────────────────
try:
    enc = tiktoken.get_encoding("cl100k_base")
    logger.info("Using tiktoken cl100k_base")
except:
    enc = None
    logger.warning("tiktoken unavailable, falling back to char-estimate")

def count_tokens(text: str) -> int:
    return len(enc.encode(text)) if enc else max(1, len(text)//4)

# ─── JSON Schema ───────────────────────────────────────────────────────────
QA_SCHEMA = {
    "type": "object",
    "required": ["id","question","answer","context","source","metadata"],
    "properties": {
        "id":       {"type":"string","pattern":"^[0-9a-fA-F\\-]{36}$"},
        "question": {"type":"string","minLength":10},
        "answer":   {"type":"string","minLength":10},
        "context":  {"type":"string"},
        "source":   {"type":"string"},
        "category": {"type":"string"},
        "metadata": {
            "type":"object",
            "required":["chunk_id","generated_at","model","confidence"],
            "properties":{
                "chunk_id":     {"type":"string"},
                "generated_at": {"type":"string","format":"date-time"},
                "model":        {"type":"string"},
                "confidence":   {"type":"number","minimum":0,"maximum":1}
            }
        }
    }
}

# ─── Data Model ─────────────────────────────────────────────────────────────
@dataclass
class QAPair:
    id:       str
    question: str
    answer:   str
    context:  str
    source:   str = "EU AI Act"
    category: str = "Other"
    metadata: Dict[str,Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str,Any]:
        d = asdict(self)
        for k,v in d.items():
            if isinstance(v,str):
                d[k] = v.strip()
        return d

    @staticmethod
    def validate(d: Dict[str,Any]) -> bool:
        try:
            jsonschema.validate(instance=d, schema=QA_SCHEMA)
            return True
        except jsonschema.ValidationError as e:
            logger.warning(f"Validation failed: {e.message}")
            return False

# ─── Processor Class ───────────────────────────────────────────────────────
class EUAIActProcessor:
    def __init__(self, inp:str, out:str, workers:int):
        self.input_path  = Path(inp)
        self.output_file = Path(out)
        self.max_workers = workers
        self.session     = requests.Session()
        self.ollama_url  = f"{OLLAMA_API_BASE}/generate"
        self._prepare_output()
        self._load_hashes()
        logger.info(f"Init: in={self.input_path} out={self.output_file} workers={self.max_workers}")

    def _prepare_output(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_file.exists():
            self.output_file.write_text("")

    def _load_hashes(self):
        self.hashes = set()
        if self.output_file.exists():
            for line in self.output_file.read_text().splitlines():
                try:
                    p = json.loads(line)
                    h = hash((p['question'].lower(), p['answer'].lower()))
                    self.hashes.add(h)
                except:
                    pass
        logger.info(f"Loaded {len(self.hashes)} existing pairs")

    def _check_ollama(self) -> bool:
        try:
            r = self.session.get(f"{OLLAMA_API_BASE}/tags", timeout=10)
            r.raise_for_status()
            return True
        except:
            return False

    def _ensure_model(self) -> bool:
        if not self._check_ollama():
            return False
        r = self.session.get(f"{OLLAMA_API_BASE}/tags", timeout=30)
        r.raise_for_status()
        names = [m['name'] for m in r.json().get('models',[])]
        if MODEL_NAME in names:
            return True
        pull = self.session.post(f"{OLLAMA_API_BASE}/pull", json={"name":MODEL_NAME},
                                 stream=True, timeout=600)
        pull.raise_for_status()
        for ln in pull.iter_lines():
            try:
                st = json.loads(ln)
                logger.info(st.get('status', st.get('error','')))
            except:
                pass
        return True

    @backoff.on_exception(backoff.expo, (requests.RequestException, ValueError), max_tries=MAX_RETRIES)
    def _call_model(self, prompt:str) -> str:
        if not self._ensure_model():
            raise ConnectionError("Model unavailable")
        pt = count_tokens(prompt)
        maxr = MAX_TOKENS - pt - RESERVED_TOKENS
        if maxr < 50:
            raise ValueError("Prompt too large")
        r = self.session.post(
            self.ollama_url,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": MAX_TOKENS,
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            },
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("response","").strip()

    def _clean(self, text:str) -> str:
        t = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
        return re.sub(r'\n\s*\n', '\n\n', t).strip()

    def _truncate(self, text:str, limit:int) -> str:
        if enc:
            toks = enc.encode(text)
            return enc.decode(toks[:limit])
        return text[: limit * 4]

    def _extract_json(self, resp:str) -> List[Dict]:
        s = resp.find('[')
        e = resp.rfind(']') + 1
        if s < 0 or e <= s:
            return []
        try:
            return json.loads(resp[s:e])
        except:
            return []

    def _process_raw(self, raw:List[Dict], ctx:str, cid:str) -> List[QAPair]:
        out = []
        for qa in raw[:MAX_Q_PER_CHUNK]:
            q = qa.get('question','').strip()
            a = qa.get('answer','').strip()
            if not q or not a:
                continue
            w = len(a.split())
            if w < MIN_A_WORDS or w > MAX_A_WORDS:
                continue
            cat = qa.get('category','Other').strip()
            if cat not in ['Definitions','Requirements','Prohibitions','Obligations',
                           'Enforcement','Risk Management','Compliance','Other']:
                cat = 'Other'
            obj = QAPair(
                id=str(uuid.uuid4()),
                question=q,
                answer=a,
                context=ctx[:1200] + ("..." if len(ctx)>1200 else ""),
                category=cat,
                metadata={
                    "chunk_id": cid,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model": MODEL_NAME,
                    "confidence": 1.0
                }
            )
            d = obj.to_dict()
            if QAPair.validate(d):
                out.append(obj)
        return out

    def _extract_chunk(self, text:str, idx:int) -> List[QAPair]:
        cid = f"chunk_{idx:04d}"
        clean = self._clean(text)
        if count_tokens(clean) > MAX_TOKENS - RESERVED_TOKENS:
            clean = self._truncate(clean, MAX_TOKENS - RESERVED_TOKENS)
        prompt = f"""Generate exactly {MAX_Q_PER_CHUNK} Q&A pairs from this excerpt. Return ONLY a JSON array:

TEXT:
{clean}

FORMAT:
[{{"question":"...","answer":"...","category":"..."}}]
"""
        resp = self._call_model(prompt)
        raw = self._extract_json(resp)
        return self._process_raw(raw, clean, cid)

    def process(self):
        content = self.input_path.read_text(encoding='utf-8')
        paras = [p for p in self._clean(content).split('\n\n') if p.strip()]
        chunks, buf, length = [], [], 0
        for p in paras:
            if buf and length + len(p) > CHUNK_CHAR_LIMIT:
                chunks.append('\n\n'.join(buf))
                buf, length = [], 0
            buf.append(p)
            length += len(p)
        if buf:
            chunks.append('\n\n'.join(buf))
        logger.info(f"Split into {len(chunks)} chunks")

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            for qa_list in tqdm(
                exe.map(self._extract_chunk, chunks, range(1, len(chunks)+1)),
                total=len(chunks),
                desc="Chunks"
            ):
                self._save(qa_list)
                time.sleep(0.5)

    def _save(self, pairs:List[QAPair]):
        if not pairs:
            return
        written = 0
        # Open in line-buffered mode and force flush after each write
        with open(self.output_file, 'a', encoding='utf-8', buffering=1) as f:
            for qa in pairs:
                d = qa.to_dict()
                if d['id'] not in self.hashes:
                    json.dump(d, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()  # Force flush after each line
                    os.fsync(f.fileno())  # Force OS to write to disk
                    self.hashes.add(d['id'])
                    written += 1
        if written:
            logger.debug(f"Wrote {written} new Q&A pairs to {self.output_file}")

# ─── Splitter ──────────────────────────────────────────────────────────────
def split_dataset(input_path:Path):
    lines = [json.loads(l) for l in input_path.read_text().splitlines()]
    groups: Dict[str, List[Dict]] = {}
    for obj in lines:
        cid = obj['metadata']['chunk_id']
        groups.setdefault(cid, []).append(obj)
    cids = list(groups.keys())
    random.shuffle(cids)
    n = len(cids)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)
    splits = {
        'train': cids[:n_train],
        'dev':   cids[n_train:n_train+n_dev],
        'test':  cids[n_train+n_dev:]
    }
    for name, id_list in splits.items():
        out_file = input_path.parent / f"{name}.jsonl"
        with open(out_file, 'w', encoding='utf-8') as f:
            for cid in id_list:
                for obj in groups[cid]:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {name}.jsonl ({sum(len(groups[c]) for c in id_list)} examples)")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input','-i', default='euaiact.md')
    p.add_argument('--output','-o', default='data/qa.jsonl')
    p.add_argument('--workers','-w', type=int, default=4)
    args = p.parse_args()

    proc = EUAIActProcessor(args.input, args.output, args.workers)
    proc.process()

    split_dataset(Path(args.output))
    logger.info("Dataset split into train/dev/test complete.")

if __name__=="__main__":
    main()
