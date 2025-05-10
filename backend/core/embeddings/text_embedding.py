from typing import List, Dict, Any, Optional
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)

class TextEmbeddingService:
    """Service for converting text into vector embeddings."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding service with the specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       If None, uses the model specified in settings.
        """
        try:
            self.model_name = model_name or settings.EMBEDDING_MODEL
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            # Fallback to a simpler model in case of error
            self.model_name = "all-MiniLM-L6-v2"
            logger.info(f"Falling back to model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Convert text to vector embedding.
        
        Args:
            text: Input text to convert to embedding
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zeros as fallback
            return [0.0] * self.embedding_dim
    
    def embed_tokens(self, tokens: List[str]) -> List[List[float]]:
        """Convert list of tokens to embeddings.
        
        Args:
            tokens: List of token strings to embed
            
        Returns:
            List of embedding vectors (each a list of floats)
        """
        try:
            embeddings = self.model.encode(tokens)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating token embeddings: {str(e)}")
            # Return zeros as fallback
            return [[0.0] * self.embedding_dim for _ in tokens]
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2)
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
