import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings

class VectorEmbeddings:
    """Vector embeddings generator using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            # Clean and preprocess text
            text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode([text])[0]
            
            # Convert to list for JSON serialization
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            # Clean and preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(processed_texts)
            
            # Convert to list of lists for JSON serialization
            return [embedding.tolist() for embedding in embeddings]
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation"""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (model dependent)
        max_length = 512  # Most sentence transformers have this limit
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        return text
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()

# Global embedding model instance
embedding_model = VectorEmbeddings()