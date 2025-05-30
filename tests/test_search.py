import pytest
import asyncio
from fastapi.testclient import TestClient
from app import app
from elasticsearch_client import es_client
from vector_embeddings import embedding_model

client = TestClient(app)

class TestVectorRAG:
    """Test cases for Vector RAG POC"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "elasticsearch" in data
        assert "embedding_model" in data
    
    def test_search_endpoint(self):
        """Test search functionality"""
        search_request = {
            "query": "machine learning algorithms",
            "max_results": 5,
            "min_score": 0.5,
            "search_type": "vector"
        }
        
        response = client.post("/search", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert "execution_time_ms" in data
        assert data["query"] == search_request["query"]
    
    def test_rag_query_endpoint(self):
        """Test RAG query functionality"""
        rag_request = {
            "query": "What are the latest trends in artificial intelligence?",
            "max_context": 3,
            "include_sources": True,
            "min_score": 0.5
        }
        
        response = client.post("/rag-query", json=rag_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "original_query" in data
        assert "enhanced_prompt" in data
        assert "context_sources" in data
        assert "execution_time_ms" in data
        assert data["original_query"] == rag_request["query"]
    
    def test_add_document(self):
        """Test adding a new document"""
        document_request = {
            "title": "Test Document",
            "content": "This is a test document for the vector RAG system.",
            "category": "test",
            "tags": ["test", "document"],
            "metadata": {"test": True}
        }
        
        response = client.post("/documents", json=document_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["title"] == document_request["title"]
        assert data["category"] == document_request["category"]
        assert data["status"] == "indexed"
        
        # Clean up - delete the test document
        doc_id = data["id"]
        delete_response = client.delete(f"/documents/{doc_id}")
        assert delete_response.status_code == 200
    
    def test_get_document(self):
        """Test retrieving a document"""
        # First add a document
        document_request = {
            "title": "Test Document for Retrieval",
            "content": "This document will be retrieved in the test.",
            "category": "test",
            "tags": ["test"]
        }
        
        add_response = client.post("/documents", json=document_request)
        assert add_response.status_code == 200
        doc_id = add_response.json()["id"]
        
        # Retrieve the document
        get_response = client.get(f"/documents/{doc_id}")
        assert get_response.status_code == 200
        
        data = get_response.json()
        assert data["title"] == document_request["title"]
        assert data["content"] == document_request["content"]
        assert data["has_embedding"] == True
        
        # Clean up
        client.delete(f"/documents/{doc_id}")
    
    def test_stats_endpoint(self):
        """Test stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "index_name" in data
        assert "document_count" in data
        assert "embedding_model" in data
        assert "embedding_dimension" in data
    
    def test_invalid_search_request(self):
        """Test invalid search request"""
        invalid_request = {
            "max_results": -1,  # Invalid: negative value
            "min_score": 1.5   # Invalid: > 1.0
        }
        
        response = client.post("/search", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_document(self):
        """Test getting non-existent document"""
        response = client.get("/documents/nonexistent-id")
        assert response.status_code == 404
    
    def test_embedding_generation(self):
        """Test embedding generation directly"""
        test_text = "This is a test text for embedding generation."
        
        embedding = embedding_model.generate_embedding(test_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedding_model.get_embedding_dimension()
        assert all(isinstance(x, float) for x in embedding)
    
    def test_elasticsearch_connection(self):
        """Test Elasticsearch connection"""
        health = es_client.health_check()
        
        assert "elasticsearch_ping" in health
        assert health["elasticsearch_ping"] == True

# Run tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])