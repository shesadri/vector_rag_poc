# Vector RAG POC - API Reference

Complete API documentation for the Vector RAG POC system.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for the POC. In production deployments, consider implementing API keys or OAuth 2.0.

## Content Type

All requests should use `Content-Type: application/json` for POST requests.

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error description"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (validation error)
- `404` - Not Found
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error

---

## Endpoints

### Health Check

#### GET /health

Returns system health status and configuration information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-05-30T13:00:00Z",
  "elasticsearch": {
    "elasticsearch_ping": true,
    "cluster_status": "green",
    "number_of_nodes": 1,
    "active_shards": 1
  },
  "embedding_model": {
    "name": "all-MiniLM-L6-v2",
    "dimension": 384
  }
}
```

---

### Vector Search

#### POST /search

Perform semantic search across documents using vector similarity.

**Request Body:**
```json
{
  "query": "machine learning algorithms",
  "max_results": 10,
  "min_score": 0.7,
  "search_type": "vector",
  "filters": {
    "category": "technology",
    "tags": ["AI", "algorithms"]
  }
}
```

**Parameters:**
- `query` (string, required): Search query text
- `max_results` (integer, optional): Maximum results to return (1-100, default: 10)
- `min_score` (float, optional): Minimum similarity score (0.0-1.0, default: 0.7)
- `search_type` (string, optional): Search type - "vector", "hybrid" (default: "vector")
- `filters` (object, optional): Additional filters to apply

**Response:**
```json
{
  "query": "machine learning algorithms",
  "total_results": 3,
  "search_type": "vector",
  "results": [
    {
      "id": "doc_001",
      "title": "Introduction to Machine Learning Algorithms",
      "content": "Machine learning algorithms are computational methods that enable computers to learn patterns from data...",
      "category": "technology",
      "tags": ["machine-learning", "algorithms", "data-science", "AI"],
      "score": 0.92,
      "metadata": {
        "author": "Dr. Sarah Chen",
        "reading_time": "8 minutes"
      }
    }
  ],
  "execution_time_ms": 45.2
}
```

---

### RAG Query

#### POST /rag-query

Generate an enhanced prompt with relevant context for LLM consumption.

**Request Body:**
```json
{
  "query": "What are the benefits of microservices architecture?",
  "max_context": 3,
  "include_sources": true,
  "min_score": 0.6
}
```

**Parameters:**
- `query` (string, required): User query for RAG enhancement
- `max_context` (integer, optional): Maximum context documents (1-10, default: 3)
- `include_sources` (boolean, optional): Include source document references (default: true)
- `min_score` (float, optional): Minimum similarity score (0.0-1.0, default: 0.7)

**Response:**
```json
{
  "original_query": "What are the benefits of microservices architecture?",
  "enhanced_prompt": "Based on the following context, please answer the query. Use the provided information to give accurate and contextual responses.\n\nContext 1: [Microservices Architecture: Design Patterns and Implementation]\nMicroservices architecture breaks down monolithic applications into smaller, independent services that communicate through APIs...\n\nQuery: What are the benefits of microservices architecture?\n\nInstructions: Please provide a comprehensive answer based on the context above.",
  "context_sources": [
    {
      "id": "doc_015",
      "title": "Microservices Architecture: Design Patterns and Implementation",
      "category": "technology",
      "score": 0.89,
      "excerpt": "Microservices architecture breaks down monolithic applications into smaller, independent services..."
    }
  ],
  "execution_time_ms": 78.3
}
```

---

### Document Management

#### POST /documents

Add a new document to the vector database.

**Request Body:**
```json
{
  "title": "Kubernetes Best Practices",
  "content": "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications. Best practices include using namespaces for organization, implementing resource limits, setting up health checks, and following security guidelines.",
  "category": "technology",
  "tags": ["kubernetes", "containers", "devops", "orchestration"],
  "metadata": {
    "author": "DevOps Team",
    "source": "Internal Documentation",
    "difficulty": "intermediate"
  }
}
```

**Parameters:**
- `title` (string, required): Document title
- `content` (string, required): Document content
- `category` (string, required): Document category
- `tags` (array, optional): List of tags
- `metadata` (object, optional): Additional metadata

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Kubernetes Best Practices",
  "category": "technology",
  "status": "indexed",
  "embedding_dimension": 384
}
```

#### GET /documents/{doc_id}

Retrieve a specific document by ID.

**Parameters:**
- `doc_id` (string, required): Document ID

**Response:**
```json
{
  "id": "doc_001",
  "title": "Introduction to Machine Learning Algorithms",
  "content": "Machine learning algorithms are computational methods...",
  "category": "technology",
  "tags": ["machine-learning", "algorithms", "data-science", "AI"],
  "metadata": {
    "author": "Dr. Sarah Chen",
    "source": "ML Research Journal"
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "has_embedding": true
}
```

#### DELETE /documents/{doc_id}

Delete a document by ID.

**Parameters:**
- `doc_id` (string, required): Document ID

**Response:**
```json
{
  "id": "doc_001",
  "status": "deleted"
}
```

---

### System Information

#### GET /stats

Get system and index statistics.

**Response:**
```json
{
  "index_name": "vector_rag_documents",
  "document_count": 25,
  "index_size_bytes": 2048576,
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "elasticsearch_version": "8.11.1"
}
```

---

## Response Models

### SearchResult

```json
{
  "id": "string",
  "title": "string",
  "content": "string",
  "category": "string",
  "tags": ["string"],
  "score": 0.95,
  "metadata": {}
}
```

### ContextSource

```json
{
  "id": "string",
  "title": "string",
  "category": "string",
  "score": 0.89,
  "excerpt": "string"
}
```

---

## Usage Examples

### Python Client

```python
import requests

class VectorRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def search(self, query, **kwargs):
        response = requests.post(
            f"{self.base_url}/search",
            json={"query": query, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def rag_query(self, query, **kwargs):
        response = requests.post(
            f"{self.base_url}/rag-query",
            json={"query": query, **kwargs}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = VectorRAGClient()
results = client.search("machine learning")
rag_response = client.rag_query("Explain neural networks")
```

### JavaScript Client

```javascript
class VectorRAGClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async search(query, options = {}) {
        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, ...options })
        });
        return response.json();
    }
    
    async ragQuery(query, options = {}) {
        const response = await fetch(`${this.baseUrl}/rag-query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, ...options })
        });
        return response.json();
    }
}

// Usage
const client = new VectorRAGClient();
const results = await client.search('cloud computing');
const ragResponse = await client.ragQuery('What is containerization?');
```

---

## Rate Limiting

Currently, no rate limiting is implemented in the POC. For production deployments, consider implementing:

- Request rate limits per IP/API key
- Concurrent request limits
- Resource usage quotas

## Caching

The system implements internal caching for:

- Embedding model loading
- Elasticsearch connection pooling
- Query result caching (future enhancement)

## Performance Considerations

- Embedding generation: ~50-200ms per query
- Vector search: ~10-100ms depending on index size
- Document indexing: ~100-500ms per document
- Batch operations recommended for multiple documents

## OpenAPI Specification

Complete OpenAPI specification available at:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- JSON schema: `http://localhost:8000/openapi.json`