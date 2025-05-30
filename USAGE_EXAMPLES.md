# Vector RAG POC - Usage Examples

This document provides practical examples of how to use the Vector RAG POC system to demonstrate the benefits of vector databases with Retrieval-Augmented Generation.

## Quick Start Examples

### 1. Basic Vector Search

Search for documents using semantic similarity:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms for classification",
    "max_results": 5,
    "min_score": 0.7,
    "search_type": "vector"
  }'
```

**Response:**
```json
{
  "query": "machine learning algorithms for classification",
  "total_results": 3,
  "search_type": "vector",
  "results": [
    {
      "id": "doc_001",
      "title": "Introduction to Machine Learning Algorithms",
      "content": "Machine learning algorithms are computational methods...",
      "category": "technology",
      "tags": ["machine-learning", "algorithms", "data-science"],
      "score": 0.92
    }
  ],
  "execution_time_ms": 45.2
}
```

### 2. RAG-Enhanced Query

Generate an enhanced prompt with relevant context for LLM consumption:

```bash
curl -X POST "http://localhost:8000/rag-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can businesses implement sustainable practices?",
    "max_context": 3,
    "include_sources": true,
    "min_score": 0.6
  }'
```

**Response:**
```json
{
  "original_query": "How can businesses implement sustainable practices?",
  "enhanced_prompt": "Based on the following context, please answer the query...\n\nContext 1: [Sustainable Business Practices and ESG Reporting]\nEnvironmental, Social, and Governance (ESG) criteria...\n\nQuery: How can businesses implement sustainable practices?",
  "context_sources": [
    {
      "id": "doc_012",
      "title": "Sustainable Business Practices and ESG Reporting",
      "category": "business",
      "score": 0.89,
      "excerpt": "Companies are implementing sustainable practices..."
    }
  ],
  "execution_time_ms": 78.3
}
```

### 3. Add New Document

Add a custom document to the vector database:

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Artificial Intelligence in Healthcare",
    "content": "AI is transforming healthcare through diagnostic imaging, drug discovery, personalized treatment plans, and predictive analytics. Machine learning models can analyze medical images with accuracy matching or exceeding human specialists.",
    "category": "healthcare",
    "tags": ["AI", "healthcare", "diagnostics", "machine-learning"],
    "metadata": {
      "author": "Dr. Jane Smith",
      "publication_date": "2024-01-15",
      "source": "Medical AI Journal"
    }
  }'
```

## Advanced Examples

### 4. Filtered Search

Search within specific categories or tags:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "optimization strategies",
    "max_results": 10,
    "min_score": 0.5,
    "filters": {
      "category": "business",
      "tags": ["strategy"]
    }
  }'
```

### 5. Hybrid Search

Combine vector similarity with traditional text search:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum computing applications",
    "max_results": 5,
    "search_type": "hybrid"
  }'
```

### 6. Multi-Context RAG Query

Get comprehensive context from multiple relevant documents:

```bash
curl -X POST "http://localhost:8000/rag-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key considerations for digital transformation in enterprises?",
    "max_context": 5,
    "include_sources": true,
    "min_score": 0.5
  }'
```

## Demonstrating RAG Benefits

### Before RAG (Standard LLM Query)

**Query:** "What are the latest trends in AI for business?"

**Standard LLM Response:** Generic information based on training data, potentially outdated, no specific sources.

### After RAG (Enhanced with Context)

**Query:** "What are the latest trends in AI for business?"

**RAG-Enhanced Prompt:**
```
Based on the following context, please answer the query. Use the provided information to give accurate and contextual responses.

Context 1: [Digital Transformation Strategy for Traditional Industries]
Digital transformation is no longer optional for traditional industries. Companies are leveraging IoT, AI, and analytics to create competitive advantages...

Context 2: [The Future of Artificial Intelligence and Ethics]
Artificial Intelligence is rapidly advancing with developments in large language models, computer vision, robotics, and autonomous systems...

Query: What are the latest trends in AI for business?

Instructions: Please provide a comprehensive answer based on the context above.
```

**Enhanced LLM Response:** Specific, sourced information from your knowledge base, current and relevant to your domain.

## Python Client Example

```python
import requests
import json

class VectorRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def search(self, query, max_results=5, search_type="vector"):
        response = requests.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_type": search_type
            }
        )
        return response.json()
    
    def rag_query(self, query, max_context=3):
        response = requests.post(
            f"{self.base_url}/rag-query",
            json={
                "query": query,
                "max_context": max_context,
                "include_sources": True
            }
        )
        return response.json()
    
    def add_document(self, title, content, category, tags=None):
        response = requests.post(
            f"{self.base_url}/documents",
            json={
                "title": title,
                "content": content,
                "category": category,
                "tags": tags or []
            }
        )
        return response.json()

# Usage
client = VectorRAGClient()

# Search for relevant documents
results = client.search("cloud computing security")
print(f"Found {results['total_results']} documents")

# Generate RAG-enhanced prompt
rag_response = client.rag_query("How to secure cloud infrastructure?")
print("Enhanced prompt for LLM:")
print(rag_response['enhanced_prompt'])

# Add new knowledge
new_doc = client.add_document(
    title="Zero Trust Security Model",
    content="Zero trust security assumes no trust and verifies every transaction...",
    category="security",
    tags=["zero-trust", "security", "network"]
)
print(f"Added document: {new_doc['id']}")
```

## Use Case Scenarios

### 1. Customer Support Chatbot

```python
# Get relevant context for customer query
query = "How do I troubleshoot network connectivity issues?"
rag_response = client.rag_query(query, max_context=2)

# Send enhanced prompt to LLM
enhanced_prompt = rag_response['enhanced_prompt']
# ... send to OpenAI, Anthropic, etc.
```

### 2. Research Assistant

```python
# Find related research papers
results = client.search("gene therapy CRISPR applications", max_results=10)

# Generate literature review context
rag_response = client.rag_query(
    "What are the current applications of CRISPR in gene therapy?",
    max_context=5
)
```

### 3. Business Intelligence

```python
# Search for market analysis
results = client.search("e-commerce market trends 2024", search_type="hybrid")

# Generate strategic insight prompt
rag_response = client.rag_query(
    "What market opportunities should we focus on for e-commerce growth?",
    max_context=4
)
```

## Performance Monitoring

### Get System Statistics

```bash
curl -X GET "http://localhost:8000/stats"
```

**Response:**
```json
{
  "index_name": "vector_rag_documents",
  "document_count": 25,
  "index_size_bytes": 2048576,
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384
}
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## Tips for Best Results

1. **Query Formulation**: Use descriptive, specific queries for better semantic matching
2. **Content Quality**: Ensure documents have clear titles and well-structured content
3. **Metadata Usage**: Add relevant tags and metadata for better filtering
4. **Threshold Tuning**: Adjust `min_score` based on your quality requirements
5. **Context Size**: Balance between comprehensive context and prompt length limits

## Integration with LLM APIs

### OpenAI Integration Example

```python
import openai
from vector_rag_client import VectorRAGClient

rag_client = VectorRAGClient()
openai.api_key = "your-api-key"

def enhanced_chat_completion(user_query):
    # Get RAG context
    rag_response = rag_client.rag_query(user_query, max_context=3)
    
    # Send enhanced prompt to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": rag_response['enhanced_prompt']}
        ]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": rag_response['context_sources']
    }

# Usage
result = enhanced_chat_completion("Explain quantum computing applications")
print(result['answer'])
print("Sources:", [s['title'] for s in result['sources']])
```

This demonstrates how vector databases with RAG significantly improve LLM responses by providing relevant, accurate, and sourced context from your specific knowledge base.