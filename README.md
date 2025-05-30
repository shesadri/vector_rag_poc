# Vector RAG POC with Elasticsearch

A Proof of Concept demonstrating how vector databases with Retrieval-Augmented Generation (RAG) can enhance LLM responses through contextual document retrieval.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG systems:

1. **Retrieve** relevant documents from a vector database based on semantic similarity
2. **Augment** the original query with contextual information from retrieved documents
3. **Generate** more accurate and contextual responses using the enhanced prompt

## How Vector Databases Help

Vector databases like Elasticsearch store documents as high-dimensional vectors (embeddings) that capture semantic meaning. This enables:

- **Semantic Search**: Find documents based on meaning, not just keyword matching
- **Contextual Retrieval**: Get the most relevant context for any query
- **Scalable Knowledge**: Handle millions of documents efficiently
- **Real-time Updates**: Add new knowledge without retraining models

## Project Structure

```
vector_rag_poc/
├── app.py                 # FastAPI application with search endpoints
├── elasticsearch_client.py # Elasticsearch operations
├── vector_embeddings.py   # Embedding generation utilities
├── data_ingestion.py      # Script to populate sample data
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Elasticsearch setup
├── .env.example         # Environment variables template
├── sample_data/         # Sample documents
│   ├── tech_articles.json
│   ├── business_docs.json
│   └── science_papers.json
└── tests/              # Unit tests
    └── test_search.py
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/shesadri/vector_rag_poc.git
cd vector_rag_poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

### 2. Start Elasticsearch

```bash
# Using Docker Compose
docker-compose up -d

# Wait for Elasticsearch to be ready
curl -X GET "localhost:9200/_cluster/health?wait_for_status=yellow&timeout=30s"
```

### 3. Ingest Sample Data

```bash
# Load sample documents into Elasticsearch
python data_ingestion.py
```

### 4. Start the API Server

```bash
# Run the FastAPI application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the RAG System

```bash
# Semantic search query
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "max_results": 5}'

# RAG-enhanced query for LLM
curl -X POST "http://localhost:8000/rag-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest trends in AI?", "max_context": 3}'
```

## API Endpoints

### POST /search
Performs semantic search and returns relevant documents.

**Request:**
```json
{
  "query": "artificial intelligence trends",
  "max_results": 5,
  "min_score": 0.7
}
```

**Response:**
```json
{
  "query": "artificial intelligence trends",
  "total_results": 12,
  "results": [
    {
      "id": "doc_001",
      "title": "AI Trends 2024",
      "content": "Latest developments in AI...",
      "score": 0.95,
      "category": "technology"
    }
  ]
}
```

### POST /rag-query
Generates an enhanced prompt with relevant context for LLM consumption.

**Request:**
```json
{
  "query": "Explain quantum computing applications",
  "max_context": 3,
  "include_sources": true
}
```

**Response:**
```json
{
  "original_query": "Explain quantum computing applications",
  "enhanced_prompt": "Based on the following context, explain quantum computing applications:\n\nContext 1: [Quantum Computing in Cryptography]...\nContext 2: [Quantum Algorithms for Optimization]...\n\nQuery: Explain quantum computing applications",
  "context_sources": [
    {"id": "doc_15", "title": "Quantum Computing in Cryptography", "score": 0.92},
    {"id": "doc_23", "title": "Quantum Algorithms for Optimization", "score": 0.89}
  ]
}
```

### GET /health
Health check endpoint to verify service status.

## Sample Data Categories

The POC includes diverse sample data to demonstrate various use cases:

- **Technology Articles**: AI, machine learning, cloud computing, cybersecurity
- **Business Documents**: Market analysis, financial reports, strategy papers
- **Scientific Papers**: Research findings, experimental results, theoretical concepts
- **Product Documentation**: User guides, API references, troubleshooting guides

## Key Features

1. **Multiple Embedding Models**: Support for different embedding models (sentence-transformers, OpenAI, etc.)
2. **Configurable Search**: Adjustable similarity thresholds and result limits
3. **Metadata Filtering**: Filter results by category, date, or custom fields
4. **Hybrid Search**: Combine vector similarity with traditional keyword search
5. **Real-time Indexing**: Add new documents without service interruption
6. **Performance Monitoring**: Track search latency and relevance metrics

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   FastAPI App    │───▶│  Elasticsearch  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Embedding Model  │    │ Vector Index    │
                    │ (sentence-bert)  │    │ (dense_vector)  │
                    └──────────────────┘    └─────────────────┘
```

## Benefits for LLM Integration

1. **Reduced Hallucination**: Provide factual context from your knowledge base
2. **Domain-Specific Knowledge**: Include specialized information not in training data
3. **Up-to-date Information**: Access real-time data and recent documents
4. **Cost Optimization**: Reduce token usage by providing only relevant context
5. **Improved Accuracy**: Ground responses in verifiable source material

## Next Steps

- Integrate with actual LLM APIs (OpenAI, Anthropic, etc.)
- Add document chunking strategies for long texts
- Implement re-ranking algorithms for better relevance
- Add evaluation metrics for RAG quality
- Scale with production-ready Elasticsearch cluster

## Contributing

Contributions are welcome! Please read the contribution guidelines and submit pull requests for any improvements.

## License

MIT License - see LICENSE file for details.