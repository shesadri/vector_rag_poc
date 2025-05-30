from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from loguru import logger

from config import settings
from elasticsearch_client import es_client
from vector_embeddings import embedding_model

# Initialize FastAPI app
app = FastAPI(
    title="Vector RAG POC API",
    description="Proof of Concept for Vector Database with RAG using Elasticsearch",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    min_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")
    search_type: str = Field(default="vector", description="Search type: vector, text, or hybrid")

class RAGRequest(BaseModel):
    query: str = Field(..., description="User query for RAG enhancement")
    max_context: int = Field(default=3, ge=1, le=10, description="Maximum context documents")
    include_sources: bool = Field(default=True, description="Include source document references")
    min_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")

class DocumentRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(..., description="Document category")
    tags: List[str] = Field(default=[], description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    total_results: int
    search_type: str
    results: List[SearchResult]
    execution_time_ms: float

class RAGResponse(BaseModel):
    original_query: str
    enhanced_prompt: str
    context_sources: List[Dict[str, Any]]
    execution_time_ms: float

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Vector RAG POC API...")
    
    try:
        # Create index if it doesn't exist
        es_client.create_index(force_recreate=False)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        es_health = es_client.health_check()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "elasticsearch": es_health,
            "embedding_model": {
                "name": embedding_model.model_name,
                "dimension": embedding_model.get_embedding_dimension()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using vector similarity, text search, or hybrid approach"""
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Search request: {request.query} (type: {request.search_type})")
        
        # Generate query embedding for vector/hybrid search
        query_embedding = None
        if request.search_type in ["vector", "hybrid"]:
            query_embedding = embedding_model.generate_embedding(request.query)
        
        # Perform search based on type
        if request.search_type == "vector":
            results = es_client.vector_search(
                query_embedding=query_embedding,
                max_results=request.max_results,
                min_score=request.min_score,
                filters=request.filters
            )
        elif request.search_type == "hybrid":
            results = es_client.hybrid_search(
                query_text=request.query,
                query_embedding=query_embedding,
                max_results=request.max_results,
                min_score=request.min_score
            )
        else:
            # Text-only search (fallback)
            # Note: This would require implementing text_search method in es_client
            results = es_client.vector_search(
                query_embedding=embedding_model.generate_embedding(request.query),
                max_results=request.max_results,
                min_score=request.min_score,
                filters=request.filters
            )
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Format results
        formatted_results = [
            SearchResult(
                id=result['id'],
                title=result['title'],
                content=result['content'][:500] + "..." if len(result['content']) > 500 else result['content'],
                category=result['category'],
                tags=result.get('tags', []),
                score=result['score'],
                metadata=result.get('metadata')
            )
            for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            total_results=len(formatted_results),
            search_type=request.search_type,
            results=formatted_results,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/rag-query", response_model=RAGResponse)
async def rag_enhanced_query(request: RAGRequest):
    """Generate RAG-enhanced prompt with relevant context for LLM consumption"""
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"RAG query: {request.query}")
        
        # Generate query embedding
        query_embedding = embedding_model.generate_embedding(request.query)
        
        # Search for relevant context
        context_docs = es_client.vector_search(
            query_embedding=query_embedding,
            max_results=request.max_context,
            min_score=request.min_score
        )
        
        # Build enhanced prompt
        if not context_docs:
            enhanced_prompt = f"Query: {request.query}\n\nNote: No relevant context found in the knowledge base."
            context_sources = []
        else:
            # Construct context sections
            context_sections = []
            for i, doc in enumerate(context_docs, 1):
                context_section = f"Context {i}: [{doc['title']}]\n{doc['content'][:800]}{'...' if len(doc['content']) > 800 else ''}"
                context_sections.append(context_section)
            
            # Build the enhanced prompt
            context_text = "\n\n".join(context_sections)
            enhanced_prompt = f"""Based on the following context, please answer the query. Use the provided information to give accurate and contextual responses.

{context_text}

Query: {request.query}

Instructions: Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the query, please indicate what additional information would be needed."""
            
            # Prepare source information
            context_sources = [
                {
                    "id": doc['id'],
                    "title": doc['title'],
                    "category": doc['category'],
                    "score": doc['score'],
                    "excerpt": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                }
                for doc in context_docs
            ] if request.include_sources else []
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return RAGResponse(
            original_query=request.query,
            enhanced_prompt=enhanced_prompt,
            context_sources=context_sources,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/documents")
async def add_document(request: DocumentRequest):
    """Add a new document to the vector database"""
    try:
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Generate embedding for the content
        content_for_embedding = f"{request.title} {request.content}"
        embedding = embedding_model.generate_embedding(content_for_embedding)
        
        # Prepare document
        document = {
            "id": doc_id,
            "title": request.title,
            "content": request.content,
            "category": request.category,
            "tags": request.tags,
            "metadata": request.metadata or {},
            "embedding": embedding,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Index document
        indexed_id = es_client.index_document(document)
        
        logger.info(f"Added document: {doc_id}")
        
        return {
            "id": indexed_id,
            "title": request.title,
            "category": request.category,
            "status": "indexed",
            "embedding_dimension": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID"""
    try:
        document = es_client.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove embedding from response (too large)
        response_doc = {k: v for k, v in document.items() if k != 'embedding'}
        response_doc['has_embedding'] = 'embedding' in document
        
        return response_doc
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document by ID"""
    try:
        success = es_client.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"id": doc_id, "status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system and index statistics"""
    try:
        index_stats = es_client.get_index_stats()
        
        return {
            "index_name": settings.documents_index,
            "document_count": index_stats['total']['docs']['count'],
            "index_size_bytes": index_stats['total']['store']['size_in_bytes'],
            "embedding_model": embedding_model.model_name,
            "embedding_dimension": embedding_model.get_embedding_dimension(),
            "elasticsearch_version": index_stats.get('version', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )