from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from loguru import logger
from config import settings

class ElasticsearchClient:
    """Elasticsearch client for vector operations"""
    
    def __init__(self):
        self.client = None
        self.index_name = settings.documents_index
        self._connect()
    
    def _connect(self):
        """Connect to Elasticsearch"""
        try:
            # Create Elasticsearch client
            if settings.elasticsearch_username and settings.elasticsearch_password:
                self.client = Elasticsearch(
                    [{
                        'host': settings.elasticsearch_host,
                        'port': settings.elasticsearch_port,
                        'scheme': settings.elasticsearch_scheme
                    }],
                    basic_auth=(settings.elasticsearch_username, settings.elasticsearch_password),
                    verify_certs=False,
                    ssl_show_warn=False
                )
            else:
                self.client = Elasticsearch(
                    [{
                        'host': settings.elasticsearch_host,
                        'port': settings.elasticsearch_port,
                        'scheme': settings.elasticsearch_scheme
                    }],
                    verify_certs=False
                )
            
            # Test connection
            if self.client.ping():
                logger.info(f"Connected to Elasticsearch at {settings.elasticsearch_url}")
            else:
                raise ConnectionError("Failed to ping Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def create_index(self, force_recreate: bool = False):
        """Create the documents index with vector mapping"""
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                if force_recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    return
            
            # Define index mapping with vector field
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "category": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        "metadata": {
                            "type": "object",
                            "dynamic": True
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": settings.embedding_dimension,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "standard": {
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop"]
                            }
                        }
                    }
                }
            }
            
            # Create index
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def index_document(self, document: Dict[str, Any]) -> str:
        """Index a single document"""
        try:
            response = self.client.index(
                index=self.index_name,
                id=document.get('id'),
                body=document
            )
            
            doc_id = response['_id']
            logger.debug(f"Indexed document: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise
    
    def index_documents_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Index multiple documents in batch"""
        try:
            from elasticsearch.helpers import bulk
            
            # Prepare documents for bulk indexing
            actions = []
            for doc in documents:
                action = {
                    "_index": self.index_name,
                    "_id": doc.get('id'),
                    "_source": doc
                }
                actions.append(action)
            
            # Bulk index
            success_count, failed_items = bulk(self.client, actions)
            
            if failed_items:
                logger.warning(f"Failed to index {len(failed_items)} documents")
            
            logger.info(f"Successfully indexed {success_count} documents")
            return [doc.get('id') for doc in documents]
            
        except Exception as e:
            logger.error(f"Failed to batch index documents: {e}")
            raise
    
    def vector_search(self, query_embedding: List[float], 
                     max_results: int = 10, 
                     min_score: float = 0.7,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            # Build the query
            query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": max_results,
                    "num_candidates": max_results * 2
                },
                "_source": ["id", "title", "content", "category", "tags", "metadata", "created_at"]
            }
            
            # Add filters if provided
            if filters:
                query["knn"]["filter"] = self._build_filters(filters)
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query,
                size=max_results
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= min_score:
                    result = hit['_source']
                    result['score'] = hit['_score']
                    results.append(result)
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    def hybrid_search(self, query_text: str, query_embedding: List[float],
                     max_results: int = 10, min_score: float = 0.7,
                     text_weight: float = 0.3, vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector similarity"""
        try:
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["title^2", "content"],
                                    "boost": text_weight
                                }
                            },
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"cosineSimilarity(params.query_vector, 'embedding') * {vector_weight}",
                                        "params": {"query_vector": query_embedding}
                                    }
                                }
                            }
                        ]
                    }
                },
                "min_score": min_score,
                "_source": ["id", "title", "content", "category", "tags", "metadata", "created_at"]
            }
            
            response = self.client.search(
                index=self.index_name,
                body=query,
                size=max_results
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                results.append(result)
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def _build_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Elasticsearch filters from filter dictionary"""
        filter_clauses = []
        
        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"terms": {field: value}})
            else:
                filter_clauses.append({"term": {field: value}})
        
        if len(filter_clauses) == 1:
            return filter_clauses[0]
        else:
            return {"bool": {"must": filter_clauses}}
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            response = self.client.get(index=self.index_name, id=doc_id)
            return response['_source']
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            self.client.delete(index=self.index_name, id=doc_id)
            logger.info(f"Deleted document: {doc_id}")
            return True
        except NotFoundError:
            logger.warning(f"Document not found: {doc_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            response = self.client.indices.stats(index=self.index_name)
            return response['indices'][self.index_name]
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch cluster health"""
        try:
            health = self.client.cluster.health()
            ping = self.client.ping()
            
            return {
                "elasticsearch_ping": ping,
                "cluster_status": health['status'],
                "number_of_nodes": health['number_of_nodes'],
                "active_shards": health['active_shards']
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": str(e)}

# Global Elasticsearch client instance
es_client = ElasticsearchClient()