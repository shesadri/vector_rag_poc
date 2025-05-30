#!/usr/bin/env python3
"""
Data ingestion script for Vector RAG POC
Loads sample documents into Elasticsearch with embeddings
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger

from elasticsearch_client import es_client
from vector_embeddings import embedding_model
from config import settings

def create_sample_data() -> List[Dict[str, Any]]:
    """Create diverse sample documents for testing"""
    
    tech_articles = [
        {
            "title": "Introduction to Machine Learning Algorithms",
            "content": "Machine learning algorithms are computational methods that enable computers to learn patterns from data without being explicitly programmed. The main categories include supervised learning (classification and regression), unsupervised learning (clustering and dimensionality reduction), and reinforcement learning. Popular algorithms include linear regression, decision trees, random forests, support vector machines, neural networks, and k-means clustering. Each algorithm has its strengths and is suitable for different types of problems and data characteristics.",
            "category": "technology",
            "tags": ["machine-learning", "algorithms", "data-science", "AI"]
        },
        {
            "title": "Cloud Computing Best Practices for Enterprise",
            "content": "Cloud computing has revolutionized how enterprises manage their IT infrastructure. Key best practices include implementing proper security measures with multi-factor authentication, using Infrastructure as Code (IaC) for consistent deployments, establishing cost monitoring and optimization strategies, designing for scalability and fault tolerance, and maintaining data backup and disaster recovery plans. Popular cloud platforms include AWS, Azure, and Google Cloud Platform, each offering unique services and pricing models.",
            "category": "technology",
            "tags": ["cloud-computing", "enterprise", "infrastructure", "security"]
        },
        {
            "title": "Cybersecurity Threats in 2024: Prevention and Response",
            "content": "The cybersecurity landscape continues to evolve with sophisticated threats including advanced persistent threats (APTs), ransomware attacks, phishing campaigns, and supply chain attacks. Organizations must implement layered security approaches including endpoint detection and response (EDR), security information and event management (SIEM), zero-trust architecture, regular security training for employees, and incident response planning. Emerging threats include AI-powered attacks and quantum computing implications for encryption.",
            "category": "technology",
            "tags": ["cybersecurity", "threats", "prevention", "enterprise-security"]
        },
        {
            "title": "The Future of Artificial Intelligence and Ethics",
            "content": "Artificial Intelligence is rapidly advancing with developments in large language models, computer vision, robotics, and autonomous systems. However, these advances raise important ethical considerations including bias in AI systems, privacy concerns, job displacement, and the need for AI governance. Organizations are implementing responsible AI frameworks, establishing AI ethics committees, and developing guidelines for fair and transparent AI deployment. The future will require balancing innovation with ethical responsibility.",
            "category": "technology",
            "tags": ["artificial-intelligence", "ethics", "governance", "future-tech"]
        },
        {
            "title": "Microservices Architecture: Design Patterns and Implementation",
            "content": "Microservices architecture breaks down monolithic applications into smaller, independent services that communicate through APIs. Key design patterns include API Gateway, Circuit Breaker, Service Discovery, Event Sourcing, and CQRS (Command Query Responsibility Segregation). Implementation considerations include containerization with Docker, orchestration with Kubernetes, service mesh for communication, distributed tracing for monitoring, and handling eventual consistency. Benefits include scalability, technology diversity, and team autonomy.",
            "category": "technology",
            "tags": ["microservices", "architecture", "design-patterns", "scalability"]
        }
    ]
    
    business_docs = [
        {
            "title": "Digital Transformation Strategy for Traditional Industries",
            "content": "Digital transformation is no longer optional for traditional industries. Successful transformation requires a comprehensive strategy that includes technology modernization, process automation, data-driven decision making, and cultural change management. Key components include cloud migration, customer experience digitization, supply chain optimization, and workforce reskilling. Industries like manufacturing, healthcare, finance, and retail are leveraging IoT, AI, and analytics to create competitive advantages and improve operational efficiency.",
            "category": "business",
            "tags": ["digital-transformation", "strategy", "innovation", "change-management"]
        },
        {
            "title": "Sustainable Business Practices and ESG Reporting",
            "content": "Environmental, Social, and Governance (ESG) criteria have become central to business strategy and investor decisions. Companies are implementing sustainable practices including carbon footprint reduction, circular economy principles, ethical supply chain management, and diverse hiring practices. ESG reporting frameworks like GRI, SASB, and TCFD provide standards for measuring and communicating sustainability performance. Benefits include improved brand reputation, risk mitigation, cost savings, and access to sustainable financing.",
            "category": "business",
            "tags": ["sustainability", "ESG", "reporting", "corporate-responsibility"]
        },
        {
            "title": "Market Analysis: Global E-commerce Trends 2024",
            "content": "The global e-commerce market continues to grow rapidly, driven by mobile commerce, social commerce, and cross-border trade. Key trends include personalization through AI and machine learning, voice commerce, augmented reality shopping experiences, subscription-based models, and sustainable packaging. Emerging markets show the highest growth rates, while established markets focus on omnichannel experiences and customer retention. Challenges include supply chain disruptions, cybersecurity threats, and regulatory compliance across different regions.",
            "category": "business",
            "tags": ["e-commerce", "market-analysis", "trends", "retail"]
        },
        {
            "title": "Financial Planning and Risk Management in Uncertain Times",
            "content": "Financial planning has become more complex due to economic volatility, geopolitical tensions, and technological disruption. Effective risk management strategies include diversification across asset classes and geographies, stress testing financial scenarios, maintaining adequate liquidity reserves, and implementing hedging strategies. Organizations are adopting dynamic budgeting, real-time financial monitoring, and scenario-based planning. Key considerations include inflation hedging, currency risk, regulatory changes, and climate-related financial risks.",
            "category": "business",
            "tags": ["financial-planning", "risk-management", "economic-uncertainty", "strategy"]
        }
    ]
    
    science_papers = [
        {
            "title": "Climate Change Impacts on Biodiversity and Ecosystem Services",
            "content": "Climate change is fundamentally altering Earth's ecosystems, affecting species distribution, migration patterns, and ecosystem services. Research shows that rising temperatures, changing precipitation patterns, and extreme weather events are causing habitat loss, species extinction, and disruption of ecological networks. Ecosystem services such as pollination, water purification, and carbon sequestration are being compromised. Conservation strategies include protected area expansion, corridor creation, assisted migration, and ecosystem restoration. Urgent action is needed to prevent irreversible biodiversity loss.",
            "category": "science",
            "tags": ["climate-change", "biodiversity", "ecosystem", "conservation"]
        },
        {
            "title": "Quantum Computing: From Theory to Practical Applications",
            "content": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally new ways. Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously, enabling exponential computational speedups for certain problems. Current applications include cryptography, optimization, drug discovery, and financial modeling. Challenges include quantum decoherence, error correction, and scalability. Leading companies and research institutions are making progress toward fault-tolerant quantum computers that could revolutionize computing.",
            "category": "science",
            "tags": ["quantum-computing", "physics", "technology", "research"]
        },
        {
            "title": "Gene Therapy Advances in Treating Genetic Disorders",
            "content": "Gene therapy has emerged as a promising treatment for genetic disorders, offering the potential to correct defective genes at their source. Recent advances include CRISPR-Cas9 gene editing, viral vector delivery systems, and base editing techniques. Successful treatments have been developed for conditions like severe combined immunodeficiency (SCID), sickle cell disease, and inherited blindness. Challenges include delivery efficiency, immune responses, off-target effects, and ethical considerations. Ongoing research focuses on improving safety, expanding applications, and reducing costs.",
            "category": "science",
            "tags": ["gene-therapy", "genetics", "medicine", "biotechnology"]
        },
        {
            "title": "Renewable Energy Technologies: Efficiency and Storage Solutions",
            "content": "Renewable energy technologies have achieved significant improvements in efficiency and cost-effectiveness. Solar photovoltaic systems now achieve over 26% efficiency in commercial applications, while wind turbines have grown larger and more efficient. Key challenges include energy storage and grid integration. Battery technologies, particularly lithium-ion and emerging solid-state batteries, are improving energy density and reducing costs. Other storage solutions include pumped hydro, compressed air, and hydrogen production. Smart grid technologies enable better integration of variable renewable sources.",
            "category": "science",
            "tags": ["renewable-energy", "solar", "wind", "energy-storage"]
        }
    ]
    
    product_docs = [
        {
            "title": "API Authentication Guide: OAuth 2.0 Implementation",
            "content": "This guide covers implementing OAuth 2.0 authentication for our API services. OAuth 2.0 provides secure authorization flows for web applications, mobile apps, and server-to-server communication. The implementation includes authorization code flow for web apps, client credentials flow for server applications, and PKCE (Proof Key for Code Exchange) for mobile and single-page applications. Required endpoints include authorization, token, and token introspection. Security considerations include using HTTPS, validating redirect URIs, implementing rate limiting, and proper token storage.",
            "category": "documentation",
            "tags": ["API", "authentication", "OAuth", "security", "development"]
        },
        {
            "title": "Database Performance Optimization Best Practices",
            "content": "Database performance optimization is crucial for application scalability and user experience. Key strategies include proper indexing based on query patterns, query optimization using execution plans, database normalization and denormalization trade-offs, connection pooling, and caching strategies. Monitoring tools help identify slow queries, lock contention, and resource bottlenecks. Advanced techniques include partitioning, sharding, read replicas, and database clustering. Regular maintenance tasks include statistics updates, index rebuilding, and query plan cache management.",
            "category": "documentation",
            "tags": ["database", "performance", "optimization", "indexing", "scalability"]
        },
        {
            "title": "Troubleshooting Common Network Connectivity Issues",
            "content": "Network connectivity issues can significantly impact application performance and user experience. Common problems include DNS resolution failures, firewall blocking, SSL/TLS certificate issues, and network latency. Diagnostic tools include ping, traceroute, nslookup, telnet, and packet capture analysis. Systematic troubleshooting involves checking physical connectivity, network configuration, DNS settings, firewall rules, and application logs. Best practices include implementing health checks, monitoring network metrics, maintaining network documentation, and establishing escalation procedures.",
            "category": "documentation",
            "tags": ["networking", "troubleshooting", "connectivity", "diagnostics", "support"]
        }
    ]
    
    # Combine all documents
    all_documents = tech_articles + business_docs + science_papers + product_docs
    
    # Add metadata and timestamps
    base_date = datetime.utcnow() - timedelta(days=365)
    for i, doc in enumerate(all_documents):
        doc.update({
            "id": f"doc_{i+1:03d}",
            "metadata": {
                "author": f"Author {(i % 5) + 1}",
                "source": "Vector RAG POC Dataset",
                "language": "en",
                "word_count": len(doc["content"].split()),
                "reading_time_minutes": max(1, len(doc["content"].split()) // 200)
            },
            "created_at": (base_date + timedelta(days=i*10)).isoformat(),
            "updated_at": (base_date + timedelta(days=i*10 + 5)).isoformat()
        })
    
    return all_documents

def generate_embeddings_for_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for all documents"""
    logger.info("Generating embeddings for documents...")
    
    # Prepare texts for embedding
    texts = []
    for doc in documents:
        # Combine title and content for embedding
        text = f"{doc['title']} {doc['content']}"
        texts.append(text)
    
    # Generate embeddings in batch
    embeddings = embedding_model.generate_embeddings_batch(texts)
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc['embedding'] = embedding
    
    return documents

def load_data_to_elasticsearch(documents: List[Dict[str, Any]]):
    """Load documents into Elasticsearch"""
    try:
        logger.info(f"Loading {len(documents)} documents into Elasticsearch...")
        
        # Create index (will skip if exists)
        es_client.create_index(force_recreate=False)
        
        # Index documents in batch
        indexed_ids = es_client.index_documents_batch(documents)
        
        logger.info(f"Successfully indexed {len(indexed_ids)} documents")
        
        # Verify indexing
        stats = es_client.get_index_stats()
        doc_count = stats['total']['docs']['count']
        index_size = stats['total']['store']['size_in_bytes']
        
        logger.info(f"Index now contains {doc_count} documents ({index_size:,} bytes)")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def save_sample_data_files(documents: List[Dict[str, Any]]):
    """Save sample data to JSON files for reference"""
    # Create sample_data directory
    os.makedirs("sample_data", exist_ok=True)
    
    # Group documents by category
    by_category = {}
    for doc in documents:
        category = doc['category']
        if category not in by_category:
            by_category[category] = []
        
        # Remove embedding for file storage (too large)
        doc_copy = {k: v for k, v in doc.items() if k != 'embedding'}
        by_category[category].append(doc_copy)
    
    # Save each category to separate file
    for category, docs in by_category.items():
        filename = f"sample_data/{category}_docs.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(docs)} {category} documents to {filename}")

def main():
    """Main data ingestion function"""
    logger.info("Starting data ingestion for Vector RAG POC...")
    
    try:
        # Create sample documents
        logger.info("Creating sample documents...")
        documents = create_sample_data()
        logger.info(f"Created {len(documents)} sample documents")
        
        # Generate embeddings
        documents_with_embeddings = generate_embeddings_for_documents(documents)
        
        # Load to Elasticsearch
        load_data_to_elasticsearch(documents_with_embeddings)
        
        # Save sample data files
        save_sample_data_files(documents)
        
        logger.info("Data ingestion completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("DATA INGESTION SUMMARY")
        print("="*60)
        print(f"Total documents indexed: {len(documents)}")
        print(f"Embedding model: {embedding_model.model_name}")
        print(f"Embedding dimension: {embedding_model.get_embedding_dimension()}")
        print(f"Elasticsearch index: {settings.documents_index}")
        print("\nDocument categories:")
        
        categories = {}
        for doc in documents:
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for category, count in categories.items():
            print(f"  - {category}: {count} documents")
        
        print("\nNext steps:")
        print("1. Start the API server: uvicorn app:app --reload")
        print("2. Test search: curl -X POST 'http://localhost:8000/search' -H 'Content-Type: application/json' -d '{\"query\": \"machine learning\", \"max_results\": 5}'")
        print("3. Try RAG query: curl -X POST 'http://localhost:8000/rag-query' -H 'Content-Type: application/json' -d '{\"query\": \"What are AI trends?\", \"max_context\": 3}'")
        print("4. Access API docs: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()