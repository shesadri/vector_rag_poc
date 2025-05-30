#!/usr/bin/env python3
"""
Benchmarking script for Vector RAG POC
Measures search performance, embedding generation, and system throughput
"""

import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm
from loguru import logger

class VectorRAGBenchmark:
    """Benchmark suite for Vector RAG system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self) -> bool:
        """Check if the system is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform a search request"""
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": query, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def rag_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform a RAG query request"""
        response = self.session.post(
            f"{self.base_url}/rag-query",
            json={"query": query, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def measure_search_performance(self, queries: List[str], iterations: int = 10) -> Dict[str, Any]:
        """Measure search performance across multiple queries"""
        logger.info(f"Measuring search performance with {len(queries)} queries, {iterations} iterations")
        
        all_times = []
        query_results = {}
        
        for query in tqdm(queries, desc="Testing queries"):
            query_times = []
            
            for _ in range(iterations):
                start_time = time.time()
                try:
                    result = self.search(query, max_results=5)
                    query_times.append(time.time() - start_time)
                    all_times.append(time.time() - start_time)
                except Exception as e:
                    logger.error(f"Query failed: {query[:50]}... - {e}")
                    continue
            
            if query_times:
                query_results[query] = {
                    "avg_time": statistics.mean(query_times),
                    "min_time": min(query_times),
                    "max_time": max(query_times),
                    "std_dev": statistics.stdev(query_times) if len(query_times) > 1 else 0
                }
        
        return {
            "total_queries": len(queries),
            "total_requests": len(all_times),
            "overall_avg_time": statistics.mean(all_times) if all_times else 0,
            "overall_min_time": min(all_times) if all_times else 0,
            "overall_max_time": max(all_times) if all_times else 0,
            "overall_std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0,
            "per_query_results": query_results
        }
    
    def measure_rag_performance(self, queries: List[str], iterations: int = 5) -> Dict[str, Any]:
        """Measure RAG query performance"""
        logger.info(f"Measuring RAG performance with {len(queries)} queries, {iterations} iterations")
        
        all_times = []
        context_counts = []
        
        for query in tqdm(queries, desc="Testing RAG queries"):
            for _ in range(iterations):
                start_time = time.time()
                try:
                    result = self.rag_query(query, max_context=3)
                    all_times.append(time.time() - start_time)
                    context_counts.append(len(result.get('context_sources', [])))
                except Exception as e:
                    logger.error(f"RAG query failed: {query[:50]}... - {e}")
                    continue
        
        return {
            "total_queries": len(queries),
            "total_requests": len(all_times),
            "avg_time": statistics.mean(all_times) if all_times else 0,
            "min_time": min(all_times) if all_times else 0,
            "max_time": max(all_times) if all_times else 0,
            "std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0,
            "avg_context_count": statistics.mean(context_counts) if context_counts else 0
        }
    
    def measure_concurrent_performance(self, query: str, concurrent_requests: int = 10, total_requests: int = 100) -> Dict[str, Any]:
        """Measure performance under concurrent load"""
        logger.info(f"Testing concurrent performance: {concurrent_requests} concurrent, {total_requests} total")
        
        def make_request():
            start_time = time.time()
            try:
                self.search(query, max_results=5)
                return time.time() - start_time
            except Exception as e:
                logger.error(f"Concurrent request failed: {e}")
                return None
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        successful_requests = [r for r in results if r is not None]
        
        return {
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": total_requests - len(successful_requests),
            "total_time": total_time,
            "requests_per_second": len(successful_requests) / total_time if total_time > 0 else 0,
            "avg_response_time": statistics.mean(successful_requests) if successful_requests else 0,
            "min_response_time": min(successful_requests) if successful_requests else 0,
            "max_response_time": max(successful_requests) if successful_requests else 0
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run a comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark")
        
        if not self.health_check():
            raise RuntimeError("System health check failed - ensure the API is running")
        
        # Test queries across different categories
        test_queries = [
            "machine learning algorithms",
            "cloud computing security",
            "sustainable business practices",
            "quantum computing applications",
            "microservices architecture",
            "artificial intelligence ethics",
            "database performance optimization",
            "renewable energy technologies",
            "gene therapy advances",
            "digital transformation strategy"
        ]
        
        rag_queries = [
            "How can businesses implement AI responsibly?",
            "What are the benefits of microservices?",
            "How does quantum computing work?",
            "What are sustainable energy solutions?",
            "How to optimize database performance?"
        ]
        
        results = {
            "timestamp": time.time(),
            "system_info": self.get_system_info(),
            "search_performance": self.measure_search_performance(test_queries),
            "rag_performance": self.measure_rag_performance(rag_queries),
            "concurrent_performance": self.measure_concurrent_performance(
                "machine learning algorithms", 
                concurrent_requests=5, 
                total_requests=50
            )
        }
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            health_response = self.session.get(f"{self.base_url}/health")
            stats_response = self.session.get(f"{self.base_url}/stats")
            
            return {
                "health": health_response.json() if health_response.status_code == 200 else {},
                "stats": stats_response.json() if stats_response.status_code == 200 else {}
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a readable format"""
        print("\n" + "="*80)
        print("VECTOR RAG POC - BENCHMARK RESULTS")
        print("="*80)
        
        # System Info
        print("\n📊 SYSTEM INFORMATION:")
        system_info = results.get('system_info', {})
        stats = system_info.get('stats', {})
        health = system_info.get('health', {})
        
        print(f"  Document Count: {stats.get('document_count', 'N/A')}")
        print(f"  Index Size: {stats.get('index_size_bytes', 0):,} bytes")
        print(f"  Embedding Model: {stats.get('embedding_model', 'N/A')}")
        print(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")
        
        # Search Performance
        print("\n🔍 SEARCH PERFORMANCE:")
        search_perf = results.get('search_performance', {})
        print(f"  Total Queries Tested: {search_perf.get('total_queries', 0)}")
        print(f"  Total Requests: {search_perf.get('total_requests', 0)}")
        print(f"  Average Response Time: {search_perf.get('overall_avg_time', 0)*1000:.1f}ms")
        print(f"  Min Response Time: {search_perf.get('overall_min_time', 0)*1000:.1f}ms")
        print(f"  Max Response Time: {search_perf.get('overall_max_time', 0)*1000:.1f}ms")
        print(f"  Standard Deviation: {search_perf.get('overall_std_dev', 0)*1000:.1f}ms")
        
        # RAG Performance
        print("\n🤖 RAG PERFORMANCE:")
        rag_perf = results.get('rag_performance', {})
        print(f"  Total RAG Queries: {rag_perf.get('total_queries', 0)}")
        print(f"  Average Response Time: {rag_perf.get('avg_time', 0)*1000:.1f}ms")
        print(f"  Min Response Time: {rag_perf.get('min_time', 0)*1000:.1f}ms")
        print(f"  Max Response Time: {rag_perf.get('max_time', 0)*1000:.1f}ms")
        print(f"  Average Context Documents: {rag_perf.get('avg_context_count', 0):.1f}")
        
        # Concurrent Performance
        print("\n⚡ CONCURRENT PERFORMANCE:")
        concurrent_perf = results.get('concurrent_performance', {})
        print(f"  Concurrent Requests: {concurrent_perf.get('concurrent_requests', 0)}")
        print(f"  Total Requests: {concurrent_perf.get('total_requests', 0)}")
        print(f"  Successful Requests: {concurrent_perf.get('successful_requests', 0)}")
        print(f"  Failed Requests: {concurrent_perf.get('failed_requests', 0)}")
        print(f"  Requests per Second: {concurrent_perf.get('requests_per_second', 0):.1f}")
        print(f"  Average Response Time: {concurrent_perf.get('avg_response_time', 0)*1000:.1f}ms")
        
        # Performance Summary
        print("\n📈 PERFORMANCE SUMMARY:")
        search_avg = search_perf.get('overall_avg_time', 0) * 1000
        rag_avg = rag_perf.get('avg_time', 0) * 1000
        rps = concurrent_perf.get('requests_per_second', 0)
        
        if search_avg < 100:
            print(f"  ✅ Search Performance: EXCELLENT ({search_avg:.1f}ms)")
        elif search_avg < 500:
            print(f"  ✅ Search Performance: GOOD ({search_avg:.1f}ms)")
        else:
            print(f"  ⚠️  Search Performance: NEEDS OPTIMIZATION ({search_avg:.1f}ms)")
        
        if rag_avg < 500:
            print(f"  ✅ RAG Performance: EXCELLENT ({rag_avg:.1f}ms)")
        elif rag_avg < 1000:
            print(f"  ✅ RAG Performance: GOOD ({rag_avg:.1f}ms)")
        else:
            print(f"  ⚠️  RAG Performance: NEEDS OPTIMIZATION ({rag_avg:.1f}ms)")
        
        if rps > 10:
            print(f"  ✅ Throughput: EXCELLENT ({rps:.1f} req/s)")
        elif rps > 5:
            print(f"  ✅ Throughput: GOOD ({rps:.1f} req/s)")
        else:
            print(f"  ⚠️  Throughput: NEEDS OPTIMIZATION ({rps:.1f} req/s)")
        
        print("\n" + "="*80)

def main():
    """Run benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector RAG POC Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--search-only", action="store_true", help="Run only search benchmarks")
    parser.add_argument("--rag-only", action="store_true", help="Run only RAG benchmarks")
    parser.add_argument("--concurrent-only", action="store_true", help="Run only concurrent benchmarks")
    
    args = parser.parse_args()
    
    benchmark = VectorRAGBenchmark(args.url)
    
    try:
        if args.search_only:
            test_queries = ["machine learning", "cloud computing", "artificial intelligence"]
            results = {"search_performance": benchmark.measure_search_performance(test_queries)}
        elif args.rag_only:
            rag_queries = ["How does AI work?", "What is cloud computing?"]
            results = {"rag_performance": benchmark.measure_rag_performance(rag_queries)}
        elif args.concurrent_only:
            results = {"concurrent_performance": benchmark.measure_concurrent_performance("test query")}
        else:
            results = benchmark.run_comprehensive_benchmark()
        
        benchmark.print_results(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()