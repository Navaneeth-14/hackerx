#!/usr/bin/env python3
"""
Performance analysis script to identify bottlenecks
"""

import os
import sys
import time
import cProfile
import pstats
from pathlib import Path

def analyze_performance():
    """Analyze performance of each component"""
    print("⚡ PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    try:
        from rag_system import AdvancedRAGSystem
        from document_processer import AdvancedDocumentProcessor
        from vector_database import VectorDatabase
        from query_parser import AdvancedQueryParser
        from llm_reasoning import AdvancedLLMReasoning
        
        file_path = "doc2.pdf"
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📄 Testing with file: {file_path}")
        
        # Test 1: Document Processing Performance
        print("\n1️⃣ DOCUMENT PROCESSING PERFORMANCE")
        print("-" * 40)
        
        doc_processor = AdvancedDocumentProcessor()
        start_time = time.time()
        chunks = doc_processor.process_document(file_path, use_ocr=False)
        doc_time = time.time() - start_time
        
        print(f"✅ Document processing: {doc_time:.2f}s")
        print(f"📊 Chunks created: {len(chunks)}")
        print(f"📊 Average time per chunk: {doc_time/len(chunks):.4f}s")
        
        # Test 2: Vector Database Performance
        print("\n2️⃣ VECTOR DATABASE PERFORMANCE")
        print("-" * 40)
        
        vector_db = VectorDatabase()
        start_time = time.time()
        success = vector_db.add_documents(chunks)
        vector_time = time.time() - start_time
        
        print(f"✅ Vector database addition: {vector_time:.2f}s")
        print(f"📊 Success: {success}")
        print(f"📊 Average time per chunk: {vector_time/len(chunks):.4f}s")
        
        # Test 3: Query Parser Performance
        print("\n3️⃣ QUERY PARSER PERFORMANCE")
        print("-" * 40)
        
        query_parser = AdvancedQueryParser()
        test_query = "Does the policy cover newborn care after hospital discharge?"
        
        start_time = time.time()
        parsed = query_parser.parse_query(test_query)
        parser_time = time.time() - start_time
        
        print(f"✅ Query parsing: {parser_time:.2f}s")
        print(f"📊 Query type: {parsed.query_type}")
        print(f"📊 Confidence: {parsed.confidence}")
        
        # Test 4: LLM Reasoning Performance
        print("\n4️⃣ LLM REASONING PERFORMANCE")
        print("-" * 40)
        
        reasoning_engine = AdvancedLLMReasoning(use_gpu=False)
        test_context = [{"content": "Sample policy content", "source_file": "test.pdf"}]
        
        start_time = time.time()
        result = reasoning_engine.analyze_query(test_query, test_context, "coverage_inquiry")
        reasoning_time = time.time() - start_time
        
        print(f"✅ LLM reasoning: {reasoning_time:.2f}s")
        print(f"📊 Decision: {result.decision}")
        print(f"📊 Confidence: {result.confidence_score}")
        
        # Test 5: Full RAG System Performance
        print("\n5️⃣ FULL RAG SYSTEM PERFORMANCE")
        print("-" * 40)
        
        rag_system = AdvancedRAGSystem(use_gpu=False)
        
        # Document ingestion
        start_time = time.time()
        rag_chunks = rag_system.ingest_document(file_path, use_ocr=False)
        ingestion_time = time.time() - start_time
        
        print(f"✅ Document ingestion: {ingestion_time:.2f}s")
        print(f"📊 Chunks ingested: {len(rag_chunks)}")
        
        # Query processing
        start_time = time.time()
        query_result = rag_system.process_query(test_query)
        query_time = time.time() - start_time
        
        print(f"✅ Query processing: {query_time:.2f}s")
        print(f"📊 Total time: {ingestion_time + query_time:.2f}s")
        
        # Performance Summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Document Processing: {doc_time:.2f}s ({doc_time/(ingestion_time + query_time)*100:.1f}%)")
        print(f"Vector Database: {vector_time:.2f}s ({vector_time/(ingestion_time + query_time)*100:.1f}%)")
        print(f"Query Parsing: {parser_time:.2f}s ({parser_time/(ingestion_time + query_time)*100:.1f}%)")
        print(f"LLM Reasoning: {reasoning_time:.2f}s ({reasoning_time/(ingestion_time + query_time)*100:.1f}%)")
        print(f"TOTAL TIME: {ingestion_time + query_time:.2f}s")
        
        # Optimization Recommendations
        print("\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 50)
        
        if doc_time > 10:
            print("🔧 Document processing is slow - consider:")
            print("   - Reduce chunk size")
            print("   - Use parallel processing")
            print("   - Optimize OCR settings")
        
        if vector_time > 20:
            print("🔧 Vector database is slow - consider:")
            print("   - Use GPU for embeddings")
            print("   - Batch processing")
            print("   - Reduce embedding dimensions")
        
        if reasoning_time > 30:
            print("🔧 LLM reasoning is slow - consider:")
            print("   - Use smaller model")
            print("   - Enable GPU acceleration")
            print("   - Reduce max tokens")
            print("   - Use caching")
        
        if ingestion_time + query_time > 30:
            print("🔧 Overall system is slow - consider:")
            print("   - Enable GPU for all components")
            print("   - Use model quantization")
            print("   - Implement caching")
            print("   - Parallel processing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_performance() 