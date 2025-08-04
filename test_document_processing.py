#!/usr/bin/env python3
"""
Test script to isolate document processing issues
"""

import os
import sys
import time
from pathlib import Path

def test_document_processor():
    """Test the document processor directly"""
    print("🔍 TESTING DOCUMENT PROCESSOR")
    print("=" * 40)
    
    try:
        from document_processer import AdvancedDocumentProcessor
        
        # Initialize processor
        processor = AdvancedDocumentProcessor()
        print("✅ Document processor initialized")
        
        # Test with doc2.pdf
        file_path = "policy.pdf"
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📄 Processing file: {file_path}")
        
        # Test without OCR
        print("\n--- Testing without OCR ---")
        start_time = time.time()
        try:
            chunks = processor.process_document(file_path, use_ocr=False)
            processing_time = time.time() - start_time
            print(f"✅ Success! Processed {len(chunks)} chunks in {processing_time:.2f}s")
            
            # Show first chunk
            if chunks:
                print(f"📋 First chunk preview:")
                print(f"   ID: {chunks[0].chunk_id}")
                print(f"   Content: {chunks[0].content[:100]}...")
                print(f"   Source: {chunks[0].source_file}")
        except Exception as e:
            print(f"❌ Failed without OCR: {e}")
        
        # Test with OCR
        print("\n--- Testing with OCR ---")
        start_time = time.time()
        try:
            chunks = processor.process_document(file_path, use_ocr=True)
            processing_time = time.time() - start_time
            print(f"✅ Success! Processed {len(chunks)} chunks in {processing_time:.2f}s")
            
            # Show first chunk
            if chunks:
                print(f"📋 First chunk preview:")
                print(f"   ID: {chunks[0].chunk_id}")
                print(f"   Content: {chunks[0].content[:100]}...")
                print(f"   Source: {chunks[0].source_file}")
        except Exception as e:
            print(f"❌ Failed with OCR: {e}")
            
    except Exception as e:
        print(f"❌ Error initializing document processor: {e}")

def test_vector_database():
    """Test the vector database directly"""
    print("\n🔍 TESTING VECTOR DATABASE")
    print("=" * 40)
    
    try:
        from vector_database import VectorDatabase
        
        # Initialize vector database
        vector_db = VectorDatabase()
        print("✅ Vector database initialized")
        
        # Test adding a simple document
        test_content = "This is a test document for vector database testing."
        test_metadata = {
            'source_file': 'test.txt',
            'file_type': 'text',
            'section_type': 'test'
        }
        
        print("📝 Adding test document...")
        success = vector_db.add_document(test_content, test_metadata)
        
        if success:
            print("✅ Successfully added test document")
            
            # Test search
            print("🔍 Testing search...")
            results = vector_db.search_documents("test document", n_results=3)
            print(f"✅ Search returned {len(results)} results")
        else:
            print("❌ Failed to add test document")
            
    except Exception as e:
        print(f"❌ Error with vector database: {e}")

def test_rag_system():
    """Test the RAG system directly"""
    print("\n🔍 TESTING RAG SYSTEM")
    print("=" * 40)
    
    try:
        from rag_system import AdvancedRAGSystem
        
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag_system = AdvancedRAGSystem(use_gpu=False)  # Use CPU for testing
        print("✅ RAG system initialized")
        
        # Test document ingestion
        file_path = "doc2.pdf"
        if os.path.exists(file_path):
            print(f"📄 Testing document ingestion: {file_path}")
            try:
                chunks = rag_system.ingest_document(file_path, use_ocr=False)
                print(f"✅ Successfully ingested {len(chunks)} chunks")
            except Exception as e:
                print(f"❌ Document ingestion failed: {e}")
        else:
            print(f"❌ File not found: {file_path}")
            
    except Exception as e:
        print(f"❌ Error with RAG system: {e}")

def main():
    """Run all tests"""
    print("🧪 DOCUMENT PROCESSING DIAGNOSTICS")
    print("=" * 50)
    
    # Test 1: Document processor
    test_document_processor()
    
    # Test 2: Vector database
    test_vector_database()
    
    # Test 3: RAG system
    test_rag_system()
    
    print("\n✅ Diagnostics completed!")

if __name__ == "__main__":
    main() 