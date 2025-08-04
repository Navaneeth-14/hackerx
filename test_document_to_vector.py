"""
Document to Vector Database Integration Test
Demonstrates the complete workflow: document_processer.py -> vector_database.py
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import tempfile
import json
from datetime import datetime

def select_file():
    """Open file dialog to select any supported document file"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select a document to process and store in vector database",
        filetypes=[
            ("All supported files", "*.pdf;*.txt;*.docx;*.html;*.htm;*.eml;*.msg;*.csv;*.json"),
            ("PDF files", "*.pdf"),
            ("Text files", "*.txt"),
            ("Word documents", "*.docx"),
            ("HTML files", "*.html;*.htm"),
            ("Email files", "*.eml;*.msg"),
            ("CSV files", "*.csv"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def process_and_store_document(file_path, use_ocr=False):
    """Process a document and store it in the vector database"""
    try:
        print(f"🔄 Step 1: Processing document with document_processer.py")
        print(f"📄 File: {file_path}")
        print(f"📏 File size: {os.path.getsize(file_path) / 1024:.1f} KB")
        
        # Import and use document processor
        from document_processer import AdvancedDocumentProcessor
        
        # Initialize document processor
        doc_processor = AdvancedDocumentProcessor()
        
        # Process the document
        chunks = doc_processor.process_document(file_path, use_ocr=use_ocr)
        
        if not chunks:
            print("❌ No chunks extracted from document")
            return False, "No chunks extracted"
        
        print(f"✅ Successfully processed {len(chunks)} chunks")
        
        # Display chunk information
        print(f"\n📋 Chunk Analysis:")
        text_chunks = [c for c in chunks if c.section_type == 'main_text']
        table_chunks = [c for c in chunks if c.section_type == 'table']
        metadata_chunks = [c for c in chunks if c.section_type == 'metadata']
        
        print(f"   📝 Text chunks: {len(text_chunks)}")
        print(f"   📊 Table chunks: {len(table_chunks)}")
        print(f"   🏷️  Metadata chunks: {len(metadata_chunks)}")
        
        # Show sample chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n   Chunk {i+1}:")
            print(f"     ID: {chunk.chunk_id}")
            print(f"     Type: {chunk.section_type}")
            print(f"     Content: {chunk.content[:100]}...")
        
        print(f"\n🔄 Step 2: Storing in vector database")
        
        # Import and use vector database
        from vector_database import VectorDatabase
        
        # Initialize vector database
        vector_db = VectorDatabase(
            embedding_model="all-MiniLM-L6-v2",
            collection_name="processed_documents",
            persist_directory="./vector_db",
            use_gpu=True
        )
        
        # Add documents to vector database
        success = vector_db.add_documents(chunks)
        
        if not success:
            print("❌ Failed to store documents in vector database")
            return False, "Vector database storage failed"
        
        print(f"✅ Successfully stored {len(chunks)} chunks in vector database")
        
        # Get database statistics
        stats = vector_db.get_document_statistics()
        print(f"\n📊 Vector Database Statistics:")
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   Unique sources: {stats.get('unique_sources', 0)}")
        print(f"   File types: {stats.get('file_types', [])}")
        
        return True, chunks
        
    except Exception as e:
        print(f"❌ Error in process_and_store_document: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def test_search_functionality(vector_db, original_file):
    """Test search functionality with the stored document"""
    print(f"\n🔍 Step 3: Testing search functionality")
    
    # Get filename for search terms
    filename = Path(original_file).stem
    
    # Create some test queries
    test_queries = [
        filename,  # Search by filename
        "document",  # Generic search
        "text content",  # Content search
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        
        # Semantic search
        semantic_results = vector_db.search_similar(query, n_results=3)
        print(f"   📝 Semantic search results: {len(semantic_results)}")
        
        for i, result in enumerate(semantic_results[:2]):
            print(f"     Result {i+1}: Score {result.similarity_score:.3f}")
            print(f"       Source: {result.source_file}")
            print(f"       Content: {result.content[:80]}...")
        
        # Hybrid search
        hybrid_results = vector_db.hybrid_search(query, n_results=3)
        print(f"   🔄 Hybrid search results: {len(hybrid_results)}")
        
        for i, result in enumerate(hybrid_results[:2]):
            print(f"     Result {i+1}: Score {result.similarity_score:.3f}")
            print(f"       Source: {result.source_file}")
            print(f"       Content: {result.content[:80]}...")

def create_sample_documents():
    """Create sample documents for testing"""
    test_dir = tempfile.mkdtemp()
    print(f"📁 Created test directory: {test_dir}")
    
    # Create sample TXT file
    txt_content = """
    Sample Document for Testing
    
    This is a sample text document that will be processed and stored in the vector database.
    It contains multiple paragraphs with various topics including:
    
    1. Technology and AI
    2. Business processes
    3. Data analysis
    4. Machine learning applications
    
    The document processor should extract this content and create chunks.
    The vector database should then store these chunks with embeddings.
    """
    
    txt_path = os.path.join(test_dir, "sample_document.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    # Create sample JSON file
    json_data = {
        "title": "Sample JSON Document",
        "author": "Test User",
        "content": "This is a sample JSON document for testing the document processor and vector database integration.",
        "topics": ["document processing", "vector database", "AI", "machine learning"],
        "metadata": {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "tags": ["test", "sample", "integration"]
        }
    }
    
    json_path = os.path.join(test_dir, "sample_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    return test_dir, {
        'txt': txt_path,
        'json': json_path
    }

def main():
    """Main function to test document to vector database workflow"""
    print("🚀 Document to Vector Database Integration Test")
    print("="*60)
    print("This test demonstrates the complete workflow:")
    print("1. Process document with document_processer.py")
    print("2. Store processed chunks in vector_database.py")
    print("3. Test search functionality")
    print()
    
    # Check if required modules are available
    try:
        from document_processer import AdvancedDocumentProcessor
        print("✅ Document processor available")
    except ImportError as e:
        print(f"❌ Document processor not available: {e}")
        return
    
    try:
        from vector_database import VectorDatabase
        print("✅ Vector database available")
    except ImportError as e:
        print(f"❌ Vector database not available: {e}")
        return
    
    print("\nChoose an option:")
    print("1. Select a file to process")
    print("2. Use sample documents")
    print("3. Process from command line")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Select file
        print("\n📁 Please select a document to process...")
        file_path = select_file()
        
        if not file_path:
            print("❌ No file selected")
            return
        
        # Ask about OCR
        use_ocr = input("Use OCR for PDFs? (y/n): ").lower().strip() in ['y', 'yes']
        
        # Process and store
        success, result = process_and_store_document(file_path, use_ocr)
        
        if success:
            # Test search functionality
            vector_db = VectorDatabase()
            test_search_functionality(vector_db, file_path)
            
            print(f"\n🎉 Complete workflow successful!")
            print(f"📄 Processed: {file_path}")
            print(f"📊 Stored in vector database")
            print(f"🔍 Search functionality tested")
        
    elif choice == "2":
        # Use sample documents
        test_dir, sample_files = create_sample_documents()
        
        print(f"\n🧪 Testing with sample documents...")
        
        for file_type, file_path in sample_files.items():
            print(f"\n📄 Processing {file_type.upper()} file...")
            success, result = process_and_store_document(file_path)
            
            if success:
                print(f"✅ {file_type.upper()} file processed successfully")
            else:
                print(f"❌ {file_type.upper()} file failed: {result}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test directory")
        
    elif choice == "3":
        # Command line processing
        if len(sys.argv) < 2:
            print("❌ Usage: python test_document_to_vector.py <file_path> [--ocr]")
            return
        
        file_path = sys.argv[1]
        use_ocr = "--ocr" in sys.argv
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"🔄 Processing file from command line: {file_path}")
        success, result = process_and_store_document(file_path, use_ocr)
        
        if success:
            print(f"🎉 Successfully processed and stored: {file_path}")
        else:
            print(f"❌ Failed: {result}")
    
    else:
        print("❌ Invalid choice")

def batch_process_directory():
    """Process all supported files in a directory"""
    print("🔄 Batch Processing Directory")
    print("="*40)
    
    # Select directory
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select directory containing documents")
    root.destroy()
    
    if not directory:
        print("❌ No directory selected")
        return
    
    # Find supported files
    supported_extensions = {'.pdf', '.txt', '.docx', '.html', '.htm', '.eml', '.msg', '.csv', '.json'}
    files_to_process = []
    
    for ext in supported_extensions:
        files_to_process.extend(Path(directory).glob(f"*{ext}"))
    
    if not files_to_process:
        print("❌ No supported files found in directory")
        return
    
    print(f"📁 Found {len(files_to_process)} files to process")
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Process each file
    results = {}
    for file_path in files_to_process:
        print(f"\n🔄 Processing: {file_path.name}")
        
        success, result = process_and_store_document(str(file_path))
        
        if success:
            print(f"✅ Success: {len(result)} chunks stored")
            results[file_path.name] = len(result)
        else:
            print(f"❌ Failed: {result}")
            results[file_path.name] = "ERROR"
    
    # Summary
    print(f"\n📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*40}")
    successful = sum(1 for result in results.values() if isinstance(result, int))
    total = len(results)
    
    for filename, result in results.items():
        status = f"{result} chunks" if isinstance(result, int) else result
        print(f"{filename}: {status}")
    
    print(f"\n✅ Successfully processed: {successful}/{total} files")
    
    # Test search with all documents
    print(f"\n🔍 Testing search with all processed documents...")
    test_search_functionality(vector_db, "batch_processed")

if __name__ == "__main__":
    print("Choose workflow:")
    print("1. Single file processing")
    print("2. Batch directory processing")
    
    workflow_choice = input("Enter choice (1 or 2): ").strip()
    
    if workflow_choice == "1":
        main()
    elif workflow_choice == "2":
        batch_process_directory()
    else:
        print("❌ Invalid choice") 