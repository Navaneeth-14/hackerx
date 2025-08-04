"""
Comprehensive RAG System Demo
Demonstrates all features of the RAG system including document ingestion, query processing, and audit trail
"""

import os
import json
import time
from pathlib import Path
from rag_system_gpu import RAGSystem

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

def demo_document_ingestion():
    """Interactive document ingestion with file upload from anywhere"""
    print_section("DOCUMENT INGESTION")
    
    try:
        rag_system = RAGSystem(use_gpu=True)  # Use GPU for better performance
        print("✅ GPU-optimized RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("💡 This might be due to missing model file or dependencies")
        return None
    
    print_subsection("PDF Document Upload")
    print("💡 Please provide the path to your PDF document from anywhere on your system.")
    print("💡 Supported formats: PDF files")
    print("💡 Type 'sample' to use the default sample.pdf (if available)")
    print("💡 Type 'browse' to open file browser (if available)")
    print("💡 Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get user input for file path
            file_path = input("📁 Enter PDF file path (or 'browse'/'sample'): ").strip()
            
            # Check for exit commands
            if file_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                return None
            
            # Check for browse command
            if file_path.lower() == 'browse':
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    
                    # Create a hidden root window
                    root = tk.Tk()
                    root.withdraw()  # Hide the main window
                    
                    # Open file dialog
                    file_path = filedialog.askopenfilename(
                        title="Select PDF File",
                        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                    )
                    
                    root.destroy()  # Close the hidden window
                    
                    if not file_path:
                        print("❌ No file selected.")
                        continue
                    
                    print(f"✅ Selected file: {file_path}")
                    
                except ImportError:
                    print("❌ File browser not available. Please enter the file path manually.")
                    continue
                except Exception as e:
                    print(f"❌ Error opening file browser: {e}")
                    print("💡 Please enter the file path manually.")
                    continue
            
            # Check for sample command
            elif file_path.lower() == 'sample':
                if os.path.exists("sample.pdf"):
                    file_path = "sample.pdf"
                    print("📄 Using sample.pdf...")
                else:
                    print("❌ sample.pdf not found. Please provide a different file path.")
                    continue
            
            # Skip empty input
            elif not file_path:
                print("⚠️  Please enter a file path.")
                continue
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                print("💡 Please check the file path and try again.")
                continue
            
            # Check if it's a PDF file
            if not file_path.lower().endswith('.pdf'):
                print("❌ Only PDF files are supported.")
                print("💡 Please provide a PDF file.")
                continue
            
            print_subsection(f"Processing PDF Document")
            print(f"📄 File: {os.path.basename(file_path)}")
            print(f"📂 Path: {file_path}")
            
            # Ask about OCR
            use_ocr_input = input("🔍 Use OCR for scanned documents? (y/n, default: n): ").strip().lower()
            use_ocr = use_ocr_input in ['y', 'yes']
            
            if use_ocr:
                print("🔍 OCR enabled - processing scanned document...")
            else:
                print("📄 Processing as text-based PDF...")
            
            # Ingest the document
            print("⏳ Processing document...")
            start_time = time.time()
            chunks = rag_system.ingest_document(file_path, use_ocr=use_ocr)
            processing_time = time.time() - start_time
            
            print(f"✅ Successfully processed {len(chunks)} chunks in {processing_time:.2f} seconds")
            print(f"📊 Document chunks created:")
            
            for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                print(f"   Chunk {i+1}: {chunk.chunk_id}")
                print(f"   Content preview: {chunk.content[:100]}...")
                print()
            
            if len(chunks) > 5:
                print(f"   ... and {len(chunks) - 5} more chunks")
            
            print("🎉 Document processing completed successfully!")
            return rag_system
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            return None
        except Exception as e:
            print(f"❌ Error during document ingestion: {e}")
            print("💡 Please check if the file is a valid PDF and try again.")
            print("💡 For scanned documents, try enabling OCR.")
            continue
            
            print_subsection(f"Processing PDF Document")
            print(f"📄 File: {selected_file.name}")
            
            # Ask about OCR
            use_ocr_input = input("🔍 Use OCR for scanned documents? (y/n, default: n): ").strip().lower()
            use_ocr = use_ocr_input in ['y', 'yes']
            
            if use_ocr:
                print("🔍 OCR enabled - processing scanned document...")
            else:
                print("📄 Processing as text-based PDF...")
            
            # Ingest the document
            print("⏳ Processing document...")
            start_time = time.time()
            chunks = rag_system.ingest_document(str(selected_file), use_ocr=use_ocr)
            processing_time = time.time() - start_time
            
            print(f"✅ Successfully processed {len(chunks)} chunks in {processing_time:.2f} seconds")
            print(f"📊 Document chunks created:")
            
            for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                print(f"   Chunk {i+1}: {chunk.chunk_id}")
                print(f"   Content preview: {chunk.content[:100]}...")
                print()
            
            if len(chunks) > 5:
                print(f"   ... and {len(chunks) - 5} more chunks")
            
            print("🎉 Document processing completed successfully!")
            return rag_system
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            return None
        except Exception as e:
            print(f"❌ Error during document ingestion: {e}")
            print("💡 Please check if the file is a valid PDF and try again.")
            print("💡 For scanned documents, try enabling OCR.")
            continue

def demo_query_processing(rag_system):
    """Interactive query processing with user input"""
    print_section("INTERACTIVE QUERY PROCESSING")
    
    print("💡 Enter your insurance policy questions below.")
    print("💡 Type 'quit' or 'exit' to stop asking questions.")
    print("💡 Type 'help' for example questions.")
    print()
    
    results = []
    query_count = 0
    
    while True:
        try:
            # Get user input
            user_query = input("🤔 Enter your question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Check for help command
            if user_query.lower() == 'help':
                print("\n📋 Example questions you can ask:")
                print("   • Is heart surgery covered under this policy?")
                print("   • What is the waiting period for pre-existing diseases?")
                print("   • Can I claim for dental treatment?")
                print("   • What is the maximum coverage amount?")
                print("   • Are there any exclusions for chronic diseases?")
                print("   • What documents are required for claim submission?")
                print("   • Is cancer treatment covered?")
                print("   • What is the claim process?")
                print()
                continue
            
            # Skip empty queries
            if not user_query:
                print("⚠️  Please enter a question.")
                continue
            
            query_count += 1
            print_subsection(f"Processing Query #{query_count}")
            print(f"🤔 Query: {user_query}")
            
            # Process the query
            start_time = time.time()
            result = rag_system.process_query(user_query)
            processing_time = time.time() - start_time
            
            # Display results
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"📋 Decision: {result.decision.upper()}")
            print(f"🎯 Confidence: {result.confidence_score:.2%}")
            
            if result.amount:
                print(f"💰 Amount: ₹{result.amount:,.2f}")
            
            print(f"📝 Justification: {result.justification}")
            
            if result.relevant_clauses:
                print(f"📄 Relevant Clauses: {', '.join(result.relevant_clauses)}")
            
            results.append({
                "query": user_query,
                "result": result,
                "processing_time": processing_time
            })
            
            print()
            
            # Ask if user wants to continue
            if query_count % 3 == 0:  # Ask every 3 queries
                continue_choice = input("❓ Continue asking questions? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '']:
                    print("👋 Thanks for using the RAG system!")
                    break
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("💡 Try asking a different question or type 'help' for examples.")
            print()

    return results

def demo_audit_trail(rag_system):
    """Demo audit trail functionality"""
    print_section("AUDIT TRAIL DEMO")
    
    try:
        # Get audit trail
        audit_log = rag_system.get_audit_trail()
        
        print_subsection("Audit Trail Overview")
        print(f"📊 Total queries processed: {len(audit_log)}")
        
        if audit_log:
            print("\n📋 Recent audit entries:")
            for i, entry in enumerate(audit_log[-3:], 1):  # Show last 3 entries
                print(f"   Entry {i}:")
                print(f"     Timestamp: {entry.get('timestamp', 'N/A')}")
                print(f"     Query: {entry.get('query', 'N/A')}")
                print(f"     Relevant chunks: {entry.get('relevant_chunks_count', 0)}")
                
                if 'error' in entry:
                    print(f"     Error: {entry['error']}")
                print()
        
        # Save audit trail
        print_subsection("Saving Audit Trail")
        audit_file = "demo_audit_trail.json"
        rag_system.save_audit_trail(audit_file)
        print(f"✅ Audit trail saved to: {audit_file}")
        
        # Show audit file size
        if os.path.exists(audit_file):
            file_size = os.path.getsize(audit_file)
            print(f"📁 File size: {file_size:,} bytes")
        
    except Exception as e:
        print(f"❌ Error with audit trail: {e}")

def demo_system_analysis(rag_system, query_results):
    """Demo system performance and analysis"""
    print_section("SYSTEM ANALYSIS DEMO")
    
    if not query_results:
        print("❌ No query results to analyze")
        return
    
    print_subsection("Performance Statistics")
    
    # Calculate statistics
    total_queries = len(query_results)
    avg_processing_time = sum(r['processing_time'] for r in query_results) / total_queries
    avg_confidence = sum(r['result'].confidence_score for r in query_results) / total_queries
    
    decisions = [r['result'].decision for r in query_results]
    decision_counts = {}
    for decision in decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    print(f"📊 Total queries processed: {total_queries}")
    print(f"⏱️  Average processing time: {avg_processing_time:.2f} seconds")
    print(f"🎯 Average confidence score: {avg_confidence:.2%}")
    
    print("\n📋 Decision Distribution:")
    for decision, count in decision_counts.items():
        percentage = (count / total_queries) * 100
        print(f"   {decision.upper()}: {count} ({percentage:.1f}%)")
    
    print_subsection("Query Analysis")
    
    # Find best and worst performing queries
    best_query = max(query_results, key=lambda x: x['result'].confidence_score)
    worst_query = min(query_results, key=lambda x: x['result'].confidence_score)
    
    print(f"🏆 Best performing query:")
    print(f"   Query: {best_query['query']}")
    print(f"   Confidence: {best_query['result'].confidence_score:.2%}")
    
    print(f"\n⚠️  Worst performing query:")
    print(f"   Query: {worst_query['query']}")
    print(f"   Confidence: {worst_query['result'].confidence_score:.2%}")

def demo_advanced_features():
    """Demo advanced features like hybrid search and contextual compression"""
    print_section("ADVANCED FEATURES DEMO")
    
    print_subsection("Vector Database Features")
    print("🔍 Semantic search with contextual compression")
    print("📚 Document chunking with metadata preservation")
    print("🎯 Similarity scoring and ranking")
    
    print_subsection("LLM Integration")
    print("🧠 Query parsing with entity extraction")
    print("💭 Reasoning with policy clause mapping")
    print("📋 Structured JSON response generation")
    
    print_subsection("Audit and Compliance")
    print("📝 Complete audit trail with timestamps")
    print("🔗 Decision justification with clause references")
    print("💾 Exportable audit logs for compliance")

def main():
    """Main demo function"""
    print("🚀 RAG Insurance Policy Analyzer - Interactive GPU Demo")
    print("This demo allows you to upload PDF documents from anywhere on your system")
    print("and ask questions interactively. Powered by GPU-accelerated RAG system")
    
    # Step 1: Document Ingestion
    rag_system = demo_document_ingestion()
    if not rag_system:
        print("❌ Demo cannot continue without successful document ingestion")
        return
    
    # Step 2: Query Processing
    query_results = demo_query_processing(rag_system)
    
    # Step 3: Audit Trail
    demo_audit_trail(rag_system)
    
    # Step 4: System Analysis
    demo_system_analysis(rag_system, query_results)
    
    # Step 5: Advanced Features
    demo_advanced_features()
    
    print_section("INTERACTIVE DEMO COMPLETED")
    print("✅ You've successfully used the interactive RAG system!")
    print("📁 Check the following files for outputs:")
    print("   - demo_audit_trail.json (audit trail)")
    print("   - vector_db/ (vector database)")
    print("   - uploads/ (uploaded documents)")
    
    print("\n🎯 Next Steps:")
    print("   1. Start the web interface: python api_server.py")
    print("   2. Open http://localhost:8000 in your browser")
    print("   3. Upload documents and process queries interactively")
    print("   4. Or run this demo again: python demo.py")
    print("   5. Try different PDF files from anywhere on your system")

if __name__ == "__main__":
    main() 