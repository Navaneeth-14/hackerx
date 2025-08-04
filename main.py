"""
Main Entry Point for Advanced RAG System
Provides user-friendly interface for document upload and query processing
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our RAG system
from rag_system import AdvancedRAGSystem, QueryResult

# Import document processor for direct access
from document_processer import AdvancedDocumentProcessor

# Import pytesseract for direct OCR
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def print_banner():
    """Print system banner"""
    print("=" * 80)
    print("🚀 ADVANCED RAG SYSTEM - INSURANCE POLICY ANALYZER")
    print("=" * 80)
    print("📋 Features:")
    print("   • Multi-format document processing (PDF, TXT, Email, etc.)")
    print("   • Advanced OCR for scanned documents")
    print("   • Natural language query processing")
    print("   • Clause referencing and explanation")
    print("   • Comprehensive audit trail")
    print("   • GPU-accelerated processing")
    print("=" * 80)

def print_menu():
    """Print main menu options"""
    print("\n📋 MAIN MENU:")
    print("1. 📄 Upload Document")
    print("2. 🤔 Process Query")
    print("3. 📊 System Statistics")
    print("4. 🔍 View Audit Trail")
    print("5. 🧪 System Validation")
    print("6. 💾 Export System Data")
    print("7. 🗑️  Clear System")
    print("8. 🧪 Test OCR")
    print("9. ❓ Help")
    print("0. 🚪 Exit")
    print("-" * 40)

def get_user_choice() -> str:
    """Get user choice from menu"""
    try:
        choice = input("\n🎯 Enter your choice (1-9): ").strip()
        return choice
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

def simple_ocr_pdf(file_path: str) -> str:
    """Simple OCR function using pytesseract directly"""
    if not OCR_AVAILABLE:
        print("❌ pytesseract not available. Please install it with: pip install pytesseract")
        return ""
    
    try:
        import fitz  # PyMuPDF
        
        # Open PDF
        doc = fitz.open(file_path)
        all_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get page as image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            all_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
        
        doc.close()
        return all_text
        
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

def upload_document(rag_system: AdvancedRAGSystem):
    """Handle document upload"""
    print("\n📄 DOCUMENT UPLOAD")
    print("-" * 40)
    
    try:
        # Get file path
        file_path = input("📁 Enter file path (or 'browse' for file dialog): ").strip()
        
        if file_path.lower() == 'browse':
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                
                file_path = filedialog.askopenfilename(
                    title="Select Document",
                    filetypes=[
                        ("All supported", "*.pdf;*.txt;*.docx;*.html;*.eml;*.csv;*.json"),
                        ("PDF files", "*.pdf"),
                        ("Text files", "*.txt"),
                        ("Word documents", "*.docx"),
                        ("HTML files", "*.html"),
                        ("Email files", "*.eml"),
                        ("CSV files", "*.csv"),
                        ("JSON files", "*.json"),
                        ("All files", "*.*")
                    ]
                )
                
                root.destroy()
                
                if not file_path:
                    print("❌ No file selected.")
                    return
                    
            except ImportError:
                print("❌ File browser not available. Please enter the file path manually.")
                return
            except Exception as e:
                print(f"❌ Error opening file browser: {e}")
                return
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        # Check file type
        supported_extensions = {'.pdf', '.txt', '.docx', '.html', '.htm', '.eml', '.msg', '.csv', '.json'}
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in supported_extensions:
            print(f"❌ Unsupported file type: {file_extension}")
            print(f"💡 Supported types: {', '.join(supported_extensions)}")
            return
        
        # Ask about OCR for PDF files
        use_ocr = False
        if file_extension == '.pdf':
            if OCR_AVAILABLE:
                ocr_choice = input("🔍 Use OCR for scanned documents? (y/n, default: n): ").strip().lower()
                use_ocr = ocr_choice in ['y', 'yes']
            else:
                print("⚠️ OCR not available. Processing without OCR.")
        
        print(f"\n⏳ Processing document: {os.path.basename(file_path)}")
        print("💡 This may take a few moments...")
        
        # Process document
        start_time = time.time()
        try:
            chunks = rag_system.ingest_document(file_path, use_ocr=use_ocr)
            processing_time = time.time() - start_time
        except Exception as e:
            print(f"❌ Document ingestion failed: {e}")
            print("💡 Trying alternative processing method...")
            
            # Fallback: Use document processor directly
            try:
                doc_processor = AdvancedDocumentProcessor()
                chunks = doc_processor.process_document(file_path, use_ocr=use_ocr)
                
                # Add chunks to vector database manually
                if chunks:
                    success = rag_system.vector_database.add_documents(chunks)
                    if success:
                        print("✅ Document processed using fallback method")
                        processing_time = time.time() - start_time
                    else:
                        print("❌ Failed to add documents to vector database")
                        chunks = []
                else:
                    print("❌ No chunks generated from document")
                    chunks = []
                    processing_time = time.time() - start_time
            except Exception as fallback_error:
                print(f"❌ Fallback processing also failed: {fallback_error}")
                chunks = []
                processing_time = time.time() - start_time
        
        if chunks:
            print(f"✅ Successfully processed {len(chunks)} chunks in {processing_time:.2f} seconds")
            print(f"📊 Document chunks created:")
            
            for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                print(f"   Chunk {i+1}: {chunk.chunk_id}")
                print(f"   Type: {chunk.section_type}")
                print(f"   Content preview: {chunk.content[:100]}...")
                print()
            
            if len(chunks) > 5:
                print(f"   ... and {len(chunks) - 5} more chunks")
        else:
            print("❌ Failed to process document")
            if file_extension == '.pdf' and not use_ocr:
                print("💡 Try using OCR for scanned PDFs: Select 'y' when asked about OCR")
            
    except KeyboardInterrupt:
        print("\n👋 Upload cancelled.")
    except Exception as e:
        print(f"❌ Error uploading document: {e}")

def process_query(rag_system: AdvancedRAGSystem):
    """Interactive query processing with continuous questioning"""
    print("\n🤔 INTERACTIVE QUERY PROCESSING")
    print("-" * 40)
    
    print("💡 Enter your questions about the uploaded documents.")
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
                print("   • Does the policy cover newborn care after hospital discharge?")
                print()
                continue
            
            # Skip empty queries
            if not user_query:
                print("⚠️  Please enter a question.")
                continue
            
            query_count += 1
            print(f"\n🔍 Processing Query #{query_count}")
            print(f"🤔 Query: {user_query}")
            
            # Process the query
            start_time = time.time()
            try:
                result = rag_system.process_query(user_query)
                processing_time = time.time() - start_time
            except Exception as e:
                print(f"❌ Query processing failed: {e}")
                print("💡 Creating fallback result...")
                
                # Create fallback result
                from query_parser import AdvancedQueryParser
                from llm_reasoning import AdvancedLLMReasoning
                
                try:
                    # Parse query
                    query_parser = AdvancedQueryParser()
                    parsed_query = query_parser.parse_query(user_query)
                    
                    # Create fallback reasoning
                    reasoning_engine = AdvancedLLMReasoning()
                    fallback_context = [{
                        'content': f"Based on the query: {user_query}",
                        'source_file': 'fallback',
                        'similarity_score': 0.5
                    }]
                    
                    reasoning_result = reasoning_engine.analyze_query(
                        user_query, fallback_context, parsed_query.query_type
                    )
                    
                    # Create fallback result
                    from rag_system import QueryResult
                    result = QueryResult(
                        query=user_query,
                        parsed_query=parsed_query,
                        search_results=[],
                        reasoning_result=reasoning_result,
                        processing_time=time.time() - start_time,
                        timestamp=time.time(),
                        audit_trail={'status': 'fallback', 'error': str(e)}
                    )
                    processing_time = time.time() - start_time
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback processing also failed: {fallback_error}")
                    continue
        
            # Display results
            print(f"\n📋 QUERY RESULTS")
            print("=" * 50)
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"🎯 Decision: {result.reasoning_result.decision.upper()}")
            print(f"📊 Confidence: {result.reasoning_result.confidence_score:.1%}")
            
            if result.reasoning_result.amount:
                print(f"💰 Amount: ${result.reasoning_result.amount:,.2f}")
            
            if result.reasoning_result.waiting_period:
                print(f"⏰ Waiting Period: {result.reasoning_result.waiting_period}")
            
            print(f"\n📝 Justification:")
            print(result.reasoning_result.justification)
            
            if result.reasoning_result.relevant_clauses:
                print(f"\n📄 Relevant Clauses:")
                for clause in result.reasoning_result.relevant_clauses:
                    print(f"   • {clause}")
            
            if result.reasoning_result.conditions:
                print(f"\n✅ Conditions:")
                for condition in result.reasoning_result.conditions:
                    print(f"   • {condition}")
            
            if result.reasoning_result.exclusions:
                print(f"\n❌ Exclusions:")
                for exclusion in result.reasoning_result.exclusions:
                    print(f"   • {exclusion}")
            
            if result.reasoning_result.required_documents:
                print(f"\n📋 Required Documents:")
                for doc in result.reasoning_result.required_documents:
                    print(f"   • {doc}")
            
            # Show search results summary
            if result.search_results:
                print(f"\n🔍 Search Results Summary:")
                print(f"   Found {len(result.search_results)} relevant documents")
                for i, search_result in enumerate(result.search_results[:3], 1):
                    print(f"   {i}. {search_result.source_file} (Score: {search_result.similarity_score:.2f})")
            
            # Ask if user wants to see detailed explanation
            show_details = input("\n❓ Show detailed explanation? (y/n): ").strip().lower()
            if show_details in ['y', 'yes']:
                try:
                    detailed_explanation = rag_system.reasoning_engine.explain_decision(result.reasoning_result)
                    print(f"\n📖 DETAILED EXPLANATION:")
                    print("=" * 50)
                    print(detailed_explanation)
                except Exception as e:
                    print(f"❌ Error generating detailed explanation: {e}")
                    print("💡 Detailed explanation not available")
            
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

def show_system_statistics(rag_system: AdvancedRAGSystem):
    """Show system statistics"""
    print("\n📊 SYSTEM STATISTICS")
    print("-" * 40)
    
    try:
        stats = rag_system.get_system_statistics()
        
        if stats:
            # Vector database stats
            db_stats = stats.get('vector_database', {})
            print(f"📚 Vector Database:")
            print(f"   Total chunks: {db_stats.get('total_chunks', 0)}")
            print(f"   Unique sources: {db_stats.get('unique_sources', 0)}")
            print(f"   File types: {', '.join(db_stats.get('file_types', []))}")
            
            # Audit trail stats
            audit_stats = stats.get('audit_trail', {})
            print(f"\n📋 Audit Trail:")
            print(f"   Total entries: {audit_stats.get('total_entries', 0)}")
            print(f"   Successful queries: {audit_stats.get('successful_queries', 0)}")
            print(f"   Failed queries: {audit_stats.get('failed_queries', 0)}")
            print(f"   Document ingestions: {audit_stats.get('document_ingestions', 0)}")
            
            # Component info
            components = stats.get('components', {})
            print(f"\n🔧 Components:")
            print(f"   Document Processor: {components.get('document_processor', 'Unknown')}")
            print(f"   Vector Database: {components.get('vector_database', 'Unknown')}")
            print(f"   Query Parser: {components.get('query_parser', 'Unknown')}")
            print(f"   Reasoning Engine: {components.get('reasoning_engine', 'Unknown')}")
            print(f"   GPU Enabled: {components.get('use_gpu', False)}")
        else:
            print("❌ Unable to retrieve system statistics")
            
    except Exception as e:
        print(f"❌ Error getting system statistics: {e}")

def show_audit_trail(rag_system: AdvancedRAGSystem):
    """Show audit trail"""
    print("\n🔍 AUDIT TRAIL")
    print("-" * 40)
    
    try:
        audit_log = rag_system.get_audit_trail()
        
        if audit_log:
            print(f"📋 Total entries: {len(audit_log)}")
            
            # Show recent entries
            recent_entries = audit_log[-10:]  # Last 10 entries
            print(f"\n📝 Recent Entries:")
            
            for i, entry in enumerate(reversed(recent_entries), 1):
                action = entry.get('action', 'Unknown')
                timestamp = entry.get('timestamp', 'Unknown')
                status = entry.get('status', 'Unknown')
                
                print(f"   {i}. {action} - {status} ({timestamp})")
                
                if action == 'query_processing':
                    query = entry.get('query', 'Unknown')
                    print(f"      Query: {query[:50]}...")
                    
                    reasoning = entry.get('reasoning_result', {})
                    if reasoning:
                        decision = reasoning.get('decision', 'Unknown')
                        confidence = reasoning.get('confidence_score', 0.0)
                        print(f"      Decision: {decision} (Confidence: {confidence:.1%})")
                
                elif action == 'document_ingestion':
                    file_path = entry.get('file_path', 'Unknown')
                    chunks = entry.get('chunks_processed', 0)
                    print(f"      File: {os.path.basename(file_path)} ({chunks} chunks)")
                
                print()
            
            # Ask if user wants to save audit trail
            save_choice = input("💾 Save audit trail to file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                filename = input("📁 Enter filename (default: audit_trail.json): ").strip()
                if not filename:
                    filename = "audit_trail.json"
                
                if rag_system.save_audit_trail(filename):
                    print(f"✅ Audit trail saved to: {filename}")
                else:
                    print("❌ Failed to save audit trail")
        else:
            print("📋 No audit trail entries found")
            
    except Exception as e:
        print(f"❌ Error showing audit trail: {e}")

def validate_system(rag_system: AdvancedRAGSystem):
    """Validate system components"""
    print("\n🧪 SYSTEM VALIDATION")
    print("-" * 40)
    
    try:
        validation = rag_system.validate_system()
        
        print("🔍 Checking system components...")
        
        components = [
            ('Document Processor', validation.get('document_processor', False)),
            ('Vector Database', validation.get('vector_database', False)),
            ('Query Parser', validation.get('query_parser', False)),
            ('Reasoning Engine', validation.get('reasoning_engine', False))
        ]
        
        all_valid = True
        for component_name, is_valid in components:
            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"   {component_name}: {status}")
            if not is_valid:
                all_valid = False
        
        print(f"\n🎯 Overall Status: {'✅ ALL COMPONENTS VALID' if all_valid else '❌ SOME COMPONENTS FAILED'}")
        
        if not all_valid:
            print("\n❌ Errors found:")
            for error in validation.get('errors', []):
                print(f"   • {error}")
        
    except Exception as e:
        print(f"❌ Error validating system: {e}")

def export_system_data(rag_system: AdvancedRAGSystem):
    """Export system data"""
    print("\n💾 EXPORT SYSTEM DATA")
    print("-" * 40)
    
    try:
        filename = input("📁 Enter filename (default: system_export.json): ").strip()
        if not filename:
            filename = "system_export.json"
        
        print(f"⏳ Exporting system data to: {filename}")
        
        if rag_system.export_system_data(filename):
            print(f"✅ System data exported successfully to: {filename}")
            
            # Show file size
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"📁 File size: {file_size:,} bytes")
        else:
            print("❌ Failed to export system data")
            
    except Exception as e:
        print(f"❌ Error exporting system data: {e}")

def clear_system(rag_system: AdvancedRAGSystem):
    """Clear system data"""
    print("\n🗑️  CLEAR SYSTEM")
    print("-" * 40)
    
    try:
        confirm = input("⚠️  This will clear ALL system data. Are you sure? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            print("⏳ Clearing system data...")
            
            if rag_system.clear_system():
                print("✅ System data cleared successfully")
            else:
                print("❌ Failed to clear system data")
        else:
            print("❌ Operation cancelled")
            
    except Exception as e:
        print(f"❌ Error clearing system: {e}")

def test_ocr():
    """Test OCR functionality"""
    print("\n🧪 OCR TEST")
    print("-" * 40)
    
    if not OCR_AVAILABLE:
        print("❌ pytesseract not available")
        print("💡 Install with: pip install pytesseract")
        return
    
    try:
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        
        # Create test image
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text
        text = "Test OCR Text"
        draw.text((10, 40), text, fill='black', font=font)
        
        # Test OCR
        ocr_text = pytesseract.image_to_string(img)
        print(f"✅ OCR test successful")
        print(f"   Original: '{text}'")
        print(f"   OCR result: '{ocr_text.strip()}'")
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        print("💡 Make sure Tesseract is installed and configured")

def show_help():
    """Show help information"""
    print("\n❓ HELP")
    print("-" * 40)
    print("📋 This RAG system can process various document types and answer questions about them.")
    print("\n📄 Supported Document Types:")
    print("   • PDF files (with OCR support for scanned documents)")
    print("   • Text files (.txt)")
    print("   • Word documents (.docx)")
    print("   • HTML files (.html)")
    print("   • Email files (.eml, .msg)")
    print("   • CSV files (.csv)")
    print("   • JSON files (.json)")
    
    print("\n🤔 Query Examples:")
    print("   • 'Is heart surgery covered under this policy?'")
    print("   • 'How do I file a claim?'")
    print("   • 'What is the waiting period for pre-existing conditions?'")
    print("   • 'What documents are required for claim submission?'")
    print("   • 'What is the maximum coverage amount?'")
    
    print("\n💡 Tips:")
    print("   • Upload documents first before asking questions")
    print("   • Use natural language - the system understands plain English")
    print("   • The system can handle vague or incomplete queries")
    print("   • All queries are logged in the audit trail")
    print("   • Use the system validation to check component status")
    print("   • Test OCR functionality if you have issues with scanned documents")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced RAG System')
    parser.add_argument('--query', help='Process a single query from file')
    parser.add_argument('--upload', help='Upload and process a document')
    parser.add_argument('--status', action='store_true', help='Check system status')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.query:
        # Process single query from file
        try:
            with open(args.query, 'r') as f:
                question = f.read().strip()
            
            # Initialize RAG system
            rag_system = AdvancedRAGSystem(use_gpu=False)
            
            # Process the query
            result = rag_system.process_query(question)
            
            # Output results in a structured format
            print(f"Question: {question}")
            print(f"Decision: {result.reasoning_result.decision}")
            print(f"Confidence: {result.reasoning_result.confidence_score:.1%}")
            print(f"Justification: {result.reasoning_result.justification}")
            
            if result.reasoning_result.amount:
                print(f"Amount: ${result.reasoning_result.amount:,.2f}")
            
            if result.reasoning_result.waiting_period:
                print(f"Waiting Period: {result.reasoning_result.waiting_period}")
            
            if result.reasoning_result.relevant_clauses:
                print(f"Relevant Clauses: {', '.join(result.reasoning_result.relevant_clauses)}")
            
            if result.reasoning_result.conditions:
                print(f"Conditions: {', '.join(result.reasoning_result.conditions)}")
            
            if result.reasoning_result.exclusions:
                print(f"Exclusions: {', '.join(result.reasoning_result.exclusions)}")
            
            if result.reasoning_result.required_documents:
                print(f"Required Documents: {', '.join(result.reasoning_result.required_documents)}")
            
            return
        except Exception as e:
            print(f"Error processing query: {e}")
            sys.exit(1)
    
    elif args.upload:
        # Upload and process document
        try:
            # Initialize RAG system
            rag_system = AdvancedRAGSystem(use_gpu=False)
            
            # Process the document
            chunks = rag_system.ingest_document(args.upload, use_ocr=False)
            
            print(f"Document processed successfully: {args.upload}")
            print(f"Chunks processed: {len(chunks)}")
            
            return
        except Exception as e:
            print(f"Error processing document: {e}")
            sys.exit(1)
    
    elif args.status:
        # Check system status
        try:
            rag_system = AdvancedRAGSystem(use_gpu=False)
            print("System Status: READY")
            return
        except Exception as e:
            print(f"System Status: ERROR - {e}")
            sys.exit(1)
    
    # Interactive mode (default)
    print_banner()
    
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        try:
            rag_system = AdvancedRAGSystem(use_gpu=False)  # Use CPU for better compatibility
            print("✅ RAG system initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            print("💡 Trying with minimal configuration...")
            
            try:
                # Try with minimal settings
                rag_system = AdvancedRAGSystem(
                    use_gpu=False,
                    model_path=None  # Don't require model file
                )
                print("✅ RAG system initialized with minimal configuration!")
            except Exception as e2:
                print(f"❌ RAG system initialization failed: {e2}")
                print("💡 Please check your system configuration")
                return
        
        # Main loop
        while True:
            print_menu()
            choice = get_user_choice()
            
            if choice == '1':
                upload_document(rag_system)
            elif choice == '2':
                process_query(rag_system)
            elif choice == '3':
                show_system_statistics(rag_system)
            elif choice == '4':
                show_audit_trail(rag_system)
            elif choice == '5':
                validate_system(rag_system)
            elif choice == '6':
                export_system_data(rag_system)
            elif choice == '7':
                clear_system(rag_system)
            elif choice == '8':
                test_ocr()
            elif choice == '9':
                show_help()
            elif choice == '0':
                print("👋 Thank you for using the Advanced RAG System!")
                break
            else:
                print("❌ Invalid choice. Please enter a number between 1-0.")
            
            # Pause before showing menu again
            input("\n⏸️  Press Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("💡 Please check your system configuration and try again.")

if __name__ == "__main__":
    main() 