"""
PDF Document Processor Test
Allows you to choose any PDF file and process it with the document processor
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import tempfile
import shutil

def select_pdf_file():
    """Open file dialog to select a PDF file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select a PDF file to process",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path

def process_pdf_with_ocr(pdf_path, use_ocr=False):
    """Process a PDF file with optional OCR"""
    try:
        from document_processer import AdvancedDocumentProcessor
        
        print(f"🔄 Processing PDF: {pdf_path}")
        print(f"📄 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
        
        # Initialize processor
        processor = AdvancedDocumentProcessor()
        
        # Process the document
        chunks = processor.process_document(pdf_path, use_ocr=use_ocr)
        
        return chunks, None
        
    except Exception as e:
        return None, str(e)

def display_results(chunks, pdf_path):
    """Display processing results"""
    print(f"\n{'='*60}")
    print("📊 PROCESSING RESULTS")
    print(f"{'='*60}")
    
    print(f"📄 PDF File: {pdf_path}")
    print(f"📊 Total Chunks: {len(chunks)}")
    
    # Analyze chunks
    text_chunks = [c for c in chunks if c.section_type == 'main_text']
    table_chunks = [c for c in chunks if c.section_type == 'table']
    metadata_chunks = [c for c in chunks if c.section_type == 'metadata']
    
    print(f"📝 Text Chunks: {len(text_chunks)}")
    print(f"📊 Table Chunks: {len(table_chunks)}")
    print(f"🏷️  Metadata Chunks: {len(metadata_chunks)}")
    
    # Show sample chunks
    print(f"\n📋 SAMPLE CHUNKS:")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Type: {chunk.section_type}")
        print(f"  Content Preview: {chunk.content[:150]}...")
        
        if chunk.table_data:
            print(f"  Table Data: {len(chunk.table_data.get('data', []))} rows")
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    
    # Save results to file
    save_results_to_file(chunks, pdf_path)

def save_results_to_file(chunks, pdf_path):
    """Save processing results to a text file"""
    try:
        # Create output filename
        pdf_name = Path(pdf_path).stem
        output_file = f"{pdf_name}_processed_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"PDF Processing Results\n")
            f.write(f"="*50 + "\n")
            f.write(f"Source PDF: {pdf_path}\n")
            f.write(f"Total Chunks: {len(chunks)}\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i+1}:\n")
                f.write(f"  ID: {chunk.chunk_id}\n")
                f.write(f"  Type: {chunk.section_type}\n")
                f.write(f"  Source: {chunk.source_file}\n")
                f.write(f"  File Type: {chunk.file_type}\n")
                f.write(f"  Content:\n{chunk.content}\n")
                f.write(f"  {'-'*40}\n\n")
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save results to file: {e}")

def analyze_pdf_content(chunks):
    """Analyze the content of processed chunks"""
    print(f"\n🔍 CONTENT ANALYSIS")
    print(f"{'='*40}")
    
    total_text_length = sum(len(chunk.content) for chunk in chunks)
    avg_chunk_size = total_text_length / len(chunks) if chunks else 0
    
    print(f"📏 Total Text Length: {total_text_length:,} characters")
    print(f"📊 Average Chunk Size: {avg_chunk_size:.0f} characters")
    
    # Find longest and shortest chunks
    if chunks:
        longest_chunk = max(chunks, key=lambda x: len(x.content))
        shortest_chunk = min(chunks, key=lambda x: len(x.content))
        
        print(f"📏 Longest Chunk: {len(longest_chunk.content)} characters")
        print(f"📏 Shortest Chunk: {len(shortest_chunk.content)} characters")
    
    # Count unique words
    all_text = " ".join(chunk.content for chunk in chunks)
    unique_words = len(set(all_text.lower().split()))
    total_words = len(all_text.split())
    
    print(f"📝 Total Words: {total_words:,}")
    print(f"📝 Unique Words: {unique_words:,}")

def main():
    """Main function to run the PDF processor test"""
    print("🚀 PDF Document Processor Test")
    print("="*50)
    print("This tool allows you to process any PDF file by specifying its path.")
    print("You can choose whether to use OCR for better text extraction.")
    print()
    
    # Check if document processor is available
    try:
        from document_processer import AdvancedDocumentProcessor
        print("✅ Document processor loaded successfully")
    except ImportError as e:
        print(f"❌ Error loading document processor: {e}")
        print("💡 Make sure document_processer.py is in the same directory")
        return
    
    # Get PDF file path
    print("\n📁 Enter the path to your PDF file:")
    print("   Examples:")
    print("   - C:\\Users\\YourName\\Documents\\document.pdf")
    print("   - /home/username/documents/document.pdf")
    print("   - ./local_file.pdf")
    print("   - Or press Enter to use file dialog")
    
    pdf_path = input("PDF file path: ").strip()
    
    # If no path provided, use file dialog
    if not pdf_path:
        print("\n📁 Opening file dialog...")
        pdf_path = select_pdf_file()
    
    if not pdf_path:
        print("❌ No file selected. Exiting.")
        return
    
    # Expand relative paths and resolve to absolute path
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        print("💡 Please check the file path and try again.")
        return
    
    # Check if it's actually a PDF file
    if not pdf_path.lower().endswith('.pdf'):
        print(f"⚠️  Warning: File doesn't have .pdf extension: {pdf_path}")
        proceed = input("Continue anyway? (y/n): ").lower().strip()
        if proceed not in ['y', 'yes']:
            print("❌ Exiting.")
            return
    
    print(f"✅ Found file: {pdf_path}")
    print(f"📄 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    # Ask about OCR
    print("\n🤔 Do you want to use OCR for better text extraction?")
    print("   OCR is useful for scanned PDFs or PDFs with images")
    print("   OCR takes longer but provides better results for image-based PDFs")
    
    use_ocr = input("Use OCR? (y/n): ").lower().strip() in ['y', 'yes']
    
    if use_ocr:
        print("🔍 Will use OCR for text extraction")
    else:
        print("📝 Will use standard text extraction")
    
    # Process the PDF
    print(f"\n🔄 Processing PDF...")
    chunks, error = process_pdf_with_ocr(pdf_path, use_ocr)
    
    if error:
        print(f"❌ Error processing PDF: {error}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure the PDF file is not corrupted")
        print("2. Try without OCR if the PDF has text")
        print("3. Check if all dependencies are installed")
        print("4. Verify the file path is correct")
        return
    
    if not chunks:
        print("❌ No chunks were extracted from the PDF")
        print("💡 This might be because:")
        print("   - The PDF is password protected")
        print("   - The PDF contains only images")
        print("   - The PDF is corrupted")
        return
    
    # Display results
    display_results(chunks, pdf_path)
    
    # Analyze content
    analyze_pdf_content(chunks)
    
    print(f"\n🎉 PDF processing completed successfully!")
    print(f"📄 Processed: {pdf_path}")
    print(f"📊 Extracted: {len(chunks)} chunks")

def batch_process_pdfs():
    """Process multiple PDF files in a directory"""
    print("🔄 Batch PDF Processing")
    print("="*40)
    
    # Select directory
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select directory containing PDF files")
    root.destroy()
    
    if not directory:
        print("❌ No directory selected")
        return
    
    # Find PDF files
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in the selected directory")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files in {directory}")
    
    # Process each PDF
    results = {}
    for pdf_file in pdf_files:
        print(f"\n🔄 Processing: {pdf_file.name}")
        chunks, error = process_pdf_with_ocr(str(pdf_file), use_ocr=False)
        
        if error:
            print(f"❌ Error: {error}")
            results[pdf_file.name] = "ERROR"
        else:
            print(f"✅ Processed: {len(chunks)} chunks")
            results[pdf_file.name] = len(chunks)
    
    # Summary
    print(f"\n📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*40}")
    successful = sum(1 for result in results.values() if isinstance(result, int))
    total = len(results)
    
    for filename, result in results.items():
        status = f"{result} chunks" if isinstance(result, int) else result
        print(f"{filename}: {status}")
    
    print(f"\n✅ Successfully processed: {successful}/{total} files")

def process_from_command_line():
    """Process PDF from command line arguments"""
    import sys
    
    if len(sys.argv) < 2:
        print("❌ Usage: python test_pdf_processor.py <pdf_file_path> [--ocr]")
        print("   Example: python test_pdf_processor.py C:\\path\\to\\document.pdf --ocr")
        return
    
    pdf_path = sys.argv[1]
    use_ocr = "--ocr" in sys.argv
    
    # Expand relative paths and resolve to absolute path
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"🚀 Processing PDF from command line: {pdf_path}")
    print(f"🔍 OCR enabled: {use_ocr}")
    
    # Process the PDF
    chunks, error = process_pdf_with_ocr(pdf_path, use_ocr)
    
    if error:
        print(f"❌ Error processing PDF: {error}")
        return
    
    if not chunks:
        print("❌ No chunks were extracted from the PDF")
        return
    
    # Display results
    display_results(chunks, pdf_path)
    analyze_pdf_content(chunks)
    
    print(f"\n🎉 PDF processing completed successfully!")

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        process_from_command_line()
    else:
        print("Choose an option:")
        print("1. Process a single PDF file")
        print("2. Batch process all PDFs in a directory")
        print("3. Process from command line (usage: python test_pdf_processor.py <pdf_path> [--ocr])")
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            batch_process_pdfs()
        elif choice == "3":
            print("\nCommand line usage:")
            print("python test_pdf_processor.py <pdf_file_path> [--ocr]")
            print("\nExamples:")
            print("python test_pdf_processor.py C:\\path\\to\\document.pdf")
            print("python test_pdf_processor.py /home/user/document.pdf --ocr")
            print("python test_pdf_processor.py ./local_file.pdf")
        else:
            print("❌ Invalid choice. Exiting.") 