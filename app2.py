"""
Flask API Pipeline for RAG System
Dedicated API server for the Advanced RAG System
Separate from main.py (command-line pipeline)
"""

from flask import Flask, request, jsonify, send_file
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import requests

# Import RAG system components
from rag_system import AdvancedRAGSystem, QueryResult
from document_processer import AdvancedDocumentProcessor, DocumentChunk
from vector_database import VectorDatabase, SearchResult
from query_parser import AdvancedQueryParser, ParsedQuery
from llm_reasoning import AdvancedLLMReasoning, ReasoningResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global RAG system instance
rag_system = None
system_initialized = False

# Request logging middleware
@app.before_request
def log_request_info():
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url}")
    if request.method == 'POST':
        logger.info(f"Request data: {request.get_data()[:200]}...")

@app.after_request
def log_response_info(response):
    """Log all outgoing responses"""
    logger.info(f"Response: {response.status_code} for {request.method} {request.url}")
    return response

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    logger.info("Root endpoint accessed")
    return jsonify({
        'message': 'RAG System API Pipeline',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': {
            'health': 'GET /api/health',
            'status': 'GET /api/status',
            'upload': 'POST /api/upload',
            'query': 'POST /api/query',
            'batch_query': 'POST /api/batch_query',
            'validate': 'GET /api/validate',
            'audit': 'GET /api/audit',
            'statistics': 'GET /api/statistics',
            'export': 'POST /api/export',
            'clear': 'POST /api/clear',
            'keep_alive': 'GET /api/keep-alive',
            'hackrx_run': 'POST /hackrx/run',
            'hackrx_upload': 'POST /hackrx/upload'
        },
        'description': 'Advanced RAG System API Pipeline',
        'features': [
            'Document upload and processing',
            'Natural language query processing',
            'Policy coverage analysis',
            'Claim requirement extraction',
            'Audit trail and statistics',
            'Hackathon-compatible endpoints'
        ],
        'timestamp': datetime.now().isoformat()
    })

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system, system_initialized
    
    try:
        logger.info("Initializing RAG system...")
        rag_system = AdvancedRAGSystem(
            model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            use_gpu=False,  # Use CPU for better compatibility
            vector_db_path="./vector_db"
        )
        system_initialized = True
        logger.info("RAG system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"RAG system initialization failed: {e}")
        return False

def ensure_system_ready():
    """Ensure RAG system is ready"""
    if not system_initialized:
        return initialize_rag_system()
    return True

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_initialized': system_initialized,
            'rag_system_ready': system_initialized,
            'message': 'RAG System API is running'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'error': f'Health check failed: {str(e)}'}), 500

@app.route('/api/keep-alive', methods=['GET'])
def keep_alive():
    """Keep-alive endpoint to prevent server termination"""
    logger.info("Keep-alive ping received")
    return jsonify({
        'status': 'alive', 
        'timestamp': datetime.now().isoformat(), 
        'message': 'RAG System API is running'
    })

@app.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    """Hackathon run endpoint"""
    logger.info("HackRX run endpoint accessed")
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header")
            return jsonify({'error': 'Unauthorized'}), 401
        
        api_key = auth_header.split(' ')[1]
        # For now, accept any Bearer token
        logger.info(f"API key provided: {api_key[:10]}...")
        
        # Get request data
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        documents_url = data.get('documents')
        questions = data.get('questions', [])
        
        if not questions:
            logger.warning("No questions provided")
            return jsonify({'error': 'No questions provided'}), 400
        
        logger.info(f"Processing {len(questions)} questions")
        if documents_url:
            logger.info(f"Document URL provided: {documents_url}")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Download and process document if URL provided
        if documents_url:
            try:
                logger.info("Downloading document from URL...")
                response = requests.get(documents_url, timeout=30)
                response.raise_for_status()
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(response.content)
                temp_file.close()
                
                logger.info(f"Document downloaded to: {temp_file.name}")
                
                # Ingest document
                chunks = rag_system.ingest_document(temp_file.name, use_ocr=False)
                logger.info(f"Document ingested: {len(chunks)} chunks created")
                
                # Clean up
                os.unlink(temp_file.name)
                
            except Exception as e:
                logger.error(f"Document download/processing failed: {e}")
                return jsonify({'error': f'Document processing failed: {str(e)}'}), 500
        
        # Process questions
        answers = []
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                result = rag_system.process_query(question)
                
                if result and result.reasoning_result:
                    answer = result.reasoning_result.justification
                    answers.append(answer)
                    logger.info(f"Question {i+1} processed successfully")
                else:
                    answers.append("Unable to process this question.")
                    logger.warning(f"Question {i+1} failed to process")
                    
            except Exception as e:
                logger.error(f"Question {i+1} processing failed: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        response_data = {'answers': answers}
        logger.info(f"HackRX run completed: {len(answers)} answers generated")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"HackRX run failed: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/hackrx/upload', methods=['POST'])
def hackrx_upload():
    """Hackathon upload endpoint"""
    logger.info("HackRX upload endpoint accessed")
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            logger.warning("No file provided in hackrx/upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename in hackrx/upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready for upload")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Check file type
        supported_extensions = {'.pdf', '.txt', '.docx', '.html', '.htm', '.eml', '.msg', '.csv', '.json'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            logger.warning(f"Unsupported file type: {file_extension}")
            return jsonify({'error': f'Unsupported file type: {file_extension}'}), 400
        
        # Save uploaded file
        upload_dir = Path('uploads')
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        file.save(str(file_path))
        logger.info(f"File saved to: {file_path}")
        
        try:
            # Process document
            start_time = time.time()
            logger.info("Starting document ingestion...")
            chunks = rag_system.ingest_document(str(file_path), use_ocr=False)
            processing_time = time.time() - start_time
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks created in {processing_time:.2f}s")
            
            # Clean up uploaded file
            os.remove(str(file_path))
            logger.info("Temporary file cleaned up")
            
            response = {
                'success': True,
                'message': 'Document processed successfully',
                'filename': file.filename,
                'chunks_processed': len(chunks),
                'processing_time': processing_time,
                'file_type': file_extension,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"HackRX upload response: {response}")
            return jsonify(response)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(str(file_path)):
                os.remove(str(file_path))
                logger.info("Cleaned up file after error")
            logger.error(f"Document processing error: {e}")
            raise e
            
    except Exception as e:
        logger.error(f"HackRX upload failed: {e}")
        return jsonify({'error': f'Document processing failed: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def system_status():
    """Get detailed system status"""
    try:
        if not ensure_system_ready():
            return jsonify({
                'status': 'error',
                'message': 'RAG system initialization failed'
            }), 500
        
        # Get system statistics
        stats = rag_system.get_system_statistics()
        
        return jsonify({
            'status': 'ready',
            'system_statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Status check failed: {str(e)}'
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload document endpoint"""
    logger.info("Upload endpoint accessed")
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            logger.warning("No file provided in upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename in upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready for upload")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Check file type
        supported_extensions = {'.pdf', '.txt', '.docx', '.html', '.htm', '.eml', '.msg', '.csv', '.json'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            logger.warning(f"Unsupported file type: {file_extension}")
            return jsonify({'error': f'Unsupported file type: {file_extension}'}), 400
        
        # Save uploaded file
        upload_dir = Path('uploads')
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        file.save(str(file_path))
        logger.info(f"File saved to: {file_path}")
        
        try:
            # Process document
            start_time = time.time()
            logger.info("Starting document ingestion...")
            chunks = rag_system.ingest_document(str(file_path), use_ocr=False)
            processing_time = time.time() - start_time
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks created in {processing_time:.2f}s")
            
            # Clean up uploaded file
            os.remove(str(file_path))
            logger.info("Temporary file cleaned up")
            
            response = {
                'success': True,
                'message': 'Document uploaded and processed successfully',
                'filename': file.filename,
                'chunks_processed': len(chunks),
                'processing_time': processing_time,
                'file_type': file_extension,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Upload response: {response}")
            return jsonify(response)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(str(file_path)):
                os.remove(str(file_path))
                logger.info("Cleaned up file after error")
            logger.error(f"Document processing error: {e}")
            raise e
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': f'Document processing failed: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process single query endpoint"""
    logger.info("Query endpoint accessed")
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        if not query:
            logger.warning("No query provided")
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Process query
        start_time = time.time()
        result = rag_system.process_query(query)
        processing_time = time.time() - start_time
        
        if result:
            response = {
                'success': True,
                'query': query,
                'answer': result.reasoning_result.justification if result.reasoning_result else "No answer generated",
                'confidence': result.reasoning_result.confidence if result.reasoning_result else 0.0,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return jsonify(response)
        else:
            logger.warning("Query processing returned no result")
            return jsonify({'error': 'No result generated'}), 500
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/api/batch_query', methods=['POST'])
def process_batch_queries():
    """Process multiple queries endpoint"""
    logger.info("Batch query endpoint accessed")
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        queries = data.get('queries', [])
        if not queries:
            logger.warning("No queries provided")
            return jsonify({'error': 'No queries provided'}), 400
        
        logger.info(f"Processing {len(queries)} queries")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Process queries
        results = []
        start_time = time.time()
        
        for i, query in enumerate(queries):
            try:
                logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
                result = rag_system.process_query(query)
                
                if result and result.reasoning_result:
                    results.append({
                        'query': query,
                        'answer': result.reasoning_result.justification,
                        'confidence': result.reasoning_result.confidence,
                        'success': True
                    })
                else:
                    results.append({
                        'query': query,
                        'answer': "No answer generated",
                        'confidence': 0.0,
                        'success': False
                    })
                    
            except Exception as e:
                logger.error(f"Query {i+1} processing failed: {e}")
                results.append({
                    'query': query,
                    'answer': f"Error: {str(e)}",
                    'confidence': 0.0,
                    'success': False
                })
        
        processing_time = time.time() - start_time
        
        response = {
            'success': True,
            'queries_processed': len(queries),
            'successful_queries': len([r for r in results if r['success']]),
            'processing_time': processing_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch query completed: {len(results)} results in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch query processing failed: {e}")
        return jsonify({'error': f'Batch query processing failed: {str(e)}'}), 500

@app.route('/api/validate', methods=['GET'])
def validate_system():
    """Validate system endpoint"""
    logger.info("Validate endpoint accessed")
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Perform validation checks
        validation_results = {
            'rag_system_ready': system_initialized,
            'vector_db_accessible': True,  # Add actual check if needed
            'model_loaded': True,  # Add actual check if needed
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(validation_results)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500

@app.route('/api/audit', methods=['GET'])
def get_audit_trail():
    """Get audit trail endpoint"""
    logger.info("Audit endpoint accessed")
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Get audit trail from RAG system
        audit_trail = rag_system.get_audit_trail()
        
        return jsonify({
            'audit_trail': audit_trail,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Audit trail retrieval failed: {e}")
        return jsonify({'error': f'Audit trail retrieval failed: {str(e)}'}), 500

@app.route('/api/export', methods=['POST'])
def export_system_data():
    """Export system data endpoint"""
    logger.info("Export endpoint accessed")
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        data = request.get_json() or {}
        export_format = data.get('format', 'json')
        
        # Export system data
        export_data = rag_system.export_system_data(export_format)
        
        if export_data:
            return jsonify({
                'success': True,
                'format': export_format,
                'data': export_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to export system data'}), 500
            
    except Exception as e:
        logger.error(f"System export failed: {e}")
        return jsonify({'error': f'System export failed: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_system():
    """Clear system data endpoint"""
    logger.info("Clear endpoint accessed")
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        success = rag_system.clear_system()
        
        return jsonify({
            'success': success,
            'message': 'System cleared successfully' if success else 'Failed to clear system',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"System clear failed: {e}")
        return jsonify({'error': f'System clear failed: {str(e)}'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics endpoint"""
    logger.info("Statistics endpoint accessed")
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        stats = rag_system.get_system_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        return jsonify({'error': f'Statistics retrieval failed: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting RAG System API Pipeline (app2.py)")
    print("=" * 60)
    print("📋 Available endpoints:")
    print("   GET  /api/health      - Health check")
    print("   GET  /api/status      - System status")
    print("   POST /api/upload      - Upload document")
    print("   POST /api/query       - Process single query")
    print("   POST /api/batch_query - Process multiple queries")
    print("   GET  /api/validate    - Validate system")
    print("   GET  /api/audit       - Get audit trail")
    print("   POST /api/export      - Export system data")
    print("   POST /api/clear       - Clear system")
    print("   GET  /api/statistics  - Get statistics")
    print("   POST /hackrx/run      - Hackathon run endpoint")
    print("   POST /hackrx/upload   - Hackathon upload endpoint")
    print("=" * 60)
    
    # Initialize system on startup
    logger.info("Starting RAG System API Pipeline")
    if initialize_rag_system():
        print("✅ RAG system initialized successfully!")
        logger.info("RAG system initialized successfully")
    else:
        print("⚠️  RAG system initialization failed - will retry on first request")
        logger.warning("RAG system initialization failed")
    
    print(f"🌐 Server will run on http://127.0.0.1:5000")
    print("=" * 60)
    logger.info("Starting Flask server...")
    
    try:
        app.run(
            debug=False, 
            host='0.0.0.0', 
            port=5000, 
            threaded=True,
            use_reloader=False  # Prevent auto-restart issues
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
        print("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        print(f"❌ Server crashed: {e}")
        raise 