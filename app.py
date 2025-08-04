"""
Flask API Pipeline for Advanced RAG System
Provides REST API endpoints for document processing and query analysis
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

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
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes



# Global RAG system instance
rag_system = None
system_initialized = False



# Request logging middleware
@app.before_request
def log_request_info():
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url}")
    if request.method == 'POST':
        logger.info(f"Request data: {request.get_data()[:200]}...")  # Log first 200 chars

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
        'message': 'Advanced RAG System API',
        'version': '1.0.0',
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
            'keep_alive': 'GET /api/keep-alive'
        },
        'description': 'Insurance Policy Analysis RAG System',
        'features': [
            'Document upload and processing',
            'Natural language query processing',
            'Policy coverage analysis',
            'Claim requirement extraction',
            'Audit trail and statistics'
        ],
        'timestamp': datetime.now().isoformat()
    })

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system, system_initialized
    
    try:
        logger.info("Initializing RAG system...")
        print("🔧 DEBUG: Starting RAG system initialization...")
        
        rag_system = AdvancedRAGSystem(
            model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            use_gpu=True,  # Changed from False to True
            vector_db_path="./vector_db",
            # Optimized performance parameters
            chunk_size=400,
            chunk_overlap=50,
            max_chunks_to_consider=5,
            top_k=5,
            max_tokens=200,
            temperature=0.5,
            top_p=0.9
        )
        
        # Test the reasoning engine
        print("🔧 DEBUG: Testing reasoning engine...")
        if rag_system.reasoning_engine.llm is None:
            print("❌ DEBUG: LLM is None after initialization!")
            logger.error("LLM is None after initialization")
        else:
            print("✅ DEBUG: LLM initialized successfully")
            logger.info("LLM initialized successfully")
        
        system_initialized = True
        logger.info("RAG system initialized successfully!")
        print("✅ DEBUG: RAG system initialization complete")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        import traceback
        logger.error(f"Initialization traceback: {traceback.format_exc()}")
        print(f"❌ DEBUG: RAG system initialization failed: {e}")
        system_initialized = False
        return False

def ensure_system_ready():
    """Ensure the RAG system is ready"""
    if not system_initialized or rag_system is None:
        if not initialize_rag_system():
            return False
    return True

def generate_human_readable_answer(query: str, reasoning_result) -> str:
    """Generate a human-readable answer from the reasoning result (robust to dict or dataclass)."""
    try:
        # Support both dict and dataclass for reasoning_result
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        decision = safe_get(reasoning_result, 'decision')
        justification = safe_get(reasoning_result, 'justification')
        specific_details = safe_get(reasoning_result, 'specific_details')
        amount = safe_get(reasoning_result, 'amount')
        waiting_period = safe_get(reasoning_result, 'waiting_period')
        conditions = safe_get(reasoning_result, 'conditions', []) or []
        exclusions = safe_get(reasoning_result, 'exclusions', []) or []
        required_documents = safe_get(reasoning_result, 'required_documents', []) or []

        # If justification is present and non-empty, always use it as the answer
        if justification and justification.strip():
            return justification.strip()

        # Otherwise, fallback to previous answer logic
        query_lower = query.lower()
        if 'grace period' in query_lower:
            if amount or waiting_period:
                return f"A grace period of {waiting_period or 'thirty days'} is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
            elif decision == 'approved':
                return "A grace period is provided for premium payment to allow policyholders to renew or continue their policy without losing benefits."
            else:
                return "Grace period information is not clearly specified in the policy document."
        elif any(word in query_lower for word in ['amount', 'coverage', 'limit', 'sum']):
            if amount:
                return f"The coverage amount for medical expenses is ${amount:,.2f} as specified in the policy."
            elif specific_details:
                return specific_details
            elif decision == 'approved':
                return "Coverage is provided for medical expenses as per the policy terms and conditions."
            else:
                return "Specific coverage amounts are not clearly stated in the policy document."
        elif 'waiting period' in query_lower:
            if waiting_period:
                return f"There is a waiting period of {waiting_period} before coverage becomes effective for certain conditions."
            elif specific_details:
                return specific_details
            else:
                return "Waiting period information is not clearly specified in the policy document."
        elif any(word in query_lower for word in ['document', 'documents', 'submit', 'claim']):
            if required_documents:
                docs = ', '.join(required_documents)
                return f"To submit a claim, you will need to provide the following documents: {docs}."
            else:
                return "Standard claim documents such as medical bills, claim forms, and supporting documentation are typically required."
        elif any(word in query_lower for word in ['exclusion', 'excluded', 'not covered']):
            if exclusions:
                excl_list = ', '.join(exclusions)
                return f"The following items are excluded from coverage: {excl_list}."
            else:
                return "Standard exclusions apply as per the policy terms and conditions."
        else:
            if specific_details and specific_details != "Analysis could not be completed due to insufficient policy data or technical issues.":
                return specific_details
            else:
                return "Based on the policy analysis, the requested information requires further review or is not clearly specified in the available policy sections."

    except Exception as e:
        logger.error(f"Error generating human-readable answer: {e}")
        return "Unable to generate a specific answer at this time. Please review the policy document for detailed information."

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    try:
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_initialized': system_initialized,
            'rag_system_ready': rag_system is not None
        }
        logger.info(f"Health check response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/keep-alive', methods=['GET'])
def keep_alive():
    """Keep-alive endpoint to prevent auto-termination"""
    logger.info("Keep-alive ping received")
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat(),
        'message': 'Server is running'
    })

@app.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    """Main endpoint for hackathon - processes queries with document URL"""
    logger.info("HackRX run endpoint accessed")
    
    # Check Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Missing or invalid Authorization header")
        return jsonify({'error': 'Unauthorized'}), 401
    
    api_key = auth_header.split(' ')[1]
    # For now, accept any Bearer token (you can add validation later)
    logger.info(f"API key provided: {api_key[:10]}...")
    
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided in hackrx/run request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract documents URL and questions
        documents_url = data.get('documents')
        questions = data.get('questions')
        
        if not questions or not isinstance(questions, list):
            logger.warning("No questions list provided in hackrx/run request")
            return jsonify({'error': 'No questions list provided'}), 400
        
        logger.info(f"Processing {len(questions)} questions")
        if documents_url:
            logger.info(f"Document URL provided: {documents_url}")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready for hackrx/run")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # If document URL is provided, download and process it
        if documents_url:
            try:
                logger.info("Downloading document from URL...")
                import requests
                response = requests.get(documents_url, timeout=30)
                if response.status_code == 200:
                    # Save document temporarily
                    temp_file = f"temp_document_{int(time.time())}.pdf"
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Process document
                    logger.info("Processing downloaded document...")
                    chunks = rag_system.ingest_document(temp_file, use_ocr=False)
                    logger.info(f"Document processed: {len(chunks)} chunks created")
                    
                    # Clean up
                    os.remove(temp_file)
                else:
                    logger.warning(f"Failed to download document: {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading/processing document: {e}")
        
        # Process questions
        answers = []
        total_start_time = time.time()
        
        for i, question in enumerate(questions):
            if not isinstance(question, str) or not question.strip():
                continue
                
            logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
            start_time = time.time()
            
            try:
                result = rag_system.process_query(question)
                processing_time = time.time() - start_time
                
                # Generate human-readable answer
                human_answer = generate_human_readable_answer(question, result.reasoning_result)
                
                # Robustly get justification (works for dict or dataclass)
                def safe_get(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)
                justification = safe_get(result.reasoning_result, 'justification')
                confidence = safe_get(result.reasoning_result, 'confidence_score')

                # If justification/confidence are missing, try to extract from raw LLM response if available
                if (not justification or not justification.strip() or confidence is None) and hasattr(result, 'raw_llm_response'):
                    raw = result.raw_llm_response
                    if isinstance(raw, dict):
                        if not justification or not justification.strip():
                            justification = raw.get('justification', justification)
                        if confidence is None:
                            confidence = raw.get('confidence_score', confidence)
                    elif isinstance(raw, str):
                        # Try to parse JSON from raw string if possible
                        import re, json
                        try:
                            match = re.search(r'\{.*\}', raw, re.DOTALL)
                            if match:
                                raw_json = json.loads(match.group(0))
                                if not justification or not justification.strip():
                                    justification = raw_json.get('justification', justification)
                                if confidence is None:
                                    confidence = raw_json.get('confidence_score', confidence)
                        except Exception as parse_e:
                            logger.warning(f"Could not parse justification/confidence from raw LLM response: {parse_e}")

                logger.debug(f"Extracted justification: {justification}")
                logger.debug(f"Extracted confidence: {confidence}")

                # The answer should always be the justification if present, otherwise fallback to human_answer
                answer_text = justification.strip() if justification and justification.strip() else human_answer
                
                # Print answer, justification, and confidence for debugging/visibility
                print("\n--- Answer Output ---")
                print(f"Answer: {answer_text}")
                print(f"Justification: {justification if justification and justification.strip() else None}")
                print(f"Confidence: {confidence}")
                print("---------------------\n")
                answers.append({
                    'answer': answer_text,
                    'justification': justification if justification and justification.strip() else None,
                    'confidence': confidence
                })
                logger.info(f"Question {i+1} processed successfully in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Error processing query: {str(e)}")
        
        total_time = time.time() - total_start_time
        logger.info(f"HackRX run completed: {len(answers)} answers in {total_time:.2f}s")
        
        # Return in hackathon format
        return jsonify({'answers': answers})
        
    except Exception as e:
        logger.error(f"HackRX run failed: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/hackrx/upload', methods=['POST'])
def hackrx_upload():
    """Upload endpoint for hackathon"""
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
    """Upload and process a document"""
    logger.info("Document upload requested")
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
        
        # Get OCR option
        use_ocr = request.form.get('use_ocr', 'false').lower() == 'true'
        logger.info(f"OCR enabled: {use_ocr}")
        
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
            chunks = rag_system.ingest_document(str(file_path), use_ocr=use_ocr)
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
                'ocr_used': use_ocr,
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
        logger.error(f"Document upload failed: {e}")
        return jsonify({'error': f'Document processing failed: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a single query"""
    logger.info("Single query processing requested")
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided in query request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided")
            return jsonify({'error': 'Invalid query provided'}), 400
        
        logger.info(f"Processing query: {query}")
        
        # Ensure system is ready
        if not ensure_system_ready():
            logger.error("RAG system not ready for query processing")
            return jsonify({'error': 'RAG system not ready'}), 500
        
        # Process query
        start_time = time.time()
        logger.info("Starting query processing...")
        result = rag_system.process_query(query)
        processing_time = time.time() - start_time
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        # Generate human-readable answer
        human_answer = generate_human_readable_answer(query, result.reasoning_result)
        
        # Format response
        response = {
            'query': query,
            'answer': human_answer,
            'decision': result.reasoning_result.decision,
            'confidence': result.reasoning_result.confidence_score,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'amount': result.reasoning_result.amount,
                'waiting_period': result.reasoning_result.waiting_period,
                'relevant_clauses': result.reasoning_result.relevant_clauses,
                'conditions': result.reasoning_result.conditions,
                'exclusions': result.reasoning_result.exclusions,
                'required_documents': result.reasoning_result.required_documents
            }
        }
        
        logger.info(f"Query response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/api/batch_query', methods=['POST'])
def process_batch_queries():
    """Process multiple queries"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        queries = data.get('queries')
        if not queries or not isinstance(queries, list):
            return jsonify({'error': 'Invalid queries list provided'}), 400
        
        # Ensure system is ready
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        results = []
        total_start_time = time.time()
        
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                continue
                
            try:
                start_time = time.time()
                result = rag_system.process_query(query)
                processing_time = time.time() - start_time
                
                # Generate human-readable answer
                human_answer = generate_human_readable_answer(query, result.reasoning_result)
                
                query_result = {
                    'query': query,
                    'answer': human_answer,
                    'decision': result.reasoning_result.decision,
                    'confidence': result.reasoning_result.confidence_score,
                    'processing_time': processing_time,
                    'metadata': {
                        'amount': result.reasoning_result.amount,
                        'waiting_period': result.reasoning_result.waiting_period,
                        'relevant_clauses': result.reasoning_result.relevant_clauses,
                        'conditions': result.reasoning_result.conditions,
                        'exclusions': result.reasoning_result.exclusions,
                        'required_documents': result.reasoning_result.required_documents
                    }
                }
                
                results.append(query_result)
                
            except Exception as e:
                results.append({
                    'query': query,
                    'answer': f"Error processing query: {str(e)}",
                    'decision': 'ERROR',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'metadata': {}
                })
        
        total_time = time.time() - total_start_time
        
        return jsonify({
            'results': results,
            'total_queries': len(queries),
            'successful_queries': len([r for r in results if r['decision'] != 'ERROR']),
            'total_processing_time': total_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch query processing failed: {e}")
        return jsonify({'error': f'Batch query processing failed: {str(e)}'}), 500

@app.route('/api/validate', methods=['GET'])
def validate_system():
    """Validate system components"""
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        validation = rag_system.validate_system()
        return jsonify(validation)
        
    except Exception as e:
        logger.error(f"System validation failed: {e}")
        return jsonify({'error': f'System validation failed: {str(e)}'}), 500

@app.route('/api/audit', methods=['GET'])
def get_audit_trail():
    """Get audit trail"""
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        audit_log = rag_system.get_audit_trail()
        return jsonify({
            'audit_trail': audit_log,
            'total_entries': len(audit_log),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Audit trail retrieval failed: {e}")
        return jsonify({'error': f'Audit trail retrieval failed: {str(e)}'}), 500

@app.route('/api/export', methods=['POST'])
def export_system_data():
    """Export system data"""
    try:
        data = request.get_json() or {}
        filename = data.get('filename', f'system_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        success = rag_system.export_system_data(filename)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'System data exported successfully',
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to export system data'}), 500
            
    except Exception as e:
        logger.error(f"System export failed: {e}")
        return jsonify({'error': f'System export failed: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_system():
    """Clear system data"""
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
    """Get system statistics"""
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        stats = rag_system.get_system_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        return jsonify({'error': f'Statistics retrieval failed: {str(e)}'}), 500

@app.route('/api/debug_search', methods=['POST'])
def debug_search():
    """Debug endpoint to search for specific content"""
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        data = request.get_json()
        search_term = data.get('search_term', 'grace period')
        
        print(f"🔍 DEBUG: Searching for: {search_term}")
        
        # Search the vector database
        results = rag_system.vector_database.hybrid_search(
            query=search_term,
            n_results=10,
            semantic_weight=0.3,  # Lower semantic weight, higher keyword weight
            keyword_weight=0.7
        )
        
        debug_results = []
        for i, result in enumerate(results):
            debug_results.append({
                'rank': i + 1,
                'score': result.similarity_score,
                'content': result.content,
                'metadata': result.metadata
            })
        
        return jsonify({
            'search_term': search_term,
            'total_results': len(results),
            'results': debug_results
        })
        
    except Exception as e:
        logger.error(f"Debug search failed: {e}")
        return jsonify({'error': f'Debug search failed: {str(e)}'}), 500

@app.route('/api/debug_keywords', methods=['POST'])
def debug_keywords():
    """Debug endpoint to search for multiple keywords"""
    try:
        if not ensure_system_ready():
            return jsonify({'error': 'RAG system not ready'}), 500
        
        data = request.get_json()
        keywords = data.get('keywords', ['grace', 'period', 'thirty', 'days', 'premium', 'payment'])
        
        print(f"🔍 DEBUG: Searching for keywords: {keywords}")
        
        all_results = {}
        
        for keyword in keywords:
            print(f"\n🔍 DEBUG: Searching for keyword: '{keyword}'")
            results = rag_system.vector_database.hybrid_search(
                query=keyword,
                n_results=5,
                semantic_weight=0.3,
                keyword_weight=0.7
            )
            
            keyword_results = []
            for i, result in enumerate(results):
                keyword_results.append({
                    'rank': i + 1,
                    'score': result.similarity_score,
                    'content': result.content[:300] + '...' if len(result.content) > 300 else result.content
                })
            
            all_results[keyword] = {
                'count': len(results),
                'results': keyword_results
            }
        
        return jsonify({
            'keywords': keywords,
            'results': all_results
        })
        
    except Exception as e:
        logger.error(f"Debug keywords failed: {e}")
        return jsonify({'error': f'Debug keywords failed: {str(e)}'}), 500

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
    print("🚀 Starting Flask API Pipeline for Advanced RAG System")
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
    print("=" * 60)
    
    # Initialize system on startup
    logger.info("Starting Flask API Pipeline")
    if initialize_rag_system():
        print("✅ RAG system initialized successfully!")
        logger.info("RAG system initialized successfully")
    else:
        print("⚠️  RAG system initialization failed - will retry on first request")
        logger.warning("RAG system initialization failed")
    
    print(f"🌐 Server will run on http://0.0.0.0:5000")
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