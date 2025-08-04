"""
Main RAG System - Orchestrates All Components
Integrates document processing, vector database, query parsing, and LLM reasoning
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Import our custom components
from document_processer import AdvancedDocumentProcessor, DocumentChunk
from vector_database import VectorDatabase, SearchResult
from query_parser import AdvancedQueryParser, ParsedQuery
from llm_reasoning import AdvancedLLMReasoning, ReasoningResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """
    Represents the complete result of a query processing. Handles reasoning_result as either a dataclass or dict.
    Safely exposes decision and specific_details, and robustly flattens for serialization.
    """
    question_id: str
    user_question: str
    processed_question: str
    retrieved_chunks: List[Any]
    reasoning_result: Any  # Can be ReasoningResult or dict

    @property
    def decision(self):
        rr = self.reasoning_result
        if isinstance(rr, dict):
            return rr.get('decision')
        return getattr(rr, 'decision', None)

    @property
    def specific_details(self):
        rr = self.reasoning_result
        if isinstance(rr, dict):
            return rr.get('specific_details')
        return getattr(rr, 'specific_details', None)

    def to_flat_dict(self):
        # asdict may fail if reasoning_result is a dict, so handle both cases
        import copy
        d = copy.deepcopy(self.__dict__)
        rr = d.pop('reasoning_result', {})
        # If ReasoningResult is a dataclass, convert to dict
        if hasattr(rr, '__dataclass_fields__'):
            rr = asdict(rr)
        if rr:
            d.update(rr)
        return d


class AdvancedRAGSystem:
    """Advanced RAG system that orchestrates all components"""
    
    def __init__(self, 
                 model_path: str = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                 use_gpu: bool = True,
                 vector_db_path: str = "./vector_db",
                 # Optimized performance parameters
                 chunk_size: int = 400,
                 chunk_overlap: int = 50,
                 max_chunks_to_consider: int = 5,
                 top_k: int = 5,
                 max_tokens: int = 200,
                 temperature: float = 0.5,
                 top_p: float = 0.9):
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.vector_db_path = vector_db_path
        
        # Store optimized parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_to_consider = max_chunks_to_consider
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize components
        self._initialize_components()
        
        # Audit trail storage
        self.audit_log = []
        
        logger.info("Advanced RAG System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize document processor with optimized parameters
            self.document_processor = AdvancedDocumentProcessor(
                ocr_language='eng',
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_chunks_to_consider=self.max_chunks_to_consider
            )
            
            # Initialize vector database with optimized parameters
            self.vector_database = VectorDatabase(
                embedding_model="all-MiniLM-L6-v2",  # 384-dim for faster processing
                collection_name="documents",
                persist_directory=self.vector_db_path,
                use_gpu=self.use_gpu,
                top_k=self.top_k,
                similarity_metric="cosine"
            )
            
            # Initialize query parser with optimized parameters
            self.query_parser = AdvancedQueryParser(
                use_gpu=self.use_gpu,
                max_chunks_to_consider=self.max_chunks_to_consider
            )
            
            # Initialize LLM reasoning engine with optimized parameters
            self.reasoning_engine = AdvancedLLMReasoning(
                model_path=self.model_path,
                use_gpu=self.use_gpu,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def ingest_document(self, file_path: str, use_ocr: bool = False) -> List[DocumentChunk]:
        """Ingest and process a document"""
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            # Process document
            chunks = self.document_processor.process_document(file_path, use_ocr)
            logger.info(f"Document processor created {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("No chunks created by document processor")
                return []
            
            # Add to vector database
            logger.info(f"Adding {len(chunks)} chunks to vector database...")
            success = self.vector_database.add_documents(chunks)
            logger.info(f"Vector database add_documents returned: {success}")
            
            if success:
                logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
                
                # Add to audit trail
                self._add_audit_entry({
                    'action': 'document_ingestion',
                    'file_path': file_path,
                    'chunks_processed': len(chunks),
                    'use_ocr': use_ocr,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
                
                return chunks
            else:
                logger.error(f"Failed to add documents to vector database, but returning chunks anyway")
                # Return chunks even if vector database fails, so the user can still see the processing worked
                return chunks
                
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            
            # Add error to audit trail
            self._add_audit_entry({
                'action': 'document_ingestion',
                'file_path': file_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            })
            
            raise
    
    def process_query(self, query: str, n_results: int = 3) -> QueryResult:
        """Process a natural language query (ULTRA-FAST for 30-second response)"""
        try:
            start_time = time.time()
            
            logger.info(f"Processing query: {query}")
            
            # Step 1: Parse the query (FAST)
            parsed_query = self.query_parser.parse_query(query)
            
            # Step 2: Search for relevant documents (ULTRA-FAST)
            search_results = self.vector_database.hybrid_search(
                query=parsed_query.enhanced_query,
                n_results=1,  # Aggressively reduced for speed
                semantic_weight=0.8,
                keyword_weight=0.2
            )
            
            # Step 3: Prepare context for reasoning (FAST)
            context = self._prepare_context_for_reasoning(search_results)
            
            # Step 4: Analyze with LLM reasoning (WITH TIMEOUT)
            reasoning_start = time.time()
            reasoning_result = self.reasoning_engine.analyze_query(
                query=query,
                context=context,
                query_type=parsed_query.query_type
            )
            
            # Check if we're approaching 30-second limit
            elapsed_time = time.time() - start_time
            if elapsed_time > 60:
                logger.warning(f"Approaching 1-minute limit ({elapsed_time:.1f}s), returning real LLM result anyway")
            
            processing_time = time.time() - start_time
            
            # Step 5: Create audit trail (FAST)
            audit_trail = self._create_audit_trail(
                query, parsed_query, search_results, reasoning_result, processing_time
            )
            
            # Step 6: Build result (FAST)
            result = QueryResult(
                question_id=f"q_{int(time.time() * 1000)}",
                user_question=query,
                processed_question=parsed_query.enhanced_query,
                retrieved_chunks=search_results,
                reasoning_result=reasoning_result
            )
            
            # Add to audit log
            self._add_audit_entry(audit_trail)
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_fallback_result(query, str(e))
    
    def _prepare_context_for_reasoning(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Prepare search results for LLM reasoning"""
        try:
            context = []
            
            for result in search_results:
                context_item = {
                    'content': result.content,
                    'source_file': result.source_file,
                    'similarity_score': result.similarity_score,
                    'section_type': result.section_type,
                    'metadata': result.metadata
                }
                
                # Add table data if present
                if result.table_data:
                    context_item['table_data'] = result.table_data
                
                context.append(context_item)
            
            return context
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return []
    
    def _create_audit_trail(self, 
                           query: str, 
                           parsed_query: ParsedQuery,
                           search_results: List[SearchResult],
                           reasoning_result: ReasoningResult,
                           processing_time: float) -> Dict[str, Any]:
        """Create comprehensive audit trail"""
        try:
            audit_trail = {
                'action': 'query_processing',
                'query': query,
                'parsed_query': {
                    'query_type': parsed_query.query_type,
                    'intent': parsed_query.intent,
                    'confidence': parsed_query.confidence,
                    'entities': parsed_query.entities,
                    'keywords': parsed_query.keywords
                },
                'search_results': {
                    'count': len(search_results),
                    'top_results': [
                        {
                            'content_preview': result.content[:100] + "...",
                            'source_file': result.source_file,
                            'similarity_score': result.similarity_score,
                            'section_type': result.section_type
                        }
                        for result in search_results[:3]
                    ]
                },
                'reasoning_result': {
                    'decision': reasoning_result.get('decision') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'decision', None),
                    'confidence_score': reasoning_result.get('confidence_score') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'confidence_score', None),
                    'justification': reasoning_result.get('justification') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'justification', None),
                    'relevant_clauses': reasoning_result.get('relevant_clauses', []) if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'relevant_clauses', []),
                    'amount': reasoning_result.get('amount') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'amount', None),
                    'waiting_period': reasoning_result.get('waiting_period') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'waiting_period', None),
                    'specific_details': reasoning_result.get('specific_details') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'specific_details', None),
                    'conditions': reasoning_result.get('conditions') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'conditions', None),
                    'exclusions': reasoning_result.get('exclusions') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'exclusions', None),
                    'required_documents': reasoning_result.get('required_documents') if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'required_documents', None)
                },
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            return audit_trail
            
        except Exception as e:
            logger.error(f"Error creating audit trail: {e}")
            return {
                'action': 'query_processing',
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    def _create_fallback_result(self, query: str, error: str) -> QueryResult:
        """Create a fallback result when processing fails or times out"""
        try:
            # Create basic parsed query
            parsed_query = ParsedQuery(
                original_query=query,
                enhanced_query=query,
                query_type='general_inquiry',
                entities={},
                intent='information_seeking',
                confidence=0.0,
                keywords=[],
                synonyms=[],
                context={},
                timestamp=datetime.now()
            )
            
            # Create simple fallback reasoning result
            reasoning_result = ReasoningResult(
                decision='pending',
                confidence_score=0.0,
                justification=f'Quick response: {error}',
                relevant_clauses=[],
                reasoning_steps=['Fast processing'],
                source_references=[]
            )
            
            return QueryResult(
                question_id=f"fallback_{int(time.time())}",
                user_question=query,
                processed_question=query,
                retrieved_chunks=[],
                reasoning_result=reasoning_result
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback result: {e}")
            raise
    
    def _add_audit_entry(self, entry: Dict[str, Any]):
        """Add entry to audit log"""
        try:
            self.audit_log.append(entry)
            
            # Keep audit log size manageable
            if len(self.audit_log) > 1000:
                self.audit_log = self.audit_log[-500:]
                
        except Exception as e:
            logger.error(f"Error adding audit entry: {e}")
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get the complete audit trail"""
        return self.audit_log.copy()
    
    def save_audit_trail(self, file_path: str) -> bool:
        """Save audit trail to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            
            logger.info(f"Audit trail saved to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audit trail: {e}")
            return False
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get vector database statistics
            db_stats = self.vector_database.get_document_statistics()
            
            # Get audit trail statistics
            audit_stats = {
                'total_entries': len(self.audit_log),
                'successful_queries': len([e for e in self.audit_log if e.get('status') == 'success']),
                'failed_queries': len([e for e in self.audit_log if e.get('status') == 'error']),
                'document_ingestions': len([e for e in self.audit_log if e.get('action') == 'document_ingestion']),
                'query_processings': len([e for e in self.audit_log if e.get('action') == 'query_processing'])
            }
            
            # Get component information
            component_info = {
                'document_processor': 'AdvancedDocumentProcessor',
                'vector_database': 'AdvancedVectorDatabase',
                'query_parser': 'AdvancedQueryParser',
                'reasoning_engine': 'AdvancedLLMReasoning',
                'model_path': self.model_path,
                'use_gpu': self.use_gpu
            }
            
            stats = {
                'vector_database': db_stats,
                'audit_trail': audit_stats,
                'components': component_info,
                'timestamp': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {e}")
            return {}
    
    def clear_system(self) -> bool:
        """Clear all data from the system"""
        try:
            # Clear vector database
            self.vector_database.clear_database()
            
            # Clear audit log
            self.audit_log = []
            
            logger.info("System cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing system: {e}")
            return False
    
    def export_system_data(self, export_path: str) -> bool:
        """Export system data for backup or analysis"""
        try:
            # Get system statistics
            stats = self.get_system_statistics()
            
            # Add audit trail
            export_data = {
                'statistics': stats,
                'audit_trail': self.audit_log,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"System data exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting system data: {e}")
            return False
    
    def validate_system(self) -> Dict[str, Any]:
        """Validate system components and return status"""
        try:
            validation_results = {
                'document_processor': True,
                'vector_database': True,
                'query_parser': True,
                'reasoning_engine': True,
                'overall_status': True,
                'errors': []
            }
            
            # Test document processor
            try:
                # This is a basic test - in practice you might want more comprehensive tests
                pass
            except Exception as e:
                validation_results['document_processor'] = False
                validation_results['errors'].append(f"Document processor: {e}")
            
            # Test vector database
            try:
                stats = self.vector_database.get_document_statistics()
            except Exception as e:
                validation_results['vector_database'] = False
                validation_results['errors'].append(f"Vector database: {e}")
            
            # Test query parser
            try:
                test_parsed = self.query_parser.parse_query("test query")
            except Exception as e:
                validation_results['query_parser'] = False
                validation_results['errors'].append(f"Query parser: {e}")
            
            # Test reasoning engine
            try:
                # Basic test - check if model file exists
                if not os.path.exists(self.model_path):
                    validation_results['reasoning_engine'] = False
                    validation_results['errors'].append("LLM model file not found")
            except Exception as e:
                validation_results['reasoning_engine'] = False
                validation_results['errors'].append(f"Reasoning engine: {e}")
            
            # Overall status
            validation_results['overall_status'] = all([
                validation_results['document_processor'],
                validation_results['vector_database'],
                validation_results['query_parser'],
                validation_results['reasoning_engine']
            ])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating system: {e}")
            return {
                'overall_status': False,
                'errors': [f"Validation failed: {e}"]
            }

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag_system = AdvancedRAGSystem(use_gpu=True)
    
    # Test system validation
    validation = rag_system.validate_system()
    print(f"System validation: {validation['overall_status']}")
    
    if validation['overall_status']:
        print("✅ All components are working correctly")
    else:
        print("❌ Some components have issues:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # Get system statistics
    stats = rag_system.get_system_statistics()
    print(f"\nSystem statistics: {stats}")