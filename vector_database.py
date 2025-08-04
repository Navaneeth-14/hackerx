"""
Advanced Vector Database for Document Storage and Retrieval
Handles document embeddings, similarity search, and metadata management
"""

import os
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Vector database and embedding libraries
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Optional LangChain imports with error handling
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LangChain components not available: {e}")
    print("   Basic functionality will work without LangChain features")
    LANGCHAIN_AVAILABLE = False

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    chunk_id: str
    content: str
    source_file: str
    similarity_score: float
    metadata: Dict[str, Any]
    section_type: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None

class VectorDatabase:
    """Advanced vector database with GPU optimization and hybrid search"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",  # 384-dim for faster processing
                 collection_name: str = "documents",
                 persist_directory: str = "./vector_db",
                 use_gpu: bool = True,
                 top_k: int = 5,  # Reduced from default for faster retrieval
                 similarity_metric: str = "cosine"):
        
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_gpu = use_gpu
        self.top_k = top_k
        self.similarity_metric = similarity_metric
        
        # Initialize embedding model
        self._initialize_embeddings()
        
        # Initialize ChromaDB
        self._initialize_chromadb()
        
        # Initialize LangChain components
        self._initialize_langchain()
        
        logger.info(f"Vector database initialized with model: {embedding_model}")
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            # Use GPU if available and requested
            device = "cuda" if self.use_gpu and self._check_gpu_availability() else "cpu"
            
            self.embedder = SentenceTransformer(self.embedding_model, device=device)
            
            # Initialize LangChain embeddings if available
            if LANGCHAIN_AVAILABLE:
                self.langchain_embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': device}
                )
            else:
                self.langchain_embeddings = None
            
            logger.info(f"Embedding model loaded on device: {device}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _initialize_langchain(self):
        """Initialize LangChain components for advanced retrieval"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain components not available - advanced features disabled")
            self.langchain_chroma = None
            self.contextual_retriever = None
            return
        
        try:
            # Initialize LangChain Chroma
            self.langchain_chroma = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.langchain_embeddings
            )
            
            # Initialize contextual compression retriever
            self.contextual_retriever = ContextualCompressionRetriever(
                base_retriever=self.langchain_chroma.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                ),
                base_compressor=LLMChainExtractor.from_llm(
                    llm=None,  # Will be set later
                    prompt_template="Extract the most relevant information from the following text: {text}"
                )
            )
            
            logger.info("LangChain components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain components: {e}")
            # Continue without LangChain components if they fail
            self.langchain_chroma = None
            self.contextual_retriever = None
    
    def add_documents(self, chunks: List[Any]) -> bool:
        """Add document chunks to the vector database"""
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                # Generate unique ID
                chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else str(uuid.uuid4())
                
                # Get content
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                
                # Create metadata (filter out None values)
                metadata = {
                    'source_file': getattr(chunk, 'source_file', 'unknown'),
                    'file_type': getattr(chunk, 'file_type', 'unknown'),
                    'section_type': getattr(chunk, 'section_type', 'text'),
                    'chunk_index': getattr(chunk, 'chunk_index', 0),
                    'confidence_score': getattr(chunk, 'confidence_score', 1.0),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add page_number only if it's not None
                page_number = getattr(chunk, 'page_number', None)
                if page_number is not None:
                    metadata['page_number'] = page_number
                
                # Add table data if present
                if hasattr(chunk, 'table_data') and chunk.table_data:
                    metadata['table_data'] = json.dumps(chunk.table_data)
                
                # Generate embedding
                embedding = self.embedder.encode(content, convert_to_tensor=False)
                
                ids.append(chunk_id)
                texts.append(content)
                metadatas.append(metadata)
                embeddings.append(embedding.tolist())
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            return False
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a single document to the vector database"""
        try:
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            
            # Create metadata with defaults
            doc_metadata = {
                'source_file': metadata.get('source_file', 'unknown'),
                'file_type': metadata.get('file_type', 'unknown'),
                'section_type': metadata.get('section_type', 'text'),
                'chunk_index': metadata.get('chunk_index', 0),
                'confidence_score': metadata.get('confidence_score', 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add additional metadata
            for key, value in metadata.items():
                if key not in doc_metadata and value is not None:
                    doc_metadata[key] = value
            
            # Generate embedding
            embedding = self.embedder.encode(content, convert_to_tensor=False)
            
            # Add to ChromaDB
            self.collection.add(
                ids=[chunk_id],
                documents=[content],
                metadatas=[doc_metadata],
                embeddings=[embedding.tolist()]
            )
            
            logger.info(f"Successfully added document to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector database: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for documents and return as dictionary format for compatibility"""
        try:
            search_results = self.search_similar(query, n_results, similarity_threshold)
            
            # Convert to dictionary format
            results = []
            for result in search_results:
                results.append({
                    'content': result.content,
                    'source_file': result.source_file,
                    'similarity_score': result.similarity_score,
                    'metadata': result.metadata,
                    'section_type': result.section_type,
                    'table_data': result.table_data
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def search_similar(self, 
                      query: str, 
                      n_results: Optional[int] = None, 
                      similarity_threshold: float = 0.3,  # Lowered default from 0.7 to 0.3
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents using semantic similarity"""
        try:
            # Use class default top_k if n_results not specified
            if n_results is None:
                n_results = self.top_k
            
            # Generate query embedding
            query_embedding = self.embedder.encode(query, convert_to_tensor=False)
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Debug: Log raw search results
            logger.info(f"Raw search returned {len(results['ids'][0])} documents")
            if len(results['ids'][0]) > 0:
                logger.info(f"Sample distances: {results['distances'][0][:3]}")
                logger.info(f"Sample similarity scores: {[1-d for d in results['distances'][0][:3]]}")
            
            # Process results
            search_results = []
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                            # Convert distance to similarity score
            similarity_score = 1 - distance
            
            # Debug: Log raw similarity scores
            logger.debug(f"Raw similarity score: {similarity_score:.3f} for chunk {chunk_id}")
            
            # Filter by similarity threshold (lowered from 0.7 to 0.3)
            if similarity_score >= 0.3:  # Lowered threshold to 30%
                    # Parse table data if present
                    table_data = None
                    if 'table_data' in metadata and metadata['table_data']:
                        try:
                            table_data = json.loads(metadata['table_data'])
                        except:
                            pass
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        source_file=metadata.get('source_file', 'unknown'),
                        similarity_score=similarity_score,
                        metadata=metadata,
                        section_type=metadata.get('section_type', 'text'),
                        table_data=table_data
                    )
                    search_results.append(result)
            
            # Sort by similarity score
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Found {len(search_results)} similar documents for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     n_results: int = 5,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword matching"""
        try:
            # Semantic search
            semantic_results = self.search_similar(query, n_results=n_results*2)
            
            # Keyword search (simple implementation)
            keyword_results = self._keyword_search(query, n_results=n_results*2)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, 
                keyword_results, 
                semantic_weight, 
                keyword_weight
            )
            
            # Return top results
            return combined_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.search_similar(query, n_results)
    
    def _keyword_search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Simple keyword-based search"""
        try:
            # Get all documents
            all_results = self.collection.get()
            
            keyword_results = []
            query_terms = query.lower().split()
            
            for i, content in enumerate(all_results['documents']):
                content_lower = content.lower()
                
                # Calculate keyword match score
                matches = sum(1 for term in query_terms if term in content_lower)
                if matches > 0:
                    score = matches / len(query_terms)
                    
                    result = SearchResult(
                        chunk_id=all_results['ids'][i],
                        content=content,
                        source_file=all_results['metadatas'][i].get('source_file', 'unknown'),
                        similarity_score=score,
                        metadata=all_results['metadatas'][i],
                        section_type=all_results['metadatas'][i].get('section_type', 'text')
                    )
                    keyword_results.append(result)
            
            # Sort by score
            keyword_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return keyword_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _combine_search_results(self, 
                              semantic_results: List[SearchResult],
                              keyword_results: List[SearchResult],
                              semantic_weight: float,
                              keyword_weight: float) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        try:
            # Create a dictionary to store combined scores
            combined_scores = {}
            
            # Add semantic results
            for result in semantic_results:
                combined_scores[result.chunk_id] = {
                    'result': result,
                    'semantic_score': result.similarity_score,
                    'keyword_score': 0.0
                }
            
            # Add keyword results
            for result in keyword_results:
                if result.chunk_id in combined_scores:
                    combined_scores[result.chunk_id]['keyword_score'] = result.similarity_score
                else:
                    combined_scores[result.chunk_id] = {
                        'result': result,
                        'semantic_score': 0.0,
                        'keyword_score': result.similarity_score
                    }
            
            # Calculate combined scores
            combined_results = []
            for chunk_id, scores in combined_scores.items():
                combined_score = (scores['semantic_score'] * semantic_weight + 
                               scores['keyword_score'] * keyword_weight)
                
                # Update the result with combined score
                result = scores['result']
                result.similarity_score = combined_score
                combined_results.append(result)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return semantic_results
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            # Get collection info
            count = self.collection.count()
            
            # Get unique sources
            all_results = self.collection.get()
            sources = set()
            file_types = set()
            section_types = set()
            
            for metadata in all_results['metadatas']:
                sources.add(metadata.get('source_file', 'unknown'))
                file_types.add(metadata.get('file_type', 'unknown'))
                section_types.add(metadata.get('section_type', 'text'))
            
            stats = {
                'total_chunks': count,
                'unique_sources': len(sources),
                'file_types': list(file_types),
                'section_types': list(section_types),
                'sources': list(sources),
                'embedding_model': self.embedding_model,
                'collection_name': self.collection_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {}
    
    def delete_documents(self, source_file: str) -> bool:
        """Delete all documents from a specific source file"""
        try:
            # Get documents from the source
            results = self.collection.get(
                where={"source_file": source_file}
            )
            
            if results['ids']:
                # Delete the documents
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks from {source_file}")
                return True
            else:
                logger.warning(f"No documents found for source: {source_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all documents from the database"""
        try:
            self.collection.delete(where={})
            logger.info("Cleared all documents from vector database")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
    
    def export_database(self, export_path: str) -> bool:
        """Export database statistics and metadata"""
        try:
            stats = self.get_document_statistics()
            
            with open(export_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Database exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting database: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize vector database
    vector_db = VectorDatabase(use_gpu=True)
    
    # Test search
    results = vector_db.search_similar("insurance policy coverage", n_results=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Score: {result.similarity_score:.3f}")
        print(f"Source: {result.source_file}")
        print(f"Content: {result.content[:100]}...")
    
    # Get statistics
    stats = vector_db.get_document_statistics()
    print(f"\nDatabase statistics: {stats}") 