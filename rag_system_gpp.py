"""
GPU-Optimized RAG System for High Performance
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from docx import Document
from bs4 import BeautifulSoup
import requests

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document with metadata"""
    chunk_id: str
    content: str
    source_file: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    category: Optional[str] = None
    embedding: Optional[List[float]] = None

@dataclass
class QueryResult:
    """Structured result from query processing"""
    decision: str  # approved/rejected/conditional
    justification: str
    relevant_clauses: List[str]
    confidence_score: float
    audit_trail: Dict[str, Any]
    amount: Optional[float] = None

class DocumentProcessor:
    """Handles document ingestion and preprocessing with OCR support"""
    
    def __init__(self, ocr_language='eng'):
        self.ocr_language = ocr_language
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = False) -> str:
        """Extract text from PDF with optional OCR for scanned documents"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try to extract text normally first
                page_text = page.get_text()
                
                # If no text found or very little text, use OCR
                if use_ocr or len(page_text.strip()) < 50:
                    logger.info(f"Using OCR for page {page_num + 1}")
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Convert to grayscale for better OCR
                        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        
                        # Apply preprocessing for better OCR
                        img_processed = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        
                        # Extract text using OCR
                        page_text = pytesseract.image_to_string(
                            img_processed, 
                            lang=self.ocr_language,
                            config='--psm 6'
                        )
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                        logger.warning("Continuing with existing text extraction")
                        # Keep the existing page_text (from normal extraction)
                
                text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def chunk_document(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Chunk document into semantically coherent passages"""
        try:
            # Use LangChain's text splitter for better semantic chunking
            docs = [Document(page_content=text, metadata={"source": source_file})]
            split_docs = self.text_splitter.split_documents(docs)
            
            chunks = []
            for i, doc in enumerate(split_docs):
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{i+1}_{hashlib.md5(doc.page_content.encode()).hexdigest()[:8]}",
                    content=doc.page_content.strip(),
                    source_file=source_file,
                    chunk_index=i
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise

class VectorDatabase:
    """Manages vector storage and retrieval with GPU optimization"""
    
    def _clean_metadata(self, metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata by removing None values and converting to proper types"""
        clean_metadata = {}
        for key, value in metadata_dict.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
        return clean_metadata
    
    def __init__(self, persist_directory: str = "./vector_db", use_gpu: bool = True):
        self.persist_directory = persist_directory
        
        # GPU-optimized embeddings
        device = 'cuda' if use_gpu else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        # Initialize ChromaDB
        import chromadb
        from chromadb.config import Settings
        
        # Create ChromaDB client with proper settings
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Create collection for documents
        collection_name = "insurance_documents"
        try:
            self.collection = client.get_collection(collection_name)
        except:
            self.collection = client.create_collection(collection_name)
        
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            client=client,
            collection_name=collection_name
        )
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to vector database"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                documents.append(chunk.content)
                
                # Create metadata dictionary
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "category": chunk.category
                }
                
                # Clean metadata using helper function
                clean_metadata = self._clean_metadata(metadata)
                metadatas.append(clean_metadata)
                ids.append(chunk.chunk_id)
            
            # Get embeddings for documents (GPU-accelerated)
            embeddings = self.embeddings.embed_documents(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        try:
            # Get query embedding (GPU-accelerated)
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Clean metadata using helper function
                    clean_metadata = self._clean_metadata(metadata)
                    
                    search_results.append({
                        "content": doc,
                        "metadata": clean_metadata,
                        "similarity_score": 1.0 - float(distance)  # Convert distance to similarity
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            raise

class QueryParser:
    """Parses and structures natural language queries"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from natural language query"""
        try:
            prompt = f"""
            Extract structured information from the following query about insurance/policy:
            
            Query: {query}
            
            Extract the following information in JSON format:
            {{
                "age": <age if mentioned>,
                "procedure": <medical procedure if mentioned>,
                "location": <location if mentioned>,
                "policy_type": <type of policy mentioned>,
                "claim_amount": <amount if mentioned>,
                "condition": <medical condition if mentioned>,
                "intent": <what the user is asking about>
            }}
            
            JSON Response:
            """
            
            # Call LLM with proper format
            if hasattr(self.llm, 'generate_content'):  # Gemini
                response = self.llm.generate_content(prompt).text
            else:  # Llama
                response = self.llm(prompt, max_tokens=200, temperature=0.1, stop=["\n\n"])
                if isinstance(response, dict):
                    response = response.get('choices', [{}])[0].get('text', '')
                elif hasattr(response, 'choices'):
                    response = response.choices[0].text
            
            # Parse response - handle both string and dict responses
            try:
                if isinstance(response, dict):
                    entities = response
                else:
                    entities = json.loads(response)
                return entities
            except (json.JSONDecodeError, TypeError):
                # Fallback parsing
                return self._fallback_entity_extraction(query)
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return self._fallback_entity_extraction(query)
    
    def _fallback_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Simple fallback entity extraction"""
        entities = {
            "age": None,
            "procedure": None,
            "location": None,
            "policy_type": None,
            "claim_amount": None,
            "condition": None,
            "intent": "general_inquiry"
        }
        
        # Simple keyword-based extraction
        query_lower = query.lower()
        
        # Extract age
        import re
        age_match = re.search(r'(\d+)\s*(?:years?|yrs?)', query_lower)
        if age_match:
            entities["age"] = int(age_match.group(1))
        
        # Extract amount
        amount_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs?|rupees?|inr)', query_lower)
        if amount_match:
            entities["claim_amount"] = float(amount_match.group(1).replace(',', ''))
        
        return entities

class LLMReasoning:
    """Handles LLM-based reasoning and decision logic"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
    
    def analyze_query(self, query: str, relevant_chunks: List[Dict], parsed_entities: Dict) -> QueryResult:
        """Analyze query against relevant document chunks"""
        try:
            # Prepare context from relevant chunks
            context = "\n\n".join([
                f"Document {i+1}:\n{chunk['content']}\nClause ID: {chunk['metadata'].get('chunk_id', 'N/A')}"
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            # Create reasoning prompt
            prompt = f"""
            You are an insurance policy analyzer. Analyze the following query against the provided policy documents.
            
            User Query: {query}
            Extracted Entities: {json.dumps(parsed_entities, indent=2)}
            
            Relevant Policy Clauses:
            {context}
            
            Please provide a structured analysis in the following JSON format:
            {{
                "decision": "approved/rejected/conditional",
                "amount": <amount if applicable, null otherwise>,
                "justification": "<detailed explanation with specific clause references>",
                "relevant_clauses": ["<list of clause IDs that support the decision>"],
                "confidence_score": <0.0 to 1.0>,
                "conditions": ["<any conditions that must be met>"]
            }}
            
            Base your decision on:
            1. Policy coverage and exclusions
            2. Eligibility criteria
            3. Waiting periods
            4. Pre-existing conditions
            5. Specific terms and conditions
            
            JSON Response:
            """
            
            # Call LLM with proper format
            if hasattr(self.llm, 'generate_content'):  # Gemini
                response = self.llm.generate_content(prompt).text
            else:  # Llama
                response = self.llm(prompt, max_tokens=500, temperature=0.1, stop=["\n\n"])
                if isinstance(response, dict):
                    response = response.get('choices', [{}])[0].get('text', '')
                elif hasattr(response, 'choices'):
                    response = response.choices[0].text
            
            # Parse response - handle both string and dict responses
            try:
                if isinstance(response, dict):
                    result_data = response
                else:
                    result_data = json.loads(response)
                
                # Create audit trail
                audit_trail = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "parsed_entities": parsed_entities,
                    "relevant_chunks_count": len(relevant_chunks),
                    "llm_prompt": prompt,
                    "llm_response": response,
                    "chunk_ids": [chunk['metadata'].get('chunk_id') for chunk in relevant_chunks]
                }
                
                return QueryResult(
                    decision=result_data.get("decision", "conditional"),
                    amount=result_data.get("amount"),
                    justification=result_data.get("justification", "Analysis incomplete"),
                    relevant_clauses=result_data.get("relevant_clauses", []),
                    confidence_score=result_data.get("confidence_score", 0.5),
                    audit_trail=audit_trail
                )
                
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                # Fallback response
                return QueryResult(
                    decision="conditional",
                    amount=None,
                    justification=f"Unable to parse LLM response: {str(e)}. Raw response: {str(response)[:200]}...",
                    relevant_clauses=[],
                    confidence_score=0.3,
                    audit_trail={
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "error": f"JSON parsing failed: {str(e)}",
                        "raw_response": str(response)[:500]
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            return QueryResult(
                decision="conditional",
                amount=None,
                justification=f"Error in analysis: {str(e)}",
                relevant_clauses=[],
                confidence_score=0.0,
                audit_trail={
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "error": str(e)
                }
            )

class RAGSystem:
    """GPU-Optimized RAG system orchestrating all components"""
    
    def __init__(self, model_path: str = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf", use_gpu: bool = True):
        # Initialize components
        self.document_processor = DocumentProcessor()
        
        # Initialize vector database with GPU optimization
        try:
            self.vector_db = VectorDatabase(use_gpu=use_gpu)
            logger.info(f"Vector database initialized successfully (GPU: {use_gpu})")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
        
        # Initialize LLM with GPU optimization
        try:
            logger.info(f"Loading model from: {model_path} (GPU: {use_gpu})")
            
            from llama_cpp import Llama
            
            # GPU-optimized configuration
            if use_gpu:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=8,  # More threads for GPU
                    n_gpu_layers=35,  # Use GPU layers
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    seed=42
                )
                logger.info("GPU-optimized LLM model loaded successfully")
            else:
                # CPU fallback
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=4,
                    n_gpu_layers=0,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    seed=42
                )
                logger.info("CPU LLM model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load local model: {e}")
            logger.info("Falling back to Gemini model")
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.llm = genai.GenerativeModel("gemini-1.5-pro")
                logger.info("Gemini model loaded successfully")
            except ImportError:
                logger.error("Google Generative AI not available. Install with: pip install google-generativeai")
                raise Exception("No LLM model available. Please install google-generativeai or ensure local model file exists.")
            except Exception as gemini_error:
                logger.error(f"Failed to load both local and Gemini models: {gemini_error}")
                raise Exception("No LLM model available. Please check model file or API key.")
        
        self.query_parser = QueryParser(self.llm)
        self.reasoning_engine = LLMReasoning(self.llm)
        
        # Audit trail storage
        self.audit_log = []
    
    def ingest_document(self, file_path: str, use_ocr: bool = False) -> List[DocumentChunk]:
        """Ingest and process a document"""
        try:
            file_path = Path(file_path)
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.document_processor.extract_text_from_pdf(str(file_path), use_ocr)
            elif file_path.suffix.lower() == '.docx':
                text = self.document_processor.extract_text_from_docx(str(file_path))
            elif file_path.suffix.lower() == '.html':
                text = self.document_processor.extract_text_from_html(str(file_path))
            elif file_path.suffix.lower() == '.eml':
                text = self.document_processor.extract_text_from_email(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Chunk the document
            chunks = self.document_processor.chunk_document(text, str(file_path))
            
            # Add to vector database
            self.vector_db.add_documents(chunks)
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            raise
    
    def process_query(self, query: str) -> QueryResult:
        """Process a natural language query"""
        try:
            # Step 1: Parse and structure the query
            parsed_entities = self.query_parser.extract_entities(query)
            
            # Step 2: Semantic retrieval
            relevant_chunks = self.vector_db.search(query, k=5)
            
            # Step 3: LLM reasoning and decision logic
            result = self.reasoning_engine.analyze_query(query, relevant_chunks, parsed_entities)
            
            # Step 4: Store audit trail
            self.audit_log.append(result.audit_trail)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                decision="error",
                amount=None,
                justification=f"System error: {str(e)}",
                relevant_clauses=[],
                confidence_score=0.0,
                audit_trail={
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "error": str(e)
                }
            )
    
    def get_audit_trail(self) -> List[Dict]:
        """Get complete audit trail"""
        return self.audit_log
    
    def save_audit_trail(self, file_path: str):
        """Save audit trail to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            logger.info(f"Audit trail saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audit trail: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize GPU-optimized RAG system
    rag_system = RAGSystem(use_gpu=True)  # Set to False for CPU-only
    
    # Ingest sample document
    print("Ingesting sample document...")
    chunks = rag_system.ingest_document("sample.pdf", use_ocr=False)
    
    # Process example queries
    example_queries = [
        "Is heart surgery covered under this policy?",
        "What is the waiting period for pre-existing diseases?",
        "Can I claim for dental treatment?",
        "What is the maximum coverage amount?"
    ]
    
    print("\nProcessing queries...")
    for query in example_queries:
        print(f"\nQuery: {query}")
        result = rag_system.process_query(query)
        print(f"Decision: {result.decision}")
        print(f"Justification: {result.justification}")
        print(f"Confidence: {result.confidence_score}")
    
    # Save audit trail
    rag_system.save_audit_trail("audit_trail.json") 