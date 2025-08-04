"""
Advanced LLM Reasoning Engine for Query Analysis and Response Generation
Handles complex reasoning, clause referencing, and structured response generation
"""

import os
import json
import logging
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# LLM and AI libraries
from llama_cpp import Llama
from transformers import pipeline
import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Configuration (aggressively optimized for speed)
LLM_MAX_TOKENS = 32  # Very short output for fastest completion
LLM_CONTEXT_SIZE = 512  # Minimal context for speed
LLM_BATCH_SIZE = 256  # Max throughput
LLM_GPU_LAYERS = 15
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.85
# REMOVED LLM_TIMEOUT - NO TIMEOUTS

@dataclass
class ReasoningResult:
    """Represents the result of LLM reasoning"""
    decision: str  # approved, denied, pending, unclear
    confidence_score: float
    justification: str
    relevant_clauses: List[str]
    specific_details: Optional[str] = None  # For exact information requested
    amount: Optional[float] = None
    waiting_period: Optional[str] = None
    conditions: List[str] = None
    exclusions: List[str] = None
    required_documents: List[str] = None
    processing_time: Optional[str] = None
    reasoning_steps: List[str] = None
    source_references: List[Dict[str, Any]] = None
    policy_quotes: List[str] = None  # Exact policy text quotes

@dataclass
class ClauseReference:
    """Represents a reference to a specific policy clause"""
    clause_id: str
    clause_text: str
    relevance_score: float
    page_number: Optional[int] = None
    section_type: Optional[str] = None



class AdvancedLLMReasoning:
    """Advanced LLM reasoning engine with clause referencing and structured analysis"""
    
    def __init__(self, 
                 model_path: str = None,
                 use_gpu: bool = True,
                 max_tokens: int = 10,  # Reduced for speed (approx 10 seconds)
                 temperature: float = 0.2,  # Lower for more focused output
                 top_p: float = 0.9):  # Limits sampling space
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Simple cache for responses
        self.response_cache = {}
        
        print(f"🔧 DEBUG: Initializing AdvancedLLMReasoning with:")
        print(f"  Model path: {self.model_path}")
        print(f"  Use GPU: {self.use_gpu}")
        print(f"  Max tokens: {self.max_tokens}")
        
        logger.info(f"Initializing AdvancedLLMReasoning with:")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Use GPU: {self.use_gpu}")
        logger.info(f"  Max tokens: {self.max_tokens}")
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize reasoning patterns
        self._initialize_reasoning_patterns()
        
        # Initialize clause extraction
        self._initialize_clause_extraction()
        
        logger.info(f"AdvancedLLMReasoning initialization complete. LLM status: {self.llm is not None}")
        print(f"✅ DEBUG: AdvancedLLMReasoning initialization complete. LLM status: {self.llm is not None}")
        if self.llm is not None:
            logger.info("✅ LLM initialized successfully")
            print("✅ DEBUG: LLM initialized successfully")
        else:
            logger.error("❌ LLM initialization failed - will use fallback mode")
            print("❌ DEBUG: LLM initialization failed - will use fallback mode")
    
    def _initialize_llm(self):
        """Initialize the LLM model with speed optimizations"""
        try:
            # Set default model path if none provided
            if self.model_path is None:
                self.model_path = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            
            print(f"🔧 DEBUG: Checking model file: {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("LLM reasoning will use fallback mode without local model")
                print(f"❌ DEBUG: Model file not found: {self.model_path}")
                self.llm = None
                return
            else:
                logger.info(f"Model file found: {self.model_path}")
                logger.info(f"Model file size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
                print(f"✅ DEBUG: Model file found: {self.model_path}")
                print(f"✅ DEBUG: Model file size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
            
            # Initialize Llama model with SPEED optimization
            try:
                logger.info("Initializing LLM with speed optimizations...")
                print("🔧 DEBUG: Initializing LLM with speed optimizations...")
                
                load_start = time.time()
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=256,  # Reduce context for speed
                    n_batch=32,  # Lower batch size for memory and speed
                    n_gpu_layers=LLM_GPU_LAYERS if self.use_gpu else 0,
                    verbose=False,  # Disable verbose for speed
                    use_mmap=True,
                    use_mlock=False,
                    seed=42,
                    n_threads=os.cpu_count(),  # Use all CPU cores
                    n_threads_batch=os.cpu_count(),
                    rope_scaling=None,
                    mul_mat_q=True,
                    f16_kv=True,
                    vocab_only=False,
                    embedding=False,
                    offload_kqv=self.use_gpu,
                    numa=False,
                    logits_all=False,
                )
                load_time = time.time() - load_start
                logger.info(f"LLM loaded in {load_time:.2f} seconds. GPU used: {self.use_gpu}, n_gpu_layers: {LLM_GPU_LAYERS if self.use_gpu else 0}")
                print(f"✅ DEBUG: LLM loaded in {load_time:.2f}s, GPU: {self.use_gpu}, n_gpu_layers: {LLM_GPU_LAYERS if self.use_gpu else 0}")
                
                logger.info("LLM initialized successfully with speed optimizations")
                logger.info(f"LLM object created: {self.llm}")
                print("✅ DEBUG: LLM object created successfully")
                
                # Test the LLM with a simple prompt
                try:
                    logger.info("Testing LLM with simple prompt...")
                    print(" DEBUG: Testing LLM with simple prompt...")
                    test_response = self.llm.create_completion(
                        prompt="Hello, this is a test.",
                        max_tokens=10,
                        temperature=0.0
                    )
                    logger.info(f"LLM test successful: {test_response}")
                    print(f"✅ DEBUG: LLM test successful: {test_response}")
                except Exception as test_e:
                    logger.error(f"LLM test failed: {test_e}")
                    logger.error(f"Test error type: {type(test_e)}")
                    import traceback
                    logger.error(f"Test traceback: {traceback.format_exc()}")
                    print(f"❌ DEBUG: LLM test failed: {test_e}")
                    print(f"❌ DEBUG: Test traceback: {traceback.format_exc()}")
                    self.llm = None
                
            except Exception as e:
                logger.error(f"Error initializing LLM: {e}")
                import traceback
                logger.error(f"Initialization traceback: {traceback.format_exc()}")
                logger.info("Falling back to rule-based analysis only")
                print(f"❌ DEBUG: Error initializing LLM: {e}")
                print(f"❌ DEBUG: Initialization traceback: {traceback.format_exc()}")
                self.llm = None
                
        except Exception as e:
            logger.error(f"Error in LLM initialization: {e}")
            print(f"❌ DEBUG: Error in LLM initialization: {e}")
            self.llm = None
    
    def _initialize_reasoning_patterns(self):
        """Initialize reasoning patterns and templates"""
        try:
            # Decision mapping for different domains, expanded for robustness
            self.decision_mapping = {
                # Insurance domain
                'COVERED': 'approved',
                'NOT_COVERED': 'denied',
                'CONDITIONAL': 'pending',
                'APPROVED': 'approved',
                'REJECTED': 'denied',
                'DENIED': 'denied',
                'PENDING': 'pending',
                'PENDING_REVIEW': 'pending',
                'NOT_FOUND': 'denied',
                'UNCLEAR': 'pending',
                'ERROR': 'pending',
                'NO': 'denied',
                'YES': 'approved',
                # Legal compliance domain
                'COMPLIANT': 'approved',
                'NON_COMPLIANT': 'denied',
                'NEEDS_REVIEW': 'pending',
                # Contract domain
                'PERMITTED': 'approved',
                'PROHIBITED': 'denied',
                'NEEDS_CLARIFICATION': 'pending',
                # Policy review
                'CLEAR': 'approved',
                'UNCLEAR': 'pending',
            }
            # Reasoning templates for different query types and domains
            self.reasoning_templates = {
                'coverage_check': """
                You are an expert document analyst. Analyze the provided document information carefully to answer the user's query.
                
                USER QUERY: {query}
                
                DOCUMENT SECTIONS:
                {context}
                
                INSTRUCTIONS:
                1. Read and understand the document sections provided
                2. Look for specific clauses, conditions, limitations, and exclusions
                3. Pay attention to time periods, limits, and restrictions
                4. Consider both what is allowed/permitted AND what is explicitly prohibited/excluded
                5. Base your decision ONLY on the information provided in the document
                
                ANALYSIS REQUIREMENTS:
                - Decision: APPROVED, DENIED, CONDITIONAL, or PENDING (be precise based on document text)
                - Confidence Score: 0.0 to 1.0 (higher if document clearly states the answer)
                - Justification: Quote specific document text and explain your reasoning
                - Relevant Clauses: List the exact document sections that support your decision
                - Conditions: Any specific conditions, time limits, or restrictions mentioned
                - Exclusions: What is explicitly excluded or not permitted
                - Required Documents: Documents mentioned as required for this type of request
                
                IMPORTANT: If the document explicitly states something is NOT permitted or has limitations, you must reflect that in your decision. Do not assume approval unless the document clearly states it.
                
                Respond in valid JSON format.
                """,
                
                'legal_compliance': """
                You are an expert legal compliance analyst. Analyze the provided legal documents to determine compliance status.
                
                USER QUERY: {query}
                
                LEGAL DOCUMENT SECTIONS:
                {context}
                
                INSTRUCTIONS:
                1. Read and understand the legal document sections provided
                2. Look for specific regulations, requirements, and compliance criteria
                3. Pay attention to deadlines, obligations, and legal requirements
                4. Consider both what is required AND what is explicitly prohibited
                5. Base your decision ONLY on the information provided in the legal documents
                
                ANALYSIS REQUIREMENTS:
                - Decision: COMPLIANT, NON_COMPLIANT, CONDITIONAL, or NEEDS_REVIEW
                - Confidence Score: 0.0 to 1.0 (higher if document clearly states the answer)
                - Justification: Quote specific legal text and explain your reasoning
                - Relevant Regulations: List the exact legal sections that apply
                - Requirements: Any specific legal requirements or obligations mentioned
                - Violations: What would constitute non-compliance
                - Required Actions: Steps needed to achieve or maintain compliance
                
                Respond in valid JSON format.
                """,
                
                'hr_policy': """
                You are an expert HR policy analyst. Analyze the provided HR documents to answer employee-related queries.
                
                USER QUERY: {query}
                
                HR DOCUMENT SECTIONS:
                {context}
                
                INSTRUCTIONS:
                1. Read and understand the HR document sections provided
                2. Look for specific policies, procedures, and employee rights
                3. Pay attention to eligibility criteria, time limits, and benefits
                4. Consider both what is permitted AND what is explicitly prohibited
                5. Base your decision ONLY on the information provided in the HR documents
                
                ANALYSIS REQUIREMENTS:
                - Decision: APPROVED, DENIED, CONDITIONAL, or PENDING_REVIEW
                - Confidence Score: 0.0 to 1.0 (higher if document clearly states the answer)
                - Justification: Quote specific policy text and explain your reasoning
                - Relevant Policies: List the exact policy sections that apply
                - Eligibility: Any specific eligibility criteria or conditions
                - Benefits: What benefits or entitlements are available
                - Required Documentation: Documents needed to support the request
                
                Respond in valid JSON format.
                """,
                
                'contract_analysis': """
                You are an expert contract analyst. Analyze the provided contract documents to answer contract-related queries.
                
                USER QUERY: {query}
                
                CONTRACT DOCUMENT SECTIONS:
                {context}
                
                INSTRUCTIONS:
                1. Read and understand the contract document sections provided
                2. Look for specific terms, conditions, and contractual obligations
                3. Pay attention to deadlines, deliverables, and performance requirements
                4. Consider both what is required AND what is explicitly prohibited
                5. Base your decision ONLY on the information provided in the contract documents
                
                ANALYSIS REQUIREMENTS:
                - Decision: PERMITTED, PROHIBITED, CONDITIONAL, or NEEDS_CLARIFICATION
                - Confidence Score: 0.0 to 1.0 (higher if contract clearly states the answer)
                - Justification: Quote specific contract text and explain your reasoning
                - Relevant Clauses: List the exact contract sections that apply
                - Obligations: Any specific contractual obligations or requirements
                - Restrictions: What is explicitly prohibited or limited
                - Remedies: Available remedies or consequences for non-compliance
                
                Respond in valid JSON format.
                """,
                
                'claim_processing': """
                Analyze the claim processing requirements:
                
                Query: {query}
                
                Policy Information:
                {context}
                
                Please provide:
                1. Decision: APPROVED, DENIED, or PENDING
                2. Confidence Score: 0.0 to 1.0
                3. Justification: Detailed explanation
                4. Required Documents: List of needed documents
                5. Processing Time: Expected processing duration
                6. Steps: Claim processing steps
                7. Relevant Clauses: Policy clauses for claims
                
                Respond in JSON format.
                """,
                
                'policy_review': """
                Review the policy terms and conditions:
                
                Query: {query}
                
                Policy Content:
                {context}
                
                Please provide:
                1. Decision: CLEAR, UNCLEAR, or NEEDS_CLARIFICATION
                2. Confidence Score: 0.0 to 1.0
                3. Justification: Detailed explanation
                4. Relevant Clauses: Specific policy sections
                5. Key Points: Important policy points
                6. Recommendations: Suggested actions
                
                Respond in JSON format.
                """
            }
            
            # Decision mapping for different domains
            self.decision_mapping = {
                # Insurance domain
                'COVERED': 'approved',
                'NOT_COVERED': 'denied',
                'CONDITIONAL': 'denied',
                'APPROVED': 'approved',
                'REJECTED': 'denied',
                'DENIED': 'denied',
                'PENDING': 'denied',
                'PENDING_REVIEW': 'denied',
                'NOT_FOUND': 'denied',
                'UNCLEAR': 'denied',
                'NEEDS_REVIEW': 'denied',
                'ERROR': 'denied',
                'NO': 'denied',
                'YES': 'approved',
                # Legal compliance domain
                'COMPLIANT': 'approved',
                'NON_COMPLIANT': 'denied',
                # Contract domain
                'PERMITTED': 'approved',
                'PROHIBITED': 'denied',
                'NEEDS_CLARIFICATION': 'denied',
                # Policy review
                'CLEAR': 'approved',
                'UNCLEAR': 'denied'
            }
            
            logger.info("Reasoning patterns initialized")
            
        except Exception as e:
            logger.error(f"Error initializing reasoning patterns: {e}")
    
    def _initialize_clause_extraction(self):
        """Initialize clause extraction patterns"""
        try:
            # Patterns for extracting policy clauses
            self.clause_patterns = {
                'coverage_clause': [
                    r'coverage.*?shall.*?include',
                    r'covered.*?expenses.*?include',
                    r'benefits.*?shall.*?cover',
                    r'policy.*?covers.*?following'
                ],
                'exclusion_clause': [
                    r'exclusions.*?include',
                    r'not.*?covered.*?following',
                    r'excluded.*?from.*?coverage',
                    r'coverage.*?does.*?not.*?include'
                ],
                'condition_clause': [
                    r'conditions.*?precedent',
                    r'requirements.*?for.*?coverage',
                    r'must.*?meet.*?following',
                    r'coverage.*?subject.*?to'
                ],
                'amount_clause': [
                    r'maximum.*?benefit.*?\$[\d,]+',
                    r'coverage.*?limit.*?\$[\d,]+',
                    r'benefit.*?amount.*?\$[\d,]+',
                    r'up.*?to.*?\$[\d,]+'
                ],
                'waiting_period': [
                    r'waiting.*?period.*?\d+.*?(days?|weeks?|months?)',
                    r'coverage.*?begins.*?after.*?\d+',
                    r'benefits.*?available.*?after.*?\d+'
                ]
            }
            
            logger.info("Clause extraction patterns initialized")
            
        except Exception as e:
            logger.error(f"Error initializing clause extraction: {e}")
    
    def analyze_query(self, 
                     query: str, 
                     context: List[Dict[str, Any]], 
                     query_type: str = 'coverage_check') -> ReasoningResult:
        """Main method to analyze a query using LLM generation"""
        try:
            # FORCE DEBUG OUTPUT - ADD THIS AT THE VERY BEGINNING
            print("🔍 DEBUG: analyze_query method called!")
            print(f"🔍 DEBUG: Query: {query}")
            print(f"🔍 DEBUG: Context length: {len(context)}")
            print(f"🔍 DEBUG: LLM object: {self.llm}")
            
            total_start_time = time.time()
            step_times = {}
            
            logger.info("🧠 Starting LLM analysis...")
            logger.info(f"LLM object status: {self.llm is not None}")
            logger.info(f"LLM object: {self.llm}")
            
            # Step 1: Prepare context from relevant chunks
            step_start = time.time()
            logger.info("📝 Step 1: Formatting context for LLM...")
            formatted_context = self._format_context_for_llm(context)
            step_times['context_formatting'] = time.time() - step_start
            logger.info(f"✅ Context formatting completed in {step_times['context_formatting']:.2f}s")
            logger.info(f"Formatted context length: {len(formatted_context)} characters")
            
            # Step 2: Create comprehensive prompt
            step_start = time.time()
            logger.info("📋 Step 2: Creating comprehensive LLM prompt...")
            
            # Create a detailed prompt that asks for specific answers
            prompt = f"""Analyze this insurance policy query and provide a detailed answer.

QUERY: {query}

POLICY CONTEXT:
{formatted_context}

INSTRUCTIONS:
1. Provide a clear, direct answer to the query
2. Extract specific details like amounts, waiting periods, conditions
3. Explain the coverage decision (approved/denied/pending/not_found)
4. Include relevant policy clauses and conditions
5. Respond in JSON format with the following structure:
{{
    "decision": "approved/denied",
    "confidence_score": 0.0-1.0,
    "justification": "detailed explanation of the answer",
    "specific_details": "exact information requested",
    "amount": "coverage amount if applicable",
    "waiting_period": "waiting period if applicable",
    "conditions": ["list of conditions"],
    "exclusions": ["list of exclusions"],
    "relevant_clauses": ["relevant policy clauses"]
}}

IMPORTANT: 
- Quote specific policy text in your justification
- Reference exact clause IDs from the provided context
- If the policy explicitly states limitations, reflect them accurately
- Provide detailed reasoning, not just yes/no answers
- Consider both what is covered AND what is excluded
- Look for specific amounts, time periods, and conditions
- If the answer is not explicitly found, make the best inference based on the provided policy context. Do not answer 'pending', 'not_found', or 'unclear'. Always choose the most likely of 'approved' or 'denied' and explain your reasoning.

JSON Response:
"""
            
            step_times['prompt_creation'] = time.time() - step_start
            logger.info(f"✅ Prompt creation completed in {step_times['prompt_creation']:.2f}s")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            # Step 3: Generate response using LLM
            step_start = time.time()
            logger.info("🤖 Step 3: Generating LLM response...")
            
            # Check if LLM is available before calling
            if self.llm is None:
                logger.error("❌ LLM is None! Using fallback response")
                print("🔍 DEBUG: LLM is None - using fallback")
                response = self._generate_fallback_response(prompt)
            else:
                logger.info("✅ LLM is available, calling _generate_llm_response")
                print("🔍 DEBUG: LLM is available - calling generation")
                # DON'T CATCH ERRORS - let them propagate
            response = self._generate_llm_response(prompt)
            
            step_times['llm_generation'] = time.time() - step_start
            logger.info(f"✅ LLM generation completed in {step_times['llm_generation']:.2f}s")
            logger.info(f"Response received: {response[:200]}...")
            
            # Step 4: Parse the response
            step_start = time.time()
            logger.info("🔍 Step 4: Parsing LLM response...")
            parsed_result = self._parse_llm_response(response, query_type)
            step_times['response_parsing'] = time.time() - step_start
            logger.info(f"✅ Response parsing completed in {step_times['response_parsing']:.2f}s")
            
            # Step 5: Build final result
            step_start = time.time()
            logger.info("🏗️ Step 5: Building final result...")
            
            # Get the decision from parsed result, with proper fallback
            decision = parsed_result.get('decision', 'pending')
            if decision == 'not_found':
                # Keep not_found as not_found, don't change to pending
                final_decision = 'not_found'
            else:
                final_decision = decision
            
            result = ReasoningResult(
                decision=final_decision,  # Use the corrected decision
                confidence_score=parsed_result.get('confidence_score', 0.5),
                justification=parsed_result.get('justification', 'LLM analysis completed'),
                relevant_clauses=parsed_result.get('relevant_clauses', []),
                specific_details=parsed_result.get('specific_details', ''),
                amount=parsed_result.get('amount'),
                waiting_period=parsed_result.get('waiting_period'),
                conditions=parsed_result.get('conditions', []),
                exclusions=parsed_result.get('exclusions', []),
                required_documents=parsed_result.get('required_documents', []),
                processing_time=f"{time.time() - total_start_time:.2f}s",
                reasoning_steps=parsed_result.get('reasoning_steps', []),
                source_references=[],
                policy_quotes=parsed_result.get('policy_quotes', [])
            )
            step_times['result_building'] = time.time() - step_start
            logger.info(f"✅ Result building completed in {step_times['result_building']:.2f}s")
            
            total_time = time.time() - total_start_time
            
            # Log detailed LLM timing breakdown
            logger.info("🧠 LLM TIMING BREAKDOWN:")
            logger.info(f"   Context Formatting: {step_times['context_formatting']:.2f}s")
            logger.info(f"   Prompt Creation: {step_times['prompt_creation']:.2f}s")
            logger.info(f"   LLM Generation: {step_times['llm_generation']:.2f}s ⏱️")
            logger.info(f"   Response Parsing: {step_times['response_parsing']:.2f}s")
            logger.info(f"   Result Building: {step_times['result_building']:.2f}s")
            logger.info(f"   TOTAL LLM TIME: {total_time:.2f}s")
            
            logger.info(f"Query analysis completed: {result.decision} ({result.confidence_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"❌ DEBUG: Error analyzing query: {e}")
            print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
            # Don't return fallback - let the error propagate up
            raise e
    
    def _format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """Format context for LLM with smaller size limits"""
        try:
            formatted_parts = []
            total_chars = 0
            max_chars = 500  # REDUCED from 1000 to 500 for much faster processing
            
            for chunk in context:
                chunk_text = f"Section: {chunk.get('content', '')[:150]}"  # REDUCED from 200 to 150
                if total_chars + len(chunk_text) > max_chars:
                    break
                formatted_parts.append(chunk_text)
                total_chars += len(chunk_text)
            
            return "\n\n".join(formatted_parts)
        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            return "Context formatting error"

    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using the LLM with NO TIMEOUTS. Dynamically truncates prompt if needed to fit context window."""
        # --- SMART PROMPT PACKING ---
        system_prompt = (
            "You are an insurance policy analyst. Answer ONLY in one line of valid JSON, as detailed as possible, with decision, confidence_score, justification, and all key fields. Example: {\"decision\":\"approved\",\"confidence_score\":0.9,\"justification\":\"Policy covers X and Y.\",...}. Do NOT add any extra text."
        )
        timestamp = f"Timestamp: {time.time()}"
        def count_tokens(s):
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(enc.encode(s))
            except Exception:
                return max(1, int(len(s) / 4))
        max_total_tokens = 255  # Strict limit: prompt+output < n_ctx
        max_completion_tokens = 24
        # Assume prompt argument is: [User question]\n[Top document chunk]
        parts = prompt.split('\n', 1)
        user_query = parts[0] if parts else ''
        doc_chunk = parts[1] if len(parts) > 1 else ''
        # Build prompt in priority order: system, user, doc, timestamp
        prompt_sections = [system_prompt, user_query, doc_chunk, timestamp]
        packed_prompt = "\n".join([p for p in prompt_sections if p.strip()])
        # Truncate doc_chunk first, then user_query if needed
        while count_tokens(packed_prompt) + max_completion_tokens > max_total_tokens:
            if len(doc_chunk) > 32:
                doc_chunk = doc_chunk[:int(len(doc_chunk)*0.8)]
            elif len(user_query) > 16:
                user_query = user_query[:int(len(user_query)*0.8)]
            else:
                break
            prompt_sections = [system_prompt, user_query, doc_chunk, timestamp]
            packed_prompt = "\n".join([p for p in prompt_sections if p.strip()])
        # --- END SMART PACKING ---
        start_time = time.time()
        step_times = {}
        if self.llm is None:
            logger.warning("LLM not available, using fallback response")
            return self._generate_fallback_response(prompt)
        logger.info(f"LLM model loaded: {self.model_path}")
        logger.info(f"LLM object type: {type(self.llm)}")
        step_start = time.time()
        logger.info("🔧 Step 1: Preparing optimized system prompt...")
        step_times['prompt_prep'] = time.time() - step_start
        logger.info(f"✅ System prompt prepared in {step_times['prompt_prep']:.2f}s")
        
        # Step 2: Generate response with NO TIMEOUTS
        step_start = time.time()
        logger.info("🤖 Step 2: Starting LLM generation...")
        
        # FORCE FRESH GENERATION - DISABLE CACHE FOR DEBUGGING
        logger.info("🔄 FORCING FRESH LLM GENERATION (cache disabled for debugging)...")
        
        try:
            logger.info("🔄 Generating LLM response (NO TIMEOUT - will wait as long as needed)...")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"Max tokens: {self.max_tokens}")
            
            # Force a fresh generation by adding a unique timestamp
            unique_prompt = f"{system_prompt}\n\n{prompt}\n\nTimestamp: {time.time()}"
            
            # REMOVED ALL TIMEOUTS AND STOP CONDITIONS
            # Use ThreadPoolExecutor for multi-threading LLM inference
            from concurrent.futures import ThreadPoolExecutor
            # Final check: ensure packed_prompt fits context window
            total_tokens = count_tokens(packed_prompt) + max_completion_tokens
            print(f"DEBUG: Final packed_prompt token count: {count_tokens(packed_prompt)}, total with output: {total_tokens}")
            logger.info(f"Final prompt token count: {count_tokens(packed_prompt)}, total with output: {total_tokens}")
            # Fix: ensure total_tokens < max_total_tokens (strictly less than n_ctx)
            while total_tokens >= max_total_tokens:
                if len(doc_chunk) > 8:
                    doc_chunk = doc_chunk[:int(len(doc_chunk)*0.8)]
                elif len(user_query) > 8:
                    user_query = user_query[:int(len(user_query)*0.8)]
                elif max_completion_tokens > 4:
                    max_completion_tokens -= 1
                else:
                    # If we can't truncate further, stop and raise a clear error
                    raise ValueError(f"Cannot fit prompt and output into context window: prompt={count_tokens(packed_prompt)}, output={max_completion_tokens}, n_ctx={max_total_tokens+1}")
                prompt_sections = [system_prompt, user_query, doc_chunk, timestamp]
                packed_prompt = "\n".join([p for p in prompt_sections if p.strip()])
                total_tokens = count_tokens(packed_prompt) + max_completion_tokens
                print(f"⚠️ Emergency re-truncation: packed_prompt now {count_tokens(packed_prompt)} tokens, total: {total_tokens}, output: {max_completion_tokens}")
                logger.warning(f"Emergency re-truncation: packed_prompt now {count_tokens(packed_prompt)} tokens, total: {total_tokens}, output: {max_completion_tokens}")
            assert total_tokens < max_total_tokens, f"Prompt+output tokens ({total_tokens}) must be less than n_ctx ({max_total_tokens})!"
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(
                    self.llm.create_completion,
                    prompt=packed_prompt,
                    max_tokens=max_completion_tokens,
                    temperature=self.temperature
                )
                response = future.result()
            
            logger.info(f"Raw LLM response received: {response}")
                
            logger.info(f"Raw LLM response received: {response}")
            generation_time = time.time() - step_start
            logger.info(f"Generation time: {generation_time:.2f}s")
            # Step 3: Extract response text
            response_text = response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"❌ DEBUG: LLM generation error: {e}")
            print(f"❌ DEBUG: Error type: {type(e)}")
            raise e
        step_times['llm_completion'] = time.time() - step_start
        logger.info(f"✅ LLM completion finished in {step_times['llm_completion']:.2f}s")
        step_start = time.time()
        step_times['text_extraction'] = time.time() - step_start
        logger.info(f"Extracted response text: {response_text[:200]}...")
        return response_text

    def _parse_llm_response(self, response: str, query_type: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            response = response.strip()
            logger.debug(f"Raw LLM response: {response}")
            # Try to extract JSON from the response
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                # Clean up common LLM JSON issues
                import re
                # Remove trailing commas before closing brace
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                # Ensure braces are balanced
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if close_braces < open_braces:
                    json_str += '}' * (open_braces - close_braces)
                try:
                    parsed = json.loads(json_str)
                    logger.info(f"Successfully parsed LLM response: {parsed}")
                    # Fill in missing keys with sensible defaults
                    defaults = {
                        'decision': 'pending',
                        'confidence_score': 0.5,
                        'justification': 'No justification provided.',
                        'specific_details': '',
                        'amount': None,
                        'waiting_period': None,
                        'conditions': [],
                        'exclusions': [],
                        'relevant_clauses': [],
                        'required_documents': [],
                        'reasoning_steps': [],
                        'policy_quotes': []
                    }
                    for k, v in defaults.items():
                        if k not in parsed or parsed[k] is None:
                            parsed[k] = v
                    # Defensive: map decision to standard form
                    original_decision = str(parsed['decision']).upper()
                    mapped_decision = self.decision_mapping.get(original_decision, 'pending')
                    if mapped_decision != parsed['decision']:
                        logger.info(f"Mapping decision '{parsed['decision']}' -> '{mapped_decision}'")
                    parsed['decision'] = mapped_decision
                    return parsed
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Response text: {response}")
                    return self._fallback_parse_response(response)
            else:
                logger.error(f"No JSON found in response: {response}")
            return self._fallback_parse_response(response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_parse_response(response)

    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for malformed responses. Attempts to extract as much as possible."""
        try:
            response_lower = response.lower()
            # Decision extraction
            if 'not_found' in response_lower:
                decision = 'not_found'
            elif 'approved' in response_lower:
                decision = 'approved'
            elif 'denied' in response_lower:
                decision = 'denied'
            elif 'pending' in response_lower:
                decision = 'pending'
            else:
                decision = 'pending'
            # Confidence score extraction
            confidence = 0.5
            conf_match = re.search(r'confidence[_ ]?score["\s]*[:=][\s]*([0-9.]+)', response, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except Exception:
                    pass
            # Amount extraction
            amount = None
            amt_match = re.search(r'amount["\s]*[:=][\s]*([0-9,.]+)', response, re.IGNORECASE)
            if amt_match:
                try:
                    amount = float(amt_match.group(1).replace(',', ''))
                except Exception:
                    pass
            # Waiting period extraction
            waiting_period = None
            wp_match = re.search(r'waiting[_ ]?period["\s]*[:=][\s]*([\w\s]+)', response, re.IGNORECASE)
            if wp_match:
                waiting_period = wp_match.group(1).strip()
            # Justification extraction (try regex for JSON-like field, else fallback to first 500 chars)
            just_match = re.search(r'"justification"\s*:\s*"([^"]+)"', response)
            if just_match:
                justification = just_match.group(1)
            else:
                justification = response[:500] if response else 'Response parsing failed'
            # Specific details extraction (look for key or fallback)
            spec_match = re.search(r'specific[_ ]?details["\s]*[:=][\s]*([^"]+)', response, re.IGNORECASE)
            specific_details = spec_match.group(1).strip() if spec_match else ''
            # Fallback for lists
            def extract_list(key):
                match = re.search(rf'{key}["\s]*[:=][\s]*\[([^\]]*)\]', response, re.IGNORECASE)
                if match:
                    return [item.strip().strip('"\'') for item in match.group(1).split(',') if item.strip()]
                return []
            conditions = extract_list('conditions')
            exclusions = extract_list('exclusions')
            relevant_clauses = extract_list('relevant_clauses')
            required_documents = extract_list('required_documents')
            reasoning_steps = extract_list('reasoning_steps')
            policy_quotes = extract_list('policy_quotes')
            logger.warning(f"Fallback LLM response parsing used. Extracted decision: {decision}, confidence: {confidence}")
            return {
                'decision': decision,
                'confidence_score': confidence,
                'justification': justification,
                'specific_details': specific_details,
                'amount': amount,
                'waiting_period': waiting_period,
                'conditions': conditions,
                'exclusions': exclusions,
                'relevant_clauses': relevant_clauses,
                'required_documents': required_documents,
                'reasoning_steps': reasoning_steps,
                'policy_quotes': policy_quotes
            }
        except Exception as e:
            logger.error(f"Error in fallback parsing: {e}")
            return {
                'decision': 'pending',
                'confidence_score': 0.0,
                'justification': 'Failed to parse response',
                'specific_details': 'Parsing error occurred',
                'amount': None,
                'waiting_period': None,
                'conditions': [],
                'exclusions': [],
                'relevant_clauses': [],
                'required_documents': [],
                'reasoning_steps': [],
                'policy_quotes': []
            }
    
    def _extract_clause_references(self, 
                                  context: List[Dict[str, Any]], 
                                  parsed_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract specific clause references from context"""
        try:
            references = []
            
            for item in context:
                content = item.get('content', '')
                source = item.get('source_file', 'Unknown')
                
                # Extract clauses using patterns
                for clause_type, patterns in self.clause_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            references.append({
                                'clause_type': clause_type,
                                'clause_text': match,
                                'source_file': source,
                                'relevance_score': item.get('similarity_score', 0.0)
                            })
            
            return references
            
        except Exception as e:
            logger.error(f"Error extracting clause references: {e}")
            return []
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a more intelligent fallback response"""
        try:
            # Extract context from prompt to provide better fallback
            if "POLICY CONTEXT:" in prompt:
                context_start = prompt.find("POLICY CONTEXT:")
                context_end = prompt.find("INSTRUCTIONS:", context_start)
                if context_end == -1:
                    context_end = len(prompt)
                context = prompt[context_start:context_end]
            else:
                context = "Policy context not available"
            
            # Create a more informative fallback based on available context
            fallback_response = {
                "decision": "pending",
                "confidence_score": 0.3,
                "justification": "LLM analysis could not complete. Please check the policy document manually or try a simpler query.",
                "specific_details": "Analysis incomplete - manual review recommended",
                "amount": None,
                "waiting_period": None,
                "conditions": ["Manual review required"],
                "exclusions": [],
                "relevant_clauses": ["Full document review needed"],
                "policy_quotes": [f"Context available: {context[:200]}..."]
            }
            
            return json.dumps(fallback_response, indent=2)
                
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return json.dumps({
                "decision": "pending",
                "confidence_score": 0.0,
                "justification": "System error occurred during analysis",
                "specific_details": "Technical issue - please try again",
                "amount": None,
                "waiting_period": None,
                "conditions": [],
                "exclusions": [],
                "relevant_clauses": [],
                "policy_quotes": []
            })
    
    def _fast_rule_based_analysis(self, query: str, context: str) -> ReasoningResult:
        """Fast rule-based analysis without LLM"""
        try:
            query_lower = query.lower()
            context_lower = context.lower()
            
            # Initialize result
            decision = "pending"
            confidence_score = 0.5
            justification = "Analysis completed using rule-based system"
            specific_details = ""
            relevant_clauses = []
            conditions = []
            exclusions = []
            policy_quotes = []
            
            # Extract grace period information
            if 'grace period' in query_lower:
                grace_matches = re.findall(r'grace period.*?(\d+)\s*(days?|weeks?|months?)', context_lower, re.IGNORECASE)
                if grace_matches:
                    specific_details = f"Grace period: {grace_matches[0][0]} {grace_matches[0][1]}"
                    decision = "approved"
                    confidence_score = 0.9
                    justification = f"Policy specifies grace period of {grace_matches[0][0]} {grace_matches[0][1]} for premium payment."
                    policy_quotes.append(f"Grace period: {grace_matches[0][0]} {grace_matches[0][1]}")
                else:
                    # Look for grace period mentions without specific duration
                    if 'grace period' in context_lower:
                        specific_details = "Grace period mentioned but duration not specified"
                        decision = "approved"
                        confidence_score = 0.7
                        justification = "Policy mentions grace period but does not specify exact duration."
                        policy_quotes.append("Grace period mentioned in policy")
                    else:
                        specific_details = "Grace period information not found"
                        decision = "pending"
                        confidence_score = 0.3
                        justification = "Grace period details not found in policy sections."
            
            # Extract premium payment information
            elif 'premium' in query_lower and 'payment' in query_lower:
                premium_matches = re.findall(r'premium.*?payment.*?(\d+)\s*(days?|weeks?|months?)', context_lower, re.IGNORECASE)
                if premium_matches:
                    specific_details = f"Premium payment period: {premium_matches[0][0]} {premium_matches[0][1]}"
                    decision = "approved"
                    confidence_score = 0.8
                    justification = f"Policy specifies premium payment period of {premium_matches[0][0]} {premium_matches[0][1]}."
                    policy_quotes.append(f"Premium payment: {premium_matches[0][0]} {premium_matches[0][1]}")
            
            # Extract coverage amounts
            elif any(word in query_lower for word in ['amount', 'coverage', 'limit', 'sum']):
                amount_matches = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs?|rupees?|inr|\$)', context_lower, re.IGNORECASE)
                if amount_matches:
                    specific_details = f"Coverage amount: {amount_matches[0]}"
                    decision = "approved"
                    confidence_score = 0.8
                    justification = f"Policy specifies coverage amount of {amount_matches[0]}."
                    policy_quotes.append(f"Coverage amount: {amount_matches[0]}")
            
            # Extract waiting periods
            elif 'waiting period' in query_lower or 'waiting' in query_lower:
                waiting_matches = re.findall(r'waiting.*?period.*?(\d+)\s*(days?|weeks?|months?)', context_lower, re.IGNORECASE)
                if waiting_matches:
                    specific_details = f"Waiting period: {waiting_matches[0][0]} {waiting_matches[0][1]}"
                    decision = "approved"
                    confidence_score = 0.8
                    justification = f"Policy specifies waiting period of {waiting_matches[0][0]} {waiting_matches[0][1]}."
                    policy_quotes.append(f"Waiting period: {waiting_matches[0][0]} {waiting_matches[0][1]}")
            
            # Extract exclusions
            elif 'exclusion' in query_lower or 'not covered' in query_lower:
                exclusion_keywords = ['excluded', 'not covered', 'not permitted', 'not allowed']
                found_exclusions = []
                for keyword in exclusion_keywords:
                    if keyword in context_lower:
                        found_exclusions.append(keyword)
                
                if found_exclusions:
                    specific_details = f"Exclusions found: {', '.join(found_exclusions)}"
                    decision = "denied"
                    confidence_score = 0.7
                    justification = f"Policy contains exclusions: {', '.join(found_exclusions)}."
                    exclusions = found_exclusions
                    policy_quotes.append(f"Exclusions: {', '.join(found_exclusions)}")
            
            # General coverage check
            else:
                if any(word in context_lower for word in ['covered', 'coverage', 'benefits', 'permitted']):
                    decision = "approved"
                    confidence_score = 0.6
                    justification = "Policy indicates coverage is provided."
                    policy_quotes.append("Coverage mentioned in policy")
                elif any(word in context_lower for word in ['excluded', 'not covered', 'prohibited']):
                    decision = "denied"
                    confidence_score = 0.6
                    justification = "Policy indicates coverage is excluded."
                    policy_quotes.append("Exclusion mentioned in policy")
                else:
                    decision = "pending"
                    confidence_score = 0.4
                    justification = "Insufficient information to determine coverage."
            
            return ReasoningResult(
                decision=decision,
                confidence_score=confidence_score,
                justification=justification,
                relevant_clauses=relevant_clauses,
                specific_details=specific_details,
                conditions=conditions,
                exclusions=exclusions,
                required_documents=[],
                processing_time="fast",
                reasoning_steps=["Rule-based analysis"],
                source_references=[],
                policy_quotes=policy_quotes
            )
            
        except Exception as e:
            logger.error(f"Error in fast rule-based analysis: {e}")
            return self._create_fallback_result(query)
    
    def _create_fallback_result(self, query: str) -> ReasoningResult:
        """Create a fallback result when analysis fails"""
        try:
            # Try rule-based analysis as fallback
            logger.info("🔄 Attempting rule-based fallback analysis...")
            fallback_result = self._fast_rule_based_analysis(query, "")
            
            if fallback_result and fallback_result.confidence_score > 0.3:
                logger.info("✅ Fallback analysis successful")
                return fallback_result
            else:
                logger.warning("❌ Fallback analysis failed, using basic fallback")
                return ReasoningResult(
                    decision='pending',
                    confidence_score=0.0,
                    justification='Unable to analyze query due to technical issues. Please try again.',
                    relevant_clauses=[],
                    reasoning_steps=['Analysis failed - technical issue'],
                    source_references=[]
                )
        except Exception as e:
            logger.error(f"Error in fallback analysis: {e}")
        return ReasoningResult(
            decision='pending',
                confidence_score=0.0,
                justification='Unable to analyze query due to technical issues',
            relevant_clauses=[],
                reasoning_steps=['Analysis failed'],
                source_references=[]
            )
    
    def explain_decision(self, result: ReasoningResult) -> str:
        """Generate a human-readable explanation of the decision"""
        try:
            explanation_parts = []
            
            # Main decision
            explanation_parts.append(f"Decision: {result.decision.upper()}")
            explanation_parts.append(f"Confidence: {result.confidence_score:.1%}")
            
            # Justification
            if result.justification:
                explanation_parts.append(f"\nJustification:\n{result.justification}")
            
            # Relevant clauses
            if result.relevant_clauses:
                explanation_parts.append(f"\nRelevant Policy Clauses:")
                for clause in result.relevant_clauses:
                    explanation_parts.append(f"- {clause}")
            
            # Amount information
            if result.amount:
                explanation_parts.append(f"\nCoverage Amount: ${result.amount:,.2f}")
            
            # Waiting period
            if result.waiting_period:
                explanation_parts.append(f"\nWaiting Period: {result.waiting_period}")
            
            # Conditions
            if result.conditions:
                explanation_parts.append(f"\nConditions:")
                for condition in result.conditions:
                    explanation_parts.append(f"- {condition}")
            
            # Exclusions
            if result.exclusions:
                explanation_parts.append(f"\nExclusions:")
                for exclusion in result.exclusions:
                    explanation_parts.append(f"- {exclusion}")
            
            # Required documents
            if result.required_documents:
                explanation_parts.append(f"\nRequired Documents:")
                for doc in result.required_documents:
                    explanation_parts.append(f"- {doc}")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error explaining decision: {e}")
            return f"Decision: {result.decision.upper()}\nConfidence: {result.confidence_score:.1%}\nJustification: {result.justification}"
    
    def validate_decision(self, result: ReasoningResult) -> bool:
        """Validate the reasoning result for consistency"""
        try:
            # Check if confidence score is valid
            if not (0.0 <= result.confidence_score <= 1.0):
                return False
            
            # Check if decision is valid
            valid_decisions = ['approved', 'denied', 'pending']
            if result.decision not in valid_decisions:
                return False
            
            # Check if justification is provided
            if not result.justification or len(result.justification.strip()) < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating decision: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize reasoning engine
    reasoning_engine = AdvancedLLMReasoning()
    
    # Test query
    test_query = "Is heart surgery covered under this policy?"
    test_context = [
        {
            'content': 'This policy covers medical procedures including heart surgery up to $50,000.',
            'source_file': 'policy.pdf',
            'similarity_score': 0.9
        }
    ]
    
    # Analyze query
    result = reasoning_engine.analyze_query(test_query, test_context, 'coverage_check')
    
    print(f"Decision: {result.decision}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Justification: {result.justification}")
    
    # Explain decision
    explanation = reasoning_engine.explain_decision(result)
    print(f"\nExplanation:\n{explanation}") 