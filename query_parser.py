"""
Advanced Query Parser for Natural Language Processing
Handles vague, incomplete, and plain English queries with entity extraction
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# NLP and ML libraries
# import spacy  # Temporarily commented out due to installation issues
# from transformers import pipeline  # Temporarily commented out due to installation issues
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Optional imports with error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. NER functionality will be limited.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Additional NLTK data that might be needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass  # Ignore if punkt_tab is not available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParsedQuery:
    """Represents a parsed query with extracted information"""
    original_query: str
    enhanced_query: str
    query_type: str  # claim, coverage, policy, general, etc.
    entities: Dict[str, List[str]]
    intent: str
    confidence: float
    keywords: List[str]
    synonyms: List[str]
    context: Dict[str, Any]
    timestamp: datetime

@dataclass
class QueryEntity:
    """Represents an extracted entity from a query"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int

class AdvancedQueryParser:
    """Advanced query parser with entity extraction and query enhancement"""
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 use_gpu: bool = True,
                 max_chunks_to_consider: int = 5):  # Limit chunks for faster processing
        
        self.use_gpu = use_gpu
        self.spacy_model = spacy_model
        self.max_chunks_to_consider = max_chunks_to_consider
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Initialize entity extractors
        self._initialize_entity_extractors()
        
        # Initialize query enhancement
        self._initialize_query_enhancement()
        
        logger.info("Advanced Query Parser initialized")
    
    def _initialize_nlp_components(self):
        """Initialize NLP components"""
        try:
            # Load spaCy model
            # self.nlp = spacy.load(self.spacy_model)  # Temporarily commented out due to installation issues
            self.nlp = None  # Set to None temporarily
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Add custom stop words for insurance domain
            insurance_stop_words = {
                'policy', 'claim', 'coverage', 'insurance', 'document',
                'please', 'help', 'need', 'want', 'know', 'tell'
            }
            self.stop_words.update(insurance_stop_words)
            
            logger.info("NLP components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            raise
    
    def _initialize_entity_extractors(self):
        """Initialize entity extraction components"""
        try:
            # Initialize NER pipeline if transformers is available
            if TRANSFORMERS_AVAILABLE:
                try:
                    device = 0 if self.use_gpu else -1
                    self.ner_pipeline = pipeline(
                        "ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        device=device
                    )
                    logger.info("NER pipeline initialized")
                except Exception as e:
                    logger.warning(f"NER pipeline initialization failed: {e}")
                    self.ner_pipeline = None
            else:
                self.ner_pipeline = None
                logger.info("NER pipeline not available (transformers not installed)")
            
            # Insurance-specific entity patterns (always available)
            self.insurance_entities = {
                'medical_condition': [
                    r'\b(heart attack|stroke|cancer|diabetes|hypertension|asthma|arthritis)\b',
                    r'\b(surgery|operation|procedure|treatment|therapy)\b',
                    r'\b(medication|prescription|drug|medicine)\b'
                ],
                'coverage_type': [
                    r'\b(health|medical|dental|vision|life|auto|home|property)\s+(insurance|coverage|policy)\b',
                    r'\b(accident|disability|liability|comprehensive|collision)\b'
                ],
                'amount': [
                    r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                    r'\b\d+\s*(?:dollars?|rupees?|euros?)\b',
                    r'\b(?:maximum|minimum|total|sum)\s+(?:of\s+)?\$\d+\b'
                ],
                'time_period': [
                    r'\b(waiting period|grace period|coverage period|policy term)\b',
                    r'\b(\d+\s+(?:days?|weeks?|months?|years?))\b',
                    r'\b(immediate|urgent|emergency|routine)\b'
                ],
                'document_type': [
                    r'\b(claim form|medical certificate|prescription|bill|receipt|invoice)\b',
                    r'\b(doctor|physician|specialist|hospital|clinic)\s+(?:report|note|letter)\b'
                ]
            }
            
            logger.info("Entity extractors initialized")
            
        except Exception as e:
            logger.error(f"Error initializing entity extractors: {e}")
            # Set defaults if initialization fails
            self.ner_pipeline = None
            self.insurance_entities = {}
    
    def _initialize_query_enhancement(self):
        """Initialize query enhancement components"""
        try:
            # Query enhancement patterns
            self.enhancement_patterns = {
                'claim_related': {
                    'keywords': ['claim', 'file', 'submit', 'process', 'approve', 'reject'],
                    'synonyms': ['application', 'request', 'petition', 'appeal'],
                    'context': 'claim_processing'
                },
                'coverage_related': {
                    'keywords': ['cover', 'include', 'exclude', 'limit', 'maximum', 'minimum'],
                    'synonyms': ['protection', 'benefit', 'entitlement', 'eligibility'],
                    'context': 'coverage_analysis'
                },
                'policy_related': {
                    'keywords': ['policy', 'terms', 'conditions', 'clause', 'section'],
                    'synonyms': ['agreement', 'contract', 'document', 'provision'],
                    'context': 'policy_review'
                },
                'medical_related': {
                    'keywords': ['medical', 'health', 'treatment', 'surgery', 'medication'],
                    'synonyms': ['healthcare', 'therapeutic', 'clinical', 'pharmaceutical'],
                    'context': 'medical_coverage'
                }
            }
            
            # Query type classification patterns
            self.query_types = {
                'claim_inquiry': [
                    r'\b(how|what|can|is|does)\s+(?:to\s+)?(?:file|submit|process|claim)\b',
                    r'\b(claim|file|submit|process)\s+(?:a\s+)?(?:claim|request)\b',
                    # NEW: Claimable event + covered/payable/eligible/claim/benefit
                    r'\b(accidental death|death|hospitalization|hospitalisation|surgery|critical illness|disability|injury|fracture|burn|hospital cash|personal accident|permanent total disability|partial disability|loss of limb|loss of sight|cancer|cardiac arrest|stroke)\b.*\b(covered|payable|eligible|claim|benefit)\b',
                    r'\b(covered|payable|eligible|claim|benefit)\b.*\b(accidental death|death|hospitalization|hospitalisation|surgery|critical illness|disability|injury|fracture|burn|hospital cash|personal accident|permanent total disability|partial disability|loss of limb|loss of sight|cancer|cardiac arrest|stroke)\b',
                    # NEW: Questions about claim status, approval, rejection
                    r'\b(status|approved|rejected|pending)\s+(?:of\s+)?(?:my\s+)?(?:claim|request|application)\b',
                    r'\b(can|how|when|where|why|who)\s+(?:i\s+)?(?:claim|apply|request|file|submit)\b'
                ],
                'coverage_check': [
                    r'\b(cover|include|exclude|limit|maximum|minimum|entitled|eligible|applicable|available|provided)\b',
                    r'\b(is|does|can|will|are|was|were|has|have)\s+(?:.*?)\s+(?:cover|include|exclude|entitle|eligible|apply|available|provide)\b',
                    r'\b(waiting\s+period|grace\s+period|coverage\s+period|deductible|sub-limit|co-pay|sum insured|insured amount|policy limit|limit of liability)\b',
                    r'\b(what\'s|what\s+is|how much|how many|how long)\s+(?:the\s+)?(?:waiting|grace|coverage|deductible|sum insured|limit|co-pay|sub-limit)\b',
                    # NEW: Direct questions about coverage for events
                    r'\b(is|are|was|were|has|have|does|do|will|can)\s+(?:accidental death|hospitalization|surgery|critical illness|disability|injury|fracture|burn|hospital cash|personal accident|permanent total disability|partial disability|loss of limb|loss of sight|cancer|cardiac arrest|stroke)\s+(covered|included|excluded|eligible|payable|provided|available)\b',
                ],
                'policy_review': [
                    r'\b(policy|terms|conditions|clause|section|endorsement|schedule|benefit illustration|brochure|fine print|exclusion list|annexure)\b',
                    r'\b(what|which|where|show|list|explain|describe|find)\s+(?:in\s+)?(?:policy|document|contract|agreement|terms|conditions|clause|section|endorsement|schedule|annexure)\b',
                ],
                'medical_coverage': [
                    r'\b(medical|health|treatment|therapy|surgery|medication|prescription|consultation|diagnosis|investigation|test|screening|procedure|operation|rehabilitation|physiotherapy|ayush|unani|ayurveda|homeopathy|naturopathy|allopathy|alternative medicine)\b',
                    r'\b(doctor|hospital|clinic|physician|specialist|nurse|surgeon|medical facility|healthcare provider|practitioner|consultant)\b',
                    r'\b(dental|dental\s+procedures|dental\s+treatment|oral surgery|orthodontics|prosthodontics|periodontics)\b',
                    r'\b(heart\s+surgery|cardiac|surgical|angioplasty|bypass|stent|pacemaker)\b',
                ],
                'general_inquiry': [
                    r'\b(what|how|when|where|why|who|which|whom|whose)\b',
                    r'\b(explain|describe|tell|show|list|find|give|provide|details|information|clarify|help|assist|support)\b',
                    r'\b(premium|renewal|discount|no claim bonus|policy number|customer service|contact|helpline|agent|branch|portal|website|mobile app|grievance|complaint|feedback)\b',
                ]
            }
            
            logger.info("Query enhancement components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing query enhancement: {e}")
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Main method to parse and enhance a query"""
        try:
            # Clean and preprocess query
            cleaned_query = self._preprocess_query(query)
            
            # Extract entities
            entities = self._extract_entities(cleaned_query)
            
            # Determine query type and intent
            query_type, intent, confidence = self._classify_query(cleaned_query)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_query)
            
            # Generate synonyms
            synonyms = self._generate_synonyms(keywords)
            
            # Enhance query
            enhanced_query = self._enhance_query(cleaned_query, entities, query_type)
            
            # Build context
            context = self._build_context(cleaned_query, entities, query_type)
            
            parsed_query = ParsedQuery(
                original_query=query,
                enhanced_query=enhanced_query,
                query_type=query_type,
                entities=entities,
                intent=intent,
                confidence=confidence,
                keywords=keywords,
                synonyms=synonyms,
                context=context,
                timestamp=datetime.now()
            )
            
            logger.info(f"Query parsed successfully: {query_type} ({confidence:.2f})")
            return parsed_query
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Return a basic parsed query
            return self._create_basic_parsed_query(query)
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and clean the query"""
        try:
            # Convert to lowercase
            query = query.lower().strip()
            
            # Remove extra whitespace
            query = re.sub(r'\s+', ' ', query)
            
            # Remove special characters but keep important ones
            query = re.sub(r'[^\w\s\-\.\,\?\&]', '', query)
            
            # Fix common abbreviations
            query = self._fix_abbreviations(query)
            
            return query
            
        except Exception as e:
            logger.error(f"Error preprocessing query: {e}")
            return query
    
    def _fix_abbreviations(self, query: str) -> str:
        """Fix common abbreviations in insurance queries"""
        abbreviations = {
            'dr.': 'doctor',
            'doc.': 'document',
            'med.': 'medical',
            'rx': 'prescription',
            'hosp.': 'hospital',
            'clinic.': 'clinic',
            'ins.': 'insurance',
            'pol.': 'policy',
            'claim.': 'claim',
            'coverage.': 'coverage'
        }
        
        for abbr, full in abbreviations.items():
            query = query.replace(abbr, full)
        
        return query
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query"""
        entities = {}
        
        try:
            # Use spaCy for basic NER
            # doc = self.nlp(query)  # Temporarily commented out due to installation issues
            
            # Extract named entities
            # for ent in doc.ents:  # Temporarily commented out due to installation issues
            #     entity_type = ent.label_.lower()  # Temporarily commented out due to installation issues
            #     if entity_type not in entities:  # Temporarily commented out due to installation issues
            #         entities[entity_type] = []  # Temporarily commented out due to installation issues
            #     entities[entity_type].append(ent.text)  # Temporarily commented out due to installation issues
            
            # Extract insurance-specific entities using patterns
            for entity_type, patterns in self.insurance_entities.items():
                entities[entity_type] = []
                for pattern in patterns:
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    entities[entity_type].extend(matches)
            
            # Use BERT NER if available
            if hasattr(self, 'ner_pipeline') and self.ner_pipeline is not None:
                try:
                    ner_results = self.ner_pipeline(query)
                    for result in ner_results:
                        entity_type = result['entity'].lower()
                        if entity_type not in entities:
                            entities[entity_type] = []
                        entities[entity_type].append(result['word'])
                except Exception as e:
                    logger.debug(f"BERT NER failed: {e}")
            
            # Remove duplicates
            for entity_type in entities:
                entities[entity_type] = list(set(entities[entity_type]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}
    
    def _classify_query(self, query: str) -> Tuple[str, str, float]:
        """Classify the query type and determine intent (robust, prioritizes specific types)."""
        try:
            type_scores = {}
            type_first_match_idx = {}
            for query_type, patterns in self.query_types.items():
                matches = 0
                first_match_idx = None
                for i, pattern in enumerate(patterns):
                    if re.search(pattern, query, re.IGNORECASE):
                        matches += 1
                        if first_match_idx is None:
                            first_match_idx = i
                type_scores[query_type] = matches
                type_first_match_idx[query_type] = first_match_idx if first_match_idx is not None else float('inf')

            # Find types with the highest number of matches
            max_matches = max(type_scores.values())
            candidates = [t for t, m in type_scores.items() if m == max_matches and m > 0]

            # If only general_inquiry matches, use it; else prefer more specific types
            if len(candidates) == 0:
                best_type = 'general_inquiry'
            elif len(candidates) == 1:
                best_type = candidates[0]
            else:
                # Prefer non-general types if tied
                non_general = [t for t in candidates if t != 'general_inquiry']
                if non_general:
                    # Of non-general, pick the one with the earliest first match
                    best_type = min(non_general, key=lambda t: type_first_match_idx[t])
                else:
                    best_type = 'general_inquiry'

            # Confidence: proportion of patterns matched for the chosen type
            patterns = self.query_types[best_type]
            matches = type_scores[best_type]
            confidence = matches / len(patterns) if patterns else 0.0

            # Determine intent based on query type
            intent_mapping = {
                'claim_inquiry': 'claim_processing',
                'coverage_check': 'coverage_analysis',
                'policy_review': 'policy_review',
                'medical_coverage': 'medical_coverage',
                'general_inquiry': 'information_seeking'
            }
            intent = intent_mapping.get(best_type, 'information_seeking')
            return best_type, intent, confidence
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return 'general_inquiry', 'information_seeking', 0.0
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        try:
            # Tokenize with fallback
            try:
                tokens = word_tokenize(query)
            except Exception as tokenize_error:
                logger.warning(f"Word tokenization failed, using simple split: {tokenize_error}")
                tokens = query.split()
            
            # Remove stop words and lemmatize
            keywords = []
            for token in tokens:
                if token.lower() not in self.stop_words and len(token) > 2:
                    try:
                        lemmatized = self.lemmatizer.lemmatize(token.lower())
                        keywords.append(lemmatized)
                    except Exception as lemmatize_error:
                        logger.debug(f"Lemmatization failed for '{token}': {lemmatize_error}")
                        keywords.append(token.lower())
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _generate_synonyms(self, keywords: List[str]) -> List[str]:
        """Generate synonyms for keywords"""
        synonyms = []
        
        try:
            # Simple synonym mapping for insurance domain
            synonym_mapping = {
                'claim': ['application', 'request', 'petition'],
                'cover': ['include', 'protect', 'insure'],
                'policy': ['document', 'agreement', 'contract'],
                'medical': ['health', 'clinical', 'therapeutic'],
                'surgery': ['operation', 'procedure', 'treatment'],
                'hospital': ['clinic', 'medical center', 'facility'],
                'doctor': ['physician', 'specialist', 'medical practitioner'],
                'medicine': ['medication', 'drug', 'prescription'],
                'cost': ['expense', 'charge', 'fee', 'amount'],
                'limit': ['maximum', 'cap', 'ceiling', 'restriction']
            }
            
            for keyword in keywords:
                if keyword in synonym_mapping:
                    synonyms.extend(synonym_mapping[keyword])
            
            return list(set(synonyms))
            
        except Exception as e:
            logger.error(f"Error generating synonyms: {e}")
            return []
    
    def _enhance_query(self, query: str, entities: Dict[str, List[str]], query_type: str) -> str:
        """Enhance the query with additional context and synonyms"""
        try:
            enhanced_parts = [query]
            
            # Add entity context
            for entity_type, entity_list in entities.items():
                if entity_list:
                    enhanced_parts.append(f"related to {entity_type}: {', '.join(entity_list)}")
            
            # Add query type context
            if query_type in self.enhancement_patterns:
                pattern = self.enhancement_patterns[query_type]
                enhanced_parts.append(f"context: {pattern['context']}")
            
            # Add synonyms for important terms
            synonyms = self._generate_synonyms(self._extract_keywords(query))
            if synonyms:
                enhanced_parts.append(f"synonyms: {', '.join(synonyms[:5])}")
            
            return " | ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def _build_context(self, query: str, entities: Dict[str, List[str]], query_type: str) -> Dict[str, Any]:
        """Build context information for the query"""
        try:
            context = {
                'query_length': len(query),
                'has_entities': len(entities) > 0,
                'entity_types': list(entities.keys()),
                'query_type': query_type,
                'is_medical': any('medical' in entity_type for entity_type in entities.keys()),
                'has_amounts': 'amount' in entities,
                'has_time_periods': 'time_period' in entities
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return {}
    
    def _create_basic_parsed_query(self, query: str) -> ParsedQuery:
        """Create a basic parsed query when parsing fails"""
        return ParsedQuery(
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
    
    def get_query_suggestions(self, query: str) -> List[str]:
        """Generate query suggestions based on the input"""
        try:
            suggestions = []
            
            # Basic suggestions based on query type
            if 'claim' in query.lower():
                suggestions.extend([
                    "How do I file a claim?",
                    "What documents are needed for claim submission?",
                    "What is the claim processing time?",
                    "Can I track my claim status?"
                ])
            
            if 'cover' in query.lower() or 'coverage' in query.lower():
                suggestions.extend([
                    "What is covered under this policy?",
                    "What are the coverage limits?",
                    "Are pre-existing conditions covered?",
                    "What is not covered?"
                ])
            
            if 'medical' in query.lower() or 'health' in query.lower():
                suggestions.extend([
                    "What medical procedures are covered?",
                    "Are prescription drugs covered?",
                    "What is the coverage for hospital stays?",
                    "Are specialist consultations covered?"
                ])
            
            # Add general suggestions if none specific
            if not suggestions:
                suggestions.extend([
                    "What is covered under this policy?",
                    "How do I file a claim?",
                    "What are the policy terms and conditions?",
                    "What documents do I need?"
                ])
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []

# Example usage
if __name__ == "__main__":
    parser = AdvancedQueryParser()
    
    # Test queries
    test_queries = [
        "Is heart surgery covered?",
        "How do I file a claim?",
        "What's the waiting period?",
        "Can I claim for dental treatment?",
        "What documents are needed?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Original Query: {query}")
        
        parsed = parser.parse_query(query)
        
        print(f"Enhanced Query: {parsed.enhanced_query}")
        print(f"Query Type: {parsed.query_type}")
        print(f"Intent: {parsed.intent}")
        print(f"Confidence: {parsed.confidence:.2f}")
        print(f"Entities: {parsed.entities}")
        print(f"Keywords: {parsed.keywords}")
        
        suggestions = parser.get_query_suggestions(query)
        print(f"Suggestions: {suggestions[:2]}") 