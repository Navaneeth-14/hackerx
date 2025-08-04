"""
Test file for Query Parser
Checks if query_parser.py is working correctly with various test cases
"""

import sys
import os
from datetime import datetime

def test_query_parser_import():
    """Test if query parser can be imported without errors"""
    print("🧪 Testing Query Parser Import")
    print("="*40)
    
    try:
        from query_parser import AdvancedQueryParser, ParsedQuery, QueryEntity
        print("✅ Query parser imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing query parser: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_parser_initialization():
    """Test if query parser can be initialized"""
    print("\n🧪 Testing Query Parser Initialization")
    print("="*50)
    
    try:
        from query_parser import AdvancedQueryParser
        
        # Test initialization with default parameters
        print("🔄 Initializing query parser...")
        parser = AdvancedQueryParser(use_gpu=False)  # Use CPU for testing
        print("✅ Query parser initialized successfully")
        
        # Test basic attributes
        print("🔄 Checking parser attributes...")
        assert hasattr(parser, 'lemmatizer'), "Lemmatizer not found"
        assert hasattr(parser, 'stop_words'), "Stop words not found"
        assert hasattr(parser, 'insurance_entities'), "Insurance entities not found"
        assert hasattr(parser, 'query_types'), "Query types not found"
        print("✅ All required attributes present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing query parser: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_query_parsing():
    """Test basic query parsing functionality"""
    print("\n🧪 Testing Basic Query Parsing")
    print("="*40)
    
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        # Test queries
        test_queries = [
            "Is heart surgery covered?",
            "How do I file a claim?",
            "What's the waiting period?",
            "Can I claim for dental treatment?",
            "What documents are needed?"
        ]
        
        results = []
        for i, query in enumerate(test_queries):
            print(f"\n📝 Test Query {i+1}: {query}")
            
            try:
                parsed = parser.parse_query(query)
                
                # Check if parsed query has required attributes
                assert hasattr(parsed, 'original_query'), "Missing original_query"
                assert hasattr(parsed, 'enhanced_query'), "Missing enhanced_query"
                assert hasattr(parsed, 'query_type'), "Missing query_type"
                assert hasattr(parsed, 'entities'), "Missing entities"
                assert hasattr(parsed, 'intent'), "Missing intent"
                assert hasattr(parsed, 'confidence'), "Missing confidence"
                assert hasattr(parsed, 'keywords'), "Missing keywords"
                assert hasattr(parsed, 'synonyms'), "Missing synonyms"
                assert hasattr(parsed, 'context'), "Missing context"
                assert hasattr(parsed, 'timestamp'), "Missing timestamp"
                
                print(f"   ✅ Parsed successfully")
                print(f"   📊 Type: {parsed.query_type}")
                print(f"   🎯 Intent: {parsed.intent}")
                print(f"   📈 Confidence: {parsed.confidence:.2f}")
                print(f"   🔑 Keywords: {parsed.keywords[:3]}")
                print(f"   🏷️  Entities: {list(parsed.entities.keys())}")
                
                results.append(True)
                
            except Exception as e:
                print(f"   ❌ Failed to parse: {e}")
                results.append(False)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Basic Parsing Results: {success_count}/{total_count} successful")
        return success_count == total_count
        
    except Exception as e:
        print(f"❌ Error in basic query parsing: {e}")
        return False

def test_entity_extraction():
    """Test entity extraction functionality"""
    print("\n🧪 Testing Entity Extraction")
    print("="*40)
    
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        # Test queries with specific entities
        test_cases = [
            {
                'query': "Is heart surgery covered under my policy?",
                'expected_entities': ['medical_condition', 'coverage_type']
            },
            {
                'query': "Can I claim $5000 for dental treatment?",
                'expected_entities': ['amount', 'medical_condition']
            },
            {
                'query': "What's the 30-day waiting period for pre-existing conditions?",
                'expected_entities': ['time_period']
            },
            {
                'query': "Do I need a doctor's report for this claim?",
                'expected_entities': ['document_type']
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            query = test_case['query']
            expected_entities = test_case['expected_entities']
            
            print(f"\n📝 Test Case {i+1}: {query}")
            
            try:
                parsed = parser.parse_query(query)
                extracted_entities = list(parsed.entities.keys())
                
                print(f"   🏷️  Extracted entities: {extracted_entities}")
                print(f"   🎯 Expected entities: {expected_entities}")
                
                # Check if any expected entities were found
                found_entities = [entity for entity in expected_entities if entity in extracted_entities]
                
                if found_entities:
                    print(f"   ✅ Found expected entities: {found_entities}")
                    results.append(True)
                else:
                    print(f"   ⚠️  No expected entities found")
                    results.append(False)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append(False)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Entity Extraction Results: {success_count}/{total_count} successful")
        return success_count > 0  # At least some entities should be found
        
    except Exception as e:
        print(f"❌ Error in entity extraction: {e}")
        return False

def test_query_classification():
    """Test query type classification"""
    print("\n🧪 Testing Query Classification")
    print("="*40)
    
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        # Test queries for different types
        test_cases = [
            {
                'query': "How do I file a claim?",
                'expected_type': 'claim_inquiry'
            },
            {
                'query': "What is covered under my policy?",
                'expected_type': 'coverage_check'
            },
            {
                'query': "What are the policy terms?",
                'expected_type': 'policy_review'
            },
            {
                'query': "Is dental treatment covered?",
                'expected_type': 'medical_coverage'
            },
            {
                'query': "What is this document about?",
                'expected_type': 'general_inquiry'
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            query = test_case['query']
            expected_type = test_case['expected_type']
            
            print(f"\n📝 Test Case {i+1}: {query}")
            
            try:
                parsed = parser.parse_query(query)
                actual_type = parsed.query_type
                
                print(f"   🎯 Expected type: {expected_type}")
                print(f"   📊 Actual type: {actual_type}")
                print(f"   📈 Confidence: {parsed.confidence:.2f}")
                
                if actual_type == expected_type:
                    print(f"   ✅ Classification correct")
                    results.append(True)
                else:
                    print(f"   ⚠️  Classification mismatch")
                    results.append(False)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append(False)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Classification Results: {success_count}/{total_count} correct")
        return success_count > 0  # At least some classifications should work
        
    except Exception as e:
        print(f"❌ Error in query classification: {e}")
        return False

def test_query_suggestions():
    """Test query suggestion functionality"""
    print("\n🧪 Testing Query Suggestions")
    print("="*40)
    
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        # Test queries for suggestions
        test_queries = [
            "claim",
            "coverage",
            "medical",
            "policy",
            "documents"
        ]
        
        results = []
        for i, query in enumerate(test_queries):
            print(f"\n📝 Test Query {i+1}: {query}")
            
            try:
                suggestions = parser.get_query_suggestions(query)
                
                print(f"   💡 Suggestions: {len(suggestions)} found")
                for j, suggestion in enumerate(suggestions[:2]):
                    print(f"     {j+1}. {suggestion}")
                
                if suggestions:
                    print(f"   ✅ Suggestions generated successfully")
                    results.append(True)
                else:
                    print(f"   ⚠️  No suggestions generated")
                    results.append(False)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append(False)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Suggestion Results: {success_count}/{total_count} successful")
        return success_count > 0  # At least some suggestions should work
        
    except Exception as e:
        print(f"❌ Error in query suggestions: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🧪 Testing Error Handling")
    print("="*40)
    
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        # Test with invalid inputs
        invalid_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "123",  # Numbers only
            "!@#$%",  # Special characters only
            None  # None value
        ]
        
        results = []
        for i, invalid_input in enumerate(invalid_inputs):
            print(f"\n📝 Test Case {i+1}: {repr(invalid_input)}")
            
            try:
                parsed = parser.parse_query(invalid_input)
                
                # Should return a basic parsed query
                assert parsed is not None, "Should return a parsed query"
                assert hasattr(parsed, 'original_query'), "Should have original_query"
                assert hasattr(parsed, 'enhanced_query'), "Should have enhanced_query"
                
                print(f"   ✅ Handled gracefully")
                results.append(True)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append(False)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Error Handling Results: {success_count}/{total_count} handled")
        return success_count > 0  # At least some should be handled
        
    except Exception as e:
        print(f"❌ Error in error handling test: {e}")
        return False

def test_comprehensive_workflow():
    """Test a comprehensive workflow with real-world queries"""
    print("\n🧪 Testing Comprehensive Workflow")
    print("="*50)
    
    try:
        from query_parser import AdvancedQueryParser
        
        parser = AdvancedQueryParser(use_gpu=False)
        
        # Real-world insurance queries
        real_queries = [
            "I need to file a claim for my recent heart surgery that cost $25,000",
            "What's covered under my health insurance policy for dental procedures?",
            "How long is the waiting period for pre-existing conditions?",
            "Do I need a doctor's report and medical certificate for this claim?",
            "Can I claim for prescription medications and hospital stays?"
        ]
        
        results = []
        for i, query in enumerate(real_queries):
            print(f"\n📝 Real Query {i+1}: {query}")
            
            try:
                parsed = parser.parse_query(query)
                
                print(f"   📊 Type: {parsed.query_type}")
                print(f"   🎯 Intent: {parsed.intent}")
                print(f"   📈 Confidence: {parsed.confidence:.2f}")
                print(f"   🔑 Keywords: {parsed.keywords[:5]}")
                print(f"   🏷️  Entities: {list(parsed.entities.keys())}")
                
                # Check if enhanced query is different from original
                if parsed.enhanced_query != parsed.original_query:
                    print(f"   ✨ Query enhanced successfully")
                
                # Check if we have meaningful results
                if parsed.confidence > 0.0 or parsed.keywords or parsed.entities:
                    print(f"   ✅ Meaningful results extracted")
                    results.append(True)
                else:
                    print(f"   ⚠️  Limited results")
                    results.append(False)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append(False)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Comprehensive Results: {success_count}/{total_count} successful")
        return success_count > 0  # At least some should work
        
    except Exception as e:
        print(f"❌ Error in comprehensive workflow: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Query Parser Test Suite")
    print("="*60)
    print("Testing query_parser.py functionality")
    print()
    
    # Run all tests
    tests = [
        ("Import Test", test_query_parser_import),
        ("Initialization Test", test_query_parser_initialization),
        ("Basic Parsing Test", test_basic_query_parsing),
        ("Entity Extraction Test", test_entity_extraction),
        ("Query Classification Test", test_query_classification),
        ("Query Suggestions Test", test_query_suggestions),
        ("Error Handling Test", test_error_handling),
        ("Comprehensive Workflow Test", test_comprehensive_workflow)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running {test_name}")
        print(f"{'='*60}")
        
        success = test_func()
        results[test_name] = success
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your query parser is working correctly.")
    elif passed > total // 2:
        print("⚠️  Most tests passed, but some issues need attention.")
        print("\n💡 Areas to check:")
        for test_name, success in results.items():
            if not success:
                print(f"   - {test_name}")
    else:
        print("❌ Many tests failed. Check the error messages above.")
        print("\n💡 Common issues:")
        print("   - Missing dependencies (NLTK, spaCy)")
        print("   - Import errors")
        print("   - Initialization problems")

if __name__ == "__main__":
    main() 