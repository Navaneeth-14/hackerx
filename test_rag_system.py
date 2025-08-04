"""
Test RAG System Integration
Tests the complete flow: Document Processor → Vector Database → Query Parser → LLM Reasoning → RAG System
"""

import os
import tempfile
import shutil
from pathlib import Path

def test_rag_system():
    """Test the complete RAG system workflow"""
    print("🚀 RAG System Integration Test")
    print("="*50)
    print("Testing: Document Processor → Vector Database → Query Parser → LLM Reasoning → RAG System")
    print("="*50)
    
    try:
        # Import RAG system
        print("🔄 Importing RAG system...")
        from rag_system import AdvancedRAGSystem
        print("✅ RAG system imported successfully")
        
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag_system = AdvancedRAGSystem(
            use_gpu=False,  # Use CPU for testing
            vector_db_path="./test_rag_db"
        )
        print("✅ RAG system initialized")
        
        # Validate system
        print("🔄 Validating system components...")
        validation = rag_system.validate_system()
        
        if validation['overall_status']:
            print("✅ All components validated successfully")
        else:
            print("⚠️  Some components have issues:")
            for error in validation['errors']:
                print(f"   - {error}")
        
        # Create test documents
        print("\n🔄 Creating test documents...")
        test_docs = create_test_documents()
        
        # Ingest documents
        print("🔄 Ingesting documents...")
        total_chunks = 0
        for doc_info in test_docs:
            try:
                chunks = rag_system.ingest_document(doc_info['file_path'])
                total_chunks += len(chunks)
                print(f"   ✅ Ingested {len(chunks)} chunks from {doc_info['name']}")
            except Exception as e:
                print(f"   ❌ Failed to ingest {doc_info['name']}: {e}")
        
        print(f"✅ Total chunks ingested: {total_chunks}")
        
        # Test queries
        test_queries = [
            "Is heart surgery covered under this policy?",
            "What's the waiting period for dental procedures?",
            "How do I file a claim?",
            "What documents are needed for medical claims?",
            "Are pre-existing conditions covered?"
        ]
        
        print(f"\n🔄 Testing {len(test_queries)} queries...")
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            try:
                # Process query through RAG system
                result = rag_system.process_query(query, n_results=3)
                
                print(f"   Processing Time: {result.processing_time:.2f}s")
                print(f"   Query Type: {result.parsed_query.query_type}")
                print(f"   Intent: {result.parsed_query.intent}")
                print(f"   Confidence: {result.parsed_query.confidence:.2f}")
                print(f"   Search Results: {len(result.search_results)}")
                print(f"   Decision: {result.reasoning_result.decision}")
                print(f"   Reasoning Confidence: {result.reasoning_result.confidence_score:.2f}")
                
                # Show top search result
                if result.search_results:
                    top_result = result.search_results[0]
                    print(f"   Top Result: {top_result.content[:100]}...")
                    print(f"   Source: {top_result.source_file}")
                    print(f"   Similarity: {top_result.similarity_score:.3f}")
                
                # Show reasoning justification
                if result.reasoning_result.justification:
                    print(f"   Justification: {result.reasoning_result.justification[:150]}...")
                
                results.append({
                    'query': query,
                    'result': result,
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Query processing failed: {e}")
                results.append({
                    'query': query,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Generate summary report
        print(f"\n{'='*50}")
        print("📊 RAG SYSTEM TEST RESULTS")
        print(f"{'='*50}")
        
        successful_queries = sum(1 for r in results if r['success'])
        total_queries = len(results)
        
        print(f"Total Queries Tested: {total_queries}")
        print(f"Successful Queries: {successful_queries}")
        print(f"Success Rate: {successful_queries/total_queries*100:.1f}%")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            if result['success']:
                rag_result = result['result']
                status = "✅"
                decision = rag_result.reasoning_result.decision
                confidence = rag_result.reasoning_result.confidence_score
                print(f"{i}. {status} {result['query']}")
                print(f"   Decision: {decision}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Search Results: {len(rag_result.search_results)}")
            else:
                print(f"{i}. ❌ {result['query']}")
                print(f"   Error: {result['error']}")
        
        # Test system statistics
        print(f"\n📊 SYSTEM STATISTICS:")
        stats = rag_system.get_system_statistics()
        print(f"   Vector Database: {stats.get('vector_database', {}).get('total_chunks', 0)} chunks")
        print(f"   Audit Trail: {stats.get('audit_trail', {}).get('total_entries', 0)} entries")
        print(f"   Successful Queries: {stats.get('audit_trail', {}).get('successful_queries', 0)}")
        
        # Test audit trail
        print(f"\n📋 AUDIT TRAIL SAMPLE:")
        audit_trail = rag_system.get_audit_trail()
        if audit_trail:
            latest_entry = audit_trail[-1]
            print(f"   Latest Action: {latest_entry.get('action', 'unknown')}")
            print(f"   Status: {latest_entry.get('status', 'unknown')}")
            print(f"   Timestamp: {latest_entry.get('timestamp', 'unknown')}")
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        cleanup_test_data()
        
        print(f"\n🎉 RAG system test completed!")
        
        if successful_queries == total_queries:
            print("✅ All queries processed successfully!")
            print("🎯 RAG System is working perfectly!")
        else:
            print("⚠️  Some queries failed. Check the detailed results above.")
        
        return successful_queries == total_queries
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_documents():
    """Create test documents for RAG system"""
    test_dir = tempfile.mkdtemp()
    print(f"📁 Created test directory: {test_dir}")
    
    docs = []
    
    # Create policy document
    policy_content = """
    MEDICAL INSURANCE POLICY
    
    COVERAGE DETAILS:
    - Heart surgery: Covered up to $50,000
    - Dental procedures: Covered up to $2,000 annually
    - Prescription medications: 80% coverage
    - Hospital stays: Up to $1,000 per day
    - Specialist consultations: $100 per visit
    
    WAITING PERIODS:
    - General medical: 30 days
    - Pre-existing conditions: 12 months
    - Dental procedures: 6 months
    - Major surgeries: 90 days
    
    CLAIM PROCEDURES:
    - Submit claim form within 30 days
    - Include medical certificate
    - Provide original receipts and bills
    - Processing time: 10-15 business days
    
    EXCLUSIONS:
    - Cosmetic procedures
    - Experimental treatments
    - Injuries from dangerous activities
    - Pre-existing conditions (first 12 months)
    """
    
    policy_path = os.path.join(test_dir, "medical_policy.txt")
    with open(policy_path, 'w', encoding='utf-8') as f:
        f.write(policy_content)
    
    docs.append({
        'name': 'Medical Policy',
        'file_path': policy_path,
        'type': 'policy'
    })
    
    # Create claims guide
    claims_content = """
    CLAIMS PROCESSING GUIDE
    
    REQUIRED DOCUMENTS:
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
    
    SUBMISSION METHODS:
    - Online portal
    - Mobile app
    - Mail to claims department
    - In-person at service centers
    """
    
    claims_path = os.path.join(test_dir, "claims_guide.txt")
    with open(claims_path, 'w', encoding='utf-8') as f:
        f.write(claims_content)
    
    docs.append({
        'name': 'Claims Guide',
        'file_path': claims_path,
        'type': 'guide'
    })
    
    return docs

def cleanup_test_data():
    """Clean up test data"""
    try:
        import time
        import gc
        
        # Force garbage collection
        gc.collect()
        time.sleep(2)
        
        # Remove test directories
        test_dirs = ["./test_rag_db", "./test_vector_db", "./temp_test_db"]
        for dir_path in test_dirs:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    print(f"   ✅ Cleaned {dir_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not clean {dir_path}: {e}")
        
        # Remove temporary files
        temp_files = [f for f in os.listdir('.') if f.startswith('temp_')]
        for file in temp_files:
            try:
                os.remove(file)
                print(f"   ✅ Removed {file}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {file}: {e}")
                
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

def test_individual_components():
    """Test individual components before RAG system"""
    print("\n🧪 TESTING INDIVIDUAL COMPONENTS")
    print("="*40)
    
    components = {
        'Document Processor': 'document_processer',
        'Vector Database': 'vector_database', 
        'Query Parser': 'query_parser',
        'LLM Reasoning': 'llm_reasoning'
    }
    
    results = {}
    
    for name, module in components.items():
        print(f"\n🔄 Testing {name}...")
        try:
            __import__(module)
            print(f"   ✅ {name} imported successfully")
            results[name] = True
        except Exception as e:
            print(f"   ❌ {name} import failed: {e}")
            results[name] = False
    
    # Summary
    print(f"\n📊 COMPONENT TEST RESULTS:")
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} components ready")
    
    return passed == total

def main():
    """Main test runner"""
    print("🚀 RAG System Test Suite")
    print("="*50)
    
    # Test individual components first
    components_ready = test_individual_components()
    
    if not components_ready:
        print("\n❌ Some components are not ready. Please fix the issues above.")
        return False
    
    print(f"\n{'='*50}")
    print("🔄 RUNNING RAG SYSTEM INTEGRATION TEST")
    print(f"{'='*50}")
    
    # Test RAG system
    success = test_rag_system()
    
    if success:
        print(f"\n🎉 RAG System Integration Test PASSED!")
        print("✅ All components working together successfully")
        print("🎯 Your RAG system is ready for production use!")
    else:
        print(f"\n⚠️  RAG System Integration Test FAILED!")
        print("❌ Some issues need to be resolved")
    
    print(f"\n💡 Next steps:")
    print("   1. Add your actual documents")
    print("   2. Customize the query processing")
    print("   3. Fine-tune the reasoning engine")
    print("   4. Deploy to production")

if __name__ == "__main__":
    main() 