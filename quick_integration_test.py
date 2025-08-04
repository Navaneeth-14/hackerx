"""
Quick Interactive Test for Integrated System
Test the complete workflow with user input
"""

def quick_test():
    """Quick interactive test of the integrated system"""
    print("🚀 Quick Integration Test")
    print("="*40)
    print("Testing: Query Parser → Vector Database → LLM Reasoning")
    print("="*40)
    
    try:
        # Import components
        print("🔄 Loading components...")
        from query_parser import AdvancedQueryParser
        from vector_database import VectorDatabase
        from llm_reasoning import AdvancedLLMReasoning
        print("✅ Components loaded")
        
        # Initialize components
        print("🔄 Initializing...")
        query_parser = AdvancedQueryParser(use_gpu=False)
        vector_db = VectorDatabase(
            collection_name="quick_test",
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./quick_test_db"
        )
        
        # Try to initialize LLM reasoning
        try:
            reasoning_engine = AdvancedLLMReasoning(use_gpu=False)
            llm_available = True
            print("✅ LLM reasoning available")
        except Exception as e:
            print(f"⚠️  LLM reasoning not available: {e}")
            llm_available = False
        
        # Add sample documents
        print("🔄 Adding sample documents...")
        sample_docs = [
            {
                'content': 'Heart surgery is covered up to $50,000 with 90-day waiting period.',
                'metadata': {'source': 'policy.pdf', 'section': 'coverage'}
            },
            {
                'content': 'Dental treatment is covered up to $2,000 annually with 6-month waiting period.',
                'metadata': {'source': 'policy.pdf', 'section': 'dental'}
            },
            {
                'content': 'To file a claim, you need: claim form, medical certificate, receipts, and bills.',
                'metadata': {'source': 'claims.pdf', 'section': 'procedures'}
            }
        ]
        
        for doc in sample_docs:
            vector_db.add_document(doc['content'], doc['metadata'])
        print(f"✅ Added {len(sample_docs)} documents")
        
        # Interactive testing
        print("\n🎯 Interactive Testing")
        print("="*30)
        print("Enter your queries (type 'quit' to exit):")
        
        while True:
            try:
                query = input("\n❓ Your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print(f"\n🔄 Processing: {query}")
                print("-" * 40)
                
                # Step 1: Parse query
                print("📝 Step 1: Parsing query...")
                parsed = query_parser.parse_query(query)
                print(f"   Type: {parsed.query_type}")
                print(f"   Intent: {parsed.intent}")
                print(f"   Confidence: {parsed.confidence:.2f}")
                if parsed.entities:
                    print(f"   Entities: {list(parsed.entities.keys())}")
                
                # Step 2: Search vector database
                print("\n🔍 Step 2: Searching documents...")
                results = vector_db.search_documents(query, n_results=2, similarity_threshold=0.3)
                print(f"   Found {len(results)} relevant documents")
                
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Similarity: {result.get('similarity_score', 0):.2f}")
                    print(f"      Source: {result.get('source_file', 'Unknown')}")
                    print(f"      Content: {result.get('content', '')[:100]}...")
                
                # Step 3: LLM reasoning
                if llm_available and results:
                    print("\n🧠 Step 3: LLM reasoning...")
                    reasoning_result = reasoning_engine.analyze_query(
                        query=query,
                        context=results,
                        query_type=parsed.query_type
                    )
                    
                    print(f"   Decision: {reasoning_result.decision.upper()}")
                    print(f"   Confidence: {reasoning_result.confidence_score:.2f}")
                    print(f"   Justification: {reasoning_result.justification[:150]}...")
                    
                    if reasoning_result.amount:
                        print(f"   Amount: ${reasoning_result.amount:,.2f}")
                    if reasoning_result.waiting_period:
                        print(f"   Waiting Period: {reasoning_result.waiting_period}")
                    
                    # Show explanation
                    print(f"\n📋 Explanation:")
                    explanation = reasoning_engine.explain_decision(reasoning_result)
                    print(explanation)
                    
                else:
                    print("\n🧠 Step 3: LLM reasoning (not available)")
                    print("   Query parsing and document search completed successfully")
                
                print("\n" + "="*50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        import shutil
        if os.path.exists("./quick_test_db"):
            shutil.rmtree("./quick_test_db")
        print("✅ Cleanup completed")
        
        print("\n🎉 Quick test completed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

def test_specific_query(query_text):
    """Test a specific query"""
    print(f"🧪 Testing specific query: {query_text}")
    print("="*50)
    
    try:
        from query_parser import AdvancedQueryParser
        from vector_database import VectorDatabase
        from llm_reasoning import AdvancedLLMReasoning
        
        # Initialize
        query_parser = AdvancedQueryParser(use_gpu=False)
        vector_db = VectorDatabase(
            collection_name="specific_test",
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./specific_test_db"
        )
        
        # Add test document
        vector_db.add_document(
            "Heart surgery is covered up to $50,000 with 90-day waiting period.",
            {'source': 'test.pdf', 'type': 'coverage'}
        )
        
        # Process query
        parsed = query_parser.parse_query(query_text)
        results = vector_db.search_documents(query_text, n_results=1)
        
        print(f"Query Type: {parsed.query_type}")
        print(f"Confidence: {parsed.confidence:.2f}")
        print(f"Search Results: {len(results)}")
        
        if results:
            print(f"Best Match: {results[0].get('content', '')[:100]}...")
        
        # Try LLM reasoning
        try:
            reasoning_engine = AdvancedLLMReasoning(use_gpu=False)
            reasoning_result = reasoning_engine.analyze_query(
                query_text, results, parsed.query_type
            )
            print(f"LLM Decision: {reasoning_result.decision}")
            print(f"LLM Confidence: {reasoning_result.confidence_score:.2f}")
        except Exception as e:
            print(f"LLM Reasoning failed: {e}")
        
        # Cleanup
        import shutil
        if os.path.exists("./specific_test_db"):
            shutil.rmtree("./specific_test_db")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import os
    import sys
    
    # Check if specific query provided
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        test_specific_query(query)
    else:
        quick_test() 