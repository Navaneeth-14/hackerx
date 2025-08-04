#!/usr/bin/env python3
"""
Simple test script to verify the RAG system works without timeout restrictions
"""

import requests
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://0.0.0.0:5000"

def test_query_processing():
    """Test query processing without timeout restrictions"""
    logger.info("🧪 Testing query processing without timeouts...")
    
    # Test queries
    test_queries = [
        "What is the coverage amount for medical expenses?",
        "What are the waiting periods for this policy?",
        "What documents do I need to submit for a claim?",
        "What are the exclusions in this policy?",
        "How much is covered for hospitalization?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"📝 Test {i}: {query}")
        
        try:
            start_time = time.time()
            
            # Make request without timeout restrictions
            response = requests.post(
                f"{BASE_URL}/api/query",
                json={'query': query},
                headers={'Content-Type': 'application/json'}
                # No timeout parameter - let it run as long as needed
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info(f"✅ Test {i} PASSED - Completed in {processing_time:.2f}s")
                logger.info(f"   Decision: {result_data.get('decision', 'unknown')}")
                logger.info(f"   Confidence: {result_data.get('confidence', 0.0):.2f}")
                logger.info(f"   Processing Time: {result_data.get('processing_time', 'unknown')}")
                
                results.append({
                    'test': i,
                    'query': query,
                    'status': 'PASSED',
                    'processing_time': processing_time,
                    'decision': result_data.get('decision'),
                    'confidence': result_data.get('confidence'),
                    'error': None
                })
                
            else:
                logger.error(f"❌ Test {i} FAILED - Status {response.status_code}")
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Unknown error')
                except:
                    error_msg = response.text
                
                results.append({
                    'test': i,
                    'query': query,
                    'status': 'FAILED',
                    'processing_time': processing_time,
                    'decision': None,
                    'confidence': None,
                    'error': error_msg
                })
                
        except Exception as e:
            logger.error(f"❌ Test {i} ERROR - {e}")
            results.append({
                'test': i,
                'query': query,
                'status': 'ERROR',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
                'decision': None,
                'confidence': None,
                'error': str(e)
            })
        
        # Small delay between tests
        time.sleep(2)
    
    # Print summary
    logger.info("\n📊 TEST SUMMARY:")
    logger.info("=" * 40)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR'])
    
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if passed > 0:
        avg_time = sum(r['processing_time'] for r in results if r['status'] == 'PASSED') / passed
        logger.info(f"Average Processing Time: {avg_time:.2f}s")
    
    if passed > 0:
        logger.info("✅ SYSTEM: Working - Queries completed successfully")
    else:
        logger.error("❌ SYSTEM: Not working - No queries completed successfully")
    
    return results

def test_health_check():
    """Test if the API is running"""
    logger.info("🏥 Testing API health...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            logger.info("✅ API is healthy and running")
            return True
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API health check error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting No-Timeout Test")
    logger.info("=" * 40)
    
    # Check if API is running
    if not test_health_check():
        logger.error("❌ API is not running. Please start the Flask app first.")
        return
    
    # Run tests
    results = test_query_processing()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"no_timeout_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results if r['status'] == 'PASSED'),
                'failed': sum(1 for r in results if r['status'] in ['FAILED', 'ERROR'])
            }
        }, f, indent=2)
    
    logger.info(f"📄 Results saved to: {results_file}")
    logger.info("🏁 No-timeout test completed!")

if __name__ == "__main__":
    main() 