#!/usr/bin/env python3
"""
Test script to verify answer quality - should provide actual answers, not just locations
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

def test_answer_quality():
    """Test that the system provides actual answers instead of location info"""
    logger.info("🧪 Testing answer quality...")
    
    # Test queries that should get specific answers
    test_queries = [
        "What is the coverage amount for medical expenses?",
        "What are the waiting periods for this policy?",
        "What documents do I need to submit for a claim?",
        "What are the exclusions in this policy?",
        "How much is covered for hospitalization?",
        "What is the grace period for premium payment?",
        "What conditions apply to newborn coverage?",
        "What is the maximum coverage limit?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"📝 Test {i}: {query}")
        
        try:
            start_time = time.time()
            
            # Make request
            response = requests.post(
                f"{BASE_URL}/api/query",
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                answer = result_data.get('answer', '')
                decision = result_data.get('decision', 'unknown')
                confidence = result_data.get('confidence', 0.0)
                
                # Analyze answer quality
                quality_score = analyze_answer_quality(answer, query)
                
                logger.info(f"✅ Test {i} COMPLETED - {processing_time:.2f}s")
                logger.info(f"   Decision: {decision}")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Quality Score: {quality_score}/10")
                logger.info(f"   Answer Preview: {answer[:100]}...")
                
                results.append({
                    'test': i,
                    'query': query,
                    'status': 'COMPLETED',
                    'processing_time': processing_time,
                    'decision': decision,
                    'confidence': confidence,
                    'quality_score': quality_score,
                    'answer_preview': answer[:200],
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
                    'quality_score': 0,
                    'answer_preview': None,
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
                'quality_score': 0,
                'answer_preview': None,
                'error': str(e)
            })
        
        # Small delay between tests
        time.sleep(2)
    
    # Print summary
    logger.info("\n📊 ANSWER QUALITY TEST SUMMARY:")
    logger.info("=" * 50)
    
    completed = sum(1 for r in results if r['status'] == 'COMPLETED')
    failed = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR'])
    
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    
    if completed > 0:
        avg_quality = sum(r['quality_score'] for r in results if r['status'] == 'COMPLETED') / completed
        avg_time = sum(r['processing_time'] for r in results if r['status'] == 'COMPLETED') / completed
        logger.info(f"Average Quality Score: {avg_quality:.1f}/10")
        logger.info(f"Average Processing Time: {avg_time:.2f}s")
        
        # Quality assessment
        if avg_quality >= 8:
            logger.info("🎉 EXCELLENT: High quality answers provided")
        elif avg_quality >= 6:
            logger.info("✅ GOOD: Decent quality answers provided")
        elif avg_quality >= 4:
            logger.info("⚠️ FAIR: Some improvement needed")
        else:
            logger.warning("❌ POOR: Significant improvement needed")
    
    return results

def analyze_answer_quality(answer, query):
    """Analyze the quality of an answer (0-10 scale)"""
    if not answer:
        return 0
    
    score = 0
    
    # Check if answer contains actual information vs just location
    location_indicators = ['section', 'page', 'document', 'file', 'location', 'found in']
    has_location_only = any(indicator in answer.lower() for indicator in location_indicators)
    
    if has_location_only and len(answer.split()) < 20:
        score += 1  # Very low score for location-only answers
    else:
        score += 5  # Base score for non-location answers
    
    # Check for specific details
    if any(word in answer.lower() for word in ['amount', 'coverage', 'limit', 'period', 'condition']):
        score += 2
    
    # Check for numbers (specific amounts, dates, etc.)
    import re
    if re.search(r'\d+', answer):
        score += 1
    
    # Check for policy-specific terms
    policy_terms = ['covered', 'excluded', 'approved', 'denied', 'waiting', 'grace', 'premium']
    if any(term in answer.lower() for term in policy_terms):
        score += 1
    
    # Check for detailed explanation
    if len(answer.split()) > 30:
        score += 1
    
    return min(score, 10)  # Cap at 10

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
    logger.info("🚀 Starting Answer Quality Test")
    logger.info("=" * 50)
    
    # Check if API is running
    if not test_health_check():
        logger.error("❌ API is not running. Please start the Flask app first.")
        return
    
    # Run quality tests
    results = test_answer_quality()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"answer_quality_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total': len(results),
                'completed': sum(1 for r in results if r['status'] == 'COMPLETED'),
                'failed': sum(1 for r in results if r['status'] in ['FAILED', 'ERROR']),
                'avg_quality': sum(r['quality_score'] for r in results if r['status'] == 'COMPLETED') / max(1, sum(1 for r in results if r['status'] == 'COMPLETED'))
            }
        }, f, indent=2)
    
    logger.info(f"📄 Results saved to: {results_file}")
    logger.info("🏁 Answer quality test completed!")

if __name__ == "__main__":
    main() 