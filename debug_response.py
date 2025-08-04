#!/usr/bin/env python3
"""
Debug script to test current response and see what's happening
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://0.0.0.0:5000"

def test_single_query():
    """Test a single query and show detailed response"""
    query = "What is the coverage amount for medical expenses?"
    
    logger.info(f"🔍 Testing query: {query}")
    
    try:
        # Make request
        response = requests.post(
            f"{BASE_URL}/api/query",
            json={'query': query},
            headers={'Content-Type': 'application/json'}
        )
        
        logger.info(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result_data = response.json()
            
            logger.info("📋 FULL RESPONSE:")
            logger.info(json.dumps(result_data, indent=2))
            
            # Analyze the response
            answer = result_data.get('answer', '')
            decision = result_data.get('decision', 'unknown')
            confidence = result_data.get('confidence', 0.0)
            
            logger.info("\n🔍 ANALYSIS:")
            logger.info(f"Answer: {answer}")
            logger.info(f"Decision: {decision}")
            logger.info(f"Confidence: {confidence}")
            
            # Check if answer is meaningful
            if "LLM analysis completed" in answer or "analysis completed" in answer.lower():
                logger.warning("❌ PROBLEM: Answer is just a completion message, not actual information!")
            elif len(answer.split()) < 10:
                logger.warning("⚠️ WARNING: Answer seems too short")
            else:
                logger.info("✅ Answer appears to contain actual information")
                
        else:
            logger.error(f"❌ Request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def test_health():
    """Test API health"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            logger.info("✅ API is healthy")
            return True
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API health check error: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Debug Test")
    logger.info("=" * 40)
    
    # Check health first
    if not test_health():
        logger.error("❌ API is not running. Please start the Flask app first.")
        return
    
    # Test single query
    test_single_query()
    
    logger.info("\n🏁 Debug test completed!")

if __name__ == "__main__":
    main() 