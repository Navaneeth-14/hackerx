#!/usr/bin/env python3
"""
Test app2.py API Server
This script tests the new app2.py Flask API server
"""

import requests
import json
import time

# Your ngrok URL
BASE_URL = "https://0468c638cef7.ngrok-free.app"

def test_health():
    """Test health endpoint"""
    print("🧪 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n🧪 Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Root endpoint passed!")
            data = response.json()
            print(f"Message: {data.get('message')}")
            print(f"Version: {data.get('version')}")
            print(f"Endpoints: {len(data.get('endpoints', {}))} available")
        else:
            print(f"❌ Root endpoint failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_query():
    """Test simple query without document URL"""
    print("\n🧪 Testing Simple Query...")
    url = f"{BASE_URL}/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test_key_123"
    }
    payload = {
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Simple query passed!")
            result = response.json()
            print(f"Answers: {len(result.get('answers', []))} received")
            for i, answer in enumerate(result.get('answers', [])):
                print(f"  {i+1}. {answer[:100]}...")
        else:
            print(f"❌ Simple query failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_hackathon_format():
    """Test hackathon format with document URL"""
    print("\n🧪 Testing Hackathon Format...")
    url = f"{BASE_URL}/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test_key_123"
    }
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    try:
        print("⏳ Processing (this may take 30-60 seconds)...")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Hackathon format passed!")
            result = response.json()
            print(f"Answers: {len(result.get('answers', []))} received")
            for i, answer in enumerate(result.get('answers', [])):
                print(f"  {i+1}. {answer[:100]}...")
        else:
            print(f"❌ Hackathon format failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_query():
    """Test API query endpoint"""
    print("\n🧪 Testing API Query Endpoint...")
    url = f"{BASE_URL}/api/query"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "query": "What is the grace period for premium payment?"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API query passed!")
            result = response.json()
            print(f"Answer: {result.get('answer', '')[:100]}...")
            print(f"Confidence: {result.get('confidence', 0)}")
            print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ API query failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing app2.py API Server")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("\n❌ Health check failed! Make sure app2.py is running.")
        print("Run: python app2.py")
        return
    
    # Test root endpoint
    test_root()
    
    # Test simple query
    test_simple_query()
    
    # Test hackathon format
    test_hackathon_format()
    
    # Test API query
    test_api_query()
    
    print("\n" + "=" * 50)
    print("📋 APP2.PY TESTING GUIDE")
    print("=" * 50)
    print("1. Start the server:")
    print("   python app2.py")
    print()
    print("2. Health Check:")
    print(f"   GET {BASE_URL}/api/health")
    print()
    print("3. Root Endpoint:")
    print(f"   GET {BASE_URL}/")
    print()
    print("4. Simple Query:")
    print(f"   POST {BASE_URL}/hackrx/run")
    print("   Headers: Content-Type: application/json")
    print("   Headers: Authorization: Bearer test_key_123")
    print("   Body: {\"questions\": [\"Your question here\"]}")
    print()
    print("5. Hackathon Format:")
    print(f"   POST {BASE_URL}/hackrx/run")
    print("   Headers: Content-Type: application/json")
    print("   Headers: Authorization: Bearer test_key_123")
    print("   Body: {\"documents\": \"URL\", \"questions\": [\"Q1\", \"Q2\"]}")
    print()
    print("6. API Query:")
    print(f"   POST {BASE_URL}/api/query")
    print("   Headers: Content-Type: application/json")
    print("   Body: {\"query\": \"Your question here\"}")
    print()
    print("7. Upload Document:")
    print(f"   POST {BASE_URL}/hackrx/upload")
    print("   Body: form-data with 'file' field")
    print("=" * 50)

if __name__ == "__main__":
    main() 