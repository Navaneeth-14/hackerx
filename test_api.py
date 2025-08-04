#!/usr/bin/env python3
"""
Test script for the Flask API server
Demonstrates how to use the various endpoints
"""

import requests
import json
import os

# API Configuration
BASE_URL = "http://127.0.0.1:5000"
HEADERS = {
    "Content-Type": "application/json"
}

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_system_status():
    """Test the system status endpoint"""
    print("\n📊 Testing system status...")
    try:
        response = requests.get(f"{BASE_URL}/hackrx/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ System status failed: {e}")
        return False

def test_query_processing(questions):
    """Test query processing"""
    print(f"\n🤔 Testing query processing...")
    
    payload = {
        "questions": questions
    }
    
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", 
                               json=payload, 
                               headers=HEADERS)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        return False

def test_upload_document(file_path):
    """Test document upload"""
    print(f"\n📄 Testing document upload: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            
            response = requests.post(f"{BASE_URL}/hackrx/upload", 
                                  files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def test_query_processing(questions):
    """Test query processing"""
    print(f"\n🤔 Testing query processing...")
    
    payload = {
        "questions": questions
    }
    
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", 
                               json=payload, 
                               headers=HEADERS)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        return False



def main():
    """Main test function"""
    print("🚀 Starting API Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Health check failed. Make sure the server is running.")
        return
    
    # Test 2: System status
    test_system_status()
    

    
    # Test 4: Document upload (if file exists)
    test_files = ["doc2.pdf", "test_document.txt"]
    for test_file in test_files:
        if os.path.exists(test_file):
            test_upload_document(test_file)
            break
    
    # Test 3: Query processing
    test_questions = [
        "What is covered under this policy?",
        "What is the maximum coverage amount?",
        "What documents are required for claims?"
    ]
    test_query_processing(test_questions)
    
    print("\n✅ API tests completed!")

if __name__ == "__main__":
    main() 