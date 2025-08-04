#!/usr/bin/env python3
"""
Test Upload Endpoint
This script tests the file upload functionality
"""

import requests
import os

# Your ngrok URL
BASE_URL = "https://0468c638cef7.ngrok-free.app"

def test_upload():
    """Test file upload"""
    print("🧪 Testing File Upload...")
    
    # Check if we have a test file
    test_files = ["doc2.pdf", "main.py", "app.py"]
    test_file = None
    
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("❌ No test file found. Please ensure you have a PDF file in the directory.")
        return False
    
    print(f"📁 Using test file: {test_file}")
    
    url = f"{BASE_URL}/hackrx/upload"
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'application/pdf')}
            print(f"⏳ Uploading {test_file}...")
            response = requests.post(url, files=files, timeout=60)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Upload successful!")
            result = response.json()
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ Upload failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_first():
    """Test health endpoint first"""
    print("🧪 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run upload test"""
    print("🚀 Upload Test")
    print("=" * 40)
    
    # Test health first
    if not test_health_first():
        print("\n❌ Server not responding. Make sure Flask server is running!")
        print("Run: python app.py")
        return
    
    # Test upload
    test_upload()
    
    print("\n" + "=" * 40)
    print("📋 POSTMAN UPLOAD GUIDE")
    print("=" * 40)
    print("1. Method: POST")
    print(f"2. URL: {BASE_URL}/hackrx/upload")
    print("3. Body: form-data")
    print("4. Key: file (Type: File)")
    print("5. Value: Select your PDF file")
    print("=" * 40)

if __name__ == "__main__":
    main() 