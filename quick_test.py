#!/usr/bin/env python3
"""
Quick Test for app2.py Server
"""

import requests
import time

BASE_URL = "https://0468c638cef7.ngrok-free.app"

def test_server():
    print("🧪 Quick Server Test")
    print("=" * 40)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Server is running!")
        else:
            print(f"   ❌ Server error: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("   💡 Make sure to run: python app2.py")
        return False
    
    # Test 2: Simple Query
    print("\n2. Testing simple query...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_key_123"
        }
        payload = {
            "questions": ["What is the grace period for premium payment?"]
        }
        
        response = requests.post(f"{BASE_URL}/hackrx/run", 
                               json=payload, headers=headers, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Query processed successfully!")
            result = response.json()
            print(f"   Answer: {result.get('answers', [''])[0][:100]}...")
        else:
            print(f"   ❌ Query failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Query error: {e}")
    
    print("\n" + "=" * 40)
    print("📋 NEXT STEPS:")
    print("1. Make sure app2.py is running: python app2.py")
    print("2. Keep the server running in a separate terminal")
    print("3. Test your Postman requests")
    print("=" * 40)

if __name__ == "__main__":
    test_server() 