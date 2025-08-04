#!/usr/bin/env python3
"""
Test script for hackrx/run endpoint
"""

import requests
import json

# Your ngrok URL
BASE_URL = "https://c1c8ea4c476e.ngrok-free.app"

def test_hackrx_run():
    """Test the hackrx/run endpoint"""
    print("🧪 Testing hackrx/run endpoint...")
    
    url = f"{BASE_URL}/hackrx/run"
    payload = {
        "questions": [
            "What is covered under this policy?",
            "What is the maximum coverage amount?"
        ]
    }
    
    try:
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            url, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ hackrx/run endpoint is working!")
            return True
        else:
            print("❌ hackrx/run endpoint failed")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - server might not be running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Health Status: {response.status_code}")
        print(f"Health Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n🏠 Testing root endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Root Status: {response.status_code}")
        print(f"Root Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing HackRX Endpoints")
    print("=" * 50)
    
    # Test basic endpoints first
    health_ok = test_health()
    root_ok = test_root()
    
    if not health_ok:
        print("❌ Server is not responding. Please restart the Flask server.")
        return
    
    # Test the main hackrx/run endpoint
    hackrx_ok = test_hackrx_run()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Root Endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
    print(f"HackRX Run: {'✅ PASS' if hackrx_ok else '❌ FAIL'}")
    
    if hackrx_ok:
        print("\n🎉 Your endpoint is ready for hackathon submission!")
        print(f"URL: {BASE_URL}/hackrx/run")
    else:
        print("\n⚠️ Please restart your Flask server and try again.")

if __name__ == "__main__":
    main() 