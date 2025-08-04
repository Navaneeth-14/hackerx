#!/usr/bin/env python3
"""
Test script for hackathon API format
Matches the exact requirements from the hackathon
"""

import requests
import json

# Your ngrok URL
BASE_URL = "https://0468c638cef7.ngrok-free.app"

def test_hackathon_format():
    """Test the hackathon API format"""
    print("🧪 Testing Hackathon API Format...")
    
    url = f"{BASE_URL}/hackrx/run"
    
    # Headers as required by hackathon
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer e6dfe3fbc81eaccabae37a2960e15bc85abe1d3e710777adbbae47ee6b4b4fae"  # Any API key works for testing
    }
    
    # Request body as required by hackathon
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
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            url, 
            json=payload,
            headers=headers,
            timeout=60  # 60 second timeout
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Hackathon API format test passed!")
            return True
        else:
            print("❌ Hackathon API format test failed")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - server might not be running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_format():
    """Test with simple questions (no document URL)"""
    print("\n🧪 Testing Simple Format (No Document URL)...")
    
    url = f"{BASE_URL}/hackrx/run"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test_key_123"
    }
    
    payload = {
        "questions": [
            "What is covered under this policy?",
            "What is the maximum coverage amount?"
        ]
    }
    
    try:
        response = requests.post(
            url, 
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Simple format test passed!")
            return True
        else:
            print("❌ Simple format test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Hackathon API Format")
    print("=" * 60)
    
    # Test simple format first
    simple_ok = test_simple_format()
    
    # Test full hackathon format
    hackathon_ok = test_hackathon_format()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Simple Format: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"Hackathon Format: {'✅ PASS' if hackathon_ok else '❌ FAIL'}")
    
    if hackathon_ok:
        print("\n🎉 Your API is ready for hackathon submission!")
        print(f"URL: {BASE_URL}/hackrx/run")
        print("\n📋 Submission Details:")
        print(f"Webhook URL: {BASE_URL}/hackrx/run")
        print("Method: POST")
        print("Headers: Authorization: Bearer <api_key>")
        print("Content-Type: application/json")
    else:
        print("\n⚠️ Please fix the issues and try again.")

if __name__ == "__main__":
    main() 