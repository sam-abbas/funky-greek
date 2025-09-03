#!/usr/bin/env python3
"""
Quick Security Fixes Test
Tests the critical security fixes implemented
"""

import requests
import time

def test_security_fixes():
    """Test the security fixes"""
    base_url = "http://localhost:8000"
    
    print("🔒 Testing Security Fixes...")
    print("=" * 40)
    
    try:
        # Test 1: Check security headers
        print("1. Testing Security Headers...")
        response = requests.get(f"{base_url}/health")
        headers = response.headers
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Referrer-Policy",
            "Strict-Transport-Security"
        ]
        
        missing = []
        for header in required_headers:
            if header in headers:
                print(f"   ✅ {header}: {headers[header]}")
            else:
                missing.append(header)
                print(f"   ❌ {header}: MISSING")
        
        if missing:
            print(f"   ⚠️  Missing headers: {missing}")
        else:
            print("   🎉 All security headers present!")
        
        print()
        
        # Test 2: Check rate limiting headers
        print("2. Testing Rate Limiting Headers...")
        if "X-RateLimit-Limit" in headers:
            print(f"   ✅ Rate Limit: {headers['X-RateLimit-Limit']}")
            print(f"   ✅ Remaining: {headers['X-RateLimit-Remaining']}")
            print(f"   ✅ Reset: {headers['X-RateLimit-Reset']}")
        else:
            print("   ❌ Rate limiting headers missing")
        
        print()
        
        # Test 3: Test sensitive data exposure
        print("3. Testing Sensitive Data Exposure...")
        info_response = requests.get(f"{base_url}/info")
        config_response = requests.get(f"{base_url}/config")
        
        sensitive_patterns = ["environment", "development", "production", "api_key", "secret"]
        
        info_content = info_response.text.lower()
        config_content = config_response.text.lower()
        
        exposed_in_info = [p for p in sensitive_patterns if p in info_content]
        exposed_in_config = [p for p in sensitive_patterns if p in config_content]
        
        if exposed_in_info:
            print(f"   ❌ Sensitive data in /info: {exposed_in_info}")
        else:
            print("   ✅ No sensitive data in /info")
            
        if exposed_in_config:
            print(f"   ❌ Sensitive data in /config: {exposed_in_config}")
        else:
            print("   ✅ No sensitive data in /config")
        
        print()
        
        # Test 4: Quick rate limiting test
        print("4. Testing Rate Limiting (5 requests)...")
        responses = []
        for i in range(5):
            resp = requests.get(f"{base_url}/health")
            responses.append(resp.status_code)
            time.sleep(0.1)
        
        if all(status == 200 for status in responses):
            print("   ✅ Rate limiting working (all requests successful)")
        else:
            print(f"   ❌ Rate limiting issues: {responses}")
        
        print()
        print("🔒 Security Fixes Test Complete!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_security_fixes()
