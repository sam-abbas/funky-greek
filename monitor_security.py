#!/usr/bin/env python3
"""
Security Monitoring Tool
Monitor your chart analyzer's security features in real-time
"""

import requests
import time
import json
from datetime import datetime

def monitor_security():
    """Monitor security features in real-time"""
    base_url = "http://localhost:8000"
    
    print("🔒 Security Monitoring Dashboard")
    print("=" * 50)
    
    while True:
        try:
            # Test security headers
            response = requests.get(f"{base_url}/health")
            headers = response.headers
            
            print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 30)
            
            # Security Headers Status
            security_headers = {
                "X-Content-Type-Options": "XSS Protection",
                "X-Frame-Options": "Clickjacking Protection", 
                "X-XSS-Protection": "Browser XSS Filter",
                "Referrer-Policy": "Referrer Control",
                "Strict-Transport-Security": "HTTPS Enforcement"
            }
            
            print("🛡️  Security Headers:")
            for header, description in security_headers.items():
                status = "✅" if header in headers else "❌"
                print(f"  {status} {header}: {description}")
            
            # Rate Limiting Status
            if "X-RateLimit-Limit" in headers:
                limit = headers["X-RateLimit-Limit"]
                remaining = headers["X-RateLimit-Remaining"]
                reset = headers["X-RateLimit-Reset"]
                
                print(f"\n🚦 Rate Limiting:")
                print(f"  📊 Limit: {limit} requests/hour")
                print(f"  📉 Remaining: {remaining}")
                print(f"  🔄 Reset: {datetime.fromtimestamp(int(reset)).strftime('%H:%M:%S')}")
            
            # Response Status
            print(f"\n📡 Response Status: {response.status_code}")
            
            # Check for sensitive data exposure
            try:
                info_response = requests.get(f"{base_url}/info")
                info_data = info_response.json()
                
                sensitive_found = False
                for key in info_data.keys():
                    if any(sensitive in key.lower() for sensitive in ['api_key', 'secret', 'password', 'token']):
                        sensitive_found = True
                        break
                
                if sensitive_found:
                    print("⚠️  WARNING: Potential sensitive data exposure!")
                else:
                    print("✅ No sensitive data detected in responses")
                    
            except Exception as e:
                print(f"❌ Error checking data sanitization: {e}")
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_security()
