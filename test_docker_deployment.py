#!/usr/bin/env python3
"""
Test the Docker deployment
"""

import requests
import json
import time

def test_docker_deployment():
    """Test the Docker deployment"""
    
    url = "https://funky-greek.onrender.com"
    
    print(f"🚀 Testing Docker Deployment: {url}")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{url}/health", timeout=10)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Service: {data.get('service', 'Unknown')}")
            print(f"   🚀 Version: {data.get('version', 'Unknown')}")
        else:
            print(f"   📄 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Enhanced demo-local endpoint
    print("\n2️⃣ Testing enhanced demo-local endpoint...")
    try:
        response = requests.get(f"{url}/demo-local", timeout=20)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Success: {data.get('success', 'Unknown')}")
            if 'analysis' in data:
                analysis = data['analysis']
                print(f"   📈 Chart Type: {analysis.get('chart_type', 'Unknown')}")
                print(f"   🎯 Confidence: {analysis.get('confidence', 'Unknown')}")
        else:
            print(f"   📄 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Config endpoint
    print("\n3️⃣ Testing config endpoint...")
    try:
        response = requests.get(f"{url}/config", timeout=10)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   🔧 Environment: {data.get('environment', 'Unknown')}")
            print(f"   📊 Local Mode: {data.get('force_local_mode', 'Unknown')}")
        else:
            print(f"   📄 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_docker_deployment()
