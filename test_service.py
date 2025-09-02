#!/usr/bin/env python3
"""
Simple test script for the Stock Chart Analysis API
"""

import requests
import json
from PIL import Image
import numpy as np
import io

def create_test_chart():
    """Create a simple test chart image"""
    # Create a 400x300 image with a simple line chart
    width, height = 400, 300
    
    # Create image with white background
    image = Image.new('RGB', (width, height), 'white')
    
    # Create a simple line chart (simulating price movement)
    pixels = image.load()
    
    # Draw grid lines
    for x in range(0, width, 50):
        for y in range(height):
            pixels[x, y] = (200, 200, 200)  # Light gray
    
    for y in range(0, height, 50):
        for x in range(width):
            pixels[x, y] = (200, 200, 200)  # Light gray
    
    # Draw a simple price line (simulating uptrend)
    for x in range(width):
        # Create a simple upward trend with some noise
        y = int(height - 50 - (x * 0.3) + np.random.normal(0, 5))
        y = max(0, min(height - 1, y))
        
        # Draw the line
        for offset in range(-1, 2):
            y_pos = y + offset
            if 0 <= y_pos < height:
                pixels[x, y_pos] = (0, 100, 200)  # Blue
    
    return image

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing Stock Chart Analysis API...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to service. Is it running?")
        return
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: Chart analysis
    print("\n3. Testing chart analysis...")
    try:
        # Create test chart
        test_image = create_test_chart()
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Prepare files for upload
        files = {'file': ('test_chart.png', img_byte_arr, 'image/png')}
        
        # Make request
        response = requests.post(f"{base_url}/analyze-chart", files=files)
        
        if response.status_code == 200:
            print("✅ Chart analysis passed")
            result = response.json()
            
            # Print key results
            analysis = result.get('analysis', {})
            print(f"   Overall Sentiment: {analysis.get('overall_sentiment', 'N/A')}")
            print(f"   Confidence Score: {analysis.get('confidence_score', 'N/A'):.2f}")
            print(f"   Risk Level: {analysis.get('risk_level', 'N/A')}")
            print(f"   Trading Advice: {analysis.get('trading_advice', 'N/A')}")
            
            # Print indicators
            indicators = analysis.get('indicators', [])
            print(f"   Technical Indicators: {len(indicators)} found")
            for ind in indicators[:3]:  # Show first 3
                print(f"     - {ind['name']}: {ind['signal']} (strength: {ind['strength']:.2f})")
            
            # Print patterns
            patterns = analysis.get('patterns', [])
            print(f"   Chart Patterns: {len(patterns)} found")
            for pattern in patterns:
                print(f"     - {pattern['name']}: {pattern['signal']} (confidence: {pattern['confidence']:.2f})")
            
        else:
            print(f"❌ Chart analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Chart analysis error: {e}")
    
    # Test 4: API documentation
    print("\n4. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            print(f"   Visit: {base_url}/docs")
        else:
            print(f"❌ API documentation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API documentation error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    test_api()
