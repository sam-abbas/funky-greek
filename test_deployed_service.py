#!/usr/bin/env python3
"""
Test script for the deployed Stock Chart Analyzer service
"""

import requests
import json
import time
from PIL import Image, ImageDraw, ImageFont
import io

# Service URL
BASE_URL = "https://funky-greek.onrender.com"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service Status: {data.get('status', 'Unknown')}")
            print(f"✅ Version: {data.get('version', 'Unknown')}")
            print(f"✅ Uptime: {data.get('uptime', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.text}")
    except Exception as e:
        print(f"❌ Error testing health: {e}")

def test_info():
    """Test the info endpoint"""
    print("\n🔍 Testing Info Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/info")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data.get('name', 'Unknown')}")
            print(f"✅ Description: {data.get('description', 'Unknown')}")
            print(f"✅ Version: {data.get('version', 'Unknown')}")
            print(f"✅ Features: {', '.join(data.get('features', []))}")
        else:
            print(f"❌ Info check failed: {response.text}")
    except Exception as e:
        print(f"❌ Error testing info: {e}")

def test_demo():
    """Test the demo endpoint"""
    print("\n🔍 Testing Demo Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/demo")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Demo successful: {data.get('message', 'No message')}")
            if 'analysis' in data:
                analysis = data['analysis']
                print(f"✅ Sentiment: {analysis.get('overall_sentiment', 'Unknown')}")
                print(f"✅ Confidence: {analysis.get('confidence_score', 0):.1%}")
                print(f"✅ Trading Advice: {analysis.get('trading_advice', 'Unknown')}")
                print(f"✅ Risk Level: {analysis.get('risk_level', 'Unknown')}")
        else:
            print(f"❌ Demo failed: {response.text}")
    except Exception as e:
        print(f"❌ Error testing demo: {e}")

def create_test_chart():
    """Create a simple test chart image"""
    print("\n🎨 Creating Test Chart Image...")
    
    # Create a simple chart-like image
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw some lines to simulate a chart
    # Price line (simulating uptrend)
    points = [(50, 350), (200, 300), (350, 250), (500, 200), (650, 150), (750, 100)]
    draw.line(points, fill='blue', width=3)
    
    # Add some candlestick-like rectangles
    for i in range(5):
        x = 100 + i * 120
        y = 200 + (i * 20)
        draw.rectangle([x-10, y-20, x+10, y+20], fill='green', outline='black')
    
    # Add grid lines
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Add labels
    draw.text((10, 10), "Test Stock Chart", fill='black')
    draw.text((10, height-30), "Time", fill='black')
    draw.text((10, 30), "Price", fill='black')
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    print("✅ Test chart created successfully")
    return img_byte_arr

def test_file_upload():
    """Test file upload and analysis"""
    print("\n🔍 Testing File Upload and Analysis...")
    
    try:
        # Create test chart
        test_chart = create_test_chart()
        
        # Prepare file for upload
        files = {'file': ('test_chart.png', test_chart, 'image/png')}
        
        print("📤 Uploading test chart...")
        start_time = time.time()
        
        response = requests.post(f"{BASE_URL}/analyze-chart", files=files)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"Status: {response.status_code}")
        print(f"⏱️  Processing Time: {processing_time:.2f}ms")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis successful: {data.get('message', 'No message')}")
            
            if 'analysis' in data:
                analysis = data['analysis']
                print(f"✅ Sentiment: {analysis.get('overall_sentiment', 'Unknown')}")
                print(f"✅ Confidence: {analysis.get('confidence_score', 0):.1%}")
                print(f"✅ Trading Advice: {analysis.get('trading_advice', 'Unknown')}")
                print(f"✅ Risk Level: {analysis.get('risk_level', 'Unknown')}")
                
                # Show indicators if available
                if 'indicators' in analysis and analysis['indicators']:
                    print(f"✅ Indicators found: {len(analysis['indicators'])}")
                    for ind in analysis['indicators'][:3]:  # Show first 3
                        print(f"   - {ind.get('name', 'Unknown')}: {ind.get('signal', 'Unknown')}")
                
                # Show patterns if available
                if 'patterns' in analysis and analysis['patterns']:
                    print(f"✅ Patterns found: {len(analysis['patterns'])}")
                    for pattern in analysis['patterns'][:3]:  # Show first 3
                        print(f"   - {pattern.get('name', 'Unknown')}: {pattern.get('signal', 'Unknown')}")
                
        else:
            print(f"❌ Analysis failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing file upload: {e}")

def test_error_handling():
    """Test error handling with invalid requests"""
    print("\n🔍 Testing Error Handling...")
    
    # Test with no file
    try:
        response = requests.post(f"{BASE_URL}/analyze-chart")
        print(f"✅ No file test - Status: {response.status_code}")
    except Exception as e:
        print(f"❌ No file test failed: {e}")
    
    # Test with empty file
    try:
        files = {'file': ('empty.txt', io.BytesIO(b''), 'text/plain')}
        response = requests.post(f"{BASE_URL}/analyze-chart", files=files)
        print(f"✅ Empty file test - Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Empty file test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Stock Chart Analyzer Service Tests")
    print("=" * 50)
    
    # Test basic endpoints
    test_health()
    test_info()
    test_demo()
    
    # Test file analysis
    test_file_upload()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("🏁 Testing Complete!")
    print("\n💡 Tips:")
    print("- If all tests pass, your service is working correctly!")
    print("- Check the Render dashboard for any errors")
    print("- Monitor response times for performance")
    print("- Test with real stock chart images for better results")

if __name__ == "__main__":
    main()
