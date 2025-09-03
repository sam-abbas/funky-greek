#!/usr/bin/env python3
"""
Enhanced Chart Recognition Test
Tests the improved chart detection capabilities with various chart types
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import logging
from chart_analyzer_enhanced import EnhancedChartAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_candlestick_chart(width=800, height=600):
    """Create a realistic candlestick chart"""
    # Create base image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid lines
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Generate price data
    np.random.seed(42)
    base_price = 100
    prices = [base_price]
    for _ in range(width // 20):
        change = np.random.normal(0, 2)
        new_price = max(50, prices[-1] + change)
        prices.append(new_price)
    
    # Draw candlesticks
    for i, price in enumerate(prices[1:], 1):
        x = i * 20
        prev_price = prices[i-1]
        
        # Calculate open, high, low, close
        open_price = prev_price
        close_price = price
        high_price = max(open_price, close_price) + np.random.uniform(0, 3)
        low_price = min(open_price, close_price) - np.random.uniform(0, 3)
        
        # Convert prices to y-coordinates (inverted)
        y_open = height - int((open_price - 50) * height / 100)
        y_close = height - int((close_price - 50) * height / 100)
        y_high = height - int((high_price - 50) * height / 100)
        y_low = height - int((low_price - 50) * height / 100)
        
        # Draw wick
        draw.line([(x, y_high), (x, y_low)], fill='black', width=2)
        
        # Draw body
        if close_price > open_price:  # Bullish (green)
            color = 'green'
            body_top = y_close
            body_bottom = y_open
        else:  # Bearish (red)
            color = 'red'
            body_top = y_open
            body_bottom = y_close
        
        draw.rectangle([(x-8, body_top), (x+8, body_bottom)], fill=color, outline='black')
    
    # Add volume bars at bottom
    volume_height = height // 4
    for i, price in enumerate(prices[1:], 1):
        x = i * 20
        volume = np.random.randint(1000000, 5000000)
        bar_height = int((volume / 5000000) * volume_height)
        y_start = height - bar_height
        draw.rectangle([(x-8, y_start), (x+8, height)], fill='blue', outline='black')
    
    return img

def create_line_chart(width=800, height=600):
    """Create a line chart with trend lines"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Generate price data with trend
    np.random.seed(123)
    prices = []
    x_coords = []
    
    for i in range(0, width, 10):
        # Add trend component
        trend = i * 0.1
        # Add random component
        noise = np.random.normal(0, 5)
        price = 100 + trend + noise
        prices.append(price)
        x_coords.append(i)
    
    # Draw line chart
    for i in range(1, len(prices)):
        x1, x2 = x_coords[i-1], x_coords[i]
        y1 = height - int((prices[i-1] - 50) * height / 150)
        y2 = height - int((prices[i] - 50) * height / 150)
        draw.line([(x1, y1), (x2, y2)], fill='blue', width=3)
    
    # Add trend line
    start_y = height - int((prices[0] - 50) * height / 150)
    end_y = height - int((prices[-1] - 50) * height / 150)
    draw.line([(0, start_y), (width, end_y)], fill='red', width=2)
    
    return img

def create_bar_chart(width=800, height=600):
    """Create a bar chart"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Generate bar data
    np.random.seed(456)
    bar_width = 30
    spacing = 50
    
    for i in range(0, width - bar_width, spacing):
        bar_height = np.random.randint(50, height - 100)
        y_start = height - bar_height
        
        # Random color for each bar
        color = tuple(np.random.randint(0, 255, 3))
        
        draw.rectangle([(i, y_start), (i + bar_width, height)], fill=color, outline='black')
    
    return img

def test_chart_recognition():
    """Test chart recognition with different chart types"""
    print("🧪 Testing Enhanced Chart Recognition")
    print("=" * 50)
    
    # Create analyzer
    analyzer = EnhancedChartAnalyzer(openai_api_key=None)
    
    # Test 1: Candlestick Chart
    print("\n📊 Test 1: Candlestick Chart Recognition")
    candlestick_img = create_candlestick_chart()
    candlestick_img.save("test_candlestick.png")
    
    try:
        result = analyzer.analyze_chart(candlestick_img)
        print(f"✅ Candlestick analysis successful")
        print(f"   - Price points detected: {len(result.indicators)} indicators")
        print(f"   - Patterns found: {len(result.patterns)}")
        print(f"   - Sentiment: {result.overall_sentiment}")
        print(f"   - Confidence: {result.confidence_score:.1%}")
    except Exception as e:
        print(f"❌ Candlestick analysis failed: {e}")
    
    # Test 2: Line Chart
    print("\n📈 Test 2: Line Chart Recognition")
    line_img = create_line_chart()
    line_img.save("test_line.png")
    
    try:
        result = analyzer.analyze_chart(line_img)
        print(f"✅ Line chart analysis successful")
        print(f"   - Price points detected: {len(result.indicators)} indicators")
        print(f"   - Patterns found: {len(result.patterns)}")
        print(f"   - Sentiment: {result.overall_sentiment}")
        print(f"   - Confidence: {result.confidence_score:.1%}")
    except Exception as e:
        print(f"❌ Line chart analysis failed: {e}")
    
    # Test 3: Bar Chart
    print("\n📊 Test 3: Bar Chart Recognition")
    bar_img = create_bar_chart()
    bar_img.save("test_bar.png")
    
    try:
        result = analyzer.analyze_chart(bar_img)
        print(f"✅ Bar chart analysis successful")
        print(f"   - Price points detected: {len(result.indicators)} indicators")
        print(f"   - Patterns found: {len(result.patterns)}")
        print(f"   - Sentiment: {result.overall_sentiment}")
        print(f"   - Confidence: {result.confidence_score:.1%}")
    except Exception as e:
        print(f"❌ Bar chart analysis failed: {e}")
    
    print("\n🎯 Chart Recognition Test Complete!")
    print("Check the generated test images to see the detected patterns.")

if __name__ == "__main__":
    test_chart_recognition()
