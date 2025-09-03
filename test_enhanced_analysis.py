#!/usr/bin/env python3
"""
Test script for the Enhanced Chart Analyzer
Demonstrates real technical analysis on different chart types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chart_analyzer_enhanced import EnhancedChartAnalyzer
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import time

def create_uptrend_chart():
    """Create a chart showing strong uptrend"""
    print("🎨 Creating Uptrend Chart...")
    
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw uptrend line
    points = [(50, 350), (200, 300), (350, 250), (500, 200), (650, 150), (750, 100)]
    draw.line(points, fill='green', width=3)
    
    # Add candlestick-like rectangles (green for uptrend)
    for i in range(5):
        x = 100 + i * 120
        y = 200 + (i * 20)
        draw.rectangle([x-10, y-20, x+10, y+20], fill='green', outline='black')
    
    # Add grid and labels
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    draw.text((10, 10), "Uptrend Chart", fill='black')
    draw.text((10, height-30), "Time", fill='black')
    draw.text((10, 30), "Price", fill='black')
    
    return image

def create_downtrend_chart():
    """Create a chart showing strong downtrend"""
    print("🎨 Creating Downtrend Chart...")
    
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw downtrend line
    points = [(50, 100), (200, 150), (350, 200), (500, 250), (650, 300), (750, 350)]
    draw.line(points, fill='red', width=3)
    
    # Add candlestick-like rectangles (red for downtrend)
    for i in range(5):
        x = 100 + i * 120
        y = 200 - (i * 20)
        draw.rectangle([x-10, y-20, x+10, y+20], fill='red', outline='black')
    
    # Add grid and labels
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    draw.text((10, 10), "Downtrend Chart", fill='black')
    draw.text((10, height-30), "Time", fill='black')
    draw.text((10, 30), "Price", fill='black')
    
    return image

def create_sideways_chart():
    """Create a chart showing sideways movement"""
    print("🎨 Creating Sideways Chart...")
    
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw sideways line
    base_y = 200
    points = [(50, base_y)]
    for i in range(1, 16):
        x = 50 + i * 50
        # Add some random movement around the base
        y = base_y + np.random.normal(0, 20)
        points.append((x, y))
    
    draw.line(points, fill='blue', width=3)
    
    # Add candlestick-like rectangles
    for i in range(5):
        x = 100 + i * 120
        y = base_y + np.random.normal(0, 15)
        draw.rectangle([x-10, y-20, x+10, y+20], fill='blue', outline='black')
    
    # Add grid and labels
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    draw.text((10, 10), "Sideways Chart", fill='black')
    draw.text((10, height-30), "Time", fill='black')
    draw.text((10, 30), "Price", fill='black')
    
    return image

def create_double_top_chart():
    """Create a chart showing double top pattern"""
    print("🎨 Creating Double Top Chart...")
    
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw double top pattern
    points = [
        (50, 200), (100, 150), (150, 100), (200, 150),  # First peak
        (250, 200), (300, 150), (350, 100), (400, 150),  # Second peak
        (450, 200), (500, 250), (550, 300), (600, 350),  # Decline
        (650, 400), (700, 450), (750, 500)
    ]
    draw.line(points, fill='red', width=3)
    
    # Add candlestick-like rectangles
    for i in range(8):
        x = 80 + i * 80
        y = 200 + (i * 25)
        color = 'red' if i > 4 else 'green'
        draw.rectangle([x-10, y-20, x+10, y+20], fill=color, outline='black')
    
    # Add grid and labels
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    draw.text((10, 10), "Double Top Chart", fill='black')
    draw.text((10, height-30), "Time", fill='black')
    draw.text((10, 30), "Price", fill='black')
    
    return image

def test_chart_analysis(chart_image: Image.Image, chart_name: str):
    """Test analysis on a specific chart"""
    print(f"\n🔍 Testing Analysis on {chart_name}")
    print("=" * 50)
    
    try:
        # Initialize analyzer (without LLM for now)
        analyzer = EnhancedChartAnalyzer()
        
        # Analyze chart
        start_time = time.time()
        result = analyzer.analyze_chart(chart_image)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        
        print(f"✅ Analysis completed in {processing_time:.2f}ms")
        print(f"✅ Overall Sentiment: {result.overall_sentiment.upper()}")
        print(f"✅ Confidence Score: {result.confidence_score:.1%}")
        print(f"✅ Trading Advice: {result.trading_advice}")
        print(f"✅ Risk Level: {result.risk_level.upper()}")
        
        # Show indicators
        if result.indicators:
            print(f"\n📊 Technical Indicators ({len(result.indicators)}):")
            for indicator in result.indicators:
                print(f"   - {indicator.name}: {indicator.signal.upper()} "
                      f"(Strength: {indicator.strength:.1%})")
                print(f"     Value: {indicator.value:.2f}")
                print(f"     Description: {indicator.description}")
        
        # Show patterns
        if result.patterns:
            print(f"\n📈 Chart Patterns ({len(result.patterns)}):")
            for pattern in result.patterns:
                print(f"   - {pattern.name}: {pattern.signal.upper()} "
                      f"(Confidence: {pattern.confidence:.1%})")
                print(f"     Type: {pattern.pattern_type}")
                print(f"     Description: {pattern.description}")
        
        # Show support/resistance
        if result.support_levels:
            print(f"\n🛡️ Support Levels ({len(result.support_levels)}):")
            for level in result.support_levels:
                print(f"   - {level.description} (Strength: {level.strength:.1%})")
        
        if result.resistance_levels:
            print(f"\n🚫 Resistance Levels ({len(result.resistance_levels)}):")
            for level in result.resistance_levels:
                print(f"   - {level.description} (Strength: {level.strength:.1%})")
        
        # Show insights and warnings
        if result.insights:
            print(f"\n💡 Insights ({len(result.insights)}):")
            for insight in result.insights:
                print(f"   - {insight}")
        
        if result.warnings:
            print(f"\n⚠️ Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing {chart_name}: {str(e)}")
        return None

def main():
    """Run enhanced analysis tests on different chart types"""
    print("🚀 Enhanced Stock Chart Analyzer - Real Analysis Tests")
    print("=" * 60)
    
    # Test different chart types
    charts = [
        ("Uptrend Chart", create_uptrend_chart()),
        ("Downtrend Chart", create_downtrend_chart()),
        ("Sideways Chart", create_sideways_chart()),
        ("Double Top Chart", create_double_top_chart())
    ]
    
    results = []
    
    for chart_name, chart_image in charts:
        result = test_chart_analysis(chart_image, chart_name)
        if result:
            results.append((chart_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Analysis Summary")
    print("=" * 60)
    
    for chart_name, result in results:
        print(f"\n{chart_name}:")
        print(f"  Sentiment: {result.overall_sentiment.upper()}")
        print(f"  Confidence: {result.confidence_score:.1%}")
        print(f"  Advice: {result.trading_advice}")
        print(f"  Risk: {result.risk_level.upper()}")
        print(f"  Indicators: {len(result.indicators)}")
        print(f"  Patterns: {len(result.patterns)}")
    
    print(f"\n✅ Successfully analyzed {len(results)} out of {len(charts)} charts")
    
    if len(results) == len(charts):
        print("🎉 All charts analyzed successfully!")
        print("\n💡 Key Benefits of Enhanced Analysis:")
        print("- Real technical indicators calculated from chart data")
        print("- Pattern detection based on actual price movements")
        print("- Different results for different chart types")
        print("- Professional-grade technical analysis")
        print("- LLM integration ready for enhanced insights")
    else:
        print("⚠️ Some charts failed analysis - check error messages above")

if __name__ == "__main__":
    main()
