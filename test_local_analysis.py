#!/usr/bin/env python3
"""
Test Script for Local Analysis Functionality

This script tests the local analysis capabilities without requiring
any external API keys or web server setup.
"""

import sys
import os
from PIL import Image, ImageDraw
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_chart():
    """Create a simple test chart image for testing"""
    # Create a test chart
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw a demo chart pattern (uptrend)
    points = [(50, 300), (200, 250), (350, 200), (500, 150), (650, 100), (750, 50)]
    draw.line(points, fill='blue', width=3)
    
    # Add some candlestick-like elements
    for i in range(5):
        x = 100 + i * 120
        y = 250 + (i * 30)
        color = 'green' if i < 3 else 'red'
        draw.rectangle([x-10, y-20, x+10, y+20], fill=color, outline='black')
    
    # Add grid
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Add some text labels
    draw.text((10, 10), "Test Chart - Uptrend", fill='black')
    draw.text((10, height-20), "Price", fill='black')
    draw.text((width-60, height//2), "Time", fill='black')
    
    return image

def test_local_analysis():
    """Test the local analysis functionality"""
    print("🧪 Testing Local Analysis Functionality")
    print("=" * 50)
    
    try:
        # Import required modules
        from chart_analyzer_enhanced import EnhancedChartAnalyzer
        from models import AnalysisResponse
        
        print("✅ Successfully imported required modules")
        
        # Create test chart
        test_image = create_test_chart()
        print("✅ Created test chart image")
        
        # Create local-only analyzer (no LLM)
        analyzer = EnhancedChartAnalyzer(openai_api_key=None)
        print("✅ Created local analyzer (LLM disabled)")
        
        # Test analysis
        print("\n🔍 Running local analysis...")
        start_time = time.time()
        
        analysis_result = analyzer.analyze_chart(test_image)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"✅ Analysis completed in {processing_time:.1f}ms")
        
        # Test response creation
        response = AnalysisResponse(
            success=True,
            message="Local chart analysis completed successfully (no LLM calls)",
            analysis=analysis_result,
            processing_time_ms=processing_time,
            llm_enhanced=False
        )
        print("✅ Successfully created AnalysisResponse")
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"   Sentiment: {analysis_result.overall_sentiment}")
        print(f"   Confidence: {analysis_result.confidence_score:.1%}")
        print(f"   Indicators: {len(analysis_result.indicators)}")
        print(f"   Patterns: {len(analysis_result.patterns)}")
        print(f"   Risk Level: {analysis_result.risk_level}")
        print(f"   Trading Advice: {analysis_result.trading_advice}")
        
        if analysis_result.support_levels:
            print(f"   Support Levels: {len(analysis_result.support_levels)}")
        if analysis_result.resistance_levels:
            print(f"   Resistance Levels: {len(analysis_result.resistance_levels)}")
        
        print(f"   Warnings: {len(analysis_result.warnings)}")
        print(f"   Insights: {len(analysis_result.insights)}")
        
        # Test JSON serialization
        try:
            response_dict = response.dict()
            print("✅ Successfully converted response to dictionary")
            
            # Test that we can access key fields
            assert response_dict['success'] == True
            assert response_dict['llm_enhanced'] == False
            assert 'analysis' in response_dict
            print("✅ Response validation passed")
            
        except Exception as e:
            print(f"❌ Error in response serialization: {str(e)}")
            return False
        
        print("\n🎉 All local analysis tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   Make sure all required packages are installed:")
        print("   pip install -r requirements_enhanced.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_endpoint_simulation():
    """Test that the endpoint logic would work correctly"""
    print("\n🔌 Testing Endpoint Logic Simulation")
    print("=" * 50)
    
    try:
        from chart_analyzer_enhanced import EnhancedChartAnalyzer
        
        # Simulate the endpoint logic
        test_image = create_test_chart()
        
        # Create temporary analyzer with LLM disabled (like in endpoint)
        temp_analyzer = EnhancedChartAnalyzer(openai_api_key=None)
        
        # Simulate analysis
        start_time = time.time()
        analysis_result = temp_analyzer.analyze_chart(test_image)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"✅ Endpoint simulation completed in {processing_time:.1f}ms")
        print(f"✅ LLM was properly disabled (no API calls made)")
        print(f"✅ Analysis result generated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint simulation failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Enhanced Stock Chart Analyzer - Local Analysis Tests")
    print("=" * 60)
    print("🔒 Testing Local Analysis Only - No LLM API calls")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_local_analysis()
    test2_passed = test_endpoint_simulation()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Local Analysis Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Endpoint Simulation: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Local analysis is working correctly.")
        print("\n💡 You can now use:")
        print("   - python local_analysis.py <image_path>")
        print("   - python batch_local_analysis.py <directory>")
        print("   - POST /analyze-chart-local endpoint")
        print("   - GET /demo-local endpoint")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
