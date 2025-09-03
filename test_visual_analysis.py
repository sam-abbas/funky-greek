#!/usr/bin/env python3
"""
Visual Analysis Test
Tests the enhanced visual highlighting system for chart analysis
"""

import numpy as np
from PIL import Image, ImageDraw
import logging
from chart_analyzer_enhanced import EnhancedChartAnalyzer
from visual_highlighter import VisualHighlighter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_complex_test_chart(width=1200, height=800):
    """Create a complex chart with multiple patterns for visual testing"""
    # Create base image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Generate complex price data with clear patterns
    np.random.seed(123)
    base_price = 100
    prices = [base_price]
    
    # Create different market phases with distinct patterns
    phases = [
        (0, 30, 0.3, 1.5),    # Bullish phase
        (30, 50, -0.2, 1.0),  # Bearish phase with gaps
        (50, 70, 0.1, 0.8),   # Consolidation
        (70, 100, 0.4, 2.0)   # Strong bullish phase
    ]
    
    for phase_start, phase_end, trend, volatility in phases:
        for i in range(phase_start, phase_end):
            x = i * 12
            
            # Create trend with volatility
            change = np.random.normal(trend, volatility)
            new_price = max(50, prices[-1] + change)
            prices.append(new_price)
            
            # Create OHLC data
            prev_price = prices[-2]
            open_price = prev_price
            close_price = new_price
            
            # Add some realistic OHLC variation
            high_price = max(open_price, close_price) + np.random.uniform(0, 3)
            low_price = min(open_price, close_price) - np.random.uniform(0, 3)
            
            # Convert to y-coordinates
            y_open = height - int((open_price - 50) * height / 200)
            y_close = height - int((close_price - 50) * height / 200)
            y_high = height - int((high_price - 50) * height / 200)
            y_low = height - int((low_price - 50) * height / 200)
            
            # Draw wick
            draw.line([(x, y_high), (x, y_low)], fill='black', width=2)
            
            # Draw body
            if close_price > open_price:  # Bullish
                color = 'green'
                body_top = y_close
                body_bottom = y_open
            else:  # Bearish
                color = 'red'
                body_top = y_open
                body_bottom = y_close
            
            draw.rectangle([(x-5, body_top), (x+5, body_bottom)], fill=color, outline='black')
            
            # Add volume bars
            volume = np.random.randint(1000000, 5000000)
            volume_height = int((volume / 5000000) * (height // 4))
            y_vol_start = height - volume_height
            draw.rectangle([(x-5, y_vol_start), (x+5, height)], fill='blue', outline='black')
    
    return img

def test_visual_analysis():
    """Test the visual analysis system"""
    print("🎨 Testing Visual Analysis System")
    print("=" * 50)
    
    # Create analyzer
    analyzer = EnhancedChartAnalyzer(openai_api_key=None)
    
    # Create test chart
    print("\n📊 Creating Complex Test Chart...")
    test_chart = create_complex_test_chart()
    test_chart.save("test_visual_chart.png")
    print("✅ Test chart created: test_visual_chart.png")
    
    try:
        print("\n🔍 Running Analysis with Visualization...")
        
        # Analyze with visualization
        analysis_result, enhanced_image = analyzer.analyze_chart_with_visualization(
            test_chart, save_visualization=True
        )
        
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   Sentiment: {analysis_result.overall_sentiment}")
        print(f"   Confidence: {analysis_result.confidence_score:.1%}")
        print(f"   Risk Level: {analysis_result.risk_level}")
        
        print(f"\n🎯 DETECTED FEATURES:")
        print(f"   • Technical Indicators: {len(analysis_result.indicators)}")
        print(f"   • Chart Patterns: {len(analysis_result.patterns)}")
        print(f"   • Fair Value Gaps: {len(analysis_result.fair_value_gaps)}")
        print(f"   • Daily Levels: {len(analysis_result.daily_levels)}")
        print(f"   • Price Action Patterns: {len(analysis_result.price_action_patterns)}")
        print(f"   • Order Blocks: {len(analysis_result.order_blocks)}")
        print(f"   • Liquidity Zones: {len(analysis_result.liquidity_zones)}")
        print(f"   • Market Structure: {len(analysis_result.market_structure)}")
        print(f"   • Support Levels: {len(analysis_result.support_levels)}")
        print(f"   • Resistance Levels: {len(analysis_result.resistance_levels)}")
        
        # Create legend
        print(f"\n🎨 Creating Analysis Legend...")
        highlighter = VisualHighlighter()
        legend = highlighter.create_pattern_legend()
        legend.save("analysis_legend.png")
        print("✅ Analysis legend created: analysis_legend.png")
        
        print(f"\n✅ Visual Analysis Complete!")
        print(f"   Enhanced visualization saved with timestamp")
        print(f"   All patterns and indicators highlighted")
        print(f"   Professional-grade visual output created")
        
        # Show some key findings
        if analysis_result.fair_value_gaps:
            print(f"\n🎯 KEY FAIR VALUE GAPS:")
            for gap in analysis_result.fair_value_gaps[:3]:  # Show first 3
                print(f"   • {gap.gap_type.upper()} FVG: ${gap.start_price:.2f} - ${gap.end_price:.2f}")
                print(f"     Confidence: {gap.confidence:.1%}, Fill Probability: {gap.fill_probability:.1%}")
        
        if analysis_result.order_blocks:
            print(f"\n🏢 KEY ORDER BLOCKS:")
            for block in analysis_result.order_blocks[:3]:  # Show first 3
                print(f"   • {block.block_type.upper()} OB: ${block.low:.2f} - ${block.high:.2f}")
                print(f"     Strength: {block.strength:.1%}, Tested: {'Yes' if block.tested else 'No'}")
        
        if analysis_result.price_action_patterns:
            print(f"\n🕯️ PRICE ACTION PATTERNS:")
            for pattern in analysis_result.price_action_patterns[:3]:  # Show first 3
                print(f"   • {pattern.pattern_name} ({pattern.direction})")
                print(f"     Confidence: {pattern.confidence:.1%}")
        
        print(f"\n🎨 VISUAL OUTPUT FILES:")
        print(f"   • test_visual_chart.png - Original test chart")
        print(f"   • enhanced_analysis_*.png - Enhanced visualization with highlights")
        print(f"   • analysis_legend.png - Legend for visual elements")
        
    except Exception as e:
        print(f"❌ Visual analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visual_analysis()
