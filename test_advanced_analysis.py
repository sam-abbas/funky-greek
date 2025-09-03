#!/usr/bin/env python3
"""
Advanced Analysis Test
Tests all the new advanced trading indicators and patterns
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime, timedelta
import logging
from chart_analyzer_enhanced import EnhancedChartAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_advanced_test_chart(width=1000, height=800):
    """Create a complex chart with multiple advanced patterns"""
    # Create base image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid lines
    for i in range(0, width, 50):
        draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
    for i in range(0, height, 50):
        draw.line([(0, i), (width, i)], fill='lightgray', width=1)
    
    # Generate complex price data with patterns
    np.random.seed(42)
    base_price = 100
    prices = [base_price]
    volumes = [1000000]
    
    # Create price action with various patterns
    for i in range(width // 15):
        x = i * 15
        
        # Create different market phases
        if i < 20:  # Bullish phase
            change = np.random.normal(0.5, 1)
        elif i < 40:  # Consolidation with gaps
            change = np.random.normal(0, 0.5)
        elif i < 60:  # Bearish phase
            change = np.random.normal(-0.3, 1)
        else:  # Recovery phase
            change = np.random.normal(0.2, 0.8)
        
        new_price = max(50, prices[-1] + change)
        prices.append(new_price)
        
        # Volume varies with volatility
        volume = int(1000000 * (1 + abs(change) * 2) * np.random.uniform(0.8, 1.2))
        volumes.append(volume)
        
        # Draw candlestick
        prev_price = prices[-2]
        open_price = prev_price
        close_price = new_price
        high_price = max(open_price, close_price) + np.random.uniform(0, 2)
        low_price = min(open_price, close_price) - np.random.uniform(0, 2)
        
        # Convert to y-coordinates
        y_open = height - int((open_price - 50) * height / 100)
        y_close = height - int((close_price - 50) * height / 100)
        y_high = height - int((high_price - 50) * height / 100)
        y_low = height - int((low_price - 50) * height / 100)
        
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
        
        draw.rectangle([(x-6, body_top), (x+6, body_bottom)], fill=color, outline='black')
        
        # Add volume bars
        volume_height = int((volume / 5000000) * (height // 4))
        y_vol_start = height - volume_height
        draw.rectangle([(x-6, y_vol_start), (x+6, height)], fill='blue', outline='black')
    
    return img

def test_advanced_analysis():
    """Test all advanced analysis features"""
    print("🚀 Testing Advanced Trading Analysis")
    print("=" * 60)
    
    # Create analyzer
    analyzer = EnhancedChartAnalyzer(openai_api_key=None)
    
    # Test with complex chart
    print("\n📊 Creating Advanced Test Chart...")
    test_chart = create_advanced_test_chart()
    test_chart.save("test_advanced_chart.png")
    print("✅ Advanced test chart created: test_advanced_chart.png")
    
    try:
        print("\n🔍 Running Advanced Analysis...")
        result = analyzer.analyze_chart(test_chart)
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"   Overall Sentiment: {result.overall_sentiment}")
        print(f"   Confidence Score: {result.confidence_score:.1%}")
        print(f"   Risk Level: {result.risk_level}")
        
        print(f"\n📊 TECHNICAL INDICATORS ({len(result.indicators)}):")
        for indicator in result.indicators:
            print(f"   • {indicator.name}: {indicator.value:.2f} ({indicator.signal}) - {indicator.strength:.1%} strength")
        
        print(f"\n🔍 CHART PATTERNS ({len(result.patterns)}):")
        for pattern in result.patterns:
            print(f"   • {pattern.name}: {pattern.signal} - {pattern.confidence:.1%} confidence")
        
        print(f"\n📉 SUPPORT LEVELS ({len(result.support_levels)}):")
        for support in result.support_levels:
            print(f"   • ${support.price:.2f} - {support.strength:.1%} strength")
        
        print(f"\n📈 RESISTANCE LEVELS ({len(result.resistance_levels)}):")
        for resistance in result.resistance_levels:
            print(f"   • ${resistance.price:.2f} - {resistance.strength:.1%} strength")
        
        # NEW ADVANCED FEATURES
        print(f"\n🎯 FAIR VALUE GAPS ({len(result.fair_value_gaps)}):")
        for gap in result.fair_value_gaps:
            print(f"   • {gap.gap_type.upper()} FVG: ${gap.start_price:.2f} - ${gap.end_price:.2f}")
            print(f"     Confidence: {gap.confidence:.1%}, Fill Probability: {gap.fill_probability:.1%}")
            print(f"     Volume Confirmed: {'Yes' if gap.volume_confirmation else 'No'}")
        
        print(f"\n📅 DAILY LEVELS ({len(result.daily_levels)}):")
        for level in result.daily_levels:
            print(f"   • {level.level_type.upper()}: ${level.price:.2f} at {level.time}")
            print(f"     Strength: {level.strength:.1f}/10, Tested: {level.tested_count} times")
        
        print(f"\n🕯️ PRICE ACTION PATTERNS ({len(result.price_action_patterns)}):")
        for pattern in result.price_action_patterns:
            print(f"   • {pattern.pattern_name} ({pattern.pattern_type})")
            print(f"     Direction: {pattern.direction}, Confidence: {pattern.confidence:.1%}")
            print(f"     Time: {pattern.start_time} - {pattern.end_time}")
        
        print(f"\n🏢 ORDER BLOCKS ({len(result.order_blocks)}):")
        for block in result.order_blocks:
            print(f"   • {block.block_type.upper()} Order Block: ${block.low:.2f} - ${block.high:.2f}")
            print(f"     Strength: {block.strength:.1%}, Tested: {'Yes' if block.tested else 'No'}")
            print(f"     Invalidation: ${block.invalidation_level:.2f}")
        
        print(f"\n💧 LIQUIDITY ZONES ({len(result.liquidity_zones)}):")
        for zone in result.liquidity_zones:
            print(f"   • {zone.zone_type.upper()} Liquidity: ${zone.low:.2f} - ${zone.high:.2f}")
            print(f"     Strength: {zone.strength:.1%}, Tested: {zone.tested_count} times")
        
        print(f"\n🏗️ MARKET STRUCTURE ({len(result.market_structure)}):")
        for structure in result.market_structure:
            print(f"   • {structure.structure_type.replace('_', ' ').title()}")
            print(f"     Direction: {structure.direction}, Key Level: ${structure.key_level:.2f}")
            print(f"     Confidence: {structure.confidence:.1%}")
        
        print(f"\n💡 INSIGHTS ({len(result.insights)}):")
        for insight in result.insights:
            print(f"   • {insight}")
        
        print(f"\n⚠️ WARNINGS ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   • {warning}")
        
        print(f"\n🎯 TRADING ADVICE:")
        print(f"   {result.trading_advice}")
        
        if result.stop_loss_suggestions:
            print(f"\n🛑 STOP LOSS SUGGESTIONS:")
            for sl in result.stop_loss_suggestions:
                print(f"   • ${sl:.2f}")
        
        if result.take_profit_targets:
            print(f"\n🎯 TAKE PROFIT TARGETS:")
            for tp in result.take_profit_targets:
                print(f"   • ${tp:.2f}")
        
        print(f"\n✅ Advanced Analysis Complete!")
        print(f"   Processing Time: {result.timestamp}")
        print(f"   Total Features Detected: {len(result.indicators) + len(result.patterns) + len(result.fair_value_gaps) + len(result.daily_levels) + len(result.price_action_patterns) + len(result.order_blocks) + len(result.liquidity_zones) + len(result.market_structure)}")
        
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_analysis()
