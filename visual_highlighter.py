#!/usr/bin/env python3
"""
Visual Highlighter for Chart Analysis
Creates professional visual representations of detected patterns and indicators
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from models import (
    FairValueGap, DailyLevel, PriceActionPattern, 
    OrderBlock, LiquidityZone, MarketStructure,
    SupportResistance
)

class VisualHighlighter:
    """Creates visual highlights for detected trading patterns and indicators"""
    
    def __init__(self):
        self.colors = {
            'bullish': (0, 255, 0),      # Green
            'bearish': (255, 0, 0),      # Red
            'neutral': (255, 255, 0),    # Yellow
            'support': (0, 255, 255),    # Cyan
            'resistance': (255, 0, 255), # Magenta
            'gap': (255, 165, 0),        # Orange
            'order_block': (128, 0, 128), # Purple
            'liquidity': (0, 191, 255),  # Deep Sky Blue
            'structure': (255, 20, 147)  # Deep Pink
        }
    
    def create_analysis_visualization(self, 
                                    original_image: Image.Image,
                                    analysis_result,
                                    price_data: pd.DataFrame = None) -> Image.Image:
        """
        Create a comprehensive visual analysis overlay
        """
        try:
            # Create a copy of the original image
            vis_image = original_image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            # Get image dimensions
            width, height = vis_image.size
            
            # Create overlay for different elements
            self._draw_fair_value_gaps(draw, analysis_result.fair_value_gaps, width, height)
            self._draw_daily_levels(draw, analysis_result.daily_levels, width, height)
            self._draw_order_blocks(draw, analysis_result.order_blocks, width, height)
            self._draw_liquidity_zones(draw, analysis_result.liquidity_zones, width, height)
            self._draw_support_resistance(draw, analysis_result.support_levels, 
                                        analysis_result.resistance_levels, width, height)
            self._draw_market_structure(draw, analysis_result.market_structure, width, height)
            
            # Add analysis summary overlay
            self._draw_analysis_summary(draw, analysis_result, width, height)
            
            return vis_image
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return original_image
    
    def _draw_fair_value_gaps(self, draw: ImageDraw.Draw, gaps: List[FairValueGap], 
                            width: int, height: int):
        """Draw fair value gaps on the chart"""
        try:
            for gap in gaps:
                # Convert prices to y-coordinates (assuming price range 50-350)
                y_start = height - int((gap.start_price - 50) * height / 300)
                y_end = height - int((gap.end_price - 50) * height / 300)
                
                # Draw gap rectangle
                color = self.colors['gap']
                alpha = int(128 * gap.confidence)  # Transparency based on confidence
                
                # Create semi-transparent overlay
                overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # Draw gap area
                overlay_draw.rectangle([(0, min(y_start, y_end)), 
                                      (width, max(y_start, y_end))], 
                                     fill=(*color, alpha))
                
                # Add gap label
                label = f"FVG {gap.gap_type.upper()}\n{gap.confidence:.1%}"
                overlay_draw.text((10, (y_start + y_end) // 2), label, 
                                fill=(255, 255, 255, 255))
                
        except Exception as e:
            print(f"Error drawing fair value gaps: {e}")
    
    def _draw_daily_levels(self, draw: ImageDraw.Draw, levels: List[DailyLevel], 
                          width: int, height: int):
        """Draw daily high/low levels"""
        try:
            for level in levels:
                # Convert price to y-coordinate
                y = height - int((level.price - 50) * height / 300)
                
                # Choose color based on level type
                if level.level_type == 'high':
                    color = self.colors['resistance']
                elif level.level_type == 'low':
                    color = self.colors['support']
                else:
                    color = self.colors['neutral']
                
                # Draw horizontal line
                draw.line([(0, y), (width, y)], fill=color, width=2)
                
                # Add level label
                label = f"{level.level_type.upper()}: ${level.price:.2f}"
                draw.text((10, y - 20), label, fill=color)
                
        except Exception as e:
            print(f"Error drawing daily levels: {e}")
    
    def _draw_order_blocks(self, draw: ImageDraw.Draw, blocks: List[OrderBlock], 
                          width: int, height: int):
        """Draw order blocks"""
        try:
            for block in blocks:
                # Convert prices to y-coordinates
                y_high = height - int((block.high - 50) * height / 300)
                y_low = height - int((block.low - 50) * height / 300)
                
                # Choose color based on block type
                color = self.colors['bullish'] if block.block_type == 'bullish' else self.colors['bearish']
                
                # Draw order block rectangle
                alpha = int(100 * block.strength)
                overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                overlay_draw.rectangle([(0, min(y_low, y_high)), (width, max(y_low, y_high))], 
                                     fill=(*color, alpha))
                
                # Add block label
                label = f"OB {block.block_type.upper()}\n{block.strength:.1%}"
                overlay_draw.text((width - 100, (y_high + y_low) // 2), label, 
                                fill=(255, 255, 255, 255))
                
        except Exception as e:
            print(f"Error drawing order blocks: {e}")
    
    def _draw_liquidity_zones(self, draw: ImageDraw.Draw, zones: List[LiquidityZone], 
                             width: int, height: int):
        """Draw liquidity zones"""
        try:
            for zone in zones:
                # Convert prices to y-coordinates
                y_high = height - int((zone.high - 50) * height / 300)
                y_low = height - int((zone.low - 50) * height / 300)
                
                # Draw liquidity zone
                color = self.colors['liquidity']
                alpha = int(80 * zone.strength)
                
                overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                overlay_draw.rectangle([(0, min(y_low, y_high)), (width, max(y_low, y_high))], 
                                     fill=(*color, alpha))
                
                # Add zone label
                label = f"LZ {zone.zone_type.upper()}\n{zone.tested_count} tests"
                overlay_draw.text((width - 150, (y_high + y_low) // 2), label, 
                                fill=(255, 255, 255, 255))
                
        except Exception as e:
            print(f"Error drawing liquidity zones: {e}")
    
    def _draw_support_resistance(self, draw: ImageDraw.Draw, 
                               support_levels: List[SupportResistance],
                               resistance_levels: List[SupportResistance],
                               width: int, height: int):
        """Draw support and resistance levels"""
        try:
            # Draw support levels
            for support in support_levels:
                y = height - int((support.price - 50) * height / 300)
                draw.line([(0, y), (width, y)], fill=self.colors['support'], width=3)
                draw.text((10, y - 15), f"Support: ${support.price:.2f}", 
                         fill=self.colors['support'])
            
            # Draw resistance levels
            for resistance in resistance_levels:
                y = height - int((resistance.price - 50) * height / 300)
                draw.line([(0, y), (width, y)], fill=self.colors['resistance'], width=3)
                draw.text((10, y - 15), f"Resistance: ${resistance.price:.2f}", 
                         fill=self.colors['resistance'])
                
        except Exception as e:
            print(f"Error drawing support/resistance: {e}")
    
    def _draw_market_structure(self, draw: ImageDraw.Draw, structures: List[MarketStructure], 
                              width: int, height: int):
        """Draw market structure changes"""
        try:
            for structure in structures:
                # Convert price to y-coordinate
                y = height - int((structure.key_level - 50) * height / 300)
                
                # Choose color based on direction
                color = self.colors['bullish'] if structure.direction == 'bullish' else self.colors['bearish']
                
                # Draw structure line
                draw.line([(0, y), (width, y)], fill=color, width=2)
                
                # Add structure label
                label = f"{structure.structure_type.replace('_', ' ').title()}\n{structure.direction.upper()}"
                draw.text((width - 200, y - 20), label, fill=color)
                
        except Exception as e:
            print(f"Error drawing market structure: {e}")
    
    def _draw_analysis_summary(self, draw: ImageDraw.Draw, analysis_result, 
                              width: int, height: int):
        """Draw analysis summary overlay"""
        try:
            # Create summary box
            box_width = 300
            box_height = 200
            x = width - box_width - 10
            y = 10
            
            # Draw background box
            draw.rectangle([(x, y), (x + box_width, y + box_height)], 
                          fill=(0, 0, 0, 180), outline=(255, 255, 255))
            
            # Add summary text
            summary_lines = [
                f"Sentiment: {analysis_result.overall_sentiment.upper()}",
                f"Confidence: {analysis_result.confidence_score:.1%}",
                f"Risk: {analysis_result.risk_level.upper()}",
                f"Indicators: {len(analysis_result.indicators)}",
                f"Patterns: {len(analysis_result.patterns)}",
                f"FVG: {len(analysis_result.fair_value_gaps)}",
                f"Order Blocks: {len(analysis_result.order_blocks)}",
                f"Liquidity Zones: {len(analysis_result.liquidity_zones)}"
            ]
            
            for i, line in enumerate(summary_lines):
                draw.text((x + 10, y + 10 + i * 20), line, fill=(255, 255, 255))
                
        except Exception as e:
            print(f"Error drawing analysis summary: {e}")
    
    def create_pattern_legend(self, width: int = 400, height: int = 300) -> Image.Image:
        """Create a legend for all the visual elements"""
        try:
            legend = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(legend)
            
            # Title
            draw.text((10, 10), "Chart Analysis Legend", fill=(0, 0, 0))
            
            # Legend items
            legend_items = [
                ("Fair Value Gap", self.colors['gap']),
                ("Support Level", self.colors['support']),
                ("Resistance Level", self.colors['resistance']),
                ("Bullish Order Block", self.colors['bullish']),
                ("Bearish Order Block", self.colors['bearish']),
                ("Liquidity Zone", self.colors['liquidity']),
                ("Market Structure", self.colors['structure'])
            ]
            
            y_offset = 40
            for item, color in legend_items:
                # Draw color box
                draw.rectangle([(10, y_offset), (30, y_offset + 15)], fill=color)
                # Draw label
                draw.text((35, y_offset), item, fill=(0, 0, 0))
                y_offset += 25
            
            return legend
            
        except Exception as e:
            print(f"Error creating legend: {e}")
            return Image.new('RGB', (width, height), color='white')

def create_enhanced_analysis_image(original_image: Image.Image, analysis_result) -> Image.Image:
    """
    Create an enhanced analysis image with visual highlights
    """
    try:
        highlighter = VisualHighlighter()
        enhanced_image = highlighter.create_analysis_visualization(original_image, analysis_result)
        return enhanced_image
    except Exception as e:
        print(f"Error creating enhanced analysis image: {e}")
        return original_image

if __name__ == "__main__":
    # Test the visual highlighter
    highlighter = VisualHighlighter()
    legend = highlighter.create_pattern_legend()
    legend.save("analysis_legend.png")
    print("✅ Analysis legend created: analysis_legend.png")
