#!/usr/bin/env python3
"""
Local Analysis Only Script for Stock Chart Analyzer

This script allows you to analyze chart images using LOCAL ANALYSIS ONLY
without starting the web server or making any LLM API calls.

Usage:
    python local_analysis.py <image_path>
    python local_analysis.py --help

Example:
    python local_analysis.py chart.png
    python local_analysis.py --output results.json chart.png
"""

import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chart_analyzer_enhanced import EnhancedChartAnalyzer
from models import AnalysisResponse

def analyze_image_locally(image_path: str, output_file: str = None, verbose: bool = False):
    """
    Analyze a chart image using local analysis only
    
    Args:
        image_path: Path to the chart image file
        output_file: Optional output JSON file path
        verbose: Whether to print detailed information
    """
    try:
        # Validate image file
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file '{image_path}' not found")
            return False
        
        # Load and validate image
        try:
            image = Image.open(image_path)
            if verbose:
                print(f"📊 Loaded image: {image.size[0]}x{image.size[1]} pixels, mode: {image.mode}")
        except Exception as e:
            print(f"❌ Error: Could not open image file: {str(e)}")
            return False
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            if verbose:
                print(f"🔄 Converted image to RGB mode")
        
        # Create local-only analyzer (no LLM)
        print("🔍 Starting LOCAL analysis (no LLM calls)...")
        analyzer = EnhancedChartAnalyzer(openai_api_key=None)
        
        # Perform analysis
        start_time = time.time()
        analysis_result = analyzer.analyze_chart(image)
        processing_time = (time.time() - start_time) * 1000
        
        # Create response object
        response = AnalysisResponse(
            success=True,
            message="Local chart analysis completed successfully (no LLM calls)",
            analysis=analysis_result,
            processing_time_ms=processing_time,
            llm_enhanced=False
        )
        
        # Print results
        print(f"\n✅ Analysis completed in {processing_time:.1f}ms")
        print(f"📈 Overall Sentiment: {analysis_result.overall_sentiment}")
        print(f"🎯 Confidence Score: {analysis_result.confidence_score:.1%}")
        print(f"📊 Technical Indicators: {len(analysis_result.indicators)}")
        print(f"🔍 Chart Patterns: {len(analysis_result.patterns)}")
        print(f"💰 Risk Level: {analysis_result.risk_level}")
        
        if verbose:
            print(f"\n📋 Trading Advice: {analysis_result.trading_advice}")
            print(f"⚠️  Warnings: {len(analysis_result.warnings)}")
            print(f"💡 Insights: {len(analysis_result.insights)}")
            
            if analysis_result.support_levels:
                print(f"📉 Support Levels: {len(analysis_result.support_levels)}")
            if analysis_result.resistance_levels:
                print(f"📈 Resistance Levels: {len(analysis_result.resistance_levels)}")
        
        # Save to file if requested
        if output_file:
            try:
                # Convert to dict for JSON serialization
                response_dict = response.dict()
                
                with open(output_file, 'w') as f:
                    json.dump(response_dict, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {output_file}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save to file: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and run analysis"""
    parser = argparse.ArgumentParser(
        description="Local Stock Chart Analysis (No LLM calls)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python local_analysis.py chart.png
  python local_analysis.py --output results.json chart.png
  python local_analysis.py --verbose chart.png
  python local_analysis.py --help
        """
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the chart image file to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path for results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed analysis information"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 Enhanced Stock Chart Analyzer - Local Mode")
    print("=" * 50)
    print("🔒 Local Analysis Only - No LLM API calls")
    print("=" * 50)
    
    # Run analysis
    success = analyze_image_locally(
        image_path=args.image_path,
        output_file=args.output,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
