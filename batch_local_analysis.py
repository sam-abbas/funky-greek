#!/usr/bin/env python3
"""
Batch Local Analysis Script for Stock Chart Analyzer

This script allows you to analyze multiple chart images using LOCAL ANALYSIS ONLY
without starting the web server or making any LLM API calls.

Usage:
    python batch_local_analysis.py <directory_path>
    python batch_local_analysis.py --help

Example:
    python batch_local_analysis.py charts/
    python batch_local_analysis.py --output results/ --pattern "*.png" charts/
    python batch_local_analysis.py --recursive --verbose charts/
"""

import argparse
import json
import sys
import os
import glob
from pathlib import Path
from PIL import Image
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chart_analyzer_enhanced import EnhancedChartAnalyzer
from models import AnalysisResponse

def get_image_files(directory: str, pattern: str = "*.png", recursive: bool = False):
    """
    Get all image files from directory matching the pattern
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (e.g., "*.png", "*.jpg")
        recursive: Whether to search subdirectories
    
    Returns:
        List of image file paths
    """
    image_files = []
    
    if recursive:
        # Search recursively
        search_pattern = os.path.join(directory, "**", pattern)
        image_files = glob.glob(search_pattern, recursive=True)
    else:
        # Search only in specified directory
        search_pattern = os.path.join(directory, pattern)
        image_files = glob.glob(search_pattern)
    
    # Filter to only include actual files (not directories)
    image_files = [f for f in image_files if os.path.isfile(f)]
    
    return sorted(image_files)

def analyze_image_locally(image_path: str, analyzer: EnhancedChartAnalyzer, verbose: bool = False):
    """
    Analyze a single chart image using local analysis only
    
    Args:
        image_path: Path to the chart image file
        analyzer: Pre-configured analyzer instance
        verbose: Whether to print detailed information
    
    Returns:
        AnalysisResponse object or None if failed
    """
    try:
        # Load and validate image
        try:
            image = Image.open(image_path)
            if verbose:
                print(f"📊 Processing: {os.path.basename(image_path)} ({image.size[0]}x{image.size[1]} pixels)")
        except Exception as e:
            print(f"❌ Error: Could not open {image_path}: {str(e)}")
            return None
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
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
        
        if verbose:
            print(f"✅ Completed in {processing_time:.1f}ms - Sentiment: {analysis_result.overall_sentiment}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error analyzing {image_path}: {str(e)}")
        return None

def batch_analyze_images(directory: str, output_dir: str = None, pattern: str = "*.png", 
                        recursive: bool = False, verbose: bool = False):
    """
    Analyze multiple chart images using local analysis only
    
    Args:
        directory: Directory containing chart images
        output_dir: Directory to save results (optional)
        pattern: File pattern to match
        recursive: Whether to search subdirectories
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with analysis results
    """
    # Get image files
    image_files = get_image_files(directory, pattern, recursive)
    
    if not image_files:
        print(f"❌ No image files found in {directory} matching pattern '{pattern}'")
        return {}
    
    print(f"🔍 Found {len(image_files)} image files to analyze")
    print(f"📁 Directory: {directory}")
    print(f"🔍 Pattern: {pattern}")
    print(f"📂 Recursive: {'Yes' if recursive else 'No'}")
    print("=" * 60)
    
    # Create local-only analyzer (no LLM)
    analyzer = EnhancedChartAnalyzer(openai_api_key=None)
    
    # Process each image
    results = {}
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Analyze image
        result = analyze_image_locally(image_path, analyzer, verbose)
        
        if result:
            results[image_path] = result
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 BATCH ANALYSIS COMPLETED")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total: {len(image_files)}")
    
    # Save results if output directory specified
    if output_dir and results:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save individual results
            for image_path, result in results.items():
                filename = os.path.splitext(os.path.basename(image_path))[0]
                output_file = os.path.join(output_dir, f"{filename}_analysis.json")
                
                # Convert to dict for JSON serialization
                result_dict = result.dict()
                
                with open(output_file, 'w') as f:
                    json.dump(result_dict, f, indent=2, default=str)
                
                if verbose:
                    print(f"💾 Saved: {output_file}")
            
            # Save summary
            summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            summary = {
                "timestamp": datetime.now().isoformat(),
                "directory": directory,
                "pattern": pattern,
                "recursive": recursive,
                "total_files": len(image_files),
                "successful": successful,
                "failed": failed,
                "results": {os.path.basename(k): v.dict() for k, v in results.items()}
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"💾 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save results: {str(e)}")
    
    return results

def main():
    """Main function to handle command line arguments and run batch analysis"""
    parser = argparse.ArgumentParser(
        description="Batch Local Stock Chart Analysis (No LLM calls)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_local_analysis.py charts/
  python batch_local_analysis.py --output results/ charts/
  python batch_local_analysis.py --pattern "*.jpg" charts/
  python batch_local_analysis.py --recursive --verbose charts/
  python batch_local_analysis.py --help
        """
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing chart images to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        default="*.png",
        help="File pattern to match (default: *.png)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subdirectories recursively"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed analysis information"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"❌ Error: Directory '{args.directory}' not found or not a directory")
        sys.exit(1)
    
    # Print header
    print("🚀 Enhanced Stock Chart Analyzer - Batch Local Mode")
    print("=" * 60)
    print("🔒 Local Analysis Only - No LLM API calls")
    print("=" * 60)
    
    # Run batch analysis
    results = batch_analyze_images(
        directory=args.directory,
        output_dir=args.output,
        pattern=args.pattern,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    if results:
        print("\n🎉 Batch analysis completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Batch analysis failed or no results generated")
        sys.exit(1)

if __name__ == "__main__":
    main()
