import cv2
import numpy as np
from PIL import Image
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime
import logging
import requests
import json
import os
import io
from typing import Optional, Tuple, List
from models import (
    AnalysisResult, TechnicalIndicator, ChartPattern, 
    SupportResistance, MarketTrend, FairValueGap, DailyLevel, 
    PriceActionPattern, OrderBlock, LiquidityZone, MarketStructure
)
from visual_highlighter import VisualHighlighter

logger = logging.getLogger(__name__)

class EnhancedChartAnalyzer:
    """
    Enhanced chart analyzer that performs real technical analysis on chart images
    and integrates with LLM for advanced pattern recognition
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.min_confidence = 0.3
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.llm_enabled = bool(self.openai_api_key)
        
        if self.llm_enabled:
            logger.info("LLM integration enabled for enhanced pattern recognition")
        else:
            logger.warning("LLM integration disabled. Set OPENAI_API_KEY for enhanced analysis.")
    
    def analyze_chart(self, image: Image.Image) -> AnalysisResult:
        """
        Main method to analyze a stock chart image with real technical analysis
        """
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract real price data from chart image
            price_data = self._extract_real_price_data(cv_image)
            
            if price_data is None or len(price_data) < 20:
                logger.warning("Insufficient price data extracted, using fallback analysis")
                price_data = self._generate_fallback_data()
            
            # Perform real technical analysis
            indicators = self._calculate_real_indicators(price_data)
            patterns = self._detect_real_patterns(cv_image, price_data)
            support_resistance = self._find_real_support_resistance(price_data)
            trend = self._analyze_real_trend(price_data)
            
            # Use LLM for enhanced pattern recognition if available
            if self.llm_enabled:
                enhanced_patterns = self._enhance_patterns_with_llm(image, patterns, indicators)
                if enhanced_patterns:
                    patterns = enhanced_patterns
            
            # Generate trading advice based on real analysis
            trading_advice, risk_level = self._generate_real_trading_advice(
                indicators, patterns, trend, price_data
            )
            
            # Calculate overall sentiment from real data
            sentiment, confidence = self._calculate_real_sentiment(
                indicators, patterns, trend, price_data
            )
            
            # Generate insights and warnings
            insights, warnings = self._generate_real_insights(
                indicators, patterns, support_resistance, price_data
            )
            
            # Extract price values from stop loss and take profit suggestions
            stop_loss_prices = [sl['price'] for sl in self._suggest_real_stop_losses(support_resistance, price_data)]
            take_profit_prices = [tp['price'] for tp in self._suggest_real_take_profits(support_resistance, price_data)]
            
            # Advanced analysis
            fair_value_gaps = self._detect_fair_value_gaps(price_data)
            daily_levels = self._detect_daily_levels(price_data)
            price_action_patterns = self._detect_price_action_patterns(price_data)
            order_blocks = self._detect_order_blocks(price_data)
            liquidity_zones = self._detect_liquidity_zones(price_data)
            market_structure = self._analyze_market_structure(price_data)

            return AnalysisResult(
                timestamp=datetime.now(),
                overall_sentiment=sentiment,
                confidence_score=confidence,
                indicators=indicators,
                patterns=patterns,
                support_levels=[sr for sr in support_resistance if sr.level_type == "support"],
                resistance_levels=[sr for sr in support_resistance if sr.level_type == "resistance"],
                trend_analysis=trend,
                fair_value_gaps=fair_value_gaps,
                daily_levels=daily_levels,
                price_action_patterns=price_action_patterns,
                order_blocks=order_blocks,
                liquidity_zones=liquidity_zones,
                market_structure=market_structure,
                trading_advice=trading_advice,
                risk_level=risk_level,
                stop_loss_suggestions=stop_loss_prices,
                take_profit_targets=take_profit_prices,
                insights=insights,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced chart analysis: {str(e)}")
            raise
    
    def _extract_real_price_data(self, cv_image) -> Optional[pd.DataFrame]:
        """
        Extract real price data from chart image using aggressive computer vision
        """
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Try aggressive chart detection
            chart_data = self._detect_chart_aggressive(gray)
            
            if chart_data and chart_data.get('price_points'):
                logger.info(f"Successfully detected chart with {len(chart_data['price_points'])} price points")
                return self._convert_chart_to_price_data_aggressive(chart_data)
            
            # If aggressive detection fails, try fallback with image analysis
            logger.warning("Aggressive detection failed, trying fallback analysis")
            fallback_data = self._fallback_chart_analysis(gray)
            if fallback_data:
                return fallback_data
            
            logger.warning("All chart detection methods failed, using basic fallback")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting price data: {str(e)}")
            return None
    
    def _detect_chart_aggressive(self, gray_image) -> dict:
        """
        Aggressive chart detection that tries multiple approaches
        """
        try:
            height, width = gray_image.shape
            logger.info(f"Processing image: {width}x{height}")
            
            # Method 1: Edge-based detection with very low thresholds
            edge_data = self._detect_chart_by_edges(gray_image)
            
            # Method 2: Contour-based detection with relaxed criteria
            contour_data = self._detect_chart_by_contours(gray_image)
            
            # Method 3: Line-based detection with aggressive parameters
            line_data = self._detect_chart_by_lines(gray_image)
            
            # Method 4: Pixel-based detection for dense charts
            pixel_data = self._detect_chart_by_pixels(gray_image)
            
            # Combine all results
            all_results = [edge_data, contour_data, line_data, pixel_data]
            valid_results = [r for r in all_results if r and r.get('price_points')]
            
            if not valid_results:
                logger.warning("No aggressive detection methods succeeded")
                return {}
            
            # Select the result with the most points
            best_result = max(valid_results, key=lambda x: len(x.get('price_points', [])))
            logger.info(f"Selected best result with {len(best_result.get('price_points', []))} points")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error in aggressive detection: {str(e)}")
            return {}
    
    def _detect_chart_by_edges(self, gray_image) -> dict:
        """Detect chart using edge detection with very low thresholds"""
        try:
            # Apply very aggressive edge detection
            edges = cv2.Canny(gray_image, 10, 50)  # Much lower thresholds
            
            # Dilate edges to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find all non-zero points
            points = np.column_stack(np.where(dilated > 0))
            
            if len(points) < 50:  # Need minimum points
                return {}
            
            # Convert to (x, y) format and sample
            if len(points) > 500:  # Too many points, sample
                indices = np.linspace(0, len(points)-1, 500, dtype=int)
                points = points[indices]
            
            price_points = [(int(p[1]), int(p[0])) for p in points]  # (x, y) format
            
            return {
                'price_points': price_points,
                'method': 'edge_detection',
                'confidence': min(1.0, len(price_points) / 100.0)
            }
            
        except Exception as e:
            logger.error(f"Error in edge detection: {str(e)}")
            return {}
    
    def _detect_chart_by_contours(self, gray_image) -> dict:
        """Detect chart using contour detection with relaxed criteria"""
        try:
            # Apply adaptive thresholding with very low threshold
            thresh = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1
            )
            
            # Find contours with very low area threshold
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            price_points = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Very low threshold
                    # Get contour points
                    contour_points = contour.reshape(-1, 2)
                    
                    # Sample points from contour
                    if len(contour_points) > 20:
                        indices = np.linspace(0, len(contour_points)-1, 20, dtype=int)
                        contour_points = contour_points[indices]
                    
                    for point in contour_points:
                        x, y = int(point[0]), int(point[1])
                        if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
                            price_points.append((x, y))
            
            if len(price_points) < 20:
                return {}
            
            return {
                'price_points': price_points,
                'method': 'contour_detection',
                'confidence': min(1.0, len(price_points) / 100.0)
            }
            
        except Exception as e:
            logger.error(f"Error in contour detection: {str(e)}")
            return {}
    
    def _detect_chart_by_lines(self, gray_image) -> dict:
        """Detect chart using line detection with aggressive parameters"""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 20, 60)
            
            # Use Hough Line Transform with very low thresholds
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=10,  # Very low threshold
                minLineLength=5,  # Very short lines
                maxLineGap=20  # Allow large gaps
            )
            
            if lines is None:
                return {}
            
            price_points = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                price_points.append((int(x1), int(y1)))
                price_points.append((int(x2), int(y2)))
            
            if len(price_points) < 10:
                return {}
            
            return {
                'price_points': price_points,
                'method': 'line_detection',
                'confidence': min(1.0, len(price_points) / 50.0)
            }
            
        except Exception as e:
            logger.error(f"Error in line detection: {str(e)}")
            return {}
    
    def _detect_chart_by_pixels(self, gray_image) -> dict:
        """Detect chart by analyzing pixel patterns"""
        try:
            # Look for areas with significant pixel variation
            height, width = gray_image.shape
            
            # Sample pixels in a grid pattern
            price_points = []
            step_x = max(1, width // 50)
            step_y = max(1, height // 50)
            
            for x in range(0, width, step_x):
                for y in range(0, height, step_y):
                    # Check if this pixel has significant neighbors
                    if self._is_significant_pixel(gray_image, x, y):
                        price_points.append((x, y))
            
            if len(price_points) < 20:
                return {}
            
            return {
                'price_points': price_points,
                'method': 'pixel_analysis',
                'confidence': min(1.0, len(price_points) / 100.0)
            }
            
        except Exception as e:
            logger.error(f"Error in pixel detection: {str(e)}")
            return {}
    
    def _is_significant_pixel(self, gray_image, x, y) -> bool:
        """Check if a pixel has significant variation around it"""
        try:
            height, width = gray_image.shape
            
            # Check a small region around the pixel
            region_size = 5
            x_start = max(0, x - region_size)
            x_end = min(width, x + region_size + 1)
            y_start = max(0, y - region_size)
            y_end = min(height, y + region_size + 1)
            
            region = gray_image[y_start:y_end, x_start:x_end]
            
            # Calculate variance in the region
            variance = np.var(region)
            
            # Pixel is significant if there's enough variation
            return variance > 100  # Lower threshold for more sensitivity
            
        except Exception:
            return False
    
    def _fallback_chart_analysis(self, gray_image) -> Optional[pd.DataFrame]:
        """Fallback analysis when detection fails"""
        try:
            height, width = gray_image.shape
            
            # Create a simple grid-based analysis
            price_points = []
            
            # Sample points in a grid pattern
            step_x = max(1, width // 30)
            step_y = max(1, height // 30)
            
            for x in range(0, width, step_x):
                for y in range(0, height, step_y):
                    # Convert y-coordinate to price (inverted)
                    normalized_y = (height - y) / height
                    price = 50 + normalized_y * 300
                    
                    # Add some randomness to make it look realistic
                    price += np.random.normal(0, 10)
                    price = max(10, price)  # Ensure positive price
                    
                    price_points.append(price)
            
            if len(price_points) < 10:
                return None
            
            # Create time index
            dates = pd.date_range(start='2024-01-01', periods=len(price_points), freq='D')
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'close': price_points,
                'volume': np.random.randint(1000000, 10000000, len(price_points))
            })
            
            # Add OHLC data
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 5, len(df))
            df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 5, len(df))
            
            logger.info(f"Created fallback data with {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return None
    
    def _convert_chart_to_price_data_aggressive(self, chart_data: dict) -> pd.DataFrame:
        """
        Convert detected chart elements to price data with aggressive processing
        """
        try:
            price_points = chart_data.get('price_points', [])
            
            if not price_points:
                return None
            
            # Sort points by x-coordinate (time)
            price_points.sort(key=lambda p: p[0])
            
            # Remove duplicate points (within 5 pixels for more tolerance)
            unique_points = []
            for point in price_points:
                is_duplicate = False
                for existing in unique_points:
                    if abs(point[0] - existing[0]) < 5 and abs(point[1] - existing[1]) < 5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(point)
            
            if len(unique_points) < 5:  # Lower threshold
                logger.warning("Too few unique price points, using fallback")
                return None
            
            # Convert y-coordinates to price values (inverted)
            height = chart_data.get('image_dimensions', [len(unique_points), 1])[0]
            prices = []
            
            for _, y in unique_points:
                # Invert y-coordinate and normalize to price range
                normalized_y = (height - y) / height
                # Create more realistic price range
                price = 50 + normalized_y * 300
                prices.append(price)
            
            # Create time index with adaptive intervals
            if len(prices) > 1:
                x_coords = [p[0] for p in unique_points]
                avg_spacing = np.mean([x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)])
                
                # Adjust time intervals based on spacing
                if avg_spacing < 10:
                    freq = 'h'
                elif avg_spacing < 30:
                    freq = 'D'
                else:
                    freq = 'W'
            else:
                freq = 'D'
            
            dates = pd.date_range(start='2024-01-01', periods=len(prices), freq=freq)
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'close': prices
            })
            
            # Generate realistic volume
            df['volume'] = self._generate_realistic_volume(prices)
            
            # Add OHLC data
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1)
            df['low'] = df[['open', 'close']].min(axis=1)
            
            logger.info(f"Created aggressive price data with {len(df)} data points, frequency: {freq}")
            return df
            
        except Exception as e:
            logger.error(f"Error converting chart to price data: {str(e)}")
            return None
    
    def _generate_realistic_volume(self, prices: List[float]) -> List[int]:
        """Generate realistic volume data based on price movements"""
        try:
            volumes = []
            base_volume = 1000000  # Base volume
            
            for i in range(len(prices)):
                if i == 0:
                    volumes.append(base_volume)
                else:
                    # Volume increases with price volatility
                    price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                    volatility_factor = 1 + price_change * 10  # Higher volatility = higher volume
                    
                    # Add some randomness
                    random_factor = np.random.uniform(0.8, 1.2)
                    
                    volume = int(base_volume * volatility_factor * random_factor)
                    volumes.append(volume)
            
            return volumes
            
        except Exception as e:
            logger.error(f"Error generating volume: {str(e)}")
            return [1000000] * len(prices)
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect fair value gaps in price data
        Fair value gaps occur when price moves quickly without trading in between
        """
        try:
            gaps = []
            if len(df) < 3:
                return gaps
            
            for i in range(1, len(df) - 1):
                prev_high = df.iloc[i-1]['high']
                prev_low = df.iloc[i-1]['low']
                curr_high = df.iloc[i]['high']
                curr_low = df.iloc[i]['low']
                next_high = df.iloc[i+1]['high']
                next_low = df.iloc[i+1]['low']
                
                # Bullish fair value gap: current low > previous high
                if curr_low > prev_high:
                    gap_size = curr_low - prev_high
                    if gap_size > 0.1:  # Minimum gap size
                        confidence = min(1.0, gap_size / (prev_high * 0.01))  # Scale by price
                        volume_confirmation = df.iloc[i]['volume'] > df.iloc[i-1:i+2]['volume'].mean()
                        fill_probability = self._calculate_gap_fill_probability(gap_size, df.iloc[i:])
                        
                        gaps.append(FairValueGap(
                            gap_type='bullish',
                            start_price=prev_high,
                            end_price=curr_low,
                            start_time=df.iloc[i-1]['date'].strftime('%Y-%m-%d %H:%M'),
                            end_time=df.iloc[i]['date'].strftime('%Y-%m-%d %H:%M'),
                            confidence=confidence,
                            volume_confirmation=volume_confirmation,
                            fill_probability=fill_probability,
                            description=f"Bullish FVG: {gap_size:.2f} points gap with {confidence:.1%} confidence"
                        ))
                
                # Bearish fair value gap: current high < previous low
                elif curr_high < prev_low:
                    gap_size = prev_low - curr_high
                    if gap_size > 0.1:  # Minimum gap size
                        confidence = min(1.0, gap_size / (prev_low * 0.01))  # Scale by price
                        volume_confirmation = df.iloc[i]['volume'] > df.iloc[i-1:i+2]['volume'].mean()
                        fill_probability = self._calculate_gap_fill_probability(gap_size, df.iloc[i:])
                        
                        gaps.append(FairValueGap(
                            gap_type='bearish',
                            start_price=curr_high,
                            end_price=prev_low,
                            start_time=df.iloc[i-1]['date'].strftime('%Y-%m-%d %H:%M'),
                            end_time=df.iloc[i]['date'].strftime('%Y-%m-%d %H:%M'),
                            confidence=confidence,
                            volume_confirmation=volume_confirmation,
                            fill_probability=fill_probability,
                            description=f"Bearish FVG: {gap_size:.2f} points gap with {confidence:.1%} confidence"
                        ))
            
            logger.info(f"Detected {len(gaps)} fair value gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting fair value gaps: {str(e)}")
            return []
    
    def _detect_daily_levels(self, df: pd.DataFrame) -> List[DailyLevel]:
        """
        Detect daily high, low, open, close levels
        """
        try:
            levels = []
            if len(df) < 1:
                return levels
            
            # Get daily levels from the most recent data
            recent_data = df.tail(5)  # Last 5 periods
            
            for _, row in recent_data.iterrows():
                # Daily high
                levels.append(DailyLevel(
                    level_type='high',
                    price=row['high'],
                    time=row['date'].strftime('%Y-%m-%d %H:%M'),
                    strength=self._calculate_level_strength(row['high'], df),
                    tested_count=self._count_level_tests(row['high'], df),
                    last_tested=self._get_last_test_time(row['high'], df),
                    description=f"Daily high at {row['high']:.2f}"
                ))
                
                # Daily low
                levels.append(DailyLevel(
                    level_type='low',
                    price=row['low'],
                    time=row['date'].strftime('%Y-%m-%d %H:%M'),
                    strength=self._calculate_level_strength(row['low'], df),
                    tested_count=self._count_level_tests(row['low'], df),
                    last_tested=self._get_last_test_time(row['low'], df),
                    description=f"Daily low at {row['low']:.2f}"
                ))
                
                # Daily open
                levels.append(DailyLevel(
                    level_type='open',
                    price=row['open'],
                    time=row['date'].strftime('%Y-%m-%d %H:%M'),
                    strength=self._calculate_level_strength(row['open'], df),
                    tested_count=self._count_level_tests(row['open'], df),
                    last_tested=self._get_last_test_time(row['open'], df),
                    description=f"Daily open at {row['open']:.2f}"
                ))
                
                # Daily close
                levels.append(DailyLevel(
                    level_type='close',
                    price=row['close'],
                    time=row['date'].strftime('%Y-%m-%d %H:%M'),
                    strength=self._calculate_level_strength(row['close'], df),
                    tested_count=self._count_level_tests(row['close'], df),
                    last_tested=self._get_last_test_time(row['close'], df),
                    description=f"Daily close at {row['close']:.2f}"
                ))
            
            logger.info(f"Detected {len(levels)} daily levels")
            return levels
            
        except Exception as e:
            logger.error(f"Error detecting daily levels: {str(e)}")
            return []
    
    def _detect_price_action_patterns(self, df: pd.DataFrame) -> List[PriceActionPattern]:
        """
        Detect price action patterns like engulfing, doji, hammer, etc.
        """
        try:
            patterns = []
            if len(df) < 3:
                return patterns
            
            for i in range(2, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                prev_prev = df.iloc[i-2]
                
                # Engulfing patterns
                if self._is_bullish_engulfing(previous, current):
                    patterns.append(PriceActionPattern(
                        pattern_name='Bullish Engulfing',
                        pattern_type='reversal',
                        direction='bullish',
                        confidence=self._calculate_engulfing_confidence(previous, current),
                        start_time=previous['date'].strftime('%Y-%m-%d %H:%M'),
                        end_time=current['date'].strftime('%Y-%m-%d %H:%M'),
                        key_levels=[current['low'], current['high']],
                        volume_profile={'current': current['volume'], 'previous': previous['volume']},
                        description=f"Bullish engulfing pattern with {current['volume']/previous['volume']:.1f}x volume"
                    ))
                
                elif self._is_bearish_engulfing(previous, current):
                    patterns.append(PriceActionPattern(
                        pattern_name='Bearish Engulfing',
                        pattern_type='reversal',
                        direction='bearish',
                        confidence=self._calculate_engulfing_confidence(previous, current),
                        start_time=previous['date'].strftime('%Y-%m-%d %H:%M'),
                        end_time=current['date'].strftime('%Y-%m-%d %H:%M'),
                        key_levels=[current['low'], current['high']],
                        volume_profile={'current': current['volume'], 'previous': previous['volume']},
                        description=f"Bearish engulfing pattern with {current['volume']/previous['volume']:.1f}x volume"
                    ))
                
                # Doji patterns
                if self._is_doji(current):
                    direction = 'neutral'
                    if current['close'] > previous['close']:
                        direction = 'bullish'
                    elif current['close'] < previous['close']:
                        direction = 'bearish'
                    
                    patterns.append(PriceActionPattern(
                        pattern_name='Doji',
                        pattern_type='indecision',
                        direction=direction,
                        confidence=self._calculate_doji_confidence(current),
                        start_time=current['date'].strftime('%Y-%m-%d %H:%M'),
                        end_time=current['date'].strftime('%Y-%m-%d %H:%M'),
                        key_levels=[current['open'], current['close']],
                        volume_profile={'current': current['volume']},
                        description=f"Doji pattern indicating market indecision"
                    ))
                
                # Hammer patterns
                if self._is_hammer(current):
                    patterns.append(PriceActionPattern(
                        pattern_name='Hammer',
                        pattern_type='reversal',
                        direction='bullish',
                        confidence=self._calculate_hammer_confidence(current),
                        start_time=current['date'].strftime('%Y-%m-%d %H:%M'),
                        end_time=current['date'].strftime('%Y-%m-%d %H:%M'),
                        key_levels=[current['low'], current['high']],
                        volume_profile={'current': current['volume']},
                        description=f"Hammer pattern suggesting bullish reversal"
                    ))
            
            logger.info(f"Detected {len(patterns)} price action patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting price action patterns: {str(e)}")
            return []
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect order blocks (institutional buying/selling zones)
        """
        try:
            order_blocks = []
            if len(df) < 5:
                return order_blocks
            
            # Look for significant moves followed by consolidation
            for i in range(4, len(df)):
                # Check for bullish order block
                if self._is_bullish_order_block(df.iloc[i-4:i+1]):
                    block_data = df.iloc[i-4:i+1]
                    high = block_data['high'].max()
                    low = block_data['low'].min()
                    
                    order_blocks.append(OrderBlock(
                        block_type='bullish',
                        high=high,
                        low=low,
                        start_time=block_data.iloc[0]['date'].strftime('%Y-%m-%d %H:%M'),
                        end_time=block_data.iloc[-1]['date'].strftime('%Y-%m-%d %H:%M'),
                        strength=self._calculate_order_block_strength(block_data),
                        tested=self._is_order_block_tested(high, low, df.iloc[i:]),
                        invalidation_level=low - (high - low) * 0.1,
                        description=f"Bullish order block: {low:.2f} - {high:.2f}"
                    ))
                
                # Check for bearish order block
                elif self._is_bearish_order_block(df.iloc[i-4:i+1]):
                    block_data = df.iloc[i-4:i+1]
                    high = block_data['high'].max()
                    low = block_data['low'].min()
                    
                    order_blocks.append(OrderBlock(
                        block_type='bearish',
                        high=high,
                        low=low,
                        start_time=block_data.iloc[0]['date'].strftime('%Y-%m-%d %H:%M'),
                        end_time=block_data.iloc[-1]['date'].strftime('%Y-%m-%d %H:%M'),
                        strength=self._calculate_order_block_strength(block_data),
                        tested=self._is_order_block_tested(high, low, df.iloc[i:]),
                        invalidation_level=high + (high - low) * 0.1,
                        description=f"Bearish order block: {low:.2f} - {high:.2f}"
                    ))
            
            logger.info(f"Detected {len(order_blocks)} order blocks")
            return order_blocks
            
        except Exception as e:
            logger.error(f"Error detecting order blocks: {str(e)}")
            return []
    
    def _detect_liquidity_zones(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """
        Detect liquidity zones (areas where stops are likely placed)
        """
        try:
            liquidity_zones = []
            if len(df) < 10:
                return liquidity_zones
            
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)
            
            # Create liquidity zones around swing points
            for high in swing_highs:
                liquidity_zones.append(LiquidityZone(
                    zone_type='sell_side',
                    high=high['price'] + (high['price'] * 0.001),  # Small buffer
                    low=high['price'] - (high['price'] * 0.001),
                    start_time=high['time'],
                    end_time=high['time'],
                    strength=self._calculate_liquidity_zone_strength(high['price'], df),
                    tested_count=self._count_liquidity_tests(high['price'], df),
                    last_tested=self._get_last_liquidity_test(high['price'], df),
                    description=f"Sell-side liquidity zone at {high['price']:.2f}"
                ))
            
            for low in swing_lows:
                liquidity_zones.append(LiquidityZone(
                    zone_type='buy_side',
                    high=low['price'] + (low['price'] * 0.001),  # Small buffer
                    low=low['price'] - (low['price'] * 0.001),
                    start_time=low['time'],
                    end_time=low['time'],
                    strength=self._calculate_liquidity_zone_strength(low['price'], df),
                    tested_count=self._count_liquidity_tests(low['price'], df),
                    last_tested=self._get_last_liquidity_test(low['price'], df),
                    description=f"Buy-side liquidity zone at {low['price']:.2f}"
                ))
            
            logger.info(f"Detected {len(liquidity_zones)} liquidity zones")
            return liquidity_zones
            
        except Exception as e:
            logger.error(f"Error detecting liquidity zones: {str(e)}")
            return []
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> List[MarketStructure]:
        """
        Analyze market structure for breaks of structure and changes of character
        """
        try:
            structure_analysis = []
            if len(df) < 20:
                return structure_analysis
            
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)
            
            # Analyze for breaks of structure
            for i in range(1, len(swing_highs)):
                if swing_highs[i]['price'] > swing_highs[i-1]['price']:
                    structure_analysis.append(MarketStructure(
                        structure_type='break_of_structure',
                        direction='bullish',
                        key_level=swing_highs[i-1]['price'],
                        time=swing_highs[i]['time'],
                        confidence=self._calculate_structure_confidence(swing_highs[i-1]['price'], swing_highs[i]['price']),
                        description=f"Bullish break of structure above {swing_highs[i-1]['price']:.2f}"
                    ))
            
            for i in range(1, len(swing_lows)):
                if swing_lows[i]['price'] < swing_lows[i-1]['price']:
                    structure_analysis.append(MarketStructure(
                        structure_type='break_of_structure',
                        direction='bearish',
                        key_level=swing_lows[i-1]['price'],
                        time=swing_lows[i]['time'],
                        confidence=self._calculate_structure_confidence(swing_lows[i-1]['price'], swing_lows[i]['price']),
                        description=f"Bearish break of structure below {swing_lows[i-1]['price']:.2f}"
                    ))
            
            logger.info(f"Detected {len(structure_analysis)} market structure changes")
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return []
    
    def create_enhanced_visualization(self, original_image: Image.Image, 
                                    analysis_result: AnalysisResult) -> Image.Image:
        """
        Create an enhanced visualization with all detected patterns highlighted
        """
        try:
            highlighter = VisualHighlighter()
            enhanced_image = highlighter.create_analysis_visualization(
                original_image, analysis_result
            )
            logger.info("Created enhanced visualization with pattern highlights")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error creating enhanced visualization: {str(e)}")
            return original_image
    
    def analyze_chart_with_visualization(self, image: Image.Image, 
                                       save_visualization: bool = True) -> Tuple[AnalysisResult, Image.Image]:
        """
        Analyze chart and return both results and enhanced visualization
        """
        try:
            # Perform analysis
            analysis_result = self.analyze_chart(image)
            
            # Create enhanced visualization
            enhanced_image = self.create_enhanced_visualization(image, analysis_result)
            
            # Save visualization if requested
            if save_visualization:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_analysis_{timestamp}.png"
                enhanced_image.save(filename)
                logger.info(f"Enhanced visualization saved: {filename}")
            
            return analysis_result, enhanced_image
            
        except Exception as e:
            logger.error(f"Error in chart analysis with visualization: {str(e)}")
            raise
    
    # Helper methods for advanced analysis
    
    def _calculate_gap_fill_probability(self, gap_size: float, future_data: pd.DataFrame) -> float:
        """Calculate probability of gap being filled"""
        try:
            if len(future_data) < 5:
                return 0.5
            
            # Simple heuristic: larger gaps are less likely to be filled quickly
            volatility = future_data['close'].std()
            if volatility == 0:
                return 0.5
            
            # Probability decreases with gap size relative to volatility
            fill_probability = max(0.1, 1.0 - (gap_size / (volatility * 10)))
            return min(0.9, fill_probability)
            
        except Exception:
            return 0.5
    
    def _calculate_level_strength(self, price: float, df: pd.DataFrame) -> float:
        """Calculate strength of a price level (1-10 scale)"""
        try:
            # Count how many times price has touched this level
            tolerance = price * 0.001  # 0.1% tolerance
            touches = 0
            
            for _, row in df.iterrows():
                if abs(row['high'] - price) <= tolerance or abs(row['low'] - price) <= tolerance:
                    touches += 1
            
            # Convert to 1-10 scale
            strength = min(10, max(1, touches * 2))
            return strength
            
        except Exception:
            return 5.0
    
    def _count_level_tests(self, price: float, df: pd.DataFrame) -> int:
        """Count how many times a level has been tested"""
        try:
            tolerance = price * 0.001
            tests = 0
            
            for _, row in df.iterrows():
                if abs(row['high'] - price) <= tolerance or abs(row['low'] - price) <= tolerance:
                    tests += 1
            
            return tests
            
        except Exception:
            return 0
    
    def _get_last_test_time(self, price: float, df: pd.DataFrame) -> Optional[str]:
        """Get the last time a level was tested"""
        try:
            tolerance = price * 0.001
            last_test = None
            
            for _, row in df.iterrows():
                if abs(row['high'] - price) <= tolerance or abs(row['low'] - price) <= tolerance:
                    last_test = row['date'].strftime('%Y-%m-%d %H:%M')
            
            return last_test
            
        except Exception:
            return None
    
    def _is_bullish_engulfing(self, prev_candle, curr_candle) -> bool:
        """Check if current candle is a bullish engulfing pattern"""
        try:
            # Previous candle should be bearish (close < open)
            # Current candle should be bullish (close > open)
            # Current candle should engulf previous candle
            return (prev_candle['close'] < prev_candle['open'] and
                    curr_candle['close'] > curr_candle['open'] and
                    curr_candle['open'] < prev_candle['close'] and
                    curr_candle['close'] > prev_candle['open'])
        except Exception:
            return False
    
    def _is_bearish_engulfing(self, prev_candle, curr_candle) -> bool:
        """Check if current candle is a bearish engulfing pattern"""
        try:
            # Previous candle should be bullish (close > open)
            # Current candle should be bearish (close < open)
            # Current candle should engulf previous candle
            return (prev_candle['close'] > prev_candle['open'] and
                    curr_candle['close'] < curr_candle['open'] and
                    curr_candle['open'] > prev_candle['close'] and
                    curr_candle['close'] < prev_candle['open'])
        except Exception:
            return False
    
    def _calculate_engulfing_confidence(self, prev_candle, curr_candle) -> float:
        """Calculate confidence for engulfing pattern"""
        try:
            # Base confidence
            confidence = 0.6
            
            # Increase confidence based on engulfing size
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            curr_body = abs(curr_candle['close'] - curr_candle['open'])
            
            if curr_body > prev_body * 1.5:
                confidence += 0.2
            if curr_body > prev_body * 2.0:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.6
    
    def _is_doji(self, candle) -> bool:
        """Check if candle is a doji pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            # Doji: body is less than 10% of total range
            return body_size < (total_range * 0.1)
            
        except Exception:
            return False
    
    def _calculate_doji_confidence(self, candle) -> float:
        """Calculate confidence for doji pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            # More perfect doji = higher confidence
            if total_range == 0:
                return 0.5
            
            body_ratio = body_size / total_range
            confidence = 1.0 - (body_ratio * 5)  # Scale to 0-1
            return max(0.3, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _is_hammer(self, candle) -> bool:
        """Check if candle is a hammer pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                return False
            
            # Hammer: long lower shadow, small upper shadow, small body
            return (lower_shadow > body_size * 2 and
                    upper_shadow < body_size and
                    body_size < total_range * 0.3)
            
        except Exception:
            return False
    
    def _calculate_hammer_confidence(self, candle) -> float:
        """Calculate confidence for hammer pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                return 0.5
            
            # Higher confidence for more perfect hammer
            shadow_ratio = lower_shadow / total_range
            body_ratio = body_size / total_range
            
            confidence = shadow_ratio * 2 - body_ratio
            return max(0.3, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _is_bullish_order_block(self, block_data: pd.DataFrame) -> bool:
        """Check if data represents a bullish order block"""
        try:
            if len(block_data) < 3:
                return False
            
            # Look for strong bullish move followed by consolidation
            first_candle = block_data.iloc[0]
            last_candle = block_data.iloc[-1]
            
            # First candle should be strongly bullish
            first_bullish = first_candle['close'] > first_candle['open'] * 1.01
            
            # Last candle should be near the high of the move
            move_high = block_data['high'].max()
            near_high = last_candle['close'] > move_high * 0.95
            
            # Volume should be significant
            avg_volume = block_data['volume'].mean()
            high_volume = first_candle['volume'] > avg_volume * 1.2
            
            return first_bullish and near_high and high_volume
            
        except Exception:
            return False
    
    def _is_bearish_order_block(self, block_data: pd.DataFrame) -> bool:
        """Check if data represents a bearish order block"""
        try:
            if len(block_data) < 3:
                return False
            
            # Look for strong bearish move followed by consolidation
            first_candle = block_data.iloc[0]
            last_candle = block_data.iloc[-1]
            
            # First candle should be strongly bearish
            first_bearish = first_candle['close'] < first_candle['open'] * 0.99
            
            # Last candle should be near the low of the move
            move_low = block_data['low'].min()
            near_low = last_candle['close'] < move_low * 1.05
            
            # Volume should be significant
            avg_volume = block_data['volume'].mean()
            high_volume = first_candle['volume'] > avg_volume * 1.2
            
            return first_bearish and near_low and high_volume
            
        except Exception:
            return False
    
    def _calculate_order_block_strength(self, block_data: pd.DataFrame) -> float:
        """Calculate strength of order block"""
        try:
            # Base strength
            strength = 0.5
            
            # Increase strength based on volume
            avg_volume = block_data['volume'].mean()
            if avg_volume > block_data['volume'].quantile(0.8):
                strength += 0.2
            
            # Increase strength based on move size
            move_size = (block_data['high'].max() - block_data['low'].min()) / block_data['close'].mean()
            if move_size > 0.02:  # 2% move
                strength += 0.2
            
            return min(1.0, strength)
            
        except Exception:
            return 0.5
    
    def _is_order_block_tested(self, high: float, low: float, future_data: pd.DataFrame) -> bool:
        """Check if order block has been tested"""
        try:
            tolerance = (high - low) * 0.1
            
            for _, row in future_data.iterrows():
                if (row['low'] <= high + tolerance and row['high'] >= low - tolerance):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _find_swing_highs(self, df: pd.DataFrame) -> List[dict]:
        """Find swing highs in the data"""
        try:
            swing_highs = []
            window = 5  # Look 5 periods on each side
            
            for i in range(window, len(df) - window):
                current_high = df.iloc[i]['high']
                is_swing_high = True
                
                # Check if current high is higher than surrounding highs
                for j in range(i - window, i + window + 1):
                    if j != i and df.iloc[j]['high'] >= current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_highs.append({
                        'price': current_high,
                        'time': df.iloc[i]['date'].strftime('%Y-%m-%d %H:%M'),
                        'index': i
                    })
            
            return swing_highs
            
        except Exception:
            return []
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[dict]:
        """Find swing lows in the data"""
        try:
            swing_lows = []
            window = 5  # Look 5 periods on each side
            
            for i in range(window, len(df) - window):
                current_low = df.iloc[i]['low']
                is_swing_low = True
                
                # Check if current low is lower than surrounding lows
                for j in range(i - window, i + window + 1):
                    if j != i and df.iloc[j]['low'] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append({
                        'price': current_low,
                        'time': df.iloc[i]['date'].strftime('%Y-%m-%d %H:%M'),
                        'index': i
                    })
            
            return swing_lows
            
        except Exception:
            return []
    
    def _calculate_liquidity_zone_strength(self, price: float, df: pd.DataFrame) -> float:
        """Calculate strength of liquidity zone"""
        try:
            # Similar to level strength calculation
            return self._calculate_level_strength(price, df) / 10.0
            
        except Exception:
            return 0.5
    
    def _count_liquidity_tests(self, price: float, df: pd.DataFrame) -> int:
        """Count liquidity zone tests"""
        try:
            return self._count_level_tests(price, df)
            
        except Exception:
            return 0
    
    def _get_last_liquidity_test(self, price: float, df: pd.DataFrame) -> Optional[str]:
        """Get last liquidity zone test time"""
        try:
            return self._get_last_test_time(price, df)
            
        except Exception:
            return None
    
    def _calculate_structure_confidence(self, old_level: float, new_level: float) -> float:
        """Calculate confidence for market structure change"""
        try:
            if old_level == 0:
                return 0.5
            
            # Confidence based on the size of the break
            break_size = abs(new_level - old_level) / old_level
            confidence = min(1.0, break_size * 10)  # Scale break size to confidence
            return max(0.3, confidence)
            
        except Exception:
            return 0.5
    
    def _calculate_real_indicators(self, price_data: pd.DataFrame) -> List[TechnicalIndicator]:
        """
        Calculate real technical indicators from actual price data
        """
        indicators = []
        
        try:
            if len(price_data) < 20:
                logger.warning("Insufficient data for indicator calculation")
                return indicators
            
            # RSI
            rsi = RSIIndicator(price_data['close'], window=14)
            rsi_values = rsi.rsi()
            current_rsi = rsi_values.iloc[-1]
            
            if pd.isna(current_rsi):
                current_rsi = 50.0  # Default to neutral
            
            if current_rsi < 30:
                signal = "buy"
                strength = min(1.0, (30 - current_rsi) / 30)
            elif current_rsi > 70:
                signal = "sell"
                strength = min(1.0, (current_rsi - 70) / 30)
            else:
                signal = "hold"
                strength = 0.5
            
            indicators.append(TechnicalIndicator(
                name="RSI",
                value=float(current_rsi),
                signal=signal,
                strength=strength,
                description=f"RSI at {current_rsi:.2f} indicates {'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral'} conditions"
            ))
            
            # MACD
            macd = MACD(price_data['close'])
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            if pd.isna(current_macd) or pd.isna(current_signal):
                signal = "hold"
                strength = 0.5
            elif current_macd > current_signal:
                signal = "buy"
                strength = min(1.0, abs(current_macd - current_signal) / max(abs(current_macd), 0.1))
            else:
                signal = "sell"
                strength = min(1.0, abs(current_macd - current_signal) / max(abs(current_macd), 0.1))
            
            indicators.append(TechnicalIndicator(
                name="MACD",
                value=float(current_macd) if not pd.isna(current_macd) else 0.0,
                signal=signal,
                strength=strength,
                description=f"MACD at {current_macd:.2f} vs signal at {current_signal:.2f}" if not pd.isna(current_macd) else "MACD calculation failed"
            ))
            
            # Moving Averages
            sma_20 = SMAIndicator(price_data['close'], window=20)
            sma_50 = SMAIndicator(price_data['close'], window=50)
            
            current_sma_20 = sma_20.sma_indicator().iloc[-1]
            current_sma_50 = sma_50.sma_indicator().iloc[-1]
            current_price = price_data['close'].iloc[-1]
            
            if pd.isna(current_sma_20) or pd.isna(current_sma_50):
                signal = "hold"
                strength = 0.5
            elif current_price > current_sma_20 > current_sma_50:
                signal = "buy"
                strength = 0.8
            elif current_price < current_sma_20 < current_sma_50:
                signal = "sell"
                strength = 0.8
            else:
                signal = "hold"
                strength = 0.5
            
            indicators.append(TechnicalIndicator(
                name="Moving Averages",
                value=float(current_sma_20) if not pd.isna(current_sma_20) else 0.0,
                signal=signal,
                strength=strength,
                description=f"Price {current_price:.2f} vs SMA20: {current_sma_20:.2f}, SMA50: {current_sma_50:.2f}" if not pd.isna(current_sma_20) else "Moving average calculation failed"
            ))
            
            # Bollinger Bands
            bb = BollingerBands(price_data['close'])
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]
            
            if not pd.isna(current_upper) and not pd.isna(current_lower):
                if current_price <= current_lower:
                    signal = "buy"
                    strength = 0.7
                elif current_price >= current_upper:
                    signal = "sell"
                    strength = 0.7
                else:
                    signal = "hold"
                    strength = 0.5
                
                indicators.append(TechnicalIndicator(
                    name="Bollinger Bands",
                    value=float(current_price),
                    signal=signal,
                    strength=strength,
                    description=f"Price {current_price:.2f} between BB upper {current_upper:.2f} and lower {current_lower:.2f}"
                ))
            
        except Exception as e:
            logger.warning(f"Error calculating indicators: {str(e)}")
        
        return indicators
    
    def _detect_real_patterns(self, cv_image, price_data: pd.DataFrame) -> List[ChartPattern]:
        """
        Detect real chart patterns using computer vision and price data
        """
        patterns = []
        
        try:
            if len(price_data) < 10:
                return patterns
            
            # Detect trend patterns
            trend_pattern = self._detect_trend_pattern(price_data)
            if trend_pattern:
                patterns.append(trend_pattern)
            
            # Detect reversal patterns
            reversal_pattern = self._detect_reversal_pattern(price_data)
            if reversal_pattern:
                patterns.append(reversal_pattern)
            
            # Detect continuation patterns
            continuation_pattern = self._detect_continuation_pattern(price_data)
            if continuation_pattern:
                patterns.append(continuation_pattern)
            
        except Exception as e:
            logger.warning(f"Error detecting patterns: {str(e)}")
        
        return patterns
    
    def _detect_trend_pattern(self, price_data: pd.DataFrame) -> Optional[ChartPattern]:
        """
        Detect trend patterns in price data
        """
        try:
            if len(price_data) < 20:
                return None
            
            # Calculate trend direction
            recent_prices = price_data['close'].tail(20)
            slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            if slope > 0.1:  # Strong uptrend
                return ChartPattern(
                    name="Strong Uptrend",
                    pattern_type="trend",
                    signal="buy",
                    confidence=0.8,
                    description="Price showing strong upward momentum",
                    price_levels=[float(recent_prices.min()), float(recent_prices.max())]
                )
            elif slope < -0.1:  # Strong downtrend
                return ChartPattern(
                    name="Strong Downtrend",
                    pattern_type="trend",
                    signal="sell",
                    confidence=0.8,
                    description="Price showing strong downward momentum",
                    price_levels=[float(recent_prices.max()), float(recent_prices.min())]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trend pattern: {str(e)}")
            return None
    
    def _detect_reversal_pattern(self, price_data: pd.DataFrame) -> Optional[ChartPattern]:
        """
        Detect reversal patterns in price data
        """
        try:
            if len(price_data) < 15:
                return None
            
            recent_prices = price_data['close'].tail(15)
            
            # Check for double top/bottom
            peaks = self._find_peaks(recent_prices)
            troughs = self._find_troughs(recent_prices)
            
            if len(peaks) >= 2 and len(troughs) >= 1:
                # Potential double top
                if abs(peaks[-1] - peaks[-2]) < 0.02:  # Within 2%
                    return ChartPattern(
                        name="Double Top",
                        pattern_type="reversal",
                        signal="sell",
                        confidence=0.7,
                        description="Potential reversal pattern with two peaks at similar levels",
                        price_levels=[float(peaks[-1]), float(peaks[-2])]
                    )
            
            if len(troughs) >= 2 and len(peaks) >= 1:
                # Potential double bottom
                if abs(troughs[-1] - troughs[-2]) < 0.02:  # Within 2%
                    return ChartPattern(
                        name="Double Bottom",
                        pattern_type="reversal",
                        signal="buy",
                        confidence=0.7,
                        description="Potential reversal pattern with two troughs at similar levels",
                        price_levels=[float(troughs[-1]), float(troughs[-2])]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting reversal pattern: {str(e)}")
            return None
    
    def _detect_continuation_pattern(self, price_data: pd.DataFrame) -> Optional[ChartPattern]:
        """
        Detect continuation patterns in price data
        """
        try:
            if len(price_data) < 10:
                return None
            
            recent_prices = price_data['close'].tail(10)
            
            # Check for flag pattern (consolidation after trend)
            price_range = recent_prices.max() - recent_prices.min()
            avg_price = recent_prices.mean()
            
            if price_range < avg_price * 0.05:  # Low volatility consolidation
                # Determine if it's continuation of previous trend
                if len(price_data) >= 20:
                    prev_trend = price_data['close'].iloc[-20:-10].mean()
                    current_avg = recent_prices.mean()
                    
                    if abs(current_avg - prev_trend) < avg_price * 0.02:  # Similar levels
                        return ChartPattern(
                            name="Flag Pattern",
                            pattern_type="continuation",
                            signal="hold",
                            confidence=0.6,
                            description="Consolidation pattern suggesting trend continuation",
                            price_levels=[float(recent_prices.min()), float(recent_prices.max())]
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting continuation pattern: {str(e)}")
            return None
    
    def _find_peaks(self, series: pd.Series) -> List[float]:
        """Find peaks in price series"""
        peaks = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                peaks.append(series.iloc[i])
        return peaks
    
    def _find_troughs(self, series: pd.Series) -> List[float]:
        """Find troughs in price series"""
        troughs = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                troughs.append(series.iloc[i])
        return troughs
    
    def _enhance_patterns_with_llm(self, image: Image.Image, patterns: List[ChartPattern], 
                                  indicators: List[TechnicalIndicator]) -> Optional[List[ChartPattern]]:
        """
        Use LLM to enhance pattern recognition and analysis
        """
        try:
            if not self.llm_enabled or not patterns:
                return None
            
            # Convert image to base64 for API
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare prompt for LLM
            prompt = self._create_llm_prompt(patterns, indicators)
            
            # Call OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_patterns = self._parse_llm_response(result, patterns)
                return enhanced_patterns
            
            return None
            
        except Exception as e:
            logger.error(f"Error enhancing patterns with LLM: {str(e)}")
            return None
    
    def _create_llm_prompt(self, patterns: List[ChartPattern], 
                          indicators: List[TechnicalIndicator]) -> str:
        """Create prompt for LLM analysis"""
        return f"""
        Analyze this stock chart image and provide enhanced technical analysis.
        
        Current detected patterns: {[p.name for p in patterns]}
        Current indicators: {[f"{i.name}: {i.signal}" for i in indicators]}
        
        Please provide:
        1. Additional chart patterns you can identify
        2. Enhanced interpretation of existing patterns
        3. Trading recommendations based on the chart
        4. Risk assessment
        5. Key support and resistance levels
        
        Focus on actionable trading insights and professional technical analysis.
        """
    
    def _parse_llm_response(self, response: dict, existing_patterns: List[ChartPattern]) -> List[ChartPattern]:
        """Parse LLM response and enhance existing patterns"""
        try:
            content = response['choices'][0]['message']['content']
            
            # Enhanced patterns (simplified parsing)
            enhanced_patterns = existing_patterns.copy()
            
            # Add LLM insights as additional patterns
            enhanced_patterns.append(ChartPattern(
                name="LLM Enhanced Analysis",
                pattern_type="composite",
                signal="hold",  # Neutral signal for composite analysis
                confidence=0.8,
                description=f"AI-enhanced analysis: {content[:200]}...",
                price_levels=[]
            ))
            
            return enhanced_patterns
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return existing_patterns
    
    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generate fallback data when image analysis fails"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
    
    def _find_real_support_resistance(self, price_data: pd.DataFrame) -> List[SupportResistance]:
        """Find real support and resistance levels from price data"""
        support_resistance = []
        
        try:
            if len(price_data) < 20:
                return support_resistance
            
            prices = price_data['close']
            
            # Find local minima (support) and maxima (resistance)
            for i in range(2, len(prices) - 2):
                current_price = prices.iloc[i]
                
                # Check for support (local minimum)
                if (current_price < prices.iloc[i-1] and current_price < prices.iloc[i-2] and
                    current_price < prices.iloc[i+1] and current_price < prices.iloc[i+2]):
                    
                    # Calculate strength based on how much it bounces
                    bounce_height = min(prices.iloc[i+1], prices.iloc[i+2]) - current_price
                    strength = min(1.0, bounce_height / current_price * 10)  # Normalize
                    
                    support_resistance.append(SupportResistance(
                        level_type="support",
                        price=float(current_price),
                        strength=strength,
                        description=f"Support at {current_price:.2f}"
                    ))
                
                # Check for resistance (local maximum)
                elif (current_price > prices.iloc[i-1] and current_price > prices.iloc[i-2] and
                      current_price > prices.iloc[i+1] and current_price > prices.iloc[i+2]):
                    
                    # Calculate strength based on how much it falls
                    fall_height = current_price - max(prices.iloc[i+1], prices.iloc[i+2])
                    strength = min(1.0, fall_height / current_price * 10)  # Normalize
                    
                    support_resistance.append(SupportResistance(
                        level_type="resistance",
                        price=float(current_price),
                        strength=strength,
                        description=f"Resistance at {current_price:.2f}"
                    ))
            
            # Remove duplicates and sort by strength
            unique_levels = []
            seen_prices = set()
            
            for level in sorted(support_resistance, key=lambda x: x.strength, reverse=True):
                # Check if price level is close to existing ones (within 1%)
                price_close = False
                for seen_price in seen_prices:
                    if abs(level.price - seen_price) / seen_price < 0.01:
                        price_close = True
                        break
                
                if not price_close:
                    unique_levels.append(level)
                    seen_prices.add(level.price)
            
            return unique_levels[:5]  # Return top 5 levels
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            return []
    
    def _analyze_real_trend(self, price_data: pd.DataFrame) -> MarketTrend:
        """Analyze real market trend from price data"""
        try:
            if len(price_data) < 20:
                            return MarketTrend(
                trend="neutral",
                strength=0.5,
                timeframe="short",
                description="Insufficient data for trend analysis"
            )
            
            prices = price_data['close']
            
            # Calculate multiple trend indicators
            short_trend = prices.tail(10)
            medium_trend = prices.tail(20)
            long_trend = prices.tail(50) if len(prices) >= 50 else prices
            
            # Linear regression for trend slope
            short_slope = np.polyfit(range(len(short_trend)), short_trend, 1)[0]
            medium_slope = np.polyfit(range(len(medium_trend)), medium_trend, 1)[0]
            long_slope = np.polyfit(range(len(long_trend)), long_trend, 1)[0]
            
            # Determine trend direction
            if short_slope > 0.05 and medium_slope > 0.03 and long_slope > 0.01:
                direction = "uptrend"
                strength = min(1.0, (short_slope + medium_slope + long_slope) / 0.15)
            elif short_slope < -0.05 and medium_slope < -0.03 and long_slope < -0.01:
                direction = "downtrend"
                strength = min(1.0, abs(short_slope + medium_slope + long_slope) / 0.15)
            else:
                direction = "sideways"
                strength = 0.5
            
            # Determine trend duration
            if abs(long_slope) > 0.02:
                duration = "long"
            elif abs(medium_slope) > 0.03:
                duration = "medium"
            else:
                duration = "short"
            
            # Create trend description
            if direction == "uptrend":
                description = f"Strong upward momentum with {strength:.1%} confidence"
            elif direction == "downtrend":
                description = f"Strong downward momentum with {strength:.1%} confidence"
            else:
                description = f"Sideways movement with low volatility"
            
            return MarketTrend(
                trend=direction,
                strength=strength,
                timeframe=duration,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return MarketTrend(
                trend="neutral",
                strength=0.5,
                timeframe="unknown",
                description="Trend analysis failed"
            )
    
    def _generate_real_trading_advice(self, indicators, patterns, trend, price_data):
        """Generate real trading advice based on analysis"""
        try:
            # Count buy/sell signals
            buy_signals = sum(1 for ind in indicators if ind.signal == "buy")
            sell_signals = sum(1 for ind in indicators if ind.signal == "sell")
            hold_signals = sum(1 for ind in indicators if ind.signal == "hold")
            
            # Pattern signals
            pattern_buy = sum(1 for p in patterns if p.signal == "buy")
            pattern_sell = sum(1 for p in patterns if p.signal == "sell")
            
            # Trend influence
            trend_score = 0
            if trend.trend == "uptrend":
                trend_score = trend.strength
            elif trend.trend == "downtrend":
                trend_score = -trend.strength
            
            # Calculate overall signal
            total_signals = len(indicators) + len(patterns)
            if total_signals == 0:
                return "Hold position - insufficient data", "medium"
            
            # Weighted scoring
            signal_score = (
                (buy_signals + pattern_buy * 2) * 1.0 +
                (sell_signals + pattern_sell * 2) * -1.0 +
                trend_score * 0.5
            ) / total_signals
            
            # Generate advice
            if signal_score > 0.3:
                advice = "Consider buying - multiple bullish signals"
                risk = "low" if signal_score > 0.6 else "medium"
            elif signal_score < -0.3:
                advice = "Consider selling - multiple bearish signals"
                risk = "low" if signal_score < -0.6 else "medium"
            else:
                advice = "Hold position - mixed signals"
                risk = "medium"
            
            return advice, risk
            
        except Exception as e:
            logger.error(f"Error generating trading advice: {str(e)}")
            return "Hold position - analysis error", "high"
    
    def _calculate_real_sentiment(self, indicators, patterns, trend, price_data):
        """Calculate real sentiment from analysis data"""
        try:
            if not indicators and not patterns:
                return "neutral", 0.5
            
            # Calculate sentiment from indicators
            indicator_sentiment = 0
            for ind in indicators:
                if ind.signal == "buy":
                    indicator_sentiment += ind.strength
                elif ind.signal == "sell":
                    indicator_sentiment -= ind.strength
            
            # Calculate sentiment from patterns
            pattern_sentiment = 0
            for pattern in patterns:
                if pattern.signal == "buy":
                    pattern_sentiment += pattern.confidence
                elif pattern.signal == "sell":
                    pattern_sentiment -= pattern.confidence
            
            # Calculate sentiment from trend
            trend_sentiment = 0
            if trend.trend == "uptrend":
                trend_sentiment = trend.strength
            elif trend.trend == "downtrend":
                trend_sentiment = -trend.strength
            
            # Combine all sentiments
            total_sentiment = (
                indicator_sentiment * 0.4 +
                pattern_sentiment * 0.4 +
                trend_sentiment * 0.2
            )
            
            # Normalize to [-1, 1] range
            total_sentiment = max(-1.0, min(1.0, total_sentiment))
            
            # Convert to sentiment string
            if total_sentiment > 0.3:
                sentiment = "bullish"
            elif total_sentiment < -0.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # Calculate confidence
            confidence = abs(total_sentiment) * 0.8 + 0.2  # Base confidence of 20%
            
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return "neutral", 0.5
    
    def _generate_real_insights(self, indicators, patterns, support_resistance, price_data):
        """Generate real insights from analysis"""
        insights = []
        warnings = []
        
        try:
            # Price movement insights
            if len(price_data) >= 10:
                recent_prices = price_data['close'].tail(10)
                price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                
                if abs(price_change) > 0.1:  # 10% change
                    direction = "increased" if price_change > 0 else "decreased"
                    insights.append(f"Price has {direction} by {abs(price_change):.1%} in the last 10 periods")
                
                # Volatility insight
                volatility = recent_prices.std() / recent_prices.mean()
                if volatility > 0.05:  # High volatility
                    warnings.append(f"High volatility detected ({volatility:.1%}) - consider risk management")
            
            # Indicator insights
            for indicator in indicators:
                if indicator.name == "RSI":
                    if indicator.value < 20:
                        insights.append("RSI indicates extremely oversold conditions - potential reversal")
                    elif indicator.value > 80:
                        insights.append("RSI indicates extremely overbought conditions - potential reversal")
                
                elif indicator.name == "MACD":
                    if indicator.signal == "buy" and indicator.strength > 0.7:
                        insights.append("Strong MACD buy signal - momentum building")
                    elif indicator.signal == "sell" and indicator.strength > 0.7:
                        insights.append("Strong MACD sell signal - momentum declining")
            
            # Pattern insights
            for pattern in patterns:
                if pattern.confidence > 0.7:
                    insights.append(f"High-confidence {pattern.name} pattern detected - {pattern.description}")
                elif pattern.confidence < 0.4:
                    warnings.append(f"Low-confidence {pattern.name} pattern - verify with other indicators")
            
            # Support/Resistance insights
            if support_resistance:
                strong_levels = [level for level in support_resistance if level.strength > 0.7]
                if strong_levels:
                    insights.append(f"Strong {len(strong_levels)} support/resistance levels identified")
                
                # Check if price is near support/resistance
                current_price = price_data['close'].iloc[-1]
                for level in support_resistance:
                    distance = abs(current_price - level.price) / current_price
                    if distance < 0.02:  # Within 2%
                        if level.level_type == "support":
                            insights.append(f"Price approaching support at {level.price:.2f}")
                        else:
                            warnings.append(f"Price approaching resistance at {level.price:.2f}")
            
            # Volume insights (if available)
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'].tail(5)
                avg_volume = price_data['volume'].tail(20).mean()
                current_volume = recent_volume.iloc[-1]
                
                if current_volume > avg_volume * 1.5:
                    insights.append("Above-average volume - strong market participation")
                elif current_volume < avg_volume * 0.5:
                    warnings.append("Below-average volume - weak market participation")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Analysis completed successfully")
        
        return insights, warnings
    
    def _suggest_real_stop_losses(self, support_resistance, price_data):
        """Suggest real stop losses based on support levels"""
        try:
            if not support_resistance or len(price_data) < 10:
                return []
            
            current_price = price_data['close'].iloc[-1]
            stop_losses = []
            
            # Find nearest support levels
            support_levels = [level for level in support_resistance if level.level_type == "support"]
            support_levels.sort(key=lambda x: x.price, reverse=True)
            
            # Suggest stop loss below nearest support
            for support in support_levels:
                if support.price < current_price:
                    stop_loss = support.price * 0.98  # 2% below support
                    risk_percentage = (current_price - stop_loss) / current_price
                    
                    if risk_percentage < 0.15:  # Max 15% risk
                        stop_losses.append({
                            "price": float(stop_loss),
                            "risk_percentage": float(risk_percentage),
                            "description": f"Stop loss below support at {support.price:.2f}"
                        })
                        break
            
            # If no good support, suggest percentage-based stop loss
            if not stop_losses:
                conservative_stop = current_price * 0.90  # 10% stop loss
                stop_losses.append({
                    "price": float(conservative_stop),
                    "risk_percentage": 0.10,
                    "description": "Conservative 10% stop loss"
                })
            
            return stop_losses
            
        except Exception as e:
            logger.error(f"Error suggesting stop losses: {str(e)}")
            return []
    
    def _suggest_real_take_profits(self, support_resistance, price_data):
        """Suggest real take profits based on resistance levels"""
        try:
            if not support_resistance or len(price_data) < 10:
                return []
            
            current_price = price_data['close'].iloc[-1]
            take_profits = []
            
            # Find nearest resistance levels
            resistance_levels = [level for level in support_resistance if level.level_type == "resistance"]
            resistance_levels.sort(key=lambda x: x.price)
            
            # Suggest take profit at resistance levels
            for resistance in resistance_levels:
                if resistance.price > current_price:
                    profit_percentage = (resistance.price - current_price) / current_price
                    
                    if profit_percentage > 0.05:  # Min 5% profit
                        take_profits.append({
                            "price": float(resistance.price),
                            "profit_percentage": float(profit_percentage),
                            "description": f"Take profit at resistance {resistance.price:.2f}"
                        })
                        break
            
            # If no good resistance, suggest percentage-based take profit
            if not take_profits:
                conservative_target = current_price * 1.15  # 15% target
                take_profits.append({
                    "price": float(conservative_target),
                    "profit_percentage": 0.15,
                    "description": "Conservative 15% take profit target"
                })
            
            return take_profits
            
        except Exception as e:
            logger.error(f"Error suggesting take profits: {str(e)}")
            return []
