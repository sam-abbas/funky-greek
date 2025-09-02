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
from models import (
    AnalysisResult, TechnicalIndicator, ChartPattern, 
    SupportResistance, MarketTrend
)

logger = logging.getLogger(__name__)

class ChartAnalyzer:
    """
    Analyzes stock chart images and provides technical analysis
    """
    
    def __init__(self):
        self.min_confidence = 0.3
        
    def analyze_chart(self, image: Image.Image) -> AnalysisResult:
        """
        Main method to analyze a stock chart image
        """
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract price data from chart (simulated for demo)
            price_data = self._extract_price_data(cv_image)
            
            # Perform technical analysis
            indicators = self._calculate_indicators(price_data)
            patterns = self._detect_patterns(cv_image, price_data)
            support_resistance = self._find_support_resistance(price_data)
            trend = self._analyze_trend(price_data)
            
            # Generate trading advice
            trading_advice, risk_level = self._generate_trading_advice(
                indicators, patterns, trend
            )
            
            # Calculate overall sentiment
            sentiment, confidence = self._calculate_sentiment(
                indicators, patterns, trend
            )
            
            # Generate insights and warnings
            insights, warnings = self._generate_insights(
                indicators, patterns, support_resistance
            )
            
            return AnalysisResult(
                timestamp=datetime.now(),
                overall_sentiment=sentiment,
                confidence_score=confidence,
                indicators=indicators,
                patterns=patterns,
                support_levels=[sr for sr in support_resistance if sr.level_type == "support"],
                resistance_levels=[sr for sr in support_resistance if sr.level_type == "resistance"],
                trend_analysis=trend,
                trading_advice=trading_advice,
                risk_level=risk_level,
                stop_loss_suggestions=self._suggest_stop_losses(support_resistance),
                take_profit_targets=self._suggest_take_profits(resistance_levels=[sr for sr in support_resistance if sr.level_type == "resistance"]),
                insights=insights,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in chart analysis: {str(e)}")
            raise
    
    def _extract_price_data(self, cv_image) -> pd.DataFrame:
        """
        Extract price data from chart image
        This is a simplified implementation - in production you'd use OCR or chart parsing
        """
        # For demo purposes, generate synthetic price data
        # In production, this would use computer vision to extract actual price data
        
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate realistic price movement
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Add some trend
        trend = np.linspace(0, 0.1, 100)  # 10% upward trend
        prices = [p * (1 + t) for p, t in zip(prices, trend)]
        
        # Add volume (simplified)
        volumes = np.random.randint(1000000, 10000000, 100)
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
    
    def _calculate_indicators(self, price_data: pd.DataFrame) -> list[TechnicalIndicator]:
        """
        Calculate various technical indicators
        """
        indicators = []
        
        try:
            # RSI
            rsi = RSIIndicator(price_data['close'], window=14)
            rsi_values = rsi.rsi()
            current_rsi = rsi_values.iloc[-1]
            
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
            
            if current_macd > current_signal:
                signal = "buy"
                strength = min(1.0, abs(current_macd - current_signal) / abs(current_macd))
            else:
                signal = "sell"
                strength = min(1.0, abs(current_macd - current_signal) / abs(current_macd))
            
            indicators.append(TechnicalIndicator(
                name="MACD",
                value=float(current_macd),
                signal=signal,
                strength=strength,
                description=f"MACD at {current_macd:.2f} vs signal at {current_signal:.2f}"
            ))
            
            # Moving Averages
            sma_20 = SMAIndicator(price_data['close'], window=20)
            sma_50 = SMAIndicator(price_data['close'], window=50)
            
            current_sma_20 = sma_20.sma_indicator().iloc[-1]
            current_sma_50 = sma_50.sma_indicator().iloc[-1]
            current_price = price_data['close'].iloc[-1]
            
            if current_price > current_sma_20 > current_sma_50:
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
                value=float(current_sma_20),
                signal=signal,
                strength=strength,
                description=f"Price {current_price:.2f} vs SMA20: {current_sma_20:.2f}, SMA50: {current_sma_50:.2f}"
            ))
            
        except Exception as e:
            logger.warning(f"Error calculating indicators: {str(e)}")
        
        return indicators
    
    def _detect_patterns(self, cv_image, price_data: pd.DataFrame) -> list[ChartPattern]:
        """
        Detect chart patterns using computer vision and price data
        """
        patterns = []
        
        try:
            # Simple pattern detection based on price action
            prices = price_data['close'].values
            
            # Detect double top/bottom (simplified)
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                peaks = self._find_peaks(recent_prices)
                troughs = self._find_troughs(recent_prices)
                
                if len(peaks) >= 2:
                    # Check for double top
                    if abs(peaks[-1] - peaks[-2]) < 0.02:  # Within 2%
                        patterns.append(ChartPattern(
                            name="Double Top",
                            confidence=0.7,
                            signal="sell",
                            description="Potential reversal pattern with two similar peaks",
                            price_targets=[prices[-1] * 0.9]  # 10% downside target
                        ))
                
                if len(troughs) >= 2:
                    # Check for double bottom
                    if abs(troughs[-1] - troughs[-2]) < 0.02:  # Within 2%
                        patterns.append(ChartPattern(
                            name="Double Bottom",
                            confidence=0.7,
                            signal="buy",
                            description="Potential reversal pattern with two similar troughs",
                            price_targets=[prices[-1] * 1.1]  # 10% upside target
                        ))
            
            # Trend continuation patterns
            if len(prices) >= 10:
                recent_trend = np.polyfit(range(10), prices[-10:], 1)[0]
                if recent_trend > 0:
                    patterns.append(ChartPattern(
                        name="Uptrend Continuation",
                        confidence=0.6,
                        signal="buy",
                        description="Price showing consistent upward movement",
                        price_targets=[prices[-1] * 1.05]  # 5% upside target
                    ))
                elif recent_trend < 0:
                    patterns.append(ChartPattern(
                        name="Downtrend Continuation",
                        confidence=0.6,
                        signal="sell",
                        description="Price showing consistent downward movement",
                        price_targets=[prices[-1] * 0.95]  # 5% downside target
                    ))
                    
        except Exception as e:
            logger.warning(f"Error detecting patterns: {str(e)}")
        
        return patterns
    
    def _find_peaks(self, data):
        """Find peaks in data"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(data[i])
        return peaks
    
    def _find_troughs(self, data):
        """Find troughs in data"""
        troughs = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                troughs.append(data[i])
        return troughs
    
    def _find_support_resistance(self, price_data: pd.DataFrame) -> list[SupportResistance]:
        """
        Find support and resistance levels
        """
        levels = []
        prices = price_data['close'].values
        
        try:
            # Simple support/resistance detection
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                
                # Find local minima (support)
                for i in range(1, len(recent_prices) - 1):
                    if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                        levels.append(SupportResistance(
                            level_type="support",
                            price=float(recent_prices[i]),
                            strength=0.7,
                            description=f"Support level at {recent_prices[i]:.2f}"
                        ))
                
                # Find local maxima (resistance)
                for i in range(1, len(recent_prices) - 1):
                    if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                        levels.append(SupportResistance(
                            level_type="resistance",
                            price=float(recent_prices[i]),
                            strength=0.7,
                            description=f"Resistance level at {recent_prices[i]:.2f}"
                        ))
            
            # Add psychological levels
            current_price = prices[-1]
            round_levels = [round(current_price / 10) * 10, round(current_price / 5) * 5]
            
            for level in round_levels:
                if level < current_price:
                    levels.append(SupportResistance(
                        level_type="support",
                        price=float(level),
                        strength=0.5,
                        description=f"Psychological support at {level}"
                    ))
                else:
                    levels.append(SupportResistance(
                        level_type="resistance",
                        price=float(level),
                        strength=0.5,
                        description=f"Psychological resistance at {level}"
                    ))
                    
        except Exception as e:
            logger.warning(f"Error finding support/resistance: {str(e)}")
        
        return levels
    
    def _analyze_trend(self, price_data: pd.DataFrame) -> MarketTrend:
        """
        Analyze overall market trend
        """
        try:
            prices = price_data['close'].values
            
            if len(prices) >= 20:
                # Short-term trend (last 5 days)
                short_trend = np.polyfit(range(5), prices[-5:], 1)[0]
                
                # Medium-term trend (last 20 days)
                medium_trend = np.polyfit(range(20), prices[-20:], 1)[0]
                
                # Determine overall trend
                if short_trend > 0 and medium_trend > 0:
                    trend = "bullish"
                    strength = min(1.0, (short_trend + medium_trend) / 0.1)
                elif short_trend < 0 and medium_trend < 0:
                    trend = "bearish"
                    strength = min(1.0, abs(short_trend + medium_trend) / 0.1)
                else:
                    trend = "sideways"
                    strength = 0.5
                
                description = f"Short-term trend: {'upward' if short_trend > 0 else 'downward'}, Medium-term trend: {'upward' if medium_trend > 0 else 'downward'}"
                
            else:
                trend = "neutral"
                strength = 0.5
                description = "Insufficient data for trend analysis"
            
            return MarketTrend(
                trend=trend,
                strength=strength,
                timeframe="medium",
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing trend: {str(e)}")
            return MarketTrend(
                trend="neutral",
                strength=0.5,
                timeframe="unknown",
                description="Error in trend analysis"
            )
    
    def _generate_trading_advice(self, indicators, patterns, trend) -> tuple[str, str]:
        """
        Generate trading advice based on analysis
        """
        try:
            # Count buy/sell signals
            buy_signals = sum(1 for ind in indicators if ind.signal == "buy")
            sell_signals = sum(1 for ind in indicators if ind.signal == "sell")
            
            # Pattern signals
            for pattern in patterns:
                if pattern.signal == "buy":
                    buy_signals += 1
                elif pattern.signal == "sell":
                    sell_signals += 1
            
            # Trend influence
            if trend.trend == "bullish":
                buy_signals += 1
            elif trend.trend == "bearish":
                sell_signals += 1
            
            # Generate advice
            if buy_signals > sell_signals:
                advice = "Consider buying with proper risk management"
                risk_level = "medium" if abs(buy_signals - sell_signals) <= 2 else "low"
            elif sell_signals > buy_signals:
                advice = "Consider selling or taking profits"
                risk_level = "medium" if abs(sell_signals - buy_signals) <= 2 else "high"
            else:
                advice = "Hold current position, monitor for clearer signals"
                risk_level = "medium"
            
            return advice, risk_level
            
        except Exception as e:
            logger.warning(f"Error generating trading advice: {str(e)}")
            return "Unable to generate trading advice", "unknown"
    
    def _calculate_sentiment(self, indicators, patterns, trend) -> tuple[str, float]:
        """
        Calculate overall market sentiment and confidence
        """
        try:
            total_signals = len(indicators) + len(patterns)
            if total_signals == 0:
                return "neutral", 0.5
            
            # Calculate weighted sentiment
            sentiment_score = 0
            total_weight = 0
            
            for indicator in indicators:
                weight = indicator.strength
                if indicator.signal == "buy":
                    sentiment_score += weight
                elif indicator.signal == "sell":
                    sentiment_score -= weight
                total_weight += weight
            
            for pattern in patterns:
                weight = pattern.confidence
                if pattern.signal == "buy":
                    sentiment_score += weight
                elif pattern.signal == "sell":
                    sentiment_score -= weight
                total_weight += weight
            
            # Normalize sentiment
            if total_weight > 0:
                normalized_sentiment = sentiment_score / total_weight
                
                if normalized_sentiment > 0.2:
                    sentiment = "bullish"
                elif normalized_sentiment < -0.2:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
                
                confidence = min(1.0, abs(normalized_sentiment) + 0.3)
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            return sentiment, confidence
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {str(e)}")
            return "neutral", 0.5
    
    def _suggest_stop_losses(self, support_levels) -> list[float]:
        """Suggest stop loss levels"""
        try:
            if not support_levels:
                return []
            
            # Sort by strength and suggest strongest support levels
            sorted_levels = sorted(support_levels, key=lambda x: x.strength, reverse=True)
            return [level.price * 0.98 for level in sorted_levels[:2]]  # 2% below support
            
        except Exception as e:
            logger.warning(f"Error suggesting stop losses: {str(e)}")
            return []
    
    def _suggest_take_profits(self, resistance_levels) -> list[float]:
        """Suggest take profit levels"""
        try:
            if not resistance_levels:
                return []
            
            # Sort by strength and suggest strongest resistance levels
            sorted_levels = sorted(resistance_levels, key=lambda x: x.strength, reverse=True)
            return [level.price * 1.02 for level in sorted_levels[:2]]  # 2% above resistance
            
        except Exception as e:
            logger.warning(f"Error suggesting take profits: {str(e)}")
            return []
    
    def _generate_insights(self, indicators, patterns, support_resistance) -> tuple[list[str], list[str]]:
        """Generate insights and warnings"""
        insights = []
        warnings = []
        
        try:
            # Generate insights
            if indicators:
                strong_buy = [ind for ind in indicators if ind.signal == "buy" and ind.strength > 0.7]
                strong_sell = [ind for ind in indicators if ind.signal == "sell" and ind.strength > 0.7]
                
                if strong_buy:
                    insights.append(f"Strong buy signals from {len(strong_buy)} indicators")
                if strong_sell:
                    insights.append(f"Strong sell signals from {len(strong_sell)} indicators")
            
            if patterns:
                insights.append(f"Detected {len(patterns)} chart patterns")
            
            if support_resistance:
                insights.append(f"Identified {len(support_resistance)} key price levels")
            
            # Generate warnings
            if not indicators:
                warnings.append("Limited technical indicator data available")
            
            if not patterns:
                warnings.append("No clear chart patterns detected")
            
            if not support_resistance:
                warnings.append("Unable to identify clear support/resistance levels")
            
            # Add general warnings
            warnings.append("This analysis is for educational purposes only")
            warnings.append("Always use proper risk management and stop losses")
            warnings.append("Consider multiple timeframes for confirmation")
            
        except Exception as e:
            logger.warning(f"Error generating insights: {str(e)}")
            insights = ["Analysis completed with some limitations"]
            warnings = ["Error in generating detailed insights"]
        
        return insights, warnings
