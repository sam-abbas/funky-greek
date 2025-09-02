from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class AnalysisRequest(BaseModel):
    """Request model for chart analysis"""
    chart_url: Optional[str] = None
    analysis_type: Optional[str] = "comprehensive"  # comprehensive, basic, specific_indicators

class TechnicalIndicator(BaseModel):
    """Model for technical indicators"""
    name: str
    value: float
    signal: str  # buy, sell, hold, neutral
    strength: float  # 0.0 to 1.0
    description: str

class ChartPattern(BaseModel):
    """Model for chart patterns"""
    name: str
    confidence: float  # 0.0 to 1.0
    signal: str
    description: str
    price_targets: Optional[List[float]] = None

class SupportResistance(BaseModel):
    """Model for support and resistance levels"""
    level_type: str  # support or resistance
    price: float
    strength: float  # 0.0 to 1.0
    description: str

class MarketTrend(BaseModel):
    """Model for market trend analysis"""
    trend: str  # bullish, bearish, sideways
    strength: float  # 0.0 to 1.0
    timeframe: str  # short, medium, long
    description: str

class AnalysisResult(BaseModel):
    """Model for complete analysis result"""
    timestamp: datetime
    overall_sentiment: str  # bullish, bearish, neutral
    confidence_score: float  # 0.0 to 1.0
    
    # Technical indicators
    indicators: List[TechnicalIndicator]
    
    # Chart patterns
    patterns: List[ChartPattern]
    
    # Support and resistance
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]
    
    # Market trend
    trend_analysis: MarketTrend
    
    # Trading advice
    trading_advice: str
    risk_level: str  # low, medium, high
    stop_loss_suggestions: Optional[List[float]] = None
    take_profit_targets: Optional[List[float]] = None
    
    # Additional insights
    insights: List[str]
    warnings: List[str]

class AnalysisResponse(BaseModel):
    """Response model for chart analysis"""
    success: bool
    analysis: AnalysisResult
    message: str
    processing_time_ms: Optional[float] = None
