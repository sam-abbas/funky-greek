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

class FairValueGap(BaseModel):
    """Model for fair value gaps"""
    gap_type: str  # 'bullish', 'bearish'
    start_price: float
    end_price: float
    start_time: str
    end_time: str
    confidence: float
    volume_confirmation: bool
    fill_probability: float
    description: str

class DailyLevel(BaseModel):
    """Model for daily high/low levels"""
    level_type: str  # 'high', 'low', 'open', 'close'
    price: float
    time: str
    strength: float  # 1-10 scale
    tested_count: int
    last_tested: Optional[str] = None
    description: str

class PriceActionPattern(BaseModel):
    """Model for price action patterns"""
    pattern_name: str
    pattern_type: str  # 'reversal', 'continuation', 'indecision'
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    start_time: str
    end_time: str
    key_levels: List[float]
    volume_profile: Optional[dict] = None
    description: str

class OrderBlock(BaseModel):
    """Model for order blocks"""
    block_type: str  # 'bullish', 'bearish'
    high: float
    low: float
    start_time: str
    end_time: str
    strength: float
    tested: bool
    invalidation_level: float
    description: str

class LiquidityZone(BaseModel):
    """Model for liquidity zones"""
    zone_type: str  # 'buy_side', 'sell_side'
    high: float
    low: float
    start_time: str
    end_time: str
    strength: float
    tested_count: int
    last_tested: Optional[str] = None
    description: str

class MarketStructure(BaseModel):
    """Model for market structure analysis"""
    structure_type: str  # 'break_of_structure', 'change_of_character', 'continuation'
    direction: str  # 'bullish', 'bearish'
    key_level: float
    time: str
    confidence: float
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
    
    # Advanced indicators
    fair_value_gaps: List[FairValueGap]
    daily_levels: List[DailyLevel]
    price_action_patterns: List[PriceActionPattern]
    order_blocks: List[OrderBlock]
    liquidity_zones: List[LiquidityZone]
    market_structure: List[MarketStructure]
    
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
    llm_enhanced: Optional[bool] = False
