import os
import logging
from typing import List, Optional
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class EnhancedSettings(BaseSettings):
    """Enhanced configuration settings for the Stock Chart Analyzer"""
    
    # API Configuration
    API_TITLE: str = "Enhanced Stock Chart Analyzer"
    API_DESCRIPTION: str = "Advanced stock chart analysis with real technical indicators and LLM integration"
    API_VERSION: str = "2.0.0"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = ["*"]  # Configure for production
    
    # Analysis Configuration
    MIN_CHART_SIZE: int = 100
    MAX_CHART_SIZE: int = 4000
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Technical Analysis Settings
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    SMA_SHORT: int = 20
    SMA_LONG: int = 50
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    
    # Pattern Detection Settings
    MIN_PATTERN_CONFIDENCE: float = 0.3
    TREND_SLOPE_THRESHOLD: float = 0.1
    REVERSAL_THRESHOLD: float = 0.02
    CONTINUATION_VOLATILITY_THRESHOLD: float = 0.05
    
    # Computer Vision Settings
    CANNY_LOW_THRESHOLD: int = 50
    CANNY_HIGH_THRESHOLD: int = 150
    MIN_CONTOUR_AREA: int = 100
    HOUGH_THRESHOLD: int = 50
    MIN_LINE_LENGTH: int = 30
    MAX_LINE_GAP: int = 10
    
    # LLM Integration Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-vision-preview"
    OPENAI_MAX_TOKENS: int = 1000
    LLM_ENABLED: bool = False
    
    # Alternative LLM Options
    CLAUDE_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-3-sonnet-20240229"
    CLAUDE_MAX_TOKENS: int = 1000
    
    # Local LLM Options (for cost optimization)
    LOCAL_LLM_ENABLED: bool = False
    LOCAL_LLM_URL: str = "http://localhost:11434"  # Ollama default
    LOCAL_LLM_MODEL: str = "llama2:13b"
    
    # Performance Settings
    MAX_PROCESSING_TIME: int = 30000  # 30 seconds
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cost Optimization
    MAX_LLM_CALLS_PER_DAY: int = 100
    FALLBACK_TO_LOCAL_ANALYSIS: bool = True
    USE_CACHED_ANALYSIS: bool = True
    
    # Local Analysis Mode
    FORCE_LOCAL_MODE: bool = False  # Set to True to disable all LLM calls
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
enhanced_settings = EnhancedSettings()

# Auto-detect LLM availability (unless forced to local mode)
if enhanced_settings.FORCE_LOCAL_MODE:
    enhanced_settings.LLM_ENABLED = False
    logger.info("Local mode forced - LLM integration disabled")
elif enhanced_settings.OPENAI_API_KEY:
    enhanced_settings.LLM_ENABLED = True
elif enhanced_settings.CLAUDE_API_KEY:
    enhanced_settings.LLM_ENABLED = True
elif enhanced_settings.LOCAL_LLM_ENABLED:
    enhanced_settings.LLM_ENABLED = True

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    enhanced_settings.ALLOWED_ORIGINS = [
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ]
    enhanced_settings.LOG_LEVEL = "WARNING"
    enhanced_settings.CACHE_ENABLED = True
elif os.getenv("ENVIRONMENT") == "development":
    enhanced_settings.ALLOWED_ORIGINS = ["*"]
    enhanced_settings.LOG_LEVEL = "DEBUG"
    enhanced_settings.CACHE_ENABLED = False
