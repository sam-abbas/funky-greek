"""
Configuration settings for the Stock Chart Analysis API
"""

import os
from typing import List

class Settings:
    """Application settings"""
    
    # API Configuration
    API_TITLE: str = "Stock Chart Analysis API"
    API_DESCRIPTION: str = "A cost-effective service for analyzing stock charts and providing technical analysis"
    API_VERSION: str = "1.0.0"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "https://yourdomain.com",  # Replace with your actual domain
        "https://www.yourdomain.com",
        "*"  # Allow all origins for development (remove in production)
    ]
    
    # Analysis Configuration
    MIN_CONFIDENCE: float = 0.3
    MAX_ANALYSIS_TIME: int = 30  # seconds
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg", 
        "image/png",
        "image/bmp",
        "image/tiff"
    ]
    
    # Technical Analysis Configuration
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    SMA_SHORT: int = 20
    SMA_LONG: int = 50
    
    # Pattern Detection Configuration
    PATTERN_CONFIDENCE_THRESHOLD: float = 0.6
    MIN_PATTERN_BARS: int = 10
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Global settings instance
settings = Settings()
