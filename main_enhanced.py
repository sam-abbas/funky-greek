from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import io
import time
import logging
import os
import secrets
import hashlib
from datetime import datetime, timedelta

# Import enhanced components
from chart_analyzer_enhanced import EnhancedChartAnalyzer
from config_enhanced import enhanced_settings
from models import AnalysisResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, enhanced_settings.LOG_LEVEL),
    format=enhanced_settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=enhanced_settings.API_TITLE,
    description=enhanced_settings.API_DESCRIPTION,
    version=enhanced_settings.API_VERSION
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=enhanced_settings.ALLOWED_HOSTS
)

# Configure CORS with security restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=enhanced_settings.ALLOWED_ORIGINS,
    allow_credentials=False,  # Security: disable credentials
    allow_methods=["GET", "POST"],  # Restrict methods
    allow_headers=["*"],
    expose_headers=["X-Processing-Time"],  # Only expose safe headers
    max_age=3600,  # Cache preflight for 1 hour
)

# Security headers middleware will be defined after all functions

# Security components
security = HTTPBearer(auto_error=False)

# Rate limiting storage (in production, use Redis)
request_counts = {}
rate_limit_window = 3600  # 1 hour
max_requests_per_hour = 100

def cleanup_expired_rate_limits():
    """Clean up expired rate limit entries to prevent memory leaks"""
    now = datetime.now()
    expired_ips = []
    
    for ip, data in request_counts.items():
        if now > data["reset_time"]:
            expired_ips.append(ip)
    
    for ip in expired_ips:
        del request_counts[ip]
    
    if expired_ips:
        logger.debug(f"Cleaned up {len(expired_ips)} expired rate limit entries")

def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit - IMPROVED VERSION"""
    # Skip rate limiting for unknown IPs (development)
    if client_ip == "unknown":
        return True
    
    # Clean up expired entries periodically
    if len(request_counts) > 1000:  # Clean up when we have too many entries
        cleanup_expired_rate_limits()
    
    now = datetime.now()
    
    # Initialize client data if not exists
    if client_ip not in request_counts:
        request_counts[client_ip] = {
            "count": 0, 
            "reset_time": now + timedelta(hours=1)
        }
    
    client_data = request_counts[client_ip]
    
    # Reset counter if window expired
    if now > client_data["reset_time"]:
        client_data["count"] = 0
        client_data["reset_time"] = now + timedelta(hours=1)
    
    # Check if limit exceeded
    if client_data["count"] >= max_requests_per_hour:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return False
    
    # Increment counter
    client_data["count"] += 1
    logger.debug(f"Rate limit check for IP {client_ip}: {client_data['count']}/{max_requests_per_hour}")
    
    return True

def get_client_ip(request: Request) -> str:
    """Get client IP address with improved detection"""
    # Get real IP from proxy headers (common in production)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client host
    if request.client and request.client.host:
        return request.client.host
    
    # Last resort - use a default identifier
    return "unknown"

def sanitize_response(data: dict) -> dict:
    """Remove sensitive information from responses - ENHANCED VERSION"""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Remove sensitive keys (expanded list)
            if key.lower() in [
                'openai_api_key', 'api_key', 'secret', 'password', 'token',
                'environment', 'development', 'production', 'internal',
                'server', 'host', 'port', 'config', 'settings'
            ]:
                continue
            
            # Recursively sanitize nested structures
            if isinstance(value, dict):
                sanitized[key] = sanitize_response(value)
            elif isinstance(value, list):
                sanitized[key] = [sanitize_response(item) if isinstance(item, dict) else item for item in value]
            else:
                # Additional sanitization for string values
                if isinstance(value, str):
                    # Remove any strings that might contain sensitive info
                    if any(sensitive in value.lower() for sensitive in ['sk-', 'api_key', 'secret', 'password']):
                        continue
                sanitized[key] = value
        return sanitized
    return data

# Initialize enhanced chart analyzer
analyzer = EnhancedChartAnalyzer(
    openai_api_key=enhanced_settings.OPENAI_API_KEY
)

@app.get("/")
async def root(request: Request):
    """Root endpoint - minimal information"""
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    return sanitize_response({
        "message": f"{enhanced_settings.API_TITLE} v{enhanced_settings.API_VERSION}",
        "status": "running"
    })

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint - minimal information"""
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    return sanitize_response({
        "status": "healthy",
        "service": enhanced_settings.API_TITLE,
        "version": enhanced_settings.API_VERSION,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/info")
async def get_info(request: Request):
    """Get service information - sanitized"""
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    return sanitize_response({
        "name": enhanced_settings.API_TITLE,
        "version": enhanced_settings.API_VERSION,
        "description": enhanced_settings.API_DESCRIPTION,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "local_mode": enhanced_settings.FORCE_LOCAL_MODE,
        "features": [
            "Real-time chart analysis",
            "Advanced computer vision",
            "Professional technical indicators",
            "Pattern recognition",
            "Support/Resistance detection",
            "Risk assessment",
            "Trading recommendations"
        ],
        "endpoints": {
            "analyze_chart": "Full analysis with enhanced features",
            "analyze_chart_local": "Local analysis only",
            "demo": "Demo analysis",
            "demo_local": "Demo with local analysis"
        },
        "max_file_size_mb": enhanced_settings.MAX_FILE_SIZE // (1024 * 1024)
    })

@app.post("/analyze-chart", response_model=AnalysisResponse)
async def analyze_chart(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze a stock chart image with enhanced technical analysis
    """
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > enhanced_settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File size {file_size} bytes exceeds maximum {enhanced_settings.MAX_FILE_SIZE} bytes"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Validate image dimensions
        width, height = image.size
        if width < enhanced_settings.MIN_CHART_SIZE or height < enhanced_settings.MIN_CHART_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Image dimensions {width}x{height} are too small. Minimum: {enhanced_settings.MIN_CHART_SIZE}x{enhanced_settings.MIN_CHART_SIZE}"
            )
        
        if width > enhanced_settings.MAX_CHART_SIZE or height > enhanced_settings.MAX_CHART_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Image dimensions {width}x{height} are too large. Maximum: {enhanced_settings.MAX_CHART_SIZE}x{enhanced_settings.MAX_CHART_SIZE}"
            )
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze chart with enhanced analyzer
        logger.info(f"Starting enhanced analysis of {file.filename} ({width}x{height})")
        analysis_result = analyzer.analyze_chart(image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Log analysis results (without sensitive data)
        logger.info(f"Analysis completed for IP {client_ip} in {processing_time:.2f}ms")
        logger.info(f"Sentiment: {analysis_result.overall_sentiment}, Confidence: {analysis_result.confidence_score:.1%}")
        logger.info(f"Indicators: {len(analysis_result.indicators)}, Patterns: {len(analysis_result.patterns)}")
        
        # Return enhanced response
        return AnalysisResponse(
            success=True,
            message="Enhanced chart analysis completed successfully",
            analysis=analysis_result,
            processing_time_ms=processing_time,
            llm_enhanced=enhanced_settings.LLM_ENABLED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chart analysis for IP {client_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during chart analysis")

@app.post("/analyze-chart-local", response_model=AnalysisResponse)
async def analyze_chart_local_only(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze a stock chart image using LOCAL ANALYSIS ONLY (no LLM calls)
    This endpoint forces local analysis regardless of LLM configuration
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > enhanced_settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File size {file_size} bytes exceeds maximum {enhanced_settings.MAX_FILE_SIZE} bytes"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Validate image dimensions
        width, height = image.size
        if width < enhanced_settings.MIN_CHART_SIZE or height < enhanced_settings.MIN_CHART_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Image dimensions {width}x{height} are too small. Minimum: {enhanced_settings.MIN_CHART_SIZE}x{enhanced_settings.MIN_CHART_SIZE}"
            )
        
        if width > enhanced_settings.MAX_CHART_SIZE or height > enhanced_settings.MAX_CHART_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Image dimensions {width}x{height} are too large. Maximum: {enhanced_settings.MAX_CHART_SIZE}x{enhanced_settings.MAX_CHART_SIZE}"
            )
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze chart with LOCAL ONLY analyzer (temporarily disable LLM)
        logger.info(f"Starting LOCAL ONLY analysis of {file.filename} ({width}x{height})")
        
        # Create a temporary analyzer with LLM disabled
        temp_analyzer = EnhancedChartAnalyzer(openai_api_key=None)
        analysis_result = temp_analyzer.analyze_chart(image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Log analysis results
        logger.info(f"Local analysis completed in {processing_time:.2f}ms")
        logger.info(f"Sentiment: {analysis_result.overall_sentiment}, Confidence: {analysis_result.confidence_score:.1%}")
        logger.info(f"Indicators: {len(analysis_result.indicators)}, Patterns: {len(analysis_result.patterns)}")
        
        # Return local analysis response
        return AnalysisResponse(
            success=True,
            message="Local chart analysis completed successfully (no LLM calls)",
            analysis=analysis_result,
            processing_time_ms=processing_time,
            llm_enhanced=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in local chart analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/demo")
async def get_demo(request: Request):
    """Get a demo analysis result"""
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    try:
        # Create a demo chart image
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Generate a demo chart
        width, height = 800, 400
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a demo chart pattern
        points = [(50, 200), (200, 150), (350, 100), (500, 150), (650, 200), (750, 250)]
        draw.line(points, fill='blue', width=3)
        
        # Add some candlestick-like elements
        for i in range(5):
            x = 100 + i * 120
            y = 150 + (i * 20)
            color = 'green' if i < 3 else 'red'
            draw.rectangle([x-10, y-20, x+10, y+20], fill=color, outline='black')
        
        # Add grid
        for i in range(0, width, 50):
            draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
        for i in range(0, height, 50):
            draw.line([(0, i), (width, i)], fill='lightgray', width=1)
        
        # Analyze the demo chart
        start_time = time.time()
        analysis_result = analyzer.analyze_chart(image)
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            success=True,
            message="Demo analysis completed successfully",
            analysis=analysis_result,
            processing_time_ms=processing_time,
            llm_enhanced=enhanced_settings.LLM_ENABLED
        )
        
    except Exception as e:
        logger.error(f"Error in demo analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo analysis failed: {str(e)}")

@app.get("/demo-local")
async def get_demo_local(request: Request):
    """Get a demo analysis result using LOCAL ANALYSIS ONLY (no LLM calls)"""
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    try:
        # Create a demo chart image
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Generate a demo chart
        width, height = 800, 400
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a demo chart pattern
        points = [(50, 200), (200, 150), (350, 100), (500, 150), (650, 200), (750, 250)]
        draw.line(points, fill='blue', width=3)
        
        # Add some candlestick-like elements
        for i in range(5):
            x = 100 + i * 120
            y = 150 + (i * 20)
            color = 'green' if i < 3 else 'red'
            draw.rectangle([x-10, y-20, x+10, y+20], fill=color, outline='black')
        
        # Add grid
        for i in range(0, width, 50):
            draw.line([(i, 0), (i, height)], fill='lightgray', width=1)
        for i in range(0, height, 50):
            draw.line([(0, i), (width, i)], fill='lightgray', width=1)
        
        # Analyze the demo chart with LOCAL ONLY analyzer
        start_time = time.time()
        temp_analyzer = EnhancedChartAnalyzer(openai_api_key=None)
        analysis_result = temp_analyzer.analyze_chart(image)
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            success=True,
            message="Local demo analysis completed successfully (no LLM calls)",
            analysis=analysis_result,
            processing_time_ms=processing_time,
            llm_enhanced=False
        )
        
    except Exception as e:
        logger.error(f"Error in local demo analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Local demo analysis failed: {str(e)}")

@app.get("/config")
async def get_config(request: Request):
    """Get current configuration (without sensitive data)"""
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    return sanitize_response({
        "max_file_size_mb": enhanced_settings.MAX_FILE_SIZE // (1024 * 1024),
        "min_chart_size": enhanced_settings.MIN_CHART_SIZE,
        "max_chart_size": enhanced_settings.MAX_CHART_SIZE,
        "caching_enabled": enhanced_settings.CACHE_ENABLED
    })

# Security headers middleware (FIXED VERSION)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Security headers - CRITICAL for production
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Rate limiting headers
    try:
        client_ip = get_client_ip(request)
        if client_ip in request_counts:
            remaining = max(0, max_requests_per_hour - request_counts[client_ip]["count"])
            response.headers["X-RateLimit-Limit"] = str(max_requests_per_hour)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(request_counts[client_ip]["reset_time"].timestamp()))
    except Exception as e:
        logger.debug(f"Error adding rate limit headers: {e}")
    
    # Remove server information - FIXED: use del instead of pop
    if "server" in response.headers:
        del response.headers["server"]
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Enhanced Stock Chart Analyzer v{enhanced_settings.API_VERSION}")
    logger.info(f"LLM Integration: {'Enabled' if enhanced_settings.LLM_ENABLED else 'Disabled'}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    uvicorn.run(
        "main_enhanced:app",
        host=enhanced_settings.HOST,
        port=enhanced_settings.PORT,
        log_level=enhanced_settings.LOG_LEVEL.lower()
    )
