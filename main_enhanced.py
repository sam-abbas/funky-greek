from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
import logging
import os
from datetime import datetime

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=enhanced_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize enhanced chart analyzer
analyzer = EnhancedChartAnalyzer(
    openai_api_key=enhanced_settings.OPENAI_API_KEY
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Stock Chart Analyzer API",
        "version": enhanced_settings.API_VERSION,
        "status": "running",
        "llm_enabled": enhanced_settings.LLM_ENABLED
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": enhanced_settings.API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "llm_status": "enabled" if enhanced_settings.LLM_ENABLED else "disabled",
        "uptime": "running"
    }

@app.get("/info")
async def get_info():
    """Get service information"""
    return {
        "name": enhanced_settings.API_TITLE,
        "description": enhanced_settings.API_DESCRIPTION,
        "version": enhanced_settings.API_VERSION,
        "features": [
            "Real-time chart analysis",
            "Advanced computer vision",
            "Professional technical indicators",
            "Pattern recognition",
            "LLM-enhanced insights" if enhanced_settings.LLM_ENABLED else "Local analysis only",
            "Support/Resistance detection",
            "Risk assessment",
            "Trading recommendations"
        ],
        "endpoints": {
            "analyze_chart": "Full analysis with LLM if available",
            "analyze_chart_local": "Local analysis only (no LLM calls)",
            "demo": "Demo with current analysis mode",
            "demo_local": "Demo with local analysis only"
        },
        "llm_integration": enhanced_settings.LLM_ENABLED,
        "force_local_mode": enhanced_settings.FORCE_LOCAL_MODE,
        "max_file_size_mb": enhanced_settings.MAX_FILE_SIZE // (1024 * 1024)
    }

@app.post("/analyze-chart", response_model=AnalysisResponse)
async def analyze_chart(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze a stock chart image with enhanced technical analysis
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
        
        # Analyze chart with enhanced analyzer
        logger.info(f"Starting enhanced analysis of {file.filename} ({width}x{height})")
        analysis_result = analyzer.analyze_chart(image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Log analysis results
        logger.info(f"Analysis completed in {processing_time:.2f}ms")
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
        logger.error(f"Unexpected error in chart analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze-chart-local", response_model=AnalysisResponse)
async def analyze_chart_local_only(
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
async def get_demo():
    """Get a demo analysis result"""
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
async def get_demo_local():
    """Get a demo analysis result using LOCAL ANALYSIS ONLY (no LLM calls)"""
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
async def get_config():
    """Get current configuration (without sensitive data)"""
    return {
        "api_version": enhanced_settings.API_VERSION,
        "max_file_size_mb": enhanced_settings.MAX_FILE_SIZE // (1024 * 1024),
        "min_chart_size": enhanced_settings.MIN_CHART_SIZE,
        "max_chart_size": enhanced_settings.MAX_CHART_SIZE,
        "llm_enabled": enhanced_settings.LLM_ENABLED,
        "force_local_mode": enhanced_settings.FORCE_LOCAL_MODE,
        "llm_provider": "OpenAI" if enhanced_settings.OPENAI_API_KEY else "Claude" if enhanced_settings.CLAUDE_API_KEY else "Local" if enhanced_settings.LOCAL_LLM_ENABLED else "None",
        "caching_enabled": enhanced_settings.CACHE_ENABLED,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

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
