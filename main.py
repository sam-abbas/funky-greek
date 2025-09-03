from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import time
from PIL import Image
import numpy as np
import cv2
from chart_analyzer import ChartAnalyzer
from models import AnalysisRequest, AnalysisResponse
from config import settings
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chart analyzer
analyzer = ChartAnalyzer()

@app.get("/")
async def root():
    return {"message": "Stock Chart Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "chart-analysis",
        "version": settings.API_VERSION,
        "timestamp": time.time()
    }

@app.get("/info")
async def get_info():
    """Get service information and configuration"""
    return {
        "service": "Stock Chart Analysis API",
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "features": [
            "Technical indicator calculation",
            "Chart pattern recognition", 
            "Support/resistance detection",
            "Trend analysis",
            "Trading advice generation"
        ],
        "cost_optimization": [
            "No external API dependencies",
            "Open source libraries",
            "Local processing",
            "Efficient algorithms"
        ]
    }

@app.post("/analyze-chart", response_model=AnalysisResponse)
async def analyze_chart(file: UploadFile = File(...)):
    """
    Analyze a stock chart image and provide technical analysis
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image. Received: {file.content_type}"
            )
        
        # Validate file size
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File size ({len(contents) // (1024*1024)}MB) exceeds maximum allowed size of {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validate file is not empty
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate image dimensions
        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Image too small. Minimum size: 100x100, received: {image.size[0]}x{image.size[1]}"
            )
        
        # Analyze the chart
        analysis_result = analyzer.analyze_chart(image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Chart analysis completed in {processing_time:.2f}ms")
        
        return AnalysisResponse(
            success=True,
            analysis=analysis_result,
            message="Chart analysis completed successfully",
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error analyzing chart: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@app.post("/analyze-chart-url")
async def analyze_chart_url(request: AnalysisRequest):
    """
    Analyze a stock chart from URL (placeholder for future implementation)
    """
    if not request.chart_url:
        raise HTTPException(status_code=400, detail="Chart URL is required")
    
    return {
        "message": "URL analysis not yet implemented",
        "status": "development",
        "requested_url": request.chart_url,
        "analysis_type": request.analysis_type
    }

@app.get("/demo")
async def get_demo_analysis():
    """
    Get a demo analysis result for testing purposes
    """
    try:
        # Create a simple demo chart
        demo_image = Image.new('RGB', (400, 300), 'white')
        
        # Analyze the demo chart
        analysis_result = analyzer.analyze_chart(demo_image)
        
        return AnalysisResponse(
            success=True,
            analysis=analysis_result,
            message="Demo analysis completed successfully",
            processing_time_ms=0.0
        )
        
    except Exception as e:
        logger.error(f"Error in demo analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Demo analysis failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

