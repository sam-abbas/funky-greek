# 🚀 Enhanced Stock Chart Analyzer - Deployment Guide

## 🎯 **What's New in Enhanced Version**

### **Real Technical Analysis (Not Fake Data!)**
- ✅ **Actual price extraction** from chart images
- ✅ **Real RSI, MACD, Moving Averages** calculations
- ✅ **Pattern detection** based on actual data
- ✅ **Different results** for different chart types

### **Advanced Image Processing**
- ✅ **Non-chart item filtering** (ignores text, noise, UI elements)
- ✅ **Professional computer vision** techniques
- ✅ **Robust chart detection** and validation
- ✅ **High-quality price data extraction**

### **LLM Integration Ready**
- ✅ **OpenAI GPT-4 Vision** support
- ✅ **Claude 3 Vision** alternative
- ✅ **Local LLM** fallback options
- ✅ **Cost optimization** strategies

## 🛠️ **Installation & Setup**


### **Step 1: Install Enhanced Dependencies**
```bash
# Install enhanced requirements
pip install -r requirements_enhanced.txt

# Or install core packages individually
pip install fastapi uvicorn pillow opencv-python-headless pandas numpy ta scikit-image
```

### **Step 2: Set Up Environment Variables**
Create a `.env` file:
```bash
# .env
OPENAI_API_KEY=your-openai-api-key-here
ENVIRONMENT=production
LOG_LEVEL=INFO

# Optional: Claude integration
CLAUDE_API_KEY=your-claude-api-key

# Optional: Local LLM
LOCAL_LLM_ENABLED=false
LOCAL_LLM_URL=http://localhost:11434

# Force local analysis only (no LLM calls)
FORCE_LOCAL_MODE=false
```

### **Step 3: Test Enhanced Analysis**
```bash
# Test local analysis functionality
python test_local_analysis.py

# Test enhanced analysis with LLM
python test_enhanced_comprehensive.py

# Or run basic tests
python test_enhanced_analysis.py
```

### **Step 4: Test Local Analysis Tools**
```bash
# Analyze a single chart locally
python local_analysis.py chart.png --verbose

# Analyze multiple charts locally
python batch_local_analysis.py charts/ --output results/ --recursive

# Get help
python local_analysis.py --help
python batch_local_analysis.py --help
```

## 🚀 **Deployment Options**

### **Local Analysis Only Mode**
Before deploying, you can test the system in **local analysis only mode** without any LLM API calls:

```bash
# Test local analysis functionality
python test_local_analysis.py

# Analyze a single chart locally
python local_analysis.py chart.png

# Analyze multiple charts locally
python batch_local_analysis.py charts/ --output results/

# Start server in local-only mode
export FORCE_LOCAL_MODE=true
python main_enhanced.py
```

**Local Mode Benefits:**
- ✅ **No API costs** - All analysis done locally
- ✅ **Faster processing** - No network latency
- ✅ **Privacy focused** - No data sent to external services
- ✅ **Offline capable** - Works without internet connection
- ✅ **Cost optimization** - Perfect for development and testing

### **Security Features (Production Ready)**

**🔒 Comprehensive Security Implementation:**
- ✅ **Rate Limiting** - 100 requests/hour per IP address
- ✅ **CORS Protection** - Restricted origins and methods
- ✅ **Security Headers** - XSS, CSRF, Clickjacking protection
- ✅ **Input Validation** - File type, size, and content validation
- ✅ **Response Sanitization** - No sensitive data exposed
- ✅ **Trusted Host Validation** - Prevents host header attacks
- ✅ **Error Handling** - Generic error messages, no internal details
- ✅ **Request Logging** - IP-based monitoring without sensitive data

**Security Testing:**
```bash
# Run comprehensive security tests
python test_security.py

# Test against deployed service
python test_security.py https://your-app.onrender.com
```

**Security Configuration:**
```bash
# Environment variables for production
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
SECRET_KEY=your-super-secret-key-here
RATE_LIMIT_PER_HOUR=100
MAX_REQUESTS_PER_MINUTE=20
```

### **Option 1: Render (Recommended)**
```yaml
# render.yaml
services:
  - type: web
    name: enhanced-stock-analyzer
    env: python
    buildCommand: pip install -r requirements_enhanced.txt
    startCommand: uvicorn main_enhanced:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        value: ${{OPENAI_API_KEY}}
      - key: ENVIRONMENT
        value: production
      - key: PYTHON_VERSION
        value: 3.11.0
```

### **Option 2: Railway**
```bash
# Deploy to Railway
railway login
railway init
railway up

# Set environment variables
railway variables set OPENAI_API_KEY=your-key
railway variables set ENVIRONMENT=production
```

### **Option 3: Heroku**
```bash
# Create Procfile
echo "web: uvicorn main_enhanced:app --host=0.0.0.0 --port=\$PORT" > Procfile

# Deploy
git add .
git commit -m "Enhanced chart analyzer deployment"
git push heroku main

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key
heroku config:set ENVIRONMENT=production
```

### **Option 4: Docker**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main_enhanced:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔧 **Configuration Options**

### **Performance Settings**
```python
# In config_enhanced.py
MAX_PROCESSING_TIME = 30000  # 30 seconds
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour

# Image processing settings
MIN_CHART_SIZE = 100
MAX_CHART_SIZE = 4000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Local analysis mode
FORCE_LOCAL_MODE = False  # Set to True to disable all LLM calls
```

### **LLM Settings**
```python
# OpenAI integration
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-4-vision-preview"
OPENAI_MAX_TOKENS = 1000

# Claude integration
CLAUDE_API_KEY = "your-key"
CLAUDE_MODEL = "claude-3-sonnet-20240229"

# Local LLM
LOCAL_LLM_ENABLED = True
LOCAL_LLM_URL = "http://localhost:11434"
LOCAL_LLM_MODEL = "llama2:13b"
```

## 📊 **Testing Your Deployment**

### **Local Analysis Endpoints**
```bash
# Test local analysis only (no LLM calls)
curl -X POST https://your-app.onrender.com/analyze-chart-local \
  -F "file=@your_chart.png"

# Get local demo analysis
curl https://your-app.onrender.com/demo-local

# Check available endpoints
curl https://your-app.onrender.com/info
```

### **Standard Endpoints**

### **Health Check**
```bash
curl https://your-app.onrender.com/health
```

### **Info Endpoint**
```bash
curl https://your-app.onrender.com/info
```

### **Demo Analysis**
```bash
curl https://your-app.onrender.com/demo
```

### **Test with Real Chart**
```bash
# Upload a chart image
curl -X POST https://your-app.onrender.com/analyze-chart \
  -F "file=@your_chart.png"
```

## 💰 **Cost Optimization**

### **Hybrid Analysis Strategy**
```python
# Use local analysis for basic patterns
if basic_analysis_sufficient:
    return local_analysis_result
else:
    # Use LLM for complex patterns
    return llm_enhanced_analysis
```

### **Caching Implementation**
```python
from cachetools import TTLCache
import hashlib

# Cache analysis results
analysis_cache = TTLCache(maxsize=100, ttl=3600)

def analyze_chart_cached(image):
    # Create image hash
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    
    # Check cache
    if image_hash in analysis_cache:
        return analysis_cache[image_hash]
    
    # Perform analysis
    result = analyzer.analyze_chart(image)
    
    # Cache result
    analysis_cache[image_hash] = result
    
    return result
```

### **Batch Processing**
```python
def batch_analyze_charts(charts):
    """Process multiple charts in single LLM call"""
    if not enhanced_settings.LLM_ENABLED:
        return [analyzer.analyze_chart(chart) for chart in charts]
    
    # Single LLM call for multiple charts
    return llm_batch_analysis(charts)
```

## 🔍 **Monitoring & Debugging**

### **Logging Configuration**
```python
# Enhanced logging
logging.basicConfig(
    level=enhanced_settings.LOG_LEVEL,
    format=enhanced_settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler('enhanced_analyzer.log'),
        logging.StreamHandler()
    ]
)
```

### **Performance Monitoring**
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        logger.info(f"{func.__name__} completed in {processing_time:.2f}ms")
        
        return result
    return wrapper

@monitor_performance
def analyze_chart(image):
    return analyzer.analyze_chart(image)
```

### **Error Tracking**
```python
import traceback

def safe_analyze_chart(image):
    try:
        return analyzer.analyze_chart(image)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return fallback analysis
        return generate_fallback_analysis(image)
```

## 🔒 **Security Checklist (MANDATORY)**

### **Pre-Deployment Security Verification**
```bash
# 1. Run comprehensive security tests
python test_security.py

# 2. Verify no sensitive data exposure
curl -s http://localhost:8000/info | grep -i "api_key\|secret\|password"

# 3. Test rate limiting
for i in {1..105}; do curl -s http://localhost:8000/health; done

# 4. Check security headers
curl -I http://localhost:8000/health | grep -E "(X-Content-Type-Options|X-Frame-Options|X-XSS-Protection)"
```

### **Security Requirements Checklist**
- [ ] **Rate Limiting**: 100 requests/hour per IP enforced
- [ ] **CORS Security**: Restricted origins and methods configured
- [ ] **Security Headers**: All required headers present
- [ ] **Input Validation**: File type, size, and content validated
- [ ] **Response Sanitization**: No sensitive data in responses
- [ ] **Error Handling**: Generic error messages only
- [ ] **Logging**: No sensitive data in logs
- [ ] **Host Validation**: Trusted host middleware active

### **Production Security Settings**
```bash
# Environment variables for production
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
SECRET_KEY=your-super-secret-key-here
RATE_LIMIT_PER_HOUR=100
MAX_REQUESTS_PER_MINUTE=20
ENVIRONMENT=production
LOG_LEVEL=WARNING
```

### **Security Monitoring**
- Monitor rate limit violations
- Track failed authentication attempts
- Watch for unusual request patterns
- Review error logs regularly
- Set up security alerts

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. OpenCV Installation Problems**
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Use headless version
pip install opencv-python-headless
```

#### **2. Memory Issues**
```python
# Reduce image size before processing
def preprocess_image(image, max_size=1200):
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image
```

#### **3. LLM API Errors**
```python
# Implement retry logic
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def llm_analysis_with_retry(image):
    return llm_analysis(image)
```

## 📈 **Performance Benchmarks**

### **Expected Performance**
- **Small charts (400x200)**: ~50-100ms
- **Medium charts (800x400)**: ~100-200ms
- **Large charts (1200x600)**: ~200-400ms
- **With LLM**: +500-1000ms

### **Optimization Tips**
1. **Resize large images** before processing
2. **Use caching** for repeated analyses
3. **Implement background processing** for LLM calls
4. **Monitor memory usage** and implement cleanup

## 🎯 **Next Steps After Deployment**

1. **Test all endpoints** with real chart images
2. **Monitor performance** and adjust settings
3. **Set up LLM integration** if desired
4. **Implement caching** for production use
5. **Add monitoring** and alerting
6. **Scale based on usage** patterns

## 💡 **Pro Tips**

- **Start with local analysis** before adding LLM costs
- **Use image preprocessing** to improve analysis quality
- **Implement rate limiting** to control costs
- **Monitor API usage** to optimize spending
- **Test with various chart types** to ensure robustness

Your enhanced chart analyzer is now ready to provide **real, actionable trading insights** based on actual chart data! 🎉
