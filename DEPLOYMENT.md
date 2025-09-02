# Deployment Guide

This guide covers deploying the Stock Chart Analysis API in various environments.

## Local Development

### Prerequisites
- Python 3.8+
- pip
- Virtual environment support

### Quick Start
```bash
# Clone and setup
git clone <your-repo>
cd quickcopysnaptrade

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start service
python main.py
```

### Using Scripts
- **Windows**: Double-click `start_service.bat`
- **macOS/Linux**: Run `./start_service.sh`

## Production Deployment

### Using Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables
```bash
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4
export LOG_LEVEL=INFO
```

### Using Gunicorn (Linux/macOS)
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Docker Deployment

### Build Image
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run
```bash
# Build
docker build -t stock-chart-analyzer .

# Run
docker run -p 8000:8000 stock-chart-analyzer

# Run with environment variables
docker run -p 8000:8000 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e LOG_LEVEL=INFO \
  stock-chart-analyzer
```

## Cloud Deployment

### AWS EC2
1. Launch EC2 instance (t3.micro for testing, t3.small+ for production)
2. Install Python and dependencies
3. Use systemd service for auto-start

### AWS Lambda
1. Package as Lambda deployment package
2. Configure API Gateway
3. Set memory to 512MB+ for image processing

### Google Cloud Run
1. Build Docker image
2. Deploy to Cloud Run
3. Configure scaling and memory

### Heroku
1. Add `Procfile`:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
2. Deploy using Heroku CLI or GitHub integration

## Performance Optimization

### Worker Configuration
- **Development**: 1 worker
- **Production**: 2-4 workers per CPU core
- **High Load**: 8+ workers with load balancer

### Memory Optimization
- Set `MAX_FILE_SIZE` based on expected chart sizes
- Monitor memory usage during analysis
- Consider image compression for large charts

### Caching (Future Enhancement)
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Cache analysis results
@cache(expire=3600)  # 1 hour
async def analyze_chart_cached(file: UploadFile):
    # ... analysis logic
```

## Monitoring and Logging

### Health Checks
```bash
curl http://localhost:8000/health
```

### Logging Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about operations
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations

### Metrics (Future Enhancement)
```python
from prometheus_client import Counter, Histogram

# Track analysis requests
analysis_requests = Counter('analysis_requests_total', 'Total analysis requests')
analysis_duration = Histogram('analysis_duration_seconds', 'Analysis duration')
```

## Security Considerations

### CORS Configuration
```python
# Restrict origins in production
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://app.yourdomain.com"
]
```

### Rate Limiting (Future Enhancement)
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/analyze-chart")
@limiter.limit("10/minute")
async def analyze_chart(file: UploadFile):
    # ... analysis logic
```

### File Validation
- Validate file types and sizes
- Scan uploaded files for malware
- Implement virus scanning for production

## Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple instances
- Use container orchestration (Kubernetes, Docker Swarm)

### Vertical Scaling
- Increase CPU and memory
- Use more powerful instance types
- Optimize algorithms for better performance

## Backup and Recovery

### Configuration Backup
- Version control for configuration files
- Environment-specific configs
- Backup database/state if added

### Disaster Recovery
- Multi-region deployment
- Automated failover
- Regular backup testing

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   netstat -tulpn | grep :8000
   
   # Kill process
   kill -9 <PID>
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   htop
   
   # Check Python memory
   pip install memory-profiler
   ```

3. **Dependency Issues**
   ```bash
   # Reinstall dependencies
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

### Debug Mode
```bash
# Set debug logging
export LOG_LEVEL=DEBUG

# Start with debug
uvicorn main:app --reload --log-level debug
```

## Support

For deployment issues:
1. Check logs for error messages
2. Verify environment configuration
3. Test with minimal configuration
4. Check system resources (CPU, memory, disk)
