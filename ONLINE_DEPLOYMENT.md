# Online Deployment Guide

This guide covers deploying your Stock Chart Analysis API online so frontend websites can use it.

## 🚀 **Quick Deploy (Recommended for Starters)**

### **Option 1: Render (Free Tier)**

**Step 1: Prepare Your Code**
```bash
# Ensure all files are committed to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main
```

**Step 2: Deploy on Render**
1. Go to [render.com](https://render.com) and sign up
2. Click "New Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `stock-chart-analyzer`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

**Step 3: Get Your API URL**
- Render will give you a URL like: `https://your-app.onrender.com`
- Your API will be available at: `https://your-app.onrender.com/analyze-chart`

### **Option 2: Railway (Free Tier)**

**Step 1: Deploy on Railway**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and deploys

**Step 2: Get Your API URL**
- Railway provides a URL like: `https://your-app.railway.app`
- Your endpoints: `https://your-app.railway.app/analyze-chart`

## 🌐 **Production Deployment**

### **Option 3: Heroku (Paid but Reliable)**

**Step 1: Install Heroku CLI**
```bash
# Windows: Download from heroku.com
# macOS: brew install heroku
# Linux: curl https://cli-assets.heroku.com/install.sh | sh
```

**Step 2: Deploy**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Add buildpack for Python
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open your app
heroku open
```

**Step 3: Get Your API URL**
- Your API: `https://your-app-name.herokuapp.com/analyze-chart`

### **Option 4: DigitalOcean App Platform**

**Step 1: Deploy on DigitalOcean**
1. Go to [digitalocean.com](https://digitalocean.com)
2. Create account and add payment method
3. Go to "Apps" → "Create App"
4. Connect GitHub repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Basic ($5/month)

**Step 2: Get Your API URL**
- Your API: `https://your-app.ondigitalocean.app/analyze-chart`

## 🔧 **Frontend Integration**

### **JavaScript/Frontend Usage**

Once deployed, your frontend can call the API like this:

```javascript
// Example: Upload chart image for analysis
async function analyzeChart(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
        const response = await fetch('https://your-api-url.onrender.com/analyze-chart', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log('Analysis result:', result);
        
        // Display results
        displayAnalysis(result.analysis);
        
    } catch (error) {
        console.error('Error analyzing chart:', error);
    }
}

// Example: Get demo analysis
async function getDemoAnalysis() {
    try {
        const response = await fetch('https://your-api-url.onrender.com/demo');
        const result = await response.json();
        console.log('Demo analysis:', result);
    } catch (error) {
        console.error('Error getting demo:', error);
    }
}
```

### **CORS Configuration**

Your API is already configured to allow frontend requests. For production, update `config.py`:

```python
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://www.yourdomain.com",
    "https://your-frontend-app.vercel.app",  # If using Vercel
    "https://your-frontend-app.netlify.app", # If using Netlify
]
```

## 📱 **Testing Your Deployed API**

### **Test Endpoints**

```bash
# Health check
curl https://your-api-url.onrender.com/health

# Get info
curl https://your-api-url.onrender.com/info

# Demo analysis
curl https://your-api-url.onrender.com/demo
```

### **Test Chart Analysis**

```bash
# Upload a chart image
curl -X POST "https://your-api-url.onrender.com/analyze-chart" \
     -H "accept: application/json" \
     -F "file=@your_chart.png"
```

## 💰 **Cost Comparison**

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| **Render** | ✅ Yes | $7/month+ | Starters, testing |
| **Railway** | ✅ Yes | $5/month+ | Development, small apps |
| **Heroku** | ❌ No | $7/month+ | Production, reliability |
| **DigitalOcean** | ❌ No | $5/month+ | Production, control |
| **AWS Lambda** | ✅ Yes | Pay-per-use | Serverless, scaling |

## 🚨 **Important Notes**

### **Free Tier Limitations**
- **Render**: 750 hours/month (usually sufficient)
- **Railway**: $5 credit/month
- **Heroku**: No free tier anymore

### **Performance Considerations**
- Free tiers may have cold starts
- Image processing can be slow on limited resources
- Consider upgrading for production use

### **Security**
- Update CORS origins for production
- Consider adding rate limiting
- Monitor API usage

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Build Failures**
   ```bash
   # Check requirements.txt compatibility
   pip install -r requirements.txt --dry-run
   ```

2. **CORS Errors**
   - Verify `ALLOWED_ORIGINS` in `config.py`
   - Check frontend domain is included

3. **Memory Issues**
   - Reduce `MAX_FILE_SIZE` in `config.py`
   - Optimize image processing

4. **Timeout Errors**
   - Increase timeout settings on deployment platform
   - Optimize analysis algorithms

### **Debug Mode**

```python
# In config.py, set:
LOG_LEVEL = "DEBUG"

# Or via environment variable:
export LOG_LEVEL=DEBUG
```

## 📈 **Scaling Up**

### **When to Upgrade**
- **Free tier limits reached**
- **High traffic (>100 requests/hour)**
- **Production use**
- **Need better performance**

### **Upgrade Options**
1. **Paid plans** on same platform
2. **Move to more powerful platform**
3. **Add caching** (Redis)
4. **Load balancing** for multiple instances

## 🎯 **Next Steps**

1. **Choose deployment platform** (start with Render/Railway)
2. **Deploy your API**
3. **Test all endpoints**
4. **Integrate with frontend**
5. **Monitor performance**
6. **Scale as needed**

## 📞 **Support**

- **Render**: [docs.render.com](https://docs.render.com)
- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **DigitalOcean**: [docs.digitalocean.com](https://docs.digitalocean.com)

Your API will be accessible worldwide once deployed! 🚀
