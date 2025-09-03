# 🚀 Frontend Integration Guide

## Available API Endpoints

### **Core Analysis Endpoints**
- `POST /analyze-chart` - Full analysis with LLM (requires API key)
- `POST /analyze-chart-local` - Local analysis only (no API key needed)
- `GET /health` - Health check
- `GET /info` - API information
- `GET /config` - Configuration (sanitized)

### **Request Format**
```javascript
// For local analysis (recommended for external frontends)
const formData = new FormData();
formData.append('chart_image', fileInput.files[0]);

const response = await fetch('https://your-api.onrender.com/analyze-chart-local', {
    method: 'POST',
    body: formData
});
```

### **Response Format**
```json
{
    "success": true,
    "analysis": {
        "confidence": 0.85,
        "trend": "bullish",
        "patterns": [...],
        "indicators": [...],
        "support_resistance": [...],
        "fair_value_gaps": [...],
        "daily_levels": [...],
        "price_action": [...]
    },
    "processing_time": 2.3,
    "llm_enhanced": false
}
```

## Integration Methods

### **1. Vanilla JavaScript**
### **2. React**
### **3. Vue.js**
### **4. Angular**
### **5. Python (Flask/Django)**
