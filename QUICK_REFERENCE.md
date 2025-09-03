# 🚀 **Quick Reference Card**

## **Essential API Calls**

### **1. Local Analysis (Recommended)**
```javascript
const formData = new FormData();
formData.append('chart_image', file);

const response = await fetch('https://your-api.onrender.com/analyze-chart-local', {
    method: 'POST',
    body: formData
});
```

### **2. Health Check**
```javascript
const response = await fetch('https://your-api.onrender.com/health');
```

### **3. API Info**
```javascript
const response = await fetch('https://your-api.onrender.com/info');
```

## **Key Features Summary**

| Feature | Description | Endpoint |
|---------|-------------|----------|
| **Pattern Detection** | 9+ chart patterns | `/analyze-chart-local` |
| **Technical Indicators** | RSI, MACD, SMA, Bollinger Bands | `/analyze-chart-local` |
| **Support/Resistance** | Dynamic & static levels | `/analyze-chart-local` |
| **Fair Value Gaps** | Price inefficiencies | `/analyze-chart-local` |
| **Order Blocks** | Institutional zones | `/analyze-chart-local` |
| **Market Structure** | BOS, CHoCH analysis | `/analyze-chart-local` |
| **Risk Management** | Stop loss & take profit | `/analyze-chart-local` |

## **Response Structure**
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
        "price_action": [...],
        "order_blocks": [...],
        "liquidity_zones": [...],
        "market_structure": [...],
        "stop_loss_suggestions": [...],
        "take_profit_targets": [...]
    },
    "processing_time": 2.3,
    "llm_enhanced": false
}
```

## **Security Features**
- ✅ **Rate Limiting**: 100 requests/hour per IP
- ✅ **CORS Protection**: Configurable origins
- ✅ **Input Validation**: File type, size, dimensions
- ✅ **Security Headers**: XSS, CSRF, Clickjacking protection
- ✅ **Data Sanitization**: No sensitive data exposure

## **File Requirements**
- **Formats**: PNG, JPG, JPEG, GIF, BMP
- **Size**: Max 10MB
- **Dimensions**: 100x100 to 4000x4000 pixels
- **Content**: Valid image file

## **Error Codes**
- **200**: Success
- **400**: Bad Request (invalid file)
- **429**: Too Many Requests (rate limited)
- **500**: Internal Server Error
- **504**: Gateway Timeout

## **Integration Examples**
- **Vanilla JS**: `examples/vanilla_js_integration.html`
- **React**: `examples/ReactChartAnalyzer.jsx`
- **Flask**: `examples/flask_integration.py`
- **Complete Guide**: `frontend_integration_guide.md`
