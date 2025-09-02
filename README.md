# Stock Chart Analysis API

A cost-effective backend service that analyzes stock chart images and provides comprehensive technical analysis using computer vision and technical indicators.

## Features

- **Image Upload Analysis**: Upload stock chart images for instant analysis
- **Technical Indicators**: RSI, MACD, Moving Averages, and more
- **Pattern Recognition**: Detects chart patterns like double tops/bottoms, trend continuations
- **Support/Resistance**: Identifies key price levels
- **Trend Analysis**: Short and medium-term trend assessment
- **Trading Advice**: Actionable recommendations with risk assessment
- **Cost-Effective**: Uses free/open-source libraries, no expensive API calls

## Technical Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **OpenCV**: Computer vision for image processing
- **PIL/Pillow**: Image handling
- **TA-Lib**: Technical analysis indicators
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd quickcopysnaptrade
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Service

```bash
python main.py
```

The service will start on `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```
GET /health
```
Returns service status.

#### 2. Analyze Chart
```
POST /analyze-chart
```
Upload a stock chart image for analysis.

**Request**: Form data with image file
**Response**: Comprehensive technical analysis

#### 3. API Documentation
```
GET /docs
```
Interactive API documentation (Swagger UI)

## Example Usage

### Using curl
```bash
curl -X POST "http://localhost:8000/analyze-chart" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_chart.png"
```

### Using Python requests
```python
import requests

url = "http://localhost:8000/analyze-chart"
files = {"file": open("chart.png", "rb")}
response = requests.post(url, files=files)
analysis = response.json()
print(analysis)
```

## Analysis Output

The service provides:

- **Overall Sentiment**: Bullish, Bearish, or Neutral
- **Confidence Score**: 0.0 to 1.0
- **Technical Indicators**: RSI, MACD, Moving Averages with signals
- **Chart Patterns**: Detected patterns with confidence levels
- **Support/Resistance Levels**: Key price levels
- **Trend Analysis**: Market direction and strength
- **Trading Advice**: Actionable recommendations
- **Risk Assessment**: Low, Medium, or High risk
- **Stop Loss Suggestions**: Recommended stop loss levels
- **Take Profit Targets**: Price targets for profit taking

## Cost Optimization Features

1. **No External API Dependencies**: All analysis is performed locally
2. **Open Source Libraries**: Uses free technical analysis libraries
3. **Efficient Processing**: Optimized algorithms for quick analysis
4. **Scalable Architecture**: Can handle multiple requests efficiently

## Development

### Project Structure
```
├── main.py              # FastAPI application
├── chart_analyzer.py    # Core analysis logic
├── models.py            # Pydantic data models
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Indicators

To add new technical indicators, modify the `_calculate_indicators` method in `ChartAnalyzer`:

```python
def _calculate_indicators(self, price_data: pd.DataFrame) -> list[TechnicalIndicator]:
    # ... existing code ...
    
    # Add new indicator
    new_indicator = NewIndicator(price_data['close'])
    # ... calculation logic ...
    
    indicators.append(TechnicalIndicator(
        name="New Indicator",
        value=value,
        signal=signal,
        strength=strength,
        description=description
    ))
```

### Adding New Patterns

To add new chart patterns, modify the `_detect_patterns` method:

```python
def _detect_patterns(self, cv_image, price_data: pd.DataFrame) -> list[ChartPattern]:
    # ... existing code ...
    
    # Detect new pattern
    if self._detect_new_pattern(prices):
        patterns.append(ChartPattern(
            name="New Pattern",
            confidence=confidence,
            signal=signal,
            description=description
        ))
```

## Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Limitations

1. **Image Quality**: Analysis accuracy depends on chart image quality
2. **Data Extraction**: Currently uses simulated price data (can be enhanced with OCR)
3. **Pattern Recognition**: Basic pattern detection (can be enhanced with ML models)
4. **Real-time Data**: No real-time market data integration

## Future Enhancements

1. **OCR Integration**: Extract actual price data from chart images
2. **Machine Learning**: Enhanced pattern recognition using ML models
3. **Real-time Data**: Integration with market data providers
4. **Advanced Patterns**: More sophisticated chart pattern detection
5. **Backtesting**: Historical performance analysis
6. **User Management**: Multi-user support with analysis history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This service is for educational and informational purposes only. It does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions. The analysis provided is based on technical indicators and should not be the sole basis for trading decisions.
