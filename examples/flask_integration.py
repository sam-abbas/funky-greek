#!/usr/bin/env python3
"""
Flask Frontend Integration Example
Shows how to integrate the chart analyzer API with a Flask frontend
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Configuration
API_BASE_URL = 'https://your-api.onrender.com'  # Replace with your API URL
ENDPOINT = '/analyze-chart-local'  # Use local analysis (no API key needed)

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_chart():
    """Handle chart analysis request"""
    try:
        # Check if file was uploaded
        if 'chart_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['chart_image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Prepare file for API request
        files = {'chart_image': (file.filename, file.stream, file.content_type)}
        
        # Make request to chart analyzer API
        response = requests.post(
            f"{API_BASE_URL}{ENDPOINT}",
            files=files,
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return jsonify({
                    'success': True,
                    'analysis': data['analysis']
                })
            else:
                return jsonify({'error': data.get('error', 'Analysis failed')}), 500
        else:
            return jsonify({'error': f'API request failed with status {response.status_code}'}), 500
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout. Please try again.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return jsonify({'status': 'healthy', 'api_connected': True})
        else:
            return jsonify({'status': 'unhealthy', 'api_connected': False}), 500
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'api_connected': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
