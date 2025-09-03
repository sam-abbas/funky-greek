import React, { useState, useCallback } from 'react';

const ReactChartAnalyzer = () => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    // Configurationhttps://your-api.onrender.com
    const API_BASE_URL = 'https://funky-greek.onrender.com'; // Replace with your API URL
    const ENDPOINT = '/analyze-chart'; // Use main analysis endpoint (works on current deployment)

    const handleFileChange = useCallback((e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setError(null);
            setResults(null);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && droppedFile.type.startsWith('image/')) {
            setFile(droppedFile);
            setError(null);
            setResults(null);
        }
    }, []);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
    }, []);

    const analyzeChart = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('chart_image', file);

            const response = await fetch(`${API_BASE_URL}${ENDPOINT}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                setResults(data.analysis);
            } else {
                setError(data.error || 'Analysis failed');
            }

        } catch (err) {
            setError(`Failed to analyze chart: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
            <h1>📊 Chart Analyzer - React</h1>

            {/* File Upload */}
            <div
                style={{
                    border: '2px dashed #ccc',
                    padding: '40px',
                    textAlign: 'center',
                    margin: '20px 0',
                    borderRadius: '5px',
                }}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
            >
                <p>📁 Drag & drop your chart image here or select a file</p>
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ margin: '10px' }}
                />
                {file && (
                    <div>
                        <p>Selected: {file.name}</p>
                        <button
                            onClick={analyzeChart}
                            disabled={loading}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: loading ? '#ccc' : '#007bff',
                                color: 'white',
                                border: 'none',
                                borderRadius: '5px',
                                cursor: loading ? 'not-allowed' : 'pointer',
                            }}
                        >
                            {loading ? 'Analyzing...' : 'Analyze Chart'}
                        </button>
                    </div>
                )}
            </div>

            {/* Error Display */}
            {error && (
                <div style={{ color: 'red', padding: '10px', backgroundColor: '#ffe6e6', borderRadius: '5px' }}>
                    ❌ {error}
                </div>
            )}

            {/* Results Display */}
            {results && (
                <div style={{ padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
                    <h3>✅ Analysis Results</h3>

                    <div style={{ marginBottom: '20px' }}>
                        <strong>Confidence:</strong> {(results.confidence * 100).toFixed(1)}%
                        <br />
                        <strong>Trend:</strong> {results.trend}
                        <br />
                        <strong>Processing Time:</strong> {results.processing_time}s
                    </div>

                    {/* Patterns */}
                    {results.patterns && results.patterns.length > 0 && (
                        <div>
                            <h4>🎯 Patterns Detected</h4>
                            <ul>
                                {results.patterns.map((pattern, index) => (
                                    <li key={index}>
                                        {pattern.name} ({(pattern.confidence * 100).toFixed(1)}%)
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Technical Indicators */}
                    {results.indicators && results.indicators.length > 0 && (
                        <div>
                            <h4>📊 Technical Indicators</h4>
                            <ul>
                                {results.indicators.map((indicator, index) => (
                                    <li key={index}>
                                        {indicator.name}: {indicator.value}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Support & Resistance */}
                    {results.support_resistance && results.support_resistance.length > 0 && (
                        <div>
                            <h4>🔍 Support & Resistance</h4>
                            <ul>
                                {results.support_resistance.map((sr, index) => (
                                    <li key={index}>
                                        {sr.type}: {sr.price}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Fair Value Gaps */}
                    {results.fair_value_gaps && results.fair_value_gaps.length > 0 && (
                        <div>
                            <h4>💰 Fair Value Gaps</h4>
                            <ul>
                                {results.fair_value_gaps.map((fvg, index) => (
                                    <li key={index}>
                                        Gap: {fvg.start_price} - {fvg.end_price}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Daily Levels */}
                    {results.daily_levels && results.daily_levels.length > 0 && (
                        <div>
                            <h4>📅 Daily Levels</h4>
                            <ul>
                                {results.daily_levels.map((level, index) => (
                                    <li key={index}>
                                        {level.type}: {level.price}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ReactChartAnalyzer;
