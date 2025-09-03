#!/usr/bin/env python3
"""
Security Log Analysis Tool
Analyze your server logs for security events and patterns
"""

import re
from collections import Counter
from datetime import datetime

def analyze_security_logs(log_file_path="server.log"):
    """Analyze security-related log entries"""
    
    print("🔍 Security Log Analysis")
    print("=" * 50)
    
    # Security event patterns
    patterns = {
        'rate_limit_exceeded': r'Rate limit exceeded for IP: (\d+\.\d+\.\d+\.\d+)',
        'invalid_requests': r'(\d+\.\d+\.\d+\.\d+) - ".*" (4\d{2})',
        'server_errors': r'(\d+\.\d+\.\d+\.\d+) - ".*" (5\d{2})',
        'suspicious_headers': r'X-Forwarded-For: (.+)',
        'file_uploads': r'POST /analyze-chart',
        'authentication_attempts': r'Authorization: (.+)'
    }
    
    # Counters for analysis
    ip_requests = Counter()
    rate_limit_violations = Counter()
    error_codes = Counter()
    suspicious_ips = set()
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Count requests by IP
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+) - "', line)
                if ip_match:
                    ip = ip_match.group(1)
                    ip_requests[ip] += 1
                
                # Check for rate limit violations
                if 'Rate limit exceeded' in line:
                    ip_match = re.search(r'Rate limit exceeded for IP: (\d+\.\d+\.\d+\.\d+)', line)
                    if ip_match:
                        ip = ip_match.group(1)
                        rate_limit_violations[ip] += 1
                        suspicious_ips.add(ip)
                
                # Check for error codes
                error_match = re.search(r'(\d+\.\d+\.\d+\.\d+) - ".*" (\d{3})', line)
                if error_match:
                    status_code = error_match.group(2)
                    if status_code.startswith('4') or status_code.startswith('5'):
                        error_codes[status_code] += 1
        
        # Generate report
        print(f"📊 Analysis Results:")
        print(f"   Total unique IPs: {len(ip_requests)}")
        print(f"   Total requests: {sum(ip_requests.values())}")
        print(f"   Rate limit violations: {sum(rate_limit_violations.values())}")
        
        print(f"\n🚨 Top 10 Most Active IPs:")
        for ip, count in ip_requests.most_common(10):
            status = "⚠️ SUSPICIOUS" if ip in suspicious_ips else "✅ Normal"
            print(f"   {ip}: {count} requests {status}")
        
        print(f"\n🚫 Rate Limit Violations:")
        for ip, violations in rate_limit_violations.most_common(5):
            print(f"   {ip}: {violations} violations")
        
        print(f"\n❌ Error Codes:")
        for code, count in error_codes.most_common():
            print(f"   {code}: {count} occurrences")
        
        # Security recommendations
        print(f"\n💡 Security Recommendations:")
        
        if len(suspicious_ips) > 0:
            print(f"   🔴 Consider blocking these IPs: {list(suspicious_ips)}")
        
        if error_codes['404'] > 100:
            print(f"   🟡 High 404 errors - check for scanning attempts")
        
        if error_codes['500'] > 10:
            print(f"   🔴 High 500 errors - investigate server issues")
        
        if sum(rate_limit_violations.values()) > 50:
            print(f"   🟡 Consider lowering rate limits or adding IP blocking")
        
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
        print("💡 To capture logs, run: python -m uvicorn main_enhanced:app > server.log 2>&1")

if __name__ == "__main__":
    analyze_security_logs()
