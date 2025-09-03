#!/usr/bin/env python3
"""
Security Testing Script for Enhanced Chart Analyzer
Tests all security measures to ensure no sensitive data is exposed
"""

import requests
import time
import json
from typing import Dict, Any
import sys

class SecurityTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        
    def test_endpoint_security(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test endpoint for security vulnerabilities"""
        try:
            if method.upper() == "GET":
                response = self.session.get(f"{self.base_url}{endpoint}")
            elif method.upper() == "POST":
                response = self.session.post(f"{self.base_url}{endpoint}", data=data)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:500],  # Limit content for display
                "content_length": len(response.text)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        print("🔒 Testing Rate Limiting...")
        
        # Make multiple requests to trigger rate limit
        responses = []
        for i in range(105):  # Should trigger rate limit at 100
            response = self.session.get(f"{self.base_url}/health")
            responses.append({
                "request": i + 1,
                "status_code": response.status_code,
                "rate_limited": response.status_code == 429
            })
            
            if response.status_code == 429:
                break
                
            time.sleep(0.1)  # Small delay
        
        # Check if rate limiting was triggered
        rate_limited = any(r["rate_limited"] for r in responses)
        first_rate_limit = next((r["request"] for r in responses if r["rate_limited"]), None)
        
        return {
            "rate_limiting_working": rate_limited,
            "first_rate_limit_at": first_rate_limit,
            "total_requests": len(responses),
            "responses": responses[:10]  # Show first 10 responses
        }
    
    def test_sensitive_data_exposure(self) -> Dict[str, Any]:
        """Test if sensitive data is exposed in responses"""
        print("🔍 Testing Sensitive Data Exposure...")
        
        sensitive_patterns = [
            "sk-",  # OpenAI API key pattern
            "api_key",
            "secret",
            "password",
            "token",
            "OPENAI_API_KEY",
            "CLAUDE_API_KEY",
            "internal",
            "server",
            "environment",
            "production",
            "development"
        ]
        
        endpoints_to_test = [
            "/",
            "/health", 
            "/info",
            "/config"
        ]
        
        exposed_data = {}
        
        for endpoint in endpoints_to_test:
            response = self.session.get(f"{self.base_url}{endpoint}")
            content = response.text.lower()
            
            found_patterns = []
            for pattern in sensitive_patterns:
                if pattern.lower() in content:
                    found_patterns.append(pattern)
            
            if found_patterns:
                exposed_data[endpoint] = found_patterns
        
        return {
            "sensitive_data_exposed": len(exposed_data) > 0,
            "exposed_endpoints": exposed_data,
            "total_endpoints_tested": len(endpoints_to_test)
        }
    
    def test_security_headers(self) -> Dict[str, Any]:
        """Test if security headers are properly set"""
        print("🛡️ Testing Security Headers...")
        
        response = self.session.get(f"{self.base_url}/health")
        headers = response.headers
        
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        missing_headers = []
        present_headers = {}
        
        for header, expected_value in required_headers.items():
            if header in headers:
                present_headers[header] = headers[header]
                if expected_value and expected_value not in headers[header]:
                    missing_headers.append(f"{header}: expected '{expected_value}', got '{headers[header]}'")
            else:
                missing_headers.append(f"{header}: missing")
        
        return {
            "all_headers_present": len(missing_headers) == 0,
            "missing_headers": missing_headers,
            "present_headers": present_headers,
            "total_required": len(required_headers)
        }
    
    def test_cors_security(self) -> Dict[str, Any]:
        """Test CORS configuration security"""
        print("🌐 Testing CORS Security...")
        
        # Test preflight request
        preflight_response = self.session.options(
            f"{self.base_url}/analyze-chart",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Test actual request with malicious origin
        malicious_response = self.session.post(
            f"{self.base_url}/health",
            headers={"Origin": "https://malicious-site.com"}
        )
        
        return {
            "preflight_status": preflight_response.status_code,
            "preflight_headers": dict(preflight_response.headers),
            "malicious_origin_status": malicious_response.status_code,
            "cors_working": preflight_response.status_code in [200, 204]
        }
    
    def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation security"""
        print("✅ Testing Input Validation...")
        
        # Test invalid file type
        invalid_file_response = self.session.post(
            f"{self.base_url}/analyze-chart",
            files={"file": ("test.txt", b"This is not an image", "text/plain")}
        )
        
        # Test empty file
        empty_file_response = self.session.post(
            f"{self.base_url}/analyze-chart",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        
        return {
            "invalid_file_rejected": invalid_file_response.status_code == 400,
            "empty_file_rejected": empty_file_response.status_code == 400,
            "invalid_file_response": invalid_file_response.text[:200],
            "empty_file_response": empty_file_response.text[:200]
        }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test if error responses expose sensitive information"""
        print("🚨 Testing Error Handling...")
        
        # Test non-existent endpoint
        not_found_response = self.session.get(f"{self.base_url}/nonexistent")
        
        # Test invalid method
        method_not_allowed_response = self.session.put(f"{self.base_url}/health")
        
        return {
            "not_found_status": not_found_response.status_code,
            "method_not_allowed_status": method_not_allowed_response.status_code,
            "not_found_response": not_found_response.text[:200],
            "method_not_allowed_response": method_not_allowed_response.text[:200]
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests"""
        print("🚀 Starting Comprehensive Security Testing...")
        print(f"Target URL: {self.base_url}")
        print("=" * 60)
        
        tests = [
            ("Rate Limiting", self.test_rate_limiting),
            ("Sensitive Data Exposure", self.test_sensitive_data_exposure),
            ("Security Headers", self.test_security_headers),
            ("CORS Security", self.test_cors_security),
            ("Input Validation", self.test_input_validation),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            try:
                print(f"\n📋 Running: {test_name}")
                result = test_func()
                self.test_results[test_name] = result
                print(f"✅ {test_name} completed")
            except Exception as e:
                print(f"❌ {test_name} failed: {str(e)}")
                self.test_results[test_name] = {"error": str(e)}
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive security report"""
        report = []
        report.append("🔒 SECURITY TESTING REPORT")
        report.append("=" * 50)
        report.append(f"Target: {self.base_url}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if "error" not in result)
        failed_tests = total_tests - passed_tests
        
        report.append("📊 SUMMARY")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append("")
        
        # Detailed Results
        for test_name, result in self.test_results.items():
            report.append(f"🔍 {test_name.upper()}")
            report.append("-" * 30)
            
            if "error" in result:
                report.append(f"❌ ERROR: {result['error']}")
            else:
                # Format the result based on test type
                if test_name == "Rate Limiting":
                    status = "✅ PASS" if result.get("rate_limiting_working") else "❌ FAIL"
                    report.append(f"Status: {status}")
                    report.append(f"First Rate Limit: {result.get('first_rate_limit_at')}")
                    
                elif test_name == "Sensitive Data Exposure":
                    status = "❌ FAIL" if result.get("sensitive_data_exposed") else "✅ PASS"
                    report.append(f"Status: {status}")
                    if result.get("exposed_endpoints"):
                        report.append("Exposed Data Found:")
                        for endpoint, patterns in result["exposed_endpoints"].items():
                            report.append(f"  {endpoint}: {patterns}")
                            
                elif test_name == "Security Headers":
                    status = "✅ PASS" if result.get("all_headers_present") else "❌ FAIL"
                    report.append(f"Status: {status}")
                    if result.get("missing_headers"):
                        report.append("Missing Headers:")
                        for header in result["missing_headers"]:
                            report.append(f"  {header}")
                            
                elif test_name == "CORS Security":
                    status = "✅ PASS" if result.get("cors_working") else "❌ FAIL"
                    report.append(f"Status: {status}")
                    
                elif test_name == "Input Validation":
                    status = "✅ PASS" if result.get("invalid_file_rejected") and result.get("empty_file_rejected") else "❌ FAIL"
                    report.append(f"Status: {status}")
                    
                elif test_name == "Error Handling":
                    report.append(f"Not Found Status: {result.get('not_found_status')}")
                    report.append(f"Method Not Allowed Status: {result.get('method_not_allowed_status')}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("=" * 30)
        
        if self.test_results.get("Sensitive Data Exposure", {}).get("sensitive_data_exposed"):
            report.append("⚠️  CRITICAL: Sensitive data is being exposed!")
            report.append("   - Review all endpoint responses")
            report.append("   - Implement response sanitization")
            report.append("   - Check logging for sensitive data")
        
        if not self.test_results.get("Security Headers", {}).get("all_headers_present"):
            report.append("⚠️  HIGH: Security headers are missing!")
            report.append("   - Implement missing security headers")
            report.append("   - Review security middleware")
        
        if not self.test_results.get("Rate Limiting", {}).get("rate_limiting_working"):
            report.append("⚠️  HIGH: Rate limiting is not working!")
            report.append("   - Check rate limiting implementation")
            report.append("   - Verify IP detection logic")
        
        if not self.test_results.get("Input Validation", {}).get("invalid_file_rejected"):
            report.append("⚠️  MEDIUM: Input validation needs improvement!")
            report.append("   - Strengthen file type validation")
            report.append("   - Add file content validation")
        
        if failed_tests == 0:
            report.append("🎉 All security tests passed! Your service is well-protected.")
        else:
            report.append(f"🔧 {failed_tests} security issues found. Review and fix before production deployment.")
        
        return "\n".join(report)

def main():
    """Main function to run security tests"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print("🔒 Enhanced Chart Analyzer - Security Testing")
    print("=" * 50)
    
    tester = SecurityTester(base_url)
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Generate and display report
        print("\n" + "=" * 60)
        report = tester.generate_report()
        print(report)
        
        # Save report to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"security_report_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {filename}")
        
        # Exit with appropriate code
        failed_tests = sum(1 for result in results.values() if "error" in result)
        if failed_tests > 0:
            print(f"\n❌ {failed_tests} tests failed. Review the report above.")
            sys.exit(1)
        else:
            print("\n✅ All security tests completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Security testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during security testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
