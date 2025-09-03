#!/usr/bin/env python3
"""
Production Security Configuration Generator
Generate secure configuration for your deployment
"""

import os
import secrets
import string

def generate_secret_key():
    """Generate a secure secret key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(64))

def create_production_config():
    """Create production security configuration"""
    
    print("🔒 Production Security Configuration Generator")
    print("=" * 60)
    
    # Get user input
    app_name = input("Enter your app name (e.g., 'my-chart-analyzer'): ").strip()
    custom_domain = input("Enter your custom domain (optional, e.g., 'api.mydomain.com'): ").strip()
    frontend_domain = input("Enter your frontend domain (optional, e.g., 'mydomain.com'): ").strip()
    
    # Generate configuration
    secret_key = generate_secret_key()
    
    # Build allowed hosts
    allowed_hosts = [f"{app_name}.onrender.com"]
    if custom_domain:
        allowed_hosts.append(custom_domain)
    
    # Build allowed origins
    allowed_origins = [f"https://{app_name}.onrender.com"]
    if custom_domain:
        allowed_origins.append(f"https://{custom_domain}")
    if frontend_domain:
        allowed_origins.append(f"https://{frontend_domain}")
    
    print(f"\n📋 Generated Configuration:")
    print("=" * 40)
    
    print(f"\n🔑 Environment Variables (Set these in Render):")
    print("-" * 50)
    print(f"ALLOWED_HOSTS={','.join(allowed_hosts)}")
    print(f"ALLOWED_ORIGINS={','.join(allowed_origins)}")
    print(f"SECRET_KEY={secret_key}")
    print(f"RATE_LIMIT_PER_HOUR=1000")
    print(f"MAX_REQUESTS_PER_MINUTE=50")
    print(f"ENVIRONMENT=production")
    
    print(f"\n📝 Updated config_enhanced.py:")
    print("-" * 50)
    print(f"""
# Security Configuration - PRODUCTION
ALLOWED_HOSTS: List[str] = {allowed_hosts}
ALLOWED_ORIGINS: List[str] = {allowed_origins}
SECRET_KEY: str = "{secret_key}"
RATE_LIMIT_PER_HOUR: int = 1000
MAX_REQUESTS_PER_MINUTE: int = 50
""")
    
    print(f"\n🌐 Your Deployment URLs:")
    print("-" * 30)
    print(f"API URL: https://{app_name}.onrender.com")
    if custom_domain:
        print(f"Custom URL: https://{custom_domain}")
    
    print(f"\n✅ Security Checklist:")
    print("-" * 20)
    print("□ Set environment variables in Render dashboard")
    print("□ Update config_enhanced.py with production values")
    print("□ Test your API endpoints")
    print("□ Verify CORS is working with your frontend")
    print("□ Run security tests: python test_security.py")
    
    # Save to file
    config_file = "production_config.env"
    with open(config_file, "w") as f:
        f.write(f"# Production Environment Variables\n")
        f.write(f"ALLOWED_HOSTS={','.join(allowed_hosts)}\n")
        f.write(f"ALLOWED_ORIGINS={','.join(allowed_origins)}\n")
        f.write(f"SECRET_KEY={secret_key}\n")
        f.write(f"RATE_LIMIT_PER_HOUR=1000\n")
        f.write(f"MAX_REQUESTS_PER_MINUTE=50\n")
        f.write(f"ENVIRONMENT=production\n")
    
    print(f"\n💾 Configuration saved to: {config_file}")
    print("📤 Upload this file to your deployment platform or copy the values manually")

if __name__ == "__main__":
    create_production_config()
