#!/usr/bin/env python3
"""
Domain Discovery Tool
Find your production domains and IPs for security configuration
"""

import socket
import requests
import json
from urllib.parse import urlparse

def find_domains():
    """Find domains and IPs for your deployment"""
    
    print("🌐 Domain Discovery Tool")
    print("=" * 50)
    
    # Common deployment platforms
    platforms = {
        "render.com": "https://your-app-name.onrender.com",
        "heroku.com": "https://your-app-name.herokuapp.com", 
        "vercel.com": "https://your-app-name.vercel.app",
        "netlify.com": "https://your-app-name.netlify.app",
        "railway.app": "https://your-app-name.railway.app",
        "fly.io": "https://your-app-name.fly.dev",
        "digitalocean.com": "https://your-app-name.digitalocean.com"
    }
    
    print("📋 Common Deployment Platforms:")
    for platform, example in platforms.items():
        print(f"   {platform}: {example}")
    
    print("\n🔍 How to Find Your Domains:")
    print("1. Check your deployment platform dashboard")
    print("2. Look for 'Domain' or 'URL' settings")
    print("3. Check your DNS provider (Cloudflare, Route53, etc.)")
    print("4. Check your custom domain registrar")
    
    print("\n💡 Example Production Configuration:")
    print("""
# For Render.com deployment:
ALLOWED_HOSTS = [
    "your-app-name.onrender.com",
    "your-custom-domain.com"
]

ALLOWED_ORIGINS = [
    "https://your-app-name.onrender.com",
    "https://your-custom-domain.com",
    "https://your-frontend-domain.com"
]
""")
    
    print("\n🔧 Environment Variables (Recommended):")
    print("""
# Set these in your deployment platform:
ALLOWED_HOSTS=your-app-name.onrender.com,your-custom-domain.com
ALLOWED_ORIGINS=https://your-app-name.onrender.com,https://your-custom-domain.com
""")

def test_domain_resolution(domain):
    """Test if a domain resolves to an IP"""
    try:
        ip = socket.gethostbyname(domain)
        print(f"✅ {domain} → {ip}")
        return ip
    except socket.gaierror:
        print(f"❌ {domain} → Failed to resolve")
        return None

def check_https_support(domain):
    """Check if domain supports HTTPS"""
    try:
        response = requests.get(f"https://{domain}", timeout=5)
        print(f"✅ {domain} → HTTPS supported (Status: {response.status_code})")
        return True
    except:
        try:
            response = requests.get(f"http://{domain}", timeout=5)
            print(f"⚠️  {domain} → HTTP only (Status: {response.status_code})")
            return False
        except:
            print(f"❌ {domain} → No response")
            return False

if __name__ == "__main__":
    find_domains()
    
    print("\n🧪 Test Your Domains:")
    test_domains = input("Enter domains to test (comma-separated): ").strip()
    
    if test_domains:
        domains = [d.strip() for d in test_domains.split(",")]
        for domain in domains:
            if domain:
                print(f"\nTesting {domain}:")
                test_domain_resolution(domain)
                check_https_support(domain)
