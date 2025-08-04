#!/usr/bin/env python3
"""
Script to install flask-cors and restart the server with external access
"""

import subprocess
import sys
import os
import time

def install_flask_cors():
    """Install flask-cors dependency"""
    print("📦 Installing flask-cors...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-cors>=4.0.0"])
        print("✅ flask-cors installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install flask-cors: {e}")
        return False

def check_server_running():
    """Check if server is running on port 5000"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', 5000))
            return result == 0
    except:
        return False

def main():
    """Main function"""
    print("🚀 Setting up server for external access...")
    print("=" * 50)
    
    # Install flask-cors
    if not install_flask_cors():
        print("❌ Cannot proceed without flask-cors")
        return
    
    # Check if server is running
    if check_server_running():
        print("⚠️ Server is already running on port 5000")
        print("💡 Please stop the current server (Ctrl+C) and restart it")
        print("   The new configuration will enable external access")
    else:
        print("✅ Port 5000 is available")
        print("💡 You can now start the server with:")
        print("   python app.py")
    
    print("\n📋 Configuration Summary:")
    print("   - Host: 0.0.0.0 (external access enabled)")
    print("   - Port: 5000")
    print("   - CORS: Enabled for all routes")
    print("   - Timeouts: Removed (no restrictions)")
    
    print("\n🌐 External Access:")
    print("   - Local: http://localhost:5000")
    print("   - Network: http://[YOUR_IP]:5000")
    print("   - Postman: Use your server IP address")

if __name__ == "__main__":
    main() 