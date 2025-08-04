#!/usr/bin/env python3
"""
Keep-alive script to prevent server auto-termination
Pings the Flask server every 2 minutes to keep it alive
"""

import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ping_server():
    """Ping the Flask server"""
    try:
        response = requests.get("http://127.0.0.1:5000/api/keep-alive", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Server ping successful")
            return True
        else:
            logger.warning(f"⚠️ Server ping failed with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to server - is it running?")
        return False
    except Exception as e:
        logger.error(f"❌ Ping failed: {e}")
        return False

def main():
    """Main keep-alive loop"""
    logger.info("🚀 Starting keep-alive script")
    logger.info("This script will ping the server every 2 minutes")
    logger.info("Press Ctrl+C to stop")
    
    ping_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ping_count += 1
            current_time = datetime.now()
            uptime = current_time - start_time
            
            logger.info(f"Ping #{ping_count} at {current_time.strftime('%H:%M:%S')} (Uptime: {uptime})")
            
            if ping_server():
                logger.info("Server is alive and responding")
            else:
                logger.warning("Server may be having issues")
            
            # Wait 2 minutes before next ping
            logger.info("Sleeping for 2 minutes...")
            time.sleep(120)  # 2 minutes
            
    except KeyboardInterrupt:
        logger.info("👋 Keep-alive script stopped by user")
    except Exception as e:
        logger.error(f"Keep-alive script crashed: {e}")

if __name__ == "__main__":
    main() 