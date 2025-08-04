#!/usr/bin/env python3
"""
Cloud VM Performance Monitor for RAG System
Monitors CPU, Memory, and API response times
"""

import psutil
import time
import logging
from datetime import datetime
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudVMMonitor:
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.start_time = time.time()
        
    def get_system_stats(self):
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'uptime_seconds': time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def test_api_response_time(self, endpoint: str = "/api/health"):
        """Test API response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}{endpoint}", timeout=30)
            response_time = time.time() - start_time
            
            return {
                'endpoint': endpoint,
                'status_code': response.status_code,
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error testing API: {e}")
            return {'error': str(e)}
    
    def monitor_query_performance(self, query: str):
        """Monitor performance during a query"""
        try:
            # Get baseline stats
            baseline_stats = self.get_system_stats()
            
            # Start query
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/api/query",
                json={"query": query},
                timeout=120  # 2 minutes timeout
            )
            query_time = time.time() - start_time
            
            # Get stats after query
            after_stats = self.get_system_stats()
            
            return {
                'query': query,
                'query_time_seconds': query_time,
                'status_code': response.status_code,
                'baseline_stats': baseline_stats,
                'after_stats': after_stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error monitoring query: {e}")
            return {'error': str(e)}
    
    def continuous_monitoring(self, interval: int = 30):
        """Continuous monitoring with specified interval"""
        logger.info(f"Starting continuous monitoring every {interval} seconds")
        
        while True:
            try:
                stats = self.get_system_stats()
                logger.info(f"System Stats: CPU={stats.get('cpu_percent', 0):.1f}%, "
                          f"Memory={stats.get('memory_percent', 0):.1f}%, "
                          f"Uptime={stats.get('uptime_seconds', 0):.0f}s")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

def main():
    """Main function for testing"""
    monitor = CloudVMMonitor()
    
    # Test system stats
    print("=== System Statistics ===")
    stats = monitor.get_system_stats()
    print(json.dumps(stats, indent=2))
    
    # Test API health
    print("\n=== API Health Check ===")
    api_test = monitor.test_api_response_time()
    print(json.dumps(api_test, indent=2))
    
    # Test query performance
    print("\n=== Query Performance Test ===")
    query_test = monitor.monitor_query_performance("What is the grace period?")
    print(json.dumps(query_test, indent=2))

if __name__ == "__main__":
    main() 