#!/usr/bin/env python3
"""
Performance Monitor for RAG System
Tracks response times and identifies bottlenecks
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.start_time = None
        
    def start_timer(self):
        """Start performance timer"""
        self.start_time = time.time()
        logger.info("⏱️ Performance timer started")
        
    def log_step(self, step_name: str, duration: float):
        """Log a processing step duration"""
        self.metrics.append({
            'step': step_name,
            'duration': duration,
            'timestamp': datetime.now()
        })
        logger.info(f"📊 {step_name}: {duration:.2f}s")
        
    def get_total_time(self) -> float:
        """Get total processing time"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = self.get_total_time()
        
        # Find the slowest step
        slowest_step = max(self.metrics, key=lambda x: x['duration']) if self.metrics else None
        
        summary = {
            'total_time': total_time,
            'steps': len(self.metrics),
            'slowest_step': slowest_step['step'] if slowest_step else None,
            'slowest_duration': slowest_step['duration'] if slowest_step else 0.0,
            'metrics': self.metrics
        }
        
        return summary
        
    def print_summary(self):
        """Print performance summary"""
        summary = self.get_performance_summary()
        
        logger.info("📊 PERFORMANCE SUMMARY:")
        logger.info(f"   Total Time: {summary['total_time']:.2f}s")
        logger.info(f"   Steps: {summary['steps']}")
        if summary['slowest_step']:
            logger.info(f"   Slowest Step: {summary['slowest_step']} ({summary['slowest_duration']:.2f}s)")
            
        # Performance rating
        if summary['total_time'] < 30:
            logger.info("   🟢 EXCELLENT: Under 30 seconds")
        elif summary['total_time'] < 60:
            logger.info("   🟡 GOOD: Under 1 minute")
        elif summary['total_time'] < 120:
            logger.info("   🟠 ACCEPTABLE: Under 2 minutes")
        else:
            logger.info("   🔴 SLOW: Over 2 minutes - needs optimization")

# Global monitor instance
performance_monitor = PerformanceMonitor() 