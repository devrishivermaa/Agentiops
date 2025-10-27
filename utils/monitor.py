# utils/monitor.py
"""
System monitoring utilities for tracking resource usage and performance.
"""

import psutil
import time
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("Monitor")


class SystemMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        process = psutil.Process()
        
        return {
            "elapsed_time": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "memory_delta_mb": (process.memory_info().rss / (1024 * 1024)) - self.start_memory,
            "num_threads": process.num_threads()
        }
    
    def log_stats(self):
        """Log current system statistics."""
        stats = self.get_stats()
        logger.info(
            f"System Stats - CPU: {stats['cpu_percent']:.1f}%, "
            f"Memory: {stats['memory_mb']:.1f}MB "
            f"(Δ{stats['memory_delta_mb']:+.1f}MB), "
            f"Threads: {stats['num_threads']}, "
            f"Time: {stats['elapsed_time']:.1f}s"
        )
