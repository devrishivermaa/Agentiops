# utils/monitor.py
"""
System resource monitoring utility for AgentOps.
Tracks CPU, memory usage, and processing statistics.
"""

import time
import psutil
import os
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("SystemMonitor")


class SystemMonitor:
    """
    Monitor system resources during pipeline execution.
    Provides lightweight resource tracking without significant overhead.
    """
    
    def __init__(self):
        """Initialize system monitor."""
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        
        # Baseline measurements
        try:
            self.baseline_cpu = self.process.cpu_percent(interval=0.1)
            self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        except Exception as e:
            logger.warning(f"Failed to get baseline metrics: {e}")
            self.baseline_cpu = 0
            self.baseline_memory = 0
        
        logger.debug(f"SystemMonitor initialized: CPU {self.baseline_cpu:.1f}%, Memory {self.baseline_memory:.1f}MB")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current system resource statistics.
        
        Returns:
            Dictionary with elapsed time, CPU%, memory usage
        """
        try:
            elapsed = time.time() - self.start_time
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            stats = {
                "elapsed_time": round(elapsed, 2),
                "cpu_percent": round(cpu_percent, 1),
                "memory_mb": round(memory_mb, 1),
                "memory_delta_mb": round(memory_mb - self.baseline_memory, 1)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                "elapsed_time": time.time() - self.start_time,
                "cpu_percent": 0,
                "memory_mb": 0,
                "memory_delta_mb": 0
            }
    
    def log_stats(self):
        """Log current system statistics."""
        stats = self.get_stats()
        logger.info(
            f"⏱️  Elapsed: {stats['elapsed_time']:.1f}s | "
            f"CPU: {stats['cpu_percent']:.1f}% | "
            f"Memory: {stats['memory_mb']:.1f}MB (Δ{stats['memory_delta_mb']:+.1f}MB)"
        )
    
    def get_system_wide_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics (not just this process).
        
        Returns:
            Dictionary with system-wide CPU, memory, disk usage
        """
        try:
            return {
                "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
            }
        except Exception as e:
            logger.error(f"Failed to get system-wide stats: {e}")
            return {}


# Convenience function
def get_current_stats() -> Dict[str, Any]:
    """Get current process statistics without creating a monitor instance."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        return {
            "cpu_percent": round(process.cpu_percent(interval=0.1), 1),
            "memory_mb": round(memory_mb, 1)
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"cpu_percent": 0, "memory_mb": 0}


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*60)
    print("SYSTEM MONITOR TEST")
    print("="*60 + "\n")
    
    monitor = SystemMonitor()
    
    print("Process Stats:")
    stats = monitor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSystem-Wide Stats:")
    sys_stats = monitor.get_system_wide_stats()
    for key, value in sys_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
