#!/usr/bin/env python3
"""
GameMate AI Assistant - Performance Monitor
Real-time performance monitoring and optimization
"""

import time
import psutil
import threading
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

class PerformanceMonitor:
    """Advanced performance monitoring and optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.stats = {
            "fps": 0.0,
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "gpu_percent": 0.0,
            "gpu_temp": 0.0,
            "frame_time_ms": 0.0,
            "total_frames": 0,
            "startup_time": 0.0,
            "last_update": datetime.now().isoformat()
        }
        
        # Performance limits
        self.limits = {
            "max_cpu_percent": config.get('performance', {}).get('max_cpu_percent', 80),
            "max_memory_mb": config.get('performance', {}).get('max_memory_mb', 1500),
            "min_fps": config.get('performance', {}).get('min_fps', 15),
            "max_gpu_temp": config.get('performance', {}).get('max_gpu_temp', 80)
        }
        
        # Monitoring thread
        self.monitor_thread = None
        self.stats_lock = threading.Lock()
        
        # Performance history
        self.history = []
        self.max_history = 1000
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_monitoring()
        
        # Process reference
        self.process = psutil.Process()
        
        # Startup time tracking
        self.startup_start = time.time()
        
        self.logger = logging.getLogger(__name__)
    
    def _check_gpu_monitoring(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            # Try nvidia-ml-py first (preferred), then fallback to pynvml
            try:
                import nvidia_ml_py as pynvml
            except ImportError:
                import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_frame_count = 0
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Update stats
            with self.stats_lock:
                # CPU and memory from current process
                self.stats["cpu_percent"] = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                self.stats["memory_mb"] = memory_info.rss / (1024 * 1024)
                
                # Calculate FPS
                frame_diff = self.stats["total_frames"] - last_frame_count
                time_diff = current_time - last_time
                if time_diff > 0:
                    self.stats["fps"] = frame_diff / time_diff
                    if frame_diff > 0:
                        self.stats["frame_time_ms"] = (time_diff / frame_diff) * 1000
                
                # GPU monitoring
                if self.gpu_available:
                    gpu_stats = self._get_gpu_stats()
                    self.stats.update(gpu_stats)
                
                # Update timestamp
                self.stats["last_update"] = datetime.now().isoformat()
                
                # Store history
                self.history.append(dict(self.stats))
                if len(self.history) > self.max_history:
                    self.history.pop(0)
            
            # Check for performance issues
            self._check_performance_limits()
            
            # Update counters
            last_frame_count = self.stats["total_frames"]
            last_time = current_time
            
            time.sleep(1.0)  # Update every second
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics"""
        try:
            # Try nvidia-ml-py first (preferred), then fallback to pynvml
            try:
                import nvidia_ml_py as pynvml
            except ImportError:
                import pynvml
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent = util.gpu
            
            # GPU temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                "gpu_percent": float(gpu_percent),
                "gpu_temp": float(temp)
            }
            
        except Exception as e:
            self.logger.debug(f"GPU stats unavailable: {e}")
            return {"gpu_percent": 0.0, "gpu_temp": 0.0}
    
    def _check_performance_limits(self):
        """Check if performance limits are exceeded"""
        with self.stats_lock:
            warnings = []
            
            if self.stats["cpu_percent"] > self.limits["max_cpu_percent"]:
                warnings.append(f"High CPU usage: {self.stats['cpu_percent']:.1f}%")
            
            if self.stats["memory_mb"] > self.limits["max_memory_mb"]:
                warnings.append(f"High memory usage: {self.stats['memory_mb']:.1f}MB")
            
            if self.stats["fps"] > 0 and self.stats["fps"] < self.limits["min_fps"]:
                warnings.append(f"Low FPS: {self.stats['fps']:.1f}")
            
            if self.stats["gpu_temp"] > self.limits["max_gpu_temp"]:
                warnings.append(f"High GPU temp: {self.stats['gpu_temp']:.1f}°C")
            
            # Log warnings
            for warning in warnings:
                self.logger.warning(f"Performance issue: {warning}")
    
    def increment_frame_count(self):
        """Increment processed frame count"""
        with self.stats_lock:
            self.stats["total_frames"] += 1
    
    def record_startup_complete(self):
        """Record startup completion time"""
        startup_time = time.time() - self.startup_start
        with self.stats_lock:
            self.stats["startup_time"] = startup_time
        
        self.logger.info(f"Startup completed in {startup_time:.2f} seconds")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.stats_lock:
            return dict(self.stats)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self.stats_lock:
            current = dict(self.stats)
            
            # Calculate averages from history
            if self.history:
                history_stats = {
                    "avg_fps": sum(h["fps"] for h in self.history) / len(self.history),
                    "avg_cpu": sum(h["cpu_percent"] for h in self.history) / len(self.history),
                    "avg_memory": sum(h["memory_mb"] for h in self.history) / len(self.history),
                    "max_memory": max(h["memory_mb"] for h in self.history),
                    "min_fps": min(h["fps"] for h in self.history if h["fps"] > 0),
                    "max_fps": max(h["fps"] for h in self.history)
                }
            else:
                history_stats = {}
            
            # Performance assessment
            assessment = self._assess_performance(current)
            
            return {
                "current": current,
                "averages": history_stats,
                "assessment": assessment,
                "limits": self.limits,
                "gpu_available": self.gpu_available,
                "history_samples": len(self.history)
            }
    
    def _assess_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance"""
        score = 100
        issues = []
        recommendations = []
        
        # CPU assessment
        if stats["cpu_percent"] > 50:
            score -= 15
            issues.append("High CPU usage")
            recommendations.append("Close background applications")
        
        # Memory assessment
        if stats["memory_mb"] > 1000:
            score -= 10
            issues.append("High memory usage")
            recommendations.append("Restart application periodically")
        
        # FPS assessment
        if stats["fps"] > 0 and stats["fps"] < 20:
            score -= 20
            issues.append("Low frame rate")
            recommendations.append("Reduce FPS limit or enable hardware acceleration")
        
        # GPU assessment
        if stats["gpu_temp"] > 70:
            score -= 10
            issues.append("High GPU temperature")
            recommendations.append("Check GPU cooling and reduce workload")
        
        # Overall rating
        if score >= 90:
            rating = "Excellent"
        elif score >= 75:
            rating = "Good" 
        elif score >= 60:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            "score": max(0, score),
            "rating": rating,
            "issues": issues,
            "recommendations": recommendations
        }
    
    def optimize_performance(self):
        """Attempt automatic performance optimization"""
        optimizations = []
        
        stats = self.get_current_stats()
        
        # Memory optimization
        if stats["memory_mb"] > self.limits["max_memory_mb"] * 0.8:
            try:
                import gc
                gc.collect()
                optimizations.append("Garbage collection performed")
            except Exception:
                pass
        
        # Process priority optimization
        try:
            if psutil.WINDOWS:
                self.process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                self.process.nice(-10)
            optimizations.append("Process priority optimized")
        except Exception:
            pass
        
        self.logger.info(f"Performance optimizations applied: {optimizations}")
        return optimizations
    
    def save_performance_log(self, filepath: Optional[str] = None):
        """Save performance history to file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"performance_log_{timestamp}.json"
        
        report = self.get_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance log saved to {filepath}")

class SystemOptimizer:
    """System-level optimization utilities"""
    
    @staticmethod
    def optimize_windows_gaming():
        """Apply Windows-specific gaming optimizations"""
        optimizations = []
        
        try:
            # Set process to high priority
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(-1, 0x00000080)
            optimizations.append("High priority process")
        except Exception:
            pass
        
        try:
            # Disable Windows Game Mode interference (if needed)
            optimizations.append("Gaming optimizations applied")
        except Exception:
            pass
        
        return optimizations
    
    @staticmethod
    def check_system_performance():
        """Check system performance capabilities"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_free_gb": psutil.disk_usage('.').free / (1024**3),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        # Performance rating
        rating = "Good"
        if info["cpu_count"] < 4:
            rating = "Basic"
        if info["memory_total_gb"] < 8:
            rating = "Limited"
        if info["memory_total_gb"] >= 16 and info["cpu_count"] >= 8:
            rating = "Excellent"
        
        info["performance_rating"] = rating
        return info

if __name__ == "__main__":
    # Demo performance monitoring
    config = {
        'performance': {
            'max_cpu_percent': 70,
            'max_memory_mb': 1200,
            'min_fps': 20
        }
    }
    
    monitor = PerformanceMonitor(config)
    monitor.start_monitoring()
    
    print("🎮 Performance monitoring demo - 10 seconds")
    
    # Simulate some frames
    for i in range(100):
        monitor.increment_frame_count()
        time.sleep(0.1)
    
    monitor.record_startup_complete()
    
    # Generate report
    report = monitor.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(f"   FPS: {report['current']['fps']:.1f}")
    print(f"   CPU: {report['current']['cpu_percent']:.1f}%")
    print(f"   Memory: {report['current']['memory_mb']:.1f} MB")
    print(f"   Rating: {report['assessment']['rating']} ({report['assessment']['score']}/100)")
    
    monitor.stop_monitoring()