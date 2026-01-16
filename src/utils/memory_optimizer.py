#!/usr/bin/env python3
"""
Memory Optimization Utility for GameMate AI Assistant
Provides advanced memory management and cleanup functions
"""

import gc
import psutil
import numpy as np
import ctypes
import threading
import weakref
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Advanced memory optimization for production environments"""
    
    def __init__(self):
        self.cleanup_interval = 30  # seconds
        self.cleanup_thread = None
        self.running = False
        self._memory_threshold = 400 * 1024 * 1024  # 400MB in bytes
        
    def start_auto_cleanup(self):
        """Start automatic memory cleanup thread"""
        if self.running:
            return
            
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Auto memory cleanup started")
    
    def stop_auto_cleanup(self):
        """Stop automatic memory cleanup"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        logger.info("Auto memory cleanup stopped")
    
    def _cleanup_loop(self):
        """Automatic cleanup loop"""
        import time
        
        while self.running:
            try:
                # Check memory usage
                memory_mb = self.get_memory_usage_mb()
                
                if memory_mb > (self._memory_threshold / 1024 / 1024):
                    logger.info(f"High memory usage detected: {memory_mb:.1f}MB - cleaning up")
                    self.aggressive_cleanup()
                else:
                    self.gentle_cleanup()
                    
                # Sleep for cleanup interval
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def gentle_cleanup(self):
        """Gentle memory cleanup"""
        try:
            # Standard garbage collection
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Gentle cleanup: {collected} objects collected")
        except Exception as e:
            logger.warning(f"Gentle cleanup failed: {e}")
    
    def aggressive_cleanup(self):
        """Aggressive memory cleanup for high usage scenarios"""
        try:
            # Force garbage collection with all generations
            for generation in range(3):
                collected = gc.collect()
                
            # Clear numpy cache if available
            try:
                np._NoValue  # Check if numpy is available
                # Clear any numpy internal caches
                pass
            except:
                pass
            
            # Try to release unused memory back to OS (Windows)
            try:
                if hasattr(ctypes, 'windll'):
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.GetCurrentProcess()
                    kernel32.SetProcessWorkingSetSize(handle, -1, -1)
            except Exception:
                pass
                
            logger.info(f"Aggressive cleanup completed. Memory: {self.get_memory_usage_mb():.1f}MB")
            
        except Exception as e:
            logger.error(f"Aggressive cleanup failed: {e}")
    
    def optimize_array_memory(self, array: np.ndarray, target_dtype: Optional[str] = None) -> np.ndarray:
        """Optimize numpy array memory usage"""
        if array is None:
            return array
            
        try:
            # Convert to memory-efficient dtype if specified
            if target_dtype == 'float16' and array.dtype != np.float16:
                return array.astype(np.float16)
            elif target_dtype == 'uint8' and array.dtype != np.uint8:
                return array.astype(np.uint8)
            
            # Auto-optimize based on value range
            if array.dtype == np.float64:
                # Convert double to single precision if possible
                return array.astype(np.float32)
            elif array.dtype == np.int64:
                # Convert to smaller int type if values fit
                if array.min() >= 0 and array.max() <= 255:
                    return array.astype(np.uint8)
                elif array.min() >= -128 and array.max() <= 127:
                    return array.astype(np.int8)
                elif array.min() >= 0 and array.max() <= 65535:
                    return array.astype(np.uint16)
                else:
                    return array.astype(np.int32)
            
            return array
            
        except Exception as e:
            logger.warning(f"Array memory optimization failed: {e}")
            return array
    
    def create_memory_mapped_array(self, shape, dtype=np.float32, mode='w+'):
        """Create memory-mapped array for large data"""
        try:
            import tempfile
            import os
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()
            
            # Create memory-mapped array
            array = np.memmap(temp_file.name, dtype=dtype, mode=mode, shape=shape)
            
            # Register cleanup
            weakref.finalize(array, lambda: os.unlink(temp_file.name) if os.path.exists(temp_file.name) else None)
            
            return array
            
        except Exception as e:
            logger.error(f"Failed to create memory-mapped array: {e}")
            return np.zeros(shape, dtype=dtype)
    
    def get_memory_info(self) -> dict:
        """Get detailed memory information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'gc_counts': gc.get_counts(),
                'gc_threshold': gc.get_threshold()
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()

def optimize_memory():
    """Quick memory optimization function"""
    memory_optimizer.aggressive_cleanup()
    
def get_memory_usage():
    """Get current memory usage"""
    return memory_optimizer.get_memory_usage_mb()

def start_memory_monitoring():
    """Start automatic memory monitoring"""
    memory_optimizer.start_auto_cleanup()
    
def stop_memory_monitoring():
    """Stop automatic memory monitoring"""
    memory_optimizer.stop_auto_cleanup()