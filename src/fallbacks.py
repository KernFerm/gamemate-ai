"""
Fallback systems for GameMate AI Assistant
Provides basic functionality when advanced dependencies are not available
"""

import cv2
import numpy as np
import mss
import time
import logging

logger = logging.getLogger(__name__)

class FallbackScreenCapture:
    """Basic screen capture without GPU acceleration"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.monitor = None
        try:
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]  # Primary monitor
            logger.info("Fallback screen capture initialized")
        except Exception as e:
            logger.error(f"Fallback screen capture failed: {e}")
            
    def capture_frame(self):
        """Capture a frame using basic MSS"""
        try:
            if self.monitor is None:
                return None
                
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)
            
            # Convert BGRA to BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            return frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
    
    def capture(self):
        """Alias for capture_frame to match engine interface"""
        return self.capture_frame()
    
    def get_fps(self):
        """Get current FPS"""
        return 60  # Default fallback FPS

class FallbackAITracker:
    """Basic AI tracker without advanced models"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.frame_count = 0
        logger.info("Fallback AI tracker initialized")
    
    def detect(self, frame):
        """Basic detection using simple computer vision"""
        if frame is None:
            return []
        
        detections = []
        self.frame_count += 1
        
        try:
            # Simple motion detection or color detection could go here
            # For now, return empty detections
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Example: detect bright spots (could be muzzle flashes, etc.)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'class': 'bright_spot',
                        'confidence': 0.5,
                        'bbox': [x, y, w, h]
                    })
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'fps': 60,
            'frame_count': self.frame_count,
            'detection_time': 0.016
        }

class FallbackVoiceControl:
    """Basic voice control without speech recognition"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.enabled = False
        self.commands = {
            'enable_tracker': lambda: print("Voice: Tracker enabled"),
            'disable_tracker': lambda: print("Voice: Tracker disabled"),
            'toggle_overlay': lambda: print("Voice: Overlay toggled"),
        }
        logger.warning("Voice control running in fallback mode (no speech recognition)")
    
    def listen_for_commands(self):
        """Simulate listening (does nothing in fallback)"""
        pass
    
    def process_command(self, command):
        """Process voice command"""
        if command in self.commands:
            self.commands[command]()
            return True
        return False

class FallbackVisualEngine:
    """Basic visual enhancement without GPU acceleration"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        logger.info("Fallback visual engine initialized")
    
    def process(self, frame):
        """Basic image processing using OpenCV"""
        if frame is None:
            return None
            
        try:
            # Basic brightness/contrast adjustment
            processed = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness * 50)
            
            # Basic saturation adjustment
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return processed
            
        except Exception as e:
            logger.error(f"Visual processing error: {e}")
            return frame
    
    def set_brightness(self, value):
        """Set brightness level"""
        self.brightness = max(0.1, min(3.0, value))
    
    def set_contrast(self, value):
        """Set contrast level"""  
        self.contrast = max(0.1, min(3.0, value))
    
    def set_saturation(self, value):
        """Set saturation level"""
        self.saturation = max(0.0, min(3.0, value))

class FallbackGPUMonitor:
    """Basic system monitoring without GPU-specific libraries"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.cpu_usage = 0
        self.memory_usage = 0
        logger.info("Fallback GPU monitor initialized")
    
    def get_gpu_stats(self):
        """Get mock GPU statistics"""
        return {
            'gpu_usage': 50,  # Mock values
            'memory_usage': 60,
            'temperature': 65,
            'power_usage': 150
        }
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            import psutil
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'available_memory': psutil.virtual_memory().available // (1024**2)  # MB
            }
        except ImportError:
            return {
                'cpu_usage': 50,
                'memory_usage': 60, 
                'available_memory': 8192
            }

# Fallback factory functions
def get_screen_capture():
    """Get screen capture instance with fallback"""
    return FallbackScreenCapture()

def get_ai_tracker():
    """Get AI tracker instance with fallback"""
    return FallbackAITracker()

def get_voice_control():
    """Get voice control instance with fallback"""
    return FallbackVoiceControl()

def get_visual_engine():
    """Get visual engine instance with fallback"""
    return FallbackVisualEngine()

def get_gpu_monitor():
    """Get GPU monitor instance with fallback"""
    return FallbackGPUMonitor()