"""
Fallback implementations for when MSI-specific libraries are not available
"""

import cv2
import numpy as np
import mss
from typing import Optional, Tuple

# Robust logger import that works in different contexts
try:
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for testing and standalone execution
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.logger import get_logger
    except ImportError:
        # Ultimate fallback
        import logging
        def get_logger(name):
            return logging.getLogger(name)

# Optional YOLO import with fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available - object detection will use simple fallback")

logger = get_logger(__name__)


class FallbackScreenCapture:
    """Fallback screen capture using MSS when hardware capture fails"""
    
    def __init__(self, config):
        self.config = config
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
        logger.info("Using fallback screen capture")
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture screen using MSS"""
        try:
            sct_img = self.sct.grab(self.monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            logger.error(f"Fallback capture error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.sct:
            self.sct.close()


class FallbackAITracker:
    """Fallback AI tracker using YOLO when MSI model is not available"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')
                logger.info("Using fallback YOLO tracker")
            except Exception as e:
                logger.error(f"Could not load YOLO model: {e}")
                self.model = None
        else:
            logger.info("YOLO not available - using basic tracker fallback")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with basic YOLO detection or simple fallback"""
        if not self.enabled:
            return frame
            
        if self.model is not None:
            return frame
        
        try:
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return frame
        except Exception as e:
            logger.error(f"Fallback tracker error: {e}")
            return frame
    
    def toggle(self):
        """Toggle tracker on/off"""
        self.enabled = not self.enabled


class FallbackVisualEngine:
    """Fallback visual enhancement using basic OpenCV"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        logger.info("Using fallback visual enhancement")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Basic visual enhancement"""
        if not self.enabled:
            return frame
        
        try:
            # Basic brightness/contrast adjustment
            enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
            return enhanced
        except Exception as e:
            logger.error(f"Fallback visual enhancement error: {e}")
            return frame
    
    def toggle(self):
        """Toggle enhancement on/off"""
        self.enabled = not self.enabled