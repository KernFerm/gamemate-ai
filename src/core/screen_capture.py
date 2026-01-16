"""
Screen Capture Module
Fast screen capture using MSS library
"""

import mss
import numpy as np
from typing import Optional, Dict, Any
import cv2

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ScreenCapture:
    """Handles screen capture for real-time processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sct = mss.mss()
        
        # Get monitor info
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
        # Configure capture region
        region = config['performance'].get('screen_capture_region')
        if region:
            self.monitor = {
                'left': region[0],
                'top': region[1],
                'width': region[2],
                'height': region[3]
            }
        
        self.downscale = config['performance'].get('downscale_factor', 1.0)
        
        logger.info(f"Screen capture initialized: {self.monitor['width']}x{self.monitor['height']}")
        
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture current screen frame
        
        Returns:
            numpy.ndarray: BGR image frame, or None if capture fails
        """
        try:
            # Capture screenshot
            sct_img = self.sct.grab(self.monitor)
            
            # Convert to numpy array (BGRA format)
            frame = np.array(sct_img)
            
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Apply downscaling if configured
            if self.downscale != 1.0:
                new_width = int(frame.shape[1] * self.downscale)
                new_height = int(frame.shape[0] * self.downscale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            return frame
            
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None
    
    def get_resolution(self):
        """Get current capture resolution"""
        return (self.monitor['width'], self.monitor['height'])
    
    def cleanup(self):
        """Cleanup resources"""
        if self.sct:
            self.sct.close()
