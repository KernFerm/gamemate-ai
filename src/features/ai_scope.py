"""
AI Scope - Smart Zoom & Target Focus
Automatically zooms in on targets for precision aiming
"""

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AIScope:
    """AI-powered scope zoom"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.zoom_level = config['zoom_level']
        self.zoom_area_size = config['zoom_area_size']
        self.follow_crosshair = config['follow_crosshair']
        self.smooth_tracking = config['smooth_tracking']
        
        # Zoom state
        self.zoom_center = None
        
        logger.info("AI Scope initialized")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply zoom to center or tracked area
        
        Args:
            frame: Input BGR image
            
        Returns:
            Zoomed frame
        """
        if not self.enabled:
            return frame
        
        try:
            h, w = frame.shape[:2]
            
            # Default to screen center
            if self.zoom_center is None:
                self.zoom_center = (w // 2, h // 2)
            
            cx, cy = self.zoom_center
            
            # Calculate zoom region
            half_size = int(self.zoom_area_size / 2)
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)
            
            # Extract region
            zoomed_region = frame[y1:y2, x1:x2]
            
            # Resize to zoom level
            new_w = int(zoomed_region.shape[1] * self.zoom_level)
            new_h = int(zoomed_region.shape[0] * self.zoom_level)
            zoomed = cv2.resize(zoomed_region, (new_w, new_h), 
                              interpolation=cv2.INTER_LINEAR)
            
            # Create output frame
            output = frame.copy()
            
            # Calculate position to paste zoomed region
            paste_x = max(0, (w - new_w) // 2)
            paste_y = max(0, (h - new_h) // 2)
            
            # Handle overflow
            if paste_x + new_w > w:
                new_w = w - paste_x
                zoomed = zoomed[:, :new_w]
            if paste_y + new_h > h:
                new_h = h - paste_y
                zoomed = zoomed[:new_h, :]
            
            # Paste zoomed region
            output[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = zoomed
            
            # Draw crosshair at center
            cv2.drawMarker(output, (w//2, h//2), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Draw zoom indicator
            cv2.putText(output, f"ZOOM: {self.zoom_level:.1f}x", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            return output
            
        except Exception as e:
            logger.error(f"AI Scope processing error: {e}")
            return frame
    
    def set_zoom_level(self, level: float):
        """Set zoom level"""
        self.zoom_level = max(1.0, min(5.0, level))
        logger.info(f"Zoom level set to: {self.zoom_level:.1f}x")
    
    def toggle(self):
        """Toggle scope zoom on/off"""
        self.enabled = not self.enabled
        logger.info(f"AI Scope {'enabled' if self.enabled else 'disabled'}")
