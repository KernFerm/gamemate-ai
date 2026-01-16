"""
Overlay Manager
Displays processed frames and UI elements
"""

import cv2
import numpy as np
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class OverlayManager:
    """Manages on-screen overlay display"""
    
    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        self.window_name = "Smart AI Assistant"
        
        logger.info("Overlay manager initialized")
    
    def show(self):
        """Show overlay window and start display loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        logger.info("Overlay display started")
        
        try:
            while self.engine.running:
                frame = self.engine.get_current_frame()
                
                if frame is not None:
                    # Add FPS counter
                    self._draw_info(frame)
                    
                    # Display
                    cv2.imshow(self.window_name, frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
        
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            logger.info("Overlay closed")
    
    def _draw_info(self, frame: np.ndarray):
        """Draw information overlay"""
        # Draw semi-transparent panel
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, panel_height), (0, 0, 0), -1)
        frame[:panel_height, :300] = cv2.addWeighted(
            frame[:panel_height, :300], 0.3, 
            overlay[:panel_height, :300], 0.7, 0
        )
        
        # Draw text
        y_offset = 25
        cv2.putText(frame, "Smart AI Assistant", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        active_features = [name for name in self.engine.features.keys()]
        cv2.putText(frame, f"Active: {len(active_features)} features", 
                   (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, "Press 'Q' to exit", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
