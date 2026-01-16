"""
AI Tracker - Object Detection & Highlighting
Automatically highlights in-game characters using YOLO
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AITracker:
    """AI-powered object tracking and highlighting"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.confidence_threshold = config['confidence_threshold']
        self.highlight_color = tuple(config['highlight_color'])
        self.thickness = config['highlight_thickness']
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {config['model']}")
        self.model = YOLO(config['model'])
        
        # Classes to detect
        self.target_classes = config.get('detect_classes', ['person'])
        
        logger.info("AI Tracker initialized")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame and highlight detected objects
        
        Args:
            frame: Input BGR image
            
        Returns:
            Processed frame with highlights
        """
        if not self.enabled:
            return frame
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            # Draw bounding boxes
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    # Filter by confidence and class
                    if conf >= self.confidence_threshold and class_name in self.target_classes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                    self.highlight_color, self.thickness)
                        
                        # Draw label
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  self.highlight_color, 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"AI Tracker processing error: {e}")
            return frame
    
    def toggle(self):
        """Toggle tracker on/off"""
        self.enabled = not self.enabled
        logger.info(f"AI Tracker {'enabled' if self.enabled else 'disabled'}")
