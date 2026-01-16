"""
AI Gauge - Status Monitoring & OCR
Monitors in-game status using OCR
"""

import cv2
import numpy as np
import importlib
from typing import Dict, Any

# Optional OCR library - graceful fallback if not available
pytesseract = None
try:
    pytesseract = importlib.import_module('pytesseract')
except ImportError:
    pass  # pytesseract not available - OCR features disabled

# Robust logger import that works in both package and standalone contexts
try:
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from utils.logger import get_logger
    except ImportError:
        # Ultimate fallback
        import logging
        def get_logger(name):
            return logging.getLogger(name)

logger = get_logger(__name__)


class AIGauge:
    """AI-powered status gauge monitoring"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.ocr_regions = config['ocr_regions']
        self.update_interval = config['update_interval']
        
        # Last detected values
        self.last_values = {}
        
        logger.info("AI Gauge initialized")
    
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract game status from screen regions
        
        Args:
            frame: Input BGR image
            
        Returns:
            Dictionary of detected values
        """
        if not self.enabled:
            return self.last_values
        
        if pytesseract is None:
            logger.warning("OCR not available - pytesseract not installed")
            return self.last_values
        
        try:
            stats = {}
            
            for region_name, region_coords in self.ocr_regions.items():
                x, y, w, h = region_coords
                
                # Extract region
                roi = frame[y:y+h, x:x+w]
                
                # Preprocess for OCR
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Run OCR
                text = pytesseract.image_to_string(
                    thresh, 
                    config='--psm 7 digits'
                ).strip()
                
                # Try to extract numbers
                try:
                    # Look for numeric patterns
                    import re
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        stats[region_name] = int(numbers[0])
                except:
                    stats[region_name] = text
            
            self.last_values = stats
            return stats
            
        except Exception as e:
            logger.error(f"AI Gauge processing error: {e}")
            return self.last_values
    
    def toggle(self):
        """Toggle gauge monitoring on/off"""
        self.enabled = not self.enabled
        logger.info(f"AI Gauge {'enabled' if self.enabled else 'disabled'}")
