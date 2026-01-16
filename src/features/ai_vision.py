"""
AI Vision+ - Brightness & Clarity Enhancement
Smarter brightness adaptation and detail enhancement
"""

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AIVision:
    """AI-powered visual enhancement"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.brightness_boost = config['brightness_boost']
        self.contrast_boost = config['contrast_boost']
        self.sharpen = config['sharpen']
        self.denoise = config['denoise']
        self.adaptive = config['adaptive_mode']
        
        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        logger.info("AI Vision+ initialized")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame brightness and clarity
        
        Args:
            frame: Input BGR image
            
        Returns:
            Enhanced frame
        """
        if not self.enabled:
            return frame
        
        try:
            # Convert to float for processing
            enhanced = frame.astype(np.float32)
            
            # Adaptive brightness adjustment
            if self.adaptive:
                mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                adaptive_boost = 1.0 + (128 - mean_brightness) / 256
                brightness_factor = self.brightness_boost * adaptive_boost
            else:
                brightness_factor = self.brightness_boost
            
            # Apply brightness
            enhanced = enhanced * brightness_factor
            
            # Apply contrast using CLAHE for better results
            lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.contrast_boost, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR).astype(np.float32)
            
            # Apply sharpening
            if self.sharpen:
                enhanced = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
            
            # Apply denoising
            if self.denoise:
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced.astype(np.uint8), None, 10, 10, 7, 21
                )
                enhanced = enhanced.astype(np.float32)
            
            # Clip and convert back
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"AI Vision+ processing error: {e}")
            return frame
    
    def toggle(self):
        """Toggle vision enhancement on/off"""
        self.enabled = not self.enabled
        logger.info(f"AI Vision+ {'enabled' if self.enabled else 'disabled'}")
