"""
AI Goggle - Flash Protection & Recovery
Recovers faster from flashbangs and bright flashes
"""

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AIGoggle:
    """AI-powered flash protection"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.flash_threshold = config['flash_threshold']
        self.recovery_speed = config['recovery_speed']
        self.darkness_compensation = config['darkness_compensation']
        
        # Flash detection state
        self.is_flashed = False
        self.flash_intensity = 0.0
        
        logger.info("AI Goggle initialized")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and compensate for bright flashes
        
        Args:
            frame: Input BGR image
            
        Returns:
            Flash-compensated frame
        """
        if not self.enabled:
            return frame
        
        try:
            # Calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # Detect flash
            if avg_brightness > self.flash_threshold:
                self.is_flashed = True
                self.flash_intensity = min(1.0, avg_brightness / 255.0)
                logger.debug(f"Flash detected: {avg_brightness:.1f}")
            
            # Apply recovery if flashed
            if self.is_flashed and self.flash_intensity > 0:
                # Reduce flash intensity over time
                self.flash_intensity = max(0, self.flash_intensity - self.recovery_speed)
                
                # Darken bright areas
                enhanced = frame.astype(np.float32)
                
                # Create mask for bright areas
                mask = (gray > self.flash_threshold).astype(np.float32)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                
                # Apply darkening to bright areas
                darkening_factor = 1.0 - (self.flash_intensity * 0.7)
                for i in range(3):
                    enhanced[:, :, i] = enhanced[:, :, i] * (
                        darkening_factor + mask * (1 - darkening_factor)
                    )
                
                # Boost dark areas for visibility
                dark_mask = (gray < 100).astype(np.float32)
                dark_mask = cv2.GaussianBlur(dark_mask, (21, 21), 0)
                
                for i in range(3):
                    enhanced[:, :, i] = enhanced[:, :, i] + (
                        dark_mask * 30 * self.flash_intensity
                    )
                
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                
                # Reset if recovered
                if self.flash_intensity <= 0:
                    self.is_flashed = False
                
                return enhanced
            
            return frame
            
        except Exception as e:
            logger.error(f"AI Goggle processing error: {e}")
            return frame
    
    def toggle(self):
        """Toggle goggle protection on/off"""
        self.enabled = not self.enabled
        logger.info(f"AI Goggle {'enabled' if self.enabled else 'disabled'}")
