"""
AI Scene - Automatic Scene Detection & Mode Switching
Detects game content and applies optimal display settings
"""

import cv2
import numpy as np
from typing import Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AIScene:
    """AI-powered scene detection and mode switching"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.auto_detect = config['auto_detect']
        self.modes = config['modes']
        self.current_mode = 'fps'  # Default mode
        
        # Scene detection parameters
        self.frame_buffer = []
        self.buffer_size = 30  # Analyze last 30 frames
        
        logger.info("AI Scene initialized")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect scene type and apply optimal settings
        
        Args:
            frame: Input BGR image
            
        Returns:
            Enhanced frame based on scene type
        """
        if not self.enabled:
            return frame
        
        try:
            # Auto-detect scene if enabled
            if self.auto_detect:
                detected_mode = self._detect_scene(frame)
                if detected_mode != self.current_mode:
                    logger.info(f"Scene changed: {self.current_mode} -> {detected_mode}")
                    self.current_mode = detected_mode
            
            # Apply mode settings
            mode_settings = self.modes.get(self.current_mode, {})
            enhanced = self._apply_mode(frame, mode_settings)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"AI Scene processing error: {e}")
            return frame
    
    def _detect_scene(self, frame: np.ndarray) -> str:
        """
        Detect scene type based on image characteristics
        
        Args:
            frame: Input BGR image
            
        Returns:
            Detected mode name
        """
        # Add frame to buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Analyze color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation mean
        saturation = np.mean(hsv[:, :, 1])
        
        # Calculate edge density (action intensity)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple heuristic for scene detection
        if edge_density > 0.15:
            return 'fps'  # High action = FPS game
        elif saturation > 100:
            return 'rpg'  # Colorful = RPG
        else:
            return 'moba'  # Default to MOBA
    
    def _apply_mode(self, frame: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply mode-specific enhancements"""
        enhanced = frame.astype(np.float32)
        
        # Apply brightness
        brightness = settings.get('brightness', 1.0)
        enhanced = enhanced * brightness
        
        # Apply contrast
        contrast = settings.get('contrast', 1.0)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=0)
        
        # Apply saturation
        saturation = settings.get('saturation', 1.0)
        if saturation != 1.0:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def set_mode(self, mode: str):
        """Manually set scene mode"""
        if mode in self.modes:
            self.current_mode = mode
            logger.info(f"Scene mode set to: {mode}")
    
    def toggle(self):
        """Toggle scene detection on/off"""
        self.enabled = not self.enabled
        logger.info(f"AI Scene {'enabled' if self.enabled else 'disabled'}")
