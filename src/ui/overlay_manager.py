"""
GameMate AI Assistant - Overlay Manager
Manages the gaming overlay UI with GameMate branding
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class OverlayManager:
    """Manages the GameMate overlay interface"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.opacity = config.get('opacity', 0.8)
        self.position = config.get('position', 'top_right')
        
        # MSI Colors
        self.msi_red = (0, 0, 255)  # BGR format
        self.msi_cyan = (255, 255, 0)  # BGR format  
        self.msi_white = (255, 255, 255)
        self.msi_black = (0, 0, 0)
        
        # Overlay components
        self.components = {
            'fps_counter': True,
            'ai_status': True,
            'detection_overlay': True,
            'performance_stats': True,
        }
        
        # Fonts
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        logger.info("MSI Overlay Manager initialized")
    
    def draw_msi_banner(self, frame: np.ndarray) -> np.ndarray:
        """Draw MSI branding banner"""
        if not self.enabled:
            return frame
            
        try:
            h, w = frame.shape[:2]
            
            # Top banner
            banner_height = 40
            cv2.rectangle(frame, (0, 0), (w, banner_height), self.msi_black, -1)
            
            # MSI logo text
            cv2.putText(frame, "MSI GAMING AI", (10, 25), 
                       self.font, 0.7, self.msi_red, 2)
            
            # Status indicator
            status_text = "ACTIVE"
            text_size = cv2.getTextSize(status_text, self.font, 0.5, 1)[0]
            cv2.putText(frame, status_text, (w - text_size[0] - 10, 25), 
                       self.font, 0.5, self.msi_cyan, 1)
            
        except Exception as e:
            logger.error(f"Banner drawing error: {e}")
            
        return frame
    
    def draw_fps_counter(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter"""
        if not self.enabled or not self.components['fps_counter']:
            return frame
            
        try:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 60), 
                       self.font, self.font_scale, self.msi_cyan, self.font_thickness)
                       
        except Exception as e:
            logger.error(f"FPS counter error: {e}")
            
        return frame
    
    def draw_ai_status(self, frame: np.ndarray, ai_stats: Dict) -> np.ndarray:
        """Draw AI status information"""
        if not self.enabled or not self.components['ai_status']:
            return frame
            
        try:
            y_offset = 90
            line_height = 25
            
            # AI Tracker status
            tracker_status = "ON" if ai_stats.get('tracker_enabled', False) else "OFF"
            color = self.msi_cyan if ai_stats.get('tracker_enabled', False) else self.msi_red
            cv2.putText(frame, f"AI Tracker: {tracker_status}", (10, y_offset), 
                       self.font, 0.5, color, 1)
            
            # Detection count
            detections = ai_stats.get('detections', 0)
            cv2.putText(frame, f"Detections: {detections}", (10, y_offset + line_height), 
                       self.font, 0.5, self.msi_white, 1)
            
            # Processing time
            proc_time = ai_stats.get('processing_time', 0)
            cv2.putText(frame, f"Process: {proc_time:.1f}ms", (10, y_offset + line_height * 2), 
                       self.font, 0.5, self.msi_white, 1)
                       
        except Exception as e:
            logger.error(f"AI status error: {e}")
            
        return frame
    
    def draw_detection_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection bounding boxes"""
        if not self.enabled or not self.components['detection_overlay']:
            return frame
            
        try:
            for detection in detections:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    confidence = detection.get('confidence', 0)
                    class_name = detection.get('class', 'unknown')
                    
                    # Choose color based on class
                    color = self.msi_cyan
                    if class_name in ['enemy', 'enemy_player']:
                        color = self.msi_red
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, self.font, 0.4, 1)[0]
                    cv2.rectangle(frame, (int(x), int(y) - label_size[1] - 10), 
                                 (int(x) + label_size[0], int(y)), color, -1)
                    cv2.putText(frame, label, (int(x), int(y) - 5), 
                               self.font, 0.4, self.msi_black, 1)
                               
        except Exception as e:
            logger.error(f"Detection boxes error: {e}")
            
        return frame
    
    def draw_performance_stats(self, frame: np.ndarray, gpu_stats: Dict) -> np.ndarray:
        """Draw performance statistics"""
        if not self.enabled or not self.components['performance_stats']:
            return frame
            
        try:
            h, w = frame.shape[:2]
            x_pos = w - 200
            y_offset = 60
            line_height = 20
            
            # Background panel
            cv2.rectangle(frame, (x_pos - 10, y_offset - 30), 
                         (w - 10, y_offset + line_height * 4), 
                         (0, 0, 0), -1)
            
            # GPU usage
            gpu_usage = gpu_stats.get('gpu_usage', 0)
            cv2.putText(frame, f"GPU: {gpu_usage}%", (x_pos, y_offset), 
                       self.font, 0.4, self.msi_white, 1)
            
            # Memory usage
            mem_usage = gpu_stats.get('memory_usage', 0)
            cv2.putText(frame, f"VRAM: {mem_usage}%", (x_pos, y_offset + line_height), 
                       self.font, 0.4, self.msi_white, 1)
            
            # Temperature
            temp = gpu_stats.get('temperature', 0)
            cv2.putText(frame, f"Temp: {temp}°C", (x_pos, y_offset + line_height * 2), 
                       self.font, 0.4, self.msi_white, 1)
            
            # Power
            power = gpu_stats.get('power_usage', 0)
            cv2.putText(frame, f"Power: {power}W", (x_pos, y_offset + line_height * 3), 
                       self.font, 0.4, self.msi_white, 1)
                       
        except Exception as e:
            logger.error(f"Performance stats error: {e}")
            
        return frame
    
    def draw_crosshair(self, frame: np.ndarray, style: str = 'default') -> np.ndarray:
        """Draw gaming crosshair"""
        if not self.enabled:
            return frame
            
        try:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            if style == 'default':
                # Simple cross
                cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), 
                        self.msi_cyan, 1)
                cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), 
                        self.msi_cyan, 1)
            elif style == 'circle':
                # Circle crosshair
                cv2.circle(frame, (center_x, center_y), 15, self.msi_cyan, 1)
                cv2.circle(frame, (center_x, center_y), 2, self.msi_red, -1)
                
        except Exception as e:
            logger.error(f"Crosshair error: {e}")
            
        return frame
    
    def render_overlay(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Render complete overlay"""
        if not self.enabled:
            return frame
            
        try:
            # Draw all components
            frame = self.draw_msi_banner(frame)
            
            if 'fps' in kwargs:
                frame = self.draw_fps_counter(frame, kwargs['fps'])
                
            if 'ai_stats' in kwargs:
                frame = self.draw_ai_status(frame, kwargs['ai_stats'])
                
            if 'detections' in kwargs:
                frame = self.draw_detection_boxes(frame, kwargs['detections'])
                
            if 'gpu_stats' in kwargs:
                frame = self.draw_performance_stats(frame, kwargs['gpu_stats'])
                
            if kwargs.get('show_crosshair', False):
                frame = self.draw_crosshair(frame, kwargs.get('crosshair_style', 'default'))
                
        except Exception as e:
            logger.error(f"Overlay rendering error: {e}")
            
        return frame
    
    def toggle_component(self, component: str):
        """Toggle overlay component on/off"""
        if component in self.components:
            self.components[component] = not self.components[component]
            logger.info(f"Toggled {component}: {self.components[component]}")
    
    def set_opacity(self, opacity: float):
        """Set overlay opacity"""
        self.opacity = max(0.0, min(1.0, opacity))
    
    def cleanup(self):
        """Cleanup overlay resources"""
        logger.info("Overlay manager cleaned up")