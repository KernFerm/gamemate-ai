"""
Hardware-Accelerated Screen Capture
Uses DirectX/DXGI for low-latency screen capture
"""

import numpy as np
import ctypes
from ctypes import wintypes
import win32gui
import win32api
import win32con
import cv2
import importlib
from typing import Optional, Tuple

# Optional GPU monitoring - graceful fallback if not available
pynvml = None
try:
    # Try nvidia-ml-py first (preferred), then fallback to pynvml
    try:
        import nvidia_ml_py as pynvml
    except ImportError:
        import pynvml
    pynvml.nvmlInit()
except ImportError:
    pass  # GPU monitoring not available - disabled
except Exception as e:
    pynvml = None  # GPU monitoring available but failed to initialize

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


# Windows API structures
class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long)]


class GameMateScreenCapture:
    """GameMate-style hardware-accelerated screen capture"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize Windows GDI+ for hardware capture
        self.gdi32 = ctypes.windll.gdi32
        self.user32 = ctypes.windll.user32
        
        # Get primary monitor
        self.monitor_handle = self.user32.GetDC(0)
        self.monitor_width = self.user32.GetSystemMetrics(0)
        self.monitor_height = self.user32.GetSystemMetrics(1)
        
        # Create compatible DC for fast blitting
        self.memory_dc = self.gdi32.CreateCompatibleDC(self.monitor_handle)
        self.bitmap = self.gdi32.CreateCompatibleBitmap(
            self.monitor_handle, self.monitor_width, self.monitor_height
        )
        self.gdi32.SelectObject(self.memory_dc, self.bitmap)
        
        # Initialize NVIDIA hardware acceleration if available
        self._init_nvidia_acceleration()
        
        logger.info(f"MSI Screen Capture initialized: {self.monitor_width}x{self.monitor_height}")
    
    def _init_nvidia_acceleration(self):
        """Initialize NVIDIA GPU for accelerated capture"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            # Handle both string and bytes return types
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            if 'RTX' in name or 'GTX' in name:
                logger.info(f"NVIDIA acceleration enabled: {name}")
                self.nvidia_acceleration = True
            else:
                self.nvidia_acceleration = False
                
        except Exception as e:
            logger.warning(f"NVIDIA acceleration unavailable: {e}")
            self.nvidia_acceleration = False
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Hardware-accelerated screen capture using Windows GDI
        
        Returns:
            numpy.ndarray: BGR image frame
        """
        try:
            # Fast BitBlt operation (hardware-accelerated)
            self.gdi32.BitBlt(
                self.memory_dc, 0, 0, self.monitor_width, self.monitor_height,
                self.monitor_handle, 0, 0, win32con.SRCCOPY
            )
            
            # Get bitmap data
            bitmap_header = self._create_bitmap_header()
            bitmap_data = ctypes.create_string_buffer(self.monitor_width * self.monitor_height * 4)
            
            self.gdi32.GetDIBits(
                self.memory_dc, self.bitmap, 0, self.monitor_height,
                bitmap_data, ctypes.byref(bitmap_header), win32con.DIB_RGB_COLORS
            )
            
            # Convert to numpy array
            frame = np.frombuffer(bitmap_data.raw, dtype=np.uint8)
            frame = frame.reshape((self.monitor_height, self.monitor_width, 4))
            
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Flip vertically (Windows bitmap convention)
            frame = cv2.flip(frame, 0)
            
            return frame
            
        except Exception as e:
            logger.error(f"GameMate screen capture error: {e}")
            return None
    
    def _create_bitmap_header(self):
        """Create Windows BITMAPINFOHEADER structure"""
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD),
                ('biWidth', wintypes.LONG),
                ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD),
                ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD),
                ('biXPelsPerMeter', wintypes.LONG),
                ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD)
            ]
        
        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = self.monitor_width
        header.biHeight = -self.monitor_height  # Negative for top-down
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0  # BI_RGB
        header.biSizeImage = 0
        
        return header
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        Capture specific screen region for AI Scope feature
        
        Args:
            x, y: Top-left coordinates
            width, height: Region dimensions
            
        Returns:
            numpy.ndarray: BGR image of the region
        """
        try:
            # Create compatible DC for region
            region_dc = self.gdi32.CreateCompatibleDC(self.monitor_handle)
            region_bitmap = self.gdi32.CreateCompatibleBitmap(self.monitor_handle, width, height)
            self.gdi32.SelectObject(region_dc, region_bitmap)
            
            # Copy region
            self.gdi32.BitBlt(
                region_dc, 0, 0, width, height,
                self.monitor_handle, x, y, win32con.SRCCOPY
            )
            
            # Get bitmap data
            header = self._create_region_bitmap_header(width, height)
            data = ctypes.create_string_buffer(width * height * 4)
            
            self.gdi32.GetDIBits(
                region_dc, region_bitmap, 0, height,
                data, ctypes.byref(header), win32con.DIB_RGB_COLORS
            )
            
            # Convert to numpy array
            frame = np.frombuffer(data.raw, dtype=np.uint8)
            frame = frame.reshape((height, width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.flip(frame, 0)
            
            # Cleanup
            self.gdi32.DeleteDC(region_dc)
            self.gdi32.DeleteObject(region_bitmap)
            
            return frame
            
        except Exception as e:
            logger.error(f"Region capture error: {e}")
            return None
    
    def _create_region_bitmap_header(self, width: int, height: int):
        """Create bitmap header for region capture"""
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD),
                ('biWidth', wintypes.LONG),
                ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD),
                ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD),
                ('biXPelsPerMeter', wintypes.LONG),
                ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD)
            ]
        
        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0
        header.biSizeImage = 0
        
        return header
    
    def get_game_window_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the region of the active game window
        
        Returns:
            Tuple of (x, y, width, height) or None if no game detected
        """
        try:
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd)
            
            # Check if it's a game window (basic heuristic)
            game_keywords = ['valorant', 'csgo', 'counter-strike', 'apex', 'overwatch', 'call of duty']
            is_game = any(keyword in window_title.lower() for keyword in game_keywords)
            
            if is_game:
                rect = win32gui.GetWindowRect(hwnd)
                return rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
            
            return None
            
        except Exception as e:
            logger.error(f"Game window detection error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup Windows GDI resources"""
        if hasattr(self, 'memory_dc'):
            self.gdi32.DeleteDC(self.memory_dc)
        if hasattr(self, 'bitmap'):
            self.gdi32.DeleteObject(self.bitmap)
        if hasattr(self, 'monitor_handle'):

            self.user32.ReleaseDC(0, self.monitor_handle)
