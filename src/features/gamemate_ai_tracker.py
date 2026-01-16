"""
GameMate AI Detector - Hardware Optimized Object Detection
Uses advanced gaming-specific AI models and TensorRT optimization
"""

import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from typing import List, Tuple, Dict
import importlib

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

try:
    import cupy as cp
except ImportError:
    cp = None

# Optional CUDA acceleration - graceful fallback if not available
cuda = None
float32 = None
try:
    from numba import cuda, float32
except ImportError:
    pass  # numba not available - CUDA acceleration disabled
except Exception:
    cuda = None
    float32 = None
import win32gui
import win32api
import ctypes

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
from pathlib import Path

logger = get_logger(__name__)


class GameMateAITracker:
    """GameMateAI Intelligence - Hardware-optimized object detection"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = True
        
        # Initialize NVIDIA GPU
        self._init_gpu()
        
        # Initialize GPU kernels
        self._init_gpu_kernels()
        
        # Load GameMateAI gaming-specific ONNX model
        self._load_gamemateai_model()
        
        # Game detection
        self.current_game = None
        self.game_detector = GameDetector()
        
        # Performance optimization
        self.cuda_context = None
        self.gpu_memory_pool = None
        
        logger.info("GameMate AI Tracker initialized with hardware acceleration")
    
    def cleanup_memory(self):
        """Clean up memory resources"""
        import gc
        
        if self.session is not None:
            del self.session
            self.session = None
        
        # Clean up GPU resources
        if hasattr(self, 'cuda_context') and self.cuda_context:
            self.cuda_context = None
        
        if hasattr(self, 'gpu_memory_pool') and self.gpu_memory_pool:
            self.gpu_memory_pool = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("GameMate AI Tracker memory cleaned up")
    
    def _init_gpu(self):
        """Initialize NVIDIA GPU for GameMateAI acceleration"""
        try:
            if pynvml is None:
                logger.warning("pynvml not available - GPU monitoring disabled")
                self.gpu_handle = None
                return
                
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            # Handle both string and bytes return types
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            if 'RTX' in name or 'GTX' in name:
                logger.info(f"GameMate Gaming GPU detected: {name}")
                
                # Initialize CUDA context for GameMateAI acceleration
                if cuda is not None:
                    cuda.select_device(0)
                    self.cuda_context = cuda.current_context()
                
                # Allocate dedicated GPU memory pool
                if cp is not None:
                    self.gpu_memory_pool = cp.get_default_memory_pool()
                    self.gpu_memory_pool.set_limit(size=2**31)  # 2GB for AI processing
                
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            self.gpu_handle = None
    
    def _load_gamemateai_model(self):
        """Load GameMateAI's proprietary gaming detection model"""
        try:
            if ort is None:
                logger.warning("ONNX Runtime not available, using YOLO fallback")
                self._load_yolo_fallback()
                return
            
            # GameMateAI uses TensorRT-optimized ONNX models
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            
            model_path = 'models/gamemate_gaming_detector.onnx'
            if not Path(model_path).exists():
                logger.warning(f"GameMate model not found at {model_path}, using YOLO fallback")
                self._load_yolo_fallback()
                return
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info("GameMate gaming detection model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load GameMate model: {e}")
            self._load_yolo_fallback()
    
    def _load_yolo_fallback(self):
        """Load YOLO as fallback with memory optimization"""
        # Implement lazy loading - don't load until actually needed
        self.session = None
        self._yolo_model_path = 'yolov8n.pt'  # Store path for lazy loading
        logger.info("YOLO fallback configured (lazy loading)")
    
    def _ensure_yolo_loaded(self):
        """Lazy load YOLO model only when needed"""
        if self.session is None and hasattr(self, '_yolo_model_path'):
            try:
                from ultralytics import YOLO
                import torch
                
                # Load with memory optimizations
                self.session = YOLO(self._yolo_model_path)
                
                # Optimize model for inference
                self.session.model.eval()
                
                # Move to GPU and use FP16 precision if CUDA available
                if torch.cuda.is_available():
                    self.session.to('cuda')  # Move model to GPU
                    self.session.model.half()  # Use FP16 precision
                    logger.info("YOLO model loaded with GPU acceleration (CUDA + FP16)")
                else:
                    logger.info("YOLO model loaded with CPU (CUDA not available)")
                
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.session = None
    
    def _init_gpu_kernels(self):
        """Initialize GPU kernels if Numba CUDA is available (optional optimization)"""
        if cuda is None:
            logger.info("Numba CUDA not available - using PyTorch GPU acceleration instead")
            self.gpu_preprocess_kernel = None
            return
            
        try:
            @cuda.jit
            def _gpu_preprocess_kernel(image, output, mean, std):
                """CUDA kernel for image preprocessing"""
                i, j = cuda.grid(2)
                if i < image.shape[0] and j < image.shape[1]:
                    for c in range(3):
                        output[c, i, j] = (image[i, j, c] / 255.0 - mean[c]) / std[c]
            
            self.gpu_preprocess_kernel = _gpu_preprocess_kernel
            logger.info("CUDA kernels initialized")
            
        except Exception as e:
            logger.warning(f"CUDA kernel initialization failed: {e}")
            self.gpu_preprocess_kernel = None
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        GameMateAI Intelligence processing with hardware acceleration
        """
        if not self.enabled:
            return frame
        
        try:
            # Detect current game
            current_game = self.game_detector.detect_game()
            if current_game != self.current_game:
                self.current_game = current_game
                self._load_game_profile(current_game)
            
            # GPU-accelerated preprocessing
            processed_frame = self._gpu_preprocess(frame)
            
            # Run GameMateAI gaming detection
            detections = self._run_gamemateai_detection(processed_frame)
            
            # Apply GameMateAI highlighting
            result = self._apply_gamemateai_highlights(frame, detections)
            
            return result
            
        except Exception as e:
            logger.error(f"GameMate AI Tracker error: {e}")
            return frame
    
    def _gpu_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated image preprocessing using CuPy"""
        with cp.cuda.Device(0):
            # Transfer to GPU
            gpu_frame = cp.asarray(frame)
            
            # Resize using GPU
            gpu_resized = cp.array(cv2.resize(cp.asnumpy(gpu_frame), (640, 640)))
            
            # Normalize
            gpu_normalized = (gpu_resized.astype(cp.float32) / 255.0)
            
            # Convert to CHW format
            gpu_chw = cp.transpose(gpu_normalized, (2, 0, 1))
            
            # Add batch dimension
            gpu_batch = cp.expand_dims(gpu_chw, axis=0)
            
            return cp.asnumpy(gpu_batch)
    
    def _run_gamemateai_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run GameMateAI gaming-specific detection with lazy loading"""
        if hasattr(self.session, 'run'):
            # ONNX Runtime
            outputs = self.session.run(self.output_names, {self.input_name: frame})
            return self._parse_onnx_outputs(outputs)
        else:
            # YOLO fallback with lazy loading
            self._ensure_yolo_loaded()  # Load model only when needed
            if self.session is not None:
                # Use smaller input size to save memory
                results = self.session(frame, verbose=False, imgsz=320)
                return self._parse_yolo_outputs(results)
            else:
                return []  # Return empty if model failed to load
    
    def _parse_onnx_outputs(self, outputs) -> List[Dict]:
        """Parse GameMateAI model outputs"""
        detections = []
        
        # GameMateAI model output format: [batch, detections, 6] (x1,y1,x2,y2,conf,class)
        for detection in outputs[0][0]:
            if detection[4] > self.config['confidence_threshold']:
                detections.append({
                    'bbox': detection[:4],
                    'confidence': detection[4],
                    'class_id': int(detection[5]),
                    'class_name': self._get_class_name(int(detection[5]))
                })
        
        return detections
    
    def _apply_gamemateai_highlights(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Apply GameMateAI-style highlighting with gaming-optimized colors"""
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']
            
            # GameMateAI gaming color scheme
            color = self._get_gamemateai_color(class_name)
            
            # Enhanced highlighting for gaming
            # Outer glow effect
            cv2.rectangle(result, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 3)
            cv2.rectangle(result, (x1-1, y1-1), (x2+1, y2+1), color, 2)
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # GameMateAI-style label with background
            label = f"{class_name} {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 2)[0]
            
            # Label background
            cv2.rectangle(result, (x1, y1-25), (x1+label_size[0]+10, y1-5), (0, 0, 0), -1)
            cv2.rectangle(result, (x1, y1-25), (x1+label_size[0]+10, y1-5), color, 2)
            
            # Label text
            cv2.putText(result, label, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def _get_gamemateai_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get GameMateAI gaming-specific colors"""
        color_map = {
            'enemy_player': (0, 0, 255),      # Red for enemies
            'enemy_head': (0, 100, 255),      # Orange for headshots
            'teammate': (0, 255, 0),          # Green for teammates
            'weapon': (255, 255, 0),          # Cyan for weapons
            'loot': (0, 255, 255),           # Yellow for loot
            'loot_chest': (0, 255, 255),     # Yellow for Fortnite chests
            'shield_potion': (255, 0, 255),  # Magenta for shields
            'building': (128, 128, 255),     # Light blue for Fortnite builds
            'bomb': (255, 0, 255),           # Magenta for objectives
        }
        return color_map.get(class_name, (0, 255, 255))  # Default GameMateAI cyan
    
    def _load_game_profile(self, game: str):
        """Load game-specific detection profile"""
        profiles = self.config.get('game_profiles', {})
        if game in profiles:
            profile = profiles[game]
            self.config['confidence_threshold'] = profile.get('confidence', 0.3)
            self.config['detect_classes'] = profile.get('classes', ['enemy_player'])
            logger.info(f"Loaded profile for {game}")


class GameDetector:
    """Detect currently running game using Windows APIs"""
    
    def __init__(self):
        self.known_games = {
            'VALORANT  ': 'valorant',
            'Counter-Strike: Global Offensive': 'csgo',
            'Apex Legends': 'apex_legends',
            'Fortnite': 'fortnite',
            'Call of Duty': 'cod',
            'League of Legends': 'lol',
            'Overwatch': 'overwatch'
        }
    
    def detect_game(self) -> str:
        """Detect active game window"""
        try:
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd)
            
            # Match against known games
            for title, game_id in self.known_games.items():
                if title.lower() in window_title.lower():
                    return game_id
            
            return 'unknown'
            
        except Exception:
            return 'unknown'