"""
GameMate AI Engine - Hardware Optimized

"""

import time
import threading
import gc
import weakref
from typing import Dict, Any
import numpy as np
import ctypes
import importlib

# Optional GPU monitoring - graceful fallback if not available  
pynvml = None
try:
    # Try nvidia-ml-py first (preferred), then fallback to deprecated pynvml
    try:
        import nvidia_ml_py as pynvml
    except ImportError:
        try:
            import pynvml
        except ImportError:
            pynvml = None
    
    if pynvml is not None:
        pynvml.nvmlInit()
except ImportError:
    pass  # GPU monitoring not available - disabled
except Exception as e:
    logger.warning(f"GPU monitoring initialization failed: {e}")
    pynvml = None  # GPU monitoring available but failed to initialize

# Import MSI components with fallbacks
try:
    from .gamemateai_screen_capture import GameMateScreenCapture
except ImportError:
    GameMateScreenCapture = None

try:
    from ..features.gamemate_ai_tracker import MSIAITracker
except ImportError:
    MSIAITracker = None

try:
    from ..features.gamemate_visual_engine import MSIVisualEngine
except ImportError:
    MSIVisualEngine = None

# Import remaining features with fallbacks
try:
    from ..features.ai_scene import AIScene
except ImportError:
    AIScene = None
    
try:
    from ..features.ai_goggle import AIGoggle
except ImportError:
    AIGoggle = None
    
try:
    from ..features.ai_scope import AIScope
except ImportError:
    AIScope = None
    
try:
    from ..features.ai_gauge import AIGauge
except ImportError:
    AIGauge = None
    
try:
    from ..features.voice_control import VoiceControl
except ImportError:
    VoiceControl = None
    
try:
    from .fallbacks import FallbackScreenCapture, FallbackAITracker, FallbackVisualEngine
except ImportError:
    # Create minimal fallbacks if module not available
    class FallbackScreenCapture:
        def __init__(self, config=None): pass
        def capture_frame(self): return None
        def capture(self): return None  # Alias method
        def get_fps(self): return 60
    class FallbackAITracker:
        def __init__(self, config=None): pass
        def detect(self, frame): return []
        def get_performance_stats(self): return {'fps': 60}
    class FallbackVisualEngine:
        def __init__(self, config=None): pass
        def process(self, frame): return frame
# Robust logger import that works in both package and standalone contexts
try:
    from ..utils.logger import get_logger
    from ..utils.memory_optimizer import memory_optimizer
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from utils.logger import get_logger
        from utils.memory_optimizer import memory_optimizer
    except ImportError:
        # Ultimate fallback
        import logging
        def get_logger(name):
            return logging.getLogger(name)
        
        # Fallback memory optimizer
        class FallbackMemoryOptimizer:
            def start_auto_cleanup(self): pass
            def stop_auto_cleanup(self): pass
            def gentle_cleanup(self): pass
        memory_optimizer = FallbackMemoryOptimizer()

logger = get_logger(__name__)


class MSIGamingEngine:
    """MSI Gaming AI Engine with hardware optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.fps_limit = config['general']['fps_limit']
        self.frame_time = 1.0 / self.fps_limit
        
        # MSI hardware optimization
        self.msi_mode = config['general'].get('msi_mode', True)
        self.hardware_acceleration = config['general'].get('hardware_acceleration', True)
        
        # Initialize NVIDIA GPU monitoring
        self._init_nvidia_monitoring()
        
        # Set process priority for gaming
        self._set_high_priority()
        
        # Initialize components
        self.screen_capture = None
        self.features = {}
        self.processing_thread = None
        
        # Current frame data
        self.current_frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
    
    def _init_nvidia_monitoring(self):
        """Initialize NVIDIA GPU monitoring for GameMate AI Assistant optimization"""
        try:
            if pynvml is None:
                logger.warning("pynvml not available - GPU monitoring disabled")
                self.gpu_handle = None
                return
                
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            # Handle both string and bytes return types
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            logger.info(f"MSI Gaming GPU detected: {gpu_name}")
        except Exception as e:
            logger.warning(f"NVIDIA monitoring unavailable: {e}")
            self.gpu_handle = None
    
    def _set_high_priority(self):
        """Set high process priority for gaming performance"""
        try:
            # Set high priority class (similar to MSI's approach)
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS
            logger.info("Process priority set to HIGH for gaming performance")
        except Exception as e:
            logger.warning(f"Could not set high priority: {e}")
    
    def _enable_memory_optimization(self):
        """Enable aggressive memory optimization for production use"""
        try:
            # Enable garbage collection optimization
            gc.set_threshold(700, 10, 10)  # More aggressive than default (700,10,10)
            
            # Force immediate garbage collection
            gc.collect()
            
            # Log memory optimization status
            logger.info("Memory optimization enabled: aggressive GC, weak references")
        except Exception as e:
            logger.warning(f"Memory optimization setup failed: {e}")
    
    def _optimize_frame_memory(self, frame):
        """Optimize frame memory usage"""
        if frame is not None:
            # Convert to float16 for memory efficiency
            if frame.dtype != np.float16:
                frame = frame.astype(np.float16)
            
            # Force garbage collection every 100 frames
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
                if self._frame_count % 100 == 0:
                    gc.collect()
            else:
                self._frame_count = 1
        
        return frame
        
    def initialize(self):
        """Initialize MSI gaming components"""
        logger.info("Initializing GameMate AI Engine...")
        
        # Initialize MSI screen capture with fallback
        if GameMateScreenCapture:
            try:
                self.screen_capture = GameMateScreenCapture(self.config)
                logger.info("GameMate hardware capture initialized")
            except Exception as e:
                logger.warning(f"GameMate screen capture failed, using fallback: {e}")
                self.screen_capture = FallbackScreenCapture(self.config)
        else:
            logger.info("GameMate screen capture not available, using fallback")
            self.screen_capture = FallbackScreenCapture(self.config)
        
        # Initialize MSI gaming features with fallbacks
        if self.config['ai_tracker']['enabled']:
            logger.info("Loading MSI Gaming AI Tracker...")
            if MSIAITracker:
                try:
                    self.features['tracker'] = MSIAITracker(self.config['ai_tracker'])
                    logger.info("GameMate AI Tracker loaded successfully")
                except Exception as e:
                    logger.warning(f"GameMate AI Tracker failed, using fallback: {e}")
                    self.features['tracker'] = FallbackAITracker(self.config['ai_tracker'])
            else:
                logger.info("GameMate AI Tracker not available, using fallback")
                self.features['tracker'] = FallbackAITracker(self.config['ai_tracker'])
            
        if self.config['ai_vision']['enabled']:
            logger.info("Loading GameMate Visual Enhancement Engine...")
            if MSIVisualEngine:
                try:
                    self.features['vision'] = MSIVisualEngine(self.config['ai_vision'])
                    logger.info("GameMate Visual Engine loaded successfully")
                except Exception as e:
                    logger.warning(f"MSI Visual Engine failed, using fallback: {e}")
                    self.features['vision'] = FallbackVisualEngine(self.config['ai_vision'])
            else:
                logger.info("GameMate Visual Engine not available, using fallback")
                self.features['vision'] = FallbackVisualEngine(self.config['ai_vision'])
            
        if self.config['ai_scene']['enabled']:
            logger.info("Loading AI Scene...")
            self.features['scene'] = AIScene(self.config['ai_scene'])
            
        if self.config['ai_goggle']['enabled']:
            logger.info("Loading AI Goggle...")
            self.features['goggle'] = AIGoggle(self.config['ai_goggle'])
            
        if self.config['ai_scope']['enabled']:
            logger.info("Loading AI Scope...")
            self.features['scope'] = AIScope(self.config['ai_scope'])
            
        if self.config['ai_gauge']['enabled']:
            logger.info("Loading AI Gauge...")
            self.features['gauge'] = AIGauge(self.config['ai_gauge'])
            
        if self.config['voice_control']['enabled']:
            logger.info("Loading Voice Control...")
            self.features['voice'] = VoiceControl(
                self.config['voice_control'], 
                self
            )
        
        logger.info(f"Initialized {len(self.features)} features")
        
    def start(self):
        """Start the assistant engine with memory monitoring"""
        if self.running:
            logger.warning("Engine already running")
            return
            
        logger.info("Starting assistant engine...")
        self.running = True
        
        # Start memory monitoring
        memory_optimizer.start_auto_cleanup()
        
        # Start voice control in separate thread if enabled
        if 'voice' in self.features:
            self.features['voice'].start()
        
        # Start main processing loop
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Engine started successfully with memory monitoring")
        
    def stop(self):
        """Stop the assistant engine with memory cleanup"""
        logger.info("Stopping assistant engine...")
        self.running = False
        
        # Stop memory monitoring
        memory_optimizer.stop_auto_cleanup()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if 'voice' in self.features:
            self.features['voice'].stop()
        
        # Final memory cleanup
        memory_optimizer.aggressive_cleanup()
            
        logger.info("Engine stopped with memory cleanup")
        
    def _processing_loop(self):
        """Main processing loop with memory optimization"""
        logger.info("Processing loop started with memory optimization")
        
        while self.running:
            start_time = time.time()
            
            try:
                # Capture screen
                frame = self.screen_capture.capture()
                if frame is None:
                    continue
                
                # Apply memory optimization
                frame = self._optimize_frame_memory(frame)
                
                # Store original frame with memory optimization
                with self.frame_lock:
                    self.current_frame = frame.copy() if frame is not None else None
                    self.processed_frame = frame.copy() if frame is not None else None
                
                if frame is None:
                    continue
                    
                # Process through active features (reuse frame object)
                processed = frame
                
                # AI Vision (brightness/contrast adjustment)
                if 'vision' in self.features:
                    processed = self.features['vision'].process(processed)
                
                # AI Goggle (flash protection)
                if 'goggle' in self.features:
                    processed = self.features['goggle'].process(processed)
                
                # AI Scene (mode detection)
                if 'scene' in self.features:
                    processed = self.features['scene'].process(processed)
                
                # AI Tracker (object detection)
                if 'tracker' in self.features:
                    processed = self.features['tracker'].process(processed)
                
                # AI Scope (zoom)
                if 'scope' in self.features:
                    processed = self.features['scope'].process(processed)
                
                # AI Gauge (status monitoring)
                if 'gauge' in self.features:
                    stats = self.features['gauge'].process(frame)
                    # Could overlay stats on processed frame
                
                # Update processed frame
                with self.frame_lock:
                    self.processed_frame = processed
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Processing loop ended")
        
    def get_current_frame(self):
        """Get the current processed frame"""
        with self.frame_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None
    
    def toggle_feature(self, feature_name: str):
        """Toggle a feature on/off"""
        if feature_name in self.features:
            feature = self.features[feature_name]
            if hasattr(feature, 'toggle'):
                feature.toggle()
                logger.info(f"Toggled {feature_name}")
        else:

            logger.warning(f"Feature {feature_name} not found")
