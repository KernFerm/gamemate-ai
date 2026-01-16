#!/usr/bin/env python3
"""
GameMate AI Assistant - Safe Startup Script
Ensures proper initialization and handles edge cases for reliable startup
"""

import os
import sys
import time
import logging
from pathlib import Path

def setup_environment():
    """Setup the environment for safe execution"""
    # Set CUDA environment variables for Numba CUDA support
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    ]
    
    # Find and set CUDA_PATH if not already set
    if 'CUDA_PATH' not in os.environ:
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                os.environ['CUDA_PATH'] = cuda_path
                os.environ['CUDA_HOME'] = cuda_path
                print(f"✅ CUDA Toolkit found: {cuda_path}")
                break
    
    # Add src directory to Python path and ensure folders exist
    script_dir = Path(__file__).parent.absolute()
    src_dir = script_dir / 'src'

    print("🎮 GameMate AI - Starting Up...")

    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # Ensure output directories exist
    dirs_to_create = [
        'logs',
        'models',
        'temp',
        'screenshots'
    ]

    for dir_name in dirs_to_create:
        dir_path = script_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    
    # Setup basic logging for early messages
    log_file = script_dir / 'logs' / f'gamemate_assistant_{int(time.time())}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return script_dir

def check_dependencies():
    """Check for critical dependencies and provide helpful messages"""
    missing_deps = []
    optional_deps = []
    
    # Critical dependencies
    critical = [
        ('cv2', 'opencv-python', 'pip install opencv-python'),
        ('numpy', 'numpy', 'pip install numpy'),
        ('yaml', 'pyyaml', 'pip install pyyaml'),
        ('mss', 'mss', 'pip install mss'),
        ('colorama', 'colorama', 'pip install colorama'),
    ]
    
    for module, package, install_cmd in critical:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append((package, install_cmd))
    
    # Optional dependencies
    optional = [
        ('torch', 'PyTorch', 'pip install torch torchvision'),
        ('ultralytics', 'YOLO', 'pip install ultralytics'),
        ('cupy', 'CuPy', 'pip install cupy-cuda11x'),
        ('nvidia_ml_py', 'NVIDIA-ML', 'pip install nvidia-ml-py'),
        ('speech_recognition', 'Speech Recognition', 'pip install SpeechRecognition'),
        ('pytesseract', 'Tesseract OCR', 'pip install pytesseract'),
    ]
    
    for module, package, install_cmd in optional:
        try:
            __import__(module)
        except ImportError:
            optional_deps.append((package, install_cmd))
    
    return missing_deps, optional_deps

def safe_import_config():
    """Safely import and validate configuration"""
    try:
        import yaml
        
        config_path = Path('config.yaml')
        if not config_path.exists():
            logging.error("config.yaml not found!")
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections - support both legacy 'msi' key and 'gamemate'
        required_sections = ['features', 'games', 'ai_settings']
        top_level_ok = any(k in config for k in ('msi', 'gamemate'))
        for section in required_sections:
            if section not in config:
                logging.error(f"Missing required config section: {section}")
                return None
        if not top_level_ok:
            logging.error("Missing top-level 'gamemate' or legacy 'msi' section in config")
            return None
        
        logging.info("Configuration loaded successfully")
        return config
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return None

def initialize_logging_system():
    """Initialize comprehensive logging system"""
    try:
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Setup rotating log files (max 10MB each, keep 5 files)
        file_handler = RotatingFileHandler(
            log_dir / 'gamemate_assistant.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logging.info("Logging system initialized")
        return True
        
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
        return False

def check_system_requirements():
    """Check system requirements and capabilities"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check available memory
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 2.0:
            issues.append(f"Low memory: {available_memory:.1f}GB available, 2GB+ recommended")
    except ImportError:
        pass
    
    # Check if running on Windows (for GameMate features)
    if os.name != 'nt':
        issues.append("GameMate features optimized for Windows")
    
    return issues

def main():
    """Main startup function with comprehensive error handling"""
    print("🎮 GameMate AI Assistant - Starting Up...")
    print("=" * 50)
    
    # Setup environment
    try:
        script_dir = setup_environment()
        print(f"✅ Environment setup complete: {script_dir}")
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False
    
    # Initialize logging
    try:
        initialize_logging_system()
        print("✅ Logging system ready")
    except Exception as e:
        print(f"⚠️  Logging setup warning: {e}")
    
    # Check system requirements
    system_issues = check_system_requirements()
    if system_issues:
        print("⚠️  System warnings:")
        for issue in system_issues:
            print(f"   • {issue}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing_deps, optional_deps = check_dependencies()
    
    if missing_deps:
        print("❌ Missing critical dependencies:")
        for package, install_cmd in missing_deps:
            print(f"   • {package}: {install_cmd}")
        print("\nPlease install missing dependencies before continuing.")
        return False
    
    if optional_deps:
        print("⚠️  Optional dependencies not found (reduced functionality):")
        for package, install_cmd in optional_deps:
            print(f"   • {package}: {install_cmd}")
        print("Install these for full MSI Gaming features.")
    
    # Load configuration
    print("\n⚙️  Loading configuration...")
    config = safe_import_config()
    if config is None:
        print("❌ Configuration error - cannot continue")
        return False
    
    # Import and start the main application
    print("\n🚀 Initializing GameMate AI Assistant...")
    
    # Check for active game profile from launcher
    active_profile = os.environ.get('GAMEMATE_ACTIVE_PROFILE', 'default')
    if active_profile != 'default':
        print(f"🎮 Active Game Profile: {active_profile.upper()}")
    
    try:
        # Import main components using robust module loading
        import importlib.util
        import sys
        
        # Add src to path
        src_path = str(script_dir / 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Load MSI Gaming Engine
        engine_spec = importlib.util.spec_from_file_location(
            "gamemate_engine",
            str(script_dir / 'src' / 'core' / 'engine.py')
        )
        engine_module = importlib.util.module_from_spec(engine_spec)
        engine_spec.loader.exec_module(engine_module)
        MSIGamingEngine = getattr(engine_module, 'MSIGamingEngine', None)
        if MSIGamingEngine is None:
            raise ImportError('GameMateAIEngine class not found in engine module')
        
        # Load Overlay Manager with fallback
        try:
            overlay_spec = importlib.util.spec_from_file_location(
                "overlay_manager",
                str(script_dir / 'src' / 'ui' / 'overlay_manager.py')
            )
            overlay_module = importlib.util.module_from_spec(overlay_spec)
            overlay_spec.loader.exec_module(overlay_module)
            OverlayManager = overlay_module.OverlayManager
        except Exception:
            # Create dummy overlay manager if not available
            class OverlayManager:
                def __init__(self, config): pass
                def cleanup(self): pass
        
        print("✅ Core components imported")
        
        # Create engine instance
        engine = MSIGamingEngine(config)
        print("✅ GameMate Engine created")
        
        # Create overlay manager
        overlay = OverlayManager(config.get('ui', {}))
        print("✅ Overlay manager ready")
        
        # Start the system
        print("\n🎯 Starting GameMate AI Assistant...")
        print("=" * 50)
        
        # Display startup banner
        print("""
██╗   ██╗███████╗██╗     ██████╗ ██████╗ ███╗   ███╗███████╗
██║   ██║██╔════╝██║    ██╔════╝██╔═══██╗████╗ ████║██╔════╝
██║   ██║█████╗  ██║    ██║     ██║   ██║██╔████╔██║█████╗  
██║   ██║██╔══╝  ██║    ██║     ██║   ██║██║╚██╔╝██║██╔══╝  
╚██████╔╝███████╗██║    ╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚══════╝╚═╝     ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝

        GameMate AI Assistant - READY FOR ACTION! 🎮
        """)
        
        # Initialize and start engine
        engine.initialize()
        
        print("🎉 GameMate AI Assistant is now running!")
        print("Press Ctrl+C to stop the application.")
        
        # Start the main loop
        engine.start()
        
        # Keep the main thread alive
        try:
            while engine.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Shutting down GameMate AI Assistant...")
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down GameMate AI Assistant...")
        try:
            engine.stop()
            overlay.cleanup()
        except:
            pass
        print("👋 Goodbye!")
        
    except Exception as e:
        logging.error(f"Critical error in main application: {e}", exc_info=True)
        print(f"❌ Critical error: {e}")
        print("Check logs for detailed error information.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()

    sys.exit(0 if success else 1)
