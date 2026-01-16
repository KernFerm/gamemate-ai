"""
GameMate AI Assistant
"""

import os
import sys
import yaml
from pathlib import Path
from colorama import init, Fore, Style

# Setup CUDA environment for GPU acceleration
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
]

if 'CUDA_PATH' not in os.environ:
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ['CUDA_PATH'] = cuda_path
            os.environ['CUDA_HOME'] = cuda_path
            break

from src.core.engine import GameMateAIGamingEngine
from src.ui.overlay import OverlayManager
from src.utils.logger import setup_logger

# Initialize colorama for colored console output
init(autoreset=True)

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"{Fore.RED}Error: config.yaml not found!")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_banner():
    """Print application banner"""
    banner = f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     {Fore.RED}GameMateAI Assistant{Fore.CYAN}                             ║
║     {Fore.YELLOW}Hardware-Accelerated Gaming Intelligence{Fore.CYAN}        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)

def print_features(config):
    """Print enabled features"""
    print(f"\n{Fore.YELLOW}Active Features:{Style.RESET_ALL}")
    
    features = [
        ("AI Tracker", config['ai_tracker']['enabled'], "🎯"),
        ("AI Vision+", config['ai_vision']['enabled'], "👁️"),
        ("AI Scene", config['ai_scene']['enabled'], "🎮"),
        ("AI Goggle", config['ai_goggle']['enabled'], "🕶️"),
        ("AI Scope", config['ai_scope']['enabled'], "🔭"),
        ("AI Gauge", config['ai_gauge']['enabled'], "📊"),
        ("Voice Control", config['voice_control']['enabled'], "🤖"),
    ]
    
    for name, enabled, icon in features:
        status = f"{Fore.GREEN}ON{Style.RESET_ALL}" if enabled else f"{Fore.RED}OFF{Style.RESET_ALL}"
        print(f"  {icon} {name:15} [{status}]")
    
    print(f"\n{Fore.CYAN}Press Ctrl+Q to exit{Style.RESET_ALL}\n")

def main():
    """Main application entry point"""
    print_banner()
    
    # Load configuration
    try:
        config = load_config()
        logger = setup_logger(config['general']['debug_mode'])
        logger.info("Configuration loaded successfully")
    except Exception as e:
        print(f"{Fore.RED}Error loading configuration: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Print active features
    print_features(config)
    
    # Initialize assistant engine
    try:
        print(f"{Fore.YELLOW}Initializing GameMateAI Engine...{Style.RESET_ALL}")
        engine = GameMateAIGamingEngine(config)
        engine.initialize()
        
        print(f"{Fore.YELLOW}Starting overlay manager...{Style.RESET_ALL}")
        overlay = OverlayManager(config, engine)
        
        print(f"{Fore.GREEN}✓ Assistant ready!{Style.RESET_ALL}\n")
        
        # Start the main loop
        engine.start()
        overlay.show()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutting down gracefully...{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.stop()
        print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
