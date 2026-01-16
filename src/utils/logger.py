"""
Logging Configuration
"""

import logging
import sys
from pathlib import Path

# Create logs directory
Path("logs").mkdir(exist_ok=True)

def setup_logger(debug_mode=False):
    """Setup main application logger"""
    level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/assistant.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('SmartAI')

def get_logger(name):
    """Get logger for specific module"""
    return logging.getLogger(f'SmartAI.{name}')
