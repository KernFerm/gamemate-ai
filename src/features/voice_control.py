"""
Voice Control - Hands-Free Assistant
Voice-activated control for all features
"""

import importlib
import threading
import time

# Optional speech recognition - graceful fallback if not available
sr = None
try:
    sr = importlib.import_module('speech_recognition')
except ImportError:
    pass  # speech_recognition not available - voice control disabled

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


class VoiceControl:
    """Voice-activated feature control"""
    
    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        self.enabled = True
        self.language = config['language']
        self.wake_word = config['wake_word'].lower()
        self.commands = config['commands']
        
        if sr is None:
            logger.warning("Voice control disabled - speech_recognition not available")
            self.enabled = False
            return
        
        # Speech recognition
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
        except Exception as e:
            logger.error(f"Voice control initialization failed: {e}")
            self.enabled = False
            return
        
        # Threading
        self.listening = False
        self.listen_thread = None
        
        logger.info("Voice Control initialized")
    
    def start(self):
        """Start voice control thread"""
        if self.listening:
            return
        
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        logger.info("Voice control started")
    
    def stop(self):
        """Stop voice control"""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
        logger.info("Voice control stopped")
    
    def _listen_loop(self):
        """Main listening loop"""
        logger.info("Voice listening loop started")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while self.listening:
            try:
                with self.microphone as source:
                    logger.debug("Listening for wake word...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(
                        audio, 
                        language=self.language
                    ).lower()
                    
                    logger.debug(f"Heard: {text}")
                    
                    # Check for wake word
                    if self.wake_word in text:
                        logger.info("Wake word detected!")
                        self._process_command(text)
                    
                except sr.UnknownValueError:
                    pass  # Could not understand audio
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {e}")
                
            except sr.WaitTimeoutError:
                pass  # Timeout, continue listening
            except Exception as e:
                logger.error(f"Voice control error: {e}")
                time.sleep(1)
    
    def _process_command(self, text: str):
        """Process recognized command"""
        for cmd in self.commands:
            phrase = cmd['phrase'].lower()
            action = cmd['action']
            
            if phrase in text:
                logger.info(f"Executing command: {action}")
                self._execute_action(action)
                return
        
        logger.warning(f"Unknown command: {text}")
    
    def _execute_action(self, action: str):
        """Execute the specified action"""
        action_map = {
            'enable_tracker': lambda: self.engine.toggle_feature('tracker'),
            'disable_tracker': lambda: self.engine.toggle_feature('tracker'),
            'enable_vision': lambda: self.engine.toggle_feature('vision'),
            'disable_vision': lambda: self.engine.toggle_feature('vision'),
            'enable_scope': lambda: self.engine.toggle_feature('scope'),
            'disable_scope': lambda: self.engine.toggle_feature('scope'),
        }
        
        if action in action_map:
            action_map[action]()
        else:
            logger.warning(f"Unknown action: {action}")
    
    def toggle(self):
        """Toggle voice control on/off"""
        self.enabled = not self.enabled
        logger.info(f"Voice Control {'enabled' if self.enabled else 'disabled'}")
