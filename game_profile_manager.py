#!/usr/bin/env python3
"""
GameMate AI Assistant - Game Profile Manager
Advanced game-specific configuration and optimization profiles
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil
import win32gui
import win32process
import logging
from datetime import datetime

class GameProfileManager:
    """Advanced game profile management system"""
    
    def __init__(self):
        self.profiles_dir = Path("game_profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.active_profile = None
        self.default_profile = "default"
        self.logger = logging.getLogger(__name__)
        
        # Built-in game profiles
        self.create_builtin_profiles()
        
        # Game detection
        self.game_processes = {}
        self.detection_enabled = True
    
    def create_builtin_profiles(self):
        """Create built-in game profiles"""
        profiles = {
            "fortnite": {
                "name": "Fortnite",
                "description": "Optimized for Fortnite Battle Royale",
                "executables": ["FortniteClient-Win64-Shipping.exe", "FortniteLauncher.exe"],
                "window_titles": ["Fortnite"],
                "settings": {
                    "ai_tracker": {
                        "enabled": True,
                        "confidence_threshold": 0.8,
                        "model": "yolo11s",
                        "track_players": True,
                        "track_weapons": True,
                        "track_vehicles": False
                    },
                    "ai_vision": {
                        "enabled": True,
                        "brightness_boost": 1.3,
                        "contrast_boost": 1.2,
                        "visibility_enhancement": True,
                        "storm_clarity": True
                    },
                    "ai_goggle": {
                        "enabled": True,
                        "flash_protection": 0.7,
                        "explosion_dampening": True
                    },
                    "ai_scope": {
                        "enabled": True,
                        "zoom_factor": 2.0,
                        "auto_zoom": True
                    },
                    "performance": {
                        "fps_limit": 60,
                        "priority": "high",
                        "gpu_optimization": True
                    }
                }
            },
            "valorant": {
                "name": "Valorant",
                "description": "Tactical FPS optimization for Valorant",
                "executables": ["VALORANT-Win64-Shipping.exe", "Valorant.exe"],
                "window_titles": ["VALORANT"],
                "settings": {
                    "ai_tracker": {
                        "enabled": True,
                        "confidence_threshold": 0.9,
                        "model": "yolo11m",
                        "track_players": True,
                        "track_weapons": True,
                        "crosshair_detection": True
                    },
                    "ai_vision": {
                        "enabled": True,
                        "brightness_boost": 1.1,
                        "contrast_boost": 1.4,
                        "enemy_highlighting": True
                    },
                    "ai_goggle": {
                        "enabled": True,
                        "flash_protection": 0.9,
                        "smoke_clarity": True
                    },
                    "performance": {
                        "fps_limit": 120,
                        "priority": "realtime",
                        "latency_optimization": True,
                        "gpu_optimization": True
                    }
                }
            },
            "csgo": {
                "name": "Counter-Strike 2",
                "description": "Competitive FPS settings for CS2",
                "executables": ["cs2.exe", "csgo.exe"],
                "window_titles": ["Counter-Strike 2", "Counter-Strike: Global Offensive"],
                "settings": {
                    "ai_tracker": {
                        "enabled": True,
                        "confidence_threshold": 0.85,
                        "model": "yolo11s",
                        "track_players": True,
                        "track_grenades": True,
                        "sound_visualization": True
                    },
                    "ai_vision": {
                        "enabled": True,
                        "brightness_boost": 1.2,
                        "contrast_boost": 1.3,
                        "smoke_penetration": True
                    },
                    "ai_goggle": {
                        "enabled": True,
                        "flash_protection": 0.95,
                        "adaptive_brightness": True
                    },
                    "performance": {
                        "fps_limit": 300,
                        "priority": "high",
                        "input_optimization": True,
                        "gpu_optimization": True
                    }
                }
            },
            "apex": {
                "name": "Apex Legends",
                "description": "Battle royale optimization for Apex",
                "executables": ["r5apex.exe"],
                "window_titles": ["Apex Legends"],
                "settings": {
                    "ai_tracker": {
                        "enabled": True,
                        "confidence_threshold": 0.75,
                        "model": "yolo11m",
                        "track_players": True,
                        "track_loot": True,
                        "third_party_detection": True
                    },
                    "ai_vision": {
                        "enabled": True,
                        "brightness_boost": 1.25,
                        "contrast_boost": 1.15,
                        "legend_ability_highlight": True
                    },
                    "ai_scope": {
                        "enabled": True,
                        "zoom_factor": 1.5,
                        "scope_assistance": True
                    },
                    "performance": {
                        "fps_limit": 144,
                        "priority": "high",
                        "prediction_optimization": True,
                        "gpu_optimization": True
                    }
                }
            },
            "cod": {
                "name": "Call of Duty",
                "description": "COD series optimization",
                "executables": ["ModernWarfare.exe", "BlackOpsColdWar.exe", "cod.exe"],
                "window_titles": ["Call of Duty", "Modern Warfare", "Black Ops"],
                "settings": {
                    "ai_tracker": {
                        "enabled": True,
                        "confidence_threshold": 0.8,
                        "model": "yolo11s",
                        "track_players": True,
                        "killstreak_detection": True
                    },
                    "ai_vision": {
                        "enabled": True,
                        "brightness_boost": 1.3,
                        "contrast_boost": 1.2,
                        "muzzle_flash_reduction": True
                    },
                    "ai_goggle": {
                        "enabled": True,
                        "flash_protection": 0.8,
                        "stun_protection": True
                    },
                    "performance": {
                        "fps_limit": 120,
                        "priority": "high",
                        "network_optimization": True,
                        "gpu_optimization": True
                    }
                }
            },
            "default": {
                "name": "Default Profile",
                "description": "Generic gaming profile for unknown games",
                "executables": ["*"],
                "window_titles": ["*"],
                "settings": {
                    "ai_tracker": {
                        "enabled": True,
                        "confidence_threshold": 0.7,
                        "model": "yolo11n"
                    },
                    "ai_vision": {
                        "enabled": True,
                        "brightness_boost": 1.1,
                        "contrast_boost": 1.1
                    },
                    "ai_goggle": {
                        "enabled": False
                    },
                    "ai_scope": {
                        "enabled": False
                    },
                    "performance": {
                        "fps_limit": 60,
                        "priority": "normal",
                        "gpu_optimization": True
                    }
                }
            }
        }
        
        # Save built-in profiles
        for profile_id, profile_data in profiles.items():
            self.save_profile(profile_id, profile_data)
    
    def save_profile(self, profile_id: str, profile_data: Dict[str, Any]):
        """Save game profile to file"""
        try:
            profile_path = self.profiles_dir / f"{profile_id}.yaml"
            with open(profile_path, 'w') as f:
                yaml.safe_dump(profile_data, f, indent=2)
            
            self.logger.info(f"Profile saved: {profile_id}")
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile_id}: {e}")
    
    def load_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Load game profile from file"""
        try:
            profile_path = self.profiles_dir / f"{profile_id}.yaml"
            if not profile_path.exists():
                return None
            
            with open(profile_path, 'r') as f:
                profile_data = yaml.safe_load(f)
            
            return profile_data
        except Exception as e:
            self.logger.error(f"Failed to load profile {profile_id}: {e}")
            return None
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all available profiles"""
        profiles = []
        
        for profile_file in self.profiles_dir.glob("*.yaml"):
            profile_id = profile_file.stem
            profile_data = self.load_profile(profile_id)
            
            if profile_data:
                profiles.append({
                    "id": profile_id,
                    "name": profile_data.get("name", profile_id),
                    "description": profile_data.get("description", ""),
                    "active": profile_id == self.active_profile
                })
        
        return profiles
    
    def detect_active_game(self) -> Optional[str]:
        """Detect currently running game"""
        if not self.detection_enabled:
            return None
        
        try:
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            
            window_title = win32gui.GetWindowText(hwnd)
            
            # Get process info
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(process_id)
            executable = process.name()
            
            self.logger.debug(f"Active window: '{window_title}' ({executable})")
            
            # Match against profiles
            for profile_file in self.profiles_dir.glob("*.yaml"):
                profile_id = profile_file.stem
                profile_data = self.load_profile(profile_id)
                
                if not profile_data:
                    continue
                
                # Check executables
                executables = profile_data.get("executables", [])
                if executable.lower() in [exe.lower() for exe in executables]:
                    return profile_id
                
                # Check window titles
                window_titles = profile_data.get("window_titles", [])
                for title_pattern in window_titles:
                    if title_pattern == "*" or title_pattern.lower() in window_title.lower():
                        return profile_id
            
            # No specific profile found, use default
            return self.default_profile
            
        except Exception as e:
            self.logger.debug(f"Game detection failed: {e}")
            return None
    
    def switch_profile(self, profile_id: str) -> bool:
        """Switch to a specific game profile"""
        try:
            profile_data = self.load_profile(profile_id)
            if not profile_data:
                self.logger.error(f"Profile not found: {profile_id}")
                return False
            
            self.active_profile = profile_id
            self.logger.info(f"Switched to profile: {profile_data.get('name', profile_id)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch profile {profile_id}: {e}")
            return False
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get settings for currently active profile"""
        if not self.active_profile:
            # Auto-detect game
            detected_game = self.detect_active_game()
            if detected_game:
                self.switch_profile(detected_game)
            else:
                self.active_profile = self.default_profile
        
        profile_data = self.load_profile(self.active_profile)
        if profile_data:
            return profile_data.get("settings", {})
        
        # Fallback to default
        default_profile = self.load_profile(self.default_profile)
        if default_profile:
            return default_profile.get("settings", {})
        
        return {}
    
    def create_custom_profile(self, profile_id: str, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new custom profile"""
        # Start with default settings
        default_profile = self.load_profile(self.default_profile)
        
        custom_profile = {
            "name": name,
            "description": description,
            "custom": True,
            "created": datetime.now().isoformat(),
            "executables": [],
            "window_titles": [],
            "settings": default_profile.get("settings", {}) if default_profile else {}
        }
        
        self.save_profile(profile_id, custom_profile)
        return custom_profile
    
    def update_profile_settings(self, profile_id: str, settings: Dict[str, Any]) -> bool:
        """Update settings for a specific profile"""
        try:
            profile_data = self.load_profile(profile_id)
            if not profile_data:
                return False
            
            profile_data["settings"].update(settings)
            profile_data["modified"] = datetime.now().isoformat()
            
            self.save_profile(profile_id, profile_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update profile {profile_id}: {e}")
            return False
    
    def get_profile_statistics(self) -> Dict[str, Any]:
        """Get statistics about profile usage"""
        stats = {
            "total_profiles": len(list(self.profiles_dir.glob("*.yaml"))),
            "active_profile": self.active_profile,
            "detection_enabled": self.detection_enabled,
            "builtin_profiles": 6,
            "custom_profiles": 0
        }
        
        # Count custom profiles
        for profile_file in self.profiles_dir.glob("*.yaml"):
            profile_data = self.load_profile(profile_file.stem)
            if profile_data and profile_data.get("custom", False):
                stats["custom_profiles"] += 1
        
        return stats

def create_profile_gui():
    """Create GUI for profile management"""
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    root = tk.Tk()
    root.title("🎮 Game Profile Manager")
    root.geometry("800x600")
    
    manager = GameProfileManager()
    
    # Profile list
    profile_frame = ttk.Frame(root)
    profile_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    ttk.Label(profile_frame, text="Game Profiles", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
    
    # Listbox for profiles
    listbox = tk.Listbox(profile_frame, height=15)
    listbox.pack(fill='both', expand=True)
    
    # Load profiles into listbox
    profiles = manager.list_profiles()
    for profile in profiles:
        status = " (ACTIVE)" if profile["active"] else ""
        listbox.insert(tk.END, f"{profile['name']}{status}")
    
    # Buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(fill='x', padx=10, pady=10)
    
    ttk.Button(button_frame, text="Activate Profile", 
               command=lambda: activate_selected_profile()).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Create Custom", 
               command=lambda: create_custom_profile()).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Detect Game", 
               command=lambda: detect_current_game()).pack(side='left', padx=5)
    
    def activate_selected_profile():
        selection = listbox.curselection()
        if selection:
            profile = profiles[selection[0]]
            if manager.switch_profile(profile["id"]):
                messagebox.showinfo("Success", f"Activated profile: {profile['name']}")
                root.destroy()
    
    def create_custom_profile():
        # Simple dialog for custom profile
        name = tk.simpledialog.askstring("Custom Profile", "Enter profile name:")
        if name:
            profile_id = name.lower().replace(" ", "_")
            manager.create_custom_profile(profile_id, name)
            messagebox.showinfo("Success", f"Created custom profile: {name}")
    
    def detect_current_game():
        detected = manager.detect_active_game()
        if detected:
            profile = manager.load_profile(detected)
            game_name = profile.get("name", detected) if profile else detected
            messagebox.showinfo("Game Detected", f"Detected game: {game_name}")
        else:
            messagebox.showinfo("No Game", "No supported game detected")
    
    root.mainloop()

if __name__ == "__main__":
    # Demo game profile system
    print("🎮 GameMate AI Assistant - Game Profile Manager Demo")
    
    manager = GameProfileManager()
    
    # List profiles
    profiles = manager.list_profiles()
    print(f"\n📋 Available Profiles ({len(profiles)}):")
    for profile in profiles:
        print(f"   • {profile['name']}: {profile['description']}")
    
    # Try to detect current game
    detected = manager.detect_active_game()
    if detected:
        profile_data = manager.load_profile(detected)
        game_name = profile_data.get("name", detected) if profile_data else detected
        print(f"\n🎯 Detected Game: {game_name}")
        
        # Show settings for detected game
        settings = manager.get_current_settings()
        print(f"📊 Active Settings:")
        for category, config in settings.items():
            if isinstance(config, dict) and config.get("enabled"):
                print(f"   ✅ {category.replace('_', ' ').title()}")
    else:
        print("\n❓ No game detected - using default profile")
    
    # Statistics
    stats = manager.get_profile_statistics()
    print(f"\n📈 Profile Statistics:")
    print(f"   Total profiles: {stats['total_profiles']}")
    print(f"   Built-in: {stats['builtin_profiles']}")
    print(f"   Custom: {stats['custom_profiles']}")
    print(f"   Active: {stats['active_profile']}")
    
    # Optional GUI
    try:
        import tkinter.simpledialog
        response = input("\n🖥️ Open profile manager GUI? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            create_profile_gui()
    except ImportError:
        print("   GUI unavailable (tkinter not found)")