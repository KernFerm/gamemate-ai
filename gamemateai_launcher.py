#!/usr/bin/env python3
"""
GameMate AI Assistant - GUI Launcher
Simple GUI to control the GameMate AI Assistant
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import subprocess
import sys
from pathlib import Path
import os
import time
from game_profile_manager import GameProfileManager

class GameMateLauncherGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎮 GameMate AI Launcher")
        self.root.geometry("850x650")
        self.root.configure(bg='#1e1e1e')  # GameMate Dark theme
        
        # GameMate Color scheme
        self.brand_red = "#d50000"
        self.brand_dark = "#1e1e1e"
        self.brand_gray = "#333333"
        self.brand_white = "#ffffff"
        # Backwards-compatible aliases used elsewhere in the UI
        self.gamemate_red = self.brand_red
        self.gamemate_dark = self.brand_dark
        self.gamemate_gray = self.brand_gray
        self.gamemate_white = self.brand_white
        
        self.assistant_process = None
        self.demo_process = None
        
        # Initialize Game Profile Manager
        self.profile_manager = GameProfileManager()
        self.current_profile = "default"
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Main title
        title_frame = tk.Frame(self.root, bg=self.gamemate_dark)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🎮 GameMate AI",
            font=('Arial', 18, 'bold'),
            fg=self.brand_red,
            bg=self.brand_dark
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Professional Gaming AI - Hardware Optimized",
            font=('Arial', 10),
            fg=self.brand_white,
            bg=self.brand_dark
        )
        subtitle_label.pack()
        
        # Control Panel
        control_frame = tk.Frame(self.root, bg=self.gamemate_gray, relief='ridge', bd=2)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            control_frame,
            text="🎯 Control Panel",
            font=('Arial', 12, 'bold'),
            fg=self.gamemate_white,
            bg=self.gamemate_gray
        ).pack(pady=5)
        
        # Game Profile Selection
        profile_frame = tk.Frame(control_frame, bg=self.gamemate_gray)
        profile_frame.pack(pady=10)
        
        tk.Label(
            profile_frame,
            text="🎮 Game Profile:",
            font=('Arial', 10, 'bold'),
            fg=self.gamemate_white,
            bg=self.gamemate_gray
        ).pack(side='left', padx=5)
        
        # Get available profiles
        profiles = self.profile_manager.list_profiles()
        self.profile_names = {p['id']: p['name'] for p in profiles}
        
        # Profile dropdown
        self.profile_var = tk.StringVar(value=self.profile_names.get(self.current_profile, "Default"))
        self.profile_dropdown = ttk.Combobox(
            profile_frame,
            textvariable=self.profile_var,
            values=list(self.profile_names.values()),
            state='readonly',
            width=25,
            font=('Arial', 10)
        )
        self.profile_dropdown.pack(side='left', padx=5)
        self.profile_dropdown.bind('<<ComboboxSelected>>', self.on_profile_change)
        
        # Active profile status
        self.profile_status = tk.Label(
            profile_frame,
            text=f"✅ Active",
            font=('Arial', 9),
            fg="#00ff00",
            bg=self.gamemate_gray
        )
        self.profile_status.pack(side='left', padx=10)
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg=self.gamemate_gray)
        button_frame.pack(pady=10)
        
        # Start/Stop buttons
        self.start_btn = tk.Button(
            button_frame,
            text="🚀 Start GameMate Assistant (Background)",
            command=self.start_assistant,
            bg=self.gamemate_red,
            fg=self.gamemate_white,
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=2,
            width=30
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Stop Assistant",
            command=self.stop_assistant,
            bg=self.gamemate_gray,
            fg=self.gamemate_white,
            font=('Arial', 10),
            relief='raised',
            bd=2,
            width=20,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        # Exit button
        self.exit_btn = tk.Button(
            button_frame,
            text="❌ Exit Launcher",
            command=self.exit_application,
            bg="#666666",
            fg=self.gamemate_white,
            font=('Arial', 10),
            relief='raised',
            bd=2,
            width=15
        )
        self.exit_btn.pack(side='left', padx=5)
        
        # Demo button
        demo_frame = tk.Frame(control_frame, bg=self.gamemate_gray)
        demo_frame.pack(pady=5)
        
        self.demo_btn = tk.Button(
            demo_frame,
            text="👁️ Visual Demo (With Overlay)",
            command=self.start_demo,
            bg="#0066cc",
            fg=self.gamemate_white,
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=2,
            width=30
        )
        self.demo_btn.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg=self.gamemate_gray, relief='ridge', bd=2)
        status_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(
            status_frame,
            text="📊 System Status",
            font=('Arial', 12, 'bold'),
            fg=self.gamemate_white,
            bg=self.gamemate_gray
        ).pack(pady=5)
        
        # Status display
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            height=15,
            bg=self.gamemate_dark,
            fg=self.gamemate_white,
            font=('Consolas', 9),
            relief='sunken',
            bd=2
        )
        self.status_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg=self.gamemate_dark)
        footer_frame.pack(fill='x', pady=5)
        
        tk.Label(
            footer_frame,
            text="GameMate AI v1.0 | Ready for Action! 🎮",
            font=('Arial', 8),
            fg=self.brand_white,
            bg=self.brand_dark
        ).pack()
        
        # Initial status
        self.log_status("🎮 GameMate AI Launcher Ready!")
        self.log_status(f"📋 Available Profiles: {', '.join(self.profile_names.values())}")
        self.log_status(f"✅ Current Profile: {self.profile_var.get()}")
        self.log_status("👆 Click 'Start GameMate Assistant' for background mode")
        self.log_status("👁️ Click 'Visual Demo' to see the overlay in action")
        
        # Load current profile settings
        self.on_profile_change()
    
    def log_status(self, message):
        """Add message to status log"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert('end', f"[{timestamp}] {message}\n")
        self.status_text.see('end')
        self.root.update()
    
    def on_profile_change(self, event=None):
        """Handle game profile selection change"""
        selected_name = self.profile_var.get()
        
        # Find profile ID by name
        profile_id = None
        for pid, pname in self.profile_names.items():
            if pname == selected_name:
                profile_id = pid
                break
        
        if profile_id:
            self.log_status(f"🎮 Switching to {selected_name} profile...")
            
            # Activate the profile
            success = self.profile_manager.switch_profile(profile_id)
            
            if success:
                self.current_profile = profile_id
                self.profile_status.config(text="✅ Active", fg="#00ff00")
                self.log_status(f"✅ {selected_name} profile activated!")
                
                # Get profile details
                profile = self.profile_manager.load_profile(profile_id)
                if profile:
                    settings = profile.get('settings', {})
                    perf = settings.get('performance', {})
                    ai_tracker = settings.get('ai_tracker', {})
                    self.log_status(f"   ├─ AI Tracker: {'Enabled' if ai_tracker.get('enabled') else 'Disabled'}")
                    self.log_status(f"   ├─ Priority: {perf.get('priority', 'normal')}")
                    self.log_status(f"   └─ GPU Optimization: {'Yes' if perf.get('gpu_optimization') else 'No'}")
                
            else:
                self.profile_status.config(text="❌ Failed", fg="#ff0000")
                self.log_status(f"❌ Failed to activate profile: {profile_id}")
    
    def start_assistant(self):
        """Start the GameMate AI Assistant in background mode"""
        if self.assistant_process and self.assistant_process.poll() is None:
            self.log_status("⚠️ Assistant already running!")
            return
        
        try:
            python_exe = sys.executable
            script_path = Path(__file__).parent / "start_gamemateai.py"
            
            self.log_status("🚀 Starting GameMate AI...")
            self.log_status(f"🎮 Active Profile: {self.profile_var.get()}")
            
            # Set environment variable for the active profile
            env = os.environ.copy()
            env['GAMEMATE_ACTIVE_PROFILE'] = self.current_profile
            
            # Start the assistant
            self.assistant_process = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(__file__).parent),
                env=env
            )
            
            self.start_btn.configure(state='disabled', bg=self.gamemate_gray)
            self.stop_btn.configure(state='normal', bg=self.gamemate_red)
            
            # Monitor the process
            threading.Thread(target=self.monitor_assistant, daemon=True).start()
            
            self.log_status("✅ GameMate AI started in background!")
            self.log_status("💡 Running invisibly - processing your screen for gaming optimization")
            
        except Exception as e:
            self.log_status(f"❌ Failed to start assistant: {e}")
    
    def stop_assistant(self):
        """Stop the GameMate AI Assistant"""
        if self.assistant_process and self.assistant_process.poll() is None:
            self.log_status("⏹️ Stopping GameMate AI...")
            self.assistant_process.terminate()
            self.assistant_process.wait()
            
        self.assistant_process = None
        self.start_btn.configure(state='normal', bg=self.gamemate_red)
        self.stop_btn.configure(state='disabled', bg=self.gamemate_gray)
        self.log_status("✅ GameMate AI stopped")
    
    def exit_application(self):
        """Exit the launcher application"""
        self.log_status("🚪 Exiting GameMate AI Launcher...")
        self.on_closing()
    
    def start_demo(self):
        """Start the visual demo"""
        try:
            python_exe = sys.executable
            script_path = Path(__file__).parent / "visual_demo.py"
            
            self.log_status("👁️ Starting Visual Demo...")
            
            # Start the demo
            self.demo_process = subprocess.Popen(
                [python_exe, str(script_path)],
                cwd=str(Path(__file__).parent)
            )
            
            self.log_status("✅ Visual Demo started - check for overlay window!")
            self.log_status("💡 Press 'q' in the demo window to quit")
            
        except Exception as e:
            self.log_status(f"❌ Failed to start demo: {e}")
    
    def monitor_assistant(self):
        """Monitor the assistant process"""
        while self.assistant_process and self.assistant_process.poll() is None:
            time.sleep(1)
        
        # Process ended
        if self.assistant_process:
            self.root.after(0, lambda: self.log_status("⚠️ Assistant process ended"))
            self.root.after(0, lambda: self.start_btn.configure(state='normal', bg=self.gamemate_red))
            self.root.after(0, lambda: self.stop_btn.configure(state='disabled', bg=self.gamemate_gray))
    
    def on_closing(self):
        """Handle window closing"""
        if self.assistant_process and self.assistant_process.poll() is None:
            self.log_status("🛑 Shutting down GameMate AI Assistant...")
            self.assistant_process.terminate()
        
        if self.demo_process and self.demo_process.poll() is None:
            self.demo_process.terminate()
        
        self.root.destroy()
    
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    launcher = GameMateLauncherGUI()
    launcher.run()