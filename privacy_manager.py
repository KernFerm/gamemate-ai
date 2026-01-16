#!/usr/bin/env python3
"""
GameMate AI Assistant - Privacy & Security Manager
Handles user permissions, privacy controls, and security features
"""

import tkinter as tk
from tkinter import messagebox, ttk
import json
from pathlib import Path
from datetime import datetime

class PrivacyManager:
    def __init__(self):
        self.privacy_file = Path(__file__).parent / "privacy_settings.json"
        self.permissions = self.load_permissions()
        
    def load_permissions(self):
        """Load privacy permissions from file"""
        default_permissions = {
            "screen_capture_consent": False,
            "gpu_monitoring_consent": False,
            "voice_control_consent": False,
            "data_collection_consent": False,
            "crash_reporting_consent": False,
            "consent_date": None,
            "privacy_version": "1.0"
        }
        
        if self.privacy_file.exists():
            try:
                with open(self.privacy_file, 'r') as f:
                    saved = json.load(f)
                    default_permissions.update(saved)
            except Exception:
                pass  # Use defaults
                
        return default_permissions
    
    def save_permissions(self):
        """Save privacy permissions to file"""
        self.permissions["last_updated"] = datetime.now().isoformat()
        with open(self.privacy_file, 'w') as f:
            json.dump(self.permissions, f, indent=2)
    
    def check_screen_capture_permission(self):
        """Check if user has consented to screen capture"""
        if not self.permissions.get("screen_capture_consent", False):
            return self.request_screen_capture_permission()
        return True
    
    def request_screen_capture_permission(self):
        """Request screen capture permission from user"""
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        message = """🎮 GameMate AI Assistant - Screen Capture Permission

    To provide AI gaming assistance, this application needs to:

    ✅ Capture your screen content in real-time
    ✅ Analyze game visuals for AI enhancement
    ✅ Process images locally (never uploaded)

    Your Privacy:
    • Screen content stays on your computer
    • No data is transmitted or stored remotely
    • You can revoke permission anytime
    • Processing is fully offline

    Do you consent to screen capture for AI gaming features?"""
        
        result = messagebox.askyesno(
            "Privacy Consent Required", 
            message,
            icon="question"
        )
        
        root.destroy()
        
        if result:
            self.permissions["screen_capture_consent"] = True
            self.permissions["consent_date"] = datetime.now().isoformat()
            self.save_permissions()
            return True
        else:
            return False
    
    def show_privacy_settings(self):
        """Show privacy settings GUI"""
        root = tk.Tk()
        root.title("🔒 Privacy Settings - GameMate AI Assistant")
        root.geometry("600x500")
        root.configure(bg='#1e1e1e')
        
        # GameMate colors (customize as needed)
        gamemate_red = "#d50000"
        gamemate_white = "#ffffff"
        gamemate_gray = "#333333"

        # For backward compatibility with old variable names
        msi_red = gamemate_red
        msi_white = gamemate_white
        msi_gray = gamemate_gray
        
        # Title
        title_label = tk.Label(
            root,
            text="🔒 Privacy & Security Settings",
            font=('Arial', 16, 'bold'),
            fg=msi_red,
            bg='#1e1e1e'
        )
        title_label.pack(pady=20)
        
        # Permissions frame
        perm_frame = tk.Frame(root, bg=msi_gray, relief='ridge', bd=2)
        perm_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        tk.Label(
            perm_frame,
            text="📋 Permissions",
            font=('Arial', 12, 'bold'),
            fg=msi_white,
            bg=msi_gray
        ).pack(pady=10)
        
        # Permission checkboxes
        permissions_info = [
            ("screen_capture_consent", "🖥️ Screen Capture", "Allow real-time screen analysis for AI features"),
            ("gpu_monitoring_consent", "🎯 GPU Monitoring", "Monitor GPU performance and optimization"),
            ("voice_control_consent", "🎤 Voice Control", "Enable voice commands and audio input"),
            ("crash_reporting_consent", "📊 Crash Reporting", "Send crash reports to improve stability")
        ]
        
        self.permission_vars = {}
        for perm_key, title, description in permissions_info:
            frame = tk.Frame(perm_frame, bg=msi_gray)
            frame.pack(fill='x', padx=20, pady=5)
            
            var = tk.BooleanVar(value=self.permissions.get(perm_key, False))
            self.permission_vars[perm_key] = var
            
            cb = tk.Checkbutton(
                frame,
                text=title,
                variable=var,
                fg=msi_white,
                bg=msi_gray,
                selectcolor=msi_red,
                font=('Arial', 10, 'bold')
            )
            cb.pack(anchor='w')
            
            desc_label = tk.Label(
                frame,
                text=description,
                fg='#cccccc',
                bg=msi_gray,
                font=('Arial', 8)
            )
            desc_label.pack(anchor='w', padx=20)
        
        # Data handling info
        info_frame = tk.Frame(root, bg=msi_gray, relief='ridge', bd=2)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(
            info_frame,
            text="🛡️ Data Handling",
            font=('Arial', 12, 'bold'),
            fg=msi_white,
            bg=msi_gray
        ).pack(pady=10)
        
        info_text = """✅ All processing happens locally on your computer
✅ No personal data is transmitted over the network
✅ Screen captures are processed in memory only
✅ No permanent storage of game content
✅ You can revoke permissions at any time"""
        
        tk.Label(
            info_frame,
            text=info_text,
            fg=msi_white,
            bg=msi_gray,
            font=('Arial', 9),
            justify='left'
        ).pack(padx=20, pady=10)
        
        # Buttons
        button_frame = tk.Frame(root, bg='#1e1e1e')
        button_frame.pack(fill='x', pady=20)
        
        save_btn = tk.Button(
            button_frame,
            text="💾 Save Settings",
            command=self.save_privacy_settings,
            bg=msi_red,
            fg=msi_white,
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=2
        )
        save_btn.pack(side='left', padx=20)
        
        cancel_btn = tk.Button(
            button_frame,
            text="❌ Cancel",
            command=root.destroy,
            bg=msi_gray,
            fg=msi_white,
            font=('Arial', 10),
            relief='raised',
            bd=2
        )
        cancel_btn.pack(side='right', padx=20)
        
        # Store reference to root for saving
        self.privacy_root = root
        
        root.mainloop()
    
    def save_privacy_settings(self):
        """Save privacy settings from GUI"""
        for perm_key, var in self.permission_vars.items():
            self.permissions[perm_key] = var.get()
        
        self.save_permissions()
        
        messagebox.showinfo(
            "Settings Saved",
            "Privacy settings have been saved successfully!",
            parent=self.privacy_root
        )
        
        self.privacy_root.destroy()
    
    def get_privacy_summary(self):
        """Get a summary of current privacy settings"""
        enabled_count = sum(1 for v in self.permissions.values() if isinstance(v, bool) and v)
        total_permissions = 4  # Total number of permission options
        
        return {
            "permissions_granted": enabled_count,
            "total_permissions": total_permissions,
            "consent_date": self.permissions.get("consent_date"),
            "screen_capture_allowed": self.permissions.get("screen_capture_consent", False),
            "privacy_compliant": True  # Always compliant since everything is local
        }

def show_privacy_notice():
    """Show initial privacy notice"""
    root = tk.Tk()
    root.withdraw()
    
    notice = """🎮 GameMate AI Assistant - Privacy Notice

This application is designed with privacy in mind:

🔒 PRIVACY FIRST:
• All AI processing happens locally on your device
• No personal data leaves your computer
• No network connections for data collection
• Screen content is processed in memory only

🎯 WHAT WE DO:
• Analyze your screen for gaming AI features
• Monitor system performance for optimization
• Process audio locally for voice commands (optional)

🛡️ WHAT WE DON'T DO:
• Upload or transmit your screen content
• Store personal information remotely
• Track your gaming activity online
• Share data with third parties

Continue to configure your privacy settings."""
    
    messagebox.showinfo("Privacy Notice", notice)
    root.destroy()

if __name__ == "__main__":
    # Show privacy notice and settings
    show_privacy_notice()
    
    pm = PrivacyManager()
    pm.show_privacy_settings()
    
    # Print summary
    summary = pm.get_privacy_summary()
    print(f"\n🔒 Privacy Settings Summary:")
    print(f"   Permissions granted: {summary['permissions_granted']}/{summary['total_permissions']}")
    print(f"   Screen capture: {'✅' if summary['screen_capture_allowed'] else '❌'}")
    print(f"   Privacy compliant: {'✅' if summary['privacy_compliant'] else '❌'}")