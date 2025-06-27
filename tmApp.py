#!/usr/bin/env python3
import os
import sys
import json
import threading
import time
import subprocess
import platform
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font, scrolledtext
import cv2
import numpy as np
import re
from plyer import notification
import schedule
import pandas as pd
from PIL import Image, ImageEnhance, ImageTk

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class TimetableReminder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Timetable Reminder - AI Enhanced")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Set custom icon
        self.set_app_icon()
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.tts_engine = None
        self.tts_method = 'espeak'
        self.setup_voice()
        self.tts_enabled = tk.BooleanVar(value=False)
        self.schedule_data = {}
        self.current_image_path = None
        self.reminders_active = False
        self.scheduler_thread = None
        self.ocr_engine = None
        self.upsampler = None
        self.current_day = datetime.now().strftime('%A')
        
        # Setup persistent storage
        self.setup_persistent_storage()
        self.check_notification_system()
        
        # Setup auto-start on first run
        self.setup_autostart()
        
        self.setup_gui()
        
        # Auto-load previous timetable data
        self.auto_load_schedule()
        
        # Setup auto-save on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Ensure dock icon is properly set up
        self.root.after(2000, self.ensure_dock_integration)
        
        threading.Thread(target=self.initialize_models, daemon=True).start()
    
    def ensure_dock_integration(self):
        """Ensure proper dock integration after app is fully loaded"""
        try:
            # Set window manager hints for better dock integration
            self.root.wm_class("TimetableReminder", "TimetableReminder")
            
            # Update desktop files if needed to fix icon paths
            if hasattr(self, 'icon_file_path') and self.icon_file_path:
                self.update_desktop_files()
                
        except Exception as e:
            print(f"⚠️  Dock integration setup: {e}")
    
    def update_desktop_files(self):
        """Update existing desktop files with correct icon paths"""
        try:
            desktop_files = [
                os.path.expanduser("~/.config/autostart/TimetableReminder.desktop"),
                os.path.expanduser("~/.local/share/applications/TimetableReminder.desktop")
            ]
            
            for desktop_file in desktop_files:
                if os.path.exists(desktop_file):
                    # Read current content
                    with open(desktop_file, 'r') as f:
                        content = f.read()
                    
                    # Update icon line if using system icon path
                    system_icon_path = os.path.expanduser("~/.local/share/pixmaps/timetable-reminder.png")
                    if os.path.exists(system_icon_path):
                        # Replace icon line
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if line.startswith('Icon='):
                                lines[i] = f"Icon=timetable-reminder"
                                break
                        
                        # Write back
                        with open(desktop_file, 'w') as f:
                            f.write('\n'.join(lines))
                        
                        print(f"✅ Updated desktop file: {desktop_file}")
                        
        except Exception as e:
            print(f"⚠️  Could not update desktop files: {e}")
    
    def set_app_icon(self):
        """Set custom app icon from ttApp.png"""
        try:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(script_dir, "ttApp.png")
            
            if os.path.exists(icon_path):
                # Load and set the icon
                icon_image = Image.open(icon_path)
                # Create multiple sizes for better dock/taskbar display
                icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)]
                self.app_icons = []
                
                for size in icon_sizes:
                    resized_icon = icon_image.resize(size, Image.Resampling.LANCZOS)
                    tk_icon = ImageTk.PhotoImage(resized_icon)
                    self.app_icons.append(tk_icon)
                
                # Set the main icon (largest one for best quality)
                self.app_icon = self.app_icons[-1]  # 128x128
                self.root.iconphoto(True, *self.app_icons)  # All sizes for best compatibility
                
                # Also set as window manager class for better dock integration
                self.root.wm_class("TimetableReminder", "TimetableReminder")
                
                # Store icon path for desktop file creation
                self.icon_file_path = icon_path
                
                print(f"✅ Custom icon loaded from: {icon_path}")
            else:
                print(f"⚠️  Icon file not found at: {icon_path}")
                print("Place 'ttApp.png' in the same directory as this script")
                # Fallback to default icon
                self.create_default_icon()
        except Exception as e:
            print(f"⚠️  Error loading custom icon: {e}")
            self.create_default_icon()
    
    def create_default_icon(self):
        """Create a simple default icon if custom icon fails"""
        try:
            # Create a simple colored icon as fallback
            icon_img = Image.new('RGB', (64, 64), color='#e74c3c')
            self.app_icon = ImageTk.PhotoImage(icon_img)
            self.root.iconphoto(True, self.app_icon)
            self.root.wm_class("TimetableReminder", "TimetableReminder")
            self.icon_file_path = None
        except:
            self.icon_file_path = None
    
    def setup_autostart(self):
        """Setup auto-start functionality for Ubuntu"""
        try:
            # Check if we're on a Linux system
            if platform.system() != 'Linux':
                print("⚠️  Auto-start setup is designed for Ubuntu/Linux systems")
                return
            
            # Get the current script path
            script_path = os.path.abspath(__file__)
            script_name = os.path.basename(__file__)
            app_name = "TimetableReminder"
            
            # Create autostart directory if it doesn't exist
            autostart_dir = os.path.expanduser("~/.config/autostart")
            os.makedirs(autostart_dir, exist_ok=True)
            
            # Desktop entry file path
            desktop_file = os.path.join(autostart_dir, f"{app_name}.desktop")
            
            # Check if autostart is already configured
            if os.path.exists(desktop_file):
                print("✅ Auto-start already configured")
                # Still try to install icon for dock integration
                if hasattr(self, 'icon_file_path') and self.icon_file_path and os.path.exists(self.icon_file_path):
                    self.install_icon_to_system()
                return
            
            # Install icon to system first for proper dock integration
            if hasattr(self, 'icon_file_path') and self.icon_file_path and os.path.exists(self.icon_file_path):
                self.install_icon_to_system()
            
            # Determine icon path - prefer system installed icon
            icon_name = "timetable-reminder"  # Standard name for system icon
            
            # Create desktop entry for autostart with proper dock integration
            desktop_content = f"""[Desktop Entry]
Type=Application
Name=Timetable Reminder
GenericName=AI-Enhanced Timetable Reminder
Comment=Smart Timetable Reminder with AI-powered OCR and voice alerts
Exec=python3 "{script_path}"
Icon={icon_name}
Terminal=false
NoDisplay=false
StartupNotify=true
Categories=Utility;Office;Education;Calendar;
Keywords=timetable;reminder;schedule;calendar;AI;OCR;
StartupWMClass=TimetableReminder
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
Hidden=false
MimeType=text/csv;application/vnd.ms-excel;application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;
"""
            
            # Write the desktop file
            with open(desktop_file, 'w') as f:
                f.write(desktop_content)
            
            # Make it executable
            os.chmod(desktop_file, 0o755)
            
            print(f"✅ Auto-start configured! App will start automatically on boot")
            print(f"Desktop file created: {desktop_file}")
            
            # Also create a .desktop file for applications menu with same content
            apps_dir = os.path.expanduser("~/.local/share/applications")
            os.makedirs(apps_dir, exist_ok=True)
            apps_desktop_file = os.path.join(apps_dir, f"{app_name}.desktop")
            
            with open(apps_desktop_file, 'w') as f:
                f.write(desktop_content)
            os.chmod(apps_desktop_file, 0o755)
            
            print(f"✅ Application menu entry created: {apps_desktop_file}")
            
        except Exception as e:
            print(f"⚠️  Could not setup auto-start: {e}")
            print("You can manually add this app to startup applications in System Settings")
    
    def install_icon_to_system(self):
        """Install icon to system directories for better dock integration"""
        try:
            if not hasattr(self, 'icon_file_path') or not self.icon_file_path:
                return
                
            # Standard icon directories
            icon_dirs = [
                os.path.expanduser("~/.local/share/icons/hicolor/48x48/apps/"),
                os.path.expanduser("~/.local/share/icons/hicolor/64x64/apps/"),
                os.path.expanduser("~/.local/share/icons/hicolor/128x128/apps/"),
                os.path.expanduser("~/.local/share/pixmaps/")
            ]
            
            # Create directories and copy icon
            for icon_dir in icon_dirs:
                try:
                    os.makedirs(icon_dir, exist_ok=True)
                    import shutil
                    
                    if "48x48" in icon_dir:
                        # Create 48x48 version
                        icon_img = Image.open(self.icon_file_path)
                        icon_img = icon_img.resize((48, 48), Image.Resampling.LANCZOS)
                        target_path = os.path.join(icon_dir, "timetable-reminder.png")
                        icon_img.save(target_path)
                    elif "64x64" in icon_dir:
                        # Create 64x64 version
                        icon_img = Image.open(self.icon_file_path)
                        icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
                        target_path = os.path.join(icon_dir, "timetable-reminder.png")
                        icon_img.save(target_path)
                    elif "128x128" in icon_dir:
                        # Create 128x128 version
                        icon_img = Image.open(self.icon_file_path)
                        icon_img = icon_img.resize((128, 128), Image.Resampling.LANCZOS)
                        target_path = os.path.join(icon_dir, "timetable-reminder.png")
                        icon_img.save(target_path)
                    else:
                        # Copy original to pixmaps
                        target_path = os.path.join(icon_dir, "timetable-reminder.png")
                        shutil.copy2(self.icon_file_path, target_path)
                    
                    print(f"✅ Icon installed to: {target_path}")
                except Exception as e:
                    print(f"⚠️  Could not install icon to {icon_dir}: {e}")
            
            # Update icon cache
            try:
                subprocess.run(['gtk-update-icon-cache', '-f', '-t', 
                               os.path.expanduser("~/.local/share/icons/hicolor/")], 
                              capture_output=True)
                print("✅ Icon cache updated")
            except:
                pass
                
        except Exception as e:
            print(f"⚠️  Error installing system icon: {e}")
    
    def setup_persistent_storage(self):
        """Setup persistent storage for timetable data"""
        try:
            # Create config directory if it doesn't exist
            self.config_dir = os.path.expanduser("~/.config/timetable_reminder")
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Define paths for persistent storage
            self.auto_save_file = os.path.join(self.config_dir, "timetable_data.json")
            self.settings_file = os.path.join(self.config_dir, "app_settings.json")
            self.backup_file = os.path.join(self.config_dir, "timetable_backup.json")
            
            print(f"✅ Persistent storage setup at: {self.config_dir}")
            
        except Exception as e:
            print(f"⚠️  Could not setup persistent storage: {e}")
            # Fallback to current directory
            self.auto_save_file = "timetable_data.json"
            self.settings_file = "app_settings.json"
            self.backup_file = "timetable_backup.json"
    
    def auto_save_schedule(self):
        """Automatically save schedule data whenever it changes"""
        try:
            if self.schedule_data:
                # Create backup before saving
                if os.path.exists(self.auto_save_file):
                    import shutil
                    shutil.copy2(self.auto_save_file, self.backup_file)
                
                # Save current data
                save_data = {
                    'schedule_data': self.schedule_data,
                    'tts_enabled': self.tts_enabled.get(),
                    'last_saved': datetime.now().isoformat(),
                    'version': '2.0'
                }
                
                with open(self.auto_save_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                print(f"🔄 Auto-saved timetable data")
                
                # Update status to show auto-save
                try:
                    current_status = self.status_label.cget('text')
                    if not current_status.endswith('💾'):
                        self.root.after(0, lambda: self.status_label.config(
                            text=current_status + " 💾"))
                        # Remove the save indicator after 2 seconds
                        self.root.after(2000, self.remove_save_indicator)
                except:
                    pass
                
        except Exception as e:
            print(f"⚠️  Auto-save failed: {e}")
    
    def remove_save_indicator(self):
        """Remove the save indicator from status"""
        try:
            current_status = self.status_label.cget('text')
            if current_status.endswith(' 💾'):
                self.status_label.config(text=current_status[:-2])
        except:
            pass
    
    def auto_load_schedule(self):
        """Automatically load schedule data on startup"""
        try:
            if os.path.exists(self.auto_save_file):
                with open(self.auto_save_file, 'r') as f:
                    save_data = json.load(f)
                
                # Load schedule data
                if 'schedule_data' in save_data:
                    self.schedule_data = save_data['schedule_data']
                    
                    # Load settings
                    if 'tts_enabled' in save_data:
                        self.tts_enabled.set(save_data['tts_enabled'])
                    
                    # Display loaded schedule
                    self.display_schedule()
                    
                    # Enable controls if data exists
                    if self.schedule_data and any(self.schedule_data.values()):
                        self.root.after(1000, self.enable_controls_after_load)
                    
                    last_saved = save_data.get('last_saved', 'Unknown')
                    total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
                    
                    print(f"✅ Auto-loaded {total_tasks} tasks from previous session")
                    self.root.after(2000, lambda: self.status_label.config(
                        text=f"✅ Loaded {total_tasks} tasks from previous session"))
                    
                    return True
                    
        except Exception as e:
            print(f"⚠️  Auto-load failed: {e}")
            # Try to load from backup
            try:
                if os.path.exists(self.backup_file):
                    with open(self.backup_file, 'r') as f:
                        backup_data = json.load(f)
                    if 'schedule_data' in backup_data:
                        self.schedule_data = backup_data['schedule_data']
                        self.display_schedule()
                        print("✅ Restored from backup file")
                        self.root.after(2000, lambda: self.status_label.config(
                            text="✅ Restored schedule from backup"))
                        return True
            except:
                pass
        
        return False
    
    def enable_controls_after_load(self):
        """Enable controls after successful auto-load"""
        self.start_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
    
    def on_closing(self):
        """Handle application closing - save data and cleanup"""
        try:
            # Auto-save current data
            self.auto_save_schedule()
            
            # Stop reminders gracefully
            if self.reminders_active:
                self.stop_reminders()
            
            print("✅ Application closed gracefully - data saved")
            
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
        finally:
            self.root.destroy()
    
    def remove_autostart(self):
        """Remove auto-start functionality (useful for uninstall)"""
        try:
            autostart_file = os.path.expanduser("~/.config/autostart/TimetableReminder.desktop")
            apps_file = os.path.expanduser("~/.local/share/applications/TimetableReminder.desktop")
            
            for file_path in [autostart_file, apps_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✅ Removed: {file_path}")
            
            print("✅ Auto-start removed successfully")
        except Exception as e:
            print(f"⚠️  Error removing auto-start: {e}")
    
    def check_notification_system(self):
        try:
            result = subprocess.run(['which', 'notify-send'], capture_output=True, text=True)
            if result.returncode != 0:
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libnotify-bin'])
        except Exception as e:
            print(f"Notification system check: {e}")
    
    def setup_voice(self):
        try:
            result = subprocess.run(['which', 'espeak-ng'], capture_output=True)
            if result.returncode == 0:
                self.tts_method = 'espeak-ng'
                return
        except:
            pass
        try:
            result = subprocess.run(['which', 'festival'], capture_output=True)
            if result.returncode == 0:
                self.tts_method = 'festival'
                return
        except:
            pass
        if GTTS_AVAILABLE:
            try:
                pygame.mixer.init()
                self.tts_method = 'gtts'
                return
            except:
                pass
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                best_voice = None
                for voice in voices:
                    if any(name in voice.name.lower() for name in ['female', 'zira', 'hazel', 'susan']):
                        best_voice = voice.id
                        break
                    elif any(name in voice.name.lower() for name in ['enhanced', 'premium', 'neural']):
                        best_voice = voice.id
                if best_voice:
                    self.tts_engine.setProperty('voice', best_voice)
                self.tts_engine.setProperty('rate', 160)
                self.tts_engine.setProperty('volume', 0.9)
                self.tts_engine.setProperty('pitch', 1.0)
                self.tts_method = 'pyttsx3'
                return
            except Exception as e:
                print(f"pyttsx3 setup error: {e}")
        self.tts_method = 'espeak'
        self.suggest_better_tts()
    
    def suggest_better_tts(self):
        print("\n💡 For better voice quality, you can install:")
        print("1. espeak-ng: sudo apt-get install espeak-ng")
        print("2. Festival: sudo apt-get install festival")
        print("3. Google TTS: pip install gtts pygame")
        print("4. MaryTTS or Mimic for even better quality\n")
    
    def speak_reminder(self, task):
        if not self.tts_enabled.get():
            return
        try:
            clean_task = re.sub(r'[^\w\s.,!?-]', '', task)
            speech_text = f"Reminder: {clean_task}"
            if self.tts_method == 'gtts' and GTTS_AVAILABLE:
                try:
                    tts = gTTS(text=speech_text, lang='en', slow=False)
                    temp_file = "/tmp/reminder_speech.mp3"
                    tts.save(temp_file)
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    os.remove(temp_file)
                except Exception as e:
                    print(f"gTTS failed: {e}, falling back")
                    self.speak_with_fallback(speech_text)
            elif self.tts_method == 'espeak-ng':
                subprocess.run([
                    'espeak-ng',
                    '-v', 'en+f3',
                    '-s', '150',
                    '-p', '50',
                    '-a', '150',
                    speech_text
                ])
            elif self.tts_method == 'festival':
                process = subprocess.Popen(['festival', '--tts'], stdin=subprocess.PIPE)
                process.communicate(input=speech_text.encode())
            elif self.tts_method == 'espeak':
                subprocess.run([
                    'espeak',
                    '-v', 'en+f4',
                    '-s', '140',
                    '-p', '60',
                    '-a', '150',
                    '-g', '10',
                    speech_text
                ])
            elif self.tts_method == 'pyttsx3' and self.tts_engine:
                self.tts_engine.say(speech_text)
                self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    
    def speak_with_fallback(self, text):
        try:
            subprocess.run(['espeak', text])
        except:
            pass
    
    def initialize_models(self):
        try:
            if PADDLE_AVAILABLE:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=torch.cuda.is_available() if 'torch' in sys.modules else False,
                    show_log=False
                )
                self.root.after(0, lambda: self.status_label.config(
                    text="✅ Advanced OCR models loaded"))
            if ESRGAN_AVAILABLE and torch.cuda.is_available():
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                self.upsampler = RealESRGANer(
                    scale=4,
                    model_path='RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True
                )
        except Exception as e:
            print(f"Model initialization: {e}")
    
    def setup_gui(self):
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        title_font = font.Font(family="Arial", size=20, weight="bold")
        title_label = tk.Label(header_frame, text="📅 Smart Timetable Reminder - AI Enhanced", 
                              font=title_font, fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        self.ai_status = tk.Label(header_frame, text="🤖 AI Ready" if PADDLE_AVAILABLE else "⚠️ Basic Mode", 
                                 font=('Arial', 10), fg='#ecf0f1', bg='#2c3e50')
        self.ai_status.place(x=10, y=45)
        voice_status = f"🔊 Voice: {self.tts_method}"
        voice_label = tk.Label(header_frame, text=voice_status, 
                              font=('Arial', 10), fg='#ecf0f1', bg='#2c3e50')
        voice_label.place(x=150, y=45)
        current_day_label = tk.Label(header_frame, text=f"Today: {self.current_day}", 
                                   font=('Arial', 11), fg='#ecf0f1', bg='#2c3e50')
        current_day_label.place(x=10, y=10)
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        left_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=1, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        input_tab = tk.Frame(notebook, bg='white')
        notebook.add(input_tab, text="Input Methods")
        tk.Button(input_tab, text="✏️ Manual Entry (Grid)", 
                 command=self.open_manual_entry, bg='#27ae60', fg='white',
                 font=('Arial', 12, 'bold'), padx=20, pady=12, cursor='hand2').pack(pady=15)
        tk.Button(input_tab, text="📊 Import CSV/Excel", 
                 command=self.import_csv, bg='#3498db', fg='white',
                 font=('Arial', 11), padx=15, pady=10, cursor='hand2').pack(pady=10)
        ocr_btn_text = "🤖 AI-Powered OCR" if PADDLE_AVAILABLE else "📷 Basic OCR"
        ocr_btn_color = "#9b59b6" if PADDLE_AVAILABLE else "#95a5a6"
        tk.Button(input_tab, text=ocr_btn_text, 
                 command=self.upload_image_advanced, bg=ocr_btn_color, fg='white',
                 font=('Arial', 11, 'bold'), padx=15, pady=10, cursor='hand2').pack(pady=10)
        ocr_settings = tk.LabelFrame(input_tab, text="OCR Settings", bg='white', 
                                    font=('Arial', 10, 'bold'))
        ocr_settings.pack(pady=10, padx=10, fill=tk.X)
        self.enhance_image = tk.BooleanVar(value=True)
        tk.Checkbutton(ocr_settings, text="🔍 Enhance image quality", 
                      variable=self.enhance_image, bg='white',
                      font=('Arial', 9)).pack(anchor=tk.W, padx=10, pady=2)
        self.auto_detect_layout = tk.BooleanVar(value=True)
        tk.Checkbutton(ocr_settings, text="📐 Auto-detect table layout", 
                      variable=self.auto_detect_layout, bg='white',
                      font=('Arial', 9)).pack(anchor=tk.W, padx=10, pady=2)
        control_tab = tk.Frame(notebook, bg='white')
        notebook.add(control_tab, text="Controls")
        voice_frame = tk.LabelFrame(control_tab, text="Voice Settings", bg='white',
                                   font=('Arial', 10, 'bold'))
        voice_frame.pack(pady=10, padx=10, fill=tk.X)
        tk.Checkbutton(voice_frame, text="🔊 Enable Voice Alerts", 
                      variable=self.tts_enabled, bg='white',
                      font=('Arial', 11), cursor='hand2',
                      command=self.on_tts_setting_change).pack(pady=5)
        tk.Label(voice_frame, text=f"Using: {self.tts_method}", 
                bg='white', font=('Arial', 9), fg='#666').pack()
        self.start_btn = tk.Button(control_tab, text="▶️ Start Reminders", 
                                  command=self.toggle_reminders, bg='#e74c3c', fg='white',
                                  font=('Arial', 12, 'bold'), padx=20, pady=10, cursor='hand2',
                                  state=tk.DISABLED)
        self.start_btn.pack(pady=10)
        test_frame = tk.Frame(control_tab, bg='white')
        test_frame.pack(pady=10)
        tk.Button(test_frame, text="🔔 Test Notification", 
                 command=self.test_notification, bg='#9b59b6', fg='white',
                 font=('Arial', 10), padx=10, pady=8, cursor='hand2').pack(side=tk.LEFT, padx=3)
        tk.Button(test_frame, text="🔊 Test Voice", 
                 command=lambda: self.speak_reminder("This is a test of the voice alert system"),
                 bg='#16a085', fg='white',
                 font=('Arial', 10), padx=10, pady=8, cursor='hand2').pack(side=tk.LEFT, padx=3)
        save_load_frame = tk.Frame(control_tab, bg='white')
        save_load_frame.pack(pady=15)
        self.save_btn = tk.Button(save_load_frame, text="💾 Save", 
                                 command=self.save_schedule, bg='#16a085', fg='white',
                                 font=('Arial', 10), padx=15, pady=5, cursor='hand2',
                                 state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=3)
        tk.Button(save_load_frame, text="📂 Load", 
                 command=self.load_schedule, bg='#16a085', fg='white',
                 font=('Arial', 10), padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=3)
        
        # Add auto-start controls
        autostart_frame = tk.LabelFrame(control_tab, text="Auto-Start Settings", bg='white',
                                       font=('Arial', 10, 'bold'))
        autostart_frame.pack(pady=10, padx=10, fill=tk.X)
        
        autostart_status = "✅ Enabled" if os.path.exists(os.path.expanduser("~/.config/autostart/TimetableReminder.desktop")) else "❌ Disabled"
        tk.Label(autostart_frame, text=f"Auto-start on boot: {autostart_status}", 
                bg='white', font=('Arial', 9)).pack(pady=2)
        
        autostart_btn_frame = tk.Frame(autostart_frame, bg='white')
        autostart_btn_frame.pack(pady=5)
        
        tk.Button(autostart_btn_frame, text="🚀 Enable Auto-start", 
                 command=self.setup_autostart, bg='#27ae60', fg='white',
                 font=('Arial', 9), padx=10, pady=3, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        tk.Button(autostart_btn_frame, text="🛑 Disable Auto-start", 
                 command=self.remove_autostart, bg='#e74c3c', fg='white',
                 font=('Arial', 9), padx=10, pady=3, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        status_frame = tk.Frame(left_panel, bg='#ecf0f1', padx=20, pady=15)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_frame, text="⏳ Ready to input timetable", 
                                    bg='#ecf0f1', fg='#2c3e50', font=('Arial', 10), 
                                    wraplength=300)
        self.status_label.pack()
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        schedule_header = tk.Frame(right_panel, bg='#34495e', height=40)
        schedule_header.pack(fill=tk.X)
        schedule_header.pack_propagate(False)
        schedule_title = tk.Label(schedule_header, text="📋 Your Schedule", 
                                 font=('Arial', 14, 'bold'), fg='white', bg='#34495e')
        schedule_title.pack(pady=8)
        schedule_container = tk.Frame(right_panel, bg='white')
        schedule_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_frame = tk.Frame(schedule_container)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.schedule_text = tk.Text(text_frame, font=('Consolas', 11), wrap=tk.WORD,
                                    bg='#fafafa', fg='#2c3e50')
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", 
                                 command=self.schedule_text.yview)
        self.schedule_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_text.config(yscrollcommand=scrollbar.set)
        self.schedule_text.tag_config("day", font=('Arial', 13, 'bold'), foreground='#2c3e50')
        self.schedule_text.tag_config("current_day", font=('Arial', 13, 'bold'), 
                                     foreground='white', background='#e74c3c')
        self.schedule_text.tag_config("time", font=('Consolas', 11), foreground='#e74c3c')
        self.schedule_text.tag_config("task", font=('Arial', 11), foreground='#34495e')
        self.schedule_text.tag_config("header", font=('Arial', 16, 'bold'), foreground='#2c3e50')
        self.display_welcome_message()
    
    def display_welcome_message(self):
        self.schedule_text.delete(1.0, tk.END)
        self.schedule_text.insert(tk.END, "Welcome to AI-Enhanced Timetable Reminder!\n\n", "header")
        if PADDLE_AVAILABLE:
            self.schedule_text.insert(tk.END, "🤖 AI Features Available:\n", "day")
            self.schedule_text.insert(tk.END, "• Advanced table detection with PaddleOCR\n", "task")
            self.schedule_text.insert(tk.END, "• Multi-language support\n", "task")
            self.schedule_text.insert(tk.END, "• Automatic layout understanding\n\n", "task")
        self.schedule_text.insert(tk.END, "🔊 Voice System:\n", "day")
        self.schedule_text.insert(tk.END, f"• Using {self.tts_method} for voice alerts\n", "task")
        self.schedule_text.insert(tk.END, "• Multiple notification methods for Ubuntu\n\n", "task")
        self.schedule_text.insert(tk.END, "🚀 Auto-Start & Persistence:\n", "day")
        self.schedule_text.insert(tk.END, "• App starts automatically on boot\n", "task")
        self.schedule_text.insert(tk.END, "• Timetable data auto-saves and persists\n", "task")
        self.schedule_text.insert(tk.END, "• Custom icon support (ttApp.png)\n\n", "task")
        self.schedule_text.insert(tk.END, "💾 Data Storage:\n", "day")
        self.schedule_text.insert(tk.END, "• Auto-saves after every change\n", "task")
        self.schedule_text.insert(tk.END, "• Auto-loads on app startup\n", "task")
        self.schedule_text.insert(tk.END, "• No need to manually save/load\n\n", "task")
        self.schedule_text.insert(tk.END, "Choose an input method:\n\n", "day")
        self.schedule_text.insert(tk.END, "1. ✏️ Manual Entry\n", "task")
        self.schedule_text.insert(tk.END, "   Easy grid interface for quick entry\n\n", "task")
        self.schedule_text.insert(tk.END, "2. 📊 Import CSV/Excel\n", "task")
        self.schedule_text.insert(tk.END, "   Import from spreadsheet files\n\n", "task")
        self.schedule_text.insert(tk.END, "3. 🤖 AI-Powered OCR\n", "task")
        self.schedule_text.insert(tk.END, "   Extract from images with AI accuracy\n\n", "task")
        self.schedule_text.insert(tk.END, f"📍 Today is {self.current_day}\n", "current_day")
    
    def upload_image_advanced(self):
        if not PADDLE_AVAILABLE and not TESSERACT_AVAILABLE:
            messagebox.showerror("OCR Not Available", 
                               "No OCR engine available. Please install:\n\n" +
                               "For AI OCR: pip install paddlepaddle paddleocr\n" +
                               "For basic OCR: sudo apt-get install tesseract-ocr\n" +
                               "                pip install pytesseract")
            return
        if not PADDLE_AVAILABLE:
            response = messagebox.askyesno("Basic OCR Mode", 
                                         "Advanced OCR not available. Use basic mode?\n\n" +
                                         "For advanced features, install:\n" +
                                         "pip install paddlepaddle paddleocr")
            if not response:
                return
        file_path = filedialog.askopenfilename(
            title="Select Timetable Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text="🔄 Processing with AI... Please wait.")
            self.progress.pack(pady=5, fill=tk.X)
            self.progress.start()
            threading.Thread(target=self.process_image_advanced, args=(file_path,)).start()
    
    def enhance_image_quality(self, img_path):
        img = cv2.imread(img_path)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        enhanced = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        height, width = denoised.shape[:2]
        if width < 1500:
            scale = 1500 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            denoised = cv2.resize(denoised, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
        return denoised
    
    def process_image_advanced(self, image_path):
        try:
            if self.enhance_image.get():
                enhanced_img = self.enhance_image_quality(image_path)
                temp_path = "enhanced_timetable.png"
                cv2.imwrite(temp_path, enhanced_img)
                process_path = temp_path
            else:
                process_path = image_path
            if PADDLE_AVAILABLE and self.ocr_engine:
                result = self.ocr_engine.ocr(process_path, cls=True)
                self.schedule_data = self.extract_table_from_paddle(result)
            elif TESSERACT_AVAILABLE:
                img = cv2.imread(process_path)
                text = pytesseract.image_to_string(img)
                self.schedule_data = self.extract_basic_schedule(text)
            else:
                raise Exception("No OCR engine available")
            if os.path.exists("enhanced_timetable.png"):
                os.remove("enhanced_timetable.png")
            self.root.after(0, self.ocr_complete)
        except Exception as e:
            self.root.after(0, lambda: self.ocr_error(str(e)))
    
    def extract_table_from_paddle(self, ocr_result):
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            schedule[day] = []
        if not ocr_result or not ocr_result[0]:
            return schedule
        text_boxes = []
        for line in ocr_result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.8:
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                text_boxes.append({
                    'text': text,
                    'x': x_center,
                    'y': y_center,
                    'bbox': bbox
                })
        text_boxes.sort(key=lambda b: (b['y'], b['x']))
        rows = []
        current_row = []
        last_y = -1
        row_threshold = 30
        for box in text_boxes:
            if last_y == -1 or abs(box['y'] - last_y) < row_threshold:
                current_row.append(box)
                last_y = box['y']
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b['x']))
                current_row = [box]
                last_y = box['y']
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['x']))
        header_row_idx = -1
        day_columns = {}
        for idx, row in enumerate(rows[:5]):
            found_days = 0
            for cell_idx, cell in enumerate(row):
                text_lower = cell['text'].lower()
                for day in days:
                    if day.lower() in text_lower or day[:3].lower() in text_lower:
                        day_columns[cell_idx] = day
                        found_days += 1
                        break
            if found_days >= 3:
                header_row_idx = idx
                break
        for row_idx in range(header_row_idx + 1, len(rows)):
            row = rows[row_idx]
            if row and self.is_time_text(row[0]['text']):
                time_str = self.normalize_time(row[0]['text'])
                for col_idx, day in day_columns.items():
                    if col_idx < len(row):
                        task = row[col_idx]['text'].strip()
                        if task and len(task) > 2 and not self.is_time_text(task):
                            schedule[day].append({
                                'time': time_str,
                                'task': task
                            })
        return schedule
    
    def is_time_text(self, text):
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}\s*-\s*\d{1,2}',
            r'\d{1,2}\s*(?:am|pm|AM|PM)'
        ]
        for pattern in time_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def extract_basic_schedule(self, text):
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            schedule[day] = []
        lines = text.split('\n')
        current_day = None
        for line in lines:
            line = line.strip()
            for day in days:
                if day.lower() in line.lower():
                    current_day = day
                    break
            time_match = re.search(r'(\d{1,2}:\d{2})', line)
            if time_match and current_day:
                time_str = time_match.group(1)
                task = line.replace(time_str, '').strip(' -:')
                if task and len(task) > 2:
                    schedule[current_day].append({
                        'time': self.normalize_time(time_str),
                        'task': task[:50]
                    })
        return schedule
    
    def ocr_complete(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.display_schedule()
        total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
        if total_tasks > 0:
            # Auto-save the extracted data
            self.auto_save_schedule()
            
            self.status_label.config(text=f"✅ Extracted {total_tasks} tasks! Review and edit as needed.")
            self.start_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            if total_tasks < 20:
                response = messagebox.askyesno("Review Results", 
                                             f"Extracted {total_tasks} tasks.\n\n" +
                                             "Would you like to review and edit?")
                if response:
                    self.open_edit_window()
        else:
            self.status_label.config(text="⚠️ No tasks extracted. Try manual entry.")
            messagebox.showwarning("No Data", 
                                 "Could not extract schedule from image.\n\n" +
                                 "Try:\n" +
                                 "• Better image quality\n" +
                                 "• Manual entry\n" +
                                 "• CSV import")
    
    def ocr_error(self, error_msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="❌ OCR failed. Try manual entry.")
        messagebox.showerror("OCR Error", f"Failed to process image:\n{error_msg}")
    
    def open_edit_window(self):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Schedule")
        edit_window.geometry("800x600")
        tk.Label(edit_window, text="Review and edit extracted schedule", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        notebook = ttk.Notebook(edit_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_frames = {}
        for day in days:
            frame = tk.Frame(notebook)
            notebook.add(frame, text=day)
            day_frames[day] = frame
            list_frame = tk.Frame(frame)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            listbox = tk.Listbox(list_frame, font=('Consolas', 10))
            scrollbar = tk.Scrollbar(list_frame)
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            if day in self.schedule_data:
                for task in sorted(self.schedule_data[day], key=lambda x: x['time']):
                    listbox.insert(tk.END, f"{task['time']} - {task['task']}")
            control_frame = tk.Frame(frame)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            tk.Label(control_frame, text="Time:").grid(row=0, column=0, padx=5)
            time_entry = tk.Entry(control_frame, width=10)
            time_entry.grid(row=0, column=1, padx=5)
            tk.Label(control_frame, text="Task:").grid(row=0, column=2, padx=5)
            task_entry = tk.Entry(control_frame, width=30)
            task_entry.grid(row=0, column=3, padx=5)
            def add_task(d=day, lb=listbox, te=time_entry, tke=task_entry):
                time = te.get()
                task = tke.get()
                if time and task:
                    if d not in self.schedule_data:
                        self.schedule_data[d] = []
                    self.schedule_data[d].append({
                        'time': self.normalize_time(time),
                        'task': task
                    })
                    lb.insert(tk.END, f"{self.normalize_time(time)} - {task}")
                    te.delete(0, tk.END)
                    tke.delete(0, tk.END)
            tk.Button(control_frame, text="Add", command=add_task,
                     bg='#27ae60', fg='white').grid(row=0, column=4, padx=5)
        def save_edits():
            self.display_schedule()
            # Auto-save after editing
            self.auto_save_schedule()
            edit_window.destroy()
        tk.Button(edit_window, text="💾 Save Changes", command=save_edits,
                 bg='#3498db', fg='white', font=('Arial', 11),
                 padx=20, pady=10).pack(pady=10)
    
    def open_manual_entry(self):
        entry_window = tk.Toplevel(self.root)
        entry_window.title("Manual Schedule Entry")
        entry_window.geometry("1200x700")
        inst_frame = tk.Frame(entry_window, bg='#3498db', height=50)
        inst_frame.pack(fill=tk.X)
        inst_frame.pack_propagate(False)
        tk.Label(inst_frame, text="Enter your schedule in the grid below", 
                font=('Arial', 14, 'bold'), fg='white', bg='#3498db').pack(pady=10)
        main_frame = tk.Frame(entry_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(main_frame)
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        days = ['Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_slots = [
            "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00",
            "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00",
            "20:00", "21:00", "22:00", "23:00"
        ]
        for col, day in enumerate(days):
            label = tk.Label(scrollable_frame, text=day, font=('Arial', 11, 'bold'),
                           bg='#34495e', fg='white', relief=tk.RAISED, bd=1,
                           width=15, padx=5, pady=8)
            label.grid(row=0, column=col, sticky='ew')
        entry_widgets = {}
        for row, time in enumerate(time_slots, start=1):
            time_label = tk.Label(scrollable_frame, text=time, font=('Arial', 10, 'bold'),
                                bg='#ecf0f1', relief=tk.RAISED, bd=1, width=15, pady=5)
            time_label.grid(row=row, column=0, sticky='ew')
            for col in range(1, 8):
                entry = tk.Entry(scrollable_frame, font=('Arial', 10), width=20)
                entry.grid(row=row, column=col, sticky='ew', padx=1, pady=1)
                entry_widgets[(row-1, col-1)] = entry
        def add_time_row():
            new_row = len(scrollable_frame.grid_slaves()) // 8
            time_entry = tk.Entry(scrollable_frame, font=('Arial', 10, 'bold'),
                                bg='#ecf0f1', width=15)
            time_entry.grid(row=new_row, column=0, sticky='ew')
            for col in range(1, 8):
                entry = tk.Entry(scrollable_frame, font=('Arial', 10), width=20)
                entry.grid(row=new_row, column=col, sticky='ew', padx=1, pady=1)
                entry_widgets[(new_row-1, col-1)] = entry
        canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        btn_frame = tk.Frame(entry_window)
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Button(btn_frame, text="➕ Add Time Slot", command=add_time_row,
                 bg='#3498db', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        def save_manual_entry():
            self.schedule_data = {}
            days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in days_list:
                self.schedule_data[day] = []
            time_entries = []
            for widget in scrollable_frame.grid_slaves():
                if widget.grid_info()['column'] == 0 and widget.grid_info()['row'] > 0:
                    if isinstance(widget, tk.Entry):
                        time_entries.append((widget.grid_info()['row'], widget.get()))
                    elif isinstance(widget, tk.Label):
                        time_entries.append((widget.grid_info()['row'], widget.cget('text')))
            time_entries.sort(key=lambda x: x[0])
            for row_idx, (row, time_text) in enumerate(time_entries):
                if time_text and ':' in time_text:
                    normalized_time = self.normalize_time(time_text)
                    for col, day in enumerate(days_list):
                        if (row_idx, col) in entry_widgets:
                            task = entry_widgets[(row_idx, col)].get().strip()
                            if task:
                                self.schedule_data[day].append({
                                    'time': normalized_time,
                                    'task': task
                                })
            self.display_schedule()
            if any(self.schedule_data.values()):
                # Auto-save the manually entered data
                self.auto_save_schedule()
                
                self.status_label.config(text="✅ Schedule entered successfully!")
                self.start_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.NORMAL)
                entry_window.destroy()
            else:
                messagebox.showwarning("No Data", "Please enter at least one task!")
        tk.Button(btn_frame, text="💾 Save Schedule", command=save_manual_entry,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
                 padx=20, pady=8).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="❌ Cancel", command=entry_window.destroy,
                 bg='#e74c3c', fg='white', font=('Arial', 10),
                 padx=15, pady=8).pack(side=tk.RIGHT, padx=5)
    
    def import_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select Schedule File",
            filetypes=[("Spreadsheet files", "*.csv *.xlsx *.xls"), 
                      ("CSV files", "*.csv"),
                      ("Excel files", "*.xlsx *.xls"),
                      ("All files", "*.*")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                self.schedule_data = {}
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                for day in days:
                    self.schedule_data[day] = []
                if 'Time' in df.columns or df.columns[0].lower() == 'time':
                    time_col = df.columns[0]
                    for idx, row in df.iterrows():
                        time_str = str(row[time_col])
                        if pd.notna(time_str) and ':' in time_str:
                            normalized_time = self.normalize_time(time_str)
                            for col in df.columns[1:]:
                                day_found = None
                                for day in days:
                                    if day.lower() in col.lower():
                                        day_found = day
                                        break
                                if day_found and pd.notna(row[col]):
                                    task = str(row[col]).strip()
                                    if task and task.lower() != 'nan':
                                        self.schedule_data[day_found].append({
                                            'time': normalized_time,
                                            'task': task
                                        })
                self.display_schedule()
                if any(self.schedule_data.values()):
                    # Auto-save the imported data
                    self.auto_save_schedule()
                    
                    total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
                    self.status_label.config(text=f"✅ Imported {total_tasks} tasks successfully!")
                    self.start_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                else:
                    self.status_label.config(text="⚠️ No data found. Check file format.")
                    messagebox.showinfo("Format Help", 
                                      "Expected format:\n" +
                                      "- First column: Time (e.g., 08:00)\n" +
                                      "- Other columns: Days (Monday, Tuesday, etc.)\n" +
                                      "- Cells: Task descriptions")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import file: {str(e)}")
                self.status_label.config(text="❌ Error importing file")
    
    def normalize_time(self, time_str):
        time_str = str(time_str).strip()
        if '-' in time_str:
            time_str = time_str.split('-')[0].strip()
        time_str = re.sub(r'[^\d:APMapm\s]', '', time_str)
        formats = ['%H:%M', '%I:%M %p', '%I:%M%p', '%H:%M:%S']
        for fmt in formats:
            try:
                time_obj = datetime.strptime(time_str.upper(), fmt)
                return time_obj.strftime('%H:%M')
            except:
                continue
        match = re.search(r'(\d{1,2}):(\d{2})', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 'pm' in time_str.lower() and hour < 12:
                hour += 12
            elif 'am' in time_str.lower() and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute:02d}"
        return "00:00"
    
    def display_schedule(self):
        self.schedule_text.delete(1.0, tk.END)
        total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
        if not self.schedule_data or total_tasks == 0:
            self.display_welcome_message()
            return
        self.schedule_text.insert(tk.END, "📅 Your Schedule\n", "header")
        self.schedule_text.insert(tk.END, f"Total: {total_tasks} tasks\n", "task")
        self.schedule_text.insert(tk.END, "─" * 60 + "\n\n")
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_idx = days_order.index(self.current_day) if self.current_day in days_order else 0
        ordered_days = days_order[current_idx:] + days_order[:current_idx]
        for day in ordered_days:
            if day in self.schedule_data:
                tasks = self.schedule_data[day]
                if day == self.current_day:
                    self.schedule_text.insert(tk.END, f" {day} (TODAY) ", "current_day")
                else:
                    self.schedule_text.insert(tk.END, f"{day}", "day")
                self.schedule_text.insert(tk.END, f" ({len(tasks)} tasks)\n", "task")
                if tasks:
                    sorted_tasks = sorted(tasks, key=lambda x: x['time'])
                    for task in sorted_tasks:
                        self.schedule_text.insert(tk.END, "  ")
                        self.schedule_text.insert(tk.END, f"{task['time']}", "time")
                        self.schedule_text.insert(tk.END, " - ")
                        self.schedule_text.insert(tk.END, f"{task['task']}\n", "task")
                else:
                    self.schedule_text.insert(tk.END, "  No tasks\n", "task")
                self.schedule_text.insert(tk.END, "\n")
    
    def toggle_reminders(self):
        if not self.reminders_active:
            self.start_reminders()
        else:
            self.stop_reminders()
    
    def start_reminders(self):
        self.reminders_active = True
        self.start_btn.config(text="⏸️ Stop Reminders", bg='#c0392b')
        self.status_label.config(text="🔔 Reminders active - monitoring schedule")
        self.setup_reminders()
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self.scheduler_thread.start()
    
    def stop_reminders(self):
        self.reminders_active = False
        self.start_btn.config(text="▶️ Start Reminders", bg='#e74c3c')
        self.status_label.config(text="⏹️ Reminders stopped")
        schedule.clear()
    
    def setup_reminders(self):
        schedule.clear()
        for day, tasks in self.schedule_data.items():
            for task in tasks:
                day_num = self.get_day_number(day)
                time_str = task['time']
                if day_num is not None and ':' in time_str:
                    self.schedule_task_reminder(day_num, time_str, task['task'])
    
    def get_day_number(self, day_name):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        try:
            return days.index(day_name)
        except ValueError:
            return None
    
    def schedule_task_reminder(self, day_num, time_str, task):
        def remind():
            if self.reminders_active:
                current_day = datetime.now().weekday()
                if current_day == day_num:
                    self.show_reminder(task, time_str)
        try:
            schedule.every().day.at(time_str).do(remind)
            time_obj = datetime.strptime(time_str, '%H:%M')
            advance_time = (time_obj - timedelta(minutes=5)).strftime('%H:%M')
            def advance_remind():
                if self.reminders_active:
                    current_day = datetime.now().weekday()
                    if current_day == day_num:
                        self.show_reminder(f"⏰ Upcoming in 5 minutes: {task}", time_str)
            schedule.every().day.at(advance_time).do(advance_remind)
        except Exception as e:
            print(f"Error scheduling reminder: {e}")
    
    def show_reminder(self, task, time_str):
        try:
            self.show_ubuntu_notification(task, time_str)
            self.root.after(0, lambda: self.status_label.config(
                text=f"🔔 Reminder shown: {task[:30]}..."))
            if self.tts_enabled.get():
                threading.Thread(target=self.speak_reminder, args=(task,), daemon=True).start()
        except Exception as e:
            print(f"Error showing reminder: {e}")
            self.show_popup_reminder(task, time_str)
    
    def show_ubuntu_notification(self, task, time_str):
        try:
            icon_path = self.get_icon_path()
            cmd = [
                'notify-send',
                '--urgency=critical',
                '--expire-time=10000',
                '--app-name=Timetable Reminder'
            ]
            if icon_path:
                cmd.extend(['--icon', icon_path])
            cmd.extend([
                f'📅 Reminder: {time_str}',
                task
            ])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                subprocess.run([
                    'notify-send',
                    '-u', 'critical',
                    '-t', '10000',
                    f'Reminder: {time_str}',
                    task
                ])
            self.show_zenity_notification(task, time_str)
        except FileNotFoundError:
            self.show_alternative_notification(task, time_str)
        except Exception as e:
            print(f"Ubuntu notification error: {e}")
            self.show_popup_reminder(task, time_str)
    
    def show_zenity_notification(self, task, time_str):
        try:
            result = subprocess.run(['which', 'zenity'], capture_output=True)
            if result.returncode == 0:
                subprocess.Popen([
                    'zenity',
                    '--info',
                    '--title=Timetable Reminder',
                    '--text=' + f'⏰ {time_str}\n\n{task}',
                    '--timeout=10',
                    '--width=300'
                ])
        except:
            pass
    
    def show_alternative_notification(self, task, time_str):
        try:
            subprocess.run([
                'gdbus', 'call', '--session',
                '--dest=org.freedesktop.Notifications',
                '--object-path=/org/freedesktop/Notifications',
                '--method=org.freedesktop.Notifications.Notify',
                'Timetable Reminder',
                '0',
                'appointment-reminder',
                f'Reminder: {time_str}',
                task,
                '[]',
                '{"urgency": <byte 2>}',
                '10000'
            ])
        except:
            try:
                subprocess.run([
                    'dunstify',
                    '-u', 'critical',
                    '-t', '10000',
                    f'Reminder: {time_str}',
                    task
                ])
            except:
                try:
                    notification.notify(
                        title=f"Reminder: {time_str}",
                        message=task,
                        app_name="Timetable Reminder",
                        timeout=10
                    )
                except:
                    self.show_popup_reminder(task, time_str)
    
    def get_icon_path(self):
        try:
            # First try to use our custom system-installed icon
            system_icon_paths = [
                os.path.expanduser("~/.local/share/pixmaps/timetable-reminder.png"),
                os.path.expanduser("~/.local/share/icons/hicolor/64x64/apps/timetable-reminder.png"),
                os.path.expanduser("~/.local/share/icons/hicolor/48x48/apps/timetable-reminder.png")
            ]
            
            for path in system_icon_paths:
                if os.path.exists(path):
                    return path
            
            # Then try our local custom icon
            if hasattr(self, 'icon_file_path') and self.icon_file_path and os.path.exists(self.icon_file_path):
                return self.icon_file_path
            
            # Fallback to system icons
            icon_paths = [
                '/usr/share/icons/gnome/48x48/apps/calendar.png',
                '/usr/share/icons/hicolor/48x48/apps/calendar.png',
                '/usr/share/pixmaps/calendar.png'
            ]
            for path in icon_paths:
                if os.path.exists(path):
                    return path
            return None
        except:
            return None
    
    def show_popup_reminder(self, task, time_str):
        try:
            reminder_window = tk.Toplevel(self.root)
            reminder_window.title(f"Reminder: {time_str}")
            reminder_window.geometry("400x250")
            reminder_window.configure(bg='#e74c3c')
            reminder_window.attributes('-topmost', True)
            reminder_window.lift()
            reminder_window.focus_force()
            reminder_window.update_idletasks()
            x = (reminder_window.winfo_screenwidth() // 2) - 200
            y = (reminder_window.winfo_screenheight() // 2) - 125
            reminder_window.geometry(f"+{x}+{y}")
            
            # Set icon for popup window too
            try:
                reminder_window.iconphoto(True, self.app_icon)
            except:
                pass
            
            self.flash_window(reminder_window)
            header_frame = tk.Frame(reminder_window, bg='#c0392b', height=60)
            header_frame.pack(fill=tk.X)
            header_frame.pack_propagate(False)
            tk.Label(header_frame, text="⏰ REMINDER", 
                    font=('Arial', 20, 'bold'), fg='white', bg='#c0392b').pack(pady=15)
            content_frame = tk.Frame(reminder_window, bg='#e74c3c')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            tk.Label(content_frame, text=time_str, 
                    font=('Arial', 16, 'bold'), fg='white', bg='#e74c3c').pack(pady=5)
            tk.Label(content_frame, text=task, 
                    font=('Arial', 12), fg='white', bg='#e74c3c', 
                    wraplength=350).pack(pady=10)
            btn_frame = tk.Frame(reminder_window, bg='#e74c3c')
            btn_frame.pack(fill=tk.X, pady=10)
            tk.Button(btn_frame, text="Snooze 5 min", 
                     command=lambda: self.snooze_reminder(reminder_window, task, time_str),
                     bg='#f39c12', fg='white', font=('Arial', 11, 'bold'),
                     padx=15, pady=5).pack(side=tk.LEFT, padx=20)
            tk.Button(btn_frame, text="Dismiss", 
                     command=reminder_window.destroy,
                     bg='white', fg='#e74c3c', font=('Arial', 11, 'bold'),
                     padx=20, pady=5).pack(side=tk.RIGHT, padx=20)
            reminder_window.after(30000, lambda: reminder_window.destroy() if reminder_window.winfo_exists() else None)
            self.play_notification_sound()
        except Exception as e:
            print(f"Popup reminder failed: {e}")
    
    def flash_window(self, window):
        def flash():
            current = window.attributes('-alpha')
            window.attributes('-alpha', 0.3 if current == 1.0 else 1.0)
        for i in range(6):
            window.after(i * 200, flash)
    
    def play_notification_sound(self):
        try:
            sound_files = [
                '/usr/share/sounds/freedesktop/stereo/message.oga',
                '/usr/share/sounds/freedesktop/stereo/complete.oga',
                '/usr/share/sounds/ubuntu/stereo/message.ogg'
            ]
            for sound in sound_files:
                if os.path.exists(sound):
                    subprocess.Popen(['paplay', sound])
                    break
        except:
            try:
                print('\a')
            except:
                pass
    
    def snooze_reminder(self, window, task, time_str):
        window.destroy()
        def snooze_alert():
            self.show_reminder(f"[Snoozed] {task}", time_str)
        threading.Timer(300, snooze_alert).start()
        self.status_label.config(text="😴 Reminder snoozed for 5 minutes")
    
    def test_notification(self):
        test_messages = [
            "This is a test notification to check if notifications are working properly on your Ubuntu system.",
            "Testing voice alerts and desktop notifications.",
            "If you see this, notifications are working!"
        ]
        import random
        test_msg = random.choice(test_messages)
        self.show_reminder(test_msg, datetime.now().strftime("%H:%M"))
        info_window = tk.Toplevel(self.root)
        info_window.title("Notification Test Info")
        info_window.geometry("400x300")
        info_text = tk.Text(info_window, wrap=tk.WORD, font=('Consolas', 10))
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        info_text.insert(tk.END, "🔔 Notification System Test\n\n")
        info_text.insert(tk.END, f"TTS Method: {self.tts_method}\n")
        info_text.insert(tk.END, f"TTS Enabled: {self.tts_enabled.get()}\n\n")
        info_text.insert(tk.END, "Available notification methods:\n")
        checks = [
            ('notify-send', 'which notify-send'),
            ('zenity', 'which zenity'),
            ('dunstify', 'which dunstify')
        ]
        for name, cmd in checks:
            try:
                result = subprocess.run(cmd.split(), capture_output=True)
                status = "✅" if result.returncode == 0 else "❌"
                info_text.insert(tk.END, f"{status} {name}\n")
            except:
                info_text.insert(tk.END, f"❌ {name}\n")
        tk.Button(info_window, text="Close", command=info_window.destroy,
                 bg='#3498db', fg='white').pack(pady=10)
    
    def save_schedule(self):
        if not self.schedule_data:
            messagebox.showwarning("No Schedule", "No schedule data to save!")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Schedule"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.schedule_data, f, indent=4)
                self.status_label.config(text="✅ Schedule saved successfully!")
                messagebox.showinfo("Success", "Schedule saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save schedule: {str(e)}")
    
    def load_schedule(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Schedule"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.schedule_data = json.load(f)
                self.display_schedule()
                if self.schedule_data and any(self.schedule_data.values()):
                    # Auto-save the loaded data as the new persistent data
                    self.auto_save_schedule()
                    
                    self.start_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="✅ Schedule loaded successfully!")
                else:
                    self.status_label.config(text="⚠️ Loaded schedule is empty")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load schedule: {str(e)}")
                self.status_label.config(text="❌ Error loading schedule")
    
    def on_tts_setting_change(self):
        """Auto-save when TTS settings change"""
        self.auto_save_schedule()
    
    def run_scheduler(self):
        while self.reminders_active:
            schedule.run_pending()
            time.sleep(30)
    
    def run(self):
        self.root.mainloop()

def main():
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'plyer': 'plyer',
        'schedule': 'schedule',
        'pandas': 'pandas',
        'PIL': 'pillow'
    }
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print("❌ Missing basic dependencies!")
        print("\nPlease install required packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n💡 For enhanced features on Ubuntu, also install:")
        print("sudo apt-get install espeak-ng festival libnotify-bin zenity tesseract-ocr")
        print("pip install gtts pygame paddlepaddle paddleocr pytesseract pyttsx3")
        sys.exit(1)
    if not PADDLE_AVAILABLE:
        print("\n⚠️  Advanced OCR not available.")
        print("For AI-powered OCR, install PaddleOCR:")
        print("pip install paddlepaddle paddleocr")
        if not TESSERACT_AVAILABLE:
            print("\nFor basic OCR, install Tesseract:")
            print("sudo apt-get install tesseract-ocr")
            print("pip install pytesseract")
        print("\nContinuing with available OCR...\n")
    if not GTTS_AVAILABLE and not PYTTSX3_AVAILABLE:
        print("\n💡 For voice alerts, install TTS:")
        print("pip install gtts pygame pyttsx3\n")
    ubuntu_packages = [
        ('espeak', 'sudo apt-get install espeak'),
        ('notify-send', 'sudo apt-get install libnotify-bin'),
        ('paplay', 'sudo apt-get install pulseaudio-utils')
    ]
    for pkg, install_cmd in ubuntu_packages:
        try:
            result = subprocess.run(['which', pkg], capture_output=True)
            if result.returncode == 0:
                print(f"✅ {pkg} available")
            else:
                print(f"⚠️  {pkg} not found - install with: {install_cmd}")
        except:
            pass
    try:
        app = TimetableReminder()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have Python 3.7+")
        print("2. Install all required packages")
        print("3. Check if you're running on a desktop environment")
        sys.exit(1)

if __name__ == "__main__":
    main()
