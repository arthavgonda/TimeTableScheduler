#!/usr/bin/env python3
"""
Advanced Smart Desktop Reminder App for Ubuntu
With state-of-the-art OCR using PaddleOCR and image enhancement
"""

import os
import sys
import json
import threading
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font, scrolledtext
import cv2
import numpy as np
import re
from plyer import notification
import pyttsx3
import schedule
import pandas as pd
from PIL import Image, ImageEnhance, ImageTk

# Advanced OCR imports
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

class TimetableReminder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Timetable Reminder - AI Enhanced")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_enabled = tk.BooleanVar(value=False)
        
        # Store extracted schedule
        self.schedule_data = {}
        self.current_image_path = None
        self.reminders_active = False
        self.scheduler_thread = None
        
        # Initialize OCR models
        self.ocr_engine = None
        self.upsampler = None
        
        # Get current day
        self.current_day = datetime.now().strftime('%A')
        
        # Create GUI
        self.setup_gui()
        
        # Initialize models in background
        threading.Thread(target=self.initialize_models, daemon=True).start()
        
    def initialize_models(self):
        """Initialize AI models in background"""
        try:
            # Initialize PaddleOCR
            if PADDLE_AVAILABLE:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=torch.cuda.is_available() if 'torch' in sys.modules else False,
                    show_log=False
                )
                self.root.after(0, lambda: self.status_label.config(
                    text="✅ Advanced OCR models loaded"))
            
            # Initialize Real-ESRGAN for image enhancement
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
        """Setup the GUI interface"""
        # Header frame
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_font = font.Font(family="Arial", size=20, weight="bold")
        title_label = tk.Label(header_frame, text="📅 Smart Timetable Reminder - AI Enhanced", 
                              font=title_font, fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        
        # AI status indicator
        self.ai_status = tk.Label(header_frame, text="🤖 AI Ready" if PADDLE_AVAILABLE else "⚠️ Basic Mode", 
                                 font=('Arial', 10), fg='#ecf0f1', bg='#2c3e50')
        self.ai_status.place(x=10, y=45)
        
        # Current day indicator
        current_day_label = tk.Label(header_frame, text=f"Today: {self.current_day}", 
                                   font=('Arial', 11), fg='#ecf0f1', bg='#2c3e50')
        current_day_label.place(x=10, y=10)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=1, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Notebook for tabs
        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Input Methods
        input_tab = tk.Frame(notebook, bg='white')
        notebook.add(input_tab, text="Input Methods")
        
        # Manual Entry button
        tk.Button(input_tab, text="✏️ Manual Entry (Grid)", 
                 command=self.open_manual_entry, bg='#27ae60', fg='white',
                 font=('Arial', 12, 'bold'), padx=20, pady=12, cursor='hand2').pack(pady=15)
        
        # Import CSV button
        tk.Button(input_tab, text="📊 Import CSV/Excel", 
                 command=self.import_csv, bg='#3498db', fg='white',
                 font=('Arial', 11), padx=15, pady=10, cursor='hand2').pack(pady=10)
        
        # Advanced OCR button
        ocr_btn_text = "🤖 AI-Powered OCR" if PADDLE_AVAILABLE else "📷 Basic OCR"
        ocr_btn_color = "#9b59b6" if PADDLE_AVAILABLE else "#95a5a6"
        tk.Button(input_tab, text=ocr_btn_text, 
                 command=self.upload_image_advanced, bg=ocr_btn_color, fg='white',
                 font=('Arial', 11, 'bold'), padx=15, pady=10, cursor='hand2').pack(pady=10)
        
        # OCR settings frame
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
        
        # Tab 2: Controls
        control_tab = tk.Frame(notebook, bg='white')
        notebook.add(control_tab, text="Controls")
        
        # Voice alerts checkbox
        tk.Checkbutton(control_tab, text="🔊 Enable Voice Alerts", 
                      variable=self.tts_enabled, bg='white',
                      font=('Arial', 11), cursor='hand2').pack(pady=15)
        
        # Start/Stop button
        self.start_btn = tk.Button(control_tab, text="▶️ Start Reminders", 
                                  command=self.toggle_reminders, bg='#e74c3c', fg='white',
                                  font=('Arial', 12, 'bold'), padx=20, pady=10, cursor='hand2',
                                  state=tk.DISABLED)
        self.start_btn.pack(pady=10)
        
        # Test reminder button
        tk.Button(control_tab, text="🔔 Test Notification", 
                 command=self.test_notification, bg='#9b59b6', fg='white',
                 font=('Arial', 10), padx=15, pady=8, cursor='hand2').pack(pady=10)
        
        # Save/Load frame
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
        
        # Status frame
        status_frame = tk.Frame(left_panel, bg='#ecf0f1', padx=20, pady=15)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, text="⏳ Ready to input timetable", 
                                    bg='#ecf0f1', fg='#2c3e50', font=('Arial', 10), 
                                    wraplength=300)
        self.status_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        
        # Right panel - Schedule display
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Schedule header
        schedule_header = tk.Frame(right_panel, bg='#34495e', height=40)
        schedule_header.pack(fill=tk.X)
        schedule_header.pack_propagate(False)
        
        schedule_title = tk.Label(schedule_header, text="📋 Your Schedule", 
                                 font=('Arial', 14, 'bold'), fg='white', bg='#34495e')
        schedule_title.pack(pady=8)
        
        # Schedule display
        schedule_container = tk.Frame(right_panel, bg='white')
        schedule_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(schedule_container)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.schedule_text = tk.Text(text_frame, font=('Consolas', 11), wrap=tk.WORD,
                                    bg='#fafafa', fg='#2c3e50')
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", 
                                 command=self.schedule_text.yview)
        
        self.schedule_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_text.config(yscrollcommand=scrollbar.set)
        
        # Configure text tags
        self.schedule_text.tag_config("day", font=('Arial', 13, 'bold'), foreground='#2c3e50')
        self.schedule_text.tag_config("current_day", font=('Arial', 13, 'bold'), 
                                     foreground='white', background='#e74c3c')
        self.schedule_text.tag_config("time", font=('Consolas', 11), foreground='#e74c3c')
        self.schedule_text.tag_config("task", font=('Arial', 11), foreground='#34495e')
        self.schedule_text.tag_config("header", font=('Arial', 16, 'bold'), foreground='#2c3e50')
        
        # Insert welcome message
        self.display_welcome_message()
        
    def display_welcome_message(self):
        """Display welcome message with feature information"""
        self.schedule_text.delete(1.0, tk.END)
        self.schedule_text.insert(tk.END, "Welcome to AI-Enhanced Timetable Reminder!\n\n", "header")
        
        if PADDLE_AVAILABLE:
            self.schedule_text.insert(tk.END, "🤖 AI Features Available:\n", "day")
            self.schedule_text.insert(tk.END, "• Advanced table detection with PaddleOCR\n", "task")
            self.schedule_text.insert(tk.END, "• Multi-language support\n", "task")
            self.schedule_text.insert(tk.END, "• Automatic layout understanding\n\n", "task")
        
        self.schedule_text.insert(tk.END, "Choose an input method:\n\n", "day")
        self.schedule_text.insert(tk.END, "1. ✏️ Manual Entry\n", "task")
        self.schedule_text.insert(tk.END, "   Easy grid interface for quick entry\n\n", "task")
        self.schedule_text.insert(tk.END, "2. 📊 Import CSV/Excel\n", "task")
        self.schedule_text.insert(tk.END, "   Import from spreadsheet files\n\n", "task")
        self.schedule_text.insert(tk.END, "3. 🤖 AI-Powered OCR\n", "task")
        self.schedule_text.insert(tk.END, "   Extract from images with AI accuracy\n\n", "task")
        self.schedule_text.insert(tk.END, f"📍 Today is {self.current_day}\n", "current_day")
    
    def upload_image_advanced(self):
        """Handle image upload for advanced OCR"""
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
            
            # Process in thread
            threading.Thread(target=self.process_image_advanced, args=(file_path,)).start()
    
    def enhance_image_quality(self, img_path):
        """Enhance image quality using various techniques"""
        img = cv2.imread(img_path)
        
        # Basic enhancement
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        
        # Increase resolution if too small
        height, width = denoised.shape[:2]
        if width < 1500:
            scale = 1500 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            denoised = cv2.resize(denoised, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
        
        return denoised
    
    def process_image_advanced(self, image_path):
        """Process image with advanced OCR"""
        try:
            # Enhance image if enabled
            if self.enhance_image.get():
                enhanced_img = self.enhance_image_quality(image_path)
                temp_path = "enhanced_timetable.png"
                cv2.imwrite(temp_path, enhanced_img)
                process_path = temp_path
            else:
                process_path = image_path
            
            if PADDLE_AVAILABLE and self.ocr_engine:
                # Use PaddleOCR for advanced extraction
                result = self.ocr_engine.ocr(process_path, cls=True)
                
                # Extract table structure
                self.schedule_data = self.extract_table_from_paddle(result)
                
            else:
                # Fallback to basic OCR
                import pytesseract
                img = cv2.imread(process_path)
                text = pytesseract.image_to_string(img)
                self.schedule_data = self.extract_basic_schedule(text)
            
            # Clean up
            if os.path.exists("enhanced_timetable.png"):
                os.remove("enhanced_timetable.png")
            
            # Update UI
            self.root.after(0, self.ocr_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.ocr_error(str(e)))
    
    def extract_table_from_paddle(self, ocr_result):
        """Extract table structure from PaddleOCR results"""
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days:
            schedule[day] = []
        
        if not ocr_result or not ocr_result[0]:
            return schedule
        
        # Extract all text boxes with positions
        text_boxes = []
        for line in ocr_result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence > 0.8:  # High confidence only
                # Calculate center position
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                
                text_boxes.append({
                    'text': text,
                    'x': x_center,
                    'y': y_center,
                    'bbox': bbox
                })
        
        # Sort by y-coordinate (rows) then x-coordinate (columns)
        text_boxes.sort(key=lambda b: (b['y'], b['x']))
        
        # Group into rows based on y-coordinate proximity
        rows = []
        current_row = []
        last_y = -1
        row_threshold = 30  # Pixels
        
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
        
        # Find header row with days
        header_row_idx = -1
        day_columns = {}
        
        for idx, row in enumerate(rows[:5]):  # Check first 5 rows
            found_days = 0
            for cell_idx, cell in enumerate(row):
                text_lower = cell['text'].lower()
                for day in days:
                    if day.lower() in text_lower or day[:3].lower() in text_lower:
                        day_columns[cell_idx] = day
                        found_days += 1
                        break
            
            if found_days >= 3:  # Found multiple days
                header_row_idx = idx
                break
        
        # Extract schedule from rows
        for row_idx in range(header_row_idx + 1, len(rows)):
            row = rows[row_idx]
            
            # First cell should be time
            if row and self.is_time_text(row[0]['text']):
                time_str = self.normalize_time(row[0]['text'])
                
                # Extract tasks for each day column
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
        """Check if text represents a time"""
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
        """Basic schedule extraction from text"""
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days:
            schedule[day] = []
        
        lines = text.split('\n')
        current_day = None
        
        for line in lines:
            line = line.strip()
            
            # Check for day
            for day in days:
                if day.lower() in line.lower():
                    current_day = day
                    break
            
            # Check for time pattern
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
        """Handle OCR completion"""
        self.progress.stop()
        self.progress.pack_forget()
        
        # Display results
        self.display_schedule()
        
        total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
        
        if total_tasks > 0:
            self.status_label.config(text=f"✅ Extracted {total_tasks} tasks! Review and edit as needed.")
            self.start_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            
            # Open editor for review
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
        """Handle OCR errors"""
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="❌ OCR failed. Try manual entry.")
        messagebox.showerror("OCR Error", f"Failed to process image:\n{error_msg}")
    
    def open_edit_window(self):
        """Open schedule editor window"""
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Schedule")
        edit_window.geometry("800x600")
        
        # Instructions
        tk.Label(edit_window, text="Review and edit extracted schedule", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Create notebook for days
        notebook = ttk.Notebook(edit_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_frames = {}
        
        for day in days:
            frame = tk.Frame(notebook)
            notebook.add(frame, text=day)
            day_frames[day] = frame
            
            # Listbox for tasks
            list_frame = tk.Frame(frame)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            listbox = tk.Listbox(list_frame, font=('Consolas', 10))
            scrollbar = tk.Scrollbar(list_frame)
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)
            
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Populate with current tasks
            if day in self.schedule_data:
                for task in sorted(self.schedule_data[day], key=lambda x: x['time']):
                    listbox.insert(tk.END, f"{task['time']} - {task['task']}")
            
            # Add/Edit controls
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
        
        # Save button
        def save_edits():
            self.display_schedule()
            edit_window.destroy()
        
        tk.Button(edit_window, text="💾 Save Changes", command=save_edits,
                 bg='#3498db', fg='white', font=('Arial', 11),
                 padx=20, pady=10).pack(pady=10)
    
    def open_manual_entry(self):
        """Open manual entry window with grid interface"""
        entry_window = tk.Toplevel(self.root)
        entry_window.title("Manual Schedule Entry")
        entry_window.geometry("1200x700")
        
        # Instructions
        inst_frame = tk.Frame(entry_window, bg='#3498db', height=50)
        inst_frame.pack(fill=tk.X)
        inst_frame.pack_propagate(False)
        
        tk.Label(inst_frame, text="Enter your schedule in the grid below", 
                font=('Arial', 14, 'bold'), fg='white', bg='#3498db').pack(pady=10)
        
        # Main frame with canvas for scrolling
        main_frame = tk.Frame(entry_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbars
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
        
        # Grid setup
        days = ['Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Default time slots
        time_slots = [
            "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00",
            "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00",
            "20:00", "21:00", "22:00", "23:00"
        ]
        
        # Create header
        for col, day in enumerate(days):
            label = tk.Label(scrollable_frame, text=day, font=('Arial', 11, 'bold'),
                           bg='#34495e', fg='white', relief=tk.RAISED, bd=1,
                           width=15, padx=5, pady=8)
            label.grid(row=0, column=col, sticky='ew')
        
        # Create time entries and task inputs
        entry_widgets = {}
        
        for row, time in enumerate(time_slots, start=1):
            # Time label
            time_label = tk.Label(scrollable_frame, text=time, font=('Arial', 10, 'bold'),
                                bg='#ecf0f1', relief=tk.RAISED, bd=1, width=15, pady=5)
            time_label.grid(row=row, column=0, sticky='ew')
            
            # Task entries for each day
            for col in range(1, 8):
                entry = tk.Entry(scrollable_frame, font=('Arial', 10), width=20)
                entry.grid(row=row, column=col, sticky='ew', padx=1, pady=1)
                entry_widgets[(row-1, col-1)] = entry
        
        # Add row button
        def add_time_row():
            new_row = len(scrollable_frame.grid_slaves()) // 8
            time_entry = tk.Entry(scrollable_frame, font=('Arial', 10, 'bold'),
                                bg='#ecf0f1', width=15)
            time_entry.grid(row=new_row, column=0, sticky='ew')
            
            for col in range(1, 8):
                entry = tk.Entry(scrollable_frame, font=('Arial', 10), width=20)
                entry.grid(row=new_row, column=col, sticky='ew', padx=1, pady=1)
                entry_widgets[(new_row-1, col-1)] = entry
        
        # Pack canvas and scrollbars
        canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Button frame
        btn_frame = tk.Frame(entry_window)
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="➕ Add Time Slot", command=add_time_row,
                 bg='#3498db', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        # Save function
        def save_manual_entry():
            self.schedule_data = {}
            days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Initialize days
            for day in days_list:
                self.schedule_data[day] = []
            
            # Get all time entries
            time_entries = []
            for widget in scrollable_frame.grid_slaves():
                if widget.grid_info()['column'] == 0 and widget.grid_info()['row'] > 0:
                    if isinstance(widget, tk.Entry):
                        time_entries.append((widget.grid_info()['row'], widget.get()))
                    elif isinstance(widget, tk.Label):
                        time_entries.append((widget.grid_info()['row'], widget.cget('text')))
            
            # Sort by row
            time_entries.sort(key=lambda x: x[0])
            
            # Process each row
            for row_idx, (row, time_text) in enumerate(time_entries):
                if time_text and ':' in time_text:
                    normalized_time = self.normalize_time(time_text)
                    
                    # Get tasks for this time
                    for col, day in enumerate(days_list):
                        if (row_idx, col) in entry_widgets:
                            task = entry_widgets[(row_idx, col)].get().strip()
                            if task:
                                self.schedule_data[day].append({
                                    'time': normalized_time,
                                    'task': task
                                })
            
            # Display and enable buttons
            self.display_schedule()
            if any(self.schedule_data.values()):
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
        """Import schedule from CSV or Excel file"""
        file_path = filedialog.askopenfilename(
            title="Select Schedule File",
            filetypes=[("Spreadsheet files", "*.csv *.xlsx *.xls"), 
                      ("CSV files", "*.csv"),
                      ("Excel files", "*.xlsx *.xls"),
                      ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Read file based on extension
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                self.schedule_data = {}
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Initialize days
                for day in days:
                    self.schedule_data[day] = []
                
                # Check if first column is time
                if 'Time' in df.columns or df.columns[0].lower() == 'time':
                    time_col = df.columns[0]
                    
                    # Process each row
                    for idx, row in df.iterrows():
                        time_str = str(row[time_col])
                        if pd.notna(time_str) and ':' in time_str:
                            normalized_time = self.normalize_time(time_str)
                            
                            # Check each day column
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
                
                # Display results
                self.display_schedule()
                
                if any(self.schedule_data.values()):
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
        """Normalize time string to 24-hour format"""
        time_str = str(time_str).strip()
        
        # Handle time ranges
        if '-' in time_str:
            time_str = time_str.split('-')[0].strip()
        
        # Remove any non-time characters
        time_str = re.sub(r'[^\d:APMapm\s]', '', time_str)
        
        # Try different formats
        formats = ['%H:%M', '%I:%M %p', '%I:%M%p', '%H:%M:%S']
        
        for fmt in formats:
            try:
                time_obj = datetime.strptime(time_str.upper(), fmt)
                return time_obj.strftime('%H:%M')
            except:
                continue
        
        # Basic fallback
        match = re.search(r'(\d{1,2}):(\d{2})', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            # Handle PM
            if 'pm' in time_str.lower() and hour < 12:
                hour += 12
            elif 'am' in time_str.lower() and hour == 12:
                hour = 0
                
            return f"{hour:02d}:{minute:02d}"
        
        return "00:00"
    
    def display_schedule(self):
        """Display schedule in the GUI"""
        self.schedule_text.delete(1.0, tk.END)
        
        total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
        
        if not self.schedule_data or total_tasks == 0:
            self.display_welcome_message()
            return
        
        self.schedule_text.insert(tk.END, "📅 Your Schedule\n", "header")
        self.schedule_text.insert(tk.END, f"Total: {total_tasks} tasks\n", "task")
        self.schedule_text.insert(tk.END, "─" * 60 + "\n\n")
        
        # Get days in order starting from today
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_idx = days_order.index(self.current_day) if self.current_day in days_order else 0
        ordered_days = days_order[current_idx:] + days_order[:current_idx]
        
        for day in ordered_days:
            if day in self.schedule_data:
                tasks = self.schedule_data[day]
                
                # Highlight current day
                if day == self.current_day:
                    self.schedule_text.insert(tk.END, f" {day} (TODAY) ", "current_day")
                else:
                    self.schedule_text.insert(tk.END, f"{day}", "day")
                self.schedule_text.insert(tk.END, f" ({len(tasks)} tasks)\n", "task")
                
                if tasks:
                    # Sort by time
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
        """Start or stop reminders"""
        if not self.reminders_active:
            self.start_reminders()
        else:
            self.stop_reminders()
    
    def start_reminders(self):
        """Start the reminder system"""
        self.reminders_active = True
        self.start_btn.config(text="⏸️ Stop Reminders", bg='#c0392b')
        self.status_label.config(text="🔔 Reminders active - monitoring schedule")
        
        # Setup reminders
        self.setup_reminders()
        
        # Start scheduler thread
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self.scheduler_thread.start()
    
    def stop_reminders(self):
        """Stop the reminder system"""
        self.reminders_active = False
        self.start_btn.config(text="▶️ Start Reminders", bg='#e74c3c')
        self.status_label.config(text="⏹️ Reminders stopped")
        
        # Clear scheduled jobs
        schedule.clear()
    
    def setup_reminders(self):
        """Setup scheduled reminders"""
        schedule.clear()
        
        for day, tasks in self.schedule_data.items():
            for task in tasks:
                day_num = self.get_day_number(day)
                time_str = task['time']
                
                if day_num is not None and ':' in time_str:
                    self.schedule_task_reminder(day_num, time_str, task['task'])
    
    def get_day_number(self, day_name):
        """Convert day name to number (0=Monday, 6=Sunday)"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        try:
            return days.index(day_name)
        except ValueError:
            return None
    
    def schedule_task_reminder(self, day_num, time_str, task):
        """Schedule a reminder for a specific task"""
        def remind():
            if self.reminders_active:
                current_day = datetime.now().weekday()
                if current_day == day_num:
                    self.show_reminder(task, time_str)
        
        # Schedule the reminder
        try:
            schedule.every().day.at(time_str).do(remind)
            
            # Schedule 5-minute advance reminder
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
        """Show desktop notification and optional voice alert"""
        try:
            # Desktop notification
            notification.notify(
                title=f"📅 Reminder: {time_str}",
                message=task,
                app_name="Timetable Reminder",
                timeout=10
            )
            
            # Update status in GUI
            self.root.after(0, lambda: self.status_label.config(
                text=f"🔔 Reminder shown: {task[:30]}..."))
            
            # Voice alert if enabled
            if self.tts_enabled.get():
                threading.Thread(target=self.speak_reminder, args=(task,)).start()
        except Exception as e:
            print(f"Error showing reminder: {e}")
    
    def speak_reminder(self, task):
        """Speak the reminder using text-to-speech"""
        try:
            self.tts_engine.say(f"Reminder: {task}")
            self.tts_engine.runAndWait()
        except:
            pass
    
    def test_notification(self):
        """Test the notification system"""
        self.show_reminder("This is a test notification", datetime.now().strftime("%H:%M"))
    
    def save_schedule(self):
        """Save the current schedule to a JSON file"""
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
        """Load a schedule from a JSON file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Schedule"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.schedule_data = json.load(f)
                
                # Display loaded schedule
                self.display_schedule()
                
                # Enable buttons
                if self.schedule_data and any(self.schedule_data.values()):
                    self.start_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="✅ Schedule loaded successfully!")
                else:
                    self.status_label.config(text="⚠️ Loaded schedule is empty")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load schedule: {str(e)}")
                self.status_label.config(text="❌ Error loading schedule")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while self.reminders_active:
            schedule.run_pending()
            time.sleep(30)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    # Check basic dependencies
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'plyer': 'plyer',
        'pyttsx3': 'pyttsx3',
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
        print("Missing basic dependencies!")
        print("\nPlease install required packages:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    # Check for advanced OCR
    if not PADDLE_AVAILABLE:
        print("\n⚠️  Advanced OCR not available.")
        print("For AI-powered OCR, install PaddleOCR:")
        print("pip install paddlepaddle paddleocr")
        print("\nContinuing with basic OCR...\n")
    
    # Create and run app
    app = TimetableReminder()
    app.run()

if __name__ == "__main__":
    main()
