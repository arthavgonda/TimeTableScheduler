#!/usr/bin/env python3
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
PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    print("PaddleOCR not found. Falling back to basic OCR (Pytesseract).")
PYTESSERACT_AVAILABLE = False
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    print("Pytesseract also not found. OCR features will be limited.")
ESRGAN_AVAILABLE = False
try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    print("RealESRGAN not found. Image enhancement will use basic methods.")
class TimetableReminder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Timetable Reminder - AI Enhanced")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        self.root.configure(bg='#f0f0f0')
        self.style = ttk.Style()
        self.style.theme_use('clam')
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_enabled = tk.BooleanVar(value=False)
            self.tts_available = True
        except Exception as e:
            print(f"pyttsx3 initialization failed: {e}. Voice alerts will be disabled.")
            messagebox.showwarning("Voice Alert Error", "Voice alert engine could not be initialized. Please ensure 'pyttsx3' is correctly installed with its dependencies (e.g., espeak on Linux). Voice alerts will be disabled.")
            self.tts_enabled = tk.BooleanVar(value=False)
            self.tts_available = False
        self.schedule_data = {}
        self.current_image_path = None
        self.reminders_active = False
        self.scheduler_thread = None
        self.ocr_engine = None
        self.upsampler = None
        self.current_day = datetime.now().strftime('%A')
        self.setup_gui()
        threading.Thread(target=self.initialize_models, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?\nActive reminders will be stopped."):
            self.reminders_active = False
            schedule.clear()
            if self.tts_available:
                self.tts_engine.stop()
            self.root.destroy()
    def initialize_models(self):
        try:
            if PADDLE_AVAILABLE:
                self.root.after(0, lambda: self.status_label.config(text="⏳ Loading PaddleOCR model..."))
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=torch.cuda.is_available() if 'torch' in sys.modules else False,
                    show_log=False
                )
                self.root.after(0, lambda: self.status_label.config(text="✅ Advanced OCR models loaded"))
            else:
                self.root.after(0, lambda: self.status_label.config(text="⚠️ PaddleOCR not available. Using basic OCR."))
            global ESRGAN_AVAILABLE
            if ESRGAN_AVAILABLE and 'torch' in sys.modules and torch.cuda.is_available():
                self.root.after(0, lambda: self.status_label.config(text="⏳ Loading ESRGAN model..."))
                model_path = 'RealESRGAN_x4plus.pth'
                if not os.path.exists(model_path):
                    self.root.after(0, lambda: self.ai_status.config(text="⚠️ ESRGAN model file not found."))
                    print(f"ESRGAN model file '{model_path}' not found. Skipping ESRGAN enhancement.")
                    ESRGAN_AVAILABLE = False
                else:
                    try:
                        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                        self.upsampler = RealESRGANer(
                            scale=4,
                            model_path=model_path,
                            model=model,
                            tile=0,
                            tile_pad=10,
                            pre_pad=0,
                            half=True
                        )
                        self.root.after(0, lambda: self.ai_status.config(text="🤖 AI Ready (Full)"))
                        self.root.after(0, lambda: self.status_label.config(text="✅ ESRGAN model loaded for enhancement."))
                        self.root.after(0, lambda: self.enhance_checkbox.config(state=tk.NORMAL))
                    except Exception as esrgan_e:
                        self.root.after(0, lambda: self.ai_status.config(text="⚠️ ESRGAN init failed."))
                        print(f"ESRGAN initialization failed: {esrgan_e}")
                        ESRGAN_AVAILABLE = False
            else:
                self.root.after(0, lambda: self.ai_status.config(text="⚠️ Basic Enhancement (No ESRGAN)"))
                self.root.after(0, lambda: self.enhance_checkbox.config(state=tk.DISABLED))
        except Exception as e:
            print(f"Overall model initialization failed: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text="❌ AI model loading failed. Basic mode only."))
            self.root.after(0, lambda: self.ai_status.config(text="❌ AI Failed"))
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
        current_day_label = tk.Label(header_frame, text=f"Today: {self.current_day}",
                                     font=('Arial', 11), fg='#ecf0f1', bg='#2c3e50')
        current_day_label.place(x=10, y=10)
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=0)
        main_container.grid_columnconfigure(1, weight=1)
        left_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=1)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=0)
        left_panel.grid_columnconfigure(0, weight=1)
        notebook = ttk.Notebook(left_panel)
        notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        input_tab = tk.Frame(notebook, bg='white')
        notebook.add(input_tab, text="Input Methods")
        input_tab.pack_propagate(False)
        tk.Button(input_tab, text="✏️ Manual Entry (Grid)",
                  command=self.open_manual_entry, bg='#27ae60', fg='white',
                  font=('Arial', 12, 'bold'), padx=20, pady=12, cursor='hand2',
                  activebackground='#2ecc71', activeforeground='white').pack(pady=15)
        tk.Button(input_tab, text="📊 Import CSV/Excel",
                  command=self.import_csv, bg='#3498db', fg='white',
                  font=('Arial', 11), padx=15, pady=10, cursor='hand2',
                  activebackground='#2980b9', activeforeground='white').pack(pady=10)
        ocr_btn_text = "🤖 AI-Powered OCR" if PADDLE_AVAILABLE else "📷 Basic OCR (Requires Tesseract)"
        ocr_btn_color = "#9b59b6" if PADDLE_AVAILABLE else "#95a5a6"
        tk.Button(input_tab, text=ocr_btn_text,
                  command=self.upload_image_advanced, bg=ocr_btn_color, fg='white',
                  font=('Arial', 11, 'bold'), padx=15, pady=10, cursor='hand2',
                  activebackground='#8e44ad', activeforeground='white').pack(pady=10)
        ocr_settings = tk.LabelFrame(input_tab, text="OCR Settings", bg='white',
                                     font=('Arial', 10, 'bold'))
        ocr_settings.pack(pady=10, padx=10, fill=tk.X)
        self.enhance_image = tk.BooleanVar(value=True)
        self.enhance_checkbox = tk.Checkbutton(ocr_settings, text="🔍 Enhance image quality (ESRGAN)",
                                            variable=self.enhance_image, bg='white',
                                            font=('Arial', 9), state=tk.DISABLED)
        self.enhance_checkbox.pack(anchor=tk.W, padx=10, pady=2)
        self.auto_detect_layout = tk.BooleanVar(value=True)
        tk.Checkbutton(ocr_settings, text="📐 Auto-detect table layout (PaddleOCR)",
                       variable=self.auto_detect_layout, bg='white',
                       font=('Arial', 9), state=tk.NORMAL if PADDLE_AVAILABLE else tk.DISABLED).pack(anchor=tk.W, padx=10, pady=2)
        control_tab = tk.Frame(notebook, bg='white')
        notebook.add(control_tab, text="Controls")
        control_tab.pack_propagate(False)
        tk.Checkbutton(control_tab, text="🔊 Enable Voice Alerts",
                       variable=self.tts_enabled, bg='white',
                       font=('Arial', 11), cursor='hand2',
                       state=tk.NORMAL if self.tts_available else tk.DISABLED).pack(pady=15)
        self.start_btn = tk.Button(control_tab, text="▶️ Start Reminders",
                                   command=self.toggle_reminders, bg='#e74c3c', fg='white',
                                   font=('Arial', 12, 'bold'), padx=20, pady=10, cursor='hand2',
                                   activebackground='#c0392b', activeforeground='white',
                                   state=tk.DISABLED)
        self.start_btn.pack(pady=10)
        tk.Button(control_tab, text="🔔 Test Notification",
                  command=self.test_notification, bg='#9b59b6', fg='white',
                  font=('Arial', 10), padx=15, pady=8, cursor='hand2',
                  activebackground='#8e44ad', activeforeground='white').pack(pady=10)
        save_load_frame = tk.Frame(control_tab, bg='white')
        save_load_frame.pack(pady=15)
        self.save_btn = tk.Button(save_load_frame, text="💾 Save",
                                   command=self.save_schedule, bg='#16a085', fg='white',
                                   font=('Arial', 10), padx=15, pady=5, cursor='hand2',
                                   activebackground='#1abc9c', activeforeground='white',
                                   state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=3)
        tk.Button(save_load_frame, text="📂 Load",
                  command=self.load_schedule, bg='#16a085', fg='white',
                  font=('Arial', 10), padx=15, pady=5, cursor='hand2',
                  activebackground='#1abc9c', activeforeground='white').pack(side=tk.LEFT, padx=3)
        status_frame = tk.Frame(left_panel, bg='#ecf0f1', padx=10, pady=10)
        status_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.status_label = tk.Label(status_frame, text="⏳ Ready to input timetable",
                                     bg='#ecf0f1', fg='#2c3e50', font=('Arial', 10),
                                     wraplength=300, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=1)
        right_panel.grid(row=0, column=1, sticky='nsew')
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
                                     bg='#fafafa', fg='#2c3e50', bd=0, relief=tk.FLAT,
                                     insertbackground='black')
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical",
                                  command=self.schedule_text.yview)
        self.schedule_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_text.config(yscrollcommand=scrollbar.set)
        self.schedule_text.bind("<Key>", lambda e: "break")
        self.schedule_text.bind("<Button-1>", lambda e: "break")
        self.schedule_text.tag_config("day", font=('Arial', 13, 'bold'), foreground='#2c3e50')
        self.schedule_text.tag_config("current_day", font=('Arial', 13, 'bold'),
                                      foreground='white', background='#e74c3c')
        self.schedule_text.tag_config("time", font=('Consolas', 11, 'bold'), foreground='#e74c3c')
        self.schedule_text.tag_config("task", font=('Arial', 11), foreground='#34495e')
        self.schedule_text.tag_config("header", font=('Arial', 16, 'bold'), foreground='#2c3e50')
        self.display_welcome_message()
    def display_welcome_message(self):
        self.schedule_text.config(state=tk.NORMAL)
        self.schedule_text.delete(1.0, tk.END)
        self.schedule_text.insert(tk.END, "Welcome to AI-Enhanced Timetable Reminder!\n\n", "header")
        if PADDLE_AVAILABLE:
            self.schedule_text.insert(tk.END, "🤖 AI Features Available:\n", "day")
            self.schedule_text.insert(tk.END, "• Advanced table detection with PaddleOCR\n", "task")
            self.schedule_text.insert(tk.END, "• Multi-language support (PaddleOCR)\n", "task")
            self.schedule_text.insert(tk.END, "• Automatic layout understanding\n", "task")
            if ESRGAN_AVAILABLE:
                self.schedule_text.insert(tk.END, "• AI-powered image super-resolution (Real-ESRGAN)\n\n", "task")
            else:
                self.schedule_text.insert(tk.END, "• Basic image enhancement\n\n", "task")
        elif PYTESSERACT_AVAILABLE:
             self.schedule_text.insert(tk.END, "📷 Basic OCR (Pytesseract) Available:\n", "day")
             self.schedule_text.insert(tk.END, "• Extracts text from images\n", "task")
             self.schedule_text.insert(tk.END, "• Basic image enhancement\n\n", "task")
        else:
            self.schedule_text.insert(tk.END, "⚠️ No advanced OCR features available. Install PaddleOCR or Pytesseract for image import.\n\n", "day")
        self.schedule_text.insert(tk.END, "Choose an input method:\n\n", "day")
        self.schedule_text.insert(tk.END, "1. ✏️ Manual Entry\n", "task")
        self.schedule_text.insert(tk.END, "    Easy grid interface for quick entry and editing.\n\n", "task")
        self.schedule_text.insert(tk.END, "2. 📊 Import CSV/Excel\n", "task")
        self.schedule_text.insert(tk.END, "    Import your schedule directly from spreadsheet files.\n\n", "task")
        self.schedule_text.insert(tk.END, "3. 🤖 AI-Powered OCR (or Basic OCR)\n", "task")
        self.schedule_text.insert(tk.END, "    Extract schedule data from images with high accuracy.\n\n", "task")
        self.schedule_text.insert(tk.END, f"📍 Today is {self.current_day}\n", "current_day")
        self.schedule_text.config(state=tk.DISABLED)
    def upload_image_advanced(self):
        if not PADDLE_AVAILABLE and not PYTESSERACT_AVAILABLE:
            messagebox.showerror("OCR Not Available",
                                 "Neither PaddleOCR nor Pytesseract is installed.\n"
                                 "Please install one for OCR functionality:\n\n"
                                 "For PaddleOCR: pip install paddlepaddle paddleocr\n"
                                 "For Pytesseract: pip install pytesseract && ensure Tesseract OCR engine is installed and in your system PATH (e.g., sudo apt-get install tesseract-ocr on Ubuntu)")
            return
        if not PADDLE_AVAILABLE and PYTESSERACT_AVAILABLE:
            response = messagebox.askyesno("Basic OCR Mode",
                                           "Advanced OCR (PaddleOCR) not available. Do you want to proceed with basic OCR (Pytesseract)?\n\n" +
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
            self.status_label.config(text="🔄 Processing image with AI... This may take a moment.")
            self.progress.pack(pady=5, fill=tk.X)
            self.progress.start()
            threading.Thread(target=self.process_image_advanced, args=(file_path,)).start()
    def enhance_image_quality(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image from path: {img_path}")
        global ESRGAN_AVAILABLE
        if ESRGAN_AVAILABLE and self.upsampler and self.enhance_image.get():
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                output, _ = self.upsampler.enhance(img_rgb, outscale=4)
                enhanced = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                return enhanced
            except Exception as e:
                print(f"ESRGAN enhancement failed: {e}. Falling back to basic enhancement.")
                self.root.after(0, lambda: self.status_label.config(text="⚠️ ESRGAN failed. Using basic image enhancement."))
                ESRGAN_AVAILABLE = False
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
        if width < 1500 and self.enhance_image.get():
            scale = 1500 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            denoised = cv2.resize(denoised, (new_width, new_height),
                                  interpolation=cv2.INTER_CUBIC)
        return denoised
    def process_image_advanced(self, image_path):
        try:
            process_path = image_path
            if self.enhance_image.get():
                self.root.after(0, lambda: self.status_label.config(text="🔄 Enhancing image..."))
                enhanced_img = self.enhance_image_quality(image_path)
                temp_path = "enhanced_timetable.png"
                cv2.imwrite(temp_path, enhanced_img)
                process_path = temp_path
                self.root.after(0, lambda: self.status_label.config(text="🔄 Image enhanced. Performing OCR..."))
            if PADDLE_AVAILABLE and self.ocr_engine and self.auto_detect_layout.get():
                self.root.after(0, lambda: self.status_label.config(text="🔄 Using PaddleOCR with layout detection..."))
                result = self.ocr_engine.ocr(process_path, cls=True)
                self.schedule_data = self.extract_table_from_paddle(result)
            elif PADDLE_AVAILABLE and self.ocr_engine:
                self.root.after(0, lambda: self.status_label.config(text="🔄 Using PaddleOCR for general text extraction..."))
                result = self.ocr_engine.ocr(process_path, cls=True)
                all_text = " ".join([line[1][0] for line in result[0]]) if result and result[0] else ""
                self.schedule_data = self.extract_basic_schedule(all_text)
            elif PYTESSERACT_AVAILABLE:
                self.root.after(0, lambda: self.status_label.config(text="🔄 Using Basic OCR (Pytesseract)..."))
                img = cv2.imread(process_path)
                if img is None:
                    raise ValueError("Could not read image for Tesseract processing.")
                text = pytesseract.image_to_string(img)
                self.schedule_data = self.extract_basic_schedule(text)
            else:
                raise RuntimeError("No OCR engine available or selected.")
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
            print("OCR result is empty or invalid.")
            return schedule
        text_boxes = []
        for line in ocr_result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.75:
                x_center = (bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 4
                y_center = (bbox[0][1] + bbox[1][1] + bbox[2][1] + bbox[3][1]) / 4
                mean_y = (bbox[0][1] + bbox[1][1] + bbox[2][1] + bbox[3][1]) / 4
                text_boxes.append({
                    'text': text.strip(),
                    'x': x_center,
                    'y': mean_y,
                    'bbox': bbox,
                    'width': abs(bbox[1][0] - bbox[0][0]),
                    'height': abs(bbox[2][1] - bbox[0][1])
                })
        text_boxes.sort(key=lambda b: (b['y'], b['x']))
        rows = []
        if not text_boxes: return schedule
        current_row = [text_boxes[0]]
        avg_line_height = np.mean([b['height'] for b in text_boxes]) if text_boxes else 20
        row_threshold = avg_line_height * 0.7
        for i in range(1, len(text_boxes)):
            box = text_boxes[i]
            if abs(box['y'] - current_row[-1]['y']) > row_threshold:
                rows.append(sorted(current_row, key=lambda b: b['x']))
                current_row = [box]
            else:
                current_row.append(box)
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['x']))
        header_row = None
        day_column_map = {}
        for r_idx, row in enumerate(rows[:min(len(rows), 5)]):
            found_days_in_this_row = 0
            temp_day_map = {}
            for cell in row:
                text_lower = cell['text'].lower()
                for day_full in days:
                    if day_full.lower() in text_lower or day_full[:3].lower() in text_lower:
                        temp_day_map[cell['x']] = day_full
                        found_days_in_this_row += 1
                        break
            if found_days_in_this_row >= 2:
                header_row = r_idx
                day_column_map = temp_day_map
                break
        if header_row is None:
            messagebox.showwarning("OCR Layout Warning", "Could not reliably detect timetable table layout. Falling back to basic text extraction. Please review manually.")
            all_text_content = " ".join([box['text'] for row_cells in rows for box in row_cells])
            return self.extract_basic_schedule(all_text_content)
        sorted_day_x_coords = sorted(day_column_map.keys())
        column_x_ranges = []
        for i, x_coord in enumerate(sorted_day_x_coords):
            if i == 0:
                left_bound = 0
            else:
                left_bound = (sorted_day_x_coords[i-1] + x_coord) / 2
            if i == len(sorted_day_x_coords) - 1:
                right_bound = float('inf')
            else:
                right_bound = (x_coord + sorted_day_x_coords[i+1]) / 2
            column_x_ranges.append((day_column_map[x_coord], left_bound, right_bound))
        for r_idx in range(header_row + 1, len(rows)):
            row_cells = rows[r_idx]
            time_str = None
            task_cells_by_day = {day: [] for day in days}
            potential_time_cells = []
            other_cells = []
            for cell in row_cells:
                if self.is_time_text(cell['text']):
                    potential_time_cells.append(cell)
                else:
                    other_cells.append(cell)
            if potential_time_cells:
                potential_time_cells.sort(key=lambda c: c['x'])
                time_str = self.normalize_time(potential_time_cells[0]['text'])
            if time_str:
                for task_cell in other_cells:
                    for day_name, col_left, col_right in column_x_ranges:
                        if col_left <= task_cell['x'] < col_right:
                            task_cells_by_day[day_name].append(task_cell['text'])
                            break
                for day_name, tasks_list in task_cells_by_day.items():
                    if tasks_list:
                        combined_task = " ".join(tasks_list).strip()
                        if combined_task and len(combined_task) > 2:
                            schedule[day_name].append({
                                'time': time_str,
                                'task': combined_task
                            })
        for day in schedule:
            schedule[day].sort(key=lambda x: datetime.strptime(x['time'], '%H:%M'))
        return schedule
    def is_time_text(self, text):
        text = text.strip()
        if not text: return False
        if re.fullmatch(r'^\d{1,2}:\d{2}(?:\s*[ap]m)?$', text, re.IGNORECASE): return True
        if re.fullmatch(r'^\d{1,2}\s*[ap]m$', text, re.IGNORECASE): return True
        if re.fullmatch(r'^\d{3,4}$', text) and len(text) >= 3:
            try:
                h = int(text[:-2])
                m = int(text[-2:])
                if 0 <= h <= 23 and 0 <= m <= 59: return True
            except ValueError:
                pass
        if re.fullmatch(r'^\d{1,2}$', text):
            try:
                h = int(text)
                if 0 <= h <= 23: return True
            except ValueError:
                pass
        print(f"Warning: Could not normalize time '{text}'")
        return False
    def normalize_time(self, time_str):
        time_str = time_str.strip().lower().replace('.', ':')
        match = re.match(r'(\d{1,2}):(\d{2})', time_str)
        if match:
            h = int(match.group(1))
            m = int(match.group(2))
            if 0 <= h <= 23 and 0 <= m <= 59:
                return f"{h:02d}:{m:02d}"
        match = re.match(r'(\d{1,2})\s*(a|p)m', time_str, re.IGNORECASE)
        if match:
            h = int(match.group(1))
            ampm = match.group(2).lower()
            if ampm == 'p' and h != 12: h += 12
            if ampm == 'a' and h == 12: h = 0
            if 0 <= h <= 23:
                return f"{h:02d}:00"
        match = re.fullmatch(r'(\d{3,4})', time_str)
        if match:
            s = match.group(1)
            if len(s) == 3:
                h = int(s[0])
                m = int(s[1:])
            else:
                h = int(s[:2])
                m = int(s[2:])
            if 0 <= h <= 23 and 0 <= m <= 59:
                return f"{h:02d}:{m:02d}"
        if re.fullmatch(r'^\d{1,2}$', time_str):
            h = int(time_str)
            if 0 <= h <= 23:
                return f"{h:02d}:00"
        print(f"Warning: Could not normalize time '{time_str}'")
        return None
    def extract_basic_schedule(self, text):
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            schedule[day] = []
        lines = text.split('\n')
        current_day = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            found_day = False
            for day_full in days:
                if day_full.lower() in line.lower() or day_full[:3].lower() in line.lower():
                    if not self.is_time_text(line):
                        current_day = day_full
                        found_day = True
                        break
            if found_day:
                continue
            time_match = re.search(r'(\d{1,2}:\d{2}(?:\s*[ap]m)?|\d{1,2}(?:\s*[ap]m))', line, re.IGNORECASE)
            if time_match and current_day:
                time_str = time_match.group(1)
                task = line.replace(time_str, '', 1).strip(' -:').strip()
                normalized_time = self.normalize_time(time_str)
                if normalized_time and task and len(task) > 2:
                    schedule[current_day].append({
                        'time': normalized_time,
                        'task': task[:100]
                    })
        for day in schedule:
            schedule[day].sort(key=lambda x: datetime.strptime(x['time'], '%H:%M'))
        return schedule
    def ocr_complete(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.display_schedule()
        total_tasks = sum(len(tasks) for tasks in self.schedule_data.values())
        if total_tasks > 0:
            self.status_label.config(text=f"✅ Extracted {total_tasks} tasks! Review and edit as needed.")
            self.start_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            response = messagebox.askyesno("Review Results",
                                           f"Successfully extracted {total_tasks} tasks.\n\n" +
                                           "Would you like to review and edit them now?")
            if response:
                self.open_edit_window()
        else:
            self.status_label.config(text="⚠️ No tasks extracted. Please try another image or manual entry.")
            messagebox.showwarning("No Data",
                                   "Could not extract any recognizable schedule from the image.\n\n" +
                                   "Possible reasons:\n" +
                                   "• Image quality too low\n" +
                                   "• Timetable layout is too complex or non-standard\n" +
                                   "• No actual timetable data found\n\n" +
                                   "Please try:\n" +
                                   "• Using a clearer or higher resolution image\n" +
                                   "• Manually entering the schedule\n" +
                                   "• Importing from a CSV/Excel file")
    def ocr_error(self, error_msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="❌ OCR failed. Try manual entry or check error logs.")
        messagebox.showerror("OCR Error", f"Failed to process image:\n{error_msg}\n\n"
                                          "Please ensure all required OCR dependencies are installed correctly "
                                          "(PaddleOCR, Tesseract, etc.) and try again.")
    def open_edit_window(self):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Schedule")
        edit_window.geometry("800x600")
        edit_window.transient(self.root)
        edit_window.grab_set()
        tk.Label(edit_window, text="Review and edit extracted schedule (click an item to edit, or use entries to add)",
                 font=('Arial', 12, 'bold'), wraplength=750).pack(pady=10)
        notebook = ttk.Notebook(edit_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.edit_window_day_widgets = {}
        for day in days:
            frame = tk.Frame(notebook, bg='white')
            notebook.add(frame, text=day)
            list_frame = tk.Frame(frame, bg='white')
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            listbox = tk.Listbox(list_frame, font=('Consolas', 10), selectmode=tk.SINGLE,
                                 bg='#f8f8f8', fg='#333333', selectbackground='#aed6f1', selectforeground='black',
                                 highlightbackground='#cccccc', highlightthickness=1)
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
            listbox.config(yscrollcommand=scrollbar.set)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            control_frame = tk.Frame(frame, bg='white', pady=5)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            control_frame.grid_columnconfigure(1, weight=1)
            control_frame.grid_columnconfigure(3, weight=3)
            tk.Label(control_frame, text="Time:", bg='white', font=('Arial', 10)).grid(row=0, column=0, padx=5, sticky=tk.W)
            time_entry = tk.Entry(control_frame, width=10, font=('Consolas', 10), relief=tk.FLAT, bd=1, highlightbackground='#cccccc', highlightthickness=1)
            time_entry.grid(row=0, column=1, padx=5, sticky='ew')
            tk.Label(control_frame, text="Task:", bg='white', font=('Arial', 10)).grid(row=0, column=2, padx=5, sticky=tk.W)
            task_entry = tk.Entry(control_frame, width=30, font=('Arial', 10), relief=tk.FLAT, bd=1, highlightbackground='#cccccc', highlightthickness=1)
            task_entry.grid(row=0, column=3, padx=5, sticky='ew')
            self.edit_window_day_widgets[day] = {
                'listbox': listbox,
                'time_entry': time_entry,
                'task_entry': task_entry
            }
            def populate_edit_fields(event, day_key=day):
                lb = self.edit_window_day_widgets[day_key]['listbox']
                te = self.edit_window_day_widgets[day_key]['time_entry']
                tke = self.edit_window_day_widgets[day_key]['task_entry']
                selected_indices = lb.curselection()
                if selected_indices:
                    index = selected_indices[0]
                    item_text = lb.get(index)
                    parts = item_text.split(' - ', 1)
                    if len(parts) == 2:
                        te.delete(0, tk.END)
                        te.insert(0, parts[0])
                        tke.delete(0, tk.END)
                        tke.insert(0, parts[1])
            listbox.bind("<<ListboxSelect>>", populate_edit_fields)
            def add_update_task(d=day):
                te = self.edit_window_day_widgets[d]['time_entry']
                tke = self.edit_window_day_widgets[d]['task_entry']
                lb = self.edit_window_day_widgets[d]['listbox']
                time_val = te.get().strip()
                task_val = tke.get().strip()
                if not time_val or not task_val:
                    messagebox.showwarning("Missing Information", "Please enter both Time and Task.")
                    return
                normalized_time = self.normalize_time(time_val)
                if not normalized_time:
                    messagebox.showwarning("Invalid Time Format", "Please enter time in HH:MM or H AM/PM format (e.g., 09:00, 3 PM).")
                    return
                selected_indices = lb.curselection()
                if selected_indices:
                    index_to_update = selected_indices[0]
                    original_item_text = lb.get(index_to_update)
                    original_time, original_task = original_item_text.split(' - ', 1)
                    normalized_original_time = self.normalize_time(original_time)
                    if d in self.schedule_data:
                        self.schedule_data[d] = [
                            item for item in self.schedule_data[d]
                            if not (item['time'] == normalized_original_time and item['task'] == original_task)
                        ]
                    self.schedule_data[d].append({'time': normalized_time, 'task': task_val})
                    messagebox.showinfo("Task Updated", "Task has been updated successfully!")
                else:
                    if d not in self.schedule_data:
                        self.schedule_data[d] = []
                    self.schedule_data[d].append({'time': normalized_time, 'task': task_val})
                    messagebox.showinfo("Task Added", "Task has been added successfully!")
                self.populate_listbox_for_day(lb, d)
                te.delete(0, tk.END)
                tke.delete(0, tk.END)
                lb.selection_clear(0, tk.END)
            def delete_task(d=day):
                lb = self.edit_window_day_widgets[d]['listbox']
                te = self.edit_window_day_widgets[d]['time_entry']
                tke = self.edit_window_day_widgets[d]['task_entry']
                selected_indices = lb.curselection()
                if not selected_indices:
                    messagebox.showwarning("No Selection", "Please select a task to delete.")
                    return
                index_to_delete = selected_indices[0]
                item_text = lb.get(index_to_delete)
                time_to_delete, task_to_delete = item_text.split(' - ', 1)
                normalized_time_to_delete = self.normalize_time(time_to_delete)
                if d in self.schedule_data:
                    self.schedule_data[d] = [
                        item for item in self.schedule_data[d]
                        if not (item['time'] == normalized_time_to_delete and item['task'] == task_to_delete)
                    ]
                self.populate_listbox_for_day(lb, d)
                messagebox.showinfo("Task Deleted", "Task has been deleted successfully!")
                te.delete(0, tk.END)
                tke.delete(0, tk.END)
                lb.selection_clear(0, tk.END)
            tk.Button(control_frame, text="Add/Update Task", command=add_update_task,
                      bg='#27ae60', fg='white', font=('Arial', 10, 'bold'), cursor='hand2',
                      activebackground='#2ecc71', activeforeground='white').grid(row=0, column=4, padx=5, sticky=tk.W)
            tk.Button(control_frame, text="Delete Task", command=delete_task,
                      bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'), cursor='hand2',
                      activebackground='#c0392b', activeforeground='white').grid(row=0, column=5, padx=5, sticky=tk.W)
            self.populate_listbox_for_day(listbox, day)
        def save_edits_and_close():
            self.display_schedule()
            edit_window.destroy()
            if any(self.schedule_data.values()):
                self.start_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.NORMAL)
                self.status_label.config(text="✅ Schedule updated. Ready to start reminders!")
            else:
                self.start_btn.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.DISABLED)
                self.status_label.config(text="No schedule found. Please add tasks to activate reminders.")
        tk.Button(edit_window, text="💾 Save Changes & Close", command=save_edits_and_close,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  padx=20, pady=10, cursor='hand2', activebackground='#2980b9', activeforeground='white').pack(pady=10)
        self.root.wait_window(edit_window)
    def populate_listbox_for_day(self, listbox, day):
        listbox.delete(0, tk.END)
        if day in self.schedule_data:
            sorted_tasks = sorted(self.schedule_data[day], key=lambda x: datetime.strptime(x['time'], '%H:%M'))
            for task_item in sorted_tasks:
                listbox.insert(tk.END, f"{task_item['time']} - {task_item['task']}")
    def open_manual_entry(self):
        entry_window = tk.Toplevel(self.root)
        entry_window.title("Manual Schedule Entry")
        entry_window.geometry("1200x700")
        entry_window.transient(self.root)
        entry_window.grab_set()
        inst_frame = tk.Frame(entry_window, bg='#3498db', height=50)
        inst_frame.pack(fill=tk.X)
        inst_frame.pack_propagate(False)
        tk.Label(inst_frame, text="Enter your schedule in the grid below (Time in HH:MM format)",
                 font=('Arial', 14, 'bold'), fg='white', bg='#3498db').pack(pady=10)
        main_frame = tk.Frame(entry_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(main_frame, bg='white', highlightbackground='#dddddd')
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        days_headers = ['Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.manual_grid_entries = {}
        self.manual_time_widgets = []
        for col_idx, day_name in enumerate(days_headers):
            label = tk.Label(scrollable_frame, text=day_name, font=('Arial', 11, 'bold'),
                             bg='#34495e', fg='white', relief=tk.RAISED, bd=1,
                             width=15, padx=5, pady=8)
            label.grid(row=0, column=col_idx, sticky='nsew')
        existing_times_set = set()
        for day_tasks in self.schedule_data.values():
            for task in day_tasks:
                existing_times_set.add(task['time'])
        existing_times_sorted = sorted(list(existing_times_set))
        all_times_to_display = sorted(list(set(existing_times_sorted + [
            "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00",
            "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00",
            "20:00", "21:00", "22:00", "23:00"
        ])))
        for time_val in all_times_to_display:
            row_data = {}
            for day_name_full in days_headers[1:]:
                if day_name_full in self.schedule_data:
                    tasks_at_this_time = [t['task'] for t in self.schedule_data[day_name_full] if t['time'] == time_val]
                    if tasks_at_this_time:
                        row_data[day_name_full] = ", ".join(tasks_at_this_time)
            self.add_manual_row(scrollable_frame, time_val, days_headers, self.manual_time_widgets, self.manual_grid_entries, populate_data=row_data)
        canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        scrollable_frame.grid_columnconfigure(0, weight=0)
        for col_idx in range(1, len(days_headers)):
            scrollable_frame.grid_columnconfigure(col_idx, weight=1)
        btn_frame = tk.Frame(entry_window, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Button(btn_frame, text="➕ Add New Time Slot",
                  command=lambda: self.add_manual_row(scrollable_frame, "", days_headers, self.manual_time_widgets, self.manual_grid_entries),
                  bg='#3498db', fg='white', font=('Arial', 10), cursor='hand2',
                  activebackground='#2980b9', activeforeground='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="💾 Save Schedule",
                  command=lambda: self.save_manual_entry(entry_window),
                  bg='#27ae60', fg='white', font=('Arial', 10, 'bold'), cursor='hand2',
                  activebackground='#2ecc71', activeforeground='white', padx=15, pady=5).pack(side=tk.RIGHT, padx=5)
        entry_window.wait_window(entry_window)
    def add_manual_row(self, parent_frame, initial_time, days_headers, time_widgets_list, grid_entries_dict, populate_data={}):
        current_row_idx = len(time_widgets_list) + 1
        time_entry_widget = tk.Entry(parent_frame, font=('Arial', 10, 'bold'),
                                     bg='#ecf0f1', width=15, relief=tk.FLAT, bd=1, highlightbackground='#cccccc')
        if initial_time:
            time_entry_widget.insert(0, initial_time)
        time_entry_widget.grid(row=current_row_idx, column=0, sticky='nsew', pady=1, padx=1)
        time_widgets_list.append(time_entry_widget)
        for col_offset, day_name in enumerate(days_headers[1:]):
            col_display_idx = col_offset + 1
            entry = tk.Entry(parent_frame, font=('Arial', 10), width=20, relief=tk.FLAT, bd=1, highlightbackground='#cccccc')
            if day_name in populate_data:
                entry.insert(0, populate_data[day_name])
            entry.grid(row=current_row_idx, column=col_display_idx, sticky='nsew', padx=1, pady=1)
            grid_entries_dict[(current_row_idx, col_display_idx)] = entry
        parent_frame.update_idletasks()
        parent_frame.master.config(scrollregion=parent_frame.master.bbox("all"))
    def save_manual_entry(self, entry_window):
        new_schedule_data = {day: [] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for row_display_idx, time_widget in enumerate(self.manual_time_widgets):
            time_text = time_widget.get().strip()
            if not time_text:
                continue
            normalized_time = self.normalize_time(time_text)
            if not normalized_time:
                messagebox.showwarning("Invalid Time Input", f"Could not understand time '{time_text}' in row {row_display_idx+1}. Please use HH:MM format (e.g., 09:00, 14:30) or H AM/PM (e.g., 3 PM).")
                continue
            for col_offset, day_name in enumerate(days_list):
                col_display_idx = col_offset + 1
                if (row_display_idx + 1, col_display_idx) in self.manual_grid_entries:
                    task_entry_widget = self.manual_grid_entries[(row_display_idx + 1, col_display_idx)]
                    task = task_entry_widget.get().strip()
                    if task:
                        new_schedule_data[day_name].append({
                            'time': normalized_time,
                            'task': task
                        })
        for day in new_schedule_data:
            new_schedule_data[day].sort(key=lambda x: datetime.strptime(x['time'], '%H:%M'))
        self.schedule_data = new_schedule_data
        self.display_schedule()
        if any(self.schedule_data.values()):
            self.status_label.config(text="✅ Schedule entered successfully! Ready to activate reminders.")
            self.start_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            entry_window.destroy()
            messagebox.showinfo("Schedule Saved", "Your timetable has been saved successfully!")
        else:
            messagebox.showwarning("No Tasks Saved", "No valid tasks were saved. Please ensure you entered valid times and tasks.")
    def import_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select Timetable File",
            filetypes=[("Spreadsheet files", "*.csv *.xls *.xlsx")],
            initialdir=os.path.expanduser("~")
        )
        if not file_path:
            return
        self.status_label.config(text="🔄 Importing data from file...")
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            new_schedule_data = {day: [] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
            all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if 'Time' not in df.columns:
                raise ValueError("The imported file must contain a 'Time' column.")
            day_cols_in_df = []
            for col in df.columns:
                for d in all_days:
                    if col.lower() == d.lower() or col.lower() == d[:3].lower():
                        day_cols_in_df.append((col, d))
                        break
            if not day_cols_in_df:
                raise ValueError("No recognizable day columns (e.g., 'Monday', 'Tuesday' or 'Mon', 'Tue') found in the file.")
            for index, row in df.iterrows():
                time_val = str(row['Time']).strip()
                normalized_time = self.normalize_time(time_val)
                if not normalized_time:
                    print(f"Skipping row {index+1} due to invalid time format: '{time_val}'")
                    continue
                for col_name_in_df, normalized_day_name in day_cols_in_df:
                    task = str(row[col_name_in_df]).strip()
                    if task and task.lower() != 'nan' and task.lower() != 'none':
                        new_schedule_data[normalized_day_name].append({
                            'time': normalized_time,
                            'task': task
                        })
            for day in new_schedule_data:
                new_schedule_data[day].sort(key=lambda x: datetime.strptime(x['time'], '%H:%M'))
            self.schedule_data = new_schedule_data
            self.display_schedule()
            if any(self.schedule_data.values()):
                self.status_label.config(text="✅ Schedule imported successfully!")
                self.start_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.NORMAL)
                messagebox.showinfo("Import Success", "Timetable imported successfully! Please review the schedule and activate reminders.")
            else:
                self.status_label.config(text="⚠️ No valid tasks found in the imported file.")
                messagebox.showwarning("No Data", "No valid timetable entries could be extracted from the file. "
                                          "Please check the file format (needs a 'Time' column and valid Day columns like 'Monday', 'Tuesday').")
        except Exception as e:
            self.status_label.config(text="❌ Import failed.")
            messagebox.showerror("Import Error", f"Failed to import file:\n{e}\n\n"
                                              "Ensure the file has a 'Time' column and columns for days of the week (e.g., 'Monday', 'Tuesday').")
    def display_schedule(self):
        self.schedule_text.config(state=tk.NORMAL)
        self.schedule_text.delete(1.0, tk.END)
        if not self.schedule_data or all(not val for val in self.schedule_data.values()):
            self.display_welcome_message()
            self.start_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            return
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.schedule_text.insert(tk.END, "Your Current Schedule:\n\n", "header")
        for day in days_order:
            if day in self.schedule_data and self.schedule_data[day]:
                tag = "current_day" if day == self.current_day else "day"
                self.schedule_text.insert(tk.END, f"--- {day.upper()} ---\n", tag)
                for entry in self.schedule_data[day]:
                    self.schedule_text.insert(tk.END, f"{entry['time']} - ", "time")
                    self.schedule_text.insert(tk.END, f"{entry['task']}\n", "task")
                self.schedule_text.insert(tk.END, "\n")
            else:
                tag = "current_day" if day == self.current_day else "day"
                self.schedule_text.insert(tk.END, f"--- {day.upper()} --- (No tasks scheduled)\n\n", tag)
        self.schedule_text.config(state=tk.DISABLED)
    def toggle_reminders(self):
        if not self.reminders_active:
            if not self.schedule_data or all(not val for val in self.schedule_data.values()):
                messagebox.showwarning("No Schedule", "Please input a schedule (Manual Entry, CSV, or OCR) before starting reminders.")
                return
            self.reminders_active = True
            self.start_btn.config(text="⏸️ Stop Reminders", bg='#f39c12', activebackground='#e67e22')
            self.status_label.config(text="✅ Reminders active. Checking schedule...")
            schedule.clear()
            self.setup_scheduler()
            self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self.scheduler_thread.start()
            messagebox.showinfo("Reminders Started", "Timetable reminders have been activated!\nNotifications will appear and voice alerts will play (if enabled) at scheduled times.")
        else:
            self.reminders_active = False
            self.start_btn.config(text="▶️ Start Reminders", bg='#e74c3c', activebackground='#c0392b')
            self.status_label.config(text="🛑 Reminders stopped.")
            schedule.clear()
            messagebox.showinfo("Reminders Stopped", "Timetable reminders have been paused.")
    def setup_scheduler(self):
        days_map = {
            'Monday': schedule.every().monday,
            'Tuesday': schedule.every().tuesday,
            'Wednesday': schedule.every().wednesday,
            'Thursday': schedule.every().thursday,
            'Friday': schedule.every().friday,
            'Saturday': schedule.every().saturday,
            'Sunday': schedule.every().sunday,
        }
        for day, tasks in self.schedule_data.items():
            if day in days_map:
                for task_entry in tasks:
                    time_to_remind = task_entry['time']
                    task_description = task_entry['task']
                    try:
                        days_map[day].at(time_to_remind).do(self.send_reminder, day, time_to_remind, task_description)
                        print(f"Successfully scheduled: '{task_description}' on {day} at {time_to_remind}")
                    except Exception as e:
                        print(f"Error scheduling task '{task_description}' for {day} at {time_to_remind}: {e}")
                        self.root.after(0, lambda d=day, t=time_to_remind: messagebox.showwarning("Scheduling Error", f"Could not schedule task for {d} at {t}. Please check time format in your schedule."))
    def run_scheduler(self):
        print("Scheduler thread started.")
        while self.reminders_active:
            try:
                schedule.run_pending()
            except Exception as e:
                print(f"Error occurred in scheduler run_pending: {e}")
            time.sleep(1)
        print("Scheduler thread stopped.")
    def send_reminder(self, day, time_str, task_description):
        title = "Timetable Reminder"
        message = f"It's {time_str} on {day}! Time for: {task_description}"
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="Smart Timetable Reminder",
                timeout=12
            )
            print(f"Desktop notification sent: {message}")
        except Exception as e:
            print(f"Error sending desktop notification: {e}. Ensure 'plyer' is installed and your system supports it (e.g., notify-send on Linux).")
        if self.tts_enabled.get() and self.tts_available:
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
                print(f"Voice alert played: {message}")
            except Exception as e:
                print(f"Error playing voice alert: {e}. Ensure 'pyttsx3' is correctly installed and configured.")
        elif not self.tts_enabled.get():
            print("Voice alerts are disabled.")
        elif not self.tts_available:
            print("Voice alerts not available due to initialization error.")
    def test_notification(self):
        test_message = "This is a test reminder from your Smart Timetable App! Everything seems to be working."
        title = "Smart Timetable Reminder Test"
        try:
            notification.notify(
                title=title,
                message=test_message,
                app_name="Smart Timetable Reminder",
                timeout=7
            )
            print(f"Test desktop notification sent: {test_message}")
        except Exception as e:
            messagebox.showerror("Notification Test Error", f"Failed to send test desktop notification. Ensure 'plyer' is installed correctly and your system supports desktop notifications. Error: {e}")
        if self.tts_enabled.get() and self.tts_available:
            try:
                self.tts_engine.say(test_message)
                self.tts_engine.runAndWait()
                print(f"Test voice alert played: {test_message}")
            except Exception as e:
                messagebox.showerror("Voice Alert Test Error", f"Failed to play test voice alert. Ensure 'pyttsx3' is installed correctly and its dependencies (e.g., espeak on Linux) are met. Error: {e}")
        elif not self.tts_available:
            messagebox.showinfo("Test Notification Info", "Desktop notification sent. Voice alerts are not available due to an initialization error.")
        else:
            messagebox.showinfo("Test Notification Info", "Desktop notification sent. Voice alerts are currently disabled. You can enable them in the 'Controls' tab.")
    def save_schedule(self):
        if not self.schedule_data or all(not val for val in self.schedule_data.values()):
            messagebox.showwarning("No Data to Save", "There is no schedule data entered to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Schedule As",
            initialdir=os.path.expanduser("~")
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.schedule_data, f, indent=4)
                self.status_label.config(text="✅ Schedule saved successfully!")
                messagebox.showinfo("Save Success", f"Your schedule has been saved to:\n{file_path}")
            except Exception as e:
                self.status_label.config(text="❌ Save failed.")
                messagebox.showerror("Save Error", f"Failed to save schedule to file:\n{e}")
    def load_schedule(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Schedule From",
            initialdir=os.path.expanduser("~")
        )
        if not file_path:
            return
        self.status_label.config(text="🔄 Loading schedule from file...")
        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict) or not all(isinstance(v, list) for v in loaded_data.values()):
                raise ValueError("Invalid schedule file format. Expected a dictionary of lists.")
            for day, tasks in loaded_data.items():
                if not isinstance(tasks, list):
                    print(f"Warning: Data for day '{day}' is not a list. Skipping.")
                    loaded_data[day] = []
                    continue
                cleaned_tasks = []
                for task in tasks:
                    if isinstance(task, dict) and 'time' in task and 'task' in task:
                        task['time'] = self.normalize_time(task['time'])
                        if task['time']:
                            cleaned_tasks.append(task)
                        else:
                            print(f"Warning: Skipping invalid time format in loaded task: {task}")
                    else:
                        print(f"Warning: Skipping invalid task format in loaded data: {task}")
                cleaned_tasks.sort(key=lambda x: datetime.strptime(x['time'], '%H:%M'))
                loaded_data[day] = cleaned_tasks
            self.schedule_data = loaded_data
            self.display_schedule()
            self.status_label.config(text="✅ Schedule loaded successfully!")
            self.start_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Load Success", f"Schedule loaded successfully from:\n{file_path}\n\nReview the schedule and activate reminders.")
        except json.JSONDecodeError:
            self.status_label.config(text="❌ Load failed.")
            messagebox.showerror("Load Error", f"Failed to load schedule: Invalid JSON file format.\nCheck if the file is a valid JSON.")
        except Exception as e:
            self.status_label.config(text="❌ Load failed.")
            messagebox.showerror("Load Error", f"Failed to load schedule from file:\n{e}")
    def run(self):
        self.root.mainloop()
if __name__ == "__main__":
    app = TimetableReminder()
    app.run()
