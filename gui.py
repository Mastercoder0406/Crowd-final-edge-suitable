import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import time
import platform
import logging
from crowd_analysis import CrowdAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrowdGUI")

class CrowdAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Analysis")
        
        # Device detection
        self.is_edge_device = self.detect_edge_device()
        
        # Set window size based on device
        if self.is_edge_device:
            self.root.geometry("800x480")  # Smaller for RPi
            logger.info("Running in Raspberry Pi optimized mode")
        else:
            self.root.geometry("1200x800")  # Original size
        
        # Variables
        self.source_type = tk.StringVar(value="video")
        self.file_path = tk.StringVar()
        self.running = False
        self.cap = None
        self.analyzer = CrowdAnalyzer(edge_mode=self.is_edge_device)
        self.frame_counter = 0
        self.imgtk = None  # To prevent garbage collection
        
        # GUI Setup
        self.setup_ui()

    def detect_edge_device(self):
        """Check if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    return True
            return 'arm' in platform.machine().lower()
        except:
            return False

    def setup_ui(self):
        """Setup the UI elements"""
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Device status
        device_status = "🟢 Edge Mode" if self.is_edge_device else "🔴 Desktop Mode"
        ttk.Label(control_frame, text=device_status,
                 foreground="green" if self.is_edge_device else "red").grid(row=0, column=3, padx=10)
        
        # Source selection
        ttk.Label(control_frame, text="Source:").grid(row=0, column=0)
        ttk.Radiobutton(control_frame, text="Video File", variable=self.source_type, value="video").grid(row=0, column=1)
        ttk.Radiobutton(control_frame, text="Camera", variable=self.source_type, value="camera").grid(row=0, column=2)
        
        # File browse
        ttk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=1, column=0)
        ttk.Entry(control_frame, textvariable=self.file_path, width=40).grid(row=1, column=1, columnspan=2)
        
        # Control buttons
        ttk.Button(control_frame, text="Start", command=self.start_analysis).grid(row=2, column=0, pady=5)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=2, column=1)
        ttk.Button(control_frame, text="Results", command=self.show_results).grid(row=2, column=2)
        
        # Video display
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # People counter overlay
        self.count_label = ttk.Label(
            self.video_frame,
            text="People: 0",
            font=('Arial', 24, 'bold'),
            foreground="red",
            background="black"
        )
        self.count_label.place(relx=0.95, rely=0.05, anchor=tk.NE)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var,
                 relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def browse_file(self):
        """Open file dialog to select a video file"""
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f"Selected: {filename.split('/')[-1]}")

    def start_analysis(self):
        """Start processing video"""
        if self.source_type.get() == "camera":
            source = 0  # Default camera
        else:
            source = self.file_path.get()
            if not source:
                self.status_var.set("Error: No video file selected!")
                return

        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError("Could not open video source")
                
            self.running = True
            self.frame_counter = 0
            self.status_var.set("Running...")
            self.update_frame()
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            logger.error(f"Failed to start analysis: {e}")

    def update_frame(self):
        """Process and display each frame"""
        if not self.running:
            return
            
        start_time = time.time()
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            self.status_var.set("Finished processing video")
            return
            
        # Edge optimization: Reduce resolution
        if self.is_edge_device:
            frame = cv2.resize(frame, (640, 480))
            
        # Process frame
        processed_frame, count, boxes, anomalies = self.analyzer.process_frame(frame)
        
        # Draw bounding boxes (green rectangles)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Update display
        self.update_display(processed_frame, count, start_time)
        
        # Schedule next frame
        self.root.after(30, self.update_frame)

    def update_display(self, frame, count, start_time):
        """Update the GUI with processed frame"""
        # Convert frame for display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Update PhotoImage
        if self.imgtk is None:
            self.imgtk = ImageTk.PhotoImage(image=img)
        else:
            self.imgtk.paste(img)
            
        self.video_label.config(image=self.imgtk)
        self.count_label.config(text=f"People: {count}")
        
        # Update status
        fps = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        self.status_var.set(
            f"People: {count} | FPS: {fps:.1f} | "
            f"Frame: {self.frame_counter} | "
            f"{'Edge Mode' if self.is_edge_device else 'Desktop Mode'}"
        )

    def stop(self):
        """Stop analysis and cleanup"""
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'imgtk'):
            self.video_label.config(image="")
        self.status_var.set("Stopped")
        self.analyzer.cleanup()

    def show_results(self):
        """Show analysis results"""
        if not hasattr(self.analyzer, "people_counts") or not self.analyzer.people_counts:
            self.status_var.set("No data available. Run analysis first!")
            return
            
        result_window = tk.Toplevel(self.root)
        result_window.title("Analysis Results")
        result_window.geometry("700x550")

        # Create plots
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        
        # People count plot
        axs[0].plot(self.analyzer.people_counts, color="blue", marker="o")
        axs[0].set_title("People Count Over Time")
        axs[0].grid()
        
        # Anomalies plot
        axs[1].plot(self.analyzer.anomalies_log, color="red", marker="s")
        axs[1].set_title("Anomalies Over Time")
        axs[1].grid()
        
        # Processing time plot
        axs[2].plot(self.analyzer.processing_times, color="green", marker="^")
        axs[2].set_title("Processing Time (ms)")
        axs[2].grid()

        # Embed plots
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def __del__(self):
        """Ensure cleanup"""
        self.stop()