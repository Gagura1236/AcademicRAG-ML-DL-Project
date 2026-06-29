import tkinter as tk
from tkinter import scrolledtext, font
import subprocess
import threading
import sys
import os
import webbrowser
import time
import urllib.request
import urllib.error

# Server connection parameters
PORT = 8501
URL = f"http://localhost:{PORT}"

class AppLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Academic Reading Assistant - Launcher")
        self.geometry("600x450")
        self.configure(bg="#1E2A22") # Deep forest green background
        
        # State
        self.server_process = None
        self.is_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Fonts
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.normal_font = font.Font(family="Helvetica", size=12)
        
        # Colors
        self.bg_color = "#1E2A22"
        self.fg_color = "#E5F0E6" # Soft green-white
        self.accent_color = "#D4AF37" # Gold accent for nature theme
        self.button_bg = "#2E4A35"
        self.button_active = "#3E6649"
        
        # Main Frame
        main_frame = tk.Frame(self, bg=self.bg_color, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_label = tk.Label(main_frame, text="🌿 Academic Reading Assistant", 
                               font=self.title_font, bg=self.bg_color, fg=self.accent_color)
        header_label.pack(pady=(0, 20))
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Offline 🔴")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               font=self.normal_font, bg=self.bg_color, fg=self.fg_color)
        status_label.pack(pady=(0, 20))
        
        # Buttons Frame
        btn_frame = tk.Frame(main_frame, bg=self.bg_color)
        btn_frame.pack(pady=(0, 20))
        
        self.start_btn = tk.Button(btn_frame, text="Start Server / 啟動助手", 
                                  command=self.toggle_server, 
                                  bg="white", fg="black", 
                                  activebackground="#E5E5E5", activeforeground="black",
                                  highlightbackground="#1E2A22", highlightthickness=2,
                                  font=self.normal_font, width=20, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.browser_btn = tk.Button(btn_frame, text="Open Browser / 開啟網頁", 
                                    command=self.open_browser, 
                                    bg="white", fg="black",
                                    activebackground="#E5E5E5", activeforeground="black",
                                    highlightbackground="#1E2A22", highlightthickness=2,
                                    font=self.normal_font, width=20, pady=5, state=tk.DISABLED)
        self.browser_btn.pack(side=tk.LEFT, padx=10)
        
        # Log Display
        log_label = tk.Label(main_frame, text="Server Logs:", bg=self.bg_color, fg=self.fg_color, anchor="w")
        log_label.pack(fill=tk.X)
        
        self.log_area = scrolledtext.ScrolledText(main_frame, height=10, bg="#0D1410", fg="#A3C2A6", 
                                                 font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.configure(state='disabled')
        
    def log_message(self, message):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message)
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def check_server_health(self):
        try:
            req = urllib.request.Request(f"{URL}/_stcore/health")
            with urllib.request.urlopen(req, timeout=1) as response:
                return response.status == 200
        except Exception:
            return False

    def toggle_server(self):
        if self.is_running:
            self.stop_server()
        else:
            self.start_server()
            
    def start_server(self):
        self.log_message("Starting Streamlit Server...\n")
        self.status_var.set("Status: Starting... 🟡")
        self.start_btn.config(text="Stop Server / 停止助手")
        self.is_running = True
        self.browser_opened = False
        
        # Start server process in a separate thread
        self.server_thread = threading.Thread(target=self.run_server_process, daemon=True)
        self.server_thread.start()
        
        # Start health check loop
        self.check_health_loop()

    def stop_server(self):
        self.log_message("Stopping Streamlit Server...\n")
        self.is_running = False
        self.start_btn.config(text="Start Server / 啟動助手")
        self.browser_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Offline 🔴")
        
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            self.log_message("Server stopped.\n")
            
    def run_server_process(self):
        # Assuming app.py is in the same directory as this launcher
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "app.py")
        
        # Create environment with unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Use miniforge3 Python (which has mlx_lm and all packages installed)
        # Prefer miniforge3 explicitly, fall back to sys.executable
        miniforge_python = "/Users/gagura/miniforge3/bin/python3"
        if os.path.isfile(miniforge_python):
            python_exec = miniforge_python
        else:
            python_exec = sys.executable
        self.log_message(f"Using Python: {python_exec}\n")
        
        try:
            self.server_process = subprocess.Popen(
                [python_exec, "-m", "streamlit", "run", app_path, "--server.port", str(PORT), "--server.headless", "true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                cwd=script_dir
            )
            
            # Read logs line by line
            for line in self.server_process.stdout:
                self.after(0, self.log_message, line)
                if not self.is_running:
                    break
                    
        except Exception as e:
            self.after(0, self.log_message, f"Error starting server: {e}\n")
            self.after(0, self.stop_server)

    def check_health_loop(self):
        if not self.is_running:
            return
            
        if self.check_server_health():
            self.status_var.set("Status: Online 🟢")
            self.browser_btn.config(state=tk.NORMAL)
            if not getattr(self, 'browser_opened', False):
                self.open_browser()
                self.browser_opened = True
            
            # Keep checking but less frequently
            self.after(5000, self.check_health_loop)
        else:
            self.status_var.set("Status: Starting... 🟡")
            self.after(1000, self.check_health_loop) # Check again in 1s
            
    def open_browser(self):
        self.log_message(f"Opening browser at {URL}...\n")
        webbrowser.open(URL)
        
    def destroy(self):
        # Ensure process is killed on close
        self.stop_server()
        super().destroy()

if __name__ == "__main__":
    app = AppLauncher()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.mainloop()
