import sys
import os
import time
import threading
import logging
import queue
import gc
import hashlib
import json
import atexit
import signal
import warnings
import traceback
import subprocess
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONNX_RUNTIME_EXECUTION_MODE"] = "sequential"

try:
    import customtkinter as ctk
    from customtkinter import CTk, CTkFrame, CTkButton, CTkLabel, CTkProgressBar, CTkOptionMenu, CTkEntry, CTkScrollableFrame, CTkTextbox, CTkSwitch, CTkSlider
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    import tkinter as tk
    from tkinter import ttk

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance, ImageOps
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from rembg import remove, new_session, sessions
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import restoration, filters, morphology, segmentation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

log_queue = queue.Queue()
_logger_initialized = False
_root_logger = None

class SystemLogger:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._setup_logging()
        atexit.register(self._cleanup)
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
    
    def _setup_logging(self):
        global _root_logger
        self.logger = logging.getLogger("BGRemoverPro")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[{asctime}] [{levelname}]: {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        _root_logger = self.logger
        global _logger_initialized
        _logger_initialized = True
        self.log("System logger initialized (console only)", "INFO")
    
    def log(self, message: str, level: str = "INFO"):
        level_upper = level.upper()
        if level_upper == "DEBUG":
            self.logger.debug(message)
        elif level_upper == "INFO":
            self.logger.info(message)
        elif level_upper == "WARNING":
            self.logger.warning(message)
        elif level_upper == "ERROR":
            self.logger.error(message)
        elif level_upper == "CRITICAL":
            self.logger.critical(message)
        else:
            self.logger.info(message)
        log_queue.put((message, level_upper))
    
    def _flush_worker(self):
        while True:
            time.sleep(0.1)
            try:
                while not log_queue.empty():
                    log_queue.get_nowait()
            except queue.Empty:
                pass
    
    def _cleanup(self):
        self.log("System logger shutting down", "INFO")
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()


system_logger = SystemLogger()
LOG = system_logger.log

LOG("Application bootstrap initiated", "INFO")

def _check_and_install_dependencies():
    missing_pkgs = []
    if not PIL_AVAILABLE:
        missing_pkgs.append("Pillow")
    if not REMBG_AVAILABLE:
        missing_pkgs.append("rembg[cpu]")
    if not CV2_AVAILABLE:
        missing_pkgs.append("opencv-python")
    if not SCIPY_AVAILABLE:
        missing_pkgs.append("scipy")
    if not SKIMAGE_AVAILABLE:
        missing_pkgs.append("scikit-image")
    if not CTK_AVAILABLE:
        missing_pkgs.append("customtkinter")
    
    if missing_pkgs:
        LOG(f"Missing dependencies detected: {', '.join(missing_pkgs)}", "WARNING")
        return False
    return True

DEPENDENCIES_OK = _check_and_install_dependencies()
if not DEPENDENCIES_OK:
    LOG("Some dependencies are missing. Limited functionality will be available.", "WARNING")

MODEL_REGISTRY = {
    "birefnet-general": {
        "name": "BiRefNet General",
        "size_mb": 98,
        "description": "State-of-the-art general purpose. Excellent for hair, glass, complex edges.",
        "source": "ZhengPeng7/BiRefNet",
        "type": "birefnet"
    },
    "birefnet-portrait": {
        "name": "BiRefNet Portrait",
        "size_mb": 98,
        "description": "Optimized for human portraits. Superior hair and skin edge detection.",
        "source": "ZhengPeng7/BiRefNet",
        "type": "birefnet"
    },
    "birefnet-hrsod": {
        "name": "BiRefNet High-Res",
        "size_mb": 98,
        "description": "High-resolution salient object detection. Best for detailed objects.",
        "source": "ZhengPeng7/BiRefNet",
        "type": "birefnet"
    },
    "birefnet-massive": {
        "name": "BiRefNet Massive",
        "size_mb": 98,
        "description": "Trained on massive dataset. Handles complex scenes exceptionally well.",
        "source": "ZhengPeng7/BiRefNet",
        "type": "birefnet"
    },
    "birefnet-dis": {
        "name": "BiRefNet DIS",
        "size_mb": 98,
        "description": "Dichotomous image segmentation. Sharp binary separation.",
        "source": "ZhengPeng7/BiRefNet",
        "type": "birefnet"
    },
    "isnet-general-use": {
        "name": "ISNet General",
        "size_mb": 102,
        "description": "High-quality general segmentation. Good balance of speed and accuracy.",
        "source": "rembg",
        "type": "isnet"
    },
    "isnet-anime": {
        "name": "ISNet Anime",
        "size_mb": 102,
        "description": "Optimized for anime/illustration style images.",
        "source": "rembg",
        "type": "isnet"
    },
    "u2net": {
        "name": "U2Net",
        "size_mb": 176,
        "description": "Classic robust model. Reliable for most scenarios.",
        "source": "rembg",
        "type": "u2net"
    },
    "u2net_human_seg": {
        "name": "U2Net Human Segmentation",
        "size_mb": 176,
        "description": "Specialized for human figure segmentation.",
        "source": "rembg",
        "type": "u2net"
    },
    "silueta": {
        "name": "Silueta",
        "size_mb": 43,
        "description": "Lightweight and fast. Good for simple subjects.",
        "source": "rembg",
        "type": "u2net"
    }
}

def _safe_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

def _get_model_cache_dir() -> Path:
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / "model"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def _compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

class ImagePreprocessor:
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        try:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            return image
        except Exception as e:
            LOG(f"Image enhancement failed: {e}", "WARNING")
            return image
    
    @staticmethod
    def denoise_image(image: Image.Image) -> Image.Image:
        if not CV2_AVAILABLE:
            return image
        try:
            img_array = np.array(image.convert('RGB'))
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 3, 3, 7, 21)
            return Image.fromarray(denoised)
        except Exception as e:
            LOG(f"Denoising failed: {e}", "WARNING")
            return image
    
    @staticmethod
    def auto_orient(image: Image.Image) -> Image.Image:
        try:
            return ImageOps.exif_transpose(image)
        except Exception:
            return image

class EdgeRefiner:
    @staticmethod
    def refine_alpha(alpha: Image.Image, original: Image.Image, iterations: int = 3) -> Image.Image:
        if not CV2_AVAILABLE:
            return alpha
        try:
            alpha_np = np.array(alpha.convert('L'))
            original_np = np.array(original.convert('RGB'))
            alpha_float = alpha_np.astype(np.float32) / 255.0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for _ in range(iterations):
                alpha_float = cv2.medianBlur(alpha_float, 3)
                alpha_float = cv2.GaussianBlur(alpha_float, (3, 3), 0.5)
                gradient_x = cv2.Sobel(alpha_float, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(alpha_float, cv2.CV_32F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                alpha_float = alpha_float * (1 + 0.2 * gradient_magnitude)
                alpha_float = np.clip(alpha_float, 0.0, 1.0)
                eroded = cv2.erode(alpha_float, kernel, iterations=1)
                dilated = cv2.dilate(alpha_float, kernel, iterations=1)
                alpha_float = (alpha_float + eroded + dilated) / 3.0
            alpha_refined = (alpha_float * 255).astype(np.uint8)
            return Image.fromarray(alpha_refined, mode='L')
        except Exception as e:
            LOG(f"Edge refinement failed: {e}", "WARNING")
            return alpha
    
    @staticmethod
    def matting_refinement(image: Image.Image, alpha: Image.Image) -> Image.Image:
        if not CV2_AVAILABLE:
            return alpha
        try:
            img_np = np.array(image.convert('RGB'))
            alpha_np = np.array(alpha.convert('L')).astype(np.float32) / 255.0
            trimap = np.zeros_like(alpha_np, dtype=np.uint8)
            trimap[alpha_np < 0.1] = 0
            trimap[alpha_np > 0.9] = 255
            trimap[(alpha_np >= 0.1) & (alpha_np <= 0.9)] = 128
            alpha_refined = cv2.ximgproc.createGuidedFilter(img_np, 13, 1e-6).filter(alpha_np)
            alpha_refined = np.clip(alpha_refined, 0, 1)
            alpha_refined = (alpha_refined * 255).astype(np.uint8)
            return Image.fromarray(alpha_refined, mode='L')
        except Exception as e:
            LOG(f"Matting refinement failed: {e}", "WARNING")
            return alpha

class BackgroundRemoverEngine:
    def __init__(self):
        self._session = None
        self._current_model = None
        self._model_lock = threading.Lock()
        self._cache_dir = _get_model_cache_dir()
        LOG("BackgroundRemoverEngine initialized", "INFO")
    
    def load_model(self, model_name: str) -> bool:
        with self._model_lock:
            try:
                if self._current_model == model_name and self._session is not None:
                    LOG(f"Model '{model_name}' already loaded", "INFO")
                    return True
                LOG(f"Loading model: {model_name}", "INFO")
                self._session = new_session(model_name)
                self._current_model = model_name
                LOG(f"Model '{model_name}' loaded successfully", "INFO")
                return True
            except Exception as e:
                LOG(f"Failed to load model '{model_name}': {e}", "ERROR")
                traceback.print_exc()
                return False
    
    def remove_background(self, input_path: Path, output_path: Path, 
                         refine_edges: bool = True, enhance_image: bool = True,
                         denoise: bool = False) -> bool:
        if not REMBG_AVAILABLE:
            LOG("rembg not available", "ERROR")
            return False
        if self._session is None:
            LOG("No model loaded", "ERROR")
            return False
        try:
            LOG(f"Processing: {input_path.name}", "INFO")
            with open(input_path, 'rb') as f:
                input_data = f.read()
            output_data = remove(input_data, session=self._session, 
                                alpha_matting=refine_edges,
                                alpha_matting_foreground_threshold=240,
                                alpha_matting_background_threshold=10,
                                alpha_matting_erode_size=10)
            if refine_edges and PIL_AVAILABLE:
                img = Image.open(io.BytesIO(output_data))
                if img.mode == 'RGBA':
                    alpha = img.getchannel('A')
                    rgb = img.convert('RGB')
                    if enhance_image:
                        rgb = ImagePreprocessor.enhance_image(rgb)
                    if denoise:
                        rgb = ImagePreprocessor.denoise_image(rgb)
                    alpha_refined = EdgeRefiner.refine_alpha(alpha, rgb, iterations=2)
                    alpha_refined = EdgeRefiner.matting_refinement(rgb, alpha_refined)
                    rgb.putalpha(alpha_refined)
                    output_buffer = io.BytesIO()
                    rgb.save(output_buffer, format='PNG', optimize=True)
                    output_data = output_buffer.getvalue()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(output_data)
            LOG(f"Successfully processed: {output_path.name}", "INFO")
            return True
        except Exception as e:
            LOG(f"Failed to process {input_path.name}: {e}", "ERROR")
            traceback.print_exc()
            return False

class ModernUI:
    def __init__(self, engine: BackgroundRemoverEngine):
        self.engine = engine
        self.processing = False
        self.selected_files: List[Path] = []
        self.output_dir = Path.home() / "Desktop"
        self.current_model = "birefnet-general"
        self.refine_edges = True
        self.enhance_image = True
        self.denoise = False
        self.log_queue = queue.Queue()
        self._setup_ui()
        LOG("ModernUI initialized", "INFO")
    
    def _setup_ui(self):
        if CTK_AVAILABLE:
            self._setup_ctk_ui()
        else:
            self._setup_tk_ui()
        self._update_log_display()
        threading.Thread(target=self._model_loader, daemon=True).start()
    
    def _setup_ctk_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.root = ctk.CTk()
        self.root.title("Background Remover Pro · AI Powered")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._create_header()
        self._create_model_section()
        self._create_file_section()
        self._create_options_section()
        self._create_output_section()
        self._create_progress_section()
        self._create_log_section()
        self._create_action_buttons()
    
    def _setup_tk_ui(self):
        import tkinter as tk
        from tkinter import ttk
        self.root = tk.Tk()
        self.root.title("Background Remover Pro · AI Powered")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill="both", expand=True)
        self._create_header_tk()
        self._create_model_section_tk()
        self._create_file_section_tk()
        self._create_options_section_tk()
        self._create_output_section_tk()
        self._create_progress_section_tk()
        self._create_log_section_tk()
        self._create_action_buttons_tk()
    
    def _create_header(self):
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))
        title = ctk.CTkLabel(header_frame, text="Background Remover Pro", 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(side="left")
        subtitle = ctk.CTkLabel(header_frame, text="AI-Powered · State-of-the-Art", 
                               font=ctk.CTkFont(size=12), text_color="gray")
        subtitle.pack(side="left", padx=10)
        status_indicator = ctk.CTkLabel(header_frame, text="●", text_color="green", 
                                       font=ctk.CTkFont(size=16))
        status_indicator.pack(side="right")
        self.status_label = ctk.CTkLabel(header_frame, text="Ready", 
                                        font=ctk.CTkFont(size=11), text_color="gray")
        self.status_label.pack(side="right", padx=5)
    
    def _create_model_section(self):
        model_frame = ctk.CTkFrame(self.main_frame)
        model_frame.pack(fill="x", pady=(0, 10))
        model_label = ctk.CTkLabel(model_frame, text="AI Model:", 
                                  font=ctk.CTkFont(size=12, weight="bold"))
        model_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        model_names = list(MODEL_REGISTRY.keys())
        model_display = [f"{MODEL_REGISTRY[m]['name']} ({MODEL_REGISTRY[m]['size_mb']}MB)" for m in model_names]
        self.model_var = ctk.StringVar(value=model_display[0])
        self.model_menu = ctk.CTkOptionMenu(model_frame, values=model_display, 
                                           variable=self.model_var,
                                           command=self._on_model_change,
                                           width=250)
        self.model_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.model_desc_label = ctk.CTkLabel(model_frame, text="", 
                                            font=ctk.CTkFont(size=10), 
                                            text_color="gray")
        self.model_desc_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")
        self._update_model_description()
    
    def _create_file_section(self):
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill="x", pady=(0, 10))
        select_btn = ctk.CTkButton(file_frame, text="Select Images", 
                                  command=self._select_files,
                                  width=120, height=32)
        select_btn.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.file_count_label = ctk.CTkLabel(file_frame, text="No files selected", 
                                            font=ctk.CTkFont(size=11))
        self.file_count_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.file_list_frame = ctk.CTkScrollableFrame(file_frame, height=100)
        self.file_list_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        file_frame.columnconfigure(1, weight=1)
    
    def _create_options_section(self):
        options_frame = ctk.CTkFrame(self.main_frame)
        options_frame.pack(fill="x", pady=(0, 10))
        self.refine_var = ctk.BooleanVar(value=True)
        refine_check = ctk.CTkCheckBox(options_frame, text="Edge Refinement (Hair/Glass)", 
                                      variable=self.refine_var,
                                      command=lambda: setattr(self, 'refine_edges', self.refine_var.get()))
        refine_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.enhance_var = ctk.BooleanVar(value=True)
        enhance_check = ctk.CTkCheckBox(options_frame, text="Auto Enhance (Sharpness/Contrast)", 
                                       variable=self.enhance_var,
                                       command=lambda: setattr(self, 'enhance_image', self.enhance_var.get()))
        enhance_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.denoise_var = ctk.BooleanVar(value=False)
        denoise_check = ctk.CTkCheckBox(options_frame, text="Denoise (AI Cleanup)", 
                                       variable=self.denoise_var,
                                       command=lambda: setattr(self, 'denoise', self.denoise_var.get()))
        denoise_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    
    def _create_output_section(self):
        output_frame = ctk.CTkFrame(self.main_frame)
        output_frame.pack(fill="x", pady=(0, 10))
        output_label = ctk.CTkLabel(output_frame, text="Output Folder:", 
                                   font=ctk.CTkFont(size=12, weight="bold"))
        output_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.output_var = ctk.StringVar(value=str(self.output_dir))
        output_entry = ctk.CTkEntry(output_frame, textvariable=self.output_var, width=350)
        output_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        browse_btn = ctk.CTkButton(output_frame, text="Browse", command=self._browse_output,
                                  width=80, height=32)
        browse_btn.grid(row=0, column=2, padx=10, pady=10)
        output_frame.columnconfigure(1, weight=1)
    
    def _create_progress_section(self):
        progress_frame = ctk.CTkFrame(self.main_frame)
        progress_frame.pack(fill="x", pady=(0, 10))
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(padx=10, pady=5, fill="x")
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(progress_frame, text="", 
                                          font=ctk.CTkFont(size=10))
        self.progress_label.pack(padx=10, pady=(0, 5))
    
    def _create_log_section(self):
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))
        log_label = ctk.CTkLabel(log_frame, text="System Log:", 
                                font=ctk.CTkFont(size=12, weight="bold"))
        log_label.pack(anchor="w", padx=10, pady=(10, 5))
        self.log_text = ctk.CTkTextbox(log_frame, height=150, font=ctk.CTkFont(size=10))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def _create_action_buttons(self):
        button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        button_frame.pack(fill="x")
        self.process_btn = ctk.CTkButton(button_frame, text="Process Images", 
                                        command=self._start_processing,
                                        width=150, height=40,
                                        font=ctk.CTkFont(size=14, weight="bold"),
                                        state="disabled")
        self.process_btn.pack(side="left", padx=5)
        clear_btn = ctk.CTkButton(button_frame, text="Clear All", 
                                 command=self._clear_all,
                                 width=100, height=40,
                                 fg_color="gray")
        clear_btn.pack(side="left", padx=5)
    
    def _create_header_tk(self):
        pass
    
    def _create_model_section_tk(self):
        pass
    
    def _create_file_section_tk(self):
        pass
    
    def _create_options_section_tk(self):
        pass
    
    def _create_output_section_tk(self):
        pass
    
    def _create_progress_section_tk(self):
        pass
    
    def _create_log_section_tk(self):
        pass
    
    def _create_action_buttons_tk(self):
        pass
    
    def _model_loader(self):
        success = self.engine.load_model(self.current_model)
        self.root.after(0, lambda: self._on_model_loaded(success))
    
    def _on_model_loaded(self, success: bool):
        if success:
            self.process_btn.configure(state="normal")
            self.status_label.configure(text="Ready", text_color="green")
            LOG(f"Model '{self.current_model}' loaded and ready", "INFO")
        else:
            self.status_label.configure(text="Model Failed", text_color="red")
            LOG(f"Failed to load model '{self.current_model}'", "ERROR")
    
    def _on_model_change(self, choice):
        selected_display = choice
        for key, info in MODEL_REGISTRY.items():
            if info['name'] in selected_display:
                self.current_model = key
                break
        self._update_model_description()
        self.process_btn.configure(state="disabled")
        self.status_label.configure(text="Loading...", text_color="yellow")
        threading.Thread(target=self._model_loader, daemon=True).start()
    
    def _update_model_description(self):
        info = MODEL_REGISTRY.get(self.current_model, {})
        desc = info.get('description', '')
        self.model_desc_label.configure(text=desc)
    
    def _select_files(self):
        from tkinter import filedialog
        filetypes = [
            ("All Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("WEBP", "*.webp"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff *.tif"),
        ]
        files = filedialog.askopenfilenames(
            title="Select Images for Background Removal",
            filetypes=filetypes
        )
        if files:
            self.selected_files = [Path(f) for f in files]
            self.file_count_label.configure(text=f"{len(self.selected_files)} file(s) selected")
            self._update_file_list()
            LOG(f"Selected {len(self.selected_files)} file(s)", "INFO")
    
    def _update_file_list(self):
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        for f in self.selected_files[:10]:
            label = ctk.CTkLabel(self.file_list_frame, text=f.name, 
                                font=ctk.CTkFont(size=10))
            label.pack(anchor="w", padx=5, pady=2)
        if len(self.selected_files) > 10:
            label = ctk.CTkLabel(self.file_list_frame, 
                                text=f"... and {len(self.selected_files) - 10} more",
                                font=ctk.CTkFont(size=10), text_color="gray")
            label.pack(anchor="w", padx=5, pady=2)
    
    def _browse_output(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_dir = Path(folder)
            self.output_var.set(str(self.output_dir))
            LOG(f"Output directory set to: {self.output_dir}", "INFO")
    
    def _start_processing(self):
        if not self.selected_files:
            self._show_error("No images selected", "Please select at least one image.")
            return
        if not self.output_dir.exists():
            self._show_error("Invalid output directory", "Please select a valid output folder.")
            return
        if self.processing:
            return
        self.processing = True
        self.process_btn.configure(state="disabled", text="Processing...")
        self.progress_bar.set(0)
        threading.Thread(target=self._process_images, daemon=True).start()
    
    def _process_images(self):
        total = len(self.selected_files)
        success = 0
        failed = []
        for idx, filepath in enumerate(self.selected_files):
            try:
                progress_pct = (idx + 1) / total
                self.root.after(0, lambda p=progress_pct: self.progress_bar.set(p))
                self.root.after(0, lambda i=idx+1, t=total: 
                               self.progress_label.configure(text=f"Processing {i}/{t}: {filepath.name}"))
                output_filename = f"{filepath.stem}_remover.png"
                output_path = self.output_dir / output_filename
                ok = self.engine.remove_background(
                    filepath, output_path,
                    refine_edges=self.refine_edges,
                    enhance_image=self.enhance_image,
                    denoise=self.denoise
                )
                if ok:
                    success += 1
                else:
                    failed.append(str(filepath.name))
            except Exception as e:
                failed.append(f"{filepath.name}: {e}")
                LOG(f"Exception processing {filepath.name}: {e}", "ERROR")
        self.root.after(0, lambda: self._processing_done(success, total, failed))
    
    def _processing_done(self, success: int, total: int, failed: List[str]):
        self.processing = False
        self.process_btn.configure(state="normal", text="Process Images")
        self.progress_label.configure(text="")
        LOG(f"Processing completed. Success: {success}/{total}", "INFO")
        if failed:
            LOG(f"Failed files: {failed}", "WARNING")
            msg = f"Completed with errors.\n\nSuccess: {success}/{total}\n\nFailed:\n" + "\n".join(failed[:5])
            if len(failed) > 5:
                msg += f"\n... and {len(failed)-5} more"
            self._show_error("Processing Completed", msg)
        else:
            self._show_info("Success", f"All {total} images processed successfully!\n\nSaved to:\n{self.output_dir}")
        self.selected_files = []
        self.file_count_label.configure(text="No files selected")
        self._update_file_list()
    
    def _clear_all(self):
        self.selected_files = []
        self.file_count_label.configure(text="No files selected")
        self._update_file_list()
        LOG("Cleared all selections", "INFO")
    
    def _update_log_display(self):
        try:
            while not self.log_queue.empty():
                msg, level = self.log_queue.get_nowait()
                self.log_text.insert("end", f"[{level}] {msg}\n")
            self.log_text.see("end")
        except:
            pass
        self.root.after(500, self._update_log_display)
    
    def _on_closing(self):
        LOG("Application shutting down", "INFO")
        self.root.destroy()
    
    def _show_error(self, title: str, message: str):
        from tkinter import messagebox
        messagebox.showerror(title, message)
    
    def _show_info(self, title: str, message: str):
        from tkinter import messagebox
        messagebox.showinfo(title, message)
    
    def run(self):
        self.root.mainloop()

class Application:
    def __init__(self):
        self.engine = BackgroundRemoverEngine()
        self.ui = ModernUI(self.engine)
        LOG("Application fully initialized", "INFO")
    
    def run(self):
        self.ui.run()

def _signal_handler(signum, frame):
    LOG(f"Received signal {signum}, shutting down gracefully", "INFO")
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def main():
    try:
        LOG(f"Starting Background Remover Pro v2.0.0", "INFO")
        LOG(f"Python: {sys.version}", "INFO")
        LOG(f"Platform: {sys.platform}", "INFO")
        LOG(f"RemBG available: {REMBG_AVAILABLE}", "INFO")
        LOG(f"PIL available: {PIL_AVAILABLE}", "INFO")
        LOG(f"OpenCV available: {CV2_AVAILABLE}", "INFO")
        LOG(f"CustomTkinter available: {CTK_AVAILABLE}", "INFO")
        app = Application()
        app.run()
    except Exception as e:
        LOG(f"Fatal error in main: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
