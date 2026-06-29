import time
import gc
import torch
import threading
from functools import wraps

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self.models = {}
        self.last_accessed = {}
        self.lock = threading.Lock()
        self.timeouts = {
            "embedding": 600,
            "rerank": 600,
            "ocr": 600,
            "vlm": 600
        }
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_idle_models, daemon=True)
        self.monitor_thread.start()

    def get_model(self, model_key, loader_fn):
        """
        Fetch the model from the manager. If it does not exist or was released, load it.
        """
        with self.lock:
            self.last_accessed[model_key] = time.time()
            if model_key not in self.models or self.models[model_key] is None:
                print(f"[ModelManager] 🔄 正在載入模型: `{model_key}`...")
                self.models[model_key] = loader_fn()
            return self.models[model_key]

    def update_access_time(self, model_key):
        """
        Update the last accessed timestamp for a model.
        """
        with self.lock:
            if model_key in self.models:
                self.last_accessed[model_key] = time.time()

    def release_model(self, model_key):
        """
        Explicitly release a model and reclaim memory.
        """
        with self.lock:
            if model_key in self.models and self.models[model_key] is not None:
                print(f"[ModelManager] 🗑️ 偵測到閒置超時，釋放模型: `{model_key}`...")
                self.models[model_key] = None
                if model_key in self.last_accessed:
                    del self.last_accessed[model_key]
                # Force GC and GPU/MPS memory purge
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def release_all_models(self):
        """
        Release all cached models immediately and reclaim VRAM/RAM.
        """
        with self.lock:
            for model_key in list(self.models.keys()):
                if self.models[model_key] is not None:
                    print(f"[ModelManager] 🗑️ 主動釋放模型: `{model_key}`...")
                    self.models[model_key] = None
            self.last_accessed.clear()
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _monitor_idle_models(self):
        """
        Background monitor running every 30 seconds to clean up models exceeding TTL.
        """
        while not self.stop_event.is_set():
            time.sleep(30)
            now = time.time()
            keys_to_release = []
            with self.lock:
                for key, last_time in list(self.last_accessed.items()):
                    timeout = self.timeouts.get(key, 600)
                    if now - last_time > timeout:
                        keys_to_release.append(key)
            
            for key in keys_to_release:
                self.release_model(key)

def auto_release(model_key: str, timeout: int = 600):
    """
    Decorator to wrap methods that access a model. Updates the manager's access timestamp.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = ModelManager()
            manager.update_access_time(model_key)
            # Ensure the timeout is configured
            with manager.lock:
                manager.timeouts[model_key] = timeout
            return func(*args, **kwargs)
        return wrapper
    return decorator
