import os
import torch
import threading

from src.system_utils import init_cpu_thread_environment, resolve_optimal_device

# macOS ARM-64 (M-Series) OpenMP/Tokenizers thread conflict fix
init_cpu_thread_environment()

# 設備配置：自動檢測多平台硬體加速 (Apple Silicon MPS, NVIDIA/AMD CUDA, Intel XPU)
DEVICE, NUM_GPUS = resolve_optimal_device()

# 模型選型 (針對 16GB RAM 進行輕量化優化)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"                             # v3.0: 1024維, 強大中英雙語語意嵌入
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # ~80MB, 本地極速重排
FORMULA_MODEL_NAME = "facebook/nougat-small"                     # ~250M 參數, Meta 開源論文公式解析 (2023)
VLM_MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"       # v3.2: M4 Mac 最佳化輕量雙語 VLM

# 系統路徑配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 自動初始化主資料目錄
os.makedirs(DATA_DIR, exist_ok=True)

# 讀取本地 LLM 設定檔以進行持久化
CONFIG_FILE = os.path.join(DATA_DIR, "llm_config.json")
saved_config = {}
if os.path.exists(CONFIG_FILE):
    try:
        import json
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
    except Exception as e:
        print(f"[Config] 讀取本地 LLM 設定失敗: {e}")

# LLM 模型路徑與提供者設定 (支援跨平台 Universal Portability)
# 允許從環境變數或本地設定檔覆寫
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", saved_config.get("llm_model_path", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"))
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", saved_config.get("llm_provider", "Auto"))  # オプション: Auto, MLX, API
LLM_API_KEY = os.environ.get("LLM_API_KEY", saved_config.get("llm_api_key", ""))
LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", saved_config.get("llm_api_base_url", "https://api.openai.com/v1"))
LLM_API_MODEL_NAME = os.environ.get("LLM_API_MODEL_NAME", saved_config.get("llm_api_model_name", "gpt-4o-mini"))

# LoRA 微調預設超參數（可透過 Step 6 UI 客製化並覆寫）
DEFAULT_LORA_EPOCHS = 3        # 訓練輪數：最低 3 輪確保收斂
DEFAULT_LORA_RANK = 8          # LoRA Rank：矩陣分解的秩，越高能力越強但越慢
DEFAULT_LORA_LR = 3e-5         # 學習率：標準速度
DEFAULT_LORA_MARGIN = 0.3      # Contrastive Margin：正負樣本最小分距

def save_llm_config(provider, key, base_url, model_name, model_path):
    import json
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "llm_provider": provider,
                "llm_api_key": key,
                "llm_api_base_url": base_url,
                "llm_api_model_name": model_name,
                "llm_model_path": model_path
            }, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[Config] 儲存本地 LLM 設定失敗: {e}")

# 多 Session 執行緒安全的閒置追蹤 (IdleMonitor 守護線程使用)
# SESSIONS_LOCK 保護 ACTIVE_SESSIONS 避免多 session 同時讀寫造成 RuntimeError
ACTIVE_SESSIONS: dict = {}
SESSIONS_LOCK = threading.Lock()

print(f"[System Config] 運行設備已設定為: {DEVICE}")
print(f"[System Config] 本地資料存儲路徑: {DATA_DIR}")
