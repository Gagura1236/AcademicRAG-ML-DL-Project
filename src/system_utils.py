import os
import torch

def init_cpu_thread_environment():
    """
    初始化 CPU 與執行緒的環境變數。
    - 關閉 Tokenizers 的內建平行處理，防止在 macOS ARM 架構上與其他 C++ 核心庫衝突。
    - 限制 OpenMP 只使用單一執行緒，避免多執行緒競爭造成的 SIGSEGV (Segmentation fault: 11) 崩潰。
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

def resolve_optimal_device():
    """
    自動檢測並回傳當前系統最佳硬體加速裝置與顯卡數量。
    支援 Apple Silicon (MPS), NVIDIA/AMD (CUDA), Intel (XPU), 與純 CPU。
    回傳值: (device: torch.device, num_gpus: int)
    """
    if torch.backends.mps.is_available():
        print("[Hardware Detector] 🍎 成功偵測到 Apple Silicon (M-Series)，啟用 MPS (Metal Performance Shaders) 硬體加速！")
        return torch.device("mps"), 1
    elif torch.cuda.is_available():
        # 💡 註：在 AMD ROCm 環境下，PyTorch 會將裝置名稱映射為 "cuda"，因此這行同時相容 NVIDIA 與 AMD GPU！
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "Unknown"
        print(f"[Hardware Detector] 🚀 成功偵測到 NVIDIA/AMD 架構 ({gpu_name})，啟用 CUDA/ROCm 加速！(共 {num_gpus} 張顯卡)")
        return torch.device("cuda"), num_gpus
    else:
        # 檢測 Intel 顯示卡 (XPU)：需要安裝 intel_extension_for_pytorch (IPEX) 套件
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                print("[Hardware Detector] ⚡ 成功偵測到 Intel 顯示卡，啟用 XPU 硬體加速！")
                return torch.device("xpu"), 1
        except Exception as e:
            print(f"[Hardware Detector] ⚠️ XPU 檢測失敗: {e}")
            
        print("[Hardware Detector] ⚠️ 未偵測到主流 GPU，已回退至 CPU 模式運算 (速度將受到限制)。")
        return torch.device("cpu"), 0

def resolve_safe_vision_device(primary_device: torch.device):
    """
    針對特定容易發生死鎖的模型 (如 Nougat Vision EncoderDecoderModel)，回傳安全的裝置。
    由於 MPS backend 在某些 Vision 模型 generate() 時存在死鎖風險，若主裝置為 mps，強制降級為 cpu。
    回傳值: safe_device: torch.device
    """
    if primary_device.type == "mps":
        return torch.device("cpu")
    return primary_device
