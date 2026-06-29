import torch
import gc
import config

def free_memory(device=None):
    """
    統一管理 GPU/MPS 的記憶體釋放。
    如果提供了 device 參數，則根據 device 的 type 來清理。
    如果未提供，則根據系統配置自動清理可用的硬體快取。
    """
    gc.collect()
    
    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

def extract_logits(outputs):
    """
    安全地從 HuggingFace 的輸出結構或 nn.DataParallel 退化的 Tuple 中提取 logits。
    在使用 DataParallel (NUM_GPUS > 1) 時，HuggingFace 可能會回傳 Tuple。
    """
    return outputs.logits if hasattr(outputs, "logits") else outputs[0]

def is_multi_gpu_enabled(device):
    """
    判斷是否啟用多卡協同 (Multi-GPU DataParallel / Multi-Process)。
    """
    return getattr(config, "NUM_GPUS", 0) > 1 and device.type == "cuda"
