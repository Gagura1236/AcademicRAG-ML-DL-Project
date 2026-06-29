import os
import config

def describe_figure(image_path: str) -> str:
    """
    Use mlx-vlm and the configured VLM model to generate a detailed description of the chart or image.
    """
    try:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        from src.model_manager import ModelManager
        
        def load_vlm():
            model_name = getattr(config, "VLM_MODEL_NAME", "mlx-community/Qwen2-VL-2B-Instruct-4bit")
            print(f"[VLM] 🔄 正在載入本地雙語 VLM 模型: {model_name}...")
            model, processor = load(model_name)
            cfg = load_config(model_name)
            return model, processor, cfg
            
        model, processor, cfg = ModelManager().get_model("vlm", load_vlm)
        
        prompt = (
            "Describe the charts, diagrams, or content in this figure from a research paper in detail. "
            "Focus on what it shows, the labels, variables, and axes. Write a comprehensive summary."
        )
        
        formatted_prompt = apply_chat_template(
            processor,
            cfg,
            prompt,
            num_images=1
        )
        
        print(f"[VLM] 🧠 正在描述圖表: {os.path.basename(image_path)}...")
        output = generate(
            model,
            processor,
            formatted_prompt,
            [image_path],
            verbose=False,
            max_tokens=256
        )
        if hasattr(output, "text"):
            return output.text.strip()
        else:
            return str(output).strip()
    except Exception as e:
        print(f"[VLM] ⚠️ 圖表描述生成失敗 ({os.path.basename(image_path)}): {e}")
        return f"[圖表描述生成失敗: {e}]"
