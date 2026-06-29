import os
import sys
import fitz  # PyMuPDF
import torch
from PIL import Image
import difflib

# 獲取專案根目錄，加入系統路徑以正確導入 config.py 與 utils.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.utils import crop_pdf_bbox, clean_latex_string
from src.gpu_utils import free_memory
from src.system_utils import resolve_safe_vision_device

class ScientificPDFParser:
    def __init__(self, load_ocr_model: bool = False):
        self.device = config.DEVICE
        # MPS backend has known deadlocks/hangs inside VisionEncoderDecoderModel.generate().
        # We run the OCR model on CPU if device is mps.
        self.ocr_device = resolve_safe_vision_device(self.device)
        
        if load_ocr_model:
            self.init_ocr_model()

    @property
    def ocr_model(self):
        from src.model_manager import ModelManager
        try:
            return ModelManager().get_model("ocr", self._load_ocr_model_internal)[0]
        except Exception as e:
            print(f"⚠️ 載入 Nougat 模型失敗: {e}")
            return None

    @property
    def ocr_processor(self):
        from src.model_manager import ModelManager
        try:
            return ModelManager().get_model("ocr", self._load_ocr_model_internal)[1]
        except Exception as e:
            print(f"⚠️ 載入 Nougat 處理器失敗: {e}")
            return None
            
    def init_ocr_model(self):
        """
        初始化並載入 Nougat Transformer 公式解析模型
        """
        _ = self.ocr_model

    def _load_ocr_model_internal(self):
        print(f"\n[PDF Parser] 正在載入本地公式 OCR 模型: {config.FORMULA_MODEL_NAME}...")
        from transformers import NougatProcessor, VisionEncoderDecoderModel
        
        # MPS 上 bfloat16 部分 op 會 fallback 到 CPU，推論統一用 float32 更穩定
        dtype = torch.float32
        
        processor = NougatProcessor.from_pretrained(config.FORMULA_MODEL_NAME)
        
        # ── Compatibility patch for transformers >= 4.40 ──────────────────
        try:
            import transformers.image_processing_utils as _ipu
            _orig_vtd = _ipu.validate_typed_dict

            def _lenient_validate_typed_dict(typed_dict_class, data):
                _ipu.validate_typed_dict = _orig_vtd  # restore immediately
                _orig_vtd(typed_dict_class, {k: v for k, v in data.items() if v is not None})
                _ipu.validate_typed_dict = _lenient_validate_typed_dict  # re-apply

            _ipu.validate_typed_dict = _lenient_validate_typed_dict
            print("[PDF Parser] ✅ Nougat/transformers 相容性 patch 已套用")
        except Exception as _patch_err:
            print(f"[PDF Parser] ⚠️ validate_typed_dict patch 失敗 (將繼續): {_patch_err}")
        # ─────────────────────────────────────────────────────────────────
        
        model = VisionEncoderDecoderModel.from_pretrained(
            config.FORMULA_MODEL_NAME, 
            torch_dtype=dtype
        )
        model.to(self.ocr_device)
        model.eval()
        print(f"[PDF Parser] 公式 OCR 模型載入成功！(Device: {self.ocr_device})")
        return model, processor

    def parse_layout(self, pdf_path: str, progress_callback=None) -> list:
        """
        分析 PDF 版面結構，提取內文段落並自動定位可能的數學公式區域。
        返回一個 List，每個元素包含：頁碼、類型(text/equation)、內容與邊界框(bbox)
        """
        print(f"[PDF Parser] 開始分析版面: {os.path.basename(pdf_path)}...")
        parsed_elements = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    if progress_callback:
                        progress_callback((page_num + 1) / len(doc), f"正在分析 PDF 頁面 {page_num + 1}/{len(doc)}...")
                    page = doc[page_num]
                    # 取得版面區塊 (Blocks)
                    # blocks 格式: (x0, y0, x1, y1, "text", block_no, block_type)
                    blocks = page.get_text("blocks")
                    
                    for b in blocks:
                        x0, y0, x1, y1, text, block_no, block_type = b
                        cleaned_text = text.strip()
                        
                        if not cleaned_text:
                            continue
                            
                        # 辨識是否為獨立數學公式區塊 (Heuristic Rules)
                        # 規則：包含常見數學符號、長度較短、字元密度低、或符合特定符號模式的單行區塊
                        is_equation = False
                        math_indicators = ["∑", "∫", "∏", "√", "α", "β", "λ", "θ", "±", "≠", "≤", "≥", "→", "∞", "∂", "∇"]
                        
                        # 判斷特殊 LaTeX 常見字眼
                        has_math_symbols = any(sym in cleaned_text for sym in math_indicators)
                        has_latex_syntax = any(term in cleaned_text for term in ["^", "_", "\\", "{", "}"])
                        
                        # 計算寬高比與字元長度 (獨立公式通常較窄，行數少，且包含數學指示字元)
                        height = y1 - y0
                        width = x1 - x0
                        
                        char_density = len(cleaned_text) / (width * height + 1e-5)
                        
                        if (has_math_symbols or has_latex_syntax) and char_density < 0.01 and height < 100:
                            is_equation = True
                            
                        element_type = "equation" if is_equation else "text"
                        
                        parsed_elements.append({
                            "page": page_num,
                            "type": element_type,
                            "content": cleaned_text,
                            "bbox": (x0, y0, x1, y1)
                        })
                        
            print(f"[PDF Parser] 版面分析完成。共提取 {len(parsed_elements)} 個元素。")
        except Exception as e:
            # Bug 20 Fix: re-raise so callers (app.py) can surface the error to the user
            # instead of silently returning an empty list with no user-visible feedback
            raise RuntimeError(f"PDF 版面解析失敗: {e}") from e
            
        return parsed_elements

    def image_to_latex(self, image_path: str) -> str:
        """
        利用本地 Nougat 模型將裁剪出的公式影像轉換為 LaTeX 代碼
        """
        if self.ocr_model is None or self.ocr_processor is None:
            self.init_ocr_model()
            
        if self.ocr_model is None:
            return "[模型未加載 - 無法 OCR 公式]"
            
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Pad image to Nougat's expected 672x896 size while preserving aspect ratio
            target_w, target_h = 672, 896
            w, h = image.size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            padded_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            padded_img.paste(resized_img, (paste_x, paste_y))
            image = padded_img
            
            # 使用 Nougat 處理器預處理影像。
            try:
                pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values
            except Exception as proc_err:
                print(f"⚠️ Nougat 處理器圖像預處理失敗: {proc_err}")
                raise proc_err
            
            pixel_values = pixel_values.to(self.ocr_device)
            # 確保輸入 tensor 為與模型相符的浮點數型態 (避免 uint8 vs float32 報錯)
            if not pixel_values.is_floating_point():
                model_dtype = next(self.ocr_model.parameters()).dtype
                pixel_values = pixel_values.to(model_dtype)
            elif self.ocr_model.dtype == torch.bfloat16:
                pixel_values = pixel_values.to(torch.bfloat16)
                
            # 模型推論生成 LaTeX
            with torch.no_grad():
                unk_id = self.ocr_processor.tokenizer.unk_token_id
                bad_words = [[unk_id]] if unk_id is not None else None
                outputs = self.ocr_model.generate(
                    pixel_values,
                    min_length=1,
                    max_new_tokens=150,
                    bad_words_ids=bad_words
                )
                
            # 解碼輸出
            latex_res = self.ocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 釋放設備快取避免記憶體洩漏
            del pixel_values, outputs
            free_memory(self.ocr_device)
                
            return clean_latex_string(latex_res)
        except Exception as e:
            print(f"⚠️ 公式 OCR 推論失敗: {e}")
            return "[公式 OCR 失敗]"

    def process_pdf_equations(self, pdf_path: str, layout_elements: list, max_ocr_equations: int = 5) -> list:
        """
        遍歷所有偵測為 equation 的元素，將其對齊 LaTeX 原始碼公式 (Ground Truth)，
        或裁剪影像並使用本地 OCR 轉換為真實 LaTeX。若兩者皆不可得，則優雅降級使用 PyMuPDF 提取內容。
        優化：為避免 M4 記憶體溢出，若無 TeX 資料，此處僅對前幾個公式進行轉譯。
        """
        processed_elements = []
        eq_counter = 0
        
        # 1. 嘗試載入 TeX 原始碼中的真實 LaTeX 公式 (Ground Truth)
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 動態計算專案目錄
        project_dir = os.path.dirname(os.path.dirname(pdf_path))
        tex_dir = os.path.join(project_dir, "tex")
        extracted_dir = os.path.join(project_dir, "extracted")
        
        tex_folder = os.path.join(tex_dir, paper_id)
        gt_equations = []
        
        try:
            from src.utils import extract_ground_truth_equations
            if os.path.exists(tex_folder):
                gt_equations = extract_ground_truth_equations(tex_folder)
                print(f"[PDF Parser] 發現 {len(gt_equations)} 個 TeX 原始碼公式，啟用精準 Ground Truth 公式對齊匹配！")
        except Exception as e:
            print(f"[PDF Parser] 加載 TeX 公式失敗: {e}")
        
        # 建立臨時輸出資料夾儲存裁剪的公式圖檔
        temp_crop_dir = os.path.join(extracted_dir, paper_id)
        os.makedirs(temp_crop_dir, exist_ok=True)
        
        for elem in layout_elements:
            if elem["type"] == "equation":
                eq_counter += 1
                
                # A. 優先使用 TeX 原始碼對齊
                matched_latex = ""
                if gt_equations:
                    best_match = ""
                    best_score = 0
                    # 計算字串相似度
                    for eq in gt_equations:
                        overlap = difflib.SequenceMatcher(None, elem["content"], eq).ratio()
                        if overlap > best_score:
                            best_score = overlap
                            best_match = eq
                    if best_score > 0.7:  # 相似度大於 70%
                        matched_latex = best_match
                
                if matched_latex:
                    # 成功對齊 TeX 原始碼公式，不需載入 OCR 模型，極速且 100% 精準！
                    elem["content_latex"] = matched_latex
                    elem["crop_image_path"] = ""
                else:
                    # B. 降級使用 Nougat OCR 或 PyMuPDF 提取內容
                    if eq_counter <= max_ocr_equations:
                        img_path = os.path.join(temp_crop_dir, f"eq_p{elem['page']}_{eq_counter}.png")
                        # 裁剪公式影像
                        crop_pdf_bbox(pdf_path, elem["page"], elem["bbox"], img_path)
                        
                        # 進行 OCR 轉 LaTeX
                        print(f"[PDF Parser] 正在轉譯公式 [{eq_counter}/{max_ocr_equations}] (第 {elem['page']+1} 頁)...")
                        latex_str = self.image_to_latex(img_path)
                        
                        if "失敗" in latex_str or "未加載" in latex_str:
                            # OCR 失敗，降級使用原提取字串
                            elem["content_latex"] = elem["content"]
                        else:
                            elem["content_latex"] = latex_str
                        elem["crop_image_path"] = img_path
                    else:
                        # 延遲或直接使用原提取字串
                        elem["content_latex"] = elem["content"]
                        elem["crop_image_path"] = ""
            
            processed_elements.append(elem)
            
        return processed_elements

    def extract_figures_and_charts(self, pdf_path: str, paper_id: str) -> list:
        """
        從 PDF 中萃取圖表與圖片作為裁剪影像。
        結合「圖表說明文字 (Caption) 上方定位」與「大圖標記」策略，確保各類學術圖表皆能完整擷取。
        """
        import re
        project_dir = os.path.dirname(os.path.dirname(pdf_path))
        extracted_dir = os.path.join(project_dir, "extracted")
        fig_dir = os.path.join(extracted_dir, paper_id, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        figures = []
        caption_pattern = re.compile(r'^(?:Figure|Fig\.)\s+\d+', re.IGNORECASE)
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # 1. 策略 A：基於 Figure/Fig. Caption 的上方區域裁剪 (最精準，能包含組合圖與標籤)
                    blocks = page.get_text("blocks")
                    for b_idx, b in enumerate(blocks):
                        x0, y0, x1, y1, text, block_no, block_type = b
                        if block_type == 0 and caption_pattern.match(text.strip()):
                            # 擷取說明文字上方 220 點的區域作為圖表主體
                            fig_h = 220
                            fy0 = max(0, y0 - fig_h)
                            rect = fitz.Rect(x0, fy0, x1, y0)
                            
                            pix = page.get_pixmap(clip=rect, dpi=150)
                            img_name = f"page_{page_num}_fig_{block_no}.png"
                            img_path = os.path.join(fig_dir, img_name)
                            pix.save(img_path)
                            figures.append({
                                "page": page_num,
                                "path": img_path,
                                "bbox": (x0, fy0, x1, y0)
                            })
                            
                    # 2. 策略 B：大圖 (Images) 的 xref 擷取備用
                    image_list = page.get_images(full=True)
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        rects = page.get_image_rects(xref)
                        if rects:
                            rect = rects[0]
                            # 過濾掉太小的圖片 (通常是裝飾圖案、Logo)
                            if (rect.x1 - rect.x0) > 150 and (rect.y1 - rect.y0) > 150:
                                # 檢查是否已與策略 A 擷取之圖表重疊
                                already_cropped = False
                                for fig in figures:
                                    if fig["page"] == page_num:
                                        f_x0, f_y0, f_x1, f_y1 = fig["bbox"]
                                        intersect = rect & fitz.Rect(f_x0, f_y0, f_x1, f_y1)
                                        if intersect.rect_area() / rect.rect_area() > 0.6:
                                            already_cropped = True
                                            break
                                            
                                if not already_cropped:
                                    pix = page.get_pixmap(clip=rect, dpi=150)
                                    img_name = f"page_{page_num}_img_{img_idx}.png"
                                    img_path = os.path.join(fig_dir, img_name)
                                    pix.save(img_path)
                                    figures.append({
                                        "page": page_num,
                                        "path": img_path,
                                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)
                                    })
        except Exception as e:
            print(f"[PDF Parser] ⚠️ 圖表擷取失敗: {e}")
        return figures

if __name__ == "__main__":
    from src.system_utils import get_project_paths
    paths = get_project_paths("default")
    parser = ScientificPDFParser()
    pdf_files = [f for f in os.listdir(paths['pdf_dir']) if f.endswith(".pdf")]
    if pdf_files:
        test_pdf = os.path.join(paths['pdf_dir'], pdf_files[0])
        elements = parser.parse_layout(test_pdf)
        print(f"前 5 個解析元素: {elements[:5]}")
    else:
        print("未在 data/pdfs 下發現 PDF，請先運行 data_collector.py 下載測試數據。")
