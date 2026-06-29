import os
import re
import fitz  # PyMuPDF
from PIL import Image

def crop_pdf_bbox(pdf_path: str, page_num: int, bbox: tuple, output_path: str) -> str:
    """
    從 PDF 的指定頁面中裁剪指定邊界框 (bbox) 的區域並保存為圖像。
    bbox 格式: (x0, y0, x1, y1) - 由 LayoutLM 或 OCR 工具產生的 PDF 座標系
    """
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num]
            
            # PyMuPDF 使用 Rect 物件
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # 設定渲染解析度 (zoom=2.0 提高 OCR 的圖片清晰度)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            
            # 取得裁剪區域的 pixmap
            pix = page.get_pixmap(matrix=mat, clip=rect)
            
            # 轉換為 PIL Image 並儲存
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(output_path, "PNG")
            return output_path
    except Exception as e:
        print(f"⚠️ 裁剪 PDF 失敗: {e}")
        return ""

def extract_ground_truth_equations(tex_folder: str) -> list:
    """
    從解壓的 LaTeX 資料夾中，使用正則表達式提取所有數學公式。
    用於與 PDF 偵測到的公式進行對齊與精準度評估 (Ground Truth)。
    """
    equations = []
    if not os.path.exists(tex_folder):
        return equations
        
    # 定義常用的 LaTeX 公式模式
    # 1. $$ ... $$ (塊公式)
    # 2. \begin{equation} ... \end{equation} (標號公式)
    # 3. \begin{align} ... \end{align}
    # 4. \[ ... \]
    patterns = [
        r'\$\$(.*?)\$\$',
        r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
        r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
        r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',
        r'\\\[(.*?)\\\]'
    ]
    
    for root, _, files in os.walk(tex_folder):
        for file in files:
            if file.endswith('.tex'):
                tex_path = os.path.join(root, file)
                try:
                    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # 移除 LaTeX 註解 (% 開頭的行)
                        content = re.sub(r'^\s*%.*$', '', content, flags=re.MULTILINE)
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.DOTALL)
                            for match in matches:
                                cleaned_eq = match.strip().replace('\n', ' ')
                                # 過濾掉太短或無意義的字元
                                if len(cleaned_eq) > 3 and cleaned_eq not in equations:
                                    equations.append(cleaned_eq)
                except Exception as e:
                    print(f"⚠️ 讀取 TeX 檔案失敗: {tex_path}, 錯誤: {e}")
                    
    return equations

def clean_latex_string(latex_str: str) -> str:
    """
    清洗 OCR 輸出的 LaTeX 字串，去除冗餘字元並做基本校正
    """
    # 移除頭尾標記
    latex_str = latex_str.strip()
    latex_str = re.sub(r'^\\\(', '', latex_str)
    latex_str = re.sub(r'\\\)$', '', latex_str)
    latex_str = re.sub(r'^\\\[', '', latex_str)
    latex_str = re.sub(r'\\\]$', '', latex_str)
    
    # 替換多個空格為單個空格
    latex_str = re.sub(r'\s+', ' ', latex_str)
    return latex_str.strip()
