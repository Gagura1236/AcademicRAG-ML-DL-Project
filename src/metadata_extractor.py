import os
import json
import sys
import numpy as np
import fitz  # PyMuPDF
import re

# 獲取專案根目錄，加入系統路徑以正確導入 config.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.project_utils import get_project_dir

class PaperMetadataManager:
    """
    管理論文層級的元數據 (Title, Authors, Abstract, Keywords, Category)。
    """
    def __init__(self, project_name="default"):
        self.project_name = project_name
        self.project_dir = get_project_dir(project_name)
        self.metadata_file = os.path.join(self.project_dir, "paper_metadata.json")
        self.metadata = self.load_metadata()

    def load_metadata(self) -> dict:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 載入 metadata.json 失敗: {e}")
        return {}

    def save_metadata(self):
        """JSON 序列化 metadata：自動處理 numpy 型別（np.int64, np.float32 等）防止 TypeError"""
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False, cls=_NumpyEncoder)

    def add_arxiv_metadata(self, paper_id: str, title: str, authors: list, abstract: str, categories: list):
        """新增 arXiv 來源的元數據"""
        # 自動從 arXiv ID 擷取年份
        year = "N/A"
        clean_id = paper_id.split('v')[0]
        if '.' in clean_id:
            prefix = clean_id.split('.')[0]
            if len(prefix) >= 2 and prefix.isdigit():
                yy = int(prefix[:2])
                year = f"19{yy}" if yy >= 90 else f"20{yy:02d}"
        elif '/' in clean_id:
            num_part = clean_id.split('/')[-1]
            if len(num_part) >= 2 and num_part[:2].isdigit():
                yy = int(num_part[:2])
                year = f"19{yy}" if yy >= 90 else f"20{yy:02d}"

        self.metadata[paper_id] = {
            "title": title,
            "authors": authors,
            "abstract": abstract.replace('\n', ' '),
            "categories": categories,
            "year": year,
            "keywords": [], # 將交由 paper_analyzer.py 透過 TF-IDF 產生
            "cluster_label": -1 # 將交由 KMeans 產生
        }
        self.save_metadata()

    def _validate_title(self, title: str, paper_id: str) -> bool:
        """
        雙重確認：驗證擷取出的標題是否為有效英文學術大標題
        """
        if not title:
            return False
        title_clean = title.strip().lower()
        
        # 1. 不能為空、只有幾個字元，或只有一個單字
        if len(title_clean) < 10 or len(title_clean.split()) < 2:
            return False
            
        # 2. 不能與 paper_id 相同或只是檔名
        if title_clean == paper_id.lower() or title_clean.endswith(".pdf"):
            return False
            
        # 3. 不能只是純數字、點、連接號的 ID 格式 (例如 1911.05402)
        if re.match(r'^[\d\.\-_]+$', title_clean):
            return False
            
        # 4. 英文標題必須含有起碼的英文字母
        letters = re.findall(r'[a-zA-Z]', title_clean)
        if len(letters) < 5:
            return False
            
        # 5. 不能包含 "No abstract found" 等預設錯誤提示
        if "no abstract found" in title_clean:
            return False
            
        return True

    def _is_arxiv_id(self, paper_id: str) -> bool:
        """判斷是否為 arXiv 論文 ID 格式 (例如 2306.16361 或 hep-th/9711200)"""
        return bool(re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper_id) or re.match(r'^[a-z\-]+/\d{7}', paper_id))

    def _fetch_title_from_arxiv(self, paper_id: str) -> str:
        """
        透過 arXiv API 線上查詢官方論文英文標題（最高優先權）
        """
        if not self._is_arxiv_id(paper_id):
            return ""
            
        import urllib.request
        import xml.etree.ElementTree as ET
        clean_id = re.sub(r'v\d+$', '', paper_id.strip())
        url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            if entry is not None:
                title_node = entry.find('atom:title', ns)
                if title_node is not None and title_node.text:
                    title = title_node.text.strip().replace('\n', ' ')
                    title = re.sub(r'\s+', ' ', title)
                    title = self._clean_ligatures(title)
                    return title
        except Exception as e:
            print(f"[Metadata Manager] ⚠️ 線上查詢 arXiv ID {paper_id} 失敗: {e}")
        return ""

    def _fetch_title_from_semantic_scholar(self, paper_id: str) -> str:
        """
        透過 Semantic Scholar API 查詢標題（支援 arXiv ID 與 DOI）
        """
        import requests
        clean_id = re.sub(r'v\d+$', '', paper_id.strip())
        
        # 嘗試 arXiv ID
        if self._is_arxiv_id(clean_id):
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"
        elif clean_id.startswith("10."):
            # DOI 格式
            url = f"https://api.semanticscholar.org/graph/v1/paper/{clean_id}"
        else:
            return ""
            
        try:
            resp = requests.get(url, params={"fields": "title"}, timeout=8)
            if resp.status_code == 200:
                title = resp.json().get("title", "")
                if title:
                    return self._clean_ligatures(title.strip())
        except Exception as e:
            print(f"[Metadata Manager] ⚠️ Semantic Scholar 查詢失敗: {e}")
        return ""

    def _fetch_title_from_crossref(self, doi: str) -> str:
        """
        透過 CrossRef API 查詢 DOI 對應的標題（覆蓋各類期刊/會議）
        """
        if not doi or not doi.startswith("10."):
            return ""
        import requests
        url = f"https://api.crossref.org/works/{doi}"
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "AcademicRAG/3.0 (mailto:dev@example.com)"})
            if resp.status_code == 200:
                data = resp.json().get("message", {})
                titles = data.get("title", [])
                if titles:
                    return self._clean_ligatures(titles[0].strip())
        except Exception as e:
            print(f"[Metadata Manager] ⚠️ CrossRef 查詢失敗: {e}")
        return ""

    def _extract_doi_from_pdf(self, doc) -> str:
        """
        從 PDF metadata 或首頁文字中萃取 DOI
        """
        # 1. 先嘗試 PDF metadata
        if doc.metadata:
            for key in ["doi", "DOI"]:
                doi = doc.metadata.get(key, "")
                if doi and doi.startswith("10."):
                    return doi
        
        # 2. 從首頁文字中搜尋 DOI 格式
        if len(doc) > 0:
            text = doc[0].get_text()
            doi_match = re.search(r'(?:doi[:\s]*|https?://doi\.org/)?(10\.\d{4,}/[^\s\]>]+)', text, re.IGNORECASE)
            if doi_match:
                doi = doi_match.group(1).rstrip('.')
                return doi
        return ""

    def _clean_ligatures(self, text: str) -> str:
        """清理常見字體合併符號（ligatures）與全型標點"""
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "’": "'",
            "‘": "'",
            "–": "-",
            "—": "-",
            "”": '"',
            "“": '"'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _extract_title_from_pdf(self, doc) -> str:
        """
        從 PDF 取得大標題，結合首頁 Abstract/Introduction 區塊切片邊界偵測與字型大小分析
        """
        try:
            # 1. 優先嘗試從 PDF Metadata 取得標題 (清理並驗證)
            meta_title = doc.metadata.get("title") if doc.metadata else None
            if meta_title:
                cleaned_meta = self._clean_ligatures(meta_title.strip())
                invalid_keywords = ["untitled", "layout", "template", "microsoft word", "latex", "pdf", "main", "document", "article"]
                if len(cleaned_meta) > 10 and not any(kw in cleaned_meta.lower() for kw in invalid_keywords):
                    if not re.match(r'^[\d\.\-_]+$', cleaned_meta):
                        return cleaned_meta

            # 2. 區塊切片邊界偵測：只分析 "Abstract" 或 "Introduction" 之前的區塊
            if len(doc) == 0:
                return ""
            page = doc[0]
            blocks = page.get_text("blocks")
            dict_blocks = page.get_text("dict")["blocks"]
            
            end_idx = len(blocks)
            for idx, b in enumerate(blocks):
                text = b[4].lower()
                # 碰到摘要或導論標題就停止 (以防混入本文或摘要本文)
                if re.search(r'\babstract\b', text) or re.search(r'\bintroduction\b', text):
                    end_idx = idx
                    break
                    
            if end_idx < 1:
                end_idx = 1
                
            candidate_dict_blocks = dict_blocks[:end_idx]
            
            # 3. 收集候選區塊中的所有文字 span 及字型大小
            text_instances = []
            for b in candidate_dict_blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            text = s["text"].strip()
                            font_size = s["size"]
                            # 過濾浮水印與極短詞
                            if len(text) > 2 and "arxiv" not in text.lower() and "preprint" not in text.lower():
                                text_instances.append((text, font_size))
                                
            if not text_instances:
                return ""
                
            # 4. 找出候選區塊中的最大字型
            max_size = max(inst[1] for inst in text_instances)
            
            # 5. 擷取字型在最大尺寸 80% 以上的所有文字 (以防標題換行)
            title_parts = []
            for text, size in text_instances:
                if size >= max_size * 0.80 and size > 10:
                    # 排除純數字/純標點行
                    if not re.match(r'^[\d\s\.\,\-\–\—\+\*]+$', text):
                        title_parts.append(text)
                        
            title = " ".join(title_parts).strip()
            title = self._clean_ligatures(title)
            title = re.sub(r'\s+', ' ', title)
            
            if len(title.split()) >= 2 and len(title) >= 10:
                words = title.split()
                if len(words) > 35:
                    title = " ".join(words[:35]) + "..."
                return title

        except Exception as e:
            print(f"Error extracting title: {e}")
        return ""

    def add_local_pdf_metadata(self, pdf_path: str, paper_id: str) -> str:
        """
        從本地 PDF 萃取基本元數據，並返回確定的 Title。
        
        標題解析優先順序 (Priority-Inverted Multi-Source Title Resolver):
          1. arXiv API（權威來源，100% 準確）
          2. Semantic Scholar API（覆蓋 arXiv + DOI，補強學術資料庫）
          3. CrossRef API（覆蓋各類期刊/會議 DOI）
          4. PDF 字型大小分析（本地萃取，各期刊格式不同，準確度較低）
          5. 清理檔名（終極 Fallback）
        """
        # 若快取中已有且為有效英文標題，直接沿用
        if paper_id in self.metadata:
            existing_title = self.metadata[paper_id].get("title", "")
            if self._validate_title(existing_title, paper_id):
                return existing_title
            
        title = ""
        abstract = ""
        doi = ""
        
        # ──────────────────────────────────────────────
        # 階段 A：線上權威來源查詢（不依賴 PDF 解析）
        # ──────────────────────────────────────────────
        
        # A1. arXiv API（最高優先權：arXiv 是 ground truth）
        if self._is_arxiv_id(paper_id):
            arxiv_title = self._fetch_title_from_arxiv(paper_id)
            if arxiv_title and self._validate_title(arxiv_title, paper_id):
                title = arxiv_title
                print(f"[Metadata Manager] ✅ arXiv API 取得官方標題: `{title}`")
        
        # A2. Semantic Scholar API（補強：覆蓋 arXiv + DOI）
        if not self._validate_title(title, paper_id):
            ss_title = self._fetch_title_from_semantic_scholar(paper_id)
            if ss_title and self._validate_title(ss_title, paper_id):
                title = ss_title
                print(f"[Metadata Manager] ✅ Semantic Scholar 取得標題: `{title}`")

        # ──────────────────────────────────────────────
        # 階段 B：從 PDF 萃取 DOI + Abstract + 本地標題
        # ──────────────────────────────────────────────
        try:
            with fitz.open(pdf_path) as doc:
                # B1. 從 PDF 萃取 DOI（供 CrossRef 查詢使用）
                doi = self._extract_doi_from_pdf(doc)
                
                # B2. 如果線上來源未命中，嘗試 PDF 字型分析萃取標題
                if not self._validate_title(title, paper_id):
                    detected_title = self._extract_title_from_pdf(doc)
                    if detected_title and self._validate_title(detected_title, paper_id):
                        title = detected_title
                        print(f"[Metadata Manager] 📄 PDF 字型分析取得標題: `{title}`")
                
                # B3. 讀取前兩頁尋找 Abstract
                text = ""
                for i in range(min(2, len(doc))):
                    text += doc[i].get_text()
            
            # 使用 Regex 簡單萃取 Abstract
            abs_match = re.search(r'(?i)abstract[\.:\s]*\n*(.*?)(?:\n\s*(?:introduction|1\.\s+introduction|1\.)|$)', text, re.DOTALL)
            if abs_match and len(abs_match.group(1).strip()) > 20:
                abstract = abs_match.group(1).replace('\n', ' ').strip()
            else:
                abstract = "No abstract found. (需進行 NLP 分析或 LLM 總結)"
        except Exception as e:
            print(f"⚠️ 萃取 PDF 元數據失敗: {e}")

        # A3. CrossRef API（透過 DOI 查詢，覆蓋各類期刊/會議格式）
        if not self._validate_title(title, paper_id) and doi:
            cr_title = self._fetch_title_from_crossref(doi)
            if cr_title and self._validate_title(cr_title, paper_id):
                title = cr_title
                print(f"[Metadata Manager] ✅ CrossRef (DOI: {doi}) 取得標題: `{title}`")
            # 如果 CrossRef 也沒拿到，嘗試用 DOI 查 Semantic Scholar
            if not self._validate_title(title, paper_id):
                ss_doi_title = self._fetch_title_from_semantic_scholar(doi)
                if ss_doi_title and self._validate_title(ss_doi_title, paper_id):
                    title = ss_doi_title
                    print(f"[Metadata Manager] ✅ Semantic Scholar (DOI) 取得標題: `{title}`")

        # ──────────────────────────────────────────────
        # 階段 C：終極 Fallback
        # ──────────────────────────────────────────────
        if not self._validate_title(title, paper_id):
            cleaned_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            cleaned_filename = cleaned_filename.replace('_', ' ').replace('-', ' ')
            cleaned_filename = re.sub(r'\s+', ' ', cleaned_filename).strip()
            if len(cleaned_filename) > 5 and not re.match(r'^[\d\.\-_]+$', cleaned_filename):
                title = cleaned_filename
            else:
                title = f"Paper ({paper_id})"
            print(f"[Metadata Manager] ⚠️ 所有來源皆未命中，採用清理後檔名: `{title}`")

        # 自動從 arXiv ID 擷取年份
        year = "N/A"
        clean_id = paper_id.split('v')[0]
        if '.' in clean_id:
            prefix = clean_id.split('.')[0]
            if len(prefix) >= 2 and prefix.isdigit():
                yy = int(prefix[:2])
                year = f"19{yy}" if yy >= 90 else f"20{yy:02d}"
        elif '/' in clean_id:
            num_part = clean_id.split('/')[-1]
            if len(num_part) >= 2 and num_part[:2].isdigit():
                yy = int(num_part[:2])
                year = f"19{yy}" if yy >= 90 else f"20{yy:02d}"

        self.metadata[paper_id] = {
            "title": title,
            "authors": ["Local Author"],
            "abstract": abstract[:1000],
            "categories": ["Local PDF"],
            "year": year,
            "keywords": [],
            "cluster_label": -1
        }
        if doi:
            self.metadata[paper_id]["doi"] = doi
        self.save_metadata()
        return title


    def get_metadata(self, paper_id: str) -> dict:
        return self.metadata.get(paper_id, {})
        
    def get_all_metadata(self) -> dict:
        return self.metadata

def backfill_all_keywords(top_k: int = 8):
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.semantic_scholar_client import enrich_paper

    manager = PaperMetadataManager()
    all_meta = manager.get_all_metadata()

    # 1. Backfill Semantic Scholar metadata
    arxiv_pattern = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
    old_sub_pattern = re.compile(r'^[a-z\-]+/\d{7}(v\d+)?$')
    
    for pid, meta in all_meta.items():
        is_arxiv = bool(arxiv_pattern.match(pid) or old_sub_pattern.match(pid))
        if is_arxiv and "citation_count" not in meta:
            print(f"[Metadata Manager] Querying Semantic Scholar for paper: {pid}")
            ss_data = enrich_paper(pid)
            if ss_data:
                meta.update(ss_data)
                try:
                    from src.semantic_scholar_client import get_recommendations
                    recs = get_recommendations(pid)
                    if recs:
                        meta["recommendations"] = recs
                except Exception as e:
                    print(f"Failed to get SS recommendations for {pid}: {e}")

    # 2. Backfill keywords via TF-IDF
    papers_needing_kw = {pid: m for pid, m in all_meta.items()
                         if not m.get("keywords")}
    if papers_needing_kw:
        corpus = [m.get("abstract", "") for m in papers_needing_kw.values()]
        if any(corpus):
            from src.stop_words import CUSTOM_STOP_WORDS
            tfidf = TfidfVectorizer(max_features=500, stop_words=CUSTOM_STOP_WORDS, ngram_range=(1,2))
            try:
                tfidf.fit(corpus)
                for pid, meta in papers_needing_kw.items():
                    abstract = meta.get("abstract", "")
                    if not abstract.strip():
                        continue
                    scores = tfidf.transform([abstract])
                    if not tfidf.get_feature_names_out().size:
                        continue
                    arr = scores.toarray()[0]
                    top_idx = arr.argsort()[-top_k:][::-1]
                    keywords = [tfidf.get_feature_names_out()[i] for i in top_idx if arr[i] > 0]
                    all_meta[pid]["keywords"] = keywords
                print(f"[Metadata Manager] Successfully backfilled keywords for {len(papers_needing_kw)} papers.")
            except Exception as e:
                print(f"[Metadata Manager] Keywords backfill error: {e}")

    manager.save_metadata()
