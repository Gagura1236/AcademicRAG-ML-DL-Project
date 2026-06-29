import os
import sys

import pickle
import numpy as np
import faiss
import torch
import jieba
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer, CrossEncoder

# 獲取專案根目錄，加入系統路徑以正確導入 config.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.gpu_utils import is_multi_gpu_enabled
from src.project_utils import get_project_paths

def hybrid_tokenize(text: str) -> list:
    """v2.6 Hybrid Tokenizer: 萃取英文與 LaTeX，剩餘中文用 jieba"""
    import re
    import jieba
    
    # 1. 抓取所有含有英數字、底線、減號、小數點或以 \ 開頭的 LaTeX 指令
    pattern = r"\\[a-zA-Z]+|[a-zA-Z0-9_\-\.]+"
    special_tokens = re.findall(pattern, text)
    
    # 2. 將原字串中的 special_tokens 替換為空白，剩餘的全是中文或標點
    cleaned_text = re.sub(pattern, " ", text)
    
    # 3. 對剩餘中文進行 jieba 切詞，並過濾掉無意義的純標點符號
    ignore_chars = set("，。！？、；：()[]{}<>=\"'_+~^\\|/")
    chinese_tokens = [t.strip() for t in jieba.cut(cleaned_text) if t.strip() and t.strip() not in ignore_chars]
    
    # 4. 合併並轉小寫，確保檢索時大小寫不敏感
    return [t.lower() for t in special_tokens] + chinese_tokens

class AcademicRAGEngine:
    """
    學術語意檢索 RAG 引擎：負責語意切片、向量嵌入、FAISS 向量存儲與 Cross-Encoder 二次重排。
    針對 M4 Mac 16GB 記憶體與 MPS 加速進行高度優化。
    """
    def __init__(self, project_name="default", collection_name="main"):
        self.project_name = project_name
        self.collection_name = collection_name
        self.project_paths = get_project_paths(project_name, collection_name)
        self.device = config.DEVICE
        self._use_rerank = True
        
        # 儲存 chunk 內文與元數據 (如來源頁碼、所屬論文、Bbox)
        self.chunks_metadata = []
        self.index = None
        self.bm25 = None
        self.tokenized_corpus = []
        
        # 加載現有向量庫
        self.load_index()

    @property
    def embedding_model(self):
        from src.model_manager import ModelManager
        def load_embedding():
            from sentence_transformers import SentenceTransformer
            print(f"\n[RAG Engine] 🔄 正在載入本地語意嵌入模型: {config.EMBEDDING_MODEL_NAME}...")
            model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=self.device.type)
            if self.device.type == "mps":
                model.half()
            print(f"[RAG Engine] Embedding 模型載入成功！(Device: {self.device}, FP16: {self.device.type == 'mps'})")
            return model
        try:
            return ModelManager().get_model("embedding", load_embedding)
        except Exception as e:
            print(f"⚠️ 載入 Embedding 模型失敗: {e}")
            return None

    @embedding_model.setter
    def embedding_model(self, value):
        from src.model_manager import ModelManager
        if value is None:
            ModelManager().release_model("embedding")
        else:
            with ModelManager().lock:
                ModelManager().models["embedding"] = value
                import time
                ModelManager().last_accessed["embedding"] = time.time()

    @property
    def rerank_model(self):
        if not self._use_rerank:
            return None
        from src.model_manager import ModelManager
        def load_rerank():
            from sentence_transformers import CrossEncoder
            print(f"[RAG Engine] 🔄 正在載入本地 Cross-Encoder 重排模型: {config.RERANK_MODEL_NAME}...")
            model = CrossEncoder(config.RERANK_MODEL_NAME, device=self.device.type)
            if self.device.type == "mps":
                model.model.half()
            print(f"[RAG Engine] Cross-Encoder 基礎模型載入成功！")
            
            # 動態偵測並加載 LoRA Adapter 權重
            adapter_name = "default_adapter"
            adapter_path = os.path.join(self.project_paths["lora_adapters_dir"], adapter_name)
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                print(f"[RAG Engine] 偵測到本地已微調之 LoRA Adapter ({adapter_name})，正在無縫整合載入...")
                from peft import PeftModel
                model.model = PeftModel.from_pretrained(model.model, adapter_path)
                model.model.to(self.device)
                if self.device.type == "mps":
                    model.model.half()
                model.model.eval()
                print(f"[RAG Engine] PEFT/LoRA Adapter 整合載入成功！已切換至微調重排模式 ✅")
            return model
        try:
            return ModelManager().get_model("rerank", load_rerank)
        except Exception as e:
            print(f"⚠️ 載入重排模型或 LoRA 權重失敗，檢索將不使用重排: {e}")
            return None

    @rerank_model.setter
    def rerank_model(self, value):
        from src.model_manager import ModelManager
        if value is None:
            ModelManager().release_model("rerank")
        else:
            with ModelManager().lock:
                ModelManager().models["rerank"] = value
                import time
                ModelManager().last_accessed["rerank"] = time.time()

    def chunk_document(self, layout_elements: list, paper_id: str, paper_title: str,
                       target_chunk_size: int = 512, overlap_size: int = 128) -> list:
        """
        v2.2 Semantic Chunking with Sliding Window Overlap + Sentence Boundary Detection
        (Śmigielski et al., 2026 - arXiv:2606.00881; Chen et al., 2023 - arXiv:2312.06648)
        """
        import re

        # Step 1: Flatten all layout elements into a single ordered text with page, bbox and image_path tracking
        sentences = []  # list of (sentence_text, page_number, bbox, image_path)
        for elem in layout_elements:
            page = elem["page"]
            content_type = elem["type"]
            text_content = elem.get("content_latex", elem["content"])
            bbox = elem.get("bbox", (0, 0, 0, 0))
            img_path = elem.get("image_path", "")

            if content_type == "equation":
                # Equations are atomic units - never split
                sentences.append((f" [Equation: {text_content}] ", page, bbox, img_path))
            else:
                # Split text into sentences using period, newline, or semicolon boundaries
                sent_splits = re.split(r'(?<=[.!?;])\s+|\n\n+', text_content)
                for s in sent_splits:
                    s = s.strip()
                    if s:
                        sentences.append((s, page, bbox, img_path))

        if not sentences:
            return []

        # Step 2: Build chunks using sliding window with sentence-level granularity
        chunks = []
        i = 0
        while i < len(sentences):
            current_text = ""
            current_pages = set()
            current_bboxes = []
            chunk_image_path = ""
            start_i = i

            # Fill chunk up to target_chunk_size, respecting sentence boundaries
            while i < len(sentences):
                sent_text, page, bbox, img_path = sentences[i]
                if len(current_text) + len(sent_text) + 1 > target_chunk_size and current_text:
                    break
                current_text += (" " + sent_text) if current_text else sent_text
                current_pages.add(page)
                current_bboxes.append({"page": page, "bbox": list(bbox)})
                if img_path:
                    chunk_image_path = img_path
                i += 1

            if current_text.strip():
                chunk_data = {
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "text": current_text.strip(),
                    "pages": sorted(list(current_pages)),
                    "bboxes": current_bboxes
                }
                if chunk_image_path:
                    chunk_data["image_path"] = chunk_image_path
                chunks.append(chunk_data)

            # Step 3: Sliding window overlap - backtrack by overlap_size worth of sentences
            if i < len(sentences):
                overlap_chars = 0
                backtrack = 0
                for j in range(i - 1, start_i, -1):
                    overlap_chars += len(sentences[j][0])
                    backtrack += 1
                    if overlap_chars >= overlap_size:
                        break
                i = i - backtrack if backtrack > 0 else i

        print(f"[RAG Engine] v2.2 Semantic Chunking: {len(sentences)} sentences -> {len(chunks)} chunks "
              f"(target={target_chunk_size}, overlap={overlap_size})")
        return chunks

    def add_documents(self, layout_elements: list, paper_id: str, paper_title: str, progress_callback=None):
        """
        將解析完的論文內容切片、向量化，並寫入 FAISS 向量資料庫
        """
        chunks = self.chunk_document(layout_elements, paper_id, paper_title)
        if not chunks:
            print(f"[RAG Engine] 沒有有效的文本切片，跳過寫入向量庫。")
            return
        
        # Bug 9: Guard against embedding_model being None
        if self.embedding_model is None:
            print("[RAG Engine] ✗ Embedding 模型未載入，無法建立向量索引。請檢查模型路徑與網路連線。")
            return
            
        # ==========================================================
        # [Route 1] LLM Smart Extraction: Extract keywords from Abstract
        # ==========================================================
        try:
            from src.agent import AcademicAgent
            import streamlit as st
            
            # Only run if an LLM is configured (or let AcademicAgent handle fallback)
            agent = AcademicAgent()
            
            # Approximate the abstract by combining the first 3 chunks (max 3000 chars)
            abstract_text = " ".join([c["text"] for c in chunks[:3]])[:3000]
            
            prompt = (
                "You are an AI researcher. Extract 3 to 5 core technical proper nouns or acronyms "
                "from the following text. Do not include generic words like 'model', 'paper', or 'method'. "
                "Output ONLY a comma-separated list of the extracted words, nothing else.\n\n"
                f"Text:\n{abstract_text}"
            )
            print(f"[RAG Engine] 🧠 正在呼叫 LLM 萃取 '{paper_title}' 的神級關鍵字...")
            if progress_callback:
                progress_callback(0.0, "🧠 正在使用 LLM 智慧萃取神級關鍵字...")
                
            response = agent.generate(prompt)
            
            # Clean up response (e.g. "LoRA, PEFT, LLMs" -> ["LoRA", "PEFT", "LLMs"])
            # Ignore sentences if the LLM hallucinated or returned an error block
            llm_keywords = []
            invalid_patterns = ['invalid_api_key', 'request_error', 'param', 'status', 'code', 'error', 'message', 'type', 'none', '401']
            for k in response.replace('\n', ',').split(','):
                k = k.strip().strip('"').strip("'").strip('`').strip('}').strip('{').strip(']').strip('[')
                k_lower = k.lower()
                if k and len(k.split()) <= 4 and len(k) > 2:  # Avoid full sentences or tiny words
                    # Check if keyword contains any error/API patterns
                    if not any(bad in k_lower for bad in invalid_patterns):
                        llm_keywords.append(k)
                    
            if llm_keywords:
                print(f"[RAG Engine] ✅ LLM 成功提煉關鍵字: {llm_keywords}")
                for c in chunks:
                    c["llm_keywords"] = llm_keywords
            else:
                print("[RAG Engine] ⚠️ LLM 未提供有效關鍵字（或被防禦機制過濾）。")
                
        except Exception as e:
            print(f"[RAG Engine] ⚠️ LLM 萃取失敗，跳過此步驟: {e}")
        # ==========================================================

        print(f"[RAG Engine] 正在對 {len(chunks)} 個學術切片進行語意向量化...")
        chunk_texts = [c["text"] for c in chunks]
        
        # 進行向量編碼 (支援 Multi-GPU 協同運算 Inference 加速)
        if is_multi_gpu_enabled(self.device):
            print(f"[RAG Engine] 🚀 偵測到 {config.NUM_GPUS} 張顯卡，啟用 SentenceTransformer 多卡協同 Inference！")
            
            # 重要：多進程池啟動前，主進程的模型必須在 CPU，否則 CUDA Context 會 Crash
            self.embedding_model.cpu()
            
            pool = self.embedding_model.start_multi_process_pool()
            try:
                embeddings = self.embedding_model.encode_multi_process(
                    chunk_texts, pool, batch_size=16 * config.NUM_GPUS
                )
            finally:
                self.embedding_model.stop_multi_process_pool(pool)
                # 恢復模型到預設裝置
                self.embedding_model.to(self.device)
            
            # encode_multi_process 輸出 numpy array，如果不是，強制轉換
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            # L2 Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        else:
            # Batch embedding encoding to feed progress updates to the UI callback
            batch_size = 16
            embeddings_list = []
            for start_idx in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[start_idx:start_idx+batch_size]
                batch_emb = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=False, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings_list.append(batch_emb)
                if progress_callback:
                    prog = min(1.0, (start_idx + len(batch_texts)) / len(chunk_texts))
                    progress_callback(prog, f"正在對段落進行語意向量化 {start_idx + len(batch_texts)}/{len(chunk_texts)}...")
            embeddings = np.vstack(embeddings_list)
        
        dimension = embeddings.shape[1]
        
        # 初始化 FAISS 索引
        if self.index is None:
            # 升級：使用 HNSW 確保大規模檢索效率，並使用 METRIC_INNER_PRODUCT 對齊餘弦相似度
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
            self.chunks_metadata = []
            self.tokenized_corpus = []
            
        # FP16 → float32 + nan/inf 防護
        embeddings = embeddings.astype('float32')
        if not np.isfinite(embeddings).all():
            print("[RAG Engine] ⚠️ FP16 overflow 偵測到 nan/inf，強制裁剪")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

        # 將新向量與元數據寫入
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)

        # 建立 BM25 索引 (Hybrid Search 準備)
        chunk_texts_for_bm25 = [c["text"] for c in chunks]
        print("[RAG Engine] 開始處理 BM25 Tokenization (避免 macOS 多進程死鎖，採用循序執行)...")
        
        new_tokens = [hybrid_tokenize(text) for text in chunk_texts_for_bm25]
            
        self.tokenized_corpus.extend(new_tokens)
        self.bm25 = BM25Plus(self.tokenized_corpus)

        # 存檔向量庫
        self.save_index()
        print(f"[RAG Engine] 成功新增論文 '{paper_title}' 到本地向量庫！當前總 Chunks 數: {len(self.chunks_metadata)}")

    def remove_paper(self, paper_id: str):
        """
        v2.2/v3.3: 從向量庫中移除指定論文的所有 Chunks，並重建 FAISS 索引。
        同時也會將本地的實體 PDF 檔案一併刪除。
        """
        import os
        pdf_dir = self.project_paths.get("pdf_dir", "")
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        
        pdf_removed = False
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"[RAG Engine] 已刪除實體 PDF 檔案: {pdf_path}")
                pdf_removed = True
            except Exception as e:
                print(f"[RAG Engine] ⚠️ 刪除實體 PDF 失敗: {e}")

        if not self.chunks_metadata:
            if not pdf_removed:
                print("[RAG Engine] 向量庫為空，且未找到實體 PDF。")
            return

        old_count = len(self.chunks_metadata)
        # Filter out chunks belonging to the target paper
        remaining = [c for c in self.chunks_metadata if c.get("paper_id") != paper_id]
        removed_count = old_count - len(remaining)

        if removed_count == 0:
            if pdf_removed:
                print(f"[RAG Engine] paper_id='{paper_id}' 尚未建立索引，但已移除實體 PDF。")
            else:
                print(f"[RAG Engine] 未找到 paper_id='{paper_id}' 的 chunks 與 PDF。")
            return

        print(f"[RAG Engine] 正在移除 paper_id='{paper_id}' 的 {removed_count} 個 chunks，並重建索引...")
        self._rebuild_from_chunks(remaining)
        print(f"[RAG Engine] 移除完成！剩餘 {len(self.chunks_metadata)} 個 chunks。")

    def rebuild_index_from_pdfs(self, pdf_dir: str = None, progress_callback=None, target_pdfs: list = None):
        """
        v3.2: 重建向量庫 — 重新切片並提取 VLM 圖表描述，將進度反饋給 Streamlit UI
        若提供 target_pdfs，則只會針對這些 PDF 進行完整解析與切片，其他既有論文會直接從舊 metadata 承襲。
        """
        import glob
        if pdf_dir is None:
            pdf_dir = self.project_paths.get("pdf_dir", "")

        all_pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        if target_pdfs is not None:
            pdf_files = [p for p in all_pdf_files if p in target_pdfs]
        else:
            pdf_files = all_pdf_files
        
        # 1. 備份現有的 metadata
        old_metadata = list(self.chunks_metadata) if self.chunks_metadata else []
        
        if not pdf_files and not old_metadata:
            print(f"[RAG Engine] 在 {pdf_dir} 中未找到任何 PDF 檔案，且無歷史索引。")
            return 0

        # 2. 清空現有索引
        self.index = None
        self.chunks_metadata = []
        self.tokenized_corpus = []
        self.bm25 = None

        print(f"[RAG Engine] v3.2 重建模式啟動！將針對 {len(pdf_files)} 篇 PDF 進行解析。")

        # Lazy import parser, metadata manager, and VLM helper
        from src.pdf_parser import ScientificPDFParser
        from src.metadata_extractor import PaperMetadataManager
        from src.vlm_helper import describe_figure
        
        parser = ScientificPDFParser()
        meta_mgr = PaperMetadataManager()
        
        processed_paper_ids = set()

        # 3. 優先用 PDF 重建 (使用最新切片策略 + VLM 圖表分析)
        for idx, pdf_path in enumerate(pdf_files):
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            if progress_callback:
                progress_callback(idx / len(pdf_files), f"正在處理論文 ({idx+1}/{len(pdf_files)}): {paper_id}")
            try:
                # 取得或從 PDF 萃取真實大標題
                actual_title = meta_mgr.add_local_pdf_metadata(pdf_path, paper_id)
                
                # 3a. 正常文字與公式切片
                elements = parser.parse_layout(pdf_path, progress_callback=progress_callback)
                processed = parser.process_pdf_equations(pdf_path, elements)
                
                # 3b. 擷取圖表 & 即時 VLM 描述
                if progress_callback:
                    progress_callback((idx + 0.5) / len(pdf_files), f"正在提取圖表與產生 VLM 描述: {paper_id}...")
                
                figures = parser.extract_figures_and_charts(pdf_path, paper_id)
                
                # 將圖表描述與原本文字混合
                mixed_elements = list(processed)
                for fig in figures:
                    caption = describe_figure(fig["path"])
                    mixed_elements.append({
                        "page": fig["page"],
                        "type": "text",
                        "content": f"[Figure Description] {caption}",
                        "bbox": fig["bbox"],
                        "image_path": fig["path"]
                    })
                
                # 3c. 寫入向量庫
                self.add_documents(mixed_elements, paper_id, actual_title, progress_callback=progress_callback)
                processed_paper_ids.add(paper_id)
            except Exception as e:
                print(f"[RAG Engine] ⚠️ PDF 重建時跳過 {paper_id}: {e}")

        # 4. 【持久化承襲】恢復那些 PDF 已經不見的舊論文
        missing_pdf_papers = {}
        for chunk in old_metadata:
            pid = chunk.get("paper_id")
            if pid and pid not in processed_paper_ids:
                if pid not in missing_pdf_papers:
                    missing_pdf_papers[pid] = []
                missing_pdf_papers[pid].append(chunk)
                
        if missing_pdf_papers:
            print(f"[RAG Engine] 💡 偵測到 {len(missing_pdf_papers)} 篇論文缺少原始 PDF，啟動自動承襲 (Fallback)...")
            all_fallback_chunks = []
            for pid, chunks in missing_pdf_papers.items():
                all_fallback_chunks.extend(chunks)
                
            # 將這些承襲的 chunk 重新進行向量化寫入
            chunk_texts = [c["text"] for c in all_fallback_chunks]
            embeddings = self.embedding_model.encode(chunk_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            embeddings = embeddings.astype('float32')
            embeddings = np.nan_to_num(embeddings)
            
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                
            self.index.add(embeddings)
            self.chunks_metadata.extend(all_fallback_chunks)
            
            # 更新 BM25
            new_tokens = [hybrid_tokenize(c["text"]) for c in all_fallback_chunks]
            self.tokenized_corpus.extend(new_tokens)
            self.bm25 = BM25Plus(self.tokenized_corpus)
            self.save_index()
            print(f"[RAG Engine] ✅ 成功承襲 {len(all_fallback_chunks)} 個歷史 chunks。")

        print(f"[RAG Engine] ✅ 向量庫重建與承襲完成！共 {len(self.chunks_metadata)} 個 chunks。")
        return len(processed_paper_ids) + len(missing_pdf_papers)

    def clear_index(self):
        """
        清空目前載入的所有向量資料庫與中繼資料，並覆寫本地存檔。
        """
        self.index = None
        self.chunks_metadata = []
        self.tokenized_corpus = []
        self.bm25 = None
        
        self.save_index()
        print("[RAG Engine] 向量庫已完全清除。")


    def _rebuild_from_chunks(self, chunks: list):
        """
        Internal helper: re-encode a list of chunk dicts and rebuild FAISS + BM25 from scratch.
        """
        if self.embedding_model is None:
            print("[RAG Engine] ✗ Embedding 模型未載入，無法重建索引。")
            return

        if not chunks:
            self.index = None
            self.chunks_metadata = []
            self.tokenized_corpus = []
            self.bm25 = None
            self.save_index()
            return

        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings.astype('float32')
        embeddings = np.nan_to_num(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = 64
        self.index.add(embeddings)
        self.chunks_metadata = chunks
        self.tokenized_corpus = [hybrid_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Plus(self.tokenized_corpus)
        self.save_index()

    def save_index(self):
        """
        將 FAISS 索引與 Meta 數據原子性儲存至硬碟（先寫 .tmp，再 os.replace，防止斷電損毀）
        """
        vector_db_dir = self.project_paths["vector_db_dir"]
        os.makedirs(vector_db_dir, exist_ok=True)
        index_path = os.path.join(vector_db_dir, "faiss.index")
        meta_path  = os.path.join(vector_db_dir, "metadata.pkl")
        bm25_path  = os.path.join(vector_db_dir, "bm25.pkl")

        if self.index is not None:
            # 原子性寫入：先寫 .tmp 再 replace，避免部分寫入造成下次載入失敗
            faiss.write_index(self.index, index_path + ".tmp")
            os.replace(index_path + ".tmp", index_path)

            with open(meta_path + ".tmp", 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            os.replace(meta_path + ".tmp", meta_path)

            with open(bm25_path + ".tmp", 'wb') as f:
                pickle.dump((self.tokenized_corpus, self.bm25), f)
            os.replace(bm25_path + ".tmp", bm25_path)

    def load_index(self):
        """
        從硬碟加載現有的 FAISS 索引與元數據
        """
        vector_db_dir = self.project_paths["vector_db_dir"]
        index_path = os.path.join(vector_db_dir, "faiss.index")
        meta_path = os.path.join(vector_db_dir, "metadata.pkl")
        bm25_path = os.path.join(vector_db_dir, "bm25.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            print("[RAG Engine] 未發現現有向量庫，初始化新庫。")
            return

        # 步驟 1：獨立載入核心索引（失敗才重建，不連帶清除 BM25）
        try:
            self.index = faiss.read_index(index_path)
                
            with open(meta_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            print(f"[RAG Engine] 本地向量庫載入成功！載入 {len(self.chunks_metadata)} 個 Chunks。")
        except Exception as e:
            print(f"⚠️ 核心向量庫損毀，重建: {e}")
            self.index = None
            self.chunks_metadata = []
            self.tokenized_corpus = []
            self.bm25 = None
            return

        # 步驟 2：獨立載入 BM25（失敗自動重建，不影響已成功的 FAISS）
        try:
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.tokenized_corpus, self.bm25 = pickle.load(f)
                if len(self.tokenized_corpus) != len(self.chunks_metadata):
                    raise ValueError(f"BM25 corpus 長度({len(self.tokenized_corpus)}) != metadata({len(self.chunks_metadata)})，強制重建")
            else:
                raise FileNotFoundError("bm25.pkl 不存在")
        except Exception as e:
            print(f"⚠️ BM25 索引重建中（原因: {e}）")
            self.tokenized_corpus = [hybrid_tokenize(c["text"]) for c in self.chunks_metadata]
            self.bm25 = BM25Plus(self.tokenized_corpus)
            self.save_index()

    def expand_query_with_llm(self, query: str) -> list:
        """
        v2.2 LLM Query Expansion (Xia et al., 2024 - arXiv:2410.13765)
        Generate 3 semantically equivalent sub-queries using LLM to improve recall.
        Returns a list of expanded queries (including the original).
        """
        try:
            from src.agent import AcademicAgent
            agent = AcademicAgent()
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Query Expansion Agent for academic paper search.
Given a user query, generate exactly 3 alternative search queries that capture the same intent using different terminology, synonyms, or related concepts.
Return ONLY a JSON array of 3 strings. No extra text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Original query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            import json, re
            response = agent.generate(prompt, max_tokens=256, temp=0.0)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                expanded = json.loads(match.group(0))
                expanded = [q for q in expanded if isinstance(q, str) and q.strip()]
                print(f"[RAG Engine] Query Expansion: {query} -> {expanded}")
                return [query] + expanded[:3]
        except Exception as e:
            print(f"[RAG Engine] Query Expansion 失敗，使用原始查詢: {e}")
        return [query]

    def _compute_dynamic_alpha(self, dense_top1_sim: float, bm25_top1_score: float, query_len: int) -> float:
        """
        v2.2 Dynamic Alpha Tuning (Hsu & Tzeng, 2025 - arXiv:2503.23013)
        Dynamically compute the optimal Dense vs Sparse weight per-query.
        Returns alpha in [0.3, 0.9]: higher = more weight on Dense.
        """
        # Under IndexFlatIP, dense_top1_sim is already a cosine similarity in [-1, 1]
        dense_sim = max(0.0, min(1.0, dense_top1_sim))
        # Normalize BM25 score dynamically based on query length (approximate query total IDF)
        # Average IDF per word is assumed to be around 3.0
        ref_score = max(3.0, query_len * 3.0)
        sparse_sim = (bm25_top1_score / (bm25_top1_score + ref_score)) if bm25_top1_score > 0.0 else 0.0
        
        # If Dense retriever is more confident, weight it higher and vice versa
        total = dense_sim + sparse_sim
        if total < 1e-6:
            return 0.5  # fallback to equal weight
        alpha = max(0.3, min(0.9, dense_sim / total))
        return alpha

    def _load_project_index_standalone(self, project_name):
        paths = get_project_paths(project_name)
        vector_db_dir = paths["vector_db_dir"]
        index_path = os.path.join(vector_db_dir, "faiss.index")
        meta_path = os.path.join(vector_db_dir, "metadata.pkl")
        bm25_path = os.path.join(vector_db_dir, "bm25.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(bm25_path)):
            return None, None, None
            
        try:
            index = faiss.read_index(index_path)
            if not hasattr(index, 'hnsw'):
                print(f"[RAG Engine] 專案 {project_name} 索引為舊版，需手動切換過去以觸發升級重建。")
                return None, None, None
                
            with open(meta_path, 'rb') as f:
                chunks = pickle.load(f)
            with open(bm25_path, 'rb') as f:
                _, bm25 = pickle.load(f)
            return index, chunks, bm25
        except Exception as e:
            print(f"[RAG Engine] 載入專案 {project_name} 索引失敗: {e}")
            return None, None, None

    def search(self, query: str, top_k: int = 5, use_query_expansion: bool = False, mode: str = "Balanced", target_projects: list = None) -> list:
        """
        v3.9.2 混合語意搜尋：
        - Fast: 僅 FAISS 向量檢索，無重排，反應速度極快。
        - Balanced: 混合檢索 + Rerank，反應時間與準確度平衡。
        - Thorough: 啟動 LLM 查詢擴展 + RRF 融合 + Cross-Encoder 重排，精準度最高。
        - Agentic: 以 Thorough 為基礎，在 Step 4 由自主 ReAct Loop 進階處理。
        """
        if self.index is None or not self.chunks_metadata:
            print("[RAG Engine] 向量資料庫為空，無法檢索！")
            return []
        
        if self.embedding_model is None:
            print("[RAG Engine] ✗ Embedding 模型未載入，無法執行檢索。")
            return []

        if target_projects is None or not target_projects:
            target_projects = [self.project_name]
            
        # ⚡️ 快速檢索模式 (Federated Fast Search)
        if mode == "Fast":
            query_vector = self.embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
            query_vector = np.nan_to_num(query_vector)
            
            all_results = []
            for p_name in target_projects:
                if p_name == self.project_name:
                    p_index, p_chunks = self.index, self.chunks_metadata
                else:
                    p_index, p_chunks, _ = self._load_project_index_standalone(p_name)
                    
                if p_index is None or not p_chunks: continue
                
                dense_distances, dense_indices = p_index.search(query_vector, top_k)
                for rank, idx in enumerate(dense_indices[0]):
                    if idx == -1: continue
                    candidate = dict(p_chunks[idx])
                    candidate["score"] = float(dense_distances[0][rank])
                    candidate["dense_dist"] = float(dense_distances[0][rank])
                    candidate["bm25_score"] = 0.0
                    candidate["is_fast"] = True
                    candidate["project"] = p_name
                    all_results.append(candidate)
                    
            # Sort cross-project results by distance
            all_results.sort(key=lambda x: x["score"])
            return all_results[:top_k]
        
        # Balanced / Thorough / Agentic 模式下的查詢擴展決定
        use_qe = use_query_expansion or (mode in ["Thorough", "Agentic"])
        
        # v3.0: Query Reformulation & HyDE (Hypothetical Document Embeddings)
        refined_query = None
        hyde_doc = None
        if use_qe:
            try:
                from src.agent import AcademicAgent
                agent = AcademicAgent()
                
                refined_query = agent.reformulate_query(query)
                hyde_doc = agent.generate_hyde_document(refined_query)
                
                try:
                    import streamlit as st
                    st.session_state["last_sub_queries"] = [query, refined_query, hyde_doc]
                except Exception:
                    pass
            except Exception as e:
                print(f"[RAG Engine] HyDE / Reformulation 失敗，退回原始查詢: {e}")
        
        # 分流檢索路徑 (Separated search paths)
        dense_queries = [query]
        if use_qe and refined_query and hyde_doc:
            dense_queries.extend([refined_query, hyde_doc])
            
        sparse_queries = [query]
        if use_qe and refined_query:
            sparse_queries.append(refined_query)
            
        fusion_candidates = []
        orig_tokenized = hybrid_tokenize(query)
        orig_query_vector = self.embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        orig_query_vector = np.nan_to_num(orig_query_vector)
        
        # --- Federated Search Loop ---
        for p_name in target_projects:
            if p_name == self.project_name:
                p_index, p_chunks, p_bm25 = self.index, self.chunks_metadata, self.bm25
            else:
                p_index, p_chunks, p_bm25 = self._load_project_index_standalone(p_name)
                
            if p_index is None or not p_chunks:
                continue

            # 3. 計算基於原始查詢的 Dynamic Alpha
            orig_distances, orig_indices = p_index.search(orig_query_vector, 1)
            orig_dense_top1_sim = float(orig_distances[0][0]) if orig_indices[0][0] != -1 else 0.0
            
            orig_bm25_scores = p_bm25.get_scores(orig_tokenized) if p_bm25 else np.zeros(len(p_chunks))
            if len(orig_bm25_scores) != len(p_chunks):
                orig_bm25_scores = np.zeros(len(p_chunks))
            orig_bm25_top1_score = float(np.max(orig_bm25_scores)) if len(orig_bm25_scores) > 0 else 0.0
            
            alpha = self._compute_dynamic_alpha(orig_dense_top1_sim, orig_bm25_top1_score, len(orig_tokenized))
            
            rrf_scores = {}
            rrf_k = 60
            candidate_count = min(top_k * 4, len(p_chunks))
            dense_score_map = {}
            sparse_score_map = {}
            
            # 1. 執行 Dense Retrieval (FAISS)
            for dq in dense_queries:
                q_vec = self.embedding_model.encode([dq], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
                q_vec = np.nan_to_num(q_vec)
                dense_distances, dense_indices = p_index.search(q_vec, candidate_count)
                for rank, idx in enumerate(dense_indices[0]):
                    if idx == -1: continue
                    rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + alpha / (rrf_k + rank + 1)
                    dense_score_map[int(idx)] = max(dense_score_map.get(int(idx), -1.0), float(dense_distances[0][rank]))

            # 2. 執行 Sparse Retrieval (BM25)
            for sq in sparse_queries:
                tokenized_query = hybrid_tokenize(sq)
                bm25_scores = p_bm25.get_scores(tokenized_query) if p_bm25 else np.zeros(len(p_chunks))
                if len(bm25_scores) != len(p_chunks):
                    bm25_scores = np.zeros(len(p_chunks))
                sparse_indices = np.argsort(bm25_scores)[::-1][:candidate_count]
                for rank, idx in enumerate(sparse_indices):
                    if bm25_scores[idx] <= 0: continue
                    rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + (1.0 - alpha) / (rrf_k + rank + 1)
                    sparse_score_map[int(idx)] = max(sparse_score_map.get(int(idx), 0.0), float(bm25_scores[idx]))
                
            # 將該專案的 RRF 候選加入全域候選池
            sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
            for idx, rrf in sorted_rrf[:candidate_count]:
                candidate = dict(p_chunks[idx])
                candidate["idx"] = idx
                candidate["dense_dist"] = dense_score_map.get(idx, 0.0)
                candidate["bm25_score"] = sparse_score_map.get(idx, 0.0)
                candidate["rrf_score"] = rrf
                candidate["project"] = p_name
                fusion_candidates.append(candidate)
                
        if not fusion_candidates:
            print("[RAG Engine] ⚠️ 無檢索候選段落。")
            return []
            
        # 4. 使用 Cross-Encoder 進行精確二次重排 (Re-ranking)
        if self.rerank_model is not None and mode != "Fast":
            # [Kernel Optimization] Candidate Pruning Funnel: 嚴格控制丟給 Cross-Encoder 的候選數量 (Top-15)
            # 這能為重排省下將近 70%~80% 的無效運算時間
            fusion_candidates = sorted(fusion_candidates, key=lambda x: x.get("rrf_score", 0.0), reverse=True)[:15]
            
            # 建立輸入格式: [(query, doc1_text), (query, doc2_text), ...]
            pairs = [(query, c["text"]) for c in fusion_candidates]
            
            # 針對 PEFT LoRA 模型在 SentenceTransformers 預測時的相容性處理
            is_peft = hasattr(self.rerank_model.model, "peft_config") or "peft" in str(type(self.rerank_model.model)).lower()
            if is_peft:
                # Bug 11 Fix: split tuple list into separate lists for tokenizer compatibility
                queries_list, texts_list = zip(*pairs)
                features = self.rerank_model.tokenizer(
                    list(queries_list), list(texts_list),
                    padding=True, truncation=True, return_tensors="pt", max_length=512
                )
                features = {k: v.to(self.device) for k, v in features.items()}
                with torch.no_grad():
                    logits = self.rerank_model.model(**features).logits
                    if logits.dim() > 1 and logits.shape[1] == 2:
                        # 二元分類輸出 (N, 2)：取 label=1 (相關) 的 softmax 機率
                        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    elif logits.dim() > 1 and logits.shape[1] == 1:
                        scores = torch.sigmoid(logits.view(-1)).cpu().numpy()
                    else:
                        scores = torch.sigmoid(logits).cpu().numpy()
            else:
                # M-6 Fix: normalize non-PEFT raw logits to [0,1] with sigmoid for a unified score domain
                raw_scores = self.rerank_model.predict(pairs)
                scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
            
            # 根據重排分數降序排列
            sorted_indices = np.argsort(scores)[::-1]
            ranked_results = []
            seen_texts = set()
            
            for rank_idx in sorted_indices:
                candidate = dict(fusion_candidates[rank_idx])
                if candidate["text"] in seen_texts:
                    continue
                seen_texts.add(candidate["text"])
                
                candidate["score"] = float(scores[rank_idx])
                ranked_results.append(candidate)
                if len(ranked_results) >= top_k:
                    break
            
            if not ranked_results:
                print(f"[RAG Engine] ⚠️ Cross-Encoder 重排結果為空（所有候選分數過低或重複過濾），退回 RRF 排序結果。")
                # Fallback: return top-k by RRF score so the user still sees results
                fusion_candidates_sorted = sorted(fusion_candidates, key=lambda x: x.get("rrf_score", 0.0), reverse=True)
                for c in fusion_candidates_sorted[:top_k]:
                    c["score"] = min(1.0, float(c.get("rrf_score", 0.0)) / 0.0164)
                    c["is_rrf"] = True
                    c["rerank_fallback"] = True
                    ranked_results.append(c)
                
            return ranked_results
        else:
            # 若無重排模型或為 Fast，直接根據所有專案的 RRF 進行全域排序
            fusion_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
            results = []
            for candidate in fusion_candidates[:top_k]:
                # 理論最大 RRF 約為 1/61 (0.0164)。將其正規化至 0~1 區間供 UI 顯示
                normalized_rrf = min(1.0, float(candidate["rrf_score"]) / 0.0164)
                candidate["score"] = normalized_rrf
                candidate["is_rrf"] = True
                results.append(candidate)
            
            # [Kernel Optimization] 強制釋放 MPS 快取
            if self.device.type == "mps":
                torch.mps.empty_cache()
                
            return results

if __name__ == "__main__":
    # 簡單單體測試
    engine = AcademicRAGEngine()
    print("RAG 引擎初始化成功，當前已加載索引維度與狀態。")
