import os
import sys
import gc
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# 獲取專案根目錄，加入系統路徑以正確導入 config.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.embedding_search import AcademicRAGEngine
from src.evaluation import RAGEvaluationSuite
from src.gpu_utils import free_memory, extract_logits, is_multi_gpu_enabled
from src.project_utils import get_project_paths

class TextPairDataset(Dataset):
    """
    RAG 微調專用數據集格式：(Query, Text) 雙輸入對，附帶相關性二元 Label
    """
    def __init__(self, pairs, labels, tokenizer, max_len=512):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, text = self.pairs[idx]
        label = self.labels[idx]
        
        # Tokenize pair
        inputs = self.tokenizer(
            query,
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 擠壓維度以移除 batch_size=1
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item

class InfoNCELoss(nn.Module):
    """
    Supervised Contrastive Learning (InfoNCE) Loss.
    Pulls positive pairs together and pushes negative pairs apart in the latent space.
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, logits, labels):
        # We need pairs of embeddings or logits. Since our model outputs a single scalar logit per pair in cross-encoder setup,
        # we can interpret logits as similarity scores.
        # Alternatively, if we just have logits for each pair, we can form contrastive loss across the batch.
        device = logits.device
        # Normalize logits by temperature
        logits = logits / self.temperature
        
        pos_mask = (labels == 1.0)
        neg_mask = (labels == 0.0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        
        # 使用 logsumexp 技巧提升數值穩定性，避免 exp(100) 導致的 NaN
        loss = 0.0
        for pos_logit in pos_logits:
            all_logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
            loss += -(pos_logit - torch.logsumexp(all_logits, dim=0))
            
        return loss / max(pos_logits.size(0), 1)

class AcademicLoRATuner:
    """
    學術級 LoRA 微調器：針對 M4 MPS 加速進行優化，防範 LaTeX 標記過擬合。
    """
    def __init__(self, rag_engine: AcademicRAGEngine, project_name="default", use_mps=False):
        self.project_name = project_name
        self.project_paths = get_project_paths(project_name)
        self.rag = rag_engine
        self.device = config.DEVICE
        # Apple Silicon MPS 記憶體配置不穩定，訓練時預設強行使用 CPU，但如果 use_mps=True 則啟用 MPS 加速
        if self.device.type == "mps" and not use_mps:
            print("[Fine-Tuning] Apple Silicon MPS 偵測。為防範 Metal 驅動與編譯器隨機段錯誤 (SIGSEGV)，微調預設使用 CPU 運行（不影響檢索速度）")
            self.train_device = torch.device("cpu")
        else:
            self.train_device = self.device
            
        self.eval_suite = RAGEvaluationSuite(self.rag)
        self.agent = None


    def _load_agent(self):
        if self.agent is None:
            from src.agent import AcademicAgent
            self.agent = AcademicAgent()
        
    def _stratified_split(self, pairs, labels, val_ratio=0.2):
        """
        按類別分層分割（Stratified Split）：確保訓練集與驗證集的正負比例一致。
        避免隨機分割導致驗證集全部是負樣本而無法評估正樣本的表現。
        """
        pos_indices = [i for i, l in enumerate(labels) if l == 1.0]
        neg_indices = [i for i, l in enumerate(labels) if l == 0.0]

        random.shuffle(pos_indices)
        random.shuffle(neg_indices)

        n_pos_val = max(1, int(len(pos_indices) * val_ratio))
        n_neg_val = max(1, int(len(neg_indices) * val_ratio))

        val_indices = set(pos_indices[:n_pos_val] + neg_indices[:n_neg_val])
        train_indices = [i for i in range(len(labels)) if i not in val_indices]
        val_indices = list(val_indices)

        train_pairs  = [pairs[i]  for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_pairs    = [pairs[i]  for i in val_indices]
        val_labels   = [labels[i] for i in val_indices]

        return train_pairs, train_labels, val_pairs, val_labels

    def _compute_val_auc(self, model, tokenizer, val_pairs, val_labels):
        """
        在驗證集上計算 ROC-AUC（Wilcoxon-Mann-Whitney）作為 Early Stopping 的指標。
        模型切換為 eval 模式並在 no_grad 內執行以節省記憶體。
        批次化處理以防止大量序列推論造成 MPS 記憶體溢出。
        """
        import numpy as np
        model.eval()
        all_logits = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch_pairs = val_pairs[i:i + batch_size]
                queries = [p[0] for p in batch_pairs]
                texts = [p[1] for p in batch_pairs]
                enc = tokenizer(queries, texts, max_length=512, padding=True,
                                truncation=True, return_tensors="pt")
                enc = {k: v.to(self.train_device) for k, v in enc.items()}
                out = model(**enc)
                logit = extract_logits(out)
                if logit.dim() > 1 and logit.shape[1] == 2:
                    scores = torch.softmax(logit, dim=1)[:, 1].cpu().numpy().tolist()
                elif logit.dim() > 1 and logit.shape[1] == 1:
                    scores = torch.sigmoid(logit.view(-1)).cpu().numpy().tolist()
                else:
                    scores = torch.sigmoid(logit).cpu().numpy().tolist()
                all_logits.extend(scores)
                
                # 釋放 GPU/MPS 顯存
                del enc
                del out
                del logit
                free_memory(self.train_device)
        model.train()

        scores = np.array(all_logits)
        labels = np.array(val_labels)
        pos_s = scores[labels == 1]
        neg_s = scores[labels == 0]
        if len(pos_s) == 0 or len(neg_s) == 0:
            return 0.5
        hits = sum(1 for p in pos_s for n in neg_s if p > n)
        return hits / (len(pos_s) * len(neg_s))


    def generate_training_data(self, use_llm=False):
        """
        利用已對齊的 LaTeX 結構化 Chunks，自動進行「硬負樣本(Hard Negatives)對抗生成」與數據增強。
        若開啟 use_llm=True，將啟動 LLM 進行 Self-Instruct RAG 高階合成數據生成。
        """
        print("[Fine-Tuning] 正在自動挖掘高維度對齊學術數據...")
        
        meta_path = os.path.join(self.project_paths["vector_db_dir"], "metadata.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("向量元數據 metadata.pkl 不存在，請先上傳 PDF 論文或下載 arXiv 以索引資料！")
            
        with open(meta_path, 'rb') as f:
            chunks = pickle.load(f)

        pairs = []
        labels = []

        # 5. 可選：混入使用者顯式反饋標註 (Explicit User Feedback)
        fb_file = os.path.join(self.project_paths["project_dir"], "feedback_dataset.json")
        if os.path.exists(fb_file):
            try:
                import json
                with open(fb_file, "r", encoding="utf-8") as f:
                    fb_data = json.load(f)
                
                print(f"[Fine-Tuning] 🔄 載入 {len(fb_data)} 筆真實使用者回饋 (RLHF) 作為最高權重訓練資料...")
                # 將回饋資料重複加入以提高權重 (Oversample RLHF)
                for item in fb_data:
                    for _ in range(2): 
                        pairs.append((item["query"], item["chunk_text"]))
                        labels.append(float(item["label"]))
            except Exception as e:
                print(f"[Fine-Tuning] ⚠️ 讀取 feedback_dataset.json 失敗: {e}")

        # DPO Preference Signal feedback loop integration
        try:
            from src.preference_store import load_preference_pairs
            pref_pos, pref_neg = load_preference_pairs(self.project_paths)
            if pref_pos or pref_neg:
                print(f"[Fine-Tuning] 📊 載入 DPO 偏好資料: +{len(pref_pos)} 正樣本對, +{len(pref_neg)} 負樣本對")
                # Positive pairs
                for q, t in pref_pos:
                    pairs.append((q, t))
                    labels.append(1.0)
                # Negative pairs
                for q, t in pref_neg:
                    pairs.append((q, t))
                    labels.append(0.0)
        except Exception as e:
            print(f"[Fine-Tuning] ⚠️ 載入 DPO 偏好資料失敗: {e}")

        if use_llm:
            print("[Fine-Tuning] 🚀 啟動 LLM Self-Instruct RAG 合成數據生成...")
            self._load_agent()
            
            # 挑選 15-20 個富含公式或字數較多的 chunks
            eligible_chunks = [c for c in chunks if len(c.get("text", "")) > 150]
            sample_size = min(20, len(eligible_chunks))
            if sample_size == 0:
                raise ValueError("知識庫中找不到足夠長度的段落來產生訓練資料。")
                
            selected_chunks = random.sample(eligible_chunks, sample_size)
            
            for i, chunk in enumerate(selected_chunks):
                text = chunk["text"]
                print(f"  -> [{i+1}/{sample_size}] 正在由 LLM 為此段落生成專屬學術問答對...")
                
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI Data Generator. Read the academic text below and generate 3 distinct questions that can be perfectly answered by this text.
The questions should vary in difficulty (e.g., basic concept, mathematical formula, application).
Return ONLY a valid JSON array of strings containing the 3 questions.
Format: ["Question 1?", "Question 2?", "Question 3?"]
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: {text[:1500]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                try:
                    response = self.agent.generate(prompt, max_tokens=300, temp=0.3)
                    import re, json
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                    if match:
                        generated_queries = json.loads(match.group(0))
                        
                        # 完美正樣本 (Generated Query, Original Chunk)
                        for q in generated_queries:
                            pairs.append((q, text))
                            labels.append(1.0)
                            
                            # 找硬負樣本 (隨機選一篇「不同論文」的 chunk，或者長度相似但不相關的 chunk)
                            neg_candidates = [c["text"] for c in chunks if c["paper_id"] != chunk["paper_id"] and len(c["text"]) > 100]
                            if neg_candidates:
                                pairs.append((q, random.choice(neg_candidates)))
                                labels.append(0.0)
                                pairs.append((q, random.choice(neg_candidates)))
                                labels.append(0.0)
                except Exception as e:
                    print(f"     [Error] LLM 生成失敗: {e}")
                    
            print(f"[Fine-Tuning] LLM 擴增完畢！合成 {len(pairs)} 筆完美對齊的訓練數據。")
            return pairs, labels

        # ====================================================
        # 原版 TF-IDF 基礎關鍵字擴增 + BM25 Hard Negative Mining (v2.2)
        # ====================================================
        base_queries = self.eval_suite.eval_dataset
        
        # v2.2: 準備全局 BM25 索引以挖掘硬負樣本
        from rank_bm25 import BM25Okapi
        from src.embedding_search import hybrid_tokenize
        tokenized_corpus = [hybrid_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        augmented_queries = []
        for item in base_queries:
            q = item["query"]
            keywords = item["gt_keywords"]
            target_paper = item.get("target_paper_id", "")
            
            # 手動設計高品質學術同義問句 (資料增強)
            variations = [
                q,
                f"Can you provide the formula for {keywords[0]}?",
                f"Show me the mathematical expression of {keywords[0]} in the paper.",
                f"Explain how we calculate {keywords[0]} mathematically.",
                f"Find the equations related to {keywords[0]}.",
                f"What is the derivation or formulation of {keywords[0]}?",
                f"Describe the system implementation and formula for {keywords[0]}."
            ]
            
            for v in variations:
                augmented_queries.append({
                    "query": v,
                    "keywords": keywords,
                    "target_paper_id": target_paper
                })
                
        print(f"[Fine-Tuning] 掃描 {len(chunks)} 個 Chunks，正針對 {len(augmented_queries)} 個增強問句對進行硬負樣本挖掘...")
        
        for q_item in augmented_queries:
            query = q_item["query"]
            keywords = q_item["keywords"]
            target_id = q_item["target_paper_id"]
            
            # 1. 取得該 query 的 BM25 分數
            tokenized_query = hybrid_tokenize(query)
            bm25_scores = bm25.get_scores(tokenized_query)
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
            if max_bm25 == 0: max_bm25 = 1.0

            # 2. 取得該 query 的 Dense (向量) 檢索候選結果 (用於混合硬負樣本挖掘)
            dense_top_k = min(30, len(chunks))
            dense_results = []
            try:
                if self.rag is not None and getattr(self.rag, "index", None) is not None:
                    dense_results = self.rag.search(query, top_k=dense_top_k, mode="Fast")
            except Exception as e:
                print(f"  [Negative Mining Warning] Dense search failed: {e}")
            
            # 建立稠密檢索名次對照字典
            dense_ranks = {item["text"]: rank for rank, item in enumerate(dense_results)}
            
            positives = []
            negatives_candidates = [] # 儲存格式: (text, hybrid_score)
            easy_negatives = []
            
            for idx, chunk in enumerate(chunks):
                text = chunk["text"]
                paper_id = chunk["paper_id"]
                score = bm25_scores[idx]
                
                # 計算關鍵字重合數
                kw_matches = sum(1 for kw in keywords if kw.lower() in text.lower())
                has_equation = "[Equation:" in text
                
                is_target_paper = (paper_id == target_id) if target_id else False
                
                # 正樣本判斷邏輯：有關鍵字，或屬於目標論文且有公式
                if (is_target_paper and kw_matches >= max(1, len(keywords) // 2) and has_equation) or (not target_id and kw_matches >= max(1, len(keywords) // 2)):
                    positives.append(text)
                elif not is_target_paper and kw_matches >= max(1, len(keywords) // 2):
                    positives.append(text)
                else:
                    # 負樣本：計算 Hybrid Score 硬度
                    norm_bm25 = score / max_bm25
                    
                    # Dense 排名特徵分數：第一名得 1.0，第 30 名得 0.03，未入選得 0.0
                    dense_score = 0.0
                    if text in dense_ranks:
                        rank = dense_ranks[text]
                        # ⚠️ 關鍵修復：排名太靠前的 Dense 檢索結果極可能是 False Negative（也就是真的相關，只是沒被前面的 keyword heuristic 抓到）
                        # 如果把它當作硬負樣本，會導致模型發生「語意崩潰 (Representation Collapse)」，喪失原本的排序能力。
                        if rank < 3:
                            continue
                        dense_score = 1.0 - ((rank - 3) / max(1, dense_top_k - 3))
                    
                    # 混合分數：各佔 50% 權重 (Hybrid Hard Negative score)
                    hybrid_score = 0.5 * norm_bm25 + 0.5 * dense_score
                    
                    if hybrid_score > 0.02:
                        negatives_candidates.append((text, hybrid_score))
                    else:
                        easy_negatives.append(text)
            
            # 對每個問句限制正樣本與負樣本數量，並平衡類別
            selected_pos = random.sample(positives, min(len(positives), 3)) if positives else []
            
            # Hard Negatives: 依 Hybrid Score 降序排序，取得最棘手的負樣本
            negatives_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_hard_neg = [item[0] for item in negatives_candidates[:4]]
            
            selected_easy_neg = random.sample(easy_negatives, min(len(easy_negatives), 4)) if easy_negatives else []
            
            neg_list = selected_hard_neg + selected_easy_neg
            
            # 若有正樣本，將正樣本 Oversample 複製，直到數量與負樣本完全一致 (1:1 平衡)
            if selected_pos and neg_list:
                oversampled_pos = []
                while len(oversampled_pos) < len(neg_list):
                    oversampled_pos.append(random.choice(selected_pos))
            else:
                oversampled_pos = selected_pos
                
            # 加入資料集
            for pos in oversampled_pos:
                pairs.append((query, pos))
                labels.append(1.0) # 相關
                
            for neg in neg_list:
                pairs.append((query, neg))
                labels.append(0.0) # 不相關
        print(f"[Fine-Tuning] 數據生成完畢！總訓練 Pair 數: {len(pairs)} (正樣本: {labels.count(1.0)}, 負樣本: {labels.count(0.0)})")
        return pairs, labels

    def train_lora(self, epochs=3, batch_size=8, lr=2e-5, lora_r=8,
                   temperature=1.0, margin=0.3, patience=2, use_llm=False, adapter_name="default_adapter", status_callback=None):
        """
        核心 PEFT/LoRA 訓練循環。
        """
        import numpy as np
        import copy

        # 1. 生成與分割訓練/驗證資料
        pairs, labels = self.generate_training_data(use_llm=use_llm)
        train_pairs, train_labels, val_pairs, val_labels = self._stratified_split(pairs, labels, val_ratio=0.2)

        pos_c = train_labels.count(1.0)
        neg_c = train_labels.count(0.0)
        print(f"[Fine-Tuning] 訓練集: {len(train_labels)} 筆 (正:{pos_c} 負:{neg_c}) | 驗證集: {len(val_labels)} 筆")

        # 1.5. 釋放 RAG 引擎中現有的 Rerank 模型，防範 MPS 多重載入時 PyTorch 崩潰 (SIGSEGV)
        if self.rag is not None and hasattr(self.rag, "rerank_model") and self.rag.rerank_model is not None:
            print("[Fine-Tuning] 偵測到已有 Rerank 模型，正在釋放舊模型權重以防止 MPS/GPU 記憶體衝突...", flush=True)
            try:
                if hasattr(self.rag.rerank_model, "model") and self.rag.rerank_model.model is not None:
                    self.rag.rerank_model.model.to("cpu")
            except Exception as e:
                print(f"[Fine-Tuning] ⚠️ 釋放 Rerank 失敗: {e}", flush=True)
            self.rag.rerank_model = None

        # 1.6. 暫時將 Embedding 模型遷移至 CPU，以挪出最大 MPS 記憶體空間供微調訓練使用
        if self.rag is not None and hasattr(self.rag, "embedding_model") and self.rag.embedding_model is not None:
            print("[Fine-Tuning] 正在將 Embedding 模型暫時遷移至 CPU 以挪出 MPS 顯存...", flush=True)
            try:
                self.rag.embedding_model.to("cpu")
            except Exception as e:
                print(f"[Fine-Tuning] ⚠️ 遷移 Embedding 失敗: {e}", flush=True)
            
        import gc
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

        # 2. 載入乾淨的基礎模型
        from sentence_transformers import CrossEncoder
        print(f"[Fine-Tuning] 正在載入乾淨的原始基礎模型 ({config.RERANK_MODEL_NAME})...")
        clean_encoder = CrossEncoder(config.RERANK_MODEL_NAME, device=self.train_device.type)
        base_model = clean_encoder.model
        tokenizer  = clean_encoder.tokenizer

        # 3. DataLoader (提前初始化以計算 AdaLoRA 所需的 total_steps)
        dataset = TextPairDataset(train_pairs, train_labels, tokenizer)
        num_workers = min(4, os.cpu_count() or 1) if self.train_device.type == "cuda" else 0
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=(self.train_device.type == "cuda")
        )
        if len(dataloader) == 0:
            raise ValueError("生成的訓練資料為空，請先上傳論文建立知識庫。")

        total_steps = len(dataloader) * epochs

        # 4. 注入 AdaLoRA (Adaptive Low-Rank Adaptation)
        from peft import AdaLoraConfig, get_peft_model, TaskType
        print(f"[Fine-Tuning] 正在為 Cross-Encoder 注入 AdaLoRA (init_r={lora_r+4}, target_r={lora_r}, total_steps={total_steps})...")
        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            init_r=lora_r + 4,
            target_r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            total_step=total_steps
        )

        model = get_peft_model(base_model, peft_config)
        model.to(self.train_device)
        if self.train_device.type == "mps":
            model = model.float()  # MPS FP16 backward 不穩定，訓練強制 FP32
            
        # 💡 Multi-GPU 支援：自動包裝 DataParallel
        if is_multi_gpu_enabled(self.train_device):
            print(f"[Fine-Tuning] 🚀 偵測到 {config.NUM_GPUS} 張顯卡，啟用 DataParallel 分散式協同運算！")
            model = nn.DataParallel(model)
            
        model.train()
        
        if hasattr(model, "module"):
            model.module.print_trainable_parameters()
        else:
            model.print_trainable_parameters()

        # 5. 動態類別加權（正樣本少時提升其 Loss 貢獻，防止欠擬合）
        pos_weight = torch.tensor([neg_c / max(pos_c, 1)], dtype=torch.float32, device=self.train_device)
        criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Pairwise Margin Ranking Loss 教模型知道「正樣本的分數必須高於負樣本至少 margin」
        criterion_rank = nn.MarginRankingLoss(margin=margin)
        # 深度對比學習 InfoNCE Loss
        criterion_info_nce = InfoNCELoss(temperature=0.07)

        optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        # 升級為 Cosine Scheduler (餘弦退火，學術界更推薦的 Transformer 微調收斂器)
        from transformers import get_cosine_schedule_with_warmup
        scheduler   = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.1)),
            num_training_steps=total_steps
        )

        # 5. Early Stopping 狀態
        best_val_auc    = 0.0
        best_state_dict = None
        no_improve_cnt  = 0
        stopped_early   = False

        # 6. AMP (混合精度) 與梯度累積 (Gradient Accumulation) 設定
        use_amp = (self.train_device.type == "cuda")
        # MPS 暫不支援 GradScaler，因此只在 CUDA 啟用
        scaler = torch.cuda.amp.GradScaler() if self.train_device.type == "cuda" else None
        accumulation_steps = max(1, 16 // batch_size) # 動態確保等效 Batch Size = 16

        print(f"[Fine-Tuning] 開始訓練 (Epochs≤{epochs}, lr={lr}, margin={margin}, patience={patience})")
        print(f"[Fine-Tuning] 優化策略啟用: AMP={use_amp}, Gradient Accumulation Steps={accumulation_steps}")

        for epoch in range(epochs):
            total_loss = 0.0
            model.train()
            optimizer.zero_grad() # 移動到 epoch 開頭

            for step, batch in enumerate(dataloader):
                input_ids      = batch["input_ids"].to(self.train_device)
                attention_mask = batch["attention_mask"].to(self.train_device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.train_device)

                model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    model_kwargs["token_type_ids"] = token_type_ids

                # 啟用混合精度訓練 (Automatic Mixed Precision)
                # 使用 dtype=torch.float16 如果不是 MPS 否則就用 PyTorch 預設的 autocast behavior
                device_type = "cuda" if self.train_device.type == "cuda" else "cpu"
                autocast_kwargs = {"device_type": device_type, "enabled": use_amp}
                if self.train_device.type == "cuda":
                    autocast_kwargs["dtype"] = torch.float16
                    
                with torch.autocast(**autocast_kwargs):
                    outputs = model(**model_kwargs)
                    logits  = extract_logits(outputs)

                    # 擴展包容性：logits 可能是 (N,1) 或 (N,)
                    if logits.dim() > 1 and logits.shape[1] == 1:
                        logits = logits.squeeze(1)

                    # 溫度平滑（temperature scaling）
                    smoothed_logits = logits / temperature

                    # 動態 pos_weight 的 BCE Loss
                    batch_labels = batch["labels"].to(self.train_device)
                    loss_bce = criterion_bce(smoothed_logits, batch_labels)

                    # Pairwise Margin Ranking Loss
                    pos_mask = (batch_labels == 1)
                    neg_mask = (batch_labels == 0)
                    loss_rank = torch.tensor(0.0, device=self.train_device)
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_logits = logits[pos_mask].unsqueeze(1)   # (P, 1)
                        neg_logits = logits[neg_mask].unsqueeze(0)   # (1, N)
                        pos_exp = pos_logits.expand(-1, neg_logits.shape[1])  # (P, N)
                        neg_exp = neg_logits.expand(pos_logits.shape[0], -1)  # (P, N)
                        target  = torch.ones_like(pos_exp)
                        loss_rank = criterion_rank(
                            pos_exp.reshape(-1), neg_exp.reshape(-1), target.reshape(-1)
                        )

                    # 深度對比學習 InfoNCE Loss
                    loss_info_nce = criterion_info_nce(smoothed_logits, batch_labels)

                    # 混合 Loss 與梯度累積縮放
                    loss = (0.5 * loss_bce + 0.3 * loss_rank + 0.2 * loss_info_nce) / accumulation_steps

                # 反向傳播 (透過 Scaler 若有啟用)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item() * accumulation_steps # 還原真實 loss 用於顯示

                # 執行 Gradient Accumulation
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                    # 梯度裁剪
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        
                    scheduler.step()
                    optimizer.zero_grad()

                if (step + 1) % 5 == 0 or (step + 1) == len(dataloader):
                    msg = (f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] "
                           f"Loss={loss.item()*accumulation_steps:.4f} (BCE={loss_bce.item():.4f} Rank={loss_rank.item():.4f} InfoNCE={loss_info_nce.item():.4f})")
                    print(f"  {msg}")
                    if status_callback:
                        status_callback(
                            msg=msg, 
                            progress_val=(epoch * len(dataloader) + step + 1) / total_steps, 
                            loss_dict={
                                "Total": loss.item() * accumulation_steps,
                                "BCE": loss_bce.item(),
                                "Rank": loss_rank.item(),
                                "InfoNCE": loss_info_nce.item()
                            }
                        )

                # 釋放批次顯存，避免 MPS 顯存暴漲與記憶體洩漏
                del input_ids, attention_mask, batch_labels
                if token_type_ids is not None:
                    del token_type_ids
                if "outputs" in locals():
                    del outputs
                if "logits" in locals():
                    del logits
                free_memory(self.train_device)

            mean_loss = total_loss / len(dataloader)
            
            # 7. 主動釋放 GPU/MPS 記憶體 Cache，防止 OOM
            free_memory(self.train_device)

            # 驗證與 Early Stopping
            val_auc = self._compute_val_auc(model, tokenizer, val_pairs, val_labels)

            # ── Early Stopping 修正版 ──────────────────────────────────────
            # 1. 改善閾值放寬至 0.005：小資料集 AUC 波動幅度小，原本 1e-4 過於嚴苛導致假觸發
            min_delta = 5e-3
            improved = val_auc > best_val_auc + min_delta

            # 2. 保護期：至少跑完一半 epoch 才允許 Early Stop
            min_epochs_required = max(2, (epochs + 1) // 2)
            can_early_stop = (epoch + 1) >= min_epochs_required

            epoch_msg = (
                f"Epoch [{epoch+1}/{epochs}] 完成 | 訓練 Loss={mean_loss:.4f} "
                f"| Val AUC={val_auc:.4f} "
                + ("(改善 ⬆)" if improved else f"(未改善 {no_improve_cnt+1}/{patience})")
                + ("" if can_early_stop else " [保護期：Early Stop 暫停]")
            )
            print(f"✅ {epoch_msg}")
            if status_callback:
                status_callback(epoch_msg, (epoch + 1) / epochs)

            # 3. 只要 AUC 有任b何進步就存檔（不受 min_delta 限制）
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                if hasattr(model, "module"):
                    best_state_dict = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"[Fine-Tuning] 💾 儲存最佳 Checkpoint (Val AUC={best_val_auc:.4f})")

            # 4. 計數器：只在保護期結束後才計入 no_improve
            if improved:
                no_improve_cnt = 0
            elif can_early_stop:
                no_improve_cnt += 1
                if no_improve_cnt >= patience:
                    print(f"[Fine-Tuning] Early Stopping 觸發！Val AUC 連續 {patience} Epoch 改善幅度不足 {min_delta}，復原最佳 Checkpoint。")
                    stopped_early = True
                    break
            # 保護期內不計入 no_improve_cnt

            # 確保 MPS/CUDA 記憶體在每個 Epoch 結束時釋放
            free_memory(self.train_device)

        # 還原最佳 Checkpoint
        if best_state_dict is not None:
            print("[Fine-Tuning] 載入最佳 Epoch 模型權重...")
            if hasattr(model, "module"):
                model.module.load_state_dict(best_state_dict)
            else:
                model.load_state_dict(best_state_dict)

        # 記憶體大掃除
        print("[Fine-Tuning] 正在釋放優化器與學習率排程器佔用的 RAM...")
        del optimizer
        del scheduler
        del scaler
        free_memory(self.train_device)

        print("\n[Fine-Tuning] 🚀 LoRA 微調結束！準備更新 RAG 評估報告...")

        # 7. 儲存 LoRA Adapter
        adapter_save_dir = os.path.join(self.project_paths["lora_adapters_dir"], adapter_name)
        os.makedirs(adapter_save_dir, exist_ok=True)
        print(f"[Fine-Tuning] 正在將訓練完成的 LoRA Adapter 寫入磁碟 ({adapter_name})...")
        
        # 儲存權重 (相容 DataParallel)
        if hasattr(model, "module"):
            model.module.save_pretrained(adapter_save_dir)
        else:
            model.save_pretrained(adapter_save_dir)
            
        tokenizer.save_pretrained(adapter_save_dir) 
        print(f"✅ Adapter 存儲成功！({'Early Stopped' if stopped_early else f'Full {epochs} Epochs'} | Best Val AUC={best_val_auc:.4f})")

        # 6. 重載全面評估
        model.eval()
        print("[Fine-Tuning] 重新載入含 LoRA Adapter 的模型權重...")
        
        # 釋放訓練模型權重，防範 MPS 重載時 PyTorch 崩潰
        try:
            model.to("cpu")
            base_model.to("cpu")
        except Exception:
            pass
            
        del model
        del base_model
        del clean_encoder
        free_memory(self.train_device)

        # 將 Embedding 模型恢復載入至 GPU/MPS
        if self.rag is not None and hasattr(self.rag, "embedding_model") and self.rag.embedding_model is not None:
            print("[Fine-Tuning] 正在將 Embedding 模型恢復至 GPU/MPS 裝置...", flush=True)
            try:
                self.rag.embedding_model.to(self.device)
            except Exception:
                pass

        from src.model_manager import ModelManager
        ModelManager().release_model("rerank")
        
        # 強制 Python GC 確保舊模型物件真的從記憶體移除
        import gc
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.rag.load_index()

        # 驗證新 Adapter 已正確載入
        print("[Fine-Tuning] 正在驗證新 LoRA Adapter 是否已正確載入至 RAG 引擎...")
        rerank_check = self.rag.rerank_model
        if rerank_check is not None:
            is_peft = hasattr(rerank_check.model, "peft_config") or "peft" in str(type(rerank_check.model)).lower()
            print(f"[Fine-Tuning] ✅ Rerank 模型載入成功 | PEFT/LoRA 模式: {is_peft}")
        else:
            print("[Fine-Tuning] ⚠️ Rerank 模型未能載入，評估將以 Bi-Encoder 模式進行。")

        print("[Fine-Tuning] 正在自動對微調後的模型進行 ROC-AUC 診斷評估...")
        return self.eval_suite.evaluate_retrieval(top_k=3, use_llm=use_llm)

    def run_hpo_search(self, lora_r=8, use_llm=False, status_callback=None):
        """
        自動超參數優化主動搜尋 (Auto-HPO Grid Sweep)
        """
        # HPO 候選範圍 (降低學習率，防止破壞預訓練模型的泛化能力)
        lr_candidates = [1e-5, 2e-5, 3e-5]
        margin_candidates = [0.2, 0.3, 0.4]
        
        best_auc = 0.0
        best_params = {"lr": 3e-5, "margin": 0.3}
        trials = []
        
        # 1. 先生成並快取訓練數據，避免重複生成
        pairs, labels = self.generate_training_data(use_llm=use_llm)
        train_pairs, train_labels, val_pairs, val_labels = self._stratified_split(pairs, labels, val_ratio=0.25)
        
        trial_id = 1
        total_trials = len(lr_candidates) * len(margin_candidates)
        
        for lr in lr_candidates:
            for margin in margin_candidates:
                # 換算為 10 進位科學記號上標，方便終端與 UI 閱讀
                trans = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
                lr_str = f"{lr:.0e}".replace("e-0", " × 10⁻").replace("e-", " × 10⁻")
                lr_str = lr_str.split(" × ")[0] + " × 10" + lr_str.split(" × 10")[1].translate(trans)

                trial_msg = f"🔍 HPO 嘗試 [{trial_id}/{total_trials}] — 測試 LR: {lr_str}, Margin: {margin}..."
                print(f"[Fine-Tuning HPO] {trial_msg}")
                if status_callback:
                    status_callback(trial_msg, trial_id / total_trials)
                
                try:
                    auc = self._quick_evaluate_hyperparams(
                        train_pairs, train_labels, val_pairs, val_labels,
                        lora_r=lora_r, lr=lr, margin=margin
                    )
                    trials.append({
                        "trial": trial_id,
                        "lr": lr,
                        "margin": margin,
                        "val_auc": auc,
                        "status": "Success ✅"
                    })
                    print(f"[Fine-Tuning HPO] 試驗完成 | Val AUC = {auc:.4f}")
                    if auc > best_auc:
                        best_auc = auc
                        best_params = {"lr": lr, "margin": margin}
                except Exception as e:
                    trials.append({
                        "trial": trial_id,
                        "lr": lr,
                        "margin": margin,
                        "val_auc": 0.0,
                        "status": f"Error: {e} ❌"
                    })
                    print(f"[Fine-Tuning HPO] 試驗失敗: {e}")
                
                trial_id += 1
                
        return best_params, trials

    def _quick_evaluate_hyperparams(self, train_pairs, train_labels, val_pairs, val_labels, lora_r, lr, margin):
        import gc
        from sentence_transformers import CrossEncoder
        from peft import AdaLoraConfig, get_peft_model, TaskType
        
        # 載入模型
        clean_encoder = CrossEncoder(config.RERANK_MODEL_NAME, device=self.train_device.type)
        base_model = clean_encoder.model
        tokenizer  = clean_encoder.tokenizer
        
        # 快速 DataLoader (使用較小 batch size 以節省顯存)
        dataset = TextPairDataset(train_pairs, train_labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        total_steps = len(dataloader)
        
        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            init_r=lora_r + 4,
            target_r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            total_step=total_steps
        )
        
        model = get_peft_model(base_model, peft_config)
        model.to(self.train_device)
        if self.train_device.type == "mps":
            model = model.float()
        model.train()
        
        # BCE 與 MarginLoss
        pos_c = train_labels.count(1.0)
        neg_c = train_labels.count(0.0)
        pos_weight = torch.tensor([neg_c / max(pos_c, 1)], dtype=torch.float32, device=self.train_device)
        criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_rank = nn.MarginRankingLoss(margin=margin)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 僅快速訓練前 70% 步數以加快搜尋速度
        max_steps = int(len(dataloader) * 0.7)
        for step, batch in enumerate(dataloader):
            if step > max_steps:
                break
            input_ids      = batch["input_ids"].to(self.train_device)
            attention_mask = batch["attention_mask"].to(self.train_device)
            batch_labels   = batch["labels"].to(self.train_device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = extract_logits(outputs)
            if logits.dim() > 1 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
                
            loss_bce = criterion_bce(logits, batch_labels)
            
            pos_mask = (batch_labels == 1)
            neg_mask = (batch_labels == 0)
            loss_rank = torch.tensor(0.0, device=self.train_device)
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_exp = logits[pos_mask].unsqueeze(1).expand(-1, logits[neg_mask].shape[0])
                neg_exp = logits[neg_mask].unsqueeze(0).expand(logits[pos_mask].shape[0], -1)
                loss_rank = criterion_rank(pos_exp.reshape(-1), neg_exp.reshape(-1), torch.ones_like(pos_exp).reshape(-1))
                
            loss = 0.6 * loss_bce + 0.4 * loss_rank
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            del input_ids, attention_mask, batch_labels
            
        # 計算驗證 AUC
        val_auc = self._compute_val_auc(model, tokenizer, val_pairs, val_labels)
        
        # 清理記憶體
        try:
            model.to("cpu")
            base_model.to("cpu")
        except Exception:
            pass
        del model, base_model, clean_encoder, optimizer
        free_memory(self.train_device)
        gc.collect()
        
        return val_auc

if __name__ == "__main__":
    tuner = AcademicLoRATuner()
    report = tuner.train_lora(epochs=3, lr=5e-5)
    print("\n" + "="*50)
    print("🎉 微調後評估報告：")
    print(f"ROC-AUC: {report['diagnosis']['auc']:.4f}")
    print(f"Margin: {report['diagnosis']['margin']:.4f}")
    print(f"狀態: {report['diagnosis']['status']}")
    print(f"建議: {report['diagnosis']['recommendation']}")
    print("="*50)
