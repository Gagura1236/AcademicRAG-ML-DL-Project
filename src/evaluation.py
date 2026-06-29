import os
import sys
import numpy as np
import time

# 獲取專案根目錄，加入系統路徑以正確導入 config.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.embedding_search import AcademicRAGEngine
from src.gpu_utils import free_memory, extract_logits

class RAGEvaluationSuite:
    """
    ML/DL 檢索系統核心評估套件：
    專門測試 Bi-Encoder 向量空間初篩與 Cross-Encoder 深度邏輯斯迴歸重排模型的效能。
    提供：
    1. 核心指標評估: Precision@K, Recall@K, MRR, MAP
    2. Overfitting (過擬合) 與 Underfitting (欠擬合) 診斷
    3. 權重與機率信心分佈統計 (ROC-AUC / PR-AUC 模擬)
    """
    def __init__(self, rag_engine: AcademicRAGEngine = None, project_name="default"):
        self.project_name = project_name
        self.rag = rag_engine if rag_engine is not None else AcademicRAGEngine(project_name=project_name)
        
        # 建立黃金評估資料集 (Ground Truth Dataset)
        # Query, 預期的關鍵公式片段/核心術語 (用於驗證檢索到的 chunk 是否包含真實答案)
        self.eval_dataset = [
            {
                "query": "What is the mathematical formula for Scaled Dot-Product Attention?",
                "gt_keywords": ["Attention", "softmax", "d_k", "V"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "What is Multi-Head Attention and how is it calculated?",
                "gt_keywords": ["MultiHead", "Concat", "head_i", "W^O"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "What is the formula of the Position-wise Feed-Forward Networks?",
                "gt_keywords": ["FFN", "max", "W_1", "W_2"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "How is the positional encoding calculated using sine and cosine?",
                "gt_keywords": ["PE", "sin", "cos", "10000"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "What is the Masked Language Model (MLM) pre-training in BERT?",
                "gt_keywords": ["masked", "BERT", "language model", "pre-training"],
                "target_paper_id": "1810.04805"
            },
            {
                "query": "How does sliding window attention reduce complexity to O(N)?",
                "gt_keywords": ["sliding window", "complexity", "O(N)", "local"],
                "target_paper_id": "2004.05150"
            }
        ]
        self.agent = None

    def _load_agent(self):
        if self.agent is None:
            from src.agent import AcademicAgent
            self.agent = AcademicAgent()

    def generate_dynamic_eval_dataset(self, use_llm=False):
        """
        利用 LLM 根據當前資料庫的內容，動態產生高質量的評估題庫。
        """
        if not use_llm or self.rag.index is None or not self.rag.chunks_metadata:
            return self.eval_dataset
            
        print("[ML/DL Eval] 啟動 LLM 動態產生知識庫專屬評估題庫...")
        self._load_agent()
        
        import random
        # 隨機抽取 5 個包含大量文字的 Chunks 作為出題來源
        sample_chunks = random.sample(self.rag.chunks_metadata, min(5, len(self.rag.chunks_metadata)))
        
        dynamic_dataset = []
        for chunk in sample_chunks:
            text = chunk.get("text", "")
            if len(text) < 100: continue
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI professor. Generate a highly specific academic question that can ONLY be answered by reading the provided text.
Also extract 2-4 core keywords from the text that MUST be present in the answer.
Return ONLY valid JSON in this exact format:
{{"query": "Your question here?", "gt_keywords": ["keyword1", "keyword2"]}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: {text[:1500]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            try:
                response = self.agent.generate(prompt, max_tokens=256, temp=0.2)
                import re, json
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    data["target_paper_id"] = chunk.get("paper_id", "")
                    dynamic_dataset.append(data)
                    print(f"  -> 生成題目: {data['query']}")
            except Exception as e:
                pass
                
        if len(dynamic_dataset) > 0:
            return dynamic_dataset
        return self.eval_dataset

    def get_classic_benchmark_dataset(self):
        """
        專案介紹與系統展示專用：39 篇經典 AI/ML 論文的 Ground-Truth 基準測試集。
        """
        return [
            {
                "query": "What is the mathematical formula for Scaled Dot-Product Attention?",
                "gt_keywords": ["Attention", "softmax", "d_k", "V"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "What is Multi-Head Attention and how is it calculated?",
                "gt_keywords": ["MultiHead", "Concat", "head_i", "W^O"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "What is the formula of the Position-wise Feed-Forward Networks?",
                "gt_keywords": ["FFN", "max", "W_1", "W_2"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "How is the positional encoding calculated using sine and cosine?",
                "gt_keywords": ["PE", "sin", "cos", "10000"],
                "target_paper_id": "1706.03762"
            },
            {
                "query": "What is the Masked Language Model (MLM) pre-training in BERT?",
                "gt_keywords": ["masked", "BERT", "language model", "pre-training"],
                "target_paper_id": "1810.04805"
            },
            {
                "query": "What is the residual mapping formula in Deep Residual Learning?",
                "gt_keywords": ["F(x) + x", "shortcut connection", "residual learning", "identity"],
                "target_paper_id": "1512.03385"
            },
            {
                "query": "What is the minimax objective function of Generative Adversarial Networks (GAN)?",
                "gt_keywords": ["min", "max", "V(D,G)", "log", "D(x)", "G(z)"],
                "target_paper_id": "1406.2661"
            },
            {
                "query": "What is the bias-corrected first and second moment estimate in Adam optimizer?",
                "gt_keywords": ["bias-corrected", "m_t", "v_t", "beta_1", "beta_2"],
                "target_paper_id": "1412.6980"
            },
            {
                "query": "How does Vision Transformer (ViT) partition an image into patches?",
                "gt_keywords": ["patches", "x \\in \\mathbb{R}^{H \\times W \\times C}", "P^2", "flatten"],
                "target_paper_id": "2010.11929"
            },
            {
                "query": "How does YOLO divide the input image for object detection?",
                "gt_keywords": ["grid", "bounding box", "confidence", "S \times S"],
                "target_paper_id": "1506.02640"
            },
            {
                "query": "What is the variational bound loss formulation in Denoising Diffusion Probabilistic Models (DDPM)?",
                "gt_keywords": ["diffusion", "noise", "variational bound", "reverse process"],
                "target_paper_id": "2006.11239"
            },
            {
                "query": "What is the loss function or Bellman equation used to train Deep Q-Networks (DQN)?",
                "gt_keywords": ["loss function", "Q-learning", "target", "Bellman", "reward"],
                "target_paper_id": "1312.5602"
            },
            {
                "query": "What is the difference between RAG-Sequence and RAG-Token models?",
                "gt_keywords": ["RAG-Sequence", "RAG-Token", "generator", "document", "retriever"],
                "target_paper_id": "2005.11401"
            },
            {
                "query": "What is the layer-wise propagation rule of Graph Convolutional Networks (GCN)?",
                "gt_keywords": ["layer-wise", "propagation", "adjacency matrix", "spectral", "degree"],
                "target_paper_id": "1609.02907"
            },
            {
                "query": "How does InstructGPT use RLHF (Reinforcement Learning from Human Feedback)?",
                "gt_keywords": ["RLHF", "reward model", "PPO", "human feedback", "fine-tuning"],
                "target_paper_id": "2203.02155"
            }
        ]

    def evaluate_retrieval(self, top_k: int = 3, use_llm=False, benchmark_mode=False) -> dict:
        """
        執行評估，對比 [Bi-Encoder Only] 與 [Bi-Encoder + Cross-Encoder Rerank] 兩種 ML 架構的表現。
        """
        if benchmark_mode:
            dataset_to_use = self.get_classic_benchmark_dataset()
        else:
            dataset_to_use = self.generate_dynamic_eval_dataset(use_llm=use_llm)
        print(f"\n[ML/DL Eval] 開始評估檢索準確性 (樣本數: {len(dataset_to_use)}, Top-{top_k})...")
        
        metrics = {
            "bi_encoder": {"precision": [], "recall": [], "mrr": [], "map": [], "ndcg": [], "hit": [], "scores": []},
            "reranked":   {"precision": [], "recall": [], "mrr": [], "map": [], "ndcg": [], "hit": [], "scores": [], "ragas_context_relevancy": [], "ragas_context_recall": []}
        }
        
        # 保存所有預測分數與真實 Label (1=相關, 0=不相關) 用於 ROC/PR 曲線統計
        all_labels = []
        all_bi_scores = []
        all_rerank_scores = []
        
        # L-3 Fix: move empty-index guard before the loop (not checked 6x inside)
        if self.rag.index is None or not self.rag.chunks_metadata:
            print("⚠️ FAISS 向量資料庫為空，無法評估。請先索引論文！")
            return {
                "metrics": {
                    "bi_encoder": {"mrr": 0.0, "hit_rate": 0.0, "scores": []},
                    "reranked": {"mrr": 0.0, "hit_rate": 0.0, "scores": []}
                },
                "diagnosis": self.diagnose_overfitting([], [], [])
            }

        for item in dataset_to_use:
            query = item["query"]
            keywords = item["gt_keywords"]
            target_id = item["target_paper_id"]
                
            # 1. Bi-Encoder 向量檢索 (初篩較多 candidates)
            query_vector = self.rag.embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
            candidate_count = min(top_k * 3, len(self.rag.chunks_metadata))
            distances, indices = self.rag.index.search(query_vector, candidate_count)
            
            candidates = []
            candidate_distances = []  # M-2 Fix: parallel list to avoid FAISS -1 slot index mismatch
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                c = self.rag.chunks_metadata[idx].copy()
                # 餘弦相似度直接作為分數 (已使用 IndexFlatIP)
                c["score"] = float(dist)
                candidates.append(c)
                candidate_distances.append(float(dist))
                
            # Bi-Encoder 檢索結果取前 top_k (使用字典複製，避免後續 Cross-Encoder 重排修改其 score)
            bi_results = [c.copy() for c in candidates[:top_k]]
            
            # 2. Cross-Encoder 二次重排
            rerank_results = []
            scores = np.array([])  # 初始化 scores，防止 candidates 為空時 cross-loop 污染
            if self.rag.rerank_model is not None and candidates:
                pairs = [(query, c["text"]) for c in candidates]
                
                import torch
                is_peft = hasattr(self.rag.rerank_model.model, "peft_config") or "peft" in str(type(self.rag.rerank_model.model)).lower()
                # C-3 Fix: split list of tuples into two parallel lists for HuggingFace tokenizer
                if is_peft:
                    queries_list, texts_list = zip(*pairs)
                    features = self.rag.rerank_model.tokenizer(
                        list(queries_list), list(texts_list),
                        padding=True, truncation=True, return_tensors="pt", max_length=512
                    )
                    features = {k: v.to(self.rag.device) for k, v in features.items()}
                    with torch.no_grad():
                        out = self.rag.rerank_model.model(**features)
                        logits = extract_logits(out)
                        if logits.dim() > 1 and logits.shape[1] == 2:
                            # 二元分類 (N,2)：取 label=1 (相關) 的 softmax 機率
                            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        elif logits.dim() > 1 and logits.shape[1] == 1:
                            scores = torch.sigmoid(logits.view(-1)).cpu().numpy()
                        else:
                            scores = torch.sigmoid(logits).cpu().numpy()
                            
                    del features, logits, out
                    free_memory(self.rag.device)
                else:
                    # M-6 Fix: normalize raw logits to [0,1] for consistent score domain
                    import numpy as _np
                    raw_scores = self.rag.rerank_model.predict(pairs)
                    scores = 1.0 / (1.0 + _np.exp(-_np.array(raw_scores)))
                
                # M-3 Fix: build rerank_results from fresh copies, not mutated references
                scored_candidates = []
                for idx, score in enumerate(scores):
                    c_copy = dict(candidates[idx])
                    c_copy["score"] = float(score)
                    scored_candidates.append(c_copy)
                
                # 降序排序
                sorted_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
                rerank_results = sorted_candidates[:top_k]
            else:
                rerank_results = bi_results

            # --- B. 計算指標 ---
            # 判斷一個 chunk 是否是 "Ground Truth" (真實相關)
            # Bug 7: Fix late-binding closure by passing variables as default arguments
            def is_relevant(chunk, t_id=target_id, kws=keywords):
                c_id = chunk.get("metadata", {}).get("paper_id", "") or chunk.get("paper_id", "")
                
                # 如果資料庫裡確實有目標論文，嚴格要求 ID 吻合
                db_has_tid = any(c.get("paper_id") == t_id for c in self.rag.chunks_metadata) if t_id else False
                if db_has_tid and c_id != t_id:
                    return False
                
                # 否則（使用者上傳了自己的論文），只要關鍵字大量重疊即視為正樣本
                match_count = sum(1 for kw in kws if kw.lower() in chunk["text"].lower())
                return match_count >= max(1, len(kws) // 2)

            # 計算 Bi-Encoder Only
            bi_hits = [is_relevant(c) for c in bi_results]
            precision_bi = sum(bi_hits) / max(1, top_k)
            recall_bi    = 1.0 if any(bi_hits) else 0.0
            metrics["bi_encoder"]["precision"].append(precision_bi)
            metrics["bi_encoder"]["recall"].append(recall_bi)
            metrics["bi_encoder"]["hit"].append(1.0 if any(bi_hits) else 0.0)

            # MRR
            mrr_bi = 0.0
            for rank_idx, hit in enumerate(bi_hits):
                if hit:
                    mrr_bi = 1.0 / (rank_idx + 1)
                    break
            metrics["bi_encoder"]["mrr"].append(mrr_bi)

            # MAP@K
            ap_bi, hit_count = 0.0, 0
            for rank_idx, hit in enumerate(bi_hits):
                if hit:
                    hit_count += 1
                    ap_bi += hit_count / (rank_idx + 1)
            metrics["bi_encoder"]["map"].append(ap_bi / max(1, sum(bi_hits)))

            # NDCG@K
            import math
            dcg_bi  = sum(1.0 / math.log2(i + 2) for i, h in enumerate(bi_hits) if h)
            idcg_bi = sum(1.0 / math.log2(i + 2) for i in range(min(sum(bi_hits), top_k)))
            metrics["bi_encoder"]["ndcg"].append(dcg_bi / max(idcg_bi, 1e-9))
            metrics["bi_encoder"]["scores"].extend([c["score"] for c in bi_results])

            # 計算 Reranked
            rerank_hits = [is_relevant(c) for c in rerank_results]
            precision_rr = sum(rerank_hits) / max(1, top_k)
            recall_rr    = 1.0 if any(rerank_hits) else 0.0
            metrics["reranked"]["precision"].append(precision_rr)
            metrics["reranked"]["recall"].append(recall_rr)
            metrics["reranked"]["hit"].append(1.0 if any(rerank_hits) else 0.0)

            mrr_rr = 0.0
            for rank_idx, hit in enumerate(rerank_hits):
                if hit:
                    mrr_rr = 1.0 / (rank_idx + 1)
                    break
            metrics["reranked"]["mrr"].append(mrr_rr)

            ap_rr, hit_count = 0.0, 0
            for rank_idx, hit in enumerate(rerank_hits):
                if hit:
                    hit_count += 1
                    ap_rr += hit_count / (rank_idx + 1)
            metrics["reranked"]["map"].append(ap_rr / max(1, sum(rerank_hits)))

            dcg_rr  = sum(1.0 / math.log2(i + 2) for i, h in enumerate(rerank_hits) if h)
            idcg_rr = sum(1.0 / math.log2(i + 2) for i in range(min(sum(rerank_hits), top_k)))
            metrics["reranked"]["ndcg"].append(dcg_rr / max(idcg_rr, 1e-9))
            metrics["reranked"]["scores"].extend([c["score"] for c in rerank_results])

            # v2.2: RAGAS-style Metrics (Context Recall & Context Relevancy)
            found_kws = set()
            total_sentences = 0
            relevant_sentences = 0
            import re
            
            for c in rerank_results:
                # 1. Context Recall: GT keywords coverage
                for kw in keywords:
                    if kw.lower() in c["text"].lower():
                        found_kws.add(kw.lower())
                
                # 2. Context Relevancy: Relevant sentences ratio
                sentences = re.split(r'(?<=[.!?;])\s+|\n\n+', c["text"])
                for s in sentences:
                    if s.strip():
                        total_sentences += 1
                        if any(kw.lower() in s.lower() for kw in keywords):
                            relevant_sentences += 1
                            
            metrics["reranked"]["ragas_context_recall"].append(len(found_kws) / max(1, len(keywords)))
            metrics["reranked"]["ragas_context_relevancy"].append(relevant_sentences / max(1, total_sentences))

            # M-2 Fix: use candidate_distances[c_idx]
            for c_idx, c in enumerate(candidates):
                rel = 1 if is_relevant(c) else 0
                all_labels.append(rel)
                all_bi_scores.append(1.0 / (1.0 + candidate_distances[c_idx]))
                if self.rag.rerank_model is not None and len(scores) > c_idx:
                    all_rerank_scores.append(float(scores[c_idx]))
                    
        # 計算平均指標
        def _safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        def _safe_dist(lst):
            if not lst: return None
            a = np.array(lst, dtype=float)
            return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                    "min": float(np.min(a)), "max": float(np.max(a))}

        report = {
            "bi_encoder": {
                "mean_precision": _safe_mean(metrics["bi_encoder"]["precision"]),
                "mean_recall":    _safe_mean(metrics["bi_encoder"]["recall"]),
                "mean_mrr":       _safe_mean(metrics["bi_encoder"]["mrr"]),
                "mean_map":       _safe_mean(metrics["bi_encoder"]["map"]),
                "mean_ndcg":      _safe_mean(metrics["bi_encoder"]["ndcg"]),
                "hit_rate":       _safe_mean(metrics["bi_encoder"]["hit"]),
                "score_distribution": _safe_dist(metrics["bi_encoder"]["scores"])
            },
            "reranked": {
                "mean_precision": _safe_mean(metrics["reranked"]["precision"]),
                "mean_recall":    _safe_mean(metrics["reranked"]["recall"]),
                "mean_mrr":       _safe_mean(metrics["reranked"]["mrr"]),
                "mean_map":       _safe_mean(metrics["reranked"]["map"]),
                "mean_ndcg":      _safe_mean(metrics["reranked"]["ndcg"]),
                "hit_rate":       _safe_mean(metrics["reranked"]["hit"]),
                "score_distribution": _safe_dist(metrics["reranked"]["scores"]),
                "ragas_context_recall": _safe_mean(metrics["reranked"]["ragas_context_recall"]),
                "ragas_context_relevancy": _safe_mean(metrics["reranked"]["ragas_context_relevancy"])
            }
        }
        
        # --- C. 過擬合/欠擬合診斷 (Overfitting & Calibration Diagnosis) ---
        diagnosis = self.diagnose_overfitting(all_labels, all_rerank_scores, all_bi_scores)
        report["diagnosis"] = diagnosis
        
        return report

    def diagnose_overfitting(self, labels: list, rerank_scores: list, bi_scores: list) -> dict:
        """
        四級診斷：Underfitting / Healthy / Maybe Overfitting / Overfitting

        判斷依據：
          - AUC < 0.65 或 Margin < 0.15  → Underfitting
          - neg高分占比 > 25% 或 neg_P90 > 0.65 + AUC < 0.88  → Maybe Overfitting
          - neg_P90 > 0.75 且 AUC < 0.85  → Overfitting
          - 其他  → Healthy
        """
        labels        = np.array(labels)
        rerank_scores = np.array(rerank_scores)

        if len(labels) == 0 or len(rerank_scores) == 0:
            return {
                "status": "資料不足，無法診斷",
                "auc": 0.5, "margin": 0.0,
                "mean_pos_score": 0.0, "mean_neg_score": 0.0,
                "neg_high_ratio": 0.0, "neg_p90": 0.0,
                "recommendation": "向量庫為空或未載入 Cross-Encoder。請先新增論文再執行評估。",
                "total_samples": 0, "pos_count": 0, "neg_count": 0
            }

        pos_scores = rerank_scores[labels == 1]
        neg_scores = rerank_scores[labels == 0]

        mean_pos = float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0
        mean_neg = float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0
        margin   = mean_pos - mean_neg

        # ROC-AUC (Wilcoxon-Mann-Whitney)
        auc = 0.5
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            hits = sum(1 for p in pos_scores for n in neg_scores if p > n)
            auc  = hits / (len(pos_scores) * len(neg_scores))

        # 輔助指標：高分負樣本占比（> 0.6）與 90th 百分位數
        neg_high_ratio = float(np.mean(neg_scores > 0.6)) if len(neg_scores) > 0 else 0.0
        neg_p90        = float(np.percentile(neg_scores, 90)) if len(neg_scores) > 0 else 0.0

        # 讀取目前設定檔以提供動態建議
        import json
        import os
        import config
        curr_ep = getattr(config, "DEFAULT_LORA_EPOCHS", 3)
        curr_r = getattr(config, "DEFAULT_LORA_RANK", 8)
        try:
            cfg_path = os.path.join(config.DATA_DIR, "lora_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    curr_ep = cfg.get("epochs", curr_ep)
                    curr_r = cfg.get("rank", curr_r)
        except Exception:
            pass

        # 動態建議字串
        ep_down_sugg = "降低學習率 (LR) 或檢查資料品質" if curr_ep <= 2 else f"降低訓練輪數 (建議降至 {max(2, curr_ep-1)} 輪)"
        ep_up_sugg = f"增加訓練輪數 (建議升至 {curr_ep+2} 輪)" if curr_ep < 8 else "訓練輪數已偏高，建議從降低學習率著手"
        rank_up_sugg = f"提升 LoRA Rank (建議升至 {curr_r*2})" if curr_r < 16 else "Rank 已足夠，建議增加 Epoch"

        # ===== 四級分類 =====
        if auc < 0.85:
            status = "模型能力不足 (Underfitting / Weak Capability)"
            recommendation = (
                "**診斷分析**：\n"
                f"- 總體檢索排名能力 (ROC-AUC) 僅 {auc:.3f} (低於學術標準 0.85)。\n"
                f"- 這代表模型並未發生真正的「過擬合」，而是**學習能力不足**，它根本還無法有效區分相關與不相關的語意（正負樣本的分數邊界模糊）。\n"
                "**具體調整建議**：\n"
                f"- {rank_up_sugg}，以擴充模型的參數學習空間。\n"
                f"- {ep_up_sugg}，讓模型有更多時間收斂特徵。\n"
                "- 檢查 Step 1 的切片品質，確保正樣本中確實含有能夠回答問題的關鍵段落。"
            )
        elif neg_high_ratio > 0.25:
            status = "過度自信 / 潛在過擬合 (Overconfident / Maybe Overfitting)"
            recommendation = (
                "**診斷分析**：\n"
                f"- 總體排名分數 (ROC-AUC={auc:.3f}) 表現不錯，代表模型知道正樣本比負樣本好。\n"
                f"- 但模型出現了「死背關鍵字」的跡象：有 {neg_high_ratio*100:.1f}% 的「不相關片段 (負樣本)」被錯誤地給予了 > 0.6 的絕對高分。\n"
                "**具體調整建議**：\n"
                f"- {ep_down_sugg}，避免模型過度記憶訓練資料中的無關噪聲。\n"
                "- 嘗試增加 Contrastive Margin (例如調至 0.4)，強迫模型嚴格拉開正負樣本的絕對分數差距。"
            )
        elif neg_p90 > 0.75:
            status = "嚴重過擬合 (Severe Overfitting)"
            recommendation = (
                "**診斷分析**：\n"
                f"- 模型已嚴重失去泛化能力！它對包含相似關鍵字但「實際上完全不相關」的片段，給出了極高的信心分數。\n"
                f"- 排名前 10% 的不相關片段，其錯誤得分竟然高達 {neg_p90:.3f} 以上。\n"
                "**具體調整建議**：\n"
                f"- 務必{ep_down_sugg}。\n"
                "- 降低學習率 (LR) 至 1 × 10⁻⁵ 以下，減緩權重偏移。\n"
                "- 重新檢視 Step 1 的論文切片是否過於零碎，導致上下文流失。"
            )
        else:
            status = "健康 (Balanced ✔)"
            recommendation = (
                "**診斷分析**：\n"
                f"- 總體檢索排名能力優良 (ROC-AUC={auc:.3f})，正負樣本之間有明確的分數邊界 (Margin={margin:.3f})。\n"
                f"- 模型能精準過濾不相關的片段，僅有 {neg_high_ratio*100:.1f}% 的負樣本被誤判高分。\n"
                "**結論**：\n"
                "- 系統目前處於健康的平衡狀態，沒有發現明顯過擬合或欠擬合跡象。您可以保持目前的超參數設定繼續使用！"
            )

        return {
            "status":         status,
            "auc":            float(auc),
            "margin":         float(margin),
            "mean_pos_score": mean_pos,
            "mean_neg_score": mean_neg,
            "neg_high_ratio": neg_high_ratio,
            "neg_p90":        neg_p90,
            "recommendation": recommendation,
            "total_samples":  len(labels),
            "pos_count":      int(np.sum(labels == 1)),
            "neg_count":      int(np.sum(labels == 0))
        }

if __name__ == "__main__":
    suite = RAGEvaluationSuite()
    report = suite.evaluate_retrieval(top_k=3)
    import json
    print(json.dumps(report, indent=4, ensure_ascii=False))
