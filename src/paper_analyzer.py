import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

# 獲取專案根目錄
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.metadata_extractor import PaperMetadataManager
from src.gpu_utils import free_memory

@st.cache_resource
def get_analyzer_embedding_model():
    print(f"[Paper Analyzer] 正在載入 Embedding 模型 ({config.EMBEDDING_MODEL_NAME})...")
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE.type)

@st.cache_resource
def get_analyzer_rerank_model():
    print(f"[Paper Analyzer] 正在載入 Rerank 模型 ({config.RERANK_MODEL_NAME})...")
    return CrossEncoder(config.RERANK_MODEL_NAME, device=config.DEVICE.type)

from src.metadata_extractor import PaperMetadataManager
from src.project_utils import get_project_paths

class PaperAnalyzer:
    """
    負責論文的全域分析，包含:
    1. 關鍵字萃取 (TF-IDF)
    2. 論文聚類與分類 (K-Means on SentenceBert)
    """
    def __init__(self, project_name="default"):
        self.project_name = project_name
        self.project_paths = get_project_paths(project_name)
        self.metadata_manager = PaperMetadataManager(project_name=project_name)
        self.device = config.DEVICE
        self.embedding_model = None  # Lazy loading
        self.rerank_model = None     # Lazy loading
        self.agent = None            # Lazy loading LLM Agent

    def _load_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = get_analyzer_embedding_model()
            
    def _load_rerank_model(self):
        if self.rerank_model is None:
            self.rerank_model = get_analyzer_rerank_model()
            
    def _load_agent(self):
        if self.agent is None:
            print("[Paper Analyzer] 正在載入 LLM Agent (用於知識圖譜實體抽取)...")
            from src.agent import AcademicAgent
            self.agent = AcademicAgent()

    def _get_embeddings_for_abstracts(self, abstracts, normalize=True):
        import pickle
        cache_dir = os.path.join(parent_dir, "data", "vector_db")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "abstract_embeddings.pkl")
        
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
            except Exception as e:
                print(f"[Paper Analyzer] 讀取摘要向量快取失敗: {e}")
                
        # 尋找未快取的摘要
        missing_abstracts = []
        missing_indices = []
        for idx, abs_text in enumerate(abstracts):
            abs_key = abs_text if abs_text else ""
            if abs_key not in cache:
                missing_abstracts.append(abs_key)
                missing_indices.append(idx)
                
        if missing_abstracts:
            self._load_embedding_model()
            print(f"[Paper Analyzer] 有 {len(missing_abstracts)} 篇新摘要需要進行編碼...")
            new_embs = self.embedding_model.encode(
                missing_abstracts, 
                normalize_embeddings=normalize, 
                convert_to_numpy=True
            ).astype(np.float32)
            
            for i, abs_key in enumerate(missing_abstracts):
                cache[abs_key] = new_embs[i]
                
            # 寫回快取檔案 (原子寫入)
            try:
                temp_path = cache_path + ".tmp"
                with open(temp_path, "wb") as f:
                    pickle.dump(cache, f)
                os.replace(temp_path, cache_path)
                print(f"[Paper Analyzer] 成功將 {len(missing_abstracts)} 篇摘要向量寫入快取。")
            except Exception as e:
                print(f"[Paper Analyzer] 寫入摘要向量快取失敗: {e}")
                
        # 組裝所有向量
        embs = []
        for abs_text in abstracts:
            abs_key = abs_text if abs_text else ""
            if abs_key in cache:
                embs.append(cache[abs_key])
            else:
                self._load_embedding_model()
                single_emb = self.embedding_model.encode(
                    [abs_key], 
                    normalize_embeddings=normalize, 
                    convert_to_numpy=True
                )[0].astype(np.float32)
                embs.append(single_emb)
                
        return np.array(embs, dtype=np.float32)

    def extract_keywords_all(self, top_k=5):
        """
        使用 TF-IDF 對知識庫中的所有論文摘要進行關鍵字萃取
        """
        metadata = self.metadata_manager.get_all_metadata()
        if not metadata:
            return

        paper_ids = list(metadata.keys())
        abstracts = [metadata[pid].get("abstract", "") for pid in paper_ids]

        # 若有些論文沒摘要，填補空字串
        abstracts = [abs_text if abs_text else "No content" for abs_text in abstracts]

        from src.stop_words import CUSTOM_STOP_WORDS
        vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            feature_names = vectorizer.get_feature_names_out()

            for i, pid in enumerate(paper_ids):
                # 取得這篇論文的 TF-IDF 分數
                row = tfidf_matrix.getrow(i).toarray()[0]
                # 取得前 top_k 個最高分的索引
                top_indices = row.argsort()[-top_k:][::-1]
                keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                
                # 更新 Metadata
                self.metadata_manager.metadata[pid]["keywords"] = keywords

            self.metadata_manager.save_metadata()
            print(f"[Paper Analyzer] 成功為 {len(paper_ids)} 篇論文萃取關鍵字！")
        except ValueError as e:
            print(f"[Paper Analyzer] 關鍵字萃取失敗 (文本可能太少): {e}")

    def cluster_papers(self, n_clusters=3):
        """
        Phase 3: 使用 PyTorch Autoencoder 對向量降維，並以 Gaussian Mixture Model (GMM) 進行軟分群 (Soft Clustering)
        """
        import torch
        import torch.nn as nn
        from sklearn.mixture import GaussianMixture

        metadata = self.metadata_manager.get_all_metadata()
        if len(metadata) < n_clusters:
            print("[Paper Analyzer] 論文數量過少，跳過分群。")
            return

        self._load_embedding_model()
        
        paper_ids = list(metadata.keys())
        abstracts = [metadata[pid].get("abstract", "") for pid in paper_ids]
        
        print("[Paper Analyzer] 正在進行摘要向量化 (優先使用快取)...")
        embeddings = self._get_embeddings_for_abstracts(abstracts, normalize=True)
        
        # 1. 深度學習：Autoencoder 降維
        print("[Paper Analyzer] 啟動 Deep Autoencoder 進行特徵降維 (Deep Feature Extraction)...")
        input_dim = embeddings.shape[1]
        latent_dim = min(64, input_dim // 2)
        
        class SimpleAutoencoder(nn.Module):
            def __init__(self, in_dim, lat_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, lat_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(lat_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, in_dim)
                )
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        # 轉 PyTorch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        x_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        
        autoencoder = SimpleAutoencoder(input_dim, latent_dim).to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Train Autoencoder for 50 epochs
        autoencoder.train()
        for epoch in range(50):
            optimizer.zero_grad()
            recon = autoencoder(x_tensor)
            loss = criterion(recon, x_tensor)
            loss.backward()
            optimizer.step()
            
        autoencoder.eval()
        with torch.no_grad():
            latent_embeddings = autoencoder.encoder(x_tensor).cpu().numpy()
            
        del x_tensor, autoencoder
        free_memory(device)
            
        print(f"[Paper Analyzer] 執行 Gaussian Mixture Model (clusters={n_clusters})...")
        # 改進：使用 'diag' 與 reg_covar 防止樣本過少時產生奇異矩陣 (Singular Matrix)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='diag', reg_covar=1e-3)
        gmm.fit(latent_embeddings)
        
        # 取得每個樣本屬於各群的機率 (Soft Clustering)
        cluster_probs = gmm.predict_proba(latent_embeddings)
        cluster_labels = np.argmax(cluster_probs, axis=1)
        
        theme_names = [
            "Topic A (Models/Architecture)",
            "Topic B (Optimization/Training)",
            "Topic C (Applications/NLP)",
            "Topic D (Computer Vision)",
            "Topic E (Generative Models)",
            "Topic F (Reinforcement Learning)",
            "Topic G (Graph/Structured Learning)",
            "Topic H (Efficiency/Compression)",
            "Topic I (Multimodal)",
            "Topic J (Other)"
        ]
        
        for i, pid in enumerate(paper_ids):
            c_label = int(cluster_labels[i])
            c_probs = cluster_probs[i].tolist() # 各群的機率分布
            
            self.metadata_manager.metadata[pid]["cluster_label"] = c_label
            self.metadata_manager.metadata[pid]["cluster_probs"] = c_probs
            self.metadata_manager.metadata[pid]["cluster_name"] = theme_names[c_label] if c_label < len(theme_names) else f"Topic {c_label}"
            
        self.metadata_manager.save_metadata()
        print("[Paper Analyzer] 論文主題 GMM 軟分群完成！")

        # Dynamically rename clusters via LLM topic naming
        try:
            self._load_agent()
            if self.agent:
                print("[Paper Analyzer] Auto-naming clusters using LLM...")
                cluster_names = self.auto_name_clusters(self.agent)
                for pid in paper_ids:
                    c_label = self.metadata_manager.metadata[pid]["cluster_label"]
                    if c_label in cluster_names:
                        self.metadata_manager.metadata[pid]["cluster_name"] = cluster_names[c_label]
                self.metadata_manager.save_metadata()
                print(f"[Paper Analyzer] Dynamic topic labels assigned: {cluster_names}")
        except Exception as e:
            print(f"[Paper Analyzer] Dynamic topic labeling failed: {e}")

    def auto_name_clusters(self, agent) -> dict:
        """
        LLM-based Topic Naming: Queries the LLM Agent to dynamically name clusters based on representative titles.
        """
        from collections import defaultdict
        import re
        all_meta = self.metadata_manager.get_all_metadata()
        cluster_papers = defaultdict(list)
        for pid, m in all_meta.items():
            label = m.get("cluster_label", -1)
            if label >= 0:
                cluster_papers[label].append(m.get("title", pid))

        names = {}
        for label, titles in cluster_papers.items():
            sample = "\n".join(f"- {t}" for t in titles[:4])
            prompt = (f"These research papers belong to one cluster:\n{sample}\n"
                      f"Give a precise 3-5 word research topic label (e.g. 'Optimization and Deep Learning' or 'Large Language Models'):")
            try:
                raw_name = agent.generate(prompt, max_tokens=20, temp=0.0).strip()
                cleaned_name = raw_name.replace('"', '').replace("'", "").strip()
                cleaned_name = re.sub(r'^(Topic\s+[A-Za-z]:\s*|Topic\s+\d+:\s*|Label:\s*)', '', cleaned_name)
                names[label] = cleaned_name
            except Exception as e:
                names[label] = f"Cluster {label}"
        return names

    def recommend_papers_by_interest(self, interest_query: str, top_k=3, diversity_penalty=0.5):
        """
        根據使用者的研究興趣 (Interest Query) 推薦論文。
        v2.2 Feature 5: 加入 MMR (Maximal Marginal Relevance) 確保推薦多樣性。
        diversity_penalty (1-lambda): 值越大，越強調多樣性 (排斥相似論文)
        """
        metadata = self.metadata_manager.get_all_metadata()
        if not metadata:
            return []
            
        self._load_embedding_model()
        
        paper_ids = list(metadata.keys())
        abstracts = [metadata[pid].get("abstract", "") for pid in paper_ids]
        
        query_emb = self.embedding_model.encode([interest_query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        doc_embs = self._get_embeddings_for_abstracts(abstracts, normalize=True)
        
        # 1. 計算所有文件對 Query 的相關度
        sim_to_query = cosine_similarity(query_emb, doc_embs)[0]
        
        # 2. 計算所有文件兩兩之間的相似度矩陣 (供 MMR 懲罰使用)
        sim_matrix = cosine_similarity(doc_embs)
        
        # 3. MMR 迭代選擇
        selected_indices = []
        unselected_indices = list(range(len(paper_ids)))
        
        # 第一篇直接選與 Query 最相關的
        first_best_idx = int(np.argmax(sim_to_query))
        selected_indices.append(first_best_idx)
        unselected_indices.remove(first_best_idx)
        
        # 挑選剩餘的 k-1 篇
        while len(selected_indices) < top_k and unselected_indices:
            mmr_scores = {}
            for idx in unselected_indices:
                # 相關性得分 (Lambda * Sim)
                relevance = (1.0 - diversity_penalty) * sim_to_query[idx]
                
                # 多樣性懲罰 ( (1-Lambda) * Max Sim to already selected )
                # 找出這個候選文章，與「已經選中的文章」們最相似的那篇的分數
                max_sim_to_selected = max([sim_matrix[idx][sel_idx] for sel_idx in selected_indices])
                penalty = diversity_penalty * max_sim_to_selected
                
                mmr_scores[idx] = relevance - penalty
                
            # 選出 MMR 分數最高的
            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        
        # 4. 組裝結果
        results = []
        for idx in selected_indices:
            pid = paper_ids[idx]
            sim_score = float(sim_to_query[idx])
            res = metadata[pid].copy()
            res["paper_id"] = pid
            res["similarity"] = sim_score
            results.append(res)
                
        return results

    def recommend_papers_with_llm(self, interest_query: str, top_k=3):
        """
        Two-Stage Recommendation Pipeline (參考 Hou et al., 2023)
        Stage 1: Bi-Encoder 快速檢索 Top-10 候選
        Stage 2: LLM Zero-Shot Ranker 進行重排並給出推薦理由
        """
        # 1. 取得 Top-10 候選 (Candidate Generation)
        candidates = self.recommend_papers_by_interest(interest_query, top_k=10)
        if not candidates:
            return []
            
        print("[Paper Analyzer] 啟動 LLM 零樣本重排 (Zero-Shot Reranking)...")
        from src.agent import AcademicAgent
        agent = AcademicAgent()
        
        # 2. 構建候選清單字串
        candidate_text = ""
        for i, cand in enumerate(candidates):
            title = cand.get("title", "")
            abs_text = cand.get("abstract", "")[:500] # 只取前500字避免 context 過長
            candidate_text += f"[{i+1}] Title: {title}\nAbstract: {abs_text}...\n\n"
            
        # 3. LLM Prompt 讓它作為推薦器
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI Research Director. Your task is to act as a Zero-Shot Recommender System.
Read the user's research interest and the list of candidate papers.
Select exactly {top_k} papers that BEST match the user's implicit intent, rank them, and provide a 2-3 sentence personalized justification (推薦理由) IN TRADITIONAL CHINESE for each.
Return ONLY a valid JSON array of objects with keys: "rank" (int), "index" (int, 1-10), "reason" (str).
<|eot_id|><|start_header_id|>user<|end_header_id|>
User Interest: {interest_query}

Candidate Papers:
{candidate_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        try:
            response = agent.generate(prompt, max_tokens=600, temp=0.1)
            import re, json
            rankings = []
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                try:
                    rankings = json.loads(match.group(0))
                except Exception:
                    pass
            
            # Fallback: if re.search failed or json.loads failed, try parsing the whole response
            if not isinstance(rankings, list):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        # find the first list value inside the dict
                        for val in parsed.values():
                            if isinstance(val, list):
                                rankings = val
                                break
                except Exception:
                    pass
                    
            if isinstance(rankings, list):
                final_results = []
                for item in rankings:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("index", 1)) - 1
                    if 0 <= idx < len(candidates):
                        cand = candidates[idx].copy()
                        cand["llm_reason"] = item.get("reason", "模型強烈推薦這篇論文適合您的研究方向。")
                        cand["llm_rank"] = item.get("rank", 99)
                        final_results.append(cand)
                
                # 若 LLM 有成功回傳，則返回重排結果
                if final_results:
                    # 照 LLM Rank 排序
                    final_results.sort(key=lambda x: x.get("llm_rank", 99))
                    return final_results[:top_k]
        except Exception as e:
            print(f"[Paper Analyzer] LLM 重排失敗，退回 Bi-Encoder 推薦: {e}")
            
        # 4. Fallback: 如果 LLM 失敗，回傳原本的 Bi-Encoder Top-K
        return candidates[:top_k]

    def compare_two_papers(self, paper_id_1: str, paper_id_2: str):
        """
        比較兩篇論文的相似度與關鍵字差異
        """
        metadata = self.metadata_manager.get_all_metadata()
        if paper_id_1 not in metadata or paper_id_2 not in metadata:
            return None
            
        self._load_embedding_model()
        
        abs1 = metadata[paper_id_1].get("abstract", "")
        abs2 = metadata[paper_id_2].get("abstract", "")
        
        embs = self._get_embeddings_for_abstracts([abs1, abs2], normalize=True)
        sim_score = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
        
        kw1 = set(metadata[paper_id_1].get("keywords", []))
        kw2 = set(metadata[paper_id_2].get("keywords", []))
        
        common_kws = list(kw1.intersection(kw2))
        diff_kw1 = list(kw1 - kw2)
        diff_kw2 = list(kw2 - kw1)
        
        return {
            "similarity": sim_score,
            "common_keywords": common_kws,
            "paper1_unique": diff_kw1,
            "paper2_unique": diff_kw2
        }

    def extract_triplets_with_llm(self, text: str) -> list:
        """
        利用 LLM 從摘要中抽取實體與關係 (Triplets)
        """
        self._load_agent()
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert academic knowledge graph ontology extractor.
Given the abstract of a machine learning paper, extract the core scientific entity relationships (triplets) representing the primary methodology, contributions, and improvements.

CRITICAL RULES:
1. Return ONLY a valid JSON array of triplet objects. DO NOT include any introductory or concluding text, explanations, or markdown code block fences (like ```json).
2. Format:
[
  {{"source": "Entity1", "target": "Entity2", "relation": "relationship_verb"}}
]
3. Keep entities extremely short (1-3 words). Use precise scientific concepts (e.g., "sparse attention", "language model", "attention matrix", "computational complexity").
4. NEVER use generic pronouns, authors, or paper self-references as entities (e.g., "we", "this paper", "the authors", "the proposed method", "the model", "the study", "it", "our algorithm"). Instead, extract the actual scientific concept or model name (e.g., "Big Bird", "Transformer").
5. Relations must be concise lowercase verbs or verb phrases representing physical or mathematical relationships (e.g., "improves", "solves", "uses", "extends", "mitigates", "trained on").
<|eot_id|><|start_header_id|>user<|end_header_id|>
Abstract: {text[:1500]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        response = self.agent.generate(prompt, max_tokens=512, temp=0.0)
        
        # Clean response first
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            lines = cleaned_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_response = "\n".join(lines).strip()
            
        import re
        import json
        try:
            match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Remove trailing commas that violate JSON standard
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                triplets = json.loads(json_str)
                
                valid_triplets = []
                for t in triplets:
                    if not isinstance(t, dict):
                        continue
                    if "source" not in t or "target" not in t or "relation" not in t:
                        continue
                    
                    src = str(t["source"]).strip()
                    tgt = str(t["target"]).strip()
                    rel = str(t["relation"]).strip()
                    
                    # Exclude empty entities and generic self-references
                    invalid_concepts = {
                        "we", "this paper", "the paper", "this study", "the study", "it", "our method", "the method",
                        "our proposed method", "the proposed method", "our model", "the model", "the authors", "authors",
                        "this work", "our work", "work", "paper", "study", "model", "algorithm", "our algorithm"
                    }
                    if not src or not tgt or not rel:
                        continue
                    if src.lower() in invalid_concepts or tgt.lower() in invalid_concepts:
                        continue
                        
                    valid_triplets.append({
                        "source": src,
                        "target": tgt,
                        "relation": rel
                    })
                return valid_triplets
        except Exception as e:
            print(f"[Paper Analyzer] LLM JSON 解析失敗: {e}, Response: {response}")
        return []

    def generate_knowledge_graph(self, output_path=None, use_llm=False):
        """
        生成論文與關鍵字的互動式知識圖譜 (PyVis)
        支援 v2.1 雙軌模式 (TF-IDF vs LLM 語意本體)
        """
        if output_path is None:
            output_path = os.path.join(self.project_paths["project_dir"], "knowledge_graph.html")
            
        output_abs = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_abs)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            import networkx as nx
            from pyvis.network import Network
        except ImportError:
            print("[Paper Analyzer] 請先安裝 pyvis 與 networkx")
            return None

        metadata = self.metadata_manager.get_all_metadata()
        if not metadata:
            return None

        # 如果使用 LLM，我們需要有向圖 (DiGraph) 來表達因果與方法關係
        G = nx.DiGraph() if use_llm else nx.Graph()
        
        cluster_colors = [
            "#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0",
            "#ffb3e6", "#c4e17f", "#76d7c4", "#f7dc6f", "#e59866"
        ]

        if not use_llm:
            # ==========================================
            # v2.0 原版: TF-IDF 基礎模式
            # ==========================================
            keyword_nodes = set()
            for pid, data in metadata.items():
                title = data.get("title", pid)[:30] + "..."
                cluster_id = data.get("cluster_label", 0)
                color = cluster_colors[cluster_id % len(cluster_colors)]
                
                G.add_node(pid, label=title, title=data.get("title", pid), color=color, shape="dot", size=20)
                
                keywords = data.get("keywords", [])
                for kw in keywords:
                    if kw not in keyword_nodes:
                        G.add_node(kw, label=kw, title=f"Keyword: {kw}", color="#f1c40f", shape="star", size=10)
                        keyword_nodes.add(kw)
                    G.add_edge(pid, kw, color="#bdc3c7", weight=1)

            self._load_embedding_model()
            paper_ids = list(metadata.keys())
            if len(paper_ids) > 1:
                abstracts = [metadata[pid].get("abstract", "") for pid in paper_ids]
                embs = self._get_embeddings_for_abstracts(abstracts, normalize=True)
                sim_matrix = cosine_similarity(embs)
                for i in range(len(paper_ids)):
                    for j in range(i + 1, len(paper_ids)):
                        if sim_matrix[i][j] > 0.65:
                            G.add_edge(paper_ids[i], paper_ids[j], color="#3498db", weight=float(sim_matrix[i][j]*5), title=f"Sim: {sim_matrix[i][j]:.2f}")
        else:
            # ==========================================
            # v2.1 升級版: LLM 語意本體萃取 + PageRank
            # ==========================================
            print("[Paper Analyzer] 啟動 LLM 深度本體抽取模式...")
            concept_nodes = set()
            for pid, data in metadata.items():
                title = data.get("title", pid)[:25] + "..."
                abstract = data.get("abstract", "")
                if not abstract:
                    continue
                
                # 將每篇論文作為起點節點
                G.add_node(pid, label=title, title=data.get("title", pid), color="#e74c3c", shape="box", size=30)
                
                print(f"  -> 正在解析: {title}")
                triplets = self.extract_triplets_with_llm(abstract)
                for t in triplets:
                    src = t["source"].strip().title()
                    tgt = t["target"].strip().title()
                    rel = t["relation"].strip().lower()
                    
                    if not src or not tgt: continue
                    
                    for concept in [src, tgt]:
                        if concept not in concept_nodes:
                            G.add_node(concept, label=concept, title=f"Concept: {concept}", color="#2ecc71", shape="dot", size=15)
                            concept_nodes.add(concept)
                            
                    # 論文 -> (討論了) -> Source Concept
                    G.add_edge(pid, src, color="#95a5a6", weight=1, label="discusses")
                    # Source Concept -> (Relation) -> Target Concept
                    G.add_edge(src, tgt, color="#f39c12", weight=2, label=rel, title=rel)

            # v2.4 Feature: GraphRAG Hierarchical Community Detection (Leiden)
            # 只有當節點數足夠 (>= 3) 且有邊時，才進行社群偵測
            if len(G.nodes) >= 3 and len(G.edges) > 0:
                print("[Paper Analyzer] 執行 Hierarchical Leiden Community Detection (GraphRAG v2.4)...")
                try:
                    import cdlib
                    from cdlib import algorithms
                    import json
                    # 轉換為無向圖以進行社群偵測
                    undirected_G = G.to_undirected()
                    
                    community_summaries = {"levels": {}}
                    
                    # 建立節點與整數的雙向映射，以防 cdlib 內部對字串型別節點拋出 ValueError 異常
                    node_list = list(undirected_G.nodes())
                    node_to_int = {node: i for i, node in enumerate(node_list)}
                    int_to_node = {i: node for i, node in enumerate(node_list)}
                    
                    mapped_undirected_G = nx.Graph()
                    for node in node_list:
                        mapped_undirected_G.add_node(node_to_int[node])
                    for u, v, data in undirected_G.edges(data=True):
                        mapped_undirected_G.add_edge(node_to_int[u], node_to_int[v], **data)
                    
                    # Level 0 (細粒度)
                    coms_0 = algorithms.leiden(mapped_undirected_G, weights='weight')
                    level_0_communities = []
                    if coms_0 and hasattr(coms_0, 'communities'):
                        for comm in coms_0.communities:
                            level_0_communities.append([int_to_node[idx] for idx in comm])
                    
                    # Level 1 (粗粒度): 將 Level 0 的社群縮合為節點
                    level_1_communities = []
                    if len(level_0_communities) > 1:
                        # 建立社群縮合圖 G1
                        G1 = nx.Graph()
                        node_to_c0 = {}
                        for i, nodes in enumerate(level_0_communities):
                            G1.add_node(i)
                            for n in nodes:
                                node_to_c0[n] = i
                                
                        for u, v, data in undirected_G.edges(data=True):
                            c_u = node_to_c0.get(u)
                            c_v = node_to_c0.get(v)
                            if c_u is not None and c_v is not None and c_u != c_v:
                                w = data.get('weight', 1)
                                if G1.has_edge(c_u, c_v):
                                    G1[c_u][c_v]['weight'] += w
                                else:
                                    G1.add_edge(c_u, c_v, weight=w)
                                    
                        if len(G1.nodes) > 0:
                            try:
                                coms_1 = algorithms.leiden(G1, weights='weight')
                                # 對應回原節點
                                if coms_1 and hasattr(coms_1, 'communities'):
                                    for c1_nodes in coms_1.communities:
                                        original_nodes = []
                                        for c0_idx in c1_nodes:
                                            if c0_idx < len(level_0_communities):
                                                original_nodes.extend(level_0_communities[c0_idx])
                                        level_1_communities.append(original_nodes)
                                else:
                                    level_1_communities = [level_0_communities[0]] if level_0_communities else []
                            except Exception as e_leiden1:
                                print(f"[Paper Analyzer] Level 1 Leiden 偵測失敗: {e_leiden1}")
                                level_1_communities = [nodes for nodes in level_0_communities]
                        else:
                            level_1_communities = [nodes for nodes in level_0_communities]
                    else:
                        level_1_communities = [level_0_communities[0]] if level_0_communities else []

                    # 整理要儲存的 Levels
                    levels_to_process = {
                        0: level_0_communities,
                        1: level_1_communities
                    }
                        
                    for lvl, comm_list in levels_to_process.items():
                        lvl_summaries = {}
                        
                        # 若為最高層級(最粗粒度 Level 1)，則用於畫面著色
                        if lvl == 1:
                            for c_id, nodes in enumerate(comm_list):
                                comm_color = cluster_colors[c_id % len(cluster_colors)]
                                for node in nodes:
                                    if node in G:
                                        G.nodes[node]['color'] = comm_color
                                        G.nodes[node]['group'] = f"Level_{lvl}_Comm_{c_id}"
                        
                        for c_id, nodes in enumerate(comm_list):
                            # 如果社群夠大，讓 LLM 生成全局社群摘要 (GraphRAG 精神)
                            if len(nodes) >= 3:
                                print(f"  -> 正在為 Level {lvl} Community {c_id} 生成全局主題摘要...")
                                # 只保留存在於 G 中的節點進行摘要，避免 NameError / KeyError
                                existing_nodes = [node for node in nodes if node in G]
                                if not existing_nodes:
                                    continue
                                comm_text = ", ".join(existing_nodes[:30]) # 取最多30個節點
                                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are an AI tasked with analyzing a knowledge graph community for a computer science paper database. Describe the core research theme connecting these entities: {comm_text}. Keep it under 2 sentences in Traditional Chinese.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                                summary = self.agent.generate(prompt, max_tokens=100, temp=0.1)
                                lvl_summaries[f"Community_{c_id}"] = {
                                    "nodes": existing_nodes,
                                    "summary": summary.strip()
                                }
                        
                        community_summaries["levels"][f"Level_{lvl}"] = lvl_summaries
                            
                    # 儲存分層社群摘要，供後續 Map-Reduce RAG 系統使用
                    summary_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "community_summaries.json")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(community_summaries, f, ensure_ascii=False, indent=2)
                    print(f"[Paper Analyzer] 已儲存 GraphRAG 分層社群摘要 (Leiden) 至 {summary_path}")
                        
                except Exception as e:
                    print(f"[Paper Analyzer] Hierarchical Community Detection 失敗: {e}")

            # 應用 PageRank 計算節點重要性，放大核心樞紐節點
            if len(G.nodes) > 0:
                try:
                    pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)
                    for node in G.nodes():
                        base_size = G.nodes[node].get('size', 15)
                        pr_score = pagerank_scores.get(node, 0)
                        # 放大 PageRank 高的節點
                        G.nodes[node]['size'] = base_size + (pr_score * 500)
                except Exception as e:
                    print(f"[Paper Analyzer] PageRank 計算失敗: {e}")

        # 匯出為 PyVis HTML
        # 如果是 use_llm 有向圖，我們開啟箭頭 directed=True
        net = Network(height="700px", width="100%", bgcolor="#0a0c10", font_color="white", select_menu=True, filter_menu=True, directed=use_llm)
        net.from_nx(G)
        
        if use_llm:
            net.set_options("""
            var options = {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -100,
                  "centralGravity": 0.01,
                  "springLength": 200,
                  "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
              },
              "edges": {
                "smooth": {
                  "type": "continuous"
                },
                "font": {
                  "size": 12,
                  "color": "#ffffff",
                  "strokeWidth": 0
                }
              }
            }
            """)
        else:
            net.repulsion(node_distance=150, spring_length=200)
        
        net.save_graph(output_abs)
        
        try:
            gml_path = os.path.join(output_dir, "knowledge_graph.gml")
            nx.write_gml(G, gml_path)
            print(f"[Paper Analyzer] 圖譜資料已儲存: {gml_path}")
        except Exception as e:
            print(f"[Paper Analyzer] 儲存 GML 失敗: {e}")
            
        print(f"[Paper Analyzer] 知識圖譜已生成: {output_abs}")
        return output_abs

    def recommend_papers_by_tree(self, concept: str, use_openalex: bool = False, openalex_strategy: str = "relevance") -> dict:
        """
        v2.3 Feature: Tree-based Relationship Recommendation.
        v2.3.2: Upgraded with RAG-Grounded Tree Generation and Cross-Encoder Verification.
        v2.3.3: OpenAlex Global Citation Integration.
        """
        metadata = self.metadata_manager.get_all_metadata()
        if not metadata:
            return {"tree": None, "recommendations": {}}
            
        self._load_agent()
        self._load_embedding_model()
        self._load_rerank_model()
        
        paper_ids = list(metadata.keys())
        abstracts = [metadata[pid].get("abstract", "") for pid in paper_ids]
        doc_embs = self._get_embeddings_for_abstracts(abstracts, normalize=True)
        
        if use_openalex:
            print(f"[Paper Analyzer] Retrieving Global Citation Tree via OpenAlex for '{concept}' (Strategy: {openalex_strategy})...")
            try:
                from src.openalex_client import OpenAlexClient
                openalex = OpenAlexClient()
                tree = openalex.get_citation_tree(concept, strategy=openalex_strategy)
            except Exception as e:
                print(f"[Paper Analyzer] OpenAlex client error: {e}")
                tree = self.agent.generate_grounded_concept_tree(concept, "")
        else:
            # Phase 0: Local Context Retrieval
            print(f"[Paper Analyzer] Retrieving Local Context for '{concept}'...")
            concept_emb = self.embedding_model.encode([concept], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            concept_sim = cosine_similarity(concept_emb, doc_embs)[0]
            top_context_indices = concept_sim.argsort()[-5:][::-1]
            
            local_context_parts = []
            for idx in top_context_indices:
                title = metadata[paper_ids[idx]].get("title", "Unknown")
                abs_text = abstracts[idx]
                local_context_parts.append(f"Title: {title}\nAbstract: {abs_text}\n")
            local_context = "\n".join(local_context_parts)
            
            print(f"[Paper Analyzer] Generating Grounded Concept Tree for '{concept}'...")
            tree = self.agent.generate_grounded_concept_tree(concept, local_context)
        
        results = {"tree": tree, "recommendations": {}}
        
        for branch, hyde_desc in tree.items():
            if not hyde_desc or "缺乏" in hyde_desc:
                results["recommendations"][branch] = []
                continue
                
            # isinstance check for backward compatibility if the old json is cached
            if isinstance(hyde_desc, list):
                hyde_desc = " ".join(hyde_desc)
                
            # Phase 1: Dense Retrieval using HyDE string
            query_emb = self.embedding_model.encode([hyde_desc], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            sim_scores = cosine_similarity(query_emb, doc_embs)[0]
            
            # Get top 10 papers for this branch for Cross-Encoder re-ranking
            top_k_retrieval = min(10, len(paper_ids))
            top_indices = sim_scores.argsort()[-top_k_retrieval:][::-1]
            
            # Phase 2: Cross-Encoder Verification
            cross_pairs = []
            for idx in top_indices:
                cross_pairs.append([hyde_desc, abstracts[idx]])
                
            rerank_scores = self.rerank_model.predict(cross_pairs)
            
            # Sort by Cross-Encoder score
            reranked_indices = np.argsort(rerank_scores)[::-1]
            
            branch_recs = []
            count = 0
            for r_idx in reranked_indices:
                if count >= 2:
                    break
                orig_idx = top_indices[r_idx]
                pid = paper_ids[orig_idx]
                sim = float(sim_scores[orig_idx])
                rerank_score = float(rerank_scores[r_idx])
                
                branch_recs.append({
                    "paper_id": pid,
                    "title": metadata[pid].get("title", "Unknown Title"),
                    "similarity": sim,
                    "cross_score": rerank_score,
                    "keywords": metadata[pid].get("keywords", [])
                })
                count += 1
                
            results["recommendations"][branch] = branch_recs
            
        return results
