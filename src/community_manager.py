import os
import json

class CommunityManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            try:
                import config
                self.data_dir = config.DATA_DIR
            except ImportError:
                self.data_dir = "data"
        else:
            self.data_dir = data_dir
        self.summary_path = os.path.join(self.data_dir, "community_summaries.json")
        self.gml_path = os.path.join(self.data_dir, "knowledge_graph.gml")
        self.summaries = self._load_summaries()
        self.graph = self._load_graph()

    def _load_summaries(self):
        if os.path.exists(self.summary_path):
            with open(self.summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_graph(self):
        import networkx as nx
        if os.path.exists(self.gml_path):
            try:
                return nx.read_gml(self.gml_path)
            except Exception as e:
                print(f"[Community Manager] Error loading graph: {e}")
        return None

    def get_summary_for_cluster(self, cluster_label):
        """相容舊版的單層叢集查詢"""
        label_str = str(cluster_label)
        # 如果是舊版格式
        if "levels" not in self.summaries:
            return self.summaries.get(label_str, "")
        return ""

    def get_all_summaries(self):
        return self.summaries

    def get_global_context_prompt(self, target_level=None):
        """組合社群摘要，作為 Global Context 注入 LLM Prompt (舊版做法)"""
        if not self.summaries:
            return ""
            
        context = "【Global Knowledge Graph Community Summaries】\n"
        
        # 處理 v2.4 新版階層式結構
        if "levels" in self.summaries:
            levels_dict = self.summaries["levels"]
            # 預設使用最粗粒度的層級 (通常是數字最大的)
            if target_level is None and levels_dict:
                target_level = list(levels_dict.keys())[-1]
                
            if target_level in levels_dict:
                for c_id, data in levels_dict[target_level].items():
                    context += f"- {c_id}: {data.get('summary', '')}\n"
        else:
            # 處理 v2.3 舊版結構
            for label, data in self.summaries.items():
                if isinstance(data, dict):
                    context += f"- Community {label}: {data.get('summary', '')}\n"
                else:
                    context += f"- Community {label}: {data}\n"
                    
        return context

    def pagerank_search(self, query: str, top_k_nodes: int = 5):
        """
        [v2.4 GraphRAG] Personalized PageRank 搜尋
        尋找圖譜中與 Query 最相關的核心節點
        """
        if not self.graph:
            return []
            
        import networkx as nx
        import jieba
        
        # 定義無意義的標點符號黑名單 (刻意排除可能作為數學意義的符號如 + - * / = < > 等)
        useless_punct = set(" \t\n\r,.:;!?()[]{}'\"“”‘’，。、；：？！（）【】《》〈〉")
        # 分詞並過濾，保留長度>0 且不在黑名單內的詞彙 (這樣單一英文字母或數學符號也能被保留)
        query_words = {w for w in jieba.cut(query.lower()) if w not in useless_punct and len(w.strip()) > 0}
        
        personalization = {}
        for node in self.graph.nodes():
            # 若節點名稱包含在 query_words 中，給予較高的初始權重
            # 或計算重疊度
            node_str = str(node).lower()
            overlap = len([w for w in query_words if w in node_str])
            personalization[node] = overlap + 0.1 # 基礎微小機率避免0
            
        try:
            pr_scores = nx.pagerank(self.graph, alpha=0.85, personalization=personalization, max_iter=100)
            sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:top_k_nodes]
        except Exception as e:
            print(f"[Community Manager] PPR 失敗: {e}")
            return []

    def map_reduce_search(self, agent, query: str, top_k_communities: int = 3):
        """
        [v2.4 GraphRAG Map-Reduce Global Search]
        Map: 平行評估所有社群摘要與查詢的相關性，並提取關鍵資訊。
        Reduce: 將高分社群的資訊合併，生成最終全局答案。
        """
        if not self.summaries or "levels" not in self.summaries:
            # Fallback to empty if no data or old format
            return {"answer": "No valid hierarchical community summaries found.", "mapped_communities": []}
            
        levels_dict = self.summaries["levels"]
        if not levels_dict:
             return {"answer": "No community summaries available.", "mapped_communities": []}
             
        target_level = list(levels_dict.keys())[-1]
        communities = levels_dict[target_level]
        
        mapped_results = []
        
        # [v2.4 GraphRAG] PPR 節點過濾機制
        ppr_top_nodes = self.pagerank_search(query, top_k_nodes=10)
        ppr_node_names = {node for node, score in ppr_top_nodes}
        
        # --- MAP 階段 (Pre-filter by PPR to save memory and time) ---
        # 1. 根據 PPR overlap_nodes 排序社群
        scored_communities = []
        for c_id, data in communities.items():
            summary_text = data.get('summary', '')
            if not summary_text: continue
            
            community_nodes = set(data.get('nodes', []))
            overlap_nodes = ppr_node_names.intersection(community_nodes)
            scored_communities.append({
                "c_id": c_id,
                "summary": summary_text,
                "overlap_nodes": overlap_nodes,
                "overlap_count": len(overlap_nodes)
            })
            
        # 排序：優先處理與 PPR Top Nodes 重疊最多、或有重疊的社群，最多只送前 top_k_communities * 2 個給 LLM 評估
        scored_communities.sort(key=lambda x: x["overlap_count"], reverse=True)
        candidates_to_map = scored_communities[:top_k_communities * 2]
        
        # 2. 對篩選後的候選社群進行 LLM Map
        for cand in candidates_to_map:
            c_id = cand["c_id"]
            summary_text = cand["summary"]
            overlap_nodes = cand["overlap_nodes"]
            
            ppr_hint = ""
            if overlap_nodes:
                ppr_hint = f"\nNote: This community contains highly central nodes for this query: {', '.join(overlap_nodes)}"
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert academic evaluator. 
Evaluate how relevant the following community summary is to the user query.
Rate the relevance from 0 to 100. If it is relevant (> 50), extract the key points that answer the query.
Respond in exactly this JSON format: {{"score": 85, "points": "The extracted points..."}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}
Community Summary: {summary_text}{ppr_hint}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            try:
                # 調用 Agent 生成評估
                response = agent.generate(prompt, max_tokens=150, temp=0.1)
                # Parse JSON
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    res_dict = json.loads(json_match.group())
                    score = res_dict.get("score", 0)
                    points = res_dict.get("points", "")
                    if score > 50:
                        mapped_results.append({
                            "c_id": c_id,
                            "score": score,
                            "points": points,
                            "original_summary": summary_text
                        })
            except Exception as e:
                print(f"[Map-Reduce] Map phase error on {c_id}: {e}")
                continue
                
        # 排序並取 Top K
        mapped_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = mapped_results[:top_k_communities]
        
        if not top_results:
            return {"answer": "None of the knowledge communities contained relevant information for this query.", "mapped_communities": []}
            
        # --- REDUCE 階段 ---
        combined_points = "\n".join([f"- {r['c_id']} (Score {r['score']}): {r['points']}" for r in top_results])
        reduce_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a senior academic assistant executing the Reduce phase of a GraphRAG system.
Synthesize a comprehensive global answer to the user's query using ONLY the extracted points from various knowledge communities below.
Write in Traditional Chinese.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}

Extracted Community Points:
{combined_points}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        try:
            final_answer = agent.generate(reduce_prompt, max_tokens=500, temp=0.3)
        except Exception as e:
            final_answer = f"Error in Reduce phase: {e}"
            
        return {
            "answer": final_answer.strip(),
            "mapped_communities": top_results
        }
