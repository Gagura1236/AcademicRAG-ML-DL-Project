import os
import sys

# 獲取專案根目錄
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.embedding_search import AcademicRAGEngine
from src.agent import AcademicAgent
from src.ragas_evaluator import RagasEvaluator
from src.dynamic_alpha import DynamicAlpha
from src.community_manager import CommunityManager
from src.paper_analyzer import PaperAnalyzer

class RetrievalEngine:
    def __init__(self, rag_engine=None, agent=None):
        self.rag = rag_engine if rag_engine else AcademicRAGEngine()
        self.agent = agent if agent else AcademicAgent()
        self.ragas = RagasEvaluator()
        self.alpha_manager = DynamicAlpha()
        self.community_manager = CommunityManager()
        self.analyzer = PaperAnalyzer()
        
    def run(self, query_zh: str, query_en: str = None, max_hops: int = 5, use_llm: bool = True, target_projects = None) -> dict:
        """
        端到端檢索管線 (v2.2 整合版)
        """
        results_log = []
        
        # 1. 中翻英
        if not query_en:
            query_en = self.agent.translate_to_english(query_zh)
            results_log.append(f"🔄 翻譯：{query_en}")
        else:
            results_log.append(f"🔄 沿用已翻譯指令：{query_en}")
        
        # 2. 獲取 Alpha
        alpha = self.alpha_manager.get_alpha()
        results_log.append(f"🎛️ 當前 Alpha：{alpha:.2f}")
        
        # 3. Multi-Hop 檢索迴圈
        all_chunks = []
        hop_count = 0
        current_query = query_en
        
        while hop_count < max_hops:
            hop_count += 1
            
            # 使用 AcademicRAGEngine 進行檢索 (此處底層已支援 Dynamic Alpha Tuning 與 Query Expansion)
            # 這裡我們利用 Orchestrator 再做一次高階管控
            chunks = self.rag.search(current_query, top_k=5, use_query_expansion=True, target_projects=target_projects)
            if not chunks:
                break
                
            # 去重加入
            existing_texts = {c.get("text") for c in all_chunks}
            for c in chunks:
                if c.get("text") not in existing_texts:
                    all_chunks.append(c)
                    existing_texts.add(c.get("text"))
                    
            # 將這些 chunks 整理成上下文
            context_text = "\n\n".join([f"[論文 {c.get('paper_id')}]\n{c.get('text')}" for c in all_chunks])
            
            # 判斷是否需要下一跳 (Self-Reflection)
            # 假設 LLM 能判斷資訊是否足夠
            if hop_count < max_hops:
                reflection = self.agent.evaluate_information_needs(current_query, context_text)
                # evaluate_information_needs 回傳 str。若包含 "SUFFICIENT" 則視為充足。
                if "SUFFICIENT" in reflection.upper():
                    results_log.append(f"✅ Hop {hop_count}: 資訊已充足。")
                    break
                else:
                    current_query = f"{query_en} focusing on {reflection}"
                    results_log.append(f"🔄 Hop {hop_count}: 資訊不足，擴展查詢 -> {current_query}")
            else:
                results_log.append(f"🛑 達到最大 Hop 數 {max_hops}。")

        # 4. 加入 GraphRAG 全局社群上下文 或 Map-Reduce 答案
        community_context = ""
        map_reduce_answer = None
        
        # 假設使用者可以在 UI 傳遞參數或我們自動判斷 (預設先使用 map_reduce 如果有 agent)
        use_map_reduce = True 
        
        if use_map_reduce and hasattr(self.community_manager, "map_reduce_search"):
            results_log.append(f"🌐 啟動 GraphRAG Map-Reduce Global Search...")
            mr_result = self.community_manager.map_reduce_search(self.agent, query_en)
            map_reduce_answer = mr_result.get("answer", "")
            mapped_comms = mr_result.get("mapped_communities", [])
            
            if map_reduce_answer and mapped_comms:
                results_log.append(f"✅ Map-Reduce 成功，提取了 {len(mapped_comms)} 個高度相關的知識社群。")
                community_context = f"【GraphRAG Global Knowledge Context】\n{map_reduce_answer}\n"
        else:
            community_context = self.community_manager.get_global_context_prompt()
            
        final_context = community_context + "\n\n" + "\n\n".join([f"[論文 {c.get('paper_id')}]\n{c.get('text')}" for c in all_chunks])
        
        # 5. LLM 生成初步答案 (v2.5 Multi-Agent Debate)
        answer = ""
        debate_log = []
        if use_llm:
            if hasattr(self.agent, "debate_loop"):
                results_log.append("⚔️ 啟動 Multi-Agent Debate 雙智能體辯論機制...")
                debate_res = self.agent.debate_loop(query_en, final_context, max_rounds=3)
                answer = debate_res.get("final_answer", "")
                debate_log = debate_res.get("debate_log", [])
            else:
                answer = self.agent.synthesis_agent(query_en, final_context)
            
        # 6. RAGAS 評分與 Alpha 調整
        ragas_scores = self.ragas.evaluate(query_en, all_chunks, answer)
        new_alpha = self.alpha_manager.adjust(ragas_scores.get("faithfulness", 0.5), ragas_scores.get("relevance", 0.5))
        
        # [Phase 2] Reflexion Agent: 如果評分不佳，自動反思並發起第二次檢索
        if use_llm and hasattr(self.agent, "self_critique_and_refine"):
            reflexion = self.agent.self_critique_and_refine(query_en, answer, ragas_scores)
            if reflexion["needs_refinement"]:
                results_log.append(f"⚠️ [Reflexion Agent] 檢測到幻覺或不切題！觸發自我修正機制。")
                results_log.append(f"🔄 Reflexion 新關鍵字: {reflexion['new_query']}")
                
                # 發起第二次檢索 (補充上下文)
                extra_chunks = self.rag.search(reflexion['new_query'], top_k=3, use_query_expansion=True, target_projects=target_projects)
                for ec in extra_chunks:
                    if ec.get("text") not in {c.get("text") for c in all_chunks}:
                        all_chunks.append(ec)
                        
                final_context = community_context + "\n\n" + "\n\n".join([f"[論文 {c.get('paper_id')}]\n{c.get('text')}" for c in all_chunks])
                
                # 重新合成答案
                results_log.append("✍️ [Reflexion Agent] 正在根據擴展的上下文重新合成最終答案...")
                answer = self.agent.synthesis_agent(query_en, final_context)
                
                # 重新計算 RAGAS
                ragas_scores = self.ragas.evaluate(query_en, all_chunks, answer)
                results_log.append(f"✅ Reflexion 修正完成！新 Faithfulness={ragas_scores.get('faithfulness', 0):.2f}")
        
        # 7. (Phase 1C) 為 Top 3 搜尋結果產生可解釋性推薦理由
        if use_llm and hasattr(self.agent, "explain_result"):
            results_log.append("💡 正在為 Top 3 搜尋結果產生推薦解釋...")
            for i, chunk in enumerate(all_chunks[:3]):
                try:
                    chunk["llm_explanation"] = self.agent.explain_result(query_en, chunk["text"])
                except Exception as e:
                    chunk["llm_explanation"] = f"無法產生解釋: {e}"
        
        return {
            "query_en": query_en,
            "answer": answer,
            "chunks": all_chunks,
            "ragas_scores": ragas_scores,
            "new_alpha": new_alpha,
            "log": results_log,
            "debate_log": debate_log,
            "community_context_used": bool(community_context)
        }
