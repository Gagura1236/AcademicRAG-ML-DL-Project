import json
import os

import config

class RagasEvaluator:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir if data_dir is not None else config.DATA_DIR
        self.feedback_file = os.path.join(self.data_dir, "feedback.json")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.feedback_file):
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def evaluate(self, query: str, chunks: list, answer: str) -> dict:
        """
        模擬 RAGAS 評分。
        在實際系統中，應呼叫 LLM 對 faithfulness, answer_relevance, context_precision 進行打分。
        此處給出一個基於檢索結果和答案長度的快速啟發式模擬，或直接返回一個代表性的分數。
        """
        # 模擬打分邏輯
        if not chunks:
            return {"faithfulness": 0.0, "relevance": 0.0, "diversity": 0.0}
            
        # 簡單假設：有 chunk 且 answer 長度夠，給個合理的初始分數
        # 可以在這裡加入調用 LLM 進行 RAGAS 評估的邏輯 (如原本 app.py 中實作)
        # 這裡為了系統流暢，提供一個 0.7~0.9 的隨機預設值，或是根據某些特徵算分
        import random
        
        # 如果 answer 包含 "很抱歉" (anti-hallucination)，faithfulness 依然是高 (誠實)，但 relevance 低
        if "很抱歉" in answer or "無法" in answer:
            return {"faithfulness": 1.0, "relevance": 0.2, "diversity": 0.5}
            
        return {
            "faithfulness": min(1.0, 0.6 + 0.05 * len(chunks) + random.uniform(0, 0.1)),
            "relevance": min(1.0, 0.7 + 0.02 * len(chunks) + random.uniform(0, 0.1)),
            "diversity": min(1.0, 0.5 + 0.1 * len(set([c.get('cluster_label', 0) for c in chunks])))
        }

    def save_feedback(self, query: str, answer: str, feedback_type: str):
        """
        儲存使用者回饋 (RLHF) 到 feedback.json
        feedback_type: 'thumb_up' 或 'thumb_down'
        """
        feedbacks = []
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    feedbacks = json.load(f)
            except Exception:
                feedbacks = []
                
        feedbacks.append({
            "query": query,
            "answer_preview": answer[:50] + "...",
            "feedback": feedback_type,
            "timestamp": __import__("time").time()
        })
        
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
