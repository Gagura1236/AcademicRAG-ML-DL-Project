class DynamicAlpha:
    def __init__(self, default_alpha: float = 0.5):
        self.default_alpha = default_alpha
        self.current_alpha = default_alpha

    def adjust(self, ragas_faithfulness: float, ragas_relevance: float) -> float:
        """
        根據 RAGAS 分數動態調整 Alpha。
        如果 faithfulness 很高，可以更相信檢索結果的相關性，將 alpha 偏向 relevance。
        如果 faithfulness 很低，可能是檢索結果太發散或不準確，可以稍微調降 alpha。
        """
        # 這裡實作一個簡單的動態調整邏輯
        # 假設預設為 0.5
        # 如果 faithfulness 高 (>0.8)，我們提高 alpha 以專注在 relevance (精準檢索)
        # 如果 faithfulness 低 (<0.5)，我們降低 alpha 以增加多樣性或擴大搜尋範圍
        
        if ragas_faithfulness > 0.8:
            self.current_alpha = min(0.9, self.default_alpha + 0.2)
        elif ragas_faithfulness < 0.5:
            self.current_alpha = max(0.1, self.default_alpha - 0.2)
        else:
            self.current_alpha = self.default_alpha
            
        return self.current_alpha

    def get_alpha(self) -> float:
        return self.current_alpha
