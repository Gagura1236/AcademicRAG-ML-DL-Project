# Phase 1 Checklist (v1.0 MVP Restructuring & Resource Safeguarding)

- `[x]` **1. 專案目錄重構 (v1.0 Reorganization)**
    - `[x]` 建立 `v1.0` 目錄並移動所有程式碼與 `src/` 目錄
    - `[x]` 調整 `v1.0/config.py` 中的 `DATA_DIR` 指向最外層的 `../data`
    - `[x]` 移除最外層重複的舊程式碼檔案
- `[x]` **2. 5 分鐘閒置自毀保護機制 (Resource Safeguard)**
    - `[x]` 在 `v1.0/app.py` 中引入活動更新計時器 `GLOBAL_LAST_ACTIVE`
    - `[x]` 建立背景守護線程 `IdleMonitor` 監控 5 分鐘無活動時自動終止 (`os._exit(0)`)
    - `[x]` 在 Streamlit GUI 中加強提示，顯示目前守護狀態
- `[x]` **3. 端到端管道驗證 (Verification)**
    - `[x]` 在 `v1.0` 資料夾中執行並通過 `verify_pipeline.py`
    - `[x]` 手動啟動 Streamlit `app.py` 驗證 UI 操作與 5 分鐘自毀機制
- `[x]` **4. 修正 Paper ID Dot Bug 並重新索引經典論文**
    - `[x]` 修正 `v1.0/src/pdf_parser.py` 中的 `split(".")[0]` 為 `os.path.splitext`
    - `[x]` 修正 `v1.0/verify_pipeline.py` 中的 `split(".")[0]`
    - `[x]` 修正 `v1.0/app.py` 中的 `split(".")[0]`
    - `[x]` 重新對齊並索引 "Attention Is All You Need" (`1706.03762`) 經典論文，確保 LaTeX 對齊無誤

## Phase 1 Extension: 經典論文索引與核心 ML 檢索診斷

- `[x]` **5. 經典 AI/Transformer 論文下載與高精度索引**
    - `[x]` 自動下載 BERT (`1810.04805`), Big Bird (`2007.14062`), Longformer (`2004.05150`) 論文
    - `[x]` 完美解壓 LaTeX 原始碼套件並執行 Ground Truth 公式對齊
    - `[x]` 修正 Nougat OCR 在 Longformer 中因 transformers 庫引發的 `do_crop_margin` 校驗 Bug
    - `[x]` 成功將 488 個新學術切片寫入 FAISS 向量庫 (當前總 Chunks 數: 1887)
- `[x]` **6. 深度 ML/DL 檢索系統效能評估與診斷**
    - `[x]` 執行 `evaluation.py` 對比 Bi-Encoder 與 Cross-Encoder 二次重排的表現
    - `[x]` 獲取 Precision@K, Recall@K, MRR 指標
    - `[x]` 分析對 LaTeX 數學公式的 Domain Gap (ROC-AUC / Margin 分佈)
    - `[x]` 形成過擬合/欠擬合診斷報告與 Phase 2 PEFT (LoRA) 微調建議

## Phase 2: 核心 ML/DL 參數微調 (LoRA PEFT & M4 MPS 加速)

- `[x]` **7. 開發環境與依賴配置**
    - `[x]` 在 M4 Mac 本地虛擬環境中安裝 Hugging Face `peft` 和 `accelerate`
    - `[x]` 更新主專案的 `requirements.txt` 以保留依賴記錄
- `[x]` **8. 設計並建立 LoRA 核心微調模組 (finetune.py)**
    - `[x]` 實作學術級正負樣本對自動生成器
    - `[x]` 實作同篇論文 Hard Negatives (硬負樣本) 對抗挖掘邏輯
    - `[x]` 配置 `LoraConfig` 並為 Cross-Encoder 自注意力機制注入 LoRA 旁路 (r=8)
    - `[x]` 引入 L2 Regularization (Weight Decay) 與溫度係數平滑 Loss，預防過擬合
- `[x]` **9. RAG 引擎整合與動態 Adapter 加載**
    - `[x]` 修改 `embedding_search.py`，啟動時自動檢測並無縫加載 `data/lora_adapter/`
- `[x]` **10. Streamlit GUI 升級與前後效能對比看板**
    - `[x]` 在 `app.py` 中全新建立「Tab 4: 🧠 深度微調與診斷面板」
    - `[x]` 實作網頁端一鍵啟動 MPS 加速微調、即時進度條與日誌串流
    - `[x]` 設計並列的前後對比指標卡 (ROC-AUC, Margin, 診斷狀態)
