# CHANGELOG

本文件記錄 AcademicRAG 各版本的功能演進與修正紀錄。

---

## [v3.9.4] — 2026-06-29 🚀 聯邦跨專案檢索與學術文獻 AI 診斷 (Federated Search & Academic AI Advisor)

### 核心演算法與架構升級 (Core Algorithms & Architecture Upgrades)

- **`src/tools/rag_tool.py` & `src/retrieval_engine.py` (跨專案聯邦檢索)** — 實作了跨專案聯邦式檢索 (Federated Search)。Agentic Chat 對話時能自動讀取側邊欄勾選的多個目標專案，動態並行檢索多個 FAISS 向量索引庫，並透過 Cross-Encoder 進行全域重排序 (Global Reranking) 與來源標記，打破專案間的知識孤島。
- **`src/finetune.py` (防範特徵崩塌)** — 在 Hybrid Hard Negative Mining 中實施了「正樣本保護機制」，當稠密檢索的 Rank < 3 時，即使其未包含稀疏關鍵字，也絕對禁止將其判定為硬負樣本。此優化徹底解決了因 False Negatives 導致的特徵向量邊界收縮與 Cross-Encoder 特徵崩塌 (Representation Collapse) 問題。
- **`ui/step6_eval.py` (學術文獻對比診斷)** — 實作了 AI 性能對比診斷與建議引擎 (`_get_advice`)。比對微調前後的 ROC-AUC 與 Margin 變化，結合經典文獻理論：
  - *He et al. (2021) DeBERTa* 限制學習率極限；
  - *Hu et al. (2022) LoRA* 的 Rank 分配；
  - *Nogueira & Cho (2019)* 的過擬合 Epoch 判定；
  - *Chen et al. (2020) SimCSE* 對對比學習 Margin 崩塌的診斷。
  並直接提供文獻來源與具體的參數微調建議。
- **`src/evaluation.py` (39 篇論文經典測試集)** — 擴充了 `Classic 39 Papers Benchmark` 評估測試集，內建 15 組覆蓋 ResNet, Transformer, GAN, YOLO 等經典 AI 領域的學術 Ground-Truth 查詢，供助教進行一鍵效能診斷。
- **`src/metadata_extractor.py` (動態年份萃取)** — 實作了 arXiv ID 年份自動提取算法，自動將 39 篇經典文獻與未來上傳 PDF 的出版年份解析存入 JSON 中，修正了文獻清單與介面上顯示為 N/A 的問題。

### 專案清理與瘦身 (Workspace Cleanup & Optimization)

- **無用資料夾清理** — 徹底移除了 `v3.9.4/` 內多餘的 `v1.0/` 與 `v2.0/` 暫存重複資料夾；同時刪除了未使用的本地 PDF 參考目錄 `references/`，清出空間。
- **打包管線升級 (`package_project.py`)** — 將開發歷程檔案 `dev_records/` 與環境隱私設定納入打包排除，將標準壓縮檔大小從 70.71 MB 優化精簡至 **`65.33 MB`**。

---

## [v3.9.2] — 2026-06-22 📐 數學映射與介面防呆 (Mathematical Mapping & UI Robustness)

### 準確度與介面修正 (Accuracy & UI Bug Fixes)

- **`ui/step3_alpha.py`** 與 **`ui/step5_mmr.py`** — 修復因強制截斷（Clamping `max(0.0, sim)`）導致負數餘弦相似度失去數學與語意解釋性的問題。
- 實作了完美的數學線性映射 `(sim + 1.0) / 2.0`，將底層 FAISS 或 Bi-Encoder 的 `[-1, 1]` 餘弦距離無損投射至 `[0, 1]` 的 UI 進度條安全區間，徹底排除介面被負數相似度干擾的 Bug，同時保留正確的負相關呈現。

---

## [v3.9.1] — 2026-06-21 🛠️ Bug Remediation & Interoperability Upgrades

### 系統協作性與缺陷修復 (Bug Remediation & Interoperability)

- **`src/tools/rag_tool.py`** — 將 RAG 搜尋改為 **附加 (Append)** 模式，只在分數高於 `0.35` 時附加，並在 Observation 中嵌入 `[Doc X]` 標籤。
- **`ui/step4_agentic.py`** — 呼叫 `agent.run` 時傳入 `st.session_state.messages[:-1]` 以排除當前輪的使用者問題，防止 LLM context 對話記憶重複。
- **`src/community_manager.py`** — 修正 `data_dir` 預設路徑，使其動態讀取專案配置的 `config.DATA_DIR`。
- **`src/agent.py`** — 實作並註冊 `search_knowledge_graph` 工具，讓 Agent 能讀取 Leiden 社群主題摘要 `community_summaries.json` 回答全局問題。

---

## [v3.9] — 2026-06-21 🕸️ GraphRAG Triplet Extraction Upgrades

### 準確度與穩定性優化 (Accuracy & Stability Enhancements)

- **`src/paper_analyzer.py`** — 升級學術本體關係抽取提示詞（`extract_triplets_with_llm`）。限制實體命名規則為 1-3 字精準科學概念，並過濾掉 user/paper/study 等自我參照詞，專注於學術概念提取。
- **`src/paper_analyzer.py`** — 實作了 Markdown json 標記剝除與正規表達式 trailing commas（結尾逗號）移除邏輯，提高 JSON 解析強健度。
- **`src/paper_analyzer.py`** — 實作字串節點與整數 ID 的雙向映射，使 Leiden 分層社群偵測（`cdlib.algorithms.leiden`）能在處理含有字串型別節點的無向圖時完美運作，杜絕 `ValueError` 型別錯誤。
- **`src/paper_analyzer.py`** — 在社群摘要生成與節點渲染著色中加入存在性檢查 (`if node in G`)，防範 `KeyError`。

---

## [v3.8] — 2026-06-21 💬 連續對話記憶 (Conversational UX)

### 使用者體驗優化 (User Experience Enhancements)

- **`src/agent.py`** — 更新 `router_agent` 與 `synthesis_agent` 的方法簽章與提示詞模板，引入 `chat_history`。使 Agent 的意圖路由器與最終解答合成器均能完整解讀先前的對話上下文，從而支援追問機制。
- **`ui/step4_agentic.py`** — 修改 `agent.run` 呼叫，將 Streamlit 的歷史對話紀錄 (`st.session_state.messages`) 傳入背景引擎，實作多輪對話與上下文感知能力。

---

## [v3.7] — 2026-06-21 📑 來源歸因機制 (Inline Citations)

### 使用者體驗優化 (User Experience Enhancements)

- **`src/agent.py`** — 更新 `System Prompt`，導入 `CITATION RULE`，強制限制 LLM 生成 Final Answer 時，對應本地庫檢索段落之內容必須使用 `[Doc X]` 格式標記出處來源，拒絕含糊引用。
- **`ui/step4_agentic.py`** — 實作了 `format_citations_html` 解析器。當 Agent 回答包含 `[Doc X]` 時，將其自動轉換為懸停（Hover）提示之 HTML Tag。當使用者將滑鼠指針懸停於該引用標記上時，會自動彈出**原論文標題、所在頁碼以及該檢索段落的原文摘要**，保障學術透明度。

---

## [v3.6] — 2026-06-21 🌐 前端視覺透明化 (CRAG UI Indicator)

### 使用者體驗優化 (User Experience Enhancements)

- **`ui/step4_agentic.py`** — 在對話歷程 (Thought Trace) 的 `trace_callback` 攔截 `CRAG 評估警告`。當偵測到警告時，在 UI 上動態渲染出黃色高亮的 `🛡️ CRAG 觸發 (Corrective RAG): 本地庫查無高度相關資訊，已強制拒絕使用並轉向全網檢索 🌐` 提示框，將代理人的內部判斷思維完全對使用者透明化。

---

## [v3.5] — 2026-06-21 🛡️ 防呆與防幻覺機制 (CRAG Evaluator)

### 準確度優化 (Accuracy Enhancements)

- **`src/tools/rag_tool.py`** — 實作了信心度門檻評估。將篩選門檻由原先的 `0.15` 大幅拉高至 `0.35`；且若最優結果分數低於 `0.35`，則硬性攔截不回傳無關段落，改為返回專屬的 `CRAG 評估警告 (Corrective RAG Warning)`，截斷幻覺的文本來源。

---

## [v3.4] — 2026-06-21 🤖 AI 自我修正 (Self-Correction Prompt)

### 準確度優化 (Accuracy Enhancements)

- **`src/agent.py`** — 更新 `System Prompt`，導入 `CRAG (Corrective RAG) RULE`，要求 LLM 在遇到本地庫檢索信心不足（相似度低於 0.15 或無相關文獻）時，不准憑空幻想，必須主動切換使用 `search_arxiv` 連網工具檢索，或老實告知資料庫無相關文獻。

---

## [v3.3.1] — 2026-06-21 🔖 穩定版 (Stable Release)

### 驗證與修補 (Verification & Patch)

這是 v3.3 升級後的穩定性驗收版本。所有模組均通過 `py_compile` 語法驗證與跨模組引用健康檢查。

#### ✅ 通過驗證的功能模組
- **`src/preference_store.py`**：`save_preference` / `load_preference_pairs` 匯入正常
- **`src/semantic_scholar_client.py`**：`enrich_paper` / `get_recommendations` 匯入正常
- **`src/metadata_extractor.py`**：`backfill_all_keywords` 定義完整，Semantic Scholar 整合確認
- **`src/paper_analyzer.py`**：`auto_name_clusters` 方法存在，LLM 動態命名在 GMM 分群後自動觸發
- **`src/agent.py`**：`router_agent` / `debate_loop` / `run` / `explain_result` / `_tool_paper_tldr` / `_tool_recommend_related` 全部方法確認存在並正確接線
- **`src/finetune.py`**：DPO 偏好資料 (`load_preference_pairs`) 與 RLHF 真實回饋正確整合至訓練集
- **`ui/step1_ingestion.py`**：重建向量庫後自動觸發 `backfill_all_keywords` 確認
- **`ui/step3_alpha.py`**：結構化卡片、三維分數分解、DPO 回饋按鈕確認
- **`ui/step4_agentic.py`**：Debate 開關、`use_debate` 傳遞至 `agent.run()` 確認

#### 🔍 已知邊界狀況 (Acceptable Known Limitations)
- `auto_name_clusters` 只在 `step2_graphrag.py` 透過 `paper_analyzer.py` 的 `analyze_topics()` 間接呼叫；Step 2 本身僅呈現已命名的主題，無需再顯式呼叫。
- `_tool_paper_tldr` / `_tool_recommend_related` 在 agent 內部以 `from src.semantic_scholar_client import ...` 延遲匯入，此路徑在 Streamlit 的工作目錄 (`v3.0/`) 下正確解析。

---

## [v3.3] — 2026-06-20 🏆 終極優化版

### 新增功能

#### 🤖 Agentic AI 自主思考 (ReAct Loop)
- **`src/agent.py`** — 新增 `react_loop()` 與 `debate_loop()`
  - 實作 ReAct 思考推理架構。Agent 接收問題後，強制輸出 `Thought:`，自我思考並決定下一步行動。
  - 新增多代理人對抗辯論機制 (Multi-Agent Debate)，由 Generator Agent 與 Critic Agent 進行多輪論證與反思，減少幻覺。
- **`ui/step4_agentic.py`** — 新增 Thought Trace 可視化介面
  - 使用 `st.status` 摺疊選單即時展現 Agent 的思考旅程（Thought ➔ Action ➔ Observation ➔ Final Answer）。

#### 🔌 連網外掛工具 (ArXiv Tool)
- **`src/tools/arxiv_tool.py`** — 新增 ArXiv API 連網搜尋工具
  - 當本機知識庫無法完全涵蓋使用者的查詢時，Agent 會自動呼叫 `search_arxiv` 連網抓取最新文獻的摘要，解決資訊孤島問題。

### Bug 修復 & 穩定性優化 (MPS SIGSEGV 防禦)

- **`app.py` & `test_pipeline.py`** — 實施 Early Import Guard
  - 將 `transformers` 的調度器 `get_linear_schedule_with_warmup` 移至程式最頂層導入，解決 PyTorch 與 Metal 驅動初始化衝突引發的段錯誤 (SIGSEGV / Exit Code 139)。
- **`src/finetune.py`** — 實施 Lazy PEFT Instantiation
  - 將 PEFT 相關模組的導入與實例化移至微調函數內部，避免 PEFT 動態裝飾器提前修改類別造成的記憶體混亂。
- **`src/finetune.py`** — 啟用 MPS-CPU 混合運行架構
  - 檢測到 `mps` 裝置時，微調訓練自動回退至 CPU（使用 FP32 確保梯度數值穩定）；而日常檢索與 Rerank 推理則維持使用 GPU/MPS 半精度加速。
- **`src/embedding_search.py`** — 修正多進程與 PEFT 裝置不齊問題
  - 修正了 PEFT Model 在載入後其部分權重未放置在正確裝置上的問題。

---

## [v3.2.1] — 2026-06-20

### 新增功能

- **`src/finetune.py`** — 支援 Multi-GPU DataParallel 訓練
  - 偵測到多張顯卡時，微調迴圈自動呼叫 `nn.DataParallel` 包裝模型，進行分散式平行微調。
- **`src/embedding_search.py`** — 支援 SentenceTransformer Pool 平行編碼建庫
  - 在大批量編碼建庫時，自動啟用多核 GPU 平行編碼，速度提升數倍。

---

## [v3.2] — 2026-06-20

### 新增功能

- **`ui/step6_eval.py`** — 新增即時多重 Loss 折線圖看板
  - 訓練時即時動態呈現 Total Loss、BCE Loss、Rank Loss 與 InfoNCE Loss 的收斂狀態。
- **`src/finetune.py`** — 引入梯度累積與 AMP 自動混合精度
  - 提升大批量訓練時的顯存使用效率。
- **記憶體防爆優化**
  - 主動在各個階段（如微調結束後）呼叫 `gc.collect()` 與顯存清空，有效防範 OOM 崩潰。

---

## [v3.1] — 2026-06-20

### 新增功能

- **`src/agent.py`** — 實作萬能 LLM 橋接層 (Universal Model Hub)
  - 支援自動檢測硬體環境，若本機無 GPU 資源則自動且優雅地降級切換至雲端 API，相容於 OpenAI 與 Groq 等開放 API 標準。

---

## [v3.0] — 2026-06-19 🏆 競賽版

### 新增功能

#### 🤖 Phase 1：三層 LLM Agent 整合
- **`src/agent.py`** — 新增 `reformulate_query()` 方法
  - 使用 Llama 3.1 8B 將使用者的自然語言問題，自動改寫為精確的學術查詢。
- **`src/agent.py`** — 新增 `generate_hyde_document()` 方法
  - 實作 HyDE（Hypothetical Document Embeddings）技術，提升召回率。
- **`src/agent.py`** — 新增 `explain_result()` 方法
  - 對每個 Top-K 推薦結果，讓 Llama 生成 1~2 句繁體中文推薦理由。

#### 🔍 Hybrid Agentic RAG Pipeline
- 整合 RRF (Reciprocal Rank Fusion) 融合 FAISS 密集與 BM25 稀疏檢索的排名結果。

#### 📊 Phase 3：知識庫視覺化 Dashboard
- **`ui/step1_ingestion.py`** — 新增知識庫統計儀表板與 t-SNE 2D 降維向量分布圖。

#### 🚀 Phase 4：Embedding 模型升級
- 模型更新：`all-MiniLM-L6-v2`（384 維）→ **`BAAI/bge-m3`（1024 維）**。

### Bug 修復

- **`src/embedding_search.py`** — 修復 `rebuild_index_from_pdfs()` 中錯誤的 `AcademicPDFParser` 導入。

