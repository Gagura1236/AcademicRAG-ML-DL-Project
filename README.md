# 📚 AcademicRAG v3.9.4 — 智慧學術論文推薦系統

> **成功大學 114-2 機器學習與深度學習 期末專案**  
> 學號：RA7141236 ｜ 授課教授：游濟華

---

## 🌟 系統簡介

**AcademicRAG** 是一套結合 **深度學習語意搜尋** 與 **自主思考 LLM Agent (ReAct Loop)** 的學術論文智慧推薦與問答平台。系統以 Streamlit 建構互動式 6 步驟 Workflow，讓使用者能從頂尖 AI 論文知識庫（共收錄 39 篇經典文獻，完整列表請參閱 [paper_list.md](file:///Users/gagura/NCKU/AI相關課程/成大AI課程/114-2%20ML%20&%20DL/Final%20Project/v3.9.4/paper_list.md)）中，透過自然語言問答，精準找到所需文獻，並由 LLM 協作、連網擴展搜尋，生成高品質可解釋性答案。

系統針對 Apple Silicon (M-series) 晶片與 MPS 加速環境進行了極致的記憶體管理與穩定性優化，成功解決了微調過程中隨機崩潰的 **Segmentation Fault (139)** 段錯誤，實現了高效穩定的本機 LoRA 訓練與雙卡 GPU 平行加速。

### 💡 觀念防呆：為什麼需要這麼多模型？它們如何分工？

許多人剛接觸 RAG 時會疑惑：「既然已經有搜尋模型，為什麼還要設置大語言模型 (LLM)？為什麼解析、搜尋和微調的模型全都不一樣？」
本系統採用了**「術業有專攻」**的模組化流水線設計，各模型的職責如下：

1.  **為什麼有了搜尋模型，還要用 LLM？**
    *   🔍 **搜尋模型 (Embedding & Rerank)**：像是一個**「圖書館的管理員」**。它的工作是「找書」，能在幾毫秒內從數萬個段落中精準找出與您的問題最相關的 3 段話。但**它不會寫字、無法推論、不能直接回答問題**。
    *   🧠 **大語言模型 (LLM - 如 Llama 或 GPT)**：則像是一個**「學術大師」**。它會閱讀搜尋模型幫它找出來的這 3 段話，將其融會貫通後，用流暢、符合邏輯的人類語言回答您的問題。

2.  **為什麼「解析、粗篩、精篩（微調）、寫作」要用不同的模型？**
    *   👁️ **解析模型 (Nougat OCR + Qwen2-VL VLM - 負責「眼明」)**：專責將 PDF 中的複雜數學公式翻譯成 LaTeX 程式碼，並將論文圖表解讀成詳細的英文文字描述。
    *   🎣 **檢索粗篩模型 (BAAI/bge-m3 Embedding - 負責「耳聰」)**：負責**快速篩選**。將全體論文段落向量化，快速過濾掉 99% 無關內容，挑出前 10 名最像的候選段落。
    *   ⚖️ **微調重排模型 (Cross-Encoder + LoRA - 負責「心細」)**：負責**精準篩選**。它能同時閱讀「問題」與「候選段落」進行極為細緻的對比，排出最精確的第一名。**我們在 Step 6 微調的就是這個模型**，透過 LoRA 讓它在 1 秒內載入，完美看懂您專屬領域（如會計、AI 等）的特殊名詞與公式！
    *   ✍️ **寫作推理模型 (LLM - 負責「筆健」)**：負責將精篩出的段落綜合寫成最終的學術解答。

---

## 🏗️ 系統架構

```mermaid
graph TD
    classDef agent fill:#FFECEC,stroke:#FF9999,stroke-width:2px;
    classDef search fill:#E8F0FE,stroke:#4285F4,stroke-width:2px;
    classDef data fill:#E6F4EA,stroke:#137333,stroke-width:2px;
    classDef user fill:#FEF7E0,stroke:#B06000,stroke-width:2px;
    classDef train fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px;

    U[👤 使用者學術查詢 zh/en]:::user

    subgraph 🧠 Agentic AI 推理與決策層 (ReAct & Debate)
        A1[Query Expansion 查詢擴展 & HyDE]:::agent
        A2[Thought: 推理與自我診斷]:::agent
        A3{Action: 執行檢索工具}:::agent
        A4{CRAG 判定: 檢索信心度 > 0.35?}:::agent
        A5[search_arxiv: ArXiv 聯網搜尋工具]:::agent
        A6[Multi-Agent Debate: Generator/Critic 雙體辯論]:::agent
        
        U --> A1
        A1 --> A2
        A2 --> A3
        A3 --> A4
        A4 -->|No| A5
        A5 --> A2
        A4 -->|Yes| S3
        A6 --> O[📄 最終學術回答 & Hover 引文定位]:::user
    end

    subgraph 🔍 Federated Hybrid Search 跨專案聯邦檢索 (BGE-M3)
        S1[Dense Search: FAISS 密集向量 + Dynamic Alpha]:::search
        S2[Sparse Search: BM25 關鍵字]:::search
        S3[RRF: 多路檢索倒數排名融合]:::search
        S4[LoRA Reranker: Cross-Encoder + AdaLoRA 適配]:::search
        
        A3 -->|跨專案| S1
        A3 -->|跨專案| S2
        S1 --> S3
        S2 --> S3
        S3 --> S4
        S4 --> A6
    end

    subgraph 🗄️ GraphRAG 階層知識庫與切片 (Layout OCR)
        D1[Ingestion: 多源文獻 PDF / Tex 原始碼]:::data
        D2[OCR Parser: PyMuPDF + Nougat LaTeX 公式提取]:::data
        D3[Semantic Chunking: 512+128 語意重疊切片]:::data
        D4[GraphRAG: PageRank 重點實體 + Community Map-Reduce]:::data
        
        D1 --> D2
        D2 --> D3
        D3 --> S1
        D3 --> S2
        D3 --> D4
        D4 -.-> A6
    end

    subgraph 🎓 AutoML 自我監督微調閉環 (LoRA Tuner)
        T1[Preferences Store: DPO/RLHF 人類回饋儲存]:::train
        T2[Hard Negative Miner: 假陽性硬負樣本挖掘]:::train
        T3[Auto-HPO: 學習率 & Margin 超參數 Grid Search]:::train
        T4[AdaLoRA Fine-tuning: 奇異值分解(SVD)動態權重訓練]:::train
        
        O -.->|👍/👎 回饋| T1
        T1 --> T2
        T2 --> T3
        T3 --> T4
        T4 -.->|更新 LoRA 權重| S4
    end
```

---

## 🎯 核心技術亮點（v3.9.4）

### 1. 🤖 自主思考 ReAct Agent 與連網外掛 (ArXiv Tool)
- **ReAct 推理迴圈**：LLM 在作答前，必須強制輸出 `Thought:`，自我思考並決定下一步行動。
- **連續對話記憶 (Conversational UX)**：Agent 能夠記住使用者的歷史問答上下文，支援流暢的多輪追問與主題深入探討。
- **ArXiv 連網搜尋工具**：若使用者問題超出本地知識庫範圍，Agent 會自動呼叫 `search_arxiv` 連網抓取最新論文摘要，打破資訊孤島。
- **即時 Thought Trace UI**：以 `st.status` 摺疊選單即時動態展示 Agent 的「思考旅程」（Thought ➔ Action ➔ Observation ➔ Final Answer）。

### 2. ⚔️ 多代理人對抗辯論機制 (Multi-Agent Debate)
- 引入雙智能體辯論，由 Generator Agent 與 Critic Agent 進行多輪論證與自我反思，過濾掉幻覺與無效答案。

### 3. 🛡️ CRAG (Corrective RAG) 與來源透明化
- **防幻覺門檻過濾**：設定嚴格的 `0.35` 檢索門檻，並具備前端黃色警告 Indicator，當本地庫缺乏相關知識時自動阻斷幻覺生成。
- **行內引用 (Inline Citations) 與 Hover Tooltips**：強制 Agent 使用 `[Doc X]` 標記來源，並在前端動態轉換為 Hover 氣泡提示，游標懸停即可檢視原論文段落、頁碼與標題，確保學術嚴謹度。

### 4. 🔍 Hybrid Agentic RAG Pipeline
- 整合 Query Reformulation + HyDE（假設性文件向量生成），將短查詢自動改寫為多個學術查詢。
- 使用 **FAISS 密集向量檢索** 與 **BM25 稀疏檢索** 進行多路檢索，經由 **RRF (Reciprocal Rank Fusion)** 進行排序融合。
- 採用 **Cross-Encoder 重排模型** 進行二次精排，並利用完美的數學映射 `(sim + 1.0)/2.0` 將余弦相似度無損投射至 UI。

### 5. 🛠️ Apple Silicon MPS 記憶體與 SIGSEGV 崩潰防禦
- **Early Import Guard**：將 `transformers` 調度器移至最頂層導入，避免 PyTorch 在 MPS 執行時的生命週期初始化衝突。
- **Lazy PEFT Instantiation**：PEFT 模組移至微調函數內部延遲載入，防止類別修飾器與基礎 Cross-Encoder 衝突。
- **訓練/推理混合架構**：微調訓練自動回退至 CPU（使用 FP32 確保梯度數值穩定），本機語意檢索與 Rerank 推理則維持使用 GPU/MPS 半精度加速。

### 6. 🎯 零標註本機專屬領域客製化 (Self-Supervised Domain Adaptation)
- **隨插即用自訂領域**：使用者只需上傳自己領域的論文（如金融、會計、生醫等），系統會自動呼叫本機 LLM 對上傳的論文段落進行「自我監督問答對生成 (Self-Supervised QA Generation)」，無需任何人工作業。
- **極速動態 LoRA 切換**：在 Step 6 點擊微調後，系統會自動在背景進行 LoRA 適配器微調。LoRA 權重極輕（僅約幾十 MB），載入與切換可在不到 1 秒內完成，讓使用者能自由切換不同的學術研究主軸，在多個適配領域間自如游走，且不會造成基礎模型的「災難性遺忘」。

### 7. 🌐 多專案工作區與聯邦檢索 (Workspaces & Federated Search)
- **多專案隔離 (Workspace Management)**：支援在 UI 側邊欄即時建立與切換「專案」，將不同領域的知識庫完全隔離，徹底解決專案間資料相互干擾的問題。
- **跨專案聯邦檢索 (Cross-Project Search)**：若您需要進行跨領域的研究，只需在側邊欄勾選目標專案，檢索引擎便會動態掛載多個 FAISS 向量庫並行搜尋，並將結果合併至 Cross-Encoder 進行「全域重排 (Global Reranking)」。系統會在推薦結果旁標註來源專案徽章，讓跨領域知識調閱一目了然！

### 8. 🚀 Academic LoRA Tuner v3.9.4 演算法深度優化 (AutoML & Advanced PEFT)
- **🧬 AdaLoRA (Adaptive Low-Rank Adaptation)**：取代傳統靜態 LoRA，根據矩陣權重的重要度，動態調整並分配秩（Rank）預算。利用 SVD 奇異值分解剪枝低貢獻參數，在不增加顯存的前提下，極大化複雜 LaTeX 公式與學術術語的語意學習精準度。
- **🧲 混合硬負樣本挖掘 (Hybrid Dense & Sparse Negative Mining)**：結合 RAG 的 FAISS 稠密檢索與 BM25 稀疏檢索結果，將排名前列的「假陽性 (False Positives)」自動挖掘為最具挑戰性的硬負樣本，強迫 Cross-Encoder 在特徵邊界上學會精細語意辨識。
- **📈 一鍵 Auto-HPO 超參數自動調優**：內建自動化參數調優引擎，系統在微調前會先以 Grid Search 自動測試多種 Learning Rate 與 Contrastive Margin 的排列組合，快速執行一輪驗證評估，挑選出 Validation AUC 最高的最優參數，隨後自動套用進行正式訓練，並以 pandas table 展示 HPO 探索軌跡。
- **🌀 餘弦退火排程器 (Cosine Scheduler with Warmup)**：引入 `get_cosine_schedule_with_warmup` 學習率排程器，使模型在前期快速收斂、後期平滑減速，以獲得更佳的局部最優解，避免損失震盪。

---

## 📋 Streamlit 六步驟 Workflow

| 步驟 | 功能 | 說明 |
|------|------|------|
| **Step 1** | 📥 文件處理與語意切片 | 支持 arXiv API 下載與 PDF 上傳。利用 PyMuPDF 與 Nougat 提取 LaTeX 公式，一鍵生成 **t-SNE 2D 向量分布散點圖**。 |
| **Step 2** | 🕸️ 知識圖譜與社群 | 利用 TF-IDF 與共現矩陣自動提取學術實體。基於 PageRank 評估節點重要性，並整合 GraphRAG 的階層式社群 Map-Reduce 摘要。 |
| **Step 3** | 🎛️ 動態權重與檢索設定 | 結合 FAISS 密集與 BM25 稀疏檢索，支持動態權重 (Dynamic Alpha) 自定義與語意搜尋。 |
| **Step 4** | 🔎 多跳檢索與自我反思 | 完整 Agent 工作流介面：多輪對話記憶、CRAG 狀態看板、Hover Tooltips 行內引用與 ArXiv 聯網外掛。 |
| **Step 5** | 📚 多樣性排序與推薦 | 基於 K-Means 的學術聚類可視化，並運用 MMR (最大邊際相關性) 機制進行上下游引用關係鏈推薦。 |
| **Step 6** | 📈 量化評估與微調回饋 | 整合 AdaLoRA 適配與混合負樣本挖掘，支持一鍵 Auto-HPO 自動超參數優化，並顯示即時 HPO 搜尋歷史與 Cosine 退火訓練曲線。 |

---

## 🚀 快速啟動

### 環境需求 & LLM 模型推薦

#### 1. 🖥️ 硬體與軟體需求
- **Python**: 3.10+
- **作業系統**: macOS (支援 Apple Silicon MPS 加速) 或 Windows/Linux (支援 CUDA)
- **記憶體 (RAM)**: 建議 16 GB+
- **儲存空間**: 建議空閒 10 GB+ (供下載/快取本地 Embedding, Rerank 與 VLM 模型)

#### 2. 🤖 大語言模型 (LLM) 推薦
本系統的 Agentic AI 推理與學術圖譜挖掘需要 LLM 驅動。您可以在 Streamlit 側邊欄的 **「LLM 萬能切換中心」** 自由切換您喜愛的任何本機或雲端模型。推薦機型如下：

*   **🍏 Apple Silicon macOS 本機高速推論 (MLX)**
    *   `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (預設，小巧流暢，英文理解強)
    *   `mlx-community/Qwen2.5-7B-Instruct-4bit` (極力推薦！中英雙語學術能力、程式與 LaTeX 公式解讀極佳)
*   **☁️ 雲端 API 模式 (Cloud API - 速度最快，精準度最高)**
    *   `gpt-4o-mini` (兼顧速度與經濟實惠)
    *   `gemini-2.5-flash` / `gemini-2.5-pro` (針對長文本、複雜科學公式與圖表解讀能力表現頂尖)
    *   `deepseek-chat` (DeepSeek-V3 / R1 推理模型，具備超高性價比與深度思考推理能力)
*   **🏠 通用本機 API 模式 (透過 Ollama 或 LM Studio 串接)**
    *   `qwen2.5:7b-instruct` 或 `llama3.1:8b-instruct` (預設為 `http://localhost:11434/v1`)

### 安裝與運行

您可以選擇使用終端機指令或**跨平台 GUI 一鍵啟動器**來啟動系統。

#### 方法一：使用 GUI 一鍵啟動器（推薦）
為了提供更友善的使用體驗，我們開發了跨平台的桌面 GUI 啟動器，雙擊即可開啟綠色大自然風格的控制面板，並能一鍵啟動伺服器與自動打開網頁：
- `launcher.bat`：專屬 **Windows** 的一鍵啟動捷徑（隱藏終端機黑背景視窗）。
- `launcher.command`：專屬 **macOS** 的一鍵啟動腳本（在 Finder 雙擊即可執行）。
- `launcher.sh`：專屬 **Linux** 的一鍵啟動腳本。
- `launcher_gui.py`：啟動器的核心 Python 程式，使用 tkinter 實作，負責監控 Streamlit 背景伺服器狀態與即時日誌顯示。

*(註：首次使用前，請先透過下方方法二的步驟 1~3 安裝所需套件，再使用啟動器。)*

#### 方法二：使用傳統終端機指令
```bash
# 1. 進入專案目錄
cd "Final Project/v3.9.4"

# 2. 建立並啟動虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安裝相依套件
pip install -r requirements.txt

# 4. 啟動 Streamlit 應用程式
streamlit run app.py
```

#### 💡 補充說明：如何一次性自動下載這 39 份論文 (給助教評分或新手測試用)
本專案已包含自動下載這 39 份經典學術論文的獨立腳本。助教或使用者在啟動虛擬環境並安裝 requirements 後，只需在專案目錄下執行：
```bash
python download_arxiv_papers.py
```
本腳本會遵守 arXiv API 的禮貌原則，自動將 39 份論文的 PDF 原始檔下載到專案對應的 `data/projects/default/pdfs/` 目錄中，方便執行 Ingestion 與 OCR 測試。完整收錄文獻列表請參閱 [paper_list.md](file:///Users/gagura/NCKU/AI相關課程/成大AI課程/114-2%20ML%20&%20DL/Final%20Project/v3.9.4/paper_list.md)。

---

## 📁 專案結構
```
v3.9.4/
├── app.py                    # 主程式入口（Streamlit 路由與早期導入）
├── config.py                 # 全域設定（硬體自適應、路徑）
├── requirements.txt          # 相依套件清單
├── README.md                 # 本文件
├── CHANGELOG.md              # 版本更新紀錄
├── paper_list.md             # 📚 本版本收錄的 39 篇經典論文文獻清單
│
├── src/                      # 核心演算法模組
│   ├── agent.py              # 🤖 LLM Agent (ReAct, Debate 整合)
│   ├── embedding_search.py   # 🔍 Hybrid RAG 搜尋引擎（FAISS + BM25）
│   ├── retrieval_engine.py   # 📑 端到端檢索流程協調器
│   ├── finetune.py           # 🎓 LoRA 微調訓練器 (Lazy PEFT, MPS/CPU Fallback)
│   ├── evaluation.py         # 📊 系統診斷模組 (ROC-AUC)
│   ├── pdf_parser.py         # 📄 PDF 解析與公式提取
│   ├── paper_analyzer.py     # 🔬 論文深度分析
│   ├── community_manager.py  # 🕸️ GraphRAG 社群偵測
│   ├── data_collector.py     # 📥 arXiv API 下載器
│   ├── dynamic_alpha.py      # ⚖️ 動態 Hybrid 權重調整
│   ├── metadata_extractor.py # 🏷️ 論文元數據提取
│   ├── openalex_client.py    # 🌐 OpenAlex API
│   ├── ragas_evaluator.py    # 🎯 RAGAS 評分器
│   └── utils.py              # 🔧 共用工具
│
└── ui/                       # Streamlit UI 頁面
    ├── step1_ingestion.py    # 論文導入與 t-SNE
    ├── step2_graphrag.py     # 知識圖譜與社群摘要
    ├── step3_alpha.py        # 混合檢索設定
    ├── step4_agentic.py      # Agentic 對話與 Thought Trace
    ├── step5_mmr.py          # K-Means 聚類與上下游引用推薦
    └── step6_eval.py         # 微調看板與評估雷達圖
```

---

## 🏆 課程評分對應說明

| 評分項目 | 權重 | 系統對應功能 |
|---------|------|------------|
| LLM / LLM Agent 整合 | 25% | **v3.9.4 ReAct 推理 + CRAG 評估 + 多輪對話記憶 + Hover Tooltips 行內引用 + Multi-Agent Debate** |
| 最佳推薦與結果解釋 | 15% | Rerank 得分 + FAISS 語意距離 + BM25 匹配度 + LLM 自然語言推薦解釋卡片 |
| Machine Learning / AI 方法 | 15% | bge-m3 + FAISS (ANN) + BM25 + Cross-Encoder + LoRA + t-SNE + K-Means |
| GUI 系統與使用者互動 | 15% | Streamlit 6 步驟 Workflow + t-SNE 散點圖 + 互動引文圖譜 + 即時 Thought Trace 看板 |
| Database / Dataset 建構 | 15% | 39 篇經典文獻 + v2.2 語意切片 (防止句中/公式中斷) + OpenAlex 引文網路 |
| 問題理解與系統設計 | 10% | 完整的 Data Ingestion ➔ GraphRAG ➔ Hybrid Retrieval ➔ Agent ➔ Explanation ➔ LoRA Fine-Tuning 閉環架構 |
| 程式完整性 | 5% | 模組化程式碼，解決 MPS 驅動崩潰問題，各步驟健全容錯並能完美運行 |
