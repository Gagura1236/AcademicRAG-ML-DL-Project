# 📚 AcademicRAG v3.9.4 專案檔案導覽 (Project File Guide)

本文件旨在協助助教與評審委員快速理解 **AcademicRAG v3.9.4** 專案目錄下各個核心檔案與資料夾的作用與架構關係。

---

## 📂 根目錄必要檔案清單與說明

### 1. 🚀 系統入口與配置 (Core & Configuration)
*   **`app.py`**  
    *   **作用**：系統的 Streamlit 網頁應用程式主入口。
    *   **特點**：內建 Apple Silicon MPS 早期導入防護策略，自動加載 `transformers` 全域變數以防堵本機執行 PyTorch 微調時發生的 `Segmentation Fault (139)` 段錯誤。
*   **`config.py`**  
    *   **作用**：全域中心設定檔。
    *   **特點**：自動偵測運行硬體（優先啟用 Apple Metal MPS / NVIDIA CUDA，無硬體加速時優雅降級回 CPU），並定義所有子模組的資料存取路徑。
*   **`requirements.txt`**  
    *   **作用**：專案第三方相依套件清單（包括 PyTorch, FAISS, PyMuPDF, Transformers, PEFT 等）。

### 2. 📖 專案手冊與清單 (Manuals & Documents)
*   **`README.md`**  
    *   **作用**：期末專案完整手冊與助教評分對應說明，內附全新的 **v3.9.4 進階架構 Mermaid 系統圖**。
*   **`CHANGELOG.md`**  
    *   **作用**：版本更新日誌。僅保留期末專案開發階段（v3.0 至 v3.9.4）的演進軌跡。
*   **`DATASET.md`**  
    *   **作用**：介紹本專案預建資料庫收錄的學術領域數據集特點。
*   **`paper_list.md`**  
    *   **作用**：39 篇預建文獻的完整清單，依據動態解析的**出版年份**（Year）由新到舊降序排列，清晰標記 arXiv ID 與領域分類。

### 3. 🛠️ 輔助開發與管理工具 (Scripts & Management)
*   **`download_arxiv_papers.py`**  
    *   **作用**：獨立的 39 篇經典論文 PDF 下載腳本。助教一鍵執行即可自動禮貌下載 100MB+ 的原檔 PDF 至專案目錄中，方便快速測試 Layout OCR 解析與建庫。
*   **`download_and_index.py`**  
    *   **作用**：arXiv 論文下載兼 Ingestion Pipeline 重建整合腳本。
*   **`build_index.py`**  
    *   **作用**：命令列模式的向量庫重建工具，負責掃描 PDF 目錄並重新生成 FAISS 與 BM25 索引。
*   **`package_project.py`**  
    *   **作用**：專案打包自動化工具。自動排除無用的 PDFs/Tex 等百 MB 大檔案與 `dev_records` 開發私密日誌，產生僅 **65.33 MB` 的輕量繳交標準壓縮包。

### 4. 🖱️ 桌面一鍵啟動器 (GUI Launcher)
*   **`launcher_gui.py`**  
    *   **作用**：基於 Python Tkinter 實作的桌面圖形化啟動面板。
*   **`launcher.bat` (Windows)** / **`launcher.command` (macOS)** / **`launcher.sh` (Linux)**  
    *   **作用**：雙擊即可在背景自動建立虛擬環境、啟動伺服器並彈出網頁，實現「免下指令、雙擊開箱即用」的防呆體驗。

### 5. 🎨 視覺與音效資源 (Aesthetics Assets)
*   **`nature_bg.png`**  
    *   **作用**：大自然清新風格的主題背景圖。
*   **`nature_ambient.webm`**  
    *   **作用**：大自然白噪音背景音樂檔，在 UI 側邊欄供使用者自由播放以提升學術閱讀專注力。

---

## 📁 核心架構資料夾 (Directories)

*   **`src/` (核心演算法模組)**  
    *   `agent.py`：ReAct 思考推理環與多代理人對抗辯論（Generator vs Critic）。
    *   `finetune.py`：AdaLoRA 參數微調訓練器，包含混合硬負樣本挖掘與 Auto-HPO 超參數 Grid Search。
    *   `evaluation.py`：本機 15 組 Ground-Truth 評估引擎。
    *   `retrieval_engine.py` / `embedding_search.py`：動態 Alpha 融合檢索與 RRF。
    *   `metadata_extractor.py`：雙重驗證學術標題與 arXiv 年份動態解析提取。
    *   `pdf_parser.py` / `nougat_client.py`：Layout OCR 結構化解析與 LaTeX 公式提取。
    *   `community_manager.py` / `openalex_client.py`：GraphRAG Leiden 社群摘要與 OpenAlex 前後引文推薦。
    *   `preference_store.py`：收集使用者評分 👍/👎 回饋的 DPO 偏好存儲器。
*   **`ui/` (Streamlit 分步工作流)**  
    *   `step1_ingestion.py` 至 `step6_eval.py`：對應系統運作的六大步驟互動頁面。
*   **`lib/` (前端視覺化庫)**  
    *   包含知識圖譜可視化（VisJS）與下拉多選（TomSelect）等本地依賴腳本，確保完全斷網時仍能流暢渲染網頁。
*   **`tests/` (自動化回歸測試)**  
    *   包含基於 Playwright 的自動化測試腳本，用於確保 UI 各按鈕與路由的健全。
*   **`data/` (資料庫存儲層)**  
    *   儲存預建的 `default` FAISS 密集向量索引、BM25 字典檔、Leiden 社群摘要 JSON 以及使用者偏好 log。
