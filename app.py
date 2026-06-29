import streamlit as st
import os
import sys
import config

# ==========================================
# 0. 系統初始化與基本設定
# ==========================================
st.set_page_config(
    page_title="AcademicRAG - AI 論文閱讀助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 套用全域自訂 CSS 以美化介面
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
local_css(os.path.join(config.BASE_DIR, "ui", "index.css"))

@st.cache_resource(show_spinner=False)
def get_rag_engine(project_name: str, collection_name: str = "main"):
    from src.embedding_search import AcademicRAGEngine
    return AcademicRAGEngine(project_name=project_name, collection_name=collection_name)

@st.cache_resource(show_spinner=False)
def get_parser():
    from src.pdf_parser import ScientificPDFParser
    return ScientificPDFParser()

@st.cache_resource(show_spinner=False)
def get_data_collector_fns():
    from src.data_collector import download_arxiv_papers, download_single_pdf
    return download_arxiv_papers, download_single_pdf

def clear_all_caches_and_models():
    st.cache_resource.clear()
    try:
        from src.model_manager import ModelManager
        ModelManager().release_all_models()
    except Exception as e:
        print(f"[System] 釋放模型時出錯: {e}")

from src.project_utils import get_all_projects, get_project_paths, get_collections

# Language state setup
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# Project & Collection state setup
if "project_name" not in st.session_state:
    st.session_state.project_name = "default"
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "main"

project_paths = get_project_paths(st.session_state.project_name, st.session_state.collection_name)
index_exists = os.path.exists(os.path.join(project_paths["vector_db_dir"], "faiss.index"))

if "engine_loaded" not in st.session_state:
    st.session_state.engine_loaded = False

if not st.session_state.engine_loaded:
    if not index_exists:
        spinner_msg = "⏳ 第一次開啟：需要約 25-30 秒載入模型...\n\nFirst time loading: takes approx. 25-30s to load models..."
        with st.spinner(spinner_msg):
            rag_engine = get_rag_engine(st.session_state.project_name, st.session_state.collection_name)
            st.session_state.engine_loaded = True
    else:
        spinner_msg = "⏳ 系統加載中... (System loading...)"
        with st.spinner(spinner_msg):
            rag_engine = get_rag_engine(st.session_state.project_name, st.session_state.collection_name)
            st.session_state.engine_loaded = True
else:
    rag_engine = get_rag_engine(st.session_state.project_name, st.session_state.collection_name)

parser = get_parser()
download_arxiv_papers, download_single_pdf = get_data_collector_fns()

# 初始化 session state 變數用以儲存診斷與微調結果
if 'diagnostic_report' not in st.session_state:
    st.session_state.diagnostic_report = None
if 'before_report' not in st.session_state:
    st.session_state.before_report = None



def render_latex(latex_str):
    import urllib.parse
    encoded = urllib.parse.quote(latex_str)
    url = f"https://latex.codecogs.com/svg.image?{encoded}"
    st.image(url)

# Language state setup
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

from src.locales import get_text

with st.sidebar:
    # Language selector at top
    lang_opt = st.selectbox(
        "🌐 Language",
        options=["繁體中文 (Traditional Chinese)", "English"],
        index=0 if st.session_state.lang == "zh" else 1,
        key="lang_select_top"
    )
    new_lang = "zh" if "繁體中文" in lang_opt else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.markdown("---")
    # 專案切換器
    st.subheader("📂 專案切換 (Project Workspace)")
    projects = get_all_projects()
    
    def clear_project_state():
        keys_to_clear = [
            "_search_sequence", "active_conv", "before_report", "conversations",
            "diagnostic_report", "final_query_input", "last_agent_chunks",
            "last_engine_answer", "last_engine_chunks", "last_engine_query",
            "last_sub_queries", "original_chat", "rec_interest", "rec_is_llm",
            "rec_results", "search_results", "target_projects", "translated_chat",
            "translated_interest", "tree_concept", "tree_results"
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
    
    selected_proj = st.selectbox("當前專案 (Current Project):", projects, index=projects.index(st.session_state.project_name) if st.session_state.project_name in projects else 0)
    
    if selected_proj != st.session_state.project_name:
        st.session_state.project_name = selected_proj
        st.session_state.engine_loaded = False  # 強制重新載入 RAG Engine
        clear_all_caches_and_models()
        clear_project_state()
        st.rerun()
        
    new_proj = st.text_input("➕ 新建專案 (Create New Project)", placeholder="輸入新專案名稱...")
    if st.button("創建專案 (Create)", key="create_proj_btn"):
        if new_proj.strip() and new_proj.strip() not in projects:
            st.session_state.project_name = new_proj.strip()
            st.session_state.engine_loaded = False
            clear_all_caches_and_models()
            clear_project_state()
            st.rerun()
        elif new_proj.strip() in projects:
            st.warning("專案已存在！")
    
    st.markdown("---")
    # 集合切換器
    st.subheader("📚 向量集合 (Collection)")
    collections = get_collections(st.session_state.project_name)
    
    selected_col = st.selectbox("當前向量庫 (Current Collection):", collections, index=collections.index(st.session_state.collection_name) if st.session_state.collection_name in collections else 0)
    
    if selected_col != st.session_state.collection_name:
        st.session_state.collection_name = selected_col
        st.session_state.engine_loaded = False
        clear_all_caches_and_models()
        st.rerun()
        
    new_col = st.text_input("➕ 新建向量庫 (New Collection)", placeholder="限英數字如 test_db...")
    if st.button("創建集合 (Create Collection)", key="create_col_btn"):
        import re
        if new_col.strip() and new_col.strip() not in collections:
            # 限制只能是英數字與底線減號
            if re.match(r'^[a-zA-Z0-9_\-]+$', new_col.strip()):
                st.session_state.collection_name = new_col.strip()
                st.session_state.engine_loaded = False
                clear_all_caches_and_models()
                st.rerun()
            else:
                st.error("命名只能包含英文字母、數字、底線(_)或減號(-)，請重新輸入！")
        elif new_col.strip() in collections:
            st.warning("集合已存在！")
    
    st.markdown("---")
    # 跨專案聯邦檢索設定
    st.subheader("🌐 跨專案調閱 (Federated Search)")
    if "target_projects" not in st.session_state:
        st.session_state.target_projects = [st.session_state.project_name]
    
    selected_targets = st.multiselect(
        "選擇要合併搜尋的專案:",
        options=projects,
        default=st.session_state.target_projects,
        help="勾選多個專案後，在「語意檢索」時將會自動調閱所有勾選的向量庫，並合併排名。"
    )
    if selected_targets != st.session_state.target_projects:
        st.session_state.target_projects = selected_targets
        st.rerun()

    st.markdown("---")

    st.image("https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg", width=60)
    st.title("AcademicRAG")
    st.caption("v3.9.4 Nature Theme")
    
    menu_options = [
        get_text("menu_step1", st.session_state.lang),
        get_text("menu_step2", st.session_state.lang),
        get_text("menu_step3", st.session_state.lang),
        get_text("menu_step4", st.session_state.lang),
        get_text("menu_step5", st.session_state.lang),
        get_text("menu_step6", st.session_state.lang),
    ]
    
    nav_index = st.session_state.get("nav_index", 0)
    current_page_label = st.radio(
        "📍 " + ("工作流程" if st.session_state.lang == "zh" else "Workflow") + ":",
        menu_options,
        index=nav_index,
        key="nav_radio_temp"
    )
    st.session_state.nav_index = menu_options.index(current_page_label)
    
    st.write("---")
    
    with st.expander("📊 知識庫系統狀態 (System Stats)", expanded=True):
        st.metric("索引向量維度", "384 (MiniLM-L6)")
        n_chunks = len(rag_engine.chunks_metadata)
        paper_ids = list(set(c.get("paper_id", "?") for c in rag_engine.chunks_metadata))
        st.metric("總 Chunk 數", f"{n_chunks:,}")
        st.metric("收錄文獻數", len(paper_ids))
        
        if n_chunks > 0:
            st.success("FAISS 索引已掛載 (Active)")
        else:
            st.warning("FAISS 索引為空 (Empty)")
            
        st.write("---")
        st.write("🤖 **LLM 萬能切換中心 (Model Hub)**")
        
        import platform
        sys_os = platform.system()
        
        lang = st.session_state.get('lang', 'zh')
        
        if lang == "zh":
            local_model_desc = "選擇 `Apple MLX (本地模型)`，直接用您的 Mac 晶片運算，" if sys_os == "Darwin" else "選擇 `本地模型`，直接用您的本機 GPU/CPU 運算，"
            expander_title = "💡 這個 LLM 是用來幹嘛的？"
            expander_text = f"""
            這個 LLM (大語言模型) 是您在閱讀與問答功能中，負責「開口說話」的大腦。
            
            **簡單來說：**
            1. **系統底層 (檢索模型)** 負責在幾千頁的論文中找出最相關的兩三段原文。
            2. **這個 LLM** 負責讀完這兩三段原文後，**用流暢的人類語言（中文/英文）總結並回答您的問題**。

            **如果沒有設定 LLM：**
            系統就只會死板地把找出來的英文段落直接貼給您看，無法為您做總結、翻譯、或推論分析。
            
            **為什麼要特別設計這個切換中心？**
            因為您可以自由決定這顆大腦要用誰：
            * **免費且具備極致隱私：** {local_model_desc}連線完全不用網路，機密論文不會外洩。
            * **最強大的推理能力：** 選擇 `OpenAI API (GPT-4o)` 或 `Anthropic (Claude 3.5)`，讓最頂尖的 AI 來幫您分析論文。
            * **自建伺服器：** 選擇 `Ollama / LM Studio`，連線到實驗室的工作站。
            """
        else:
            local_model_desc = "Select `Apple MLX (Local)` to use your Mac Apple Silicon, " if sys_os == "Darwin" else "Select `Local Model` to use your local GPU/CPU, "
            expander_title = "💡 What is this LLM used for?"
            expander_text = f"""
            This LLM (Large Language Model) is the "brain" responsible for speaking and answering in the reading and Q&A features.
            
            **In short:**
            1. **The underlying system (Retrieval Model)** finds the 2-3 most relevant paragraphs across thousands of pages.
            2. **This LLM** reads those paragraphs and **summarizes/answers your questions in fluent human language (English/Chinese)**.

            **If no LLM is configured:**
            The system will only rigidly paste the retrieved English paragraphs. It won't be able to summarize, translate, or infer.
            
            **Why is this Model Hub needed?**
            Because you have the freedom to choose your brain:
            * **Free & Extreme Privacy:** {local_model_desc}runs entirely offline without network, keeping your papers strictly confidential.
            * **Maximum Reasoning Power:** Select `OpenAI API (GPT-4o)` or `Anthropic (Claude 3.5)` for state-of-the-art analysis.
            * **Self-hosted Server:** Select `Ollama / LM Studio` to connect to your lab's workstation.
            """

        with st.expander(expander_title, expanded=False):
            st.markdown(expander_text)
        
        if 'llm_provider' not in st.session_state:
            st.session_state.llm_provider = config.LLM_PROVIDER
        if 'llm_api_key' not in st.session_state:
            st.session_state.llm_api_key = config.LLM_API_KEY
        if 'llm_api_base_url' not in st.session_state:
            st.session_state.llm_api_base_url = config.LLM_API_BASE_URL
        if 'llm_api_model_name' not in st.session_state:
            st.session_state.llm_api_model_name = config.LLM_API_MODEL_NAME
            
        provider_options = ["Auto (硬體偵測)", "MLX (Apple 本地高速)", "Cloud API (開放 API 標準)"]
        current_idx = 0
        if st.session_state.llm_provider == "MLX": current_idx = 1
        elif st.session_state.llm_provider == "API": current_idx = 2
        
        new_provider_str = st.selectbox("LLM 運算模式 (Provider)", options=provider_options, index=current_idx)
        
        new_provider = "Auto"
        if "MLX" in new_provider_str: new_provider = "MLX"
        elif "Cloud" in new_provider_str: new_provider = "API"

        if new_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = new_provider
            config.LLM_PROVIDER = new_provider
            config.save_llm_config(
                new_provider, 
                st.session_state.llm_api_key, 
                st.session_state.llm_api_base_url, 
                st.session_state.llm_api_model_name, 
                st.session_state.get('llm_model_path', config.LLM_MODEL_PATH)
            )
            st.rerun()

        # 如果選擇本地推論，顯示模型路徑
        if new_provider in ["Auto", "MLX"]:
            if 'llm_model_path' not in st.session_state:
                st.session_state.llm_model_path = config.LLM_MODEL_PATH
                
            new_llm_path = st.text_input(
                "MLX 模型路徑 (HuggingFace Repo)", 
                value=st.session_state.llm_model_path,
                help="例如輸入 mlx-community/Qwen2.5-7B-Instruct-4bit"
            )
            
            if new_llm_path != st.session_state.llm_model_path:
                st.session_state.llm_model_path = new_llm_path
                config.LLM_MODEL_PATH = new_llm_path
                config.save_llm_config(
                    st.session_state.llm_provider, 
                    st.session_state.llm_api_key, 
                    st.session_state.llm_api_base_url, 
                    st.session_state.llm_api_model_name, 
                    new_llm_path
                )
                st.rerun()
                
        # 如果選擇 API 推論，顯示 API 細節設定
        if new_provider in ["Auto", "API"]:
            new_api_base_url = st.text_input(
                "API Base URL", 
                value=st.session_state.llm_api_base_url,
                help="例如 OpenAI, Groq, Ollama (http://localhost:11434/v1), 或 LM Studio"
            )
            if new_api_base_url != st.session_state.llm_api_base_url:
                st.session_state.llm_api_base_url = new_api_base_url
                config.LLM_API_BASE_URL = new_api_base_url
                config.save_llm_config(
                    st.session_state.llm_provider, 
                    st.session_state.llm_api_key, 
                    new_api_base_url, 
                    st.session_state.llm_api_model_name, 
                    st.session_state.get('llm_model_path', config.LLM_MODEL_PATH)
                )
                st.rerun()

            new_api_model_name = st.text_input(
                "API 模型名稱", 
                value=st.session_state.llm_api_model_name,
                help="例如 gpt-4o-mini, llama-3.1-8b-instant"
            )
            if new_api_model_name != st.session_state.llm_api_model_name:
                st.session_state.llm_api_model_name = new_api_model_name
                config.LLM_API_MODEL_NAME = new_api_model_name
                config.save_llm_config(
                    st.session_state.llm_provider, 
                    st.session_state.llm_api_key, 
                    st.session_state.llm_api_base_url, 
                    new_api_model_name, 
                    st.session_state.get('llm_model_path', config.LLM_MODEL_PATH)
                )
                st.rerun()

            new_api_key = st.text_input(
                "API Key (本機伺服器可留空)", 
                value=st.session_state.llm_api_key, 
                type="password"
            )
            if new_api_key != st.session_state.llm_api_key:
                st.session_state.llm_api_key = new_api_key
                config.LLM_API_KEY = new_api_key
                config.save_llm_config(
                    st.session_state.llm_provider, 
                    new_api_key, 
                    st.session_state.llm_api_base_url, 
                    st.session_state.llm_api_model_name, 
                    st.session_state.get('llm_model_path', config.LLM_MODEL_PATH)
                )
                st.rerun()
                
        # 測試連線按鈕
        if st.button("🔄 測試 LLM 連線狀態 (並強制重整快取)"):
            st.cache_resource.clear()
            with st.spinner("正在重新載入模型與測試 LLM 橋接層連線..."):
                try:
                    from src.agent import AcademicAgent
                    agent = AcademicAgent()
                    
                    # 顯示本地模型加載錯誤詳情 (如果有)
                    load_err = getattr(agent.llm_provider, "load_error", None)
                    if load_err:
                        st.warning(f"⚠️ 本地 MLX 模型加載失敗：{load_err}")
                        
                    res = agent.generate("Please reply with precisely 'Connection OK'.", max_tokens=10)
                    if "失敗" in res or "未加載" in res or "Error" in res:
                        st.error(f"連線失敗 / Connection Failed: {res}")
                    else:
                        st.success(f"連線成功！回應: {res}")
                except Exception as e:
                    st.error(f"連線發生例外錯誤: {e}")
            
        st.write("---")
        st.write("🔌 **LoRA 微調權重管理**")
        
        if lang == "zh":
            lora_title = "💡 怎麼新增 Adapter？"
            lora_text = """
            **有兩種方式可以新增微調權重 (Adapter)：**
            
            1. **使用系統內建的微調功能**：
               請到畫面上方的選單切換至 **Step 6 (Deep Fine-Tuning / 深入微調)**，只要執行過一次微調訓練，系統就會自動將訓練好的權重存檔。成功儲存後，此選單就會自動多出剛訓練好的選項讓您隨時切換。
               
            2. **手動匯入外部權重**：
               如果您有下載現成的 LoRA 權重，請進入專案的 `data/lora_adapters/` 資料夾，將包含權重檔案的整個資料夾複製進去。重新整理網頁後，系統就會自動掃描到並將其加入選單。
            """
        else:
            lora_title = "💡 How to add an Adapter?"
            lora_text = """
            **There are two ways to add a new Fine-tuned Adapter:**
            
            1. **Using the built-in fine-tuning feature**:
               Switch to **Step 6 (Deep Fine-Tuning)** from the top menu. After running a training session, the system will automatically save the trained weights. Once saved, this menu will automatically show the newly trained adapter for you to select.
               
            2. **Manually importing external weights**:
               If you have downloaded existing LoRA weights, go to the project's `data/lora_adapters/` folder and copy the entire folder containing the weights into it. After refreshing the page, the system will automatically scan and add it to the menu.
            """

        with st.expander(lora_title, expanded=False):
            st.markdown(lora_text)
            
        adapters_dir = os.path.join(config.DATA_DIR, "lora_adapters")
        available_adapters = ["預設 (無 / Default)"]
        if os.path.exists(adapters_dir):
            available_adapters.extend([d for d in os.listdir(adapters_dir) if os.path.isdir(os.path.join(adapters_dir, d))])
            
        if 'current_adapter' not in st.session_state:
            st.session_state.current_adapter = "預設 (無 / Default)"
            
        selected_adapter = st.selectbox("選擇作用中的 Adapter", options=available_adapters, index=available_adapters.index(st.session_state.current_adapter))
        
        if selected_adapter != st.session_state.current_adapter:
            st.session_state.current_adapter = selected_adapter
            adapter_to_load = "default_adapter" if selected_adapter == "預設 (無 / Default)" else selected_adapter
            with st.spinner(f"正在切換模型權重至 {selected_adapter} ..."):
                rag_engine.init_models(adapter_name=adapter_to_load)
            st.rerun()
            
    with st.expander("ℹ️ 關於 (About)", expanded=False):
        st.write("此系統由成功大學研究生Gagura在游教授機器學習課程下開發，利用 Hugging Face 與 FAISS 實作本地端學術檢索系統。")

# ==========================================
# 主頁面設計
# ==========================================
st.markdown("<div><span class='gradient-text'>📖 AcademicRAG &nbsp; · &nbsp; Intelligent Paper Reading Assistant</span></div>", unsafe_allow_html=True)
st.caption("Powered by SentenceBERT · FAISS · Cross-Encoder · LoRA Fine-Tuning")

# 提醒使用者確認 LLM 運算模式 (Model Hub)
if st.session_state.get('llm_provider', 'Auto') == 'Auto':
    lang = st.session_state.get('lang', 'zh')
    if lang == "zh":
        st.info("💡 **系統提示**：目前 LLM 處於「自動檢測 (Auto)」模式。建議您前往左側邊欄 **🤖 LLM 萬能切換中心 (Model Hub)** 手動確認或選擇為 **本地端 Apple MLX** 或 **雲端 API**，以確保學術問答與 Agent 連線正常！")
    else:
        st.info("💡 **System Prompt**: LLM is currently in 'Auto' mode. It is recommended to manually select **Local Apple MLX** or **Cloud API** in the **🤖 Model Hub** on the sidebar to ensure Academic Q&A and Agent features work properly!")

# Pipeline 流程卡
st.markdown("""
<div class='glass-card' style='padding:14px 24px; margin-bottom:8px;'>
<b>🛠️ 系統工作流程 (AI Pipeline)</b><br>
<span style='font-size:1.05rem; letter-spacing:0.04em;'>
📚&nbsp;<b>資料輸入</b>&nbsp;➜&nbsp;
🔬&nbsp;<b>資料處理</b>&nbsp;➜&nbsp;
🧠&nbsp;<b>ML 模型</b>&nbsp;➜&nbsp;
🖥️&nbsp;<b>GUI 介面</b>&nbsp;➜&nbsp;
📊&nbsp;<b>結果輸出</b>
</span><br><br>
<small>• <b>資料來源</b>：arXiv 開放 API 下載 / 本地 PDF 上傳  
• <b>資料處理</b>：PyMuPDF 版面解析 → Sliding Window Overlap 語意切片 (v3.1 完全重建，替換舊版 600 字元暴力切片) → 384 維向量 → FAISS 索引  
• <b>ML 方法</b>：SentenceBERT + FAISS (ANN) + Cross-Encoder (Reranking) + LoRA + KMeans + TF-IDF</small>
</div>
""", unsafe_allow_html=True)

# 新手導覽指南
with st.expander("📖 新手導覽指南 (User Quick Start Guide)", expanded=False):
    lang = st.session_state.get('lang', 'zh')
    if lang == "zh":
        st.markdown("""
        **👋 歡迎使用 AI 論文研讀助手！建議您按照以下流程體驗系統的核心功能：**
        
        *   **📥 1. 檔案處理 (Ingestion)** 
            *   隨時上傳您的 PDF 論文或透過 arXiv 匯入，讓知識庫無限擴充！
        *   **🕸️ 2. 知識圖譜 (GraphRAG)**
            *   將論文之間的術語關聯繪製成互動式網路圖，幫助您宏觀掌握 AI 技術發展脈絡。
        *   **🔎 3. 語意檢索 (Hybrid Search)** 
            *   輸入問題（支援中文），系統會透過混合檢索與重排序 (Reranking) 挖出最相關的原文與公式，並給予解釋。
        *   **🤖 4. Agent 對話 (Agentic Chat)** 
            *   與專家團隊進行深度對話！Agent 會自主思考並呼叫 RAG、網頁搜尋等多種工具為您解答複雜問題。
        *   **📚 5. MMR 推薦 (MMR Recommendation)** 
            *   總覽經典論文，透過 GMM 軟分群與 MMR 多樣性演算法為您推薦文獻，並可利用「知識樹推演」探索上下游概念。
        *   **📈 6. 量化微調 (Evaluation & LoRA)**
            *   系統會蒐集您的點擊與反饋 (DPO) 數據，您可以在此啟動 LoRA 微調，打造您專屬的客製化排序模型！
            
        ---
        💡 **最佳建議研讀順序 (Suggested Workflow)：**
        1. **尋找切入點：** 先到 **Step 5 (MMR 推薦)** 總覽知識庫，並透過「知識樹」找到建議的閱讀文獻。
        2. **精準挖掘：** 針對不懂的概念，到 **Step 3 (語意檢索)** 搜尋並查看原始段落與公式。
        3. **深度討論：** 遇到需要綜合分析的難題，交給 **Step 4 (Agent 對話)** 的專家團隊。
        4. **宏觀俯瞰：** 到 **Step 2 (知識圖譜)** 查看您剛才學習的概念在整個 AI 領域的位置。
        5. **自訂模型：** 使用一段時間後，到 **Step 6 (量化微調)** 訓練更懂您的檢索模型。
        """)
    else:
        st.markdown("""
        **👋 Welcome to the AI Paper Reading Assistant! Here is an overview of the core features:**
        
        *   **📥 1. Ingestion** 
            *   Upload PDFs or import from arXiv at any time to infinitely expand your knowledge base!
        *   **🕸️ 2. GraphRAG**
            *   Visualize term relationships across papers in an interactive network graph to grasp the macro context of AI technology development.
        *   **🔎 3. Hybrid Search** 
            *   Ask questions (multi-lingual supported), and the system will use Hybrid Search & Reranking to find the most relevant original texts and equations, providing clear explanations.
        *   **🤖 4. Agentic Chat** 
            *   Have deep conversations with a Multi-Agent expert team! The agents autonomously think and invoke tools like RAG and Web Search to answer complex questions.
        *   **📚 5. MMR Recommendation** 
            *   Overview classic papers, get recommendations via GMM soft clustering & MMR diversity algorithms, and explore upstream/downstream concepts using the Knowledge Tree.
        *   **📈 6. Evaluation & LoRA**
            *   The system collects your click & DPO feedback data. You can start LoRA fine-tuning here to build your personalized reranking model!
            
        ---
        💡 **Suggested Workflow:**
        1. **Find a Starting Point:** Go to **Step 5 (MMR Recommendation)** to overview the knowledge base and find recommended papers using the "Knowledge Tree".
        2. **Precise Digging:** For concepts you don't understand, go to **Step 3 (Hybrid Search)** to search and view the original paragraphs and equations.
        3. **Deep Discussion:** When encountering complex problems that require comprehensive analysis, ask the expert team in **Step 4 (Agentic Chat)**.
        4. **Macro Overview:** Go to **Step 2 (GraphRAG)** to see where the concepts you just learned fit into the entire AI landscape.
        5. **Custom Model:** After using the system for a while, go to **Step 6 (Evaluation & LoRA)** to train a retrieval model that understands you better.
        """)

st.write("---")

# ==========================================
# 路由邏輯
# ==========================================
import ui.step1_ingestion as step1_ingestion
import ui.step2_graphrag as step2_graphrag
import ui.step3_alpha as step3_alpha
import ui.step4_agentic as step4_agentic
import ui.step5_mmr as step5_mmr
import ui.step6_eval as step6_eval
import importlib
importlib.reload(step1_ingestion)
importlib.reload(step2_graphrag)
importlib.reload(step3_alpha)
importlib.reload(step4_agentic)
importlib.reload(step5_mmr)
importlib.reload(step6_eval)

if st.session_state.nav_index == 0:
    step1_ingestion.render(rag_engine, parser, download_arxiv_papers, download_single_pdf)

elif st.session_state.nav_index == 1:
    step2_graphrag.render(rag_engine)

elif st.session_state.nav_index == 2:
    step3_alpha.render(rag_engine, render_latex)

elif st.session_state.nav_index == 3:
    step4_agentic.render(rag_engine)

elif st.session_state.nav_index == 4:
    from ui import step5_mmr
    step5_mmr.render()

elif st.session_state.nav_index == 5:
    from ui import step6_eval
    step6_eval.render(rag_engine)
