import streamlit as st
import os
import time
import config
from src.paper_analyzer import PaperAnalyzer
from src.metadata_extractor import PaperMetadataManager
from src.locales import get_text

@st.dialog("⚠️ 警告：清除向量庫 / Warning: Clear Vector Library")
def clear_db_dialog(rag_engine, lang):
    if lang == "zh":
        st.warning("這將會刪除您所有論文的切片資料與知識圖譜紀錄。原始的 PDF 檔案不會被刪除，但您下次需要重新執行所有解析步驟。")
        st.write("確定要清除嗎？")
        col1, col2 = st.columns(2)
        if col1.button("✅ 確定清除", type="primary", use_container_width=True):
            rag_engine.clear_index()
            st.cache_resource.clear()
            from src.model_manager import ModelManager
            ModelManager().release_all_models()
            st.success("向量庫已成功清除！")
            time.sleep(1)
            st.rerun()
        if col2.button("❌ 取消", use_container_width=True):
            st.rerun()
    else:
        st.warning("This will delete all chunk data and knowledge graph records. The original PDF files will NOT be deleted, but you will need to re-parse everything next time.")
        st.write("Are you sure you want to clear the vector library?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Confirm Clear", type="primary", use_container_width=True):
            rag_engine.clear_index()
            st.cache_resource.clear()
            from src.model_manager import ModelManager
            ModelManager().release_all_models()
            st.success("Vector library cleared successfully!")
            time.sleep(1)
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.rerun()

def render(rag_engine, parser, download_arxiv_papers, download_single_pdf):
    lang = st.session_state.get("lang", "zh")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 導入學術論文資源 / Data Ingestion")
        
        st.info("💡 **自訂專屬學術領域指引 (Personalized Domain Guide):**\n\n"
                "本系統支援**全自動本機領域自我微調**！您只需在此分頁導入任何您感興趣的領域文獻（如金融、統計、生醫、管理學等），"
                "並在 **Step 6** 點擊「一鍵開始 LoRA 微調」，系統就會自動針對您所導入的新文獻進行本機自我學習與微調，"
                "無縫客製化出最懂該領域的專屬 RAG 檢索助手！" if lang == "zh" else 
                "💡 **Personalized Domain Guide:**\n\n"
                "This system supports full local fine-tuning! Simply ingest domain-specific papers here and go to Step 6 to fine-tune a custom LoRA adapter for your specialized field.")
        
        source_option = st.radio(
            "選擇論文導入來源 / Select Paper Import Source:", 
            [
                "1. ☁️ 雲端學術庫自動下載 / Cloud DB Auto-Download", 
                "2. 📂 上傳本地 PDF 檔案 / Upload Local PDF",
                "3. 📚 瀏覽現有知識庫 / Browse Knowledge Base"
            ],
            key="source_option_radio"
        )
        
        max_ocr_eqs = st.slider(
            "公式 OCR 轉譯上限 (Nougat Limit)", 
            min_value=0, 
            max_value=50, 
            value=5, 
            step=1,
            help="設定每篇論文最多使用 Nougat 深度學習模型轉譯的公式數量。設定 0 代表僅使用 PyMuPDF 降級擷取，若硬體配置較低，設定較低的值可大幅提昇 Ingestion 速度。"
        )
        
        if "自動下載" in source_option:
            db_choice = st.selectbox("🌍 選擇資料庫來源 / Select Database", ["arXiv (CS/ML 核心)", "OpenReview (ICLR/NeurIPS)", "ACL Anthology (NLP 核心)", "直接貼上 PDF 網址"])
            
            papers = []
            download_clicked = False
            
            if db_choice == "arXiv (CS/ML 核心)":
                query_input = st.text_input("輸入 arXiv ID 或搜尋查詢 / Enter arXiv ID or Query (例如: 1706.03762)", "1706.03762")
                max_res = st.slider("最多下載篇數 / Max Download Count", min_value=1, max_value=5, value=1)
                download_clicked = st.button("開始下載並解析 / Start Download & Parse")
                if download_clicked:
                    with st.spinner("🚀 正在從 arXiv 檢索並下載..."):
                        if query_input.replace(".", "").isdigit():
                            papers = download_arxiv_papers(f"id:{query_input}", max_results=1)
                        elif "cat:" in query_input:
                            papers = download_arxiv_papers(query_input, max_results=max_res)
                        else:
                            cs_query = f'({query_input}) AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE)'
                            papers = download_arxiv_papers(cs_query, max_results=max_res)
            
            elif db_choice == "OpenReview (ICLR/NeurIPS)":
                forum_id = st.text_input("輸入 OpenReview Forum ID 以自動下載論文 / Enter Forum ID to auto-download (例如: ZJptza79R)", "")
                download_clicked = st.button("開始下載並解析 / Start Download & Parse")
                if download_clicked and forum_id:
                    with st.spinner("🚀 正在從 OpenReview 下載..."):
                        url = f"https://openreview.net/pdf?id={forum_id}"
                        papers = download_single_pdf(url, paper_id=f"OR_{forum_id}", title=f"OpenReview_{forum_id}", project_name=rag_engine.project_name)
            
            elif db_choice == "ACL Anthology (NLP 核心)":
                acl_id = st.text_input("輸入 ACL ID 以自動下載論文 / Enter ACL ID to auto-download (例如: 2020.acl-main.1)", "")
                download_clicked = st.button("開始下載並解析 / Start Download & Parse")
                if download_clicked and acl_id:
                    with st.spinner("🚀 正在從 ACL Anthology 下載..."):
                        url = f"https://aclanthology.org/{acl_id}.pdf"
                        papers = download_single_pdf(url, paper_id=f"ACL_{acl_id}", title=f"ACL_{acl_id}", project_name=rag_engine.project_name)
                        
            elif db_choice == "直接貼上 PDF 網址":
                pdf_url = st.text_input("輸入直連 PDF 網址 / Enter PDF URL", "")
                title_input = st.text_input("這篇論文的標題 (可選填) / Title (Optional)", "")
                download_clicked = st.button("開始下載並解析 / Start Download & Parse")
                if download_clicked and pdf_url.startswith("http"):
                    import hashlib
                    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
                    with st.spinner("🚀 正在從給定網址下載 PDF..."):
                        papers = download_single_pdf(pdf_url, paper_id=f"URL_{url_hash}", title=title_input, project_name=rag_engine.project_name)
 
            if papers:
                st.success(f"成功下載 {len(papers)} 篇論文！" if lang == "zh" else f"Successfully downloaded {len(papers)} papers!")
                progress_bar = st.progress(0, text="準備進行學術解析...")
                def progress_cb(prog: float, text: str):
                    progress_bar.progress(prog, text=text)
                for p in papers:
                    if p["pdf_path"] is not None and os.path.exists(p["pdf_path"]):
                        elements = parser.parse_layout(p["pdf_path"], progress_callback=progress_cb)
                        processed_elements = parser.process_pdf_equations(p["pdf_path"], elements, max_ocr_equations=max_ocr_eqs)
                        rag_engine.add_documents(processed_elements, p["id"], p["title"], progress_callback=progress_cb)
                        st.write(f"✅ 已解析並索引: `{p['title']}`")
                    else:
                        st.warning(f"⚠️ `{p['title']}` 的 PDF 下載失敗。系統啟動優雅降級，僅將其摘要 (Abstract) 寫入向量知識庫！")
                        elements = [{"type": "text", "text": "【論文摘要】" + p.get("abstract", ""), "bbox": (0,0,0,0), "page": 0}]
                        processed_elements = elements
                        rag_engine.add_documents(processed_elements, p["id"], p["title"], progress_callback=progress_cb)
                    
                    n_text = sum(1 for e in elements if e['type'] == 'text')
                    n_eq   = sum(1 for e in elements if e['type'] == 'equation')
                    n_chunks = sum(1 for c in rag_engine.chunks_metadata if c['paper_id'] == p['id'])
                    with st.expander(f"🔬 [{p['title'][:40]}...] 資料處理記錄", expanded=False):
                        st.markdown(f"""
| 處理步驟 | 結果 |
|---|---|
| 1️⃣ PDF 版面解析 (PyMuPDF) | 偵測到 **{len(elements)}** 個區塊 |
| 2️⃣ 文字區塊 | **{n_text}** 段 |
| 3️⃣ 數學公式區塊 | **{n_eq}** 個 |
| 4️⃣ 語意切片 (Semantic Chunking) | 整合為 **{n_chunks}** 個 Chunk |
| 5️⃣ SentenceBERT 向量嵌入 | 每 Chunk → **384 維向量** |
| 6️⃣ FAISS 向量索引建立 | 已加入全局向量索引 ✅ |
                        """)
                st.balloons()
                
                with st.spinner("🔄 正在自動同步全局知識圖譜與 TF-IDF 關鍵字..."):
                    _analyzer = PaperAnalyzer(project_name=st.session_state.project_name)
                    _analyzer.extract_keywords_all(top_k=5)
                    _analyzer.cluster_papers(n_clusters=3)
                    
                st.success("✅ 全局知識庫同步完成！即將自動跳轉至「論文推薦與分析」...")
                time.sleep(1.5)
                st.session_state.nav_radio = "Step 5: 📚 多樣性排序與推薦 (MMR Recommendation)"
                st.session_state.nav_index = 4
                st.rerun()
            elif download_clicked and not papers:
                st.warning("⚠️ 找不到符合條件的論文，請嘗試其他關鍵字。")
                        
        elif "上傳" in source_option:
            uploaded_file = st.file_uploader("請上傳 PDF 論文 / Upload PDF Paper:", type=["pdf"])
            if uploaded_file is not None:
                paper_id = os.path.splitext(uploaded_file.name)[0]
                pdf_dir = rag_engine.project_paths.get("pdf_dir", "")
                save_path = os.path.join(pdf_dir, uploaded_file.name)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                st.success(f"檔案上傳成功: {uploaded_file.name}")
                
                if st.button("開始進行學術解析 / Start Academic Parsing"):
                    progress_bar = st.progress(0, text="準備進行學術解析...")
                    def progress_cb(prog: float, text: str):
                        progress_bar.progress(prog, text=text)
                    elements = parser.parse_layout(save_path, progress_callback=progress_cb)
                    meta_mgr = PaperMetadataManager(project_name=st.session_state.project_name)
                    actual_title = meta_mgr.add_local_pdf_metadata(save_path, paper_id)
                    
                    processed_elements = parser.process_pdf_equations(save_path, elements, max_ocr_equations=max_ocr_eqs)
                    rag_engine.add_documents(processed_elements, paper_id, actual_title, progress_callback=progress_cb)
                    
                    st.success(f"論文已成功載入本地知識庫！標題: `{actual_title}`")
                    n_text = sum(1 for e in elements if e['type'] == 'text')
                    n_eq   = sum(1 for e in elements if e['type'] == 'equation')
                    n_chunks = sum(1 for c in rag_engine.chunks_metadata if c['paper_id'] == paper_id)
                    with st.expander("🔬 資料處理記錄", expanded=True):
                        st.markdown(f"""
| 處理步驟 | 結果 |
|---|---|
| 1️⃣ PDF 版面解析 (PyMuPDF) | 偵測到 **{len(elements)}** 個區塊 |
| 2️⃣ 文字區塊 | **{n_text}** 段 |
| 3️⃣ 數學公式區塊 | **{n_eq}** 個 |
| 4️⃣ 語意切片 (Semantic Chunking) | 整合為 **{n_chunks}** 個 Chunk |
| 5️⃣ SentenceBERT 向量嵌入 | 每 Chunk → **384 維向量** |
| 6️⃣ FAISS 向量索引建立 | 已加入全局向量索引 ✅ |
                            """
                        )
                    st.balloons()
                    
                    with st.spinner("🔄 正在自動同步全局知識圖譜與 TF-IDF 關鍵字..."):
                        _analyzer = PaperAnalyzer(project_name=st.session_state.project_name)
                        _analyzer.extract_keywords_all(top_k=5)
                        _analyzer.cluster_papers(n_clusters=3)
                        
                    st.success("✅ 全局知識庫同步完成！即將自動跳轉至「論文推薦與分析」...")
                    time.sleep(1.5)
                    st.session_state.nav_radio = "Step 5: 📚 多樣性排序與推薦 (MMR Recommendation)"
                    st.session_state.nav_index = 4
                    st.rerun()
                        
        elif "僅瀏覽" in source_option:
            st.info("💡 提示：您可以切換至其他分頁進行檢索與分析。" if lang == "zh" else "💡 Info: You can navigate to other tabs for search and analysis.")

    with col2:
        col_name = st.session_state.get("collection_name", "main")
        st.subheader(f"{get_text('kb_summary_title', lang)} (Collection: {col_name})")
        
        n_chunks = len(rag_engine.chunks_metadata)
        paper_ids = list(set(c.get("paper_id", "?") for c in rag_engine.chunks_metadata))
        
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("已索引論文數 (Papers)" if lang == "zh" else "Indexed Papers", len(paper_ids))
        col_s2.metric("語意切片數 (Chunks)" if lang == "zh" else "Semantic Chunks", n_chunks)
        
        st.markdown(f"""
**處理流程 (Pipeline):**
`PDF` → `PyMuPDF` → `Semantic Chunking` → `{'384' if 'MiniLM' in config.EMBEDDING_MODEL_NAME else '1024'} 維 Dense 向量` + `BM25 Sparse 向量`
        """)
        
        if n_chunks > 0:
            with st.expander("📍 向量空間分布圖 (t-SNE Visualization)", expanded=False):
                if st.button("生成 2D 向量分布圖 / Generate t-SNE Plot"):
                    with st.spinner("正在使用 t-SNE 將高維語意向量降維..."):
                        try:
                            import pandas as pd
                            import plotly.express as px
                            from sklearn.manifold import TSNE
                            import numpy as np
                            
                            sample_size = min(n_chunks, 1500)
                            indices = np.random.choice(n_chunks, sample_size, replace=False)
                            sampled_metadata = [rag_engine.chunks_metadata[i] for i in indices]
                            
                            embeddings = []
                            for i in indices:
                                emb = rag_engine.index.reconstruct(int(i))
                                embeddings.append(emb)
                            embeddings = np.array(embeddings)
                            
                            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
                            reduced_embeddings = tsne.fit_transform(embeddings)
                            
                            df = pd.DataFrame({
                                'x': reduced_embeddings[:, 0],
                                'y': reduced_embeddings[:, 1],
                                'Paper ID': [m.get("paper_id", "Unknown") for m in sampled_metadata],
                                'Text Preview': [m.get("text", "")[:100] + "..." for m in sampled_metadata]
                            })
                            
                            fig = px.scatter(
                                df, x='x', y='y', color='Paper ID', hover_data=['Text Preview'],
                                title="Semantic Chunks Distribution (t-SNE)",
                                template="plotly_white"
                            )
                            fig.update_traces(marker=dict(size=5, opacity=0.7))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"無法生成視覺化圖表: {e}")
        st.write("---")
        st.write("#### 🔄 重建向量庫" if lang == "zh" else "#### 🔄 Rebuild Vector Library")
        st.caption("使用最新的 Sliding Window Overlap 語意切片策略，重新處理所有已下載的 PDF。")
        
        # 顯示最近一次建立向量庫的資訊，避免使用者誤觸重建
        import os
        import time
        from datetime import datetime
        vector_db_dir = rag_engine.project_paths.get("vector_db_dir", "")
        index_path = os.path.join(vector_db_dir, "faiss.index")
        
        if os.path.exists(index_path) and rag_engine.chunks_metadata:
            mtime = os.path.getmtime(index_path)
            dt_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            n_chunks = len(rag_engine.chunks_metadata)
            n_papers = len(set(c.get("paper_id", "") for c in rag_engine.chunks_metadata))
            if lang == "zh":
                st.info(f"💡 **目前向量庫狀態**：\n- 最近建立時間：`{dt_str}`\n- 收錄論文數：`{n_papers}` 篇\n- 語意切片數：`{n_chunks}` 塊\n\n*(若您沒有剛匯入新的 PDF 或修改原始碼，請**不需要**點擊下方按鈕重新建立！)*")
            else:
                st.info(f"💡 **Current Vector Library Status**:\n- Last Built: `{dt_str}`\n- Papers Included: `{n_papers}`\n- Semantic Chunks: `{n_chunks}`\n\n*(If you haven't just imported new PDFs or modified source code, there is **NO NEED** to click the rebuild button below!)*")
        else:
            if lang == "zh":
                st.info("💡 目前尚未建立任何向量庫，請點擊下方按鈕開始建立。")
            else:
                st.info("💡 No vector library has been built yet. Please click the button below to start.")
                
        import glob
        pdf_dir = rag_engine.project_paths.get("pdf_dir", "")
        pdf_files_in_dir = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        # Build titles map for display
        from src.metadata_extractor import PaperMetadataManager
        meta_mgr = PaperMetadataManager(project_name=st.session_state.project_name)
        all_meta = meta_mgr.get_all_metadata()
        
        pdf_display_map = {}
        for p in pdf_files_in_dir:
            pid = os.path.splitext(os.path.basename(p))[0]
            title = all_meta.get(pid, {}).get("title")
            if not title or title == pid:
                for c in rag_engine.chunks_metadata:
                    if c.get("paper_id") == pid:
                        t = c.get("paper_title") or c.get("title")
                        if t and t != pid:
                            title = t
                            break
            pdf_display_map[p] = f"[{pid}] {title or 'Unknown Title'}"
            
        if os.path.exists(index_path) and rag_engine.chunks_metadata:
            db_paper_ids = set(c.get("paper_id", "") for c in rag_engine.chunks_metadata if c.get("paper_id"))
            local_pdf_ids = set(os.path.splitext(os.path.basename(p))[0] for p in pdf_files_in_dir)
            
            # Case 1: DB has papers that are missing physical PDFs
            missing_pdfs = db_paper_ids - local_pdf_ids
            if missing_pdfs:
                if lang == "zh":
                    st.info(f"💡 **歷史紀錄繼承**：系統偵測到資料庫中有 **{len(db_paper_ids)} 篇**論文向量，但本地只有 **{len(pdf_files_in_dir)} 份**實體 PDF 檔案。\n\n這代表有 {len(missing_pdfs)} 篇論文是**先前線上抓取或已移除本地 PDF** 的歷史紀錄。重建時系統會**自動繼承**它們，請安心操作！")
                else:
                    st.info(f"💡 **History Inheritance**: The database contains **{len(db_paper_ids)}** paper vectors, but only **{len(pdf_files_in_dir)}** physical PDFs exist locally. The {len(missing_pdfs)} online-fetched or missing PDF papers will automatically inherit their previous data during rebuild.")
                
                with st.expander(f"查看 {len(missing_pdfs)} 篇線上/歷史繼承論文" if lang == "zh" else f"View {len(missing_pdfs)} inherited papers"):
                    missing_display_list = []
                    for pid in missing_pdfs:
                        title = all_meta.get(pid, {}).get("title")
                        if not title or title == pid:
                            for c in rag_engine.chunks_metadata:
                                if c.get("paper_id") == pid:
                                    t = c.get("paper_title") or c.get("title")
                                    if t and t != pid:
                                        title = t
                                        break
                        missing_display_list.append(f"- **[{pid}]** {title or 'Unknown Title'}")
                    st.markdown("\n".join(missing_display_list))
                    
            # Case 2: Local folder has PDFs that are not in DB
            unindexed_pdfs = local_pdf_ids - db_paper_ids
            if unindexed_pdfs:
                if lang == "zh":
                    st.warning(f"🆕 **發現新下載的論文**：本地有 **{len(unindexed_pdfs)} 篇** PDF 檔案尚未被加入到向量庫中！已為您在下方預設勾選，請執行重建以納入知識庫。")
                else:
                    st.warning(f"🆕 **New Papers Detected**: Found **{len(unindexed_pdfs)}** local PDFs that are not yet indexed in the database! They are pre-selected below for you to rebuild.")
                
                with st.expander(f"查看 {len(unindexed_pdfs)} 篇尚未建立向量索引的新論文" if lang == "zh" else f"View {len(unindexed_pdfs)} new unindexed papers"):
                    unindexed_display_list = []
                    for pid in unindexed_pdfs:
                        title = all_meta.get(pid, {}).get("title", "Unknown Title")
                        unindexed_display_list.append(f"- **[{pid}]** {title}")
                    st.markdown("\n".join(unindexed_display_list))
        else:
            unindexed_pdfs = set()
        
        rebuild_mode = st.radio(
            "重新切片範圍 (Rebuild Scope):",
            options=["all", "manual"],
            format_func=lambda x: f"全部 / All (共 {len(pdf_files_in_dir)} 篇)" if x == "all" else "手動勾選 / Manual Select",
            horizontal=True
        )
        
        target_pdfs = None
        if rebuild_mode == "manual":
            # Default to unindexed PDFs if they exist
            default_selections = [p for p in pdf_files_in_dir if os.path.splitext(os.path.basename(p))[0] in unindexed_pdfs]
            
            selected_files = st.multiselect(
                "選擇要重新切片的論文 (未勾選的論文將保留舊的切片資料):",
                options=pdf_files_in_dir,
                default=default_selections,
                format_func=lambda x: pdf_display_map.get(x, os.path.basename(x)),
                help="選擇您剛剛新增或想要重新套用 LLM 萃取的論文。未勾選的論文將直接從舊紀錄繼承，大幅節省時間。"
            )
            target_pdfs = selected_files
            
        col_rb1, col_rb2 = st.columns([1, 1])
        with col_rb1:
            rebuild_clicked = st.button("🔄 重建向量庫 (Rebuild)", use_container_width=True)
        with col_rb2:
            if st.button("🗑️ 清除所有向量庫 (Clear Database)", type="primary", use_container_width=True):
                clear_db_dialog(rag_engine, lang)
                
        if rebuild_clicked:
            if rebuild_mode == "manual" and not target_pdfs:
                st.warning("⚠️ 請至少選擇一篇要重新切片的論文，或選擇「全部」。")
            else:
                progress_bar = st.progress(0, text="初始化重建中...")
                def progress_cb(prog: float, text: str):
                    progress_bar.progress(prog, text=text)
                count = rag_engine.rebuild_index_from_pdfs(progress_callback=progress_cb, target_pdfs=target_pdfs)
            if count > 0:
                progress_cb(0.95, "正在使用 TF-IDF 批次補齊論文關鍵字...")
                try:
                    from src.metadata_extractor import backfill_all_keywords
                    backfill_all_keywords()
                except Exception as e:
                    st.warning(f"關鍵字補齊失敗: {e}")
                progress_cb(1.0, "重建與元數據補全完成！")
                st.cache_resource.clear()  # Clear cache to reload RAG Engine state
                from src.model_manager import ModelManager
                ModelManager().release_all_models()
                st.success(f"✅ 向量庫重建與關鍵字補齊完成！已處理 {count} 篇論文，共 {len(rag_engine.chunks_metadata)} 個 chunks。")
                time.sleep(2)
                st.rerun()
            else:
                st.warning("⚠️ 未找到任何 PDF 檔案，請先在左側匯入論文。")
        
        # Remove specific paper
        st.write("---")
        st.write("#### 🗑 移除特定論文" if lang == "zh" else "#### 🗑 Remove Paper")
        
        # Combine db_paper_ids and local_pdf_ids so user can remove unindexed local PDFs too
        all_removable_ids = list(set(paper_ids) | set(os.path.splitext(os.path.basename(p))[0] for p in pdf_files_in_dir))
        
        if all_removable_ids:
            removable_titles_map = {}
            for pid in all_removable_ids:
                title = all_meta.get(pid, {}).get("title")
                if not title or title == pid:
                    for c in rag_engine.chunks_metadata:
                        if c.get("paper_id") == pid:
                            t = c.get("paper_title") or c.get("title")
                            if t and t != pid:
                                title = t
                                break
                removable_titles_map[pid] = title or pid

            paper_to_remove = st.selectbox(
                "選擇要移除的論文" if lang == "zh" else "Select paper to remove", 
                options=all_removable_ids,
                format_func=lambda x: f"[{x}] {removable_titles_map.get(x, 'Unknown Title')}"
            )
            if st.button("🗑 從專案中移除此論文" if lang == "zh" else "🗑 Remove Paper"):
                with st.spinner(f"正在移除 {paper_to_remove} 並重建索引..."):
                    rag_engine.remove_paper(paper_to_remove)
                    st.cache_resource.clear()  # Clear cache to reload RAG Engine state
                    from src.model_manager import ModelManager
                    ModelManager().release_all_models()
                    st.success(f"✅ 已移除 '{paper_to_remove}'。")
                    st.rerun()
        else:
            st.info("目前沒有可移除的論文。" if lang == "zh" else "No papers to remove.")
