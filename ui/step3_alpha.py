import streamlit as st
import os
import re
import time
import json
from src.locales import get_text
from src.embedding_search import hybrid_tokenize

@st.cache_data(max_entries=10, show_spinner=False)
def get_cached_metadata(proj_name):
    """High-performance metadata loader that caches JSON parsing in RAM"""
    from src.project_utils import get_project_paths
    paths = get_project_paths(proj_name)
    meta_path = paths["metadata"]
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def highlight_text(text, query_tokens):
    import re
    highlighted = text
    stopwords = {"a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "is", "are", "was", "were", "with", "by", "from"}
    for token in sorted(query_tokens, key=len, reverse=True):
        if len(token) < 3 or token in stopwords:
            continue
        try:
            pattern = re.compile(rf"\b({re.escape(token)})\b", re.IGNORECASE)
            highlighted = pattern.sub(r"<mark class='highlight-green'>\1</mark>", highlighted)
        except Exception:
            pass
    return highlighted

def render(rag_engine, render_latex):
    lang = st.session_state.get("lang", "zh")
    st.subheader(get_text("search_title", lang))
    st.write(get_text("search_desc", lang))

    # AI Model Info Panel
    with st.expander("🤖 AI Model Pipeline Details", expanded=False):
        if lang == "zh":
            st.markdown('''
本系統採用**多層級 ML Pipeline** 進行學術語意檢索，遠超一般關鍵字搜尋：

| 階段 | 模型/演算法 | 輸入 | 輸出 |
|---|---|---|---|
| **Stage 1：語意嵌入** | SentenceBERT (`all-MiniLM-L6-v2`) | 文字 Chunk | 384 維語意向量 |
| **Stage 2：FAISS 初篩** | 近似最近鄰搜尋 (ANN) | 查詢向量 vs. 索引向量 | Top-20 候選 Chunk |
| **Stage 3：Cross-Encoder 精排** | `ms-marco-MiniLM-L-6-v2` + LoRA | Query+Chunk 配對 | 精確相關度分數 0~1 |

**為什麼需要多層級搜尋？**  
- 快速模式直接利用 FAISS 進行極速比對，適合大量查詢。
- 平衡模式與精準搜尋利用 Cross-Encoder 計算交互注意力，並以 LoRA 微調適配特定學術風格，抑制假陽性。
            ''')
        else:
            st.markdown('''
This system runs a **multi-level ML Pipeline** for advanced semantic search:

| Stage | Model/Algorithm | Input | Output |
|---|---|---|---|
| **Stage 1: Embedding** | SentenceBERT (`all-MiniLM-L6-v2`) | Text Chunk | 384-dimensional Vector |
| **Stage 2: FAISS Filtering** | Approximate Nearest Neighbor | Query vs. Index Vectors | Top-20 Candidates |
| **Stage 3: Cross-Encoder** | `ms-marco-MiniLM-L-6-v2` + LoRA | Query+Chunk Pair | Precise Relevance 0~1 |
            ''')

    # Load metadata for autocomplete & highlights
    from src.metadata_extractor import PaperMetadataManager
    try:
        meta_mgr = PaperMetadataManager(project_name=st.session_state.project_name)
        all_meta = meta_mgr.get_all_metadata()
    except Exception:
        all_meta = {}

    # Extract all unique keywords for suggestion
    all_keywords = set()
    for pid, info in all_meta.items():
        for kw in info.get("keywords", []):
            all_keywords.add(kw.strip().lower())
    
    st.markdown(f"##### {get_text('search_step1_label', lang)}")
    raw_query = st.text_area(
        "Enter query in Chinese or English:",
        value=st.session_state.get("raw_query_input", "梯度下降中權重是如何更新的？"),
        key="raw_query_input",
        height=80
    )

    # Autocomplete Suggestions
    if raw_query:
        matches = [kw for kw in all_keywords if raw_query.lower() in kw][:5]
        if matches:
            st.markdown("💡 **" + ("搜尋建議: " if lang == "zh" else "Suggestions: ") + "** " + 
                        " ".join([f"`{m}`" for m in matches]))

    # Translate button for Chinese queries
    def has_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
        
    if has_chinese(raw_query):
        if st.button(get_text("search_translate_btn", lang)):
            with st.spinner("🤖 LLM translation in progress..."):
                try:
                    from src.agent import AcademicAgent
                    agent = AcademicAgent()
                    translated = agent.translate_to_english(raw_query)
                    
                    if "API 呼叫失敗" in translated or "sk-dummy" in translated or "模型未加載" in translated or "Final Answer:" in translated:
                        st.error("⚠️ 翻譯失敗：未配置有效的大型語言模型（API Key 遺失或本地模型未加載）。請直接手動輸入英文檢索字串。")
                    else:
                        st.session_state["final_query_input"] = translated
                        st.session_state["_auto_trigger_search"] = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Translation failed: {e}")
                    
    st.markdown(f"##### {get_text('search_step2_label', lang)}")
    query = st.text_area(
        "Confirm search query:",
        value=st.session_state.get("final_query_input", "How is the weight updated in gradient descent?" if not has_chinese(st.session_state.get("raw_query_input", "")) else ""),
        key="final_query_input",
        height=120
    )

    # Search Mode Selection
    st.markdown(f"##### {get_text('search_mode_label', lang)}")
    search_mode = st.radio(
        get_text("search_mode_label", lang),
        options=["Fast", "Balanced", "Thorough", "Agentic"],
        format_func=lambda x: get_text(f"mode_{x.lower()}", lang),
        index=1,
        key="search_mode_radio",
        label_visibility="collapsed"
    )

    use_qe = st.toggle(get_text("search_qe_toggle", lang), value=False, help=get_text("search_qe_help", lang))

    # Run search on click or via auto-trigger
    trigger_search = st.button(get_text("search_btn", lang)) or st.session_state.get("_auto_trigger_search", False)
    if st.session_state.get("_auto_trigger_search", False):
        st.session_state["_auto_trigger_search"] = False  # Reset flag immediately
        
    if trigger_search:
        if not query.strip():
            st.warning(get_text("search_warning_empty", lang))
        else:
            spinner_msg = get_text("search_spinner_qe", lang) if use_qe else get_text("search_spinner_normal", lang)
            with st.spinner(spinner_msg):
                if "last_sub_queries" in st.session_state:
                    del st.session_state["last_sub_queries"]
                
                target_projects = st.session_state.get("target_projects", [st.session_state.project_name])
                
                # Sequence-based ABA Guard
                seq_num = st.session_state.get("_search_sequence", 0) + 1
                st.session_state["_search_sequence"] = seq_num
                
                search_results = rag_engine.search(
                    query, 
                    top_k=3, 
                    use_query_expansion=use_qe, 
                    mode=search_mode, 
                    target_projects=target_projects
                )
                
                # Context Guard check (Compare Sequence Number)
                if st.session_state.get("_search_sequence") != seq_num:
                    st.stop()
                
                # Dynamic backfill of LLM Explanation
                if "agent_instance" not in st.session_state:
                    try:
                        from src.agent import AcademicAgent
                        st.session_state.agent_instance = AcademicAgent()
                    except Exception:
                        pass
                
                agent = st.session_state.get("agent_instance")
                if agent:
                    for r in search_results:
                        if not r.get("llm_explanation"):
                            try:
                                explanation = agent.explain_result(
                                    query, r.get("text", ""), max_tokens=60
                                )
                                # Clean up ugly error messages from missing API keys or MLX fallback
                                if "API 呼叫失敗" in explanation or "sk-dummy" in explanation or "模型未加載" in explanation:
                                    r["llm_explanation"] = "⚠️ 系統尚未配置有效的大型語言模型 (未填寫 API Key 或本地模型加載失敗)，請先前往配置以啟用 AI 智能解析。" if lang == "zh" else "⚠️ No valid LLM configured (API Key missing or local model failed to load). Configure it to enable AI explanations."
                                else:
                                    r["llm_explanation"] = explanation.replace("Final Answer:", "").strip()
                            except Exception as e:
                                r["llm_explanation"] = f"Error generating explanation: {e}"
                
                st.session_state["search_results"] = search_results

    if use_qe and st.session_state.get("last_sub_queries"):
        with st.expander("🔍 Query Expansion Trace", expanded=False):
            for i, sq in enumerate(st.session_state["last_sub_queries"], 1):
                label_name = "Original" if i == 1 else ("Reformulated" if i == 2 else "HyDE")
                st.markdown(f"**{label_name}**: `{sq}`")

    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        if results:
            st.write(f"### {get_text('search_results_title', lang)}")
            
            # Show notice if Cross-Encoder fell back to RRF mode
            if any(r.get("rerank_fallback") for r in results):
                st.warning(
                    "⚠️ **Cross-Encoder 精排找不到高信心度匹配（相關度過低），已退回 RRF 融合排序模式顯示候選結果。**\n\n"
                    "💡 **建議操作：**\n"
                    "- 嘗試更換更具體的英文學術關鍵字（如 `attention mechanism`, `self-attention Transformer`）\n"
                    "- 確認您的論文資料庫確實含有相關 Transformer 論文（在 Step 1 確認已匯入）\n"
                    "- 切換到 **Fast 模式** 查看純向量相似度原始結果"
                )
            
            for rank, res in enumerate(results):
                score = res["score"]
                # Normalization mapping based on search mode
                if res.get("is_fast"):
                    score_normalized = max(0.0, min(1.0, (score + 1.0) / 2.0))
                else:
                    score_normalized = max(0.0, min(1.0, score))
                score_pct = int(score_normalized * 100)
                pages_str = ", ".join(f"P.{x+1}" for x in res["pages"])

                paper_id = res.get("paper_id")
                
                # Check for federated search source project
                source_project = res.get("project", st.session_state.project_name)
                
                # Zero-parsing RAM-cached metadata retrieval
                if source_project == st.session_state.project_name:
                    meta = all_meta.get(paper_id, {})
                else:
                    meta = get_cached_metadata(source_project).get(paper_id, {})
                    
                citation_count = meta.get("citation_count", 0)
                influential_citations = meta.get("influential_citation_count", 0)
                venue = meta.get("venue", "")
                year = meta.get("year", "")
                tldr = meta.get("tldr", "")
                recommendations = meta.get("recommendations", [])
                keywords = meta.get("keywords", [])

                citation_str = f" · 📊 {citation_count} Citations ({influential_citations} influential)" if citation_count else ""
                venue_year_str = f" · {venue} ({year})" if venue and year else ""
                
                project_badge = f"<span style='background:#E8F5E9; color:#2E7D32; padding:2px 8px; border-radius:12px; font-size:0.8rem; border:1px solid #C8E6C9;'>📦 {source_project}</span>" if source_project != st.session_state.project_name else ""
                
                with st.container():
                    st.markdown(
                        f"<div style='border:1px solid var(--primary-color, rgba(46,125,50,0.5)); border-radius:12px;"
                        f"padding:16px; margin-bottom:12px; background:var(--background-color); color:var(--text-color); box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:flex-start;'>"
                        f"<h4 style='margin:0;'>📄 {res['paper_title']}</h4>"
                        f"{project_badge}"
                        f"</div>"
                        f"<div style='font-size:0.9rem; opacity:0.8; margin-top:4px;'>"
                        f"Paper ID: {paper_id}{venue_year_str}{citation_str}</div>"
                        f"<div style='font-size:0.9rem; opacity:0.8; margin-top:2px;'>"
                        f"頁碼：{pages_str} &nbsp;|&nbsp; 相關度: <b>{score_pct}%</b></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Warning for negative Cosine Similarity
                    if score < 0:
                        st.markdown(
                            f"<div class='indicator-warn'>"
                            f"<b>{get_text('negative_sim_label', lang)}</b>: "
                            f"原始相似度為 {score:.4f}，代表與檢索主題不相關或語意相反。<br>"
                            f"進度條顯示正規化後為 {score_pct}%。<br>"
                            f"<small>{get_text('negative_sim_help', lang)}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    col_score, col_badge = st.columns([3, 1])
                    with col_score:
                        if res.get("is_rrf"):
                            st.write(f"**{get_text('score_rrf', lang)}:**")
                        elif res.get("is_fast"):
                            st.write(f"**{get_text('score_dense', lang)} (Fast Cosine Score):**")
                        else:
                            st.write(f"**{get_text('score_rerank', lang)}:**")
                        st.progress(score_normalized, text=f"{score:.4f} ({score_pct}%)")
                    with col_badge:
                        if score_pct >= 60:
                            st.success("高度相關 ✅" if lang == "zh" else "Highly Related ✅")
                        elif score_pct >= 35:
                            st.warning("中度相關 ⚠️" if lang == "zh" else "Moderately Related ⚠️")
                        else:
                            st.error("低相關 ❌" if lang == "zh" else "Low Relation ❌")

                if keywords:
                    st.markdown("**🏷️ Keywords:** " + " ".join([f"`{k}`" for k in keywords]))

                if tldr:
                    st.markdown(f"💡 **TL;DR (Semantic Scholar):** *{tldr}*")

                # Keyword Highlighting in Text Card
                text_display = res["text"]
                equations = re.findall(r"\[Equation:\s*(.*?)\]", text_display, flags=re.DOTALL)
                clean_text = re.sub(r"\[Equation:\s*(.*?)\]", " [公式區塊] ", text_display, flags=re.DOTALL)
                
                # Highlight matching tokens from query
                query_tokens = hybrid_tokenize(query)
                highlighted_text = highlight_text(clean_text, query_tokens)
                
                st.markdown(
                    f"<div style='background: var(--secondary-background-color, rgba(46,125,50,0.03)); color: var(--text-color, inherit); padding: 16px;"
                    f"border: 1px solid var(--primary-color, rgba(46,125,50,0.15)); border-radius: 8px; margin-bottom: 12px; font-size: 1.1rem; line-height: 1.6;'>"
                    f"{highlighted_text}</div>",
                    unsafe_allow_html=True
                )

                if res.get("image_path") and os.path.exists(res["image_path"]):
                    st.image(res["image_path"], caption=f"🎯 Figure/Table (P.{res['pages'][0]+1})", use_container_width=True)

                try:
                    import fitz
                    from PIL import Image
                    pdf_dir = rag_engine.project_paths.get("pdf_dir", "")
                    pdf_path_for_img = os.path.join(pdf_dir, f"{res['paper_id']}.pdf")
                    if os.path.exists(pdf_path_for_img) and res.get("pages"):
                        doc = fitz.open(pdf_path_for_img)
                        try:
                            page_to_render = res["pages"][0]
                            if page_to_render < len(doc):
                                page = doc[page_to_render]
                                bboxes = res.get("bboxes", [])
                                for b_info in bboxes:
                                    if b_info.get("page") == page_to_render:
                                        bbox = b_info.get("bbox")
                                        if bbox and len(bbox) == 4:
                                            rect = fitz.Rect(bbox)
                                            page.add_highlight_annot(rect)
                                
                                pix = page.get_pixmap(dpi=150)
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                st.image(img, caption=f"Original PDF Page screenshot (Yellow highlight indicates chunk location)", use_container_width=True)
                        finally:
                            doc.close()
                except Exception as e:
                    st.warning(f"Failed to load PDF screenshot: {e}")

                if "llm_explanation" in res:
                    st.info(f"💡 **{get_text('llm_explanation_title', lang)}**\n{res['llm_explanation']}")

                # Score Decomposition Panel
                dense_dist = res.get("dense_dist", 0.0)
                bm25_score = res.get("bm25_score", 0.0)
                
                dense_pct = max(0.0, min(1.0, (dense_dist + 1.0) / 2.0))
                sparse_pct = max(0.0, min(1.0, bm25_score / 15.0))
                
                with st.expander(get_text("score_decomp_title", lang), expanded=False):
                    st.caption(get_text("score_decomp_desc", lang))
                    col1, col2, col3 = st.columns(3)
                    col1.metric(get_text("score_dense", lang), f"{dense_pct*100:.0f}%", help=f"Cosine Similarity: {dense_dist:.4f}")
                    col2.metric(get_text("score_sparse", lang), f"{sparse_pct*100:.0f}%", help=f"BM25 Score: {bm25_score:.4f}")
                    if res.get("is_rrf"):
                        col3.metric(get_text("score_rrf", lang), f"{score_normalized*100:.0f}%", help="RRF Merge")
                    else:
                        col3.metric(get_text("score_rerank", lang), f"{score_normalized*100:.0f}%", help="LoRA Cross-Encoder Rerank")

                if recommendations:
                    st.write(f"📚 **{get_text('related_papers_title', lang)}**")
                    for rec in recommendations[:3]:
                        st.markdown(f"- **[{rec['title']}](https://arxiv.org/abs/{rec['arxiv_id']})** (arXiv:{rec['arxiv_id']})")

                if equations:
                    st.write(f"📌 **{get_text('key_equations_title', lang)}**")
                    for idx, eq in enumerate(equations):
                        st.code(eq, language="latex")
                        render_latex(eq)

                # Feedback loop
                st.write("---")
                st.write(f"**{get_text('helpful_feedback_prompt', lang)}**")
                fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
                
                def save_feedback_dpo(q, c_text, label_val, r_dict):
                    fb_file = os.path.join(config.BASE_DIR, "feedback_dataset.json")
                    try:
                        if os.path.exists(fb_file):
                            with open(fb_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                        else:
                            data = []
                        data.append({
                            "query": q,
                            "chunk_text": c_text,
                            "label": label_val,
                            "timestamp": time.time()
                        })
                        with open(fb_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Failed to save local feedback: {e}")
                    
                    try:
                        from src.preference_store import save_preference
                        label_name = "positive" if label_val == 1 else "negative"
                        save_preference(q, c_text, label_name, [r_dict])
                        st.toast("感謝您的回饋！" if lang == "zh" else "Feedback saved! Thanks.")
                    except Exception as e:
                        st.error(f"Failed to write DPO preference: {e}")

                fb_key = f"fb_{rank}_{res['paper_id']}"
                feedback_disabled = st.session_state.get(fb_key, False)
                with fb_col1:
                    if st.button(get_text("helpful_yes", lang), key=f"up_{rank}_{res['paper_id']}", disabled=feedback_disabled):
                        save_feedback_dpo(query, text_display, 1, res)
                        st.session_state[fb_key] = True
                        st.rerun()
                with fb_col2:
                    if st.button(get_text("helpful_no", lang), key=f"down_{rank}_{res['paper_id']}", disabled=feedback_disabled):
                        save_feedback_dpo(query, text_display, 0, res)
                        st.session_state[fb_key] = True
                        st.rerun()
                st.write("---")
        else:
            st.warning(get_text("no_results", lang))
