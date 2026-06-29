import streamlit as st
import os
import config
from src.paper_analyzer import PaperAnalyzer
from src.locales import get_text

def render():
    lang = st.session_state.get("lang", "zh")
    st.subheader(get_text("mmr_title", lang))
    st.write(get_text("mmr_desc", lang))
    
    analyzer = PaperAnalyzer(project_name=st.session_state.project_name)
    metadata = analyzer.metadata_manager.get_all_metadata()
    
    if metadata:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"### {get_text('cluster_run_btn', lang).split('/')[0].strip()}")
            if lang == "zh":
                st.info("💡 **什麼是 GMM 軟分群？** 系統會透過高斯混合模型將論文依照語意自動分群，並計算每篇論文屬於各個主題的「機率」，幫助您快速歸納出知識庫中的主要研究領域。拉動下方滑桿即可重新分群！")
            else:
                st.info("💡 **What is GMM Soft Clustering?** The system uses a Gaussian Mixture Model to group papers based on semantics, calculating the probability of each paper belonging to different topics. Adjust the slider below to re-cluster!")
            n_clusters = st.slider(get_text("cluster_num_slider", lang), min_value=2, max_value=10, value=3)
            if st.button(get_text("cluster_run_btn", lang)):
                with st.spinner(get_text("cluster_spinner", lang)):
                    analyzer.extract_keywords_all(top_k=5)
                    analyzer.cluster_papers(n_clusters=n_clusters)
                    st.success(get_text("cluster_success", lang))
                    import time
                    time.sleep(1.5)
                    st.rerun()
            
            st.write(f"### {get_text('kg_title', lang)}")
            if lang == "zh":
                st.info("💡 **什麼是視覺化知識圖譜？** 系統會將論文間的關鍵字關聯，視覺化為網狀圖。若開啟 LLM，更會由大模型深度抽取實體關係並計算 PageRank，讓核心技術一目了然！")
            else:
                st.info("💡 **What is Knowledge Graph?** Visualizes keyword connections between papers. If LLM is enabled, it deeply extracts entity relationships and calculates PageRank to highlight core technologies!")
            use_llm = st.toggle(get_text("kg_llm_toggle", lang), value=False)
            
            if st.button(get_text("kg_render_btn", lang)):
                with st.spinner(get_text("kg_spinner_llm", lang) if use_llm else get_text("kg_spinner_normal", lang)):
                    import uuid
                    import streamlit.components.v1 as components
                    session_id = st.session_state.get("_session_id", uuid.uuid4().hex)
                    if "_session_id" not in st.session_state:
                        st.session_state["_session_id"] = session_id
                    session_graph_path = os.path.join(config.DATA_DIR, f"kg_{session_id}.html")
                    graph_path = analyzer.generate_knowledge_graph(output_path=session_graph_path, use_llm=use_llm)
                    if graph_path and os.path.exists(graph_path):
                        with open(graph_path, 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        components.html(html_data, height=700, scrolling=True)
                    else:
                        st.error(get_text("kg_error", lang))
            
            st.write(f"### {get_text('kb_summary_title', lang)}")
            for pid, info in metadata.items():
                with st.expander(f"{info.get('title', pid)}"):
                    st.write(f"**Authors:** {', '.join(info.get('authors', []))}")
                    st.write(f"**Keywords:** {', '.join(info.get('keywords', []))}")
                    st.write(f"**Cluster (Soft Clustering):** {info.get('cluster_name', '未分類')}")
                    if "cluster_probs" in info:
                        probs = info["cluster_probs"]
                        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
                        st.caption("跨領域機率分布 (GMM Probabilities):" if lang == "zh" else "Topic Probabilities (GMM):")
                        for idx in top_indices:
                            p = probs[idx]
                            if p > 0.01:
                                p_clamped = max(0.0, min(1.0, float(p)))
                                st.progress(p_clamped, text=f"Topic {idx}: {p_clamped:.1%}")
                    st.write(f"**Abstract:** {info.get('abstract', '')}")
                    
        with col2:
            st.write(f"### {get_text('rec_panel_title', lang)}")
            st.markdown(f"<p style='font-size: 0.9rem; color: #555;'>{get_text('rec_panel_desc', lang)}</p>", unsafe_allow_html=True)
            
            if lang == "zh":
                st.info("💡 **如何使用 MMR 推薦？** 請先輸入您的研究興趣（可透過按鈕翻譯為學術英文）。系統會使用最大邊際相關性 (MMR) 演算法，在確保「高相關度」的同時強制加入「多樣性」，避免推薦的論文都長得一模一樣。")
            else:
                st.info("💡 **How to use MMR Recommendation?** Enter your research interest (you can translate it to academic English). The system uses Maximal Marginal Relevance (MMR) to recommend highly relevant papers while forcing 'diversity' to avoid homogenous results.")
            
            
            use_llm_rec = st.toggle(get_text("rec_llm_toggle", lang), value=False, help="LLM will review candidate abstracts and write a recommendation report.")
            interest = st.text_area(get_text("rec_interest_label", lang), "", height=80)
            
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                if st.button(get_text("rec_trans_btn", lang)):
                    if interest.strip():
                        with st.spinner(get_text("rec_trans_spinner", lang)):
                            from src.agent import AcademicAgent
                            agent = AcademicAgent()
                            translated = agent.translate_to_english(interest)
                            st.session_state["translated_interest"] = translated
                            st.rerun()
            
            if "translated_interest" in st.session_state:
                st.info(get_text("rec_trans_preview", lang))
                final_interest = st.text_area(get_text("rec_query_label", lang), value=st.session_state["translated_interest"], height=100)
            else:
                final_interest = interest
            
            if not use_llm_rec:
                mmr_penalty = st.slider(get_text("rec_diversity_slider", lang), min_value=0.0, max_value=1.0, value=0.6, step=0.1)
                
            if st.button(get_text("rec_btn", lang)):
                if final_interest.strip():
                    with st.spinner(get_text("rec_spinner_llm", lang) if use_llm_rec else get_text("rec_spinner_normal", lang)):
                        if use_llm_rec:
                            recs = analyzer.recommend_papers_with_llm(final_interest, top_k=3)
                        else:
                            recs = analyzer.recommend_papers_by_interest(final_interest, top_k=3, diversity_penalty=mmr_penalty)
                        st.session_state["rec_results"] = recs
                        st.session_state["rec_interest"] = final_interest
                        st.session_state["rec_is_llm"] = use_llm_rec
                else:
                    st.warning("請先輸入研究興趣關鍵字。" if lang == "zh" else "Please enter your research interest first.")

            if "rec_results" in st.session_state and st.session_state.get("rec_results"):
                recs = st.session_state["rec_results"]
                disp_interest = st.session_state.get("rec_interest", "")
                is_llm = st.session_state.get("rec_is_llm", False)
                
                if recs:
                    st.write(f"{get_text('rec_results_prefix', lang)} '{disp_interest}':")
                    for i, rec in enumerate(recs):
                        with st.container():
                            st.markdown(f"**[{i+1}] {rec['title']}**")
                            
                            if is_llm and "llm_reason" in rec:
                                st.info(f"💡 **LLM Reason:**\n{rec['llm_reason']}")
                                st.caption(f"LLM Rank: #{rec.get('llm_rank', '?')} · Keywords: {', '.join(rec.get('keywords', []))}")
                            else:
                                sim = rec.get('similarity', 0.0)
                                sim_normalized = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                                sim_pct = int(sim_normalized * 100)
                                
                                if sim < 0:
                                    st.markdown(
                                        f"<div class='indicator-warn'>"
                                        f"<b>{get_text('negative_sim_label', lang)}</b>: "
                                        f"相似度為 {sim:.4f} (正規化: {sim_pct}%)，代表此論文與您輸入的興趣可能相反或極度無關。<br>"
                                        f"<small>{get_text('negative_sim_help', lang)}</small>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                
                                st.progress(sim_normalized, text=f"{get_text('rec_similarity_label', lang)}: {sim:.4f}  ({get_text('rec_normalized_label', lang)}: {sim_pct}%)")
                                if sim >= 0.5:
                                    st.success(f"{get_text('rec_highly_recommended', lang)} · Keywords: {', '.join(rec.get('keywords', []))}")
                                elif sim >= 0.3:
                                    st.warning(f"{get_text('rec_moderately_recommended', lang)} · Keywords: {', '.join(rec.get('keywords', []))}")
                                else:
                                    st.info(f"{get_text('rec_weakly_recommended', lang)} · Keywords: {', '.join(rec.get('keywords', []))}")
                                    
                    if not is_llm:
                        st.caption(get_text("rec_explain_help", lang))
                else:
                    st.warning(get_text("rec_no_found", lang))
                        
            st.write(f"### {get_text('tree_title', lang)}")
            if lang == "zh":
                st.info("💡 **什麼是知識樹推演？** 只要輸入單一概念（如 Transformer），大模型會推演出它的「前置技術」、「核心平替」與「後續發展」三個維度，並自動幫您從庫中找出對應的三批文獻。")
            else:
                st.info("💡 **What is Knowledge Tree?** Enter a single concept (e.g. Transformer), and the LLM will deduce its 'Upstream' (prerequisites), 'Core' (parallels), and 'Downstream' (future applications). It then automatically finds matching papers for all three dimensions.")
            tree_concept = st.text_input(get_text("tree_input_label", lang), "", key="tree_concept")
            use_openalex = st.toggle(get_text("tree_openalex_toggle", lang), value=False, help=get_text("tree_openalex_desc", lang))
            openalex_strategy = "relevance"
            if use_openalex:
                st.caption(get_text("tree_openalex_strategy_label", lang))
                strategy_options = {
                    "relevance": "🔮 Relevance priority" if lang == "en" else "🔮 語意關聯優先 (Relevance)",
                    "impact": "🏆 Impact priority" if lang == "en" else "🏆 影響力優先 (Impact)",
                    "recent": "🚀 Recent trends" if lang == "en" else "🚀 前沿趨勢 (Bleeding-edge)"
                }
                openalex_strategy = st.radio("Strategy", options=list(strategy_options.keys()), format_func=lambda x: strategy_options[x], label_visibility="collapsed")
            
            if st.button(get_text("tree_btn", lang)):
                if tree_concept.strip():
                    with st.spinner(get_text("tree_spinner", lang)):
                        tree_results = analyzer.recommend_papers_by_tree(tree_concept, use_openalex=use_openalex, openalex_strategy=openalex_strategy)
                        st.session_state["tree_results"] = tree_results
                        st.session_state["tree_concept"] = tree_concept
                else:
                    st.warning("請輸入欲探索的概念。" if lang == "zh" else "Please enter concept first.")
                    
            if "tree_results" in st.session_state and st.session_state["tree_results"]:
                res = st.session_state["tree_results"]
                tree = res.get("tree")
                recs = res.get("recommendations", {})
                
                if tree:
                    st.info(get_text("tree_success", lang))
                    col_u, col_c, col_d = st.columns(3)
                    with col_u:
                        st.write(f"**{get_text('tree_upstream', lang)}**")
                        u_val = tree.get("upstream", "")
                        if isinstance(u_val, list):
                            for t in u_val: st.caption(f"• {t}")
                        else:
                            st.info(f"💡 {u_val}")
                        st.write("---")
                        for i, r in enumerate(recs.get("upstream", [])):
                            st.markdown(f"**[{i+1}] {r['title']}**")
                            sim = r['similarity']
                            sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                            st.progress(sim_norm, text=f"Sim: {sim:.3f}")
                            
                    with col_c:
                        st.write(f"**{get_text('tree_core', lang)}**")
                        c_val = tree.get("core", "")
                        if isinstance(c_val, list):
                            for t in c_val: st.caption(f"• {t}")
                        else:
                            st.info(f"💡 {c_val}")
                        st.write("---")
                        for i, r in enumerate(recs.get("core", [])):
                            st.markdown(f"**[{i+1}] {r['title']}**")
                            sim = r['similarity']
                            sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                            st.progress(sim_norm, text=f"Sim: {sim:.3f}")
                            
                    with col_d:
                        st.write(f"**{get_text('tree_downstream', lang)}**")
                        d_val = tree.get("downstream", "")
                        if isinstance(d_val, list):
                            for t in d_val: st.caption(f"• {t}")
                        else:
                            st.info(f"💡 {d_val}")
                        st.write("---")
                        for i, r in enumerate(recs.get("downstream", [])):
                            st.markdown(f"**[{i+1}] {r['title']}**")
                            sim = r['similarity']
                            sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                            st.progress(sim_norm, text=f"Sim: {sim:.3f}")
                            
            st.write("---")
            st.write(f"### {get_text('compare_title', lang)}")
            paper_options = [(pid, info.get("title", pid)) for pid, info in metadata.items()]
            if len(paper_options) >= 2:
                p1 = st.selectbox(get_text("compare_p1_label", lang), options=paper_options, format_func=lambda x: x[1], key="p1")
                p2 = st.selectbox(get_text("compare_p2_label", lang), options=paper_options, format_func=lambda x: x[1], key="p2")
                if st.button(get_text("compare_btn", lang)):
                    if p1[0] == p2[0]:
                        st.warning(get_text("compare_warning_same", lang))
                    else:
                        comp = analyzer.compare_two_papers(p1[0], p2[0])
                        if comp:
                            sim = comp['similarity']
                            sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                            st.write(get_text("compare_sim_label", lang))
                            st.progress(sim_norm, text=f"{sim:.4f} ({int(sim_norm*100)}%)")
                            if sim >= 0.7:
                                st.success(get_text("compare_high_match", lang))
                            elif sim >= 0.4:
                                st.info(get_text("compare_mid_match", lang))
                            else:
                                st.warning(get_text("compare_low_match", lang))
                            st.success(f"**{get_text('compare_common_kw', lang)}** {', '.join(comp['common_keywords']) if comp['common_keywords'] else '无'}")
                            st.info(f"**{get_text('compare_p1_unique', lang)}** {', '.join(comp['paper1_unique']) if comp['paper1_unique'] else '无'}")
                            st.warning(f"**{get_text('compare_p2_unique', lang)}** {', '.join(comp['paper2_unique']) if comp['paper2_unique'] else '无'}")
                            st.caption(get_text("compare_help", lang))
            else:
                st.info(get_text("compare_need_two", lang))
    else:
        st.info(get_text("kb_empty", lang))
