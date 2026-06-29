import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from src.locales import get_text

def render(rag_engine):
    lang = st.session_state.get("lang", "zh")
    st.subheader(get_text("menu_step2", lang))
    st.write("基於當前導入的文獻，系統利用 TF-IDF 與共現矩陣自動提取核心術語，並可視化為知識圖譜。" if lang == "zh" else "Builds and visualizes a concept co-occurrence network across all ingested papers using TF-IDF and PageRank.")
    
    if rag_engine.chunks_metadata:
        corpus = [c["text"] for c in rag_engine.chunks_metadata]
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import random
        from src.stop_words import CUSTOM_STOP_WORDS

        # 1. 第一層篩選：抽取 150 個候選名詞 (套用強力的學術停用詞過濾)
        vectorizer = TfidfVectorizer(
            max_features=150, 
            stop_words=CUSTOM_STOP_WORDS,
            ngram_range=(1, 3),
            max_df=0.5,  # 忽略在超過 50% 段落中出現的無鑑別力字彙
            min_df=2     # 忽略只出現過 1 次的孤立錯字
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        candidates = vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        
        # 2. 第二層篩選：0 MB KeyBERT 語意向量排名演算法
        # 利用現有的 embedding_model 計算候選詞與整份論文庫的 Semantic Similarity
        if hasattr(rag_engine, 'embedding_model') and rag_engine.embedding_model is not None:
            try:
                # 建立「論文全局語意向量」：隨機抽樣最多 30 個 chunks 避免記憶體爆炸
                sampled_chunks = random.sample(corpus, min(30, len(corpus)))
                doc_embs = rag_engine.embedding_model.encode(sampled_chunks)
                global_emb = np.mean(doc_embs, axis=0).reshape(1, -1)
                
                # 取得候選詞的獨立向量
                cand_embs = rag_engine.embedding_model.encode(candidates)
                
                # 計算餘弦相似度 (Cosine Similarity)
                sims = cosine_similarity(cand_embs, global_emb).flatten()
                
                # 結合字詞重要性(TF-IDF)與領域語意相關度(Similarity)
                final_scores = tfidf_scores * (sims ** 2)
                
                # 演算法過濾：給予多字詞 (Bigram/Trigram) 權重加成，系統性壓抑零碎單字 (Unigram)
                length_multipliers = np.array([len(str(c).split()) for c in candidates])
                final_scores = final_scores * length_multipliers
                
                sorted_indices = final_scores.argsort()[::-1]
                sorted_candidates = [candidates[i] for i in sorted_indices]
                
                # 子字串去重 (Sub-phrase Deduplication)：如果 "deep" 和 "deep learning" 同時出現，只保留長詞
                feature_names = []
                for cand in sorted_candidates:
                    # 如果該詞是某個已被選擇之長詞的子字串，則視為冗餘 (例如 "size" 遇到 "model size")
                    if any(cand in selected for selected in feature_names):
                        continue
                    feature_names.append(cand)
                    if len(feature_names) == 30:
                        break
            except Exception as e:
                print(f"[GraphRAG] KeyBERT extraction failed, falling back to TF-IDF: {e}")
                length_multipliers = np.array([len(str(c).split()) for c in candidates])
                adjusted_scores = tfidf_scores * length_multipliers
                sorted_candidates = [candidates[i] for i in adjusted_scores.argsort()[::-1]]
                feature_names = []
                for cand in sorted_candidates:
                    if any(cand in selected for selected in feature_names):
                        continue
                    feature_names.append(cand)
                    if len(feature_names) == 30:
                        break
        else:
            # 降級保護：如果 embedding 模型尚未載入
            length_multipliers = np.array([len(str(c).split()) for c in candidates])
            adjusted_scores = tfidf_scores * length_multipliers
            sorted_candidates = [candidates[i] for i in adjusted_scores.argsort()[::-1]]
            feature_names = []
            for cand in sorted_candidates:
                if any(cand in selected for selected in feature_names):
                    continue
                feature_names.append(cand)
                if len(feature_names) == 30:
                    break
            
        # ==========================================================
        # [Route 1] 雙引擎整合：注入 LLM 神級關鍵字 (霸王條款)
        # ==========================================================
        llm_keywords_set = set()
        invalid_patterns = ['invalid_api_key', 'request_error', 'param', 'status', 'code', 'error']
        for c in rag_engine.chunks_metadata:
            if "llm_keywords" in c and isinstance(c["llm_keywords"], list):
                for k in c["llm_keywords"]:
                    k_lower = str(k).lower().strip()
                    # 1. 排除 API 錯誤訊息字串
                    if any(bad in k_lower for bad in invalid_patterns) or len(k_lower) > 25:
                        continue
                    # 2. 嚴格停用詞過濾 (避免 LLM 吐出 model, dataset 等廢話)
                    if k_lower in CUSTOM_STOP_WORDS:
                        continue
                    # 3. 避免過短的非專有名詞
                    if len(k_lower.split()) == 1 and len(k_lower) < 4 and k_lower not in ["llm", "rag", "cnn", "rnn", "nlp", "cv", "ai", "ml"]:
                        continue
                    llm_keywords_set.add(k_lower)
        
        # 為了絕對確保排除無意義字眼 (不受模組快取影響)，在此強制追加黑名單
        HARDCODED_JUNK = {"deep", "size", "large", "small", "random", "first", "similar", "rate", "log", "loss", "tuning", "gradient", "arxiv preprint arxiv", "arxiv", "preprint", "using", "dataset", "performance", "tasks", "results", "model", "used", "data", "training", "time", "language", "information"}
        
        # 將 TF-IDF 萃取的詞彙與 LLM 提煉的詞彙混合，並過濾黑名單
        all_candidates = [c for c in list(feature_names) + list(llm_keywords_set) if c.lower() not in HARDCODED_JUNK]
        
        # 進行全局子字串去重 (Global Sub-phrase Deduplication)
        all_candidates.sort(key=len, reverse=True)
        final_feature_names = []
        for cand in all_candidates:
            if any(cand in selected for selected in final_feature_names):
                continue
            final_feature_names.append(cand)
            
        feature_names = final_feature_names
        
        if len(feature_names) > 40:
            feature_names = feature_names[:40]
        # ==========================================================
        
        G = nx.Graph()
        for word in feature_names:
            if len(word) > 2 and not word.isdigit():
                G.add_node(word)
                
        for chunk in corpus:
            words_in_chunk = [w for w in feature_names if w in chunk.lower() and len(w) > 2 and not w.isdigit()]
            for w1 in words_in_chunk:
                for w2 in words_in_chunk:
                    if w1 != w2:
                        if G.has_edge(w1, w2):
                            G[w1][w2]['weight'] += 1
                        else:
                            G.add_edge(w1, w2, weight=1)
                            
        # 1. 移除完全沒有任何連線的孤立節點
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # 2. 移除低連線度邊緣節點 (度數 < 3)，只保留真正位居知識網絡核心的詞彙
        nodes_to_remove = [n for n, d in G.degree() if d < 3]
        G.remove_nodes_from(nodes_to_remove)
        
        # 3. 只保留最大的連通子圖 (Largest Connected Component)
        if len(G.nodes()) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # 若過濾後圖譜為空，提供防呆處理
        if len(G.nodes()) < 3:
            st.warning("無法建立有效的知識網絡 (節點過少)，請嘗試導入更多文獻。" if lang == "zh" else "Not enough connections found to form a graph. Try ingesting more papers.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
            
        # 更大畫布提供足夠空間避免文字被壓縮
        fig, ax = plt.subplots(figsize=(16, 16), dpi=300)
        # 背景全透明以配合 Streamlit 主題
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # 使用 spring_layout 提供有機排版，k 值越大節點推得越開，避免同一水平線重疊
        pos = nx.spring_layout(G, k=1.5, iterations=300, seed=42)

        # 離群值裁剪：計算所有節點位置的質心與標準差，移除極端離群的節點
        coords = np.array(list(pos.values()))
        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        sigma = dists.std()
        if sigma > 0:
            outlier_nodes = [n for n, d in zip(pos.keys(), dists) if d > 2.0 * sigma]
            if outlier_nodes and len(G.nodes()) - len(outlier_nodes) >= 3:
                G.remove_nodes_from(outlier_nodes)
                pos = {k: v for k, v in pos.items() if k in G.nodes()}

        # 正規化座標到 [-1, 1] 範圍，並鎖定 Matplotlib 視圖邊界，強迫圖譜展開至整個畫布
        pos_array = np.array(list(pos.values()))
        min_vals = pos_array.min(axis=0)
        max_vals = pos_array.max(axis=0)
        range_vals = max_vals - min_vals
        if np.all(range_vals > 0):
            pos = {node: (((np.array(coord) - min_vals) / range_vals) * 2 - 1) for node, coord in pos.items()}
            
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        
        try:
            pr = nx.pagerank(G, alpha=0.85)
            if pr:
                max_pr = max(pr.values())
                min_pr = min(pr.values())
                if max_pr > min_pr:
                    # 進一步放大節點尺寸，確保在更大畫布上仍然清晰可辨
                    node_sizes = [600 + 1800 * ((pr[n] - min_pr) / (max_pr - min_pr)) for n in G.nodes()]
                else:
                    node_sizes = [1200 for n in G.nodes()]
                
            else:
                node_sizes = [300 for n in G.nodes()]
        except:
            d = dict(nx.degree(G))
            node_sizes = [v * 150 for v in d.values()]
        
        # 動態社群偵測 (Community Detection)
        from networkx.algorithms.community import greedy_modularity_communities
        try:
            communities = list(greedy_modularity_communities(G))
        except:
            communities = [set(G.nodes())]
            
        # 定義高質感的柔和色系來標示不同社群
        palette = ['#FFB74D', '#64B5F6', '#81C784', '#E57373', '#BA68C8', '#4DB6AC', '#FFF176', '#FF8A65', '#90A4AE', '#A1887F']
        node_colors = []
        for node in G.nodes():
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(palette[i % len(palette)])
                    break

        # Soft leaf-green and flower-gold edges/nodes styling
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#81C784', ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
        # 加入半透明深色背景框，避免文字相鄰時字元重疊干擾閱讀
        nx.draw_networkx_labels(
            G, pos, 
            font_color='white', 
            font_size=24, 
            font_weight='bold', 
            bbox=dict(facecolor='#1e1e1e', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3'),
            ax=ax
        )
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save figure to bytes as SVG and render using iframe to avoid Streamlit/Pillow SVG marshalling bugs
        import io
        import re
        svg_io = io.BytesIO()
        plt.savefig(svg_io, format="svg", bbox_inches="tight", pad_inches=0, transparent=True)
        svg_str = svg_io.getvalue().decode("utf-8")
        
        # Make SVG fully responsive
        svg_str = re.sub(r'width="[^"]+"', 'width="100%"', svg_str, count=1)
        svg_str = re.sub(r'height="[^"]+"', 'height="100%"', svg_str, count=1)
        
        st.components.v1.html(svg_str, height=850)
        plt.close(fig)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if lang == "zh":
            with st.expander("💡 深入了解：這張圖是怎麼算出來的？它可以做什麼？", expanded=False):
                st.markdown("""
                **這張「知識圖譜 (Knowledge Graph)」揭示了您目前載入的所有論文中的隱藏技術脈絡。**
                
                ### 🧮 它是怎麼算出來的？
                1. **術語萃取 (TF-IDF)**：系統掃描所有的論文段落，利用 TF-IDF 演算法自動抓取出最具代表性的 **Top 20 核心概念**。
                2. **關聯連線 (Co-occurrence)**：接著系統會檢查這 20 個概念，如果任兩個概念**同時出現在同一個論文段落中**，就會建立一條「共現連線」。同時出現越多次，技術關聯度就越高。
                3. **節點大小 (PageRank)**：節點的圓圈大小是透過 Google 著名的 **PageRank 演算法** 計算的。被越多其他核心技術關聯的概念（代表它是領域的基礎核心），它的圓圈就會越大！

                ### 🎯 這張圖可以幹嘛？
                * **快速掌握全局**：當您剛匯入 10 篇全新的未知論文時，這張圖能一秒告訴您這些論文都在探討哪些核心技術。
                * **發現潛在關聯**：您可能會意外發現兩個原本以為無關的技術節點被緊密連線，這代表論文作者將它們結合在一起使用了。
                * **檢索靈感提示**：不知道在 **Step 3 (學術檢索)** 要問什麼問題嗎？挑選圖中最大或相連的幾個關鍵字（例如 `LoRA` 與 `Attention`），丟進去搜尋，就能精準挖出原汁原味的段落！
                """)
        else:
            with st.expander("💡 Deep Dive: How is this graph calculated and what is it for?", expanded=False):
                st.markdown("""
                **This Knowledge Graph reveals the hidden technical context across all your ingested papers.**
                
                ### 🧮 How is it calculated?
                1. **Term Extraction (TF-IDF)**: The system scans all paper chunks and uses the TF-IDF algorithm to automatically extract the **Top 20 core concepts**.
                2. **Relationship Edges (Co-occurrence)**: It then checks these concepts. If any two concepts **appear together in the same paragraph**, a "co-occurrence edge" is created. The more they co-occur, the stronger the technical relationship.
                3. **Node Size (PageRank)**: The size of each node is calculated using Google's **PageRank algorithm**. Concepts that are connected to many other core technologies (indicating they are foundational) will be drawn larger!

                ### 🎯 What can I use this for?
                * **Quick Global Overview**: When you import 10 brand new papers, this graph instantly shows you the main technologies being discussed.
                * **Discover Hidden Links**: You might unexpectedly find two seemingly unrelated concepts closely linked, indicating authors are using them together.
                * **Search Inspiration**: Don't know what to ask in **Step 3 (Semantic Search)**? Pick the largest or connected keywords from this graph and search for them to dig out the exact original paragraphs!
                """)
    else:
        st.info("向量庫目前無文獻資料，無法生成知識圖譜。請先上傳 PDF 或從 arXiv 下載論文！" if lang == "zh" else "No papers in library. Ingest papers in Step 1 first!")
        
    if lang == "zh":
        st.write("### 📂 GraphRAG 動態主題分群 (Dynamic Topic Clustering)")
        st.write("系統已使用 Modularity 演算法將上方的知識圖譜自動劃分為多個潛在的技術主題群集。透過將相近的節點染上**相同的顏色**，這正是 GraphRAG 的精髓：將離散的技術名詞收束為高階概念！")
    else:
        st.write("### 📂 GraphRAG Dynamic Topic Clustering")
        st.write("The system used the Modularity algorithm to partition the graph into technical theme clusters, marked by **matching colors**. This is the essence of GraphRAG: collapsing discrete nodes into higher-level concepts!")
        
    if 'communities' in locals() and communities:
        # Create columns to display communities nicely
        cols = st.columns(min(len(communities), 4))
        for i, comm in enumerate(communities):
            col_idx = i % 4
            comm_nodes = list(comm)
            # Find the "leader" node in this community (highest PageRank if available, else first node)
            if 'pr' in locals() and pr:
                leader = max(comm_nodes, key=lambda n: pr.get(n, 0))
            else:
                leader = comm_nodes[0]
                
            color_hex = palette[i % len(palette)]
            
            with cols[col_idx]:
                st.markdown(f"##### <span style='color:{color_hex}'>●</span> 主題 {i+1}：以 `{leader}` 為核心" if lang == "zh" else f"##### <span style='color:{color_hex}'>●</span> Theme {i+1}: Core `{leader}`", unsafe_allow_html=True)
                st.caption(", ".join(comm_nodes))
    else:
        st.info("尚未偵測到任何主題群集。" if lang == "zh" else "No topic clusters found.")
