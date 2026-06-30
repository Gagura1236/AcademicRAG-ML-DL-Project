import logging

logger = logging.getLogger(__name__)

_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        logger.info("Initializing AcademicRAGEngine (Lazy Load)...")
        try:
            # Import dynamically to avoid circular imports and loading overhead if not used
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            from embedding_search import AcademicRAGEngine
            _rag_engine = AcademicRAGEngine()
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {e}")
    return _rag_engine

def search_paper(query: str) -> str:
    """
    Researcher Agent Tool: Searches the local academic FAISS vector database for relevant paper segments.
    """
    logger.info(f"Researcher Agent invoked FAISS search with query: {query}")
    # 支援跨專案全方位查詢 (Federated Search)
    target_projects = None
    try:
        import streamlit as st
        if "target_projects" in st.session_state:
            target_projects = st.session_state["target_projects"]
    except Exception:
        pass

    engine = get_rag_engine()
    results = engine.search(query, top_k=3, target_projects=target_projects)
    if not results:
        return "Search Observation: No relevant documents found in the database. [Reflection Trigger] You must re-evaluate your query and try different, broader, or more specific keywords."
        
    formatted_results = []
    for r in results:
        score = r.get('score', 1.0)
        if score < 0.35:
            continue
            
        try:
            import streamlit as st
            if "last_agent_chunks" not in st.session_state or not isinstance(st.session_state["last_agent_chunks"], list):
                st.session_state["last_agent_chunks"] = []
                
            # Deduplicate and append to st.session_state["last_agent_chunks"]
            found_idx = -1
            for idx, existing in enumerate(st.session_state["last_agent_chunks"]):
                if existing.get('text') == r.get('text') and existing.get('paper_title') == r.get('paper_title'):
                    found_idx = idx
                    break
            
            if found_idx == -1:
                st.session_state["last_agent_chunks"].append(r)
                doc_num = len(st.session_state["last_agent_chunks"])
            else:
                doc_num = found_idx + 1
        except Exception:
            # Fallback if Streamlit is not running (e.g. CLI testing)
            doc_num = len(formatted_results) + 1
            
        formatted_results.append(
            f"[Doc {doc_num}]\n"
            f"Title: {r['paper_title']}\n"
            f"Pages: {r['pages']}\n"
            f"Content: {r['text']}\n"
            f"Relevance Score: {score:.4f}"
        )
        
    if not formatted_results:
         return "CRAG 評估警告 (Corrective RAG Warning): 本地知識庫中查無高度相關資訊 (Highest score < 0.35)。\n[Reflection Trigger] 你必須停止幻想，立即使用 `search_arxiv` 工具連網查詢最新學術資訊，或改寫關鍵字後重新搜尋。"
         
    return "Search Observation: Found the following highly relevant academic segments:\n\n" + "\n---\n".join(formatted_results)
