import streamlit as st
import re
import html
import os
import time
import json
import config
from src.locales import get_text

def format_citations_html(text, chunks):
    if not chunks:
        return text
        
    def replace_citation(match):
        doc_num = int(match.group(1))
        chunk_idx = doc_num - 1
        if 0 <= chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            title = chunk.get("paper_title", "Unknown Paper")
            pages = chunk.get("pages", "Unknown Page")
            snippet = chunk.get("text", "")[:250].strip() + "..."
            
            tooltip_text = f"Title: {title}\nPages: {pages}\n\nSnippet:\n{snippet}"
            tooltip_escaped = html.escape(tooltip_text)
            
            return f'<span class="citation-tooltip">[Doc {doc_num}]<span class="tooltiptext">{tooltip_escaped}</span></span>'
        return match.group(0)
        
    return re.sub(r'\[Doc (\d+)\]', replace_citation, text)

def render_agent_chunks(chunks, lang="zh"):
    if not chunks:
        return
    with st.expander("📚 參考文獻與檢索影像 (References & Media)" if lang == "zh" else "📚 References & Media", expanded=False):
        for i, res in enumerate(chunks):
            title = res.get("paper_title", "Unknown Paper")
            pages = res.get("pages", [])
            pages_str = ", ".join(f"P.{x+1}" for x in pages)
            st.markdown(f"**[Doc {i+1}] {title}** (頁碼: {pages_str})")
            
            # Show image if exists
            if res.get("image_path") and os.path.exists(res["image_path"]):
                st.image(res["image_path"], caption=f"Figure from {title}", use_container_width=True)
                
            # Show snippet
            snippet = res.get("text", "")[:300].replace('\n', ' ') + "..."
            st.info(snippet)

def render(rag_engine=None):
    lang = st.session_state.get("lang", "zh")
    st.subheader(get_text("menu_step4", lang))
    st.write(get_text("search_desc", lang) if lang == "en" else "與本地端 LLM 代理人對話。代理人具備意圖路由、反思能力，並可自主呼叫檢索、視覺、生成與教學工具。")
    
    # Lazy loading Agent
    if "agent_instance" not in st.session_state:
        with st.spinner("🧠 正在啟動 LLM 代理人大腦 (MLX 硬體加速中)..." if lang == "zh" else "🧠 Initializing LLM Agent (with hardware acceleration)..."):
            try:
                from src.agent import AcademicAgent
                st.session_state.agent_instance = AcademicAgent()
                st.success("✅ 代理人大腦已上線！" if lang == "zh" else "✅ Agent brain is online!")
            except Exception as e:
                st.error(f"❌ 代理人大腦啟動失敗: {e}" if lang == "zh" else f"❌ Failed to start Agent brain: {e}")
                
    # Debate toggle
    use_debate = st.sidebar.toggle(
        "⚔️ " + ("多智能體辯論驗證" if lang == "zh" else "Multi-Agent Debate"), 
        value=True,
        help="對複雜問題啟用 Generator-Critic-Judge 三方辯論，提升回答品質" if lang == "zh" else "Enable Generator-Critic-Judge debate loop for complex queries."
    )

    # Conversation thread management
    if "conversations" not in st.session_state:
        st.session_state.conversations = {
            "Default Chat": [{"role": "assistant", "content": "你好！我是您的 **Multi-Agent 專家團隊**。我們由路由專家、降噪過濾專家與學術總結專家組成。\n請輸入您的問題，我們將竭誠為您服務！" if lang == "zh" else "Hello! I am your Multi-Agent academic team. Please enter your research query."}]
        }
    if "active_conv" not in st.session_state:
        st.session_state.active_conv = "Default Chat"

    # Thread Navigation Panel
    st.write(f"### {get_text('chat_history_panel', lang)}")
    col_n1, col_n2 = st.columns([2, 1])
    with col_n1:
        conv_list = list(st.session_state.conversations.keys())
        selected_conv = st.selectbox(
            "Select Thread", 
            options=conv_list, 
            index=conv_list.index(st.session_state.active_conv),
            label_visibility="collapsed"
        )
        if selected_conv != st.session_state.active_conv:
            st.session_state.active_conv = selected_conv
            st.rerun()
    with col_n2:
        new_title = st.text_input("New Thread Name", placeholder="+ New...", label_visibility="collapsed")
        if new_title.strip() and new_title.strip() not in st.session_state.conversations:
            st.session_state.conversations[new_title.strip()] = [
                {"role": "assistant", "content": "Thread initialized. Ask your questions here!" if lang == "zh" else "New thread initialized."}
            ]
            st.session_state.active_conv = new_title.strip()
            st.rerun()

    # Active message list
    messages = st.session_state.conversations[st.session_state.active_conv]

    # Render previous messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg.get("chunks"):
                render_agent_chunks(msg["chunks"], lang)

    # Translate & Confirm Workflow
    st.write("---")
    chat_query = st.text_area("對話輸入 / Chat Input", height=150, placeholder="例如：介紹一下 GAN 的架構，並搜尋相關論文" if lang == "zh" else "e.g. explain Transformer and retrieve relevant papers")
    
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        if st.button("🌐 1. 中翻英預覽 (避免 LLM 幻覺)" if lang == "zh" else "🌐 1. Translate Query"):
            if chat_query.strip():
                with st.spinner("正在將中文意圖翻譯為高質量學術英文..." if lang == "zh" else "Translating intent..."):
                    if "agent_instance" in st.session_state and st.session_state.agent_instance is not None:
                        translated_chat = st.session_state.agent_instance.translate_to_english(chat_query)
                        st.session_state["translated_chat"] = translated_chat
                        st.session_state["original_chat"] = chat_query
                        st.rerun()

    if "translated_chat" in st.session_state:
        st.info("✅ **中翻英預覽**：請確認英文指令，或手動微調後發送。" if lang == "zh" else "✅ Translation Preview: Confirm or edit before sending.")
        final_chat_query = st.text_area("學術英文指令 (Academic English Query)", value=st.session_state["translated_chat"], height=120)
        
        if st.button("🚀 2. 傳送給 Multi-Agent 專家團隊" if lang == "zh" else "🚀 2. Send to Agent Team"):
            if final_chat_query.strip():
                st.session_state["dpo_feedback_given"] = False
                original = st.session_state.get('original_chat', final_chat_query)
                display_q = f"**[Original]**: {original}\n\n**[Academic Query]**: {final_chat_query}"
                messages.append({"role": "user", "content": display_q})
                
                with st.chat_message("user"):
                    st.markdown(display_q)

                with st.chat_message("assistant"):
                    if "agent_instance" in st.session_state and st.session_state.agent_instance is not None:
                        agent = st.session_state.agent_instance
                        st.session_state["last_agent_chunks"] = []
                        with st.status(get_text("chat_thought_trace", lang) + "...", expanded=True) as status:
                            st.write("🕵️ **Agent**: Running ReAct Loop...")
                            
                            def trace_callback(step_type, content):
                                if step_type == "thought":
                                    st.markdown(f"🤔 **Thought**: *{content}*")
                                elif step_type == "action":
                                    st.markdown(f"🛠️ **Action**: {content}")
                                elif step_type == "observation":
                                    if "CRAG" in content or "Corrective RAG" in content:
                                        st.warning("🛡️ **CRAG Active**: Local index lacks relevant info. Switched to ArXiv search.")
                                    else:
                                        st.info(f"📥 **Observation**:\n{content}")
                                elif step_type == "error":
                                    st.error(f"❌ **Error**: {content}")
                                elif step_type == "debate":
                                    st.markdown(f"⚔️ {content}")
                                elif step_type == "final_answer":
                                    st.success("🎯 **Conclusion reached!**")
                            
                            response = agent.run(
                                final_chat_query, 
                                max_steps=7, 
                                trace_callback=trace_callback, 
                                use_debate=use_debate, 
                                chat_history=messages[:-1]
                            )
                            
                            status.update(label="✅ Agent Task Complete!", state="complete", expanded=False)
                        
                        agent_chunks = st.session_state.get("last_agent_chunks", [])
                        formatted_response = format_citations_html(response, agent_chunks)
                        st.markdown(formatted_response, unsafe_allow_html=True)
                        
                        # Render rich chunks at the end
                        render_agent_chunks(agent_chunks, lang)
                        
                        messages.append({
                            "role": "assistant", 
                            "content": formatted_response,
                            "chunks": agent_chunks
                        })
                        
                        st.session_state["last_engine_query"] = final_chat_query
                        st.session_state["last_engine_answer"] = response
                        st.session_state["last_engine_chunks"] = agent_chunks
                        
                        del st.session_state["translated_chat"]
                        del st.session_state["original_chat"]
                        st.rerun()
                    else:
                        st.error("Agent not ready.")

    # Clean Button
    if st.button(get_text("chat_clear_btn", lang)):
        st.session_state.conversations[st.session_state.active_conv] = [
            {"role": "assistant", "content": "Thread cleared."}
        ]
        st.rerun()

    # DPO RLHF Feedbacks
    if "last_engine_query" in st.session_state and "last_engine_answer" in st.session_state:
        st.write("---")
        st.write(f"**{get_text('helpful_feedback_prompt', lang)}**")
        feedback_disabled = st.session_state.get("dpo_feedback_given", False)
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            if st.button(get_text("helpful_yes", lang), disabled=feedback_disabled):
                from src.ragas_evaluator import RagasEvaluator
                from src.preference_store import save_preference
                evaluator = RagasEvaluator()
                evaluator.save_feedback(st.session_state["last_engine_query"], st.session_state["last_engine_answer"], "thumb_up")
                save_preference(
                    st.session_state["last_engine_query"], 
                    st.session_state["last_engine_answer"], 
                    "positive", 
                    st.session_state.get("last_engine_chunks", [])
                )
                st.session_state["dpo_feedback_given"] = True
                st.success("感謝您的回饋！已記錄並將在背景微調檢索策略。" if lang == "zh" else "Feedback recorded!")
        with col_f2:
            if st.button(get_text("helpful_no", lang), disabled=feedback_disabled):
                from src.ragas_evaluator import RagasEvaluator
                from src.preference_store import save_preference
                evaluator = RagasEvaluator()
                evaluator.save_feedback(st.session_state["last_engine_query"], st.session_state["last_engine_answer"], "thumb_down")
                save_preference(
                    st.session_state["last_engine_query"], 
                    st.session_state["last_engine_answer"], 
                    "negative", 
                    st.session_state.get("last_engine_chunks", [])
                )
                st.session_state["dpo_feedback_given"] = True
                st.warning("感謝您的回饋！系統將調降相關段落的權重。" if lang == "zh" else "Feedback recorded! Weights downgraded.")
