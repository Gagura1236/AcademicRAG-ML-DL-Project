import os
import json
import logging
import re
import sys
from typing import Dict, Any, Callable
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

class UniversalLLMProvider:
    """
    萬能 LLM 橋接層 (Universal LLM Bridge)
    解決跨平台 (Mac/Windows) 與無 GPU 電腦的適配問題。
    支援優雅降級 (Graceful Degradation) 至雲端 API。
    """
    def __init__(self, model_path=None, provider="Auto", api_key="", api_base_url="", api_model_name=""):
        self.model_path = model_path or config.LLM_MODEL_PATH
        self.provider = provider
        self.api_key = api_key or config.LLM_API_KEY
        self.api_base_url = api_base_url or config.LLM_API_BASE_URL
        self.api_model_name = api_model_name or config.LLM_API_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.load_error = None
        
        if self.provider == "Auto":
            if sys.platform == "darwin":
                self.provider = "MLX"
            else:
                self.provider = "API"
                
        if self.provider == "MLX":
            self._load_mlx()
        elif self.provider == "API":
            self._check_api()
            
    def _load_mlx(self):
        logger.info(f"Loading MLX LLM model from {self.model_path}...")
        try:
            import mlx_lm
            self.model, self.tokenizer = mlx_lm.load(self.model_path)
            logger.info("MLX Model loaded successfully.")
            self.load_error = None
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            self.load_error = str(e)
            logger.warning("Falling back to API Provider.")
            self.provider = "API"
            self._check_api()

    def _check_api(self):
        if not self.api_key and "localhost" not in self.api_base_url and "127.0.0.1" not in self.api_base_url:
            logger.warning("API Provider selected without API Key, this might fail unless using a local server (like Ollama/LM Studio).")
            
    def generate(self, prompt: str, max_tokens: int = 4096, temp: float = 0.05) -> str:
        if self.provider == "MLX" and self.model is not None:
            try:
                import mlx_lm
                import mlx.core as mx
                response = mlx_lm.generate(
                    self.model, 
                    self.tokenizer, 
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                return response.strip()
            except Exception as e:
                return f"Final Answer: (MLX 推論失敗: {e})"
                
        elif self.provider == "API":
            try:
                # Use OpenAI package to connect to ANY OpenAI-compatible server
                from openai import OpenAI
                
                if not self.api_key:
                    return "Final Answer: (API 金鑰未設定，請在 Model Hub 中填寫有效金鑰)"
                client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
                response = client.chat.completions.create(
                    model=self.api_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                return response.choices[0].message.content.strip()
            except ImportError:
                return "Final Answer: (缺少 openai 套件，請在終端機執行 `pip install openai`)"
            except Exception as e:
                logger.error(f"API Error: {e}")
                return f"Final Answer: (API 呼叫失敗: {e})"
                
        return "Final Answer: (模型未加載，無法推理)"

@st.cache_resource(show_spinner=False)
def get_llm_provider_cached(model_path: str, provider: str, api_key: str, api_base_url: str, api_model_name: str):
    """
    Cache the UniversalLLMProvider globally to prevent OOM or redundant API connections
    when AcademicAgent is instantiated multiple times. (Cache invalidated to force MLX reload)
    """
    return UniversalLLMProvider(model_path, provider, api_key, api_base_url, api_model_name)

class AcademicAgent:
    """
    Agentic RAG Core Engine using a custom LLM.
    """
    def __init__(self, model_path=None):
        try:
            current_provider = st.session_state.get('llm_provider', config.LLM_PROVIDER)
            current_api_key = st.session_state.get('llm_api_key', config.LLM_API_KEY)
            current_api_base_url = st.session_state.get('llm_api_base_url', config.LLM_API_BASE_URL)
            current_api_model_name = st.session_state.get('llm_api_model_name', config.LLM_API_MODEL_NAME)
        except Exception:
            current_provider = config.LLM_PROVIDER
            current_api_key = config.LLM_API_KEY
            current_api_base_url = config.LLM_API_BASE_URL
            current_api_model_name = config.LLM_API_MODEL_NAME
            
        self.model_path = model_path or config.LLM_MODEL_PATH
        self.llm_provider = get_llm_provider_cached(
            self.model_path, 
            current_provider, 
            current_api_key, 
            current_api_base_url, 
            current_api_model_name
        )
            
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        
        self.system_prompt = """You are an elite autonomous Academic AI Agent running locally on Apple Silicon.
Your goal is to help users analyze computer science papers, extract data, and explain complex deep learning models.

CRAG (Corrective RAG) RULE:
If a tool returns a "CRAG 評估警告" (CRAG Warning), you MUST NOT hallucinate or guess the answer. You MUST use the `search_arxiv` tool to search the internet for the missing knowledge, or explicitly tell the user that the information is not in the database.

CITATION RULE:
When writing your Final Answer, if you reference information retrieved from the `search_paper` tool, you MUST add an inline citation in the format `[Doc X]` (e.g., [Doc 1] for the first retrieved segment, [Doc 2] for the second, etc.). Do NOT use generic tags like [1] or [Paper 1]. Every claim backed by local documents must have this tag.

AVAILABLE TOOLS:
{tools_desc}

To use a tool, you MUST output exactly in this format and nothing else:
Thought: explain what you are thinking and why you need a tool
Action: tool_name
Action Input: {{"arg_name": "arg_value"}}

Wait for the observation. Once you have the final answer, output:
Final Answer: your detailed response in Traditional Chinese (繁體中文).
"""
        self._setup_default_tools()

    def _setup_default_tools(self):
        # Register external tools from the tools folder
        try:
            from .tools.vision_tool import analyze_image_resnet
            from .tools.generator_tool import generate_augmentation_gan
            from .tools.tutor_tool import launch_transformer_tutor
            from .tools.rag_tool import search_paper
            from .tools.arxiv_tool import search_arxiv
            
            self.register_tool("analyze_image_resnet", "Use to visually analyze model architectures and plot images from papers using ResNet. Input: {'image_path': 'path/to/image'}", analyze_image_resnet)
            self.register_tool("generate_augmentation_gan", "Use to perform data augmentation using GAN when datasets are small. Input: {'num_samples': int}", generate_augmentation_gan)
            self.register_tool("launch_transformer_tutor", "Use to dynamically explain attention and Transformer concepts via an interactive GUI. Input: {'topic': 'concept to explain'}", launch_transformer_tutor)
            self.register_tool("search_paper", "Use to search the local knowledge base (vector database) for academic papers, equations, and methodologies. Input: {'query': 'search keywords'}", search_paper)
            self.register_tool("search_arxiv", "Use to search the ArXiv API for the latest academic papers and abstracts online. Input: {'keyword': 'search keyword'}", search_arxiv)
            self.register_tool("paper_tldr", "Get citation count and a brief TL;DR summary for an arXiv paper. Input: {'arxiv_id': 'arXiv paper ID (e.g. 1706.03762)'}", self._tool_paper_tldr)
            self.register_tool("recommend_related", "Get recommended academic papers related to a given arXiv ID from Semantic Scholar. Input: {'arxiv_id': 'arXiv paper ID'}", self._tool_recommend_related)
            self.register_tool("search_knowledge_graph", "Use to search or retrieve global community summaries from the Leiden hierarchical knowledge graph of the paper database to answer global thematic questions. Input: {'query': 'optional keyword filter'}", self._tool_search_knowledge_graph)
        except ImportError as e:
            logger.warning(f"Could not load some tools: {e}")

    def _tool_paper_tldr(self, arxiv_id: str) -> str:
        try:
            from src.semantic_scholar_client import enrich_paper
            d = enrich_paper(arxiv_id.strip())
            if not d:
                return f"No TL;DR found for arXiv ID: {arxiv_id}"
            return f"Title/TL;DR: {d.get('tldr', 'N/A')}\nVenue: {d.get('venue', 'N/A')}\nYear: {d.get('year', 'N/A')}\nCitations: {d.get('citation_count', 0)}\nInfluential Citations: {d.get('influential_citation_count', 0)}"
        except Exception as e:
            return f"Error fetching TL;DR: {e}"

    def _tool_recommend_related(self, arxiv_id: str) -> str:
        try:
            from src.semantic_scholar_client import get_recommendations
            recs = get_recommendations(arxiv_id.strip())
            if not recs:
                return f"No related papers found for arXiv ID: {arxiv_id}"
            return "Recommended related papers:\n" + "\n".join(f"- arXiv:{r['arxiv_id']} — {r['title']}" for r in recs)
        except Exception as e:
            return f"Error fetching recommendations: {e}"

    def _tool_search_knowledge_graph(self, query: str = None) -> str:
        """
        Search or retrieve community summaries from the Leiden hierarchical community detection of the paper database.
        Allows global reasoning about the dataset theme.
        """
        try:
            import json
            import os
            from config import DATA_DIR
            
            summary_path = os.path.join(DATA_DIR, "community_summaries.json")
            if not os.path.exists(summary_path):
                return "Observation: No knowledge graph community summaries found. You must tell the user to generate the knowledge graph (graph-rag) from the UI tab first."
                
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            summaries_dict = {}
            if "levels" in data:
                levels_dict = data["levels"]
                if levels_dict:
                    highest_level = list(levels_dict.keys())[-1]
                    summaries_dict = levels_dict[highest_level]
            else:
                summaries_dict = data
                
            if not summaries_dict:
                return "Observation: Knowledge graph community summaries file is empty."
                
            formatted = []
            for c_id, info in summaries_dict.items():
                summary_text = info.get("summary", "") if isinstance(info, dict) else str(info)
                nodes = info.get("nodes", []) if isinstance(info, dict) else []
                
                if query:
                    q_lower = query.lower()
                    if q_lower not in summary_text.lower() and not any(q_lower in n.lower() for n in nodes):
                        continue
                        
                nodes_str = ", ".join(nodes[:15]) + ("..." if len(nodes) > 15 else "")
                formatted.append(f"Community: {c_id}\nEntities/Nodes: {nodes_str}\nSummary: {summary_text}")
                
            if not formatted:
                if query:
                    return f"Observation: No community summaries matched the query '{query}'."
                return "Observation: No community summaries found."
                
            return "Observation: Found the following knowledge graph community summaries:\n\n" + "\n---\n".join(formatted)
        except Exception as e:
            return f"Error reading knowledge graph summaries: {e}"

    def register_tool(self, name: str, description: str, func: Callable):
        self.tools[name] = func
        self.tool_descriptions[name] = description
        logger.info(f"Registered tool: {name}")

    def _build_prompt(self, user_query: str, history: str = "", chat_history: list = None) -> str:
        tools_str = ""
        for name, desc in self.tool_descriptions.items():
            tools_str += f"- {name}: {desc}\n"
            
        sys_prompt = self.system_prompt.format(tools_desc=tools_str)
        
        # Format as Llama 3 Instruct prompt
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
        
        # Add past multi-turn chat history if present
        if chat_history:
            for msg in chat_history:
                role = msg["role"]
                # Skip default greeting to save context space
                if "Multi-Agent 專家團隊" in msg["content"] and role == "assistant":
                    continue
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                
        # Current turn
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}\n\n{history}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def generate(self, prompt: str, max_tokens: int = 4096, temp: float = 0.05) -> str:
        return self.llm_provider.generate(prompt, max_tokens, temp)
        
    def run(self, user_query: str, max_steps: int = 5, trace_callback: Callable = None, use_debate: bool = True, chat_history: list = None) -> str:
        """
        Executes the ReAct (Reasoning and Acting) loop.
        """
        # Intent Routing
        intent = self.router_agent(user_query, chat_history)
        if intent == "DIRECT_ANSWER":
            if trace_callback:
                trace_callback("thought", "意圖分類: 直接回答型問題，跳過檢索")
            return self.synthesis_agent(user_query, clean_context="", chat_history=chat_history)

        history = ""
        retrieved_contexts = []
        for step in range(max_steps):
            prompt = self._build_prompt(user_query, history, chat_history)
            response = self.generate(prompt)
            
            logger.info(f"Step {step+1} Agent Output: {response}")
            history += response + "\n"
            
            # Extract Thought for UI Trace
            if trace_callback:
                thought_lines = [line for line in response.split('\n') if line.startswith("Thought:")]
                if thought_lines:
                    trace_callback("thought", thought_lines[0].replace("Thought:", "").strip())
                else:
                    trace_callback("thought", "正在思考下一步...")
            
            if "Final Answer:" in response:
                final_ans = response.split("Final Answer:")[-1].strip()
                if trace_callback:
                    trace_callback("final_answer", "得出初步解答！")
                
                # Debate verification on complex/long responses
                if use_debate and len(final_ans) > 60:
                    if trace_callback:
                        trace_callback("debate", "偵測到長篇/複雜回答，啟動多智能體辯論驗證 ⚔️")
                    context_str = "\n\n".join(retrieved_contexts) if retrieved_contexts else final_ans
                    debate_res = self.debate_loop(user_query, context_str)
                    final_ans = debate_res.get("final_answer", final_ans)
                    if trace_callback:
                        for log in debate_res.get("debate_log", []):
                            trace_callback("debate", log)
                return final_ans
                
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", response)
            input_match = re.search(r"Action Input:\s*(\{.*\})", response, re.DOTALL)
            
            if action_match and input_match:
                try:
                    tool_name = action_match.group(1).strip()
                    tool_input_str = input_match.group(1).strip()
                    
                    if tool_name in self.tools:
                        tool_input = json.loads(tool_input_str)
                        logger.info(f"Executing tool {tool_name} with {tool_input}")
                        
                        if trace_callback:
                            trace_callback("action", f"啟動工具 **{tool_name}**，參數 `{tool_input_str}`")
                            
                        observation = self.tools[tool_name](**tool_input)
                        history += f"Observation: {observation}\n"
                        
                        if "search" in tool_name or tool_name == "search_paper" or tool_name == "search_arxiv":
                            retrieved_contexts.append(str(observation))
                        
                        if trace_callback:
                            obs_str = str(observation)
                            trace_callback("observation", obs_str[:300] + "..." if len(obs_str) > 300 else obs_str)
                            
                        logger.info(f"Observation: {observation}")
                    else:
                        history += f"Observation: Tool {tool_name} not found.\n"
                        if trace_callback:
                            trace_callback("error", f"工具 {tool_name} 不存在。")
                except Exception as e:
                    logger.error(f"Error parsing action: {e}")
                    history += f"Observation: Error parsing action or input. Ensure JSON format.\n"
                    if trace_callback:
                        trace_callback("error", f"指令解析錯誤: {e}")
            else:
                # Force exit if format is wrong
                if trace_callback:
                    trace_callback("final_answer", "直接給出解答 (或解析失敗，強制退出)")
                final_ans = response.strip()
                if use_debate and len(final_ans) > 60:
                    if trace_callback:
                        trace_callback("debate", "偵測到長篇/複雜回答，啟動多智能體辯論驗證 ⚔️")
                    context_str = "\n\n".join(retrieved_contexts) if retrieved_contexts else final_ans
                    debate_res = self.debate_loop(user_query, context_str)
                    final_ans = debate_res.get("final_answer", final_ans)
                    if trace_callback:
                        for log in debate_res.get("debate_log", []):
                            trace_callback("debate", log)
                return final_ans
                
        final_ans = "Final Answer: (達到最大思考步數，強制中止)"
        return final_ans

    def translate_to_english(self, text: str) -> str:
        """
        將繁體中文使用者的輸入翻譯為學術英文 (Translate & Confirm Workflow)
        以避免 LLM 陷入中文推理幻覺。
        """
        if not text.strip():
            return ""
        # 如果整句看起來幾乎都是英文，就不翻
        import re
        if len(re.findall(r'[a-zA-Z]', text)) > len(text) * 0.7:
            return text
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly professional Academic Search Query Optimizer.
The user will provide a Traditional Chinese query. Your task is to:
1. Translate it into precise, academic English suitable for a Computer Science semantic search.
2. If the query is simple or basic, expand it slightly by embedding 1-3 highly relevant academic keywords to significantly improve retrieval quality (e.g., if the user asks about "Transformer advantages", output "What are the advantages of transformers? self-attention mechanism, parallelization, long-range dependencies").
Return ONLY the final optimized English search query. Do NOT add any extra text, quotes, or explanations.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        # 使用低 temperature 來保持翻譯的一致性與精準度
        translation = self.generate(prompt, max_tokens=1024, temp=0.0)
        return translation.strip()

    # ==========================================
    # v2.1.3 Multi-Agent Architecture
    # ==========================================
    def router_agent(self, query: str, chat_history: list = None) -> str:
        """
        Router Agent: 決定是否需要檢索論文，或是可以直接回答。
        回傳: "SEARCH" 或 "DIRECT_ANSWER"
        """
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"
                
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Router Agent. Analyze the user's query and conversation history to decide if answering it requires searching the academic paper database (output SEARCH) or if it is a follow-up/general question that can be answered directly using the chat history context (output DIRECT_ANSWER).
Output ONLY one of the two keywords.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Conversation History:
{history_str}

Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        ans = self.generate(prompt, max_tokens=10, temp=0.0)
        return "SEARCH" if "SEARCH" in ans else "DIRECT_ANSWER"

    def meta_prompt_agent(self, query: str, raw_context: str) -> str:
        """
        Meta-Prompt Agent: (Rodrigues et al., 2024) 負責在送給主生成器前，過濾檢索出來的雜訊。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Meta-Prompting Filter Agent. 
You are given a user query and a large, noisy retrieved context.
Extract and summarize ONLY the sentences or facts from the context that are strictly relevant to answering the query. Discard all irrelevant noise.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}

Raw Context:
{raw_context[:2500]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        filtered_context = self.generate(prompt, max_tokens=512, temp=0.1)
        return filtered_context.strip()

    def evaluate_information_needs(self, query: str, context: str) -> str:
        """
        Meta-Prompt Agent 的延伸：評估目前檢索到的資訊是否足以完整回答問題。
        若不足，產生一句「Follow-up Query」用於第二次檢索；若充足，回傳 "SUFFICIENT"。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an Information Evaluation Agent.
Read the user's query and the retrieved context. Determine if the context contains enough information to fully answer the query.
If yes, output exactly "SUFFICIENT".
If no (e.g., missing comparison, missing definitions), generate a SINGLE follow-up search query to find the missing information.
Output ONLY "SUFFICIENT" or the follow-up query text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}
Context: {context[:2000]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        result = self.generate(prompt, max_tokens=60, temp=0.1)
        return result.strip()

    def synthesis_agent(self, query: str, clean_context: str, chat_history: list = None) -> str:
        """
        Synthesis Agent: 拿著最純淨的資料，撰寫最終解答。
        """
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"
                
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Synthesis Agent, an elite academic assistant.
Your task is to answer the user's query thoroughly using the provided filtered context and past conversation history.
CRITICAL INSTRUCTIONS:
1. Do NOT make up information or hallucinate facts that are not present in the context or chat history.
2. If neither the context nor the chat history contains sufficient information to answer the query, clearly state that the database lacks relevant documents for this specific question instead of forcing an answer.
3. Write your final response in Traditional Chinese (繁體中文) with a professional academic tone.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Conversation History:
{history_str}

Filtered Context:
{clean_context}

Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        final_answer = self.generate(prompt, max_tokens=4096, temp=0.2)
        return final_answer.strip()
    # ==========================================
    # v3.0 Competition Upgrades (Phase 1)
    # ==========================================
    def reformulate_query(self, query: str) -> str:
        """
        Query Reformulation: 將使用者模糊的自然語言問題改寫為精確的學術查詢關鍵字。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an Academic Query Reformulator.
Your task is to rewrite the user's natural language question into a highly precise, keyword-rich search query suitable for semantic vector search over computer science papers.
Output ONLY the rewritten query text in English, with no introductory or explanatory text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Original Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        reformulated = self.generate(prompt, max_tokens=64, temp=0.1)
        return reformulated.strip()

    def generate_hyde_document(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings): 根據查詢假想出一個包含解答的論文段落。
        這段落的語意向量能大幅提升搜尋命中率。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Computer Science Researcher.
Please write a short, highly technical paragraph (3-4 sentences) that directly answers the user's query as if it were an excerpt from a published academic paper. Include relevant terminology and formulas if applicable.
Output ONLY the hypothetical paragraph, no conversational filler.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        hyde_doc = self.generate(prompt, max_tokens=150, temp=0.2)
        return hyde_doc.strip()

    def explain_result(self, query: str, chunk_text: str) -> str:
        """
        Explainable Result Generation: 解釋為何這個檢索結果與使用者查詢相關。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an Explainable AI Assistant.
The user asked a query, and the system retrieved a specific paragraph from an academic paper.
Explain briefly (in 1-2 sentences in Traditional Chinese) WHY this retrieved paragraph is highly relevant to the user's query and what specific information it provides.
Output ONLY the explanation, without any introductory phrases.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}
Retrieved Paragraph: {chunk_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        explanation = self.generate(prompt, max_tokens=100, temp=0.1)
        return explanation.strip()

    def self_reflect(self, query: str, draft_answer: str, context: str) -> str:
        """
        Self-Reflection Agent: 審視 Synthesis Agent 產生的草稿是否有事實錯誤、矛盾或缺失。
        如果有，則修正並回傳最終版本；如果完美，則直接回傳原草稿。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Self-Reflection Agent. Your job is to review a draft answer for an academic query based on the provided context.
CRITICAL INSTRUCTIONS:
1. Check if the draft hallucinated any facts not present in the context.
2. Check if the draft forced an answer when the context was actually insufficient.
3. If the draft hallucinated or forced an answer without evidence, overwrite it with a polite apology stating the missing info.
4. If the draft is accurate and supported by the context, just output the draft as is.
5. Output the final version in Traditional Chinese.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context[:2000]}

Query: {query}

Draft Answer:
{draft_answer}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        final_reflected_answer = self.generate(prompt, max_tokens=4096, temp=0.1)
        return final_reflected_answer.strip()

    def self_critique_and_refine(self, query: str, draft_answer: str, ragas_scores: dict) -> dict:
        """
        Reflexion Agent (Phase 2): 透過 RAGAS 量化分數自我批判，並決定是否要重新搜尋。
        """
        faithfulness = ragas_scores.get("faithfulness", 0.0)
        relevance = ragas_scores.get("relevance", 0.0)
        
        # 門檻判斷
        if faithfulness >= 0.7 and relevance >= 0.7:
            return {"needs_refinement": False, "new_query": ""}
            
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Reflexion Agent. Your previous answer was evaluated and scored poorly.
Faithfulness score (is it hallucinating?): {faithfulness:.2f} / 1.0
Relevance score (does it answer the query?): {relevance:.2f} / 1.0

Analyze WHY the answer was poor. Then, generate ONE single highly-targeted new search query (in English) to retrieve better context.
Output ONLY the new search query, without any explanations.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Original Query: {query}
Poor Answer: {draft_answer[:500]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        new_query = self.generate(prompt, max_tokens=64, temp=0.2).strip()
        return {"needs_refinement": True, "new_query": new_query}

    # ==========================================
    # v2.5 Multi-Agent Debate Deep Expansion (ChatEval / Self-Refine)
    # ==========================================
    def fact_checker_agent(self, query: str, draft: str, context: str) -> str:
        """
        Fact-Checker Agent: 嚴苛查核幻覺。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a ruthless Fact-Checker Agent. Your sole responsibility is to audit a draft answer against the provided Context.
INSTRUCTIONS:
1. Identify any claims, numbers, or entities in the draft that are NOT explicitly stated in the context (Hallucinations).
2. If the draft is completely faithful to the context, output EXACTLY "<APPROVE>".
3. If there are hallucinations, output "<REJECT>" followed by a detailed list of the unverified claims.
4. You MUST write your critique entirely in ENGLISH.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context[:3000]}

User Query: {query}

Draft:
{draft}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return self.generate(prompt, max_tokens=512, temp=0.1)

    def logic_critic_agent(self, query: str, draft: str, context: str) -> str:
        """
        Logic Critic Agent: 審查邏輯與完整性。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict Logic and Completeness Critic. Your job is to ensure the draft fully answers the User Query in a logical manner based on the Context.
INSTRUCTIONS:
1. Check if the draft actually answers all parts of the User Query.
2. Check if the logical flow makes sense and transitions smoothly.
3. If the draft fully answers the query with sound logic, output EXACTLY "<APPROVE>".
4. If it misses parts of the query or lacks logic, output "<REJECT>" followed by specific improvements needed.
5. You MUST write your critique entirely in ENGLISH.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context[:3000]}

User Query: {query}

Draft:
{draft}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return self.generate(prompt, max_tokens=512, temp=0.1)

    def judge_agent(self, query: str, fact_critique: str, logic_critique: str) -> str:
        """
        Judge Agent: 主審官，彙整兩位 Critic 的意見。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the Chief Judge Agent. You receive critiques from a Fact-Checker and a Logic Critic regarding a draft answer.
INSTRUCTIONS:
1. If BOTH critics output <APPROVE>, you must output EXACTLY "<PASS>".
2. If any critic outputs <REJECT>, you must synthesize their critiques into a single, clear "Consolidated Feedback" list for the Generator to fix.
3. Do NOT write a new draft yourself. Only provide the feedback instructions.
4. You MUST write your consolidated feedback entirely in ENGLISH.
<|eot_id|><|start_header_id|>user<|end_header_id|>
User Query: {query}

Fact-Checker Feedback:
{fact_critique}

Logic Critic Feedback:
{logic_critique}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return self.generate(prompt, max_tokens=512, temp=0.1)

    def generator_rewrite_agent(self, query: str, context: str, previous_draft: str, judge_feedback: str) -> str:
        """
        Generator Rewrite Agent: 根據 Judge 的指示重新改寫草稿。
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Generator Agent. Your previous draft was rejected by the Chief Judge.
You must rewrite your answer to completely resolve the Judge's issues, using ONLY the provided context.
Output the finalized, polished answer in Traditional Chinese (繁體中文).
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context[:3000]}

User Query: {query}

Your Previous Draft:
{previous_draft}

Judge's Feedback (Fix these!):
{judge_feedback}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return self.generate(prompt, max_tokens=1024, temp=0.2)

    def debate_loop(self, query: str, context: str, max_rounds: int = 3) -> dict:
        """
        Orchestrator: 控制 4-Agent 架構的多回合辯論。
        回傳最終答案與辯論 Log。
        """
        debate_log = []
        
        # Round 1: Initial Draft
        debate_log.append("🤖 **[Generator]** Drafting initial response based on retrieved context...")
        current_draft = self.synthesis_agent(query, context)
        
        for round_num in range(1, max_rounds + 1):
            debate_log.append(f"--- **[Round {round_num}]** ---")
            
            # Parallel Critique (simulated sequentially)
            debate_log.append(f"🕵️‍♂️ **[Fact-Checker]** Auditing for hallucinations...")
            fact_res = self.fact_checker_agent(query, current_draft, context)
            if "<APPROVE>" in fact_res.upper():
                debate_log.append("✅ **[Fact-Checker]** <APPROVE>. No hallucinations found.")
            else:
                debate_log.append(f"❌ **[Fact-Checker]** <REJECT>.\n{fact_res}")
                
            debate_log.append(f"🧐 **[Logic Critic]** Auditing for completeness and logic...")
            logic_res = self.logic_critic_agent(query, current_draft, context)
            if "<APPROVE>" in logic_res.upper():
                debate_log.append("✅ **[Logic Critic]** <APPROVE>. Logically sound and complete.")
            else:
                debate_log.append(f"❌ **[Logic Critic]** <REJECT>.\n{logic_res}")
                
            # Judgment
            debate_log.append(f"👨‍⚖️ **[Judge]** Synthesizing critiques...")
            judge_res = self.judge_agent(query, fact_res, logic_res)
            
            if "<PASS>" in judge_res.upper() or ("<APPROVE>" in fact_res.upper() and "<APPROVE>" in logic_res.upper()):
                debate_log.append("⚖️ **[Judge]** <PASS>. The draft has survived multi-perspective scrutiny.")
                break
            else:
                debate_log.append(f"⚠️ **[Judge]** <REVISION_REQUIRED>. Consolidated Feedback:\n{judge_res}")
                if round_num < max_rounds:
                    debate_log.append(f"🔄 **[Generator]** Rewriting draft based on Judge's feedback...")
                    print(f"\n--- [Round {round_num}] Generator starts ---", flush=True)
                    current_draft = self.generator_rewrite_agent(query, context, current_draft, judge_res)
                    print(f"--- [Round {round_num}] Generator completed ---", flush=True)
                else:
                    debate_log.append("⚠️ **[Orchestrator]** Max debate rounds reached. Returning best effort draft.")
                    
        return {
            "final_answer": current_draft,
            "debate_log": debate_log
        }

    def generate_concept_tree(self, concept: str) -> dict:
        """
        Generate a concept tree (upstream, core, downstream) for Tree-based Relationship Recommendation.
        v2.3.1: Modified to generate Hypothetical Document Embeddings (HyDE) descriptions instead of keyword arrays.
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI academic researcher. Analyze the given computer science concept and output a relationship tree in STRICT JSON format.
Instead of providing keywords, you MUST write a "Hypothetical Document Description" (HyDE) for each branch. This means writing a 50-word hypothetical abstract that describes what a paper in that category would sound like.

The JSON must contain exactly three keys (and the values MUST be strings, not arrays):
- "upstream": A 50-word hypothetical abstract for a foundational or prerequisite paper that led to the development of this concept.
- "core": A 50-word hypothetical abstract for a paper that explains the core mechanism or architecture of this concept.
- "downstream": A 50-word hypothetical abstract for a paper that explores downstream applications, extensions, or successors building upon this concept.

Output ONLY valid JSON. Do NOT wrap in markdown blocks, do NOT include explanations.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Concept: {concept}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        response = self.generate(prompt, max_tokens=512, temp=0.2)
        import json
        try:
            # Attempt to parse json. Sometimes the LLM might still wrap in markdown or add text.
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"Failed to parse concept tree JSON: {e}")
            # Fallback HyDE strings
            return {
                "upstream": f"This paper explores the foundational theories and prerequisite mathematics that led to the development of {concept}, focusing on early theoretical frameworks.",
                "core": f"This paper details the core architecture and mathematical mechanism of {concept}, providing a comprehensive analysis of its internal workings.",
                "downstream": f"This paper investigates novel applications and downstream extensions of {concept}, demonstrating its effectiveness in solving complex modern tasks."
            }

    def generate_grounded_concept_tree(self, concept: str, local_context: str) -> dict:
        """
        v2.3.2: RAG-Grounded Tree Generation.
        Strictly relies on the local_context to extract upstream, core, and downstream narratives.
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI academic researcher. Analyze the given computer science concept based ONLY on the provided "Local Context".
Output a relationship tree in STRICT JSON format.

The JSON must contain exactly three keys (and the values MUST be strings):
- "upstream": A 50-word summary of the foundational or prerequisite origins of the concept AS DESCRIBED IN THE CONTEXT. If the context does NOT mention upstream origins, output EXACTLY: "本地文獻庫缺乏相關溯源資訊"
- "core": A 50-word summary of the core mechanism of the concept AS DESCRIBED IN THE CONTEXT.
- "downstream": A 50-word summary of downstream applications or extensions AS DESCRIBED IN THE CONTEXT. If not mentioned, output EXACTLY: "本地文獻庫缺乏後續應用資訊"

Output ONLY valid JSON. Do NOT wrap in markdown blocks, do NOT hallucinate outside the context.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Local Context:
{local_context[:3000]}

Concept: {concept}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        response = self.generate(prompt, max_tokens=512, temp=0.1)
        import json
        try:
            text = response.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"Failed to parse grounded tree JSON: {e}")
            return {
                "upstream": "本地文獻庫缺乏相關溯源資訊",
                "core": concept,
                "downstream": "本地文獻庫缺乏後續應用資訊"
            }
