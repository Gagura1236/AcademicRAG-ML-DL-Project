import streamlit as st
import os
import sys
import time
import json
import config
from src.evaluation import RAGEvaluationSuite
from src.finetune import AcademicLoRATuner
from src.locales import get_text

# LoRA 設定檔路徑（專案級，儲存在 data/ 下）
LORA_CONFIG_FILE = os.path.join(config.DATA_DIR, "lora_config.json")

def _load_lora_config():
    """讀取已儲存的 LoRA 客製化設定，若無則返回全域預設值。"""
    if os.path.exists(LORA_CONFIG_FILE):
        try:
            with open(LORA_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "epochs": getattr(config, "DEFAULT_LORA_EPOCHS", 3),
        "rank": getattr(config, "DEFAULT_LORA_RANK", 8),
        "lr": getattr(config, "DEFAULT_LORA_LR", 3e-5),
        "margin": getattr(config, "DEFAULT_LORA_MARGIN", 0.3),
    }

def format_lr(lr_val):
    """將 3e-05 轉換為 3 × 10⁻⁵ 格式 (Unicode 上標)"""
    s = f"{lr_val:.0e}"
    if "e" in s:
        base, exp = s.split("e")
        exp_str = str(int(exp))
        trans = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        exp_super = exp_str.translate(trans)
        return f"{base} × 10{exp_super}"
    return str(lr_val)

def _save_lora_config(epochs, rank, lr, margin):
    """將客製化 LoRA 設定儲存到磁碟，讓後續評估也能讀取。"""
    try:
        with open(LORA_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"epochs": epochs, "rank": rank, "lr": lr, "margin": margin}, f, indent=4)
    except Exception as e:
        st.error(f"儲存 LoRA 設定失敗: {e}")

def render(rag_engine):
    lang = st.session_state.get("lang", "zh")
    st.subheader(get_text("menu_step6", lang))

    if lang == "zh":
        st.write("在此頁面，您可以對現有的檢索系統進行學術級的 **ROC-AUC** 與 **Confidence Margin** 效能評估，並客製化 LoRA 超參數後啟動微調。")
    else:
        st.write("Run academic evaluation (ROC-AUC & Confidence Margin) and customize LoRA hyperparameters before fine-tuning.")

    # ── 載入當前已套用的 LoRA 設定 ──────────────────────────────────────
    saved_cfg = _load_lora_config()

    # 同步到 session state（首次或磁碟值優先）
    if "lora_epochs" not in st.session_state:
        st.session_state["lora_epochs"] = saved_cfg["epochs"]
    if "lora_r" not in st.session_state:
        st.session_state["lora_r"] = saved_cfg["rank"]
    if "lora_lr_select" not in st.session_state:
        st.session_state["lora_lr_select"] = saved_cfg["lr"]
    if "lora_margin" not in st.session_state:
        st.session_state["lora_margin"] = saved_cfg["margin"]

    eval_suite = RAGEvaluationSuite(rag_engine=rag_engine, project_name=st.session_state.project_name)

    # ── 0. 目前套用的超參數面板（全寬，一眼看清）───────────────────────
    if lang == "zh":
        st.info(
            f"⚙️ **目前套用至本專案的 LoRA 超參數** "
            f"(修改後請點下方「套用至專案」才會生效於評估與微調)：\n\n"
            f"| 參數 | 目前值 | 說明 |\n"
            f"|------|--------|------|\n"
            f"| 訓練輪數 (Epochs) | **{saved_cfg['epochs']}** | 越高 (>8) 收斂越完整但耗時且易過擬合，建議 3~7 |\n"
            f"| LoRA 矩陣秩 (Rank r) | **{saved_cfg['rank']}** | 越高 (>16) 模型能力強但極耗記憶體，過低 (≤4) 學習力不足 |\n"
            f"| 學習率 (LR) | **{format_lr(saved_cfg['lr'])}** | 太高 (≥10⁻⁴) 易震盪崩潰，太低 (≤10⁻⁵) 收斂極慢 |\n"
            f"| Contrastive Margin | **{saved_cfg['margin']}** | 高閾值 (>0.5) 讓對比學習變極難，過低 (<0.2) 則難以區分正負樣本 |"
        )

    # ── 過/欠擬合文獻參考指南（可收合）────────────────────────────────
    if lang == "zh":
        with st.expander("📚 過擬合 / 欠擬合 參數判斷指南（文獻根據）", expanded=False):
            st.markdown("""
            > 參考文獻：
            > - **ROC-AUC 0.85 門檻**：Mandrekar J. N. (2010). *Receiver operating characteristic curve in diagnostic test assessment*.
            > - **LoRA Rank**：Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
            > - **Epochs**：Nogueira & Cho (2019). *Passage Re-ranking with BERT*.
            > - **LR / Margin**：He et al. (2021) *DeBERTa* 及 Chen et al. (2020) *SimCSE*.

            ---
            ### 🟠 欠擬合 (Underfitting) — 模型沒學到足夠特徵
            **症狀**：訓練集與驗證集 Loss 都偏高；ROC-AUC < 0.6；Margin < 0.1

            | 參數 | 欠擬合跡象 | 建議調整方向 |
            |------|-----------|------------|
            | **Epochs** | ≤ 2 輪，Loss 仍高 | 增加至 5~10 輪 |
            | **LoRA Rank r** | r = 4，Loss 不收斂 | 提升到 r = 8 或 r = 16 |
            | **Learning Rate** | 1 × 10⁻⁵ 時收斂極慢 | 調升至 3 × 10⁻⁵ ~ 5 × 10⁻⁵ |
            | **Contrastive Margin** | 0.1 以下，正負樣本分不開 | 提升至 0.3 ~ 0.5 |

            > 📖 *Hu et al. (2022)*：LoRA 在 r=4 時已具基本能力，但 NLP Reranking 任務通常需要 r≥8 才能充分捕捉語意差異。

            ---
            ### 🔴 過擬合 (Overfitting) — 模型記住訓練集而無法泛化
            **症狀**：訓練 Loss 極低但驗證 AUC 不升反降；正樣本分數 > 0.99

            | 參數 | 過擬合跡象 | 建議調整方向 |
            |------|-----------|------------|
            | **Epochs** | > 8 輪後驗證 AUC 下降 | 降回 3~5 輪；或啟用 Early Stopping |
            | **LoRA Rank r** | r = 32 時，小資料集容易記憶 | 降至 r = 8，並搭配 Dropout |
            | **Learning Rate** | 1 × 10⁻⁴ 以上，Loss 震盪後驟降 | 降至 3 × 10⁻⁵；考慮加入 Warmup |
            | **Contrastive Margin** | > 0.7，負樣本被強迫推得太遠 | 降至 0.3 ~ 0.5 |

            > 📖 *Nogueira & Cho (2019)*：Cross-Encoder 在小語料庫 (< 1000 pairs) 訓練超過 5 epochs 後，
            > 驗證 nDCG@10 通常開始下滑，建議搭配 Early Stopping (patience=2)。

            ---
            ### ✅ 健康收斂區間（本系統建議值）
            | 參數 | 建議範圍 | 預設值 |
            |------|---------|--------|
            | Epochs | **3 ~ 7** | 3 |
            | LoRA Rank r | **8 ~ 16** | 8 |
            | Learning Rate | **3e-5 ~ 5e-5** | 3e-5 |
            | Contrastive Margin | **0.25 ~ 0.45** | 0.3 |
            | ROC-AUC (目標) | **> 0.75** | — |
            | Margin (Logits) | **> 0.3** | — |

            > 📖 *He et al. (2021) DeBERTa*：在 MSMARCO Reranking 任務上，
            > 學習率 3e-5 配合 Linear Warmup (10% steps) 是最穩健的組合，
            > 在小 batch size (16~32) 設定下能避免梯度噪聲過大。
            >
            > 📖 *Chen et al. (2020) SimCSE*：Contrastive Temperature / Margin 設定在 0.05~0.3 之間時，
            > 正負樣本的語意空間分離效果最佳；過大的 margin 反而造成訓練不穩定。

            💡 **本系統自動保護**：`patience=2` 的 Early Stopping 已內建，當連續 2 個 Epoch 驗證 AUC 沒有提升時，訓練會自動停止，避免過擬合。
            """)
    else:
        with st.expander("📚 Overfitting / Underfitting Parameter Guide (Literature-Based)", expanded=False):
            st.markdown("""
            > References:
            > - **ROC-AUC 0.85 Threshold**: Mandrekar J. N. (2010). *Receiver operating characteristic curve in diagnostic test assessment*.
            > - **LoRA Rank**: Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
            > - **Epochs**: Nogueira & Cho (2019). *Passage Re-ranking with BERT*.
            > - **LR / Margin**: He et al. (2021) *DeBERTa* & Chen et al. (2020) *SimCSE*.

            ---
            ### 🟠 Underfitting — Model fails to learn sufficient features
            **Symptoms**: Both train & val Loss remain high; ROC-AUC < 0.6; Margin < 0.1

            | Parameter | Underfitting Sign | Recommended Fix |
            |-----------|------------------|----------------|
            | **Epochs** | ≤ 2, Loss still high | Increase to 5~10 |
            | **LoRA Rank r** | r = 4, Loss plateau | Increase to r = 8 or 16 |
            | **Learning Rate** | 1 × 10⁻⁵ makes it too slow | Increase to 3 × 10⁻⁵ ~ 5 × 10⁻⁵ |
            | **Contrastive Margin** | < 0.1, fails to separate | Increase to 0.3 ~ 0.5 |

            > 📖 *Hu et al. (2022)*: LoRA at r=4 provides basic capacity, but NLP Reranking tasks typically need r≥8 to fully capture semantic nuances.

            ---
            ### 🔴 Overfitting — Model memorizes training set, fails to generalize
            **Symptoms**: Train Loss near zero but val AUC stagnates or drops; positive scores > 0.99

            | Parameter | Overfitting Sign | Recommended Fix |
            |-----------|-----------------|----------------|
            | **Epochs** | > 8 causes val AUC drop | Reduce to 3~5; enable Early Stopping |
            | **LoRA Rank r** | r = 32 easily memorizes | Reduce to r = 8, use Dropout |
            | **Learning Rate** | > 1 × 10⁻⁴ causes loss oscillation | Decrease to 3 × 10⁻⁵; add Warmup |
            | **Contrastive Margin** | > 0.7 forces negatives too far | Reduce to 0.3 ~ 0.5 |

            > 📖 *Nogueira & Cho (2019)*: Cross-Encoders trained on small corpora (< 1000 pairs) for > 5 epochs
            > typically show declining validation nDCG@10. Early Stopping (patience=2) is strongly recommended.

            ---
            ### ✅ Healthy Convergence Range (System Defaults)
            | Parameter | Recommended Range | Default |
            |-----------|------------------|---------|
            | Epochs | **3 ~ 7** | 3 |
            | LoRA Rank r | **8 ~ 16** | 8 |
            | Learning Rate | **3e-5 ~ 5e-5** | 3e-5 |
            | Contrastive Margin | **0.25 ~ 0.45** | 0.3 |
            | ROC-AUC (target) | **> 0.75** | — |
            | Logit Margin (target) | **> 0.3** | — |

            > 📖 *He et al. (2021) DeBERTa*: On MSMARCO Reranking, LR=3e-5 with Linear Warmup (10% steps)
            > is the most robust combination. Small batch sizes (16~32) help avoid gradient noise amplification.
            >
            > 📖 *Chen et al. (2020) SimCSE*: Contrastive temperature/margin in the 0.05~0.3 range yields
            > optimal semantic space separation. Margins that are too large cause training instability.

            ---
            💡 **Built-in protection**: `patience=2` Early Stopping is already enabled. Training auto-stops if val AUC doesn't improve for 2 consecutive epochs, preventing overfitting automatically.
            """)

    col1, col2 = st.columns([1, 1])


    # ══════════════════════════════════════════════════════════════
    # COL 1: 效能評估
    # ══════════════════════════════════════════════════════════════
    with col1:
        st.write("### 📊 1. " + ("系統診斷報告" if lang == "zh" else "System Diagnostics"))

        if lang == "zh":
            st.caption(f"📌 評估使用目前套用的超參數：Epochs={saved_cfg['epochs']}、Rank={saved_cfg['rank']}、LR={format_lr(saved_cfg['lr'])}")
            st.warning(
                "⚠️ **重要說明：修改超參數並不會立即影響評估分數！**\n\n"
                "正確操作流程：\n"
                "1️⃣ 在下方『調整 LoRA 超參數』區塊修改數值\n"
                "2️⃣ 點擊『✅ 套用至專案』儲存設定\n"
                "3️⃣ 點擊『🚀 啟動 LoRA 微調』重新訓練模型（需要一段時間）\n"
                "4️⃣ 微調完成後再點『🔄 重新評估』才會看到 AUC 分數變化"
            )
        else:
            st.caption(f"📌 Evaluation uses currently applied params: Epochs={saved_cfg['epochs']}, Rank={saved_cfg['rank']}, LR={format_lr(saved_cfg['lr'])}")
            st.warning(
                "⚠️ **Important: Changing hyperparameters does NOT immediately affect evaluation scores!**\n\n"
                "Correct workflow:\n"
                "1️⃣ Adjust values in the 'LoRA Hyperparameter' section below\n"
                "2️⃣ Click '✅ Apply to Project' to save settings\n"
                "3️⃣ Click '🚀 Start LoRA Fine-Tuning' to retrain the model (takes time)\n"
                "4️⃣ Only after training completes will '🔄 Re-Evaluate' show updated AUC scores"
            )

        test_set_type = st.selectbox(
            "📋 選擇評估測試集 / Select Evaluation Test Set",
            options=["Default (6 queries)", "Classic 39 Papers Benchmark (15 queries)", "Dynamic LLM Generator"],
            index=1,
            help="選擇要使用的評估資料集種類。專案介紹推薦使用 39 篇經典論文基準測試。" if lang == "zh" else "Select the evaluation test set to use. Classic 39 Papers Benchmark is recommended for demo."
        )
        
        use_llm_eval = (test_set_type == "Dynamic LLM Generator")
        benchmark_mode = (test_set_type == "Classic 39 Papers Benchmark (15 queries)")

        if st.session_state.diagnostic_report is None:
            st.info("💡 " + ("系統目前尚未執行效能評估。請點擊下方按鈕以開始診斷！" if lang == "zh" else "No evaluation report found. Click below to run diagnostic sweep."))
            if st.button("📊 " + ("執行效能評估 / Run Diagnostics" if lang == "zh" else "Run Diagnostics")):
                with st.spinner("🔍 " + ("正在進行語意特徵掃描與 ROC-AUC 信心模擬..." if lang == "zh" else "Scanning semantic features & simulating ROC-AUC...")):
                    report = eval_suite.evaluate_retrieval(top_k=3, use_llm=use_llm_eval, benchmark_mode=benchmark_mode)
                    if not report or "diagnosis" not in report:
                        st.error("❌ " + ("知識庫為空或評估失敗，無法產生診斷報告。請先在「檔案處理」頁新增論文！" if lang == "zh" else "Failed to generate report. Add documents in Step 1 first!"))
                    else:
                        st.session_state.diagnostic_report = report
                        st.session_state.before_report = report.copy()
                        st.rerun()
        else:
            report = st.session_state.diagnostic_report
            diag = report["diagnosis"]

            if diag["status"] == "健康 (Balanced)":
                st.success(f"✅ **{'系統健康狀態' if lang == 'zh' else 'System Status'}：{diag['status']}**")
            else:
                st.warning(f"⚠️ **{'系統健康狀態' if lang == 'zh' else 'System Status'}：{diag['status']}**")

            st.markdown(f"**{'學術級檢索識別率' if lang == 'zh' else 'Retrieval ROC-AUC'} (ROC-AUC):** `{diag['auc']:.4f}`")
            st.markdown(f"**{'正負樣本區分 Margin' if lang == 'zh' else 'Pos/Neg Margin'} (Logits):** `{diag['margin']:.4f}` (pos: `{diag['mean_pos_score']:.2f}` / neg: `{diag['mean_neg_score']:.2f}`)")
            st.write(f"💡 **{'診斷報告與調優建議' if lang == 'zh' else 'Diagnosis & Recommendation'}**：\n{diag['recommendation']}")

            if st.button("🔄 " + ("重新評估 / Re-run Diagnostics" if lang == "zh" else "Re-run Diagnostics")):
                with st.spinner("🔍 " + ("正在重新評估中..." if lang == "zh" else "Re-evaluating...")):
                    report = eval_suite.evaluate_retrieval(top_k=3, use_llm=use_llm_eval, benchmark_mode=benchmark_mode)
                    if report and "diagnosis" in report:
                        st.session_state.diagnostic_report = report
                        adapter_exists = os.path.exists(os.path.join(config.DATA_DIR, "lora_adapters", "default_adapter", "adapter_config.json"))
                        if not adapter_exists:
                            st.session_state.before_report = report.copy()
                    st.rerun()

    # ══════════════════════════════════════════════════════════════
    # COL 2: 超參數設定 + LoRA 微調
    # ══════════════════════════════════════════════════════════════
    with col2:
        st.write("### ⚙️ 2. " + ("超參數設定 & LoRA 微調" if lang == "zh" else "Hyperparameters & LoRA Fine-Tuning"))

        if lang == "zh":
            st.write("調整下方超參數，點擊 **「套用至專案」** 儲存設定後，效能評估與微調都會使用新值。")
        else:
            st.write("Adjust hyperparameters and click **'Apply to Project'** to save. Both evaluation and fine-tuning will use the new values.")

        if st.session_state.diagnostic_report is None:
            st.warning("⚠️ " + ("請先在左側完成『檢索效能評估與診斷』！" if lang == "zh" else "Please run baseline diagnostics first!"))
        else:
            # ── 超參數調整區 ─────────────────────────────
            epochs_input = st.slider(
                ("訓練輪數 / Epochs" if lang == "zh" else "Training Epochs"),
                min_value=3, max_value=15, value=st.session_state["lora_epochs"], step=1,
                help=("Epoch 越多模型越完整收斂，但訓練時間線性增加。預設 3。" if lang == "zh" else "More epochs = better convergence, longer training. Default: 3.")
            )

            lora_r_input = st.slider(
                ("LoRA 矩陣秩 / Rank (r)" if lang == "zh" else "LoRA Matrix Rank (r)"),
                min_value=4, max_value=32, value=st.session_state["lora_r"], step=4,
                help=("秩越高，模型調整能力越強，但消耗記憶體越多。建議 8~16。預設 8。" if lang == "zh" else "Higher rank = more capacity, more memory. Recommended: 8~16. Default: 8.")
            )

            lr_map = {
                ("極慢學習 / Very Slow (5e-6)" if lang == "zh" else "Very Slow (5e-6)"): 5e-6,
                ("慢速學習 / Slow (1e-5)" if lang == "zh" else "Slow (1e-5)"): 1e-5,
                ("稍慢學習 / Mild (2e-5)" if lang == "zh" else "Mild (2e-5)"): 2e-5,
                ("標準速度 / Normal (3e-5)" if lang == "zh" else "Normal (3e-5)"): 3e-5,
                ("快速學習 / Fast (5e-5)" if lang == "zh" else "Fast (5e-5)"): 5e-5,
            }
            # 找到目前儲存的 lr 對應的 label
            current_lr = st.session_state["lora_lr_select"]
            default_lr_label = list(lr_map.keys())[3]  # Normal (3e-5)
            for label, val in lr_map.items():
                if abs(val - current_lr) < 1e-10:
                    default_lr_label = label
                    break
            lr_label = st.selectbox(
                ("學習率 / Learning Rate" if lang == "zh" else "Learning Rate"),
                options=list(lr_map.keys()),
                index=list(lr_map.keys()).index(default_lr_label),
                help=("學習率決定每次參數更新幅度。太高易震盪，太低收斂慢。" if lang == "zh" else "Controls per-step parameter update size. Too high = oscillation, too low = slow.")
            )
            lr_input = lr_map[lr_label]

            margin_input = st.slider(
                ("Contrastive Margin" if lang == "zh" else "Contrastive Margin"),
                min_value=0.1, max_value=0.8, value=float(st.session_state["lora_margin"]), step=0.05,
                help=("對比學習中正負樣本的最小分距。越大對模型要求越高。預設 0.3。" if lang == "zh" else "Min score gap between pos/neg pairs in contrastive learning. Default: 0.3.")
            )

            # ── 套用至專案 ──────────────────────────────
            params_changed = (
                epochs_input != saved_cfg["epochs"] or
                lora_r_input != saved_cfg["rank"] or
                abs(lr_input - saved_cfg["lr"]) > 1e-10 or
                abs(margin_input - saved_cfg["margin"]) > 1e-4
            )

            if params_changed:
                st.warning("⚠️ " + ("您已修改超參數，請點擊下方「套用至專案」才會讓評估與微調生效！" if lang == "zh" else "Parameters changed! Click 'Apply to Project' to activate them for evaluation and fine-tuning."))

            col_apply, col_reset = st.columns([1, 1])
            with col_apply:
                if st.button("✅ " + ("套用至專案 / Apply to Project" if lang == "zh" else "Apply to Project"), type="primary"):
                    _save_lora_config(epochs_input, lora_r_input, lr_input, margin_input)
                    st.session_state["lora_epochs"] = epochs_input
                    st.session_state["lora_r"] = lora_r_input
                    st.session_state["lora_lr_select"] = lr_input
                    st.session_state["lora_margin"] = margin_input
                    st.success("✅ " + ("超參數已套用！效能評估與微調將使用新設定。" if lang == "zh" else "Hyperparameters applied! Evaluation and fine-tuning will use the new settings."))
                    st.rerun()
            with col_reset:
                if st.button("🔄 " + ("重置為預設值" if lang == "zh" else "Reset to Defaults")):
                    _save_lora_config(config.DEFAULT_LORA_EPOCHS, config.DEFAULT_LORA_RANK, config.DEFAULT_LORA_LR, config.DEFAULT_LORA_MARGIN)
                    st.session_state["lora_epochs"] = config.DEFAULT_LORA_EPOCHS
                    st.session_state["lora_r"] = config.DEFAULT_LORA_RANK
                    st.session_state["lora_lr_select"] = config.DEFAULT_LORA_LR
                    st.session_state["lora_margin"] = config.DEFAULT_LORA_MARGIN
                    st.info("🔄 " + ("已重置為預設值！" if lang == "zh" else "Reset to defaults!"))
                    st.rerun()

            st.write("---")
            st.write("### 🚀 " + ("啟動 LoRA 微調" if lang == "zh" else "Start LoRA Fine-Tuning"))

            if lang == "zh":
                st.write("使用 Hugging Face `peft` 庫，針對 Cross-Encoder 進行 Low-Rank Adaptation (LoRA) 微調。訓練包含硬負樣本 (Hard Negatives) 與 L2 Regularization。")
            else:
                st.write("Tune the Cross-Encoder using Hugging Face PEFT/LoRA with hard negatives and L2 regularization.")

            st.write("### 🧠 " + ("高階數據擴增" if lang == "zh" else "Data Augmentation"))
            
            llm_help = (
                "建議開啟！系統將自動調用本地大語言模型 (LLM) 研讀論文，並自動生成約 60 筆高品質的學術問答對作為訓練資料，能顯著提升對比學習的語意精度。缺點：需花費額外 1~2 分鐘進行模型生成推理。"
                if lang == "zh" else
                "Recommended! Uses local LLM to read chunks and synthesize ~60 high-quality academic Q&A pairs. Greatly improves semantic precision for contrastive learning but adds 1-2 mins of generation overhead."
            )
            mps_help = (
                "強烈建議開啟！這會調用您 Mac 的 M 系列晶片 GPU 進行 PyTorch 訓練。若關閉此選項，模型將在 CPU 上跑，速度會慢上 5~10 倍。"
                if lang == "zh" else
                "Strongly recommended! Uses local M-series GPU for training acceleration. If turned off, model trains on CPU which is 5-10x slower."
            )

            use_llm_finetune = st.toggle(
                "🚀 " + ("啟用 LLM 生成反向學術考題 (大幅提升準確率)" if lang == "zh" else "Enable LLM adversarial Q-gen (boosts accuracy)"),
                value=False,
                help=llm_help
            )

            import torch
            mps_available = torch.backends.mps.is_available()
            if mps_available:
                use_mps_tuner = st.toggle(
                    "⚡️ " + ("啟用 Apple Silicon GPU/MPS 訓練加速 (FP32 精度)" if lang == "zh" else "Enable Apple Silicon MPS acceleration (FP32)"), 
                    value=False,
                    help=mps_help
                )
            else:
                use_mps_tuner = False

            hpo_help = (
                "建議開啟！開啟後，系統會自動在微調前運行一輪「超參數網格搜尋」，測試不同的學習率與 Contrastive Margin 的排列組合，尋找能最大化驗證集 AUC 分數的組合，並自動將最優解套用至本次訓練。"
                if lang == "zh" else
                "Recommended! If enabled, runs a fast HPO sweep across learning rate and margin candidates, dynamically selects the combination that maximizes validation AUC, and automatically trains the final model using these parameters."
            )
            use_hpo = st.toggle(
                "📈 " + ("啟用 Auto-HPO 最佳超參數自動調優 (學術加分推薦)" if lang == "zh" else "Enable Auto-HPO grid search (recommended)"),
                value=False,
                help=hpo_help
            )

            run_ui_tests_checkbox = st.checkbox(
                get_text("eval_auto_test_label", lang),
                value=False,
                help=get_text("eval_auto_test_help", lang),
                key="run_ui_tests_checkbox"
            )

            adapter_name = st.text_input(
                "💾 " + ("微調模型儲存名稱 (Adapter Name)" if lang == "zh" else "Adapter Save Name"),
                value="default_adapter"
            )
            adapter_path = os.path.join(config.DATA_DIR, "lora_adapters", adapter_name)
            is_finetuned = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

            if is_finetuned:
                st.info(f"🎉 **{'偵測到本地已存在名為' if lang == 'zh' else 'Detected existing adapter:'} '{adapter_name}' {'的 LoRA 微調！' if lang == 'zh' else ''}**")

            # 顯示將要使用的參數摘要
            applied = _load_lora_config()
            if lang == "zh":
                st.caption(f"🚀 本次微調將使用：Epochs={applied['epochs']}、Rank={applied['rank']}、LR={format_lr(applied['lr'])}、Margin={applied['margin']}")
            else:
                st.caption(f"🚀 Fine-tuning will use: Epochs={applied['epochs']}, Rank={applied['rank']}, LR={format_lr(applied['lr'])}, Margin={applied['margin']}")

            if st.button("🚀 " + ("啟動 LoRA 微調 / Start LoRA Fine-Tuning" if lang == "zh" else "Start LoRA Fine-Tuning"), type="primary"):
                # 讀取最新套用的設定（確保不是改了但沒 Apply 就按下去）
                run_cfg = _load_lora_config()

                status_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                st.write("📈 **" + ("訓練即時曲線 (Dynamic Training Losses)" if lang == "zh" else "Live Training Loss Curve") + "**")
                loss_chart_placeholder = st.empty()
                import pandas as pd

                chart_data = {"Step": [], "Total": [], "BCE": [], "Rank": [], "InfoNCE": []}

                def update_progress(msg, progress_val, loss_dict=None, loss_val=None):
                    status_placeholder.code(msg)
                    progress_bar.progress(progress_val)
                    if loss_dict is not None:
                        chart_data["Step"].append(len(chart_data["Step"]))
                        for k, v in loss_dict.items():
                            chart_data[k].append(v)
                        df = pd.DataFrame(chart_data).set_index("Step")
                        loss_chart_placeholder.line_chart(df)
                    elif loss_val is not None:
                        chart_data["Step"].append(len(chart_data["Step"]))
                        chart_data["Total"].append(loss_val)
                        for k in ["BCE", "Rank", "InfoNCE"]:
                            chart_data[k].append(0.0)
                        df = pd.DataFrame(chart_data).set_index("Step")[["Total"]]
                        loss_chart_placeholder.line_chart(df)

                tuner = AcademicLoRATuner(
                    rag_engine=rag_engine,
                    project_name=st.session_state.project_name,
                    use_mps=use_mps_tuner
                )

                # 🚀 若啟用 HPO，執行超參數搜尋
                if use_hpo:
                    with st.spinner("📈 " + ("正在進行 Auto-HPO 超參數自動調優..." if lang == "zh" else "Running Auto-HPO Grid Sweep...")):
                        status_placeholder.info("🔍 " + ("啟動 HPO 超參數網格搜尋中，約需 1 分鐘..." if lang == "zh" else "Starting HPO Grid Sweep (approx 1 min)..."))
                        best_params, trials_data = tuner.run_hpo_search(
                            lora_r=run_cfg["rank"],
                            use_llm=use_llm_finetune,
                            status_callback=update_progress
                        )
                        
                        # 顯示 HPO 嘗試結果表格
                        st.write("📋 **" + ("Auto-HPO 搜尋結果列表" if lang == "zh" else "Auto-HPO Grid Search Trial Logs") + "**")
                        trials_df = pd.DataFrame(trials_data)
                        trials_df["lr"] = trials_df["lr"].apply(format_lr)
                        st.dataframe(trials_df.set_index("trial"), use_container_width=True)
                        
                        # 套用最佳超參數並存檔
                        st.success(f"🎯 " + (f"HPO 搜尋完畢！最佳超參數為：LR={format_lr(best_params['lr'])}、Margin={best_params['margin']}" if lang == "zh" else f"HPO Complete! Best params found: LR={format_lr(best_params['lr'])}, Margin={best_params['margin']}"))
                        _save_lora_config(run_cfg["epochs"], run_cfg["rank"], best_params["lr"], best_params["margin"])
                        run_cfg = _load_lora_config() # 重新載入最優解

                with st.spinner("🧠 " + ("正在訓練 LoRA Adapter..." if lang == "zh" else "Training LoRA Adapter...")):
                    new_report = tuner.train_lora(
                        epochs=run_cfg["epochs"],
                        lr=run_cfg["lr"],
                        lora_r=run_cfg["rank"],
                        margin=run_cfg["margin"],
                        patience=2,
                        use_llm=use_llm_finetune,
                        adapter_name=adapter_name,
                        status_callback=update_progress
                    )

                    st.session_state.diagnostic_report = new_report
                    st.success("🎉 " + ("LoRA 微調成功！" if lang == "zh" else "LoRA adapter trained successfully!"))

                    if run_ui_tests_checkbox:
                        st.info("🚀 " + ("正在啟動背景 Playwright UI 回歸測試..." if lang == "zh" else "Launching background Playwright UI regression tests..."))
                        try:
                            import subprocess
                            test_script = os.path.join(config.BASE_DIR, "tests", "run_ui_tests.py")
                            result = subprocess.run([sys.executable, test_script], capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                st.success("✅ " + ("Playwright UI 回歸測試全部通過！" if lang == "zh" else "Playwright UI tests passed!"))
                            else:
                                st.warning(f"⚠️ Playwright: {result.stderr or result.stdout}")
                        except Exception as test_err:
                            st.warning(f"{'無法執行 Playwright 測試' if lang == 'zh' else 'Failed to run Playwright tests'}: {test_err}")

                # ── 微調後重切片提醒 ────────────────────────────────
                if lang == "zh":
                    st.warning(
                        "🔁 **重要提示：微調已完成！** 建議您返回 **Step 1 (檔案處理)** 重新切片並重建向量索引，"
                        "以確保新的 LoRA 重排序模型能對最新的分塊結構發揮最佳效果。"
                    )
                else:
                    st.warning(
                        "🔁 **Important: Fine-tuning complete!** We recommend going back to **Step 1 (Ingestion)** "
                        "to re-chunk and rebuild the vector index. This ensures the new LoRA reranker performs optimally on the latest chunking structure."
                    )


                st.balloons()
                time.sleep(2)
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    # 效能雷達圖（微調前後對比）
    # ══════════════════════════════════════════════════════════════
    if st.session_state.before_report is not None and st.session_state.diagnostic_report is not None:
        st.write("---")
        st.write("### 📈 3. " + ("系統效能雷達圖與微調看板" if lang == "zh" else "Performance Comparison Radar"))

        before = st.session_state.before_report["diagnosis"]
        after = st.session_state.diagnostic_report["diagnosis"]
        before_ragas = st.session_state.before_report.get("reranked", {})
        after_ragas = st.session_state.diagnostic_report.get("reranked", {})

        import plotly.graph_objects as go
        categories = ['Context Recall', 'Context Relevancy', 'ROC-AUC', 'Margin (Normalized)']

        def normalize_margin(m):
            return max(0, min(1, m / 2.0))

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[before_ragas.get('ragas_context_recall', 0.5),
               before_ragas.get('ragas_context_relevancy', 0.5),
               before['auc'],
               normalize_margin(before['margin'])],
            theta=categories,
            fill='toself',
            name='Baseline (微調前)' if lang == "zh" else "Baseline"
        ))

        if before != after:
            fig.add_trace(go.Scatterpolar(
                r=[after_ragas.get('ragas_context_recall', 0.5),
                   after_ragas.get('ragas_context_relevancy', 0.5),
                   after['auc'],
                   normalize_margin(after['margin'])],
                theta=categories,
                fill='toself',
                name='LoRA Optimized (微調後)' if lang == "zh" else "LoRA Optimized"
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="系統多維度效能評估 (RAGAS Radar)" if lang == "zh" else "Multi-Dimensional Performance Radar (RAGAS)"
        )

        col_r, col_b, col_a = st.columns([2, 1, 1])
        with col_r:
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            label_b = "微調前" if lang == "zh" else "Baseline"
            st.metric(label=f"{label_b} ROC-AUC", value=f"{before['auc']:.4f}")
            st.metric(label=f"{label_b} Margin", value=f"{before['margin']:.4f}")
            st.metric(label=f"{label_b} Context Recall", value=f"{before_ragas.get('ragas_context_recall', 0):.4f}")
            st.error(f"{'狀態' if lang == 'zh' else 'Status'}: {before['status']}")

        with col_a:
            label_a = "微調後" if lang == "zh" else "LoRA Tuned"
            auc_delta = after['auc'] - before['auc']
            margin_delta = after['margin'] - before['margin']
            recall_delta = after_ragas.get('ragas_context_recall', 0) - before_ragas.get('ragas_context_recall', 0)

            st.metric(label=f"{label_a} ROC-AUC", value=f"{after['auc']:.4f}", delta=f"+{auc_delta:.4f}" if auc_delta >= 0 else f"{auc_delta:.4f}")
            st.metric(label=f"{label_a} Margin", value=f"{after['margin']:.4f}", delta=f"+{margin_delta:.4f}" if margin_delta >= 0 else f"{margin_delta:.4f}")
            st.metric(label=f"{label_a} Context Recall", value=f"{after_ragas.get('ragas_context_recall', 0):.4f}", delta=f"+{recall_delta:.4f}" if recall_delta >= 0 else f"{recall_delta:.4f}")
            st.success(f"{'狀態' if lang == 'zh' else 'Status'}: {after['status']}")

        # ── 🤖 AI 比較診斷與建議機制 ──
        if before != after:
            st.write("---")
            st.write("### 🤖 " + ("AI 訓練診斷與超參數建議" if lang == "zh" else "AI Training Diagnosis & Hyperparameter Advice"))
            
            def _get_advice(auc_delta, margin_delta, lang):
                if auc_delta < -0.005:
                    if lang == "zh":
                        return (
                            "**🚨 診斷：語意崩潰 (Representation Collapse) / 嚴重過擬合**\n\n"
                            "**現象**：微調後不僅沒有進步，反而破壞了模型原本的預訓練排序能力 (AUC 顯著下降)。\n\n"
                            "**建議調整 (基於文獻)**：\n"
                            "- **學習率太高**：請將學習率降至 `1e-5` 或 `5e-6`。*(He et al., 2021)* 指出 DeBERTa 架構的 Cross-Encoder 在小資料集且學習率 > 3e-5 時極易發生梯度假影，破壞原有特徵。\n"
                            "- **Rank 設定過大**：若目前 Rank ≥ 12，請降至 `8`。*(Hu et al., 2022)* 表明小語料使用過大的 Rank 會強迫模型記憶特定雜訊 (Over-parameterization)。\n"
                            "- **解決方案**：建議勾選左側的 **「Auto-HPO 自動調優」**，讓系統在安全的學習率範圍內幫您尋找最佳組合。"
                        )
                    else:
                        return "**🚨 Diagnosis: Representation Collapse / Overfitting**\n\nDecrease LR to 1e-5 or 5e-6 (He et al., 2021) and reduce Rank to 8 (Hu et al., 2022). Enable Auto-HPO."
                elif auc_delta > 0.005:
                    if lang == "zh":
                        return (
                            "**✅ 診斷：健康收斂 (Healthy Convergence)**\n\n"
                            "**現象**：微調後 ROC-AUC 顯著上升，模型已成功學習到您的領域專屬特徵！\n\n"
                            "**建議調整 (基於文獻)**：\n"
                            "- 目前參數處於**最佳甜區 (Sweet spot)**！建議保留當前設定。\n"
                            "- 若想進一步榨取效能，可嘗試將 `Epochs` 增加 1~2 輪，但請注意觀察 Validation AUC 是否開始下滑。*(Nogueira & Cho, 2019)*"
                        )
                    else:
                        return "**✅ Diagnosis: Healthy Convergence**\n\nKeep current settings. To squeeze more performance, slightly increase Epochs but watch for degradation (Nogueira & Cho, 2019)."
                elif abs(auc_delta) <= 0.005 and margin_delta > 0.01:
                    if lang == "zh":
                        return (
                            "**🟡 診斷：信心偏移 (Confidence Shift) / 欠擬合邊緣**\n\n"
                            "**現象**：整體的排序正確率 (AUC) 沒有改變，但正負樣本的差距 (Margin) 拉開了。代表模型正努力適應 Contrastive Loss，但尚未學到影響排名的關鍵特徵。\n\n"
                            "**建議調整 (基於文獻)**：\n"
                            "- **增加訓練輪數 (Epochs)**：模型可能還沒收斂，嘗試將 Epochs 增加 2~3 輪。\n"
                            "- **提升學習率**：若目前學習率在 `1e-5`，可嘗試提升至 `2e-5` 或 `3e-5` 加速收斂。*(Chen et al., 2020)*"
                        )
                    else:
                        return "**🟡 Diagnosis: Confidence Shift / Near Underfitting**\n\nIncrease Epochs by 2-3 or slightly increase LR (e.g. to 3e-5) to accelerate convergence (Chen et al., 2020)."
                else:
                    if lang == "zh":
                        return (
                            "**🧊 診斷：學習停滯 (Learning Stagnation)**\n\n"
                            "**現象**：微調前後的分數幾乎沒有變化。模型可能陷入局部最佳解，或是訓練資料的鑑別度不足。\n\n"
                            "**建議調整 (基於文獻)**：\n"
                            "- **開啟高階數據擴增 (LLM)**：模型需要更高品質的對抗樣本來學習。強烈建議勾選「啟用 LLM 生成反向學術考題」。\n"
                            "- **調整 Margin**：嘗試將 Margin 提高至 `0.4`，強迫模型更努力去區分困難樣本。*(Chen et al., 2020)*"
                        )
                    else:
                        return "**🧊 Diagnosis: Learning Stagnation**\n\nEnable LLM Data Augmentation to provide higher quality negative signals. Consider increasing Margin to 0.4 (Chen et al., 2020)."
            
            st.info(_get_advice(auc_delta, margin_delta, lang))
