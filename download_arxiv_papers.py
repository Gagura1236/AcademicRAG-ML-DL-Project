import os
import time
import urllib.request
# 引入專案的中心配置檔
# 這樣一來如果未來使用者在 config.py 更改了 PDF_DIR 的路徑 (例如變成 BBB)
# 本腳本以及切 embedding 的 src 都會自動同步使用新路徑，不需手動修改多處！
import config
from src.project_utils import get_project_paths

def download_papers(project_name="default"):
    """
    合法、免費下載公開的學術論文 (來自 arXiv)。
    下載路徑由 project_paths 控制，保證與專案 RAG 引擎的調用路徑完全一致。
    """
    project_paths = get_project_paths(project_name)
    target_dir = project_paths["pdf_dir"]
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📥 準備將公開論文下載至動態目錄：{target_dir}")
    print("-" * 50)
    
    # 39 份經典學術論文清單 (使用 arXiv IDs)
    papers = {
        "2007.14062": "Big Bird: Transformers for Longer Sequences",
        "2004.05150": "Longformer: The Long-Document Transformer",
        "2103.00020": "Learning Transferable Visual Models From Natural Language Supervision (CLIP)",
        "1512.03385": "Deep Residual Learning for Image Recognition (ResNet)",
        "1706.03762": "Attention Is All You Need (Transformer)",
        "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers",
        "1802.05365": "Deep contextualized word representations (ELMo)",
        "1606.05908": "Tutorial on Variational Autoencoders",
        "1312.6114": "Auto-Encoding Variational Bayes (VAE)",
        "1905.11946": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "1910.10683": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
        "1907.11692": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "1409.1556": "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)",
        "1707.06347": "Proximal Policy Optimization Algorithms (PPO)",
        "1710.10196": "Progressive Growing of GANs for Improved Quality, Stability, and Variation",
        "1406.2661": "Generative Adversarial Networks (GAN)",
        "1412.6980": "Adam: A Method for Stochastic Optimization",
        "1301.3781": "Efficient Estimation of Word Representations in Vector Space (Word2Vec)",
        "2010.11929": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)",
        "1409.0473": "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)",
        "2002.05709": "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)",
        "1409.3215": "Sequence to Sequence Learning with Neural Networks (Seq2Seq)",
        "1710.10903": "Graph Attention Networks (GAT)",
        "1506.01497": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
        "1406.1078": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (GRU)",
        "2006.11239": "Denoising Diffusion Probabilistic Models (DDPM)",
        "1506.02640": "You Only Look Once: Unified, Real-Time Object Detection (YOLO)",
        "1312.5602": "Playing Atari with Deep Reinforcement Learning (DQN)",
        "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "2111.06377": "Masked Autoencoders Are Scalable Vision Learners (MAE)",
        "1609.02907": "Semi-Supervised Classification with Graph Convolutional Networks (GCN)",
        "2203.02155": "Training language models to follow instructions with human feedback (InstructGPT)",
        "1606.07792": "Wide & Deep Learning for Recommender Systems",
        "2012.13293": "Fuzzy Commitments Offer Insufficient Protection Biometric Templates Produced Deep Learning",
        "2306.16361": "Beyond NTK with Vanilla Gradient Descent: Mean-Field Analysis Neural Networks with Polynomial Width, Samples, and Time",
        "2012.12104": "A Deep Reinforcement Learning Approach for Ramp Metering Based on Traffic Video Data",
        "2012.13026": "Rethink AI-based Power Grid Control: Diving Into Algorithm Design",
        "1911.05402": "QUADRATIC NUMBER NODES SUFFICIENT LEARN DATASET VIA GRADIENT DESCENT",
        "2311.15051": "Gradient Descent with Polyak's Momentum Finds Flatter Minima via Large Catapults"
    }
    
    for arxiv_id, title in papers.items():
        pdf_path = os.path.join(target_dir, f"{arxiv_id}.pdf")
        
        if os.path.exists(pdf_path):
            print(f"✅ 檔案已存在，跳過下載: {arxiv_id}.pdf ({title})")
            continue
            
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"⬇️ 正在下載: {arxiv_id} - {title} ...")
        
        try:
            # 加上 User-Agent 遵守 arXiv 的 API 禮貌原則
            req = urllib.request.Request(url, headers={'User-Agent': 'AcademicRAG_Downloader/3.0'})
            with urllib.request.urlopen(req) as response:
                with open(pdf_path, 'wb') as f:
                    f.write(response.read())
            print("  └─ ✨ 下載成功！")
            
            # 禮貌性延遲，避免對 arXiv 伺服器造成壓力 (Rate Limit)
            time.sleep(3)
        except Exception as e:
            print(f"  └─ ❌ 下載失敗: {e}")
            
    print("-" * 50)
    print("🎉 下載任務完成！接下來您可以執行向量庫重建 (build_index.py 或是 App 介面) 來處理這些文獻。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download arXiv papers.")
    parser.add_argument("--project", "-p", type=str, default="default", help="Project name")
    args = parser.parse_args()
    
    download_papers(args.project)
