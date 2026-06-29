import os
import sys
import tarfile
import urllib.request
import gzip
import arxiv

# 獲取專案根目錄，加入系統路徑以正確導入 config.py
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.pdf_parser import ScientificPDFParser
from src.embedding_search import AcademicRAGEngine
from src.metadata_extractor import PaperMetadataManager
from src.project_utils import get_project_paths

def download_and_index_by_ids(id_list: list, project_name: str = "default"):
    """
    依據 arXiv ID 列表，精確下載論文 PDF 與 LaTeX 原始碼，並進行高維公式對齊與 FAISS 語意索引。
    """
    print(f"\n[Precision Pipeline] 開始處理指定 arXiv IDs: {id_list} (Project: {project_name})...")
    
    project_paths = get_project_paths(project_name)
    client = arxiv.Client()
    search = arxiv.Search(id_list=id_list)
    
    parser = ScientificPDFParser(load_ocr_model=False)
    rag_engine = AcademicRAGEngine(project_name=project_name)
    metadata_manager = PaperMetadataManager(project_name=project_name)
    
    for i, result in enumerate(client.results(search)):
        # 移除 ID 中的版本號 (例如 1810.04805v2 -> 1810.04805)
        paper_id = result.entry_id.split("/abs/")[-1].split("v")[0]
        title = result.title.replace("/", "_").replace(":", "_")
        
        print(f"\n⚡ [{i+1}/{len(id_list)}] 正在處理論文 ID: {paper_id}")
        print(f"   標題: {result.title}")
        
        # 1. 下載 PDF
        pdf_filename = f"{paper_id}.pdf"
        pdf_path = os.path.join(project_paths["pdf_dir"], pdf_filename)
        
        if not os.path.exists(pdf_path):
            print(f"   -> 下載 PDF 到: {pdf_path}")
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                req = urllib.request.Request(result.pdf_url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response, open(pdf_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"   -> PDF 下載成功！")
            except Exception as e:
                print(f"   ⚠️ PDF 下載失敗: {e}")
                continue
        else:
            print(f"   -> PDF 已存在，跳過下載。")
            
        # 2. 下載 LaTeX 原始碼 (用於公式 Ground Truth 對齊)
        tex_folder = os.path.join(project_paths["tex_dir"], paper_id)
        os.makedirs(tex_folder, exist_ok=True)
        
        src_url = f"https://arxiv.org/src/{paper_id}"
        tar_path = os.path.join(tex_folder, f"{paper_id}.tar.gz")
        
        if not os.path.exists(tar_path):
            print(f"   -> 下載 LaTeX 原始碼到: {tar_path}")
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                req = urllib.request.Request(src_url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response, open(tar_path, 'wb') as out_file:
                    out_file.write(response.read())
                
                print(f"   -> 解壓 LaTeX 原始碼...")
                try:
                    with tarfile.open(tar_path) as tar:
                        tar.extractall(path=tex_folder)
                    print(f"   -> LaTeX 原始碼解壓成功！")
                except tarfile.TarError as tar_err:
                    print(f"   ⚠️ Tar 解壓失敗: {tar_err}，嘗試作為單個 Gzip 解壓...")
                    try:
                        decompressed_file = os.path.join(tex_folder, f"{paper_id}.tex")
                        with gzip.open(tar_path, 'rb') as f_in:
                            with open(decompressed_file, 'wb') as f_out:
                                f_out.write(f_in.read())
                        print(f"   -> LaTeX 原始碼 (單個 Gzip) 解壓成功！")
                    except Exception as gz_err:
                        print(f"   ⚠️ LaTeX 解壓失敗: {gz_err}")
            except Exception as e:
                print(f"   ⚠️ 下載 LaTeX 原始碼失敗: {e}")
        else:
            print(f"   -> LaTeX 原始碼已存在，跳過。")
            
        # 3. 執行 Layout 解析與公式 Ground-Truth 對齊
        print(f"   -> 進行 Layout 結構化與精準公式對齊...")
        elements = parser.parse_layout(pdf_path)
        processed_elements = parser.process_pdf_equations(pdf_path, elements)
        
        # 4. 寫入本地向量庫
        print(f"   -> 寫入本地向量資料庫 (FAISS)...")
        rag_engine.add_documents(processed_elements, paper_id, result.title)
        
        # 5. 儲存元數據
        authors = [author.name for author in result.authors]
        metadata_manager.add_arxiv_metadata(
            paper_id=paper_id,
            title=result.title,
            authors=authors,
            abstract=result.summary,
            categories=result.categories
        )
        
    print("\n🎉 指定經典論文下載與高精度對齊索引全部完成！")

if __name__ == "__main__":
    # 預設要下載並高精度索引的經典 AI 論文列表 (共 20 篇)：
    target_papers = [
        # === 1. Transformer & Attention ===
        "1810.04805", # BERT
        "1706.03762", # Attention Is All You Need (Transformer)
        "1409.0473",  # Bahdanau Attention
        "1910.10683", # T5
        "1907.11692", # RoBERTa
        
        # === 2. Large Language Models (LLM) & RLHF ===
        "2005.11401", # GPT-3
        "2203.02155", # InstructGPT (RLHF)
        
        # === 3. Sliding Window / Sparse Attention ===
        "2007.14062", # Big Bird
        "2004.05150", # Longformer
        
        # === 4. CNN & Vision Models ===
        "1512.03385", # ResNet
        "1409.1556",  # VGG
        "1905.11946", # EfficientNet
        "2010.11929", # ViT (Vision Transformer)
        
        # === 5. Object Detection (CNN) ===
        "1506.01497", # Faster R-CNN
        "1506.02640", # YOLO
        
        # === 6. RNN / Sequence Models ===
        "1409.3215",  # Seq2Seq (Sutskever)
        "1406.1078",  # GRU (Cho)
        
        # === 7. Graph Neural Networks (GNN) ===
        "1609.02907", # GCN (Graph Convolutional Networks)
        "1710.10903", # GAT (Graph Attention Networks)
        
        # === 8. Generative Models (GAN, VAE, Diffusion) ===
        "1406.2661",  # GANs
        "1710.10196", # Progressive Growing of GANs
        "1312.6114",  # VAE
        "1606.05908", # Tutorial on VAE
        "2006.11239", # DDPM (Denoising Diffusion Probabilistic Models)
        
        # === 9. Self-Supervised & Contrastive Learning ===
        "2002.05709", # SimCLR
        "2111.06377", # MAE (Masked Autoencoders)
        
        # === 10. Reinforcement Learning (RL) ===
        "1312.5602",  # DQN (Deep Q-Network)
        "1707.06347", # PPO (Proximal Policy Optimization)
        
        # === 11. Multimodal & Emdeddings ===
        "2103.00020", # CLIP
        "1301.3781",  # Word2Vec
        "1802.05365", # ELMo
        
        # === 12. Optimization & RecSys ===
        "1412.6980",  # Adam Optimizer
        "1606.07792"  # YouTube Deep Neural Networks for Recommendations
    ]
    
    import argparse
    parser = argparse.ArgumentParser(description="Download and index arXiv papers.")
    parser.add_argument("--project", "-p", type=str, default="default", help="Project name")
    parser.add_argument("ids", nargs="*", default=target_papers, help="List of arXiv IDs")
    args = parser.parse_args()
    
    download_and_index_by_ids(args.ids, project_name=args.project)
