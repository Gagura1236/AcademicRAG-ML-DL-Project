import os
import sys
import tarfile
import urllib.request
import arxiv
import cloudscraper
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# 獲取專案根目錄，加入系統路徑以正確導入 config.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from src.metadata_extractor import PaperMetadataManager
from src.project_utils import get_project_paths

# 初始化一個具備瀏覽器特徵的 Scraper，繞過 Cloudflare 阻擋
scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True})

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
)
def robust_download(url: str, dest_path: str, is_pdf: bool = True):
    """
    帶有指數退避重試機制的強健下載器
    """
    tmp_path = dest_path + ".tmp"
    try:
        response = scraper.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        ct = response.headers.get('Content-Type', '')
        if is_pdf and 'application/pdf' not in ct and 'octet-stream' not in ct:
            raise ValueError(f"伺服器回傳非 PDF 內容 (Content-Type: {ct})，可能遭遇 429 或 403 阻擋。")
            
        with open(tmp_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        os.rename(tmp_path, dest_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

def process_arxiv_result(result, metadata_manager, max_results, paths):
    paper_id = result.entry_id.split("/abs/")[-1].split("v")[0]
    title = result.title.replace("/", "_").replace(":", "_")
    print(f"\n[ArXiv Collector] 發現論文 ID: {paper_id} | 標題: {result.title}")
    
    # 1. 下載 PDF 檔案
    pdf_filename = f"{paper_id}.pdf"
    pdf_path = os.path.join(paths['pdf_dir'], pdf_filename)
    pdf_success = False
    
    if not os.path.exists(pdf_path):
        print(f"   [{paper_id}] -> 正在下載 PDF 到: {pdf_path}")
        pdf_url = result.pdf_url
        mirror_url = pdf_url.replace("arxiv.org", "export.arxiv.org")
        
        try:
            robust_download(pdf_url, pdf_path, is_pdf=True)
            pdf_success = True
            print(f"   [{paper_id}] -> PDF 下載完成！")
        except Exception as e1:
            print(f"   [{paper_id}] ⚠️ 主站點下載失敗 ({type(e1).__name__})，嘗試備用鏡像站點 (export.arxiv.org)...")
            try:
                robust_download(mirror_url, pdf_path, is_pdf=True)
                pdf_success = True
                print(f"   [{paper_id}] -> PDF 鏡像下載完成！")
            except Exception as e2:
                print(f"   [{paper_id}] ❌ PDF 下載徹底失敗 ({type(e2).__name__})。啟動優雅降級機制 (僅儲存摘要)...")
    else:
        print(f"   [{paper_id}] -> PDF 已存在，跳過下載。")
        pdf_success = True
        
    # 2. 下載 LaTeX 原始碼
    tex_folder = os.path.join(paths['tex_dir'], paper_id)
    os.makedirs(tex_folder, exist_ok=True)
    tar_path = os.path.join(tex_folder, f"{paper_id}.tar.gz")
    
    if not os.path.exists(tar_path) and pdf_success:
        print(f"   [{paper_id}] -> 正在下載 LaTeX 原始碼...")
        src_url = f"https://export.arxiv.org/e-print/{paper_id}" # e-print 是原始碼端點
        try:
            robust_download(src_url, tar_path, is_pdf=False)
            # 解壓
            try:
                with tarfile.open(tar_path) as tar:
                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        return os.path.commonpath([abs_directory, abs_target]) == abs_directory
                    for member in tar.getmembers():
                        member_path = os.path.join(tex_folder, member.name)
                        if not is_within_directory(tex_folder, member_path):
                            raise ValueError("Path Traversal Vulnerability")
                    tar.extractall(path=tex_folder)
                print(f"   [{paper_id}] -> 原始碼解壓完成。")
            except tarfile.ReadError:
                # 作為單個 gzip 處理
                import gzip
                decompressed_file = os.path.join(tex_folder, f"{paper_id}.tex")
                with gzip.open(tar_path, 'rb') as f_in:
                    with open(decompressed_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"   [{paper_id}] -> 原始碼單個 Gzip 解壓完成。")
        except Exception as e:
            print(f"   [{paper_id}] ⚠️ LaTeX 原始碼下載或解壓失敗 (不影響 RAG): {type(e).__name__} - {e}")
            
    # 儲存元數據
    authors = [author.name for author in result.authors]
    metadata_manager.add_arxiv_metadata(
        paper_id=paper_id,
        title=result.title,
        authors=authors,
        abstract=result.summary,
        categories=result.categories
    )
            
    return {
        "id": paper_id,
        "title": result.title,
        "pdf_path": pdf_path if pdf_success else None,
        "tex_folder": tex_folder,
        "abstract": result.summary
    }

def download_arxiv_papers(query: str, max_results: int = 5, project_name: str = "default"):
    """
    使用 arXiv API 搜尋並下載論文 PDF 與 LaTeX 原始碼，透過 ThreadPoolExecutor 平行加速
    """
    import concurrent.futures
    print(f"\n[ArXiv Collector] 開始搜尋關鍵字: '{query}' (預計下載最多 {max_results} 篇)...")
    
    paths = get_project_paths(project_name)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    metadata_manager = PaperMetadataManager(project_name=project_name)
    
    # 取得搜尋結果並列入清單
    results = list(client.results(search))
    if not results:
        print("[ArXiv Collector] 找不到符合的論文。")
        return []
        
    print(f"[ArXiv Collector] 找到 {len(results)} 篇論文，開始平行下載...")
    downloaded_papers = []
    
    # 使用 ThreadPoolExecutor 進行平行下載 (最多 5 個 Worker，避免被 arXiv Ban)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(results))) as executor:
        futures = [executor.submit(process_arxiv_result, res, metadata_manager, max_results, paths) for res in results]
        for future in concurrent.futures.as_completed(futures):
            try:
                paper_info = future.result()
                downloaded_papers.append(paper_info)
            except Exception as e:
                print(f"[ArXiv Collector] 平行處理發生未預期錯誤: {e}")
        
    print(f"\n[ArXiv Collector] 數據收集完畢！成功處理 {len(downloaded_papers)} 篇論文。")
    return downloaded_papers

def download_single_pdf(pdf_url: str, paper_id: str, title: str = None, project_name="default") -> list:
    """
    通用 PDF 下載器，搭配 robust_download 具備反爬蟲重試能力。
    """
    from src.system_utils import get_project_paths
    paths = get_project_paths(project_name)
    
    print(f"\n[Generic Collector] 準備下載論文 ID: {paper_id}")
    print(f"   來源網址: {pdf_url}")
    
    if not title:
        title = f"Document_{paper_id}"
        
    pdf_filename = f"{paper_id}.pdf"
    pdf_path = os.path.join(paths['pdf_dir'], pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f"   -> 正在下載 PDF 到: {pdf_path}")
        try:
            robust_download(pdf_url, pdf_path, is_pdf=True)
            print(f"   -> PDF 下載完成！")
        except Exception as pdf_err:
            print(f"   ❌ PDF 下載徹底失敗 ({pdf_err})。")
            return [] # 通用下載器若無 PDF 就沒什麼用了
    else:
        print(f"   -> PDF 已存在，跳過下載。")
        
    return [{"id": paper_id, "title": title, "pdf_path": pdf_path, "abstract": ""}]

if __name__ == "__main__":
    query = 'cat:cs.LG AND "gradient descent"'
    download_arxiv_papers(query, max_results=3)
