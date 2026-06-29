import os
import config

def get_project_base_dir() -> str:
    """獲取所有專案的根目錄"""
    projects_dir = os.path.join(config.DATA_DIR, "projects")
    os.makedirs(projects_dir, exist_ok=True)
    return projects_dir

def get_all_projects() -> list:
    """獲取目前所有存在的專案列表"""
    projects_dir = get_project_base_dir()
    projects = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
    if "default" not in projects:
        projects.append("default")
    return sorted(list(set(projects)))

def get_project_dir(project_name: str) -> str:
    """獲取特定專案的根目錄"""
    if not project_name:
        project_name = "default"
    project_dir = os.path.join(get_project_base_dir(), project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def get_project_paths(project_name: str, collection_name: str = "main") -> dict:
    """
    獲取特定專案內所有的動態路徑。
    支援 collection_name 來切換不同的向量庫實體。
    """
    project_dir = get_project_dir(project_name)
    
    # 確保 base vector_db 存在
    base_vector_db_dir = os.path.join(project_dir, "vector_db")
    os.makedirs(base_vector_db_dir, exist_ok=True)
    
    # 向後相容移轉 (將原本直接放在 vector_db 裡的 faiss.index 等搬到 main)
    if os.path.exists(os.path.join(base_vector_db_dir, "faiss.index")):
        main_dir = os.path.join(base_vector_db_dir, "main")
        os.makedirs(main_dir, exist_ok=True)
        import shutil
        for f in ["faiss.index", "metadata.pkl", "bm25.pkl"]:
            src = os.path.join(base_vector_db_dir, f)
            dst = os.path.join(main_dir, f)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
    
    paths = {
        "project_dir": project_dir,
        "pdf_dir": os.path.join(project_dir, "pdfs"),
        "tex_dir": os.path.join(project_dir, "tex"),
        "vector_db_dir": os.path.join(base_vector_db_dir, collection_name),
        "base_vector_db_dir": base_vector_db_dir,
        "extracted_dir": os.path.join(project_dir, "extracted"),
        "lora_adapters_dir": os.path.join(project_dir, "lora_adapters")
    }
    
    # 自動初始化目錄
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
        
    return paths

def get_collections(project_name: str) -> list:
    """獲取專案內的所有向量庫集合"""
    # 為了移轉舊版，先呼叫一次 get_project_paths 確保有 main
    paths = get_project_paths(project_name, "main")
    base_dir = paths["base_vector_db_dir"]
    
    collections = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if "main" not in collections:
        collections.append("main")
    return sorted(list(set(collections)))
