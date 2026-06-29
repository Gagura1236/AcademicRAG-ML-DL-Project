import os
import zipfile
import sys

def package_project(mode="standard"):
    """
    將 RAG 系統專案打包成壓縮檔，排除無用的暫存檔以符合繳交規範。
    
    mode:
      - "standard": 只保留程式碼、設定檔、中繼資料、雷達圖與預建向量資料庫，排除大檔案 (PDFs, Tex 原始碼, extracted 圖片) [推薦：大小約 70MB]
      - "full": 打包所有檔案，包含所有下載的 100MB+ PDF 原始檔與 230MB+ Tex 原始碼 [大小約 430MB]
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_name = f"AcademicRAG_v3.9.4_{mode}.zip"
    zip_path = os.path.join(base_dir, zip_name)
    
    print(f"📦 正在啟動專案打包管線... 模式: [{mode}]")
    
    exclude_dirs = {
        "__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", 
        "v1.0", "v2.0", "node_modules", "dev_records"
    }
    
    exclude_files = {
        ".DS_Store", "package_project.py"
    }
    
    # 針對 Standard 模式額外排除的資料夾
    standard_exclude_subdirs = {
        "pdfs", "tex", "extracted", "reference_papers"
    }
    
    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # 過濾第一層排除目錄
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # 若為 standard 模式，在 data/projects 內過濾大型目錄
            rel_path = os.path.relpath(root, base_dir)
            if mode == "standard":
                parts = rel_path.split(os.sep)
                # 判定是否處於某個專案的子目錄下，例如 data/projects/default/pdfs
                if len(parts) >= 3 and parts[0] == "data" and parts[1] == "projects":
                    # parts[2] 是專案名稱如 default，parts[3] 是子資料夾如 pdfs, tex
                    if len(parts) >= 4 and parts[3] in standard_exclude_subdirs:
                        print(f"  ❌ 排除大型資料夾: {rel_path}")
                        dirs[:] = []  # 停止往下遞迴
                        continue
            
            for file in files:
                if file in exclude_files or file.endswith(".zip") or file.endswith(".pyc"):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                
                # 再次雙重防線防堵 standard 模式的大檔案寫入
                if mode == "standard" and "data/projects/" in arcname:
                    parts = arcname.split("/")
                    if len(parts) >= 4 and parts[3] in standard_exclude_subdirs:
                        continue
                
                zipf.write(file_path, arcname)
                count += 1
                
    print(f"🎉 打包完成！共寫入 {count} 個檔案。")
    print(f"💾 壓縮檔儲存於: {zip_path}")
    print(f"📏 檔案大小: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    mode = "standard"
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["standard", "full"]:
        mode = sys.argv[1].lower()
    package_project(mode)
