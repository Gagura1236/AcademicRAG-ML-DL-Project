#!/usr/bin/env python3
"""
build_index.py — 離線向量庫重建腳本
直接從 data/pdfs/ 讀取所有 PDF，使用 BAAI/bge-m3 重新切片、向量化、存檔
"""
import os
import sys

# 確保可以 import src/*
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("AcademicRAG v3.0 — 離線向量庫重建工具")
print("=" * 60)

import config
import argparse
from src.project_utils import get_project_paths

parser = argparse.ArgumentParser(description="AcademicRAG v3.0 — 離線向量庫重建工具")
parser.add_argument("--project", "-p", type=str, default="default", help="指定要重建的專案名稱 (預設: default)")
args = parser.parse_args()

project_name = args.project
project_paths = get_project_paths(project_name)

print(f"[設定] 專案: {project_name}")
print(f"[設定] 裝置: {config.DEVICE}")
print(f"[設定] Embedding 模型: {config.EMBEDDING_MODEL_NAME}")
print(f"[設定] PDF 目錄: {project_paths['pdf_dir']}")
print(f"[設定] 向量庫目錄: {project_paths['vector_db_dir']}")
print()

from src.embedding_search import AcademicRAGEngine

print("[初始化] 載入 RAG Engine（含 bge-m3 與 Cross-Encoder）...")
engine = AcademicRAGEngine(project_name=project_name)
print("[初始化] RAG Engine 載入完成！\n")

print("[重建] 開始重建向量庫...")
print("-" * 60)
count = engine.rebuild_index_from_pdfs()
print("-" * 60)

if count > 0:
    print(f"\n✅ 向量庫重建成功！")
    print(f"   處理論文數：{count} 篇")
    print(f"   總 Chunks 數：{len(engine.chunks_metadata)}")
    print(f"   FAISS 向量維度：{engine.index.d if engine.index else 'N/A'}")
    print(f"\n向量庫已儲存至: {project_paths['vector_db_dir']}")
else:
    print("\n⚠️  向量庫重建失敗，未找到可處理的 PDF。")
    print(f"   請確認 PDF 檔案位於: {project_paths['pdf_dir']}")

print("\n完成！")
