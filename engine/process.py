from __future__ import annotations
import sys
from pathlib import Path
import uuid

from dotenv import find_dotenv
ROOT = Path(find_dotenv()).parent
sys.path.append(str(ROOT))

import json
import os
from engine.chunking import RecursiveChunker
from engine.schema import Document
from engine.embedding import LocalEmbedder

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_data(data_path):
    all_data = []

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
        
            with open(file_path, mode='r', encoding='utf-8') as f:
                content = f.read()
            
            item = {
                "content": content,
                "metadata": {
                    "title": filename  # Lưu tên file vào metadata
                }
            }
            all_data.append(item)

    return all_data

def process_chunks(data):
    final_chunks = []
    
    for item in data:
        content = item['content']
        metadata = item['metadata']
        
        chunker = RecursiveChunker()
        chunks = chunker.chunk(content)
        
        for i, chunk_text in enumerate(chunks):
            final_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "title": metadata['title'],
                }
            })
            
    return final_chunks

def process_document(chunks):
    """
    Biến danh sách các dictionary thành danh sách các đối tượng Document đã có embedding.
    """
    embedder = LocalEmbedder()
    texts_to_embed = [item['content'] for item in chunks]
    
    print(f"Đang tạo embeddings cho {len(texts_to_embed)} đoạn văn bản...")
    all_embeddings = embedder.encode_batch(texts_to_embed, prompt_name="document")
    
    # 3. Tạo danh sách các đối tượng Document
    documents = []
    for i, item in enumerate(chunks):
        # 1. Tạo doc_id riêng biệt trước
        doc_id = str(uuid.uuid4())
        
        # 2. Khởi tạo Document với metadata đã được cập nhật
        doc = Document(
            id=doc_id, # Sử dụng biến doc_id ở đây
            embeddings=all_embeddings[i],
            content=item['content'],
            metadata={
                "title": item['metadata'].get('title', ''), # Dùng .get() để tránh lỗi nếu không có key
                "doc_id": doc_id,                           # ← Thêm doc_id vào metadata
                "animal": item['metadata'].get('title', '') # ← Thêm animal nếu data có trường này
            }
        )
        documents.append(doc)
    
    return documents