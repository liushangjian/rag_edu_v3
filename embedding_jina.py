"""
查询文本 → 向量 → 相似度搜索 → 最相似向量 → 原始文本
"""

import psycopg2
import os
import json
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
import time
import requests
import hashlib
import asyncio
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor


class JSONLoader:
    def load_documents(self, file_path: str, doc_format: str = "text", test: bool = False, test_limit: int = 5) -> list[dict]:
        documents = []
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if doc_format == "text":
                     for item in data:
                        if test and count >= test_limit:
                            break
                        document = {
                            "content": item.get("Content", ""),
                            "metadata": {
                                "book": os.path.splitext(file_path)[0],
                                "chapter": item.get("Chapter"),
                                "section": item.get("Section"),
                                "subsection": item.get("Subsection"),
                                "chunk_info": item.get("chunk_info", {})
                            }
                        }
                        documents.append(document)
                        count += 1
                elif doc_format == "image":
                    for key, item in data.items():
                        if test and count >= test_limit:
                            break
                        document = {
                            "content": item,
                            "metadata": {
                                "book": os.path.splitext(file_path)[0],
                                "image_path": key
                            }
                        }
                        documents.append(document)
                        count += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file: {file_path} - {e}")
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents


class EmbeddingPipeline:
    def __init__(self):
        try:
            self.api_key = "your jina api key"
            self.api_url = "https://api.jina.ai/v1/embeddings"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.conn = psycopg2.connect(
                dbname="rag_db",
                user="edurag_user",
                password="edurag_pass",
                host="localhost",
                port="5432"
            )
            self.cur = self.conn.cursor() 
        except Exception as e:
            print(f"Initialization error: {e}")
            self.cur = None 
        
        self.processed_files_path = os.path.join(os.path.dirname(__file__), "processed_files.json")
        self.processed_files = self.load_processed_files()
        self.embedding_stats = {
            'total_documents': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'total_time': 0
        }

    def load_processed_files(self):
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, "r") as f:
                return set(json.load(f))
        return set()
    
    def save_processed_file(self, file_path):
        self.processed_files.add(file_path)
        with open(self.processed_files_path, "w") as f:
            json.dump(list(self.processed_files), f)

    def get_table_name(self, doc_format: str) -> str:
        # Define a mapping from document type to table name
        table_name_mapping = {
            "text": "rag_text_embeddings",
            "image": "rag_image_embeddings"
            # Add more mappings if there are other document types
        }
        return table_name_mapping.get(doc_format, "default_table_name")


    def create_table_if_not_exists(self, doc_format: str):
        if not self.cur:
            print("Database cursor is not initialized.")
            return
        table_name = self.get_table_name(doc_format)
        try:
            if doc_format == "text":
                self.cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        book_title VARCHAR(255),
                        chapter VARCHAR(255),
                        section VARCHAR(255),
                        subsection VARCHAR(255),
                        content TEXT,
                        metadata JSONB,
                        embedding VECTOR(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:  # 图片表新结构
                self.cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        book_title VARCHAR(255),
                        file_path TEXT,
                        page_number INTEGER,
                        figure_name VARCHAR(255),
                        folder_path VARCHAR(255),
                        content TEXT,
                        metadata JSONB,
                        embedding VECTOR(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            self.conn.commit()
        except Exception as e:
            print(f"Error creating table {table_name}: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            start_time = time.time()
            
            # 打印更多调试信息
            print(f"\nProcessing batch of {len(texts)} texts")
            print(f"Sample text length: {len(texts[0])} characters")
            
            payload = {
                "input": texts,
                "model": "jina-embeddings-v3",
                "task": "retrieval.passage",
                "late_chunking": False,
                "dimensions": 1024,
                "embedding_type": "float"
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            print(f"API Response time: {time.time() - start_time:.2f} seconds")
            print(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            
            # 验证返回的embedding数量
            print(f"Received {len(embeddings)} embeddings")
            print(f"Each embedding has {len(embeddings[0])} dimensions")
            
            return embeddings
            
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            self.embedding_stats['failed_embeddings'] += len(texts)
            return []
            
    @staticmethod 
    def parse_image_path(image_path: str) -> dict:
        """解析图片路径，按照 folder-name_origin_page_X_figure-name 格式提取信息"""
        try:
            filename = os.path.basename(image_path)
            folder_path = os.path.dirname(image_path)
            
            # 使用 "origin_page" 作为分隔符拆分
            parts = filename.split("_origin_page_")
            if len(parts) != 2:
                raise ValueError("Invalid image path format")
            
            folder_name = parts[0]
            remaining = parts[1]
            
            # 进一步解析页码和图片名称
            page_parts = remaining.split("_", 1)
            page_number = int(page_parts[0])
            figure_name = page_parts[1] if len(page_parts) > 1 else None
            
            # 构建完整的文件路径
            full_file_path = os.path.join(
                "/root/lsj/data/整体书库figures",
                folder_name,
                "figures",
                filename
            )
            
            return {
                "file_path": full_file_path,
                "page_number": page_number,
                "figure_name": figure_name,
                "folder_path": folder_path
            }
        except Exception as e:
            print(f"Error parsing image path {image_path}: {e}")
            return {}

    def load_and_embed_documents(self, documents: List[dict], doc_format: str):
        start_time = time.time()
        self.embedding_stats['total_documents'] += len(documents)
        
        if not self.cur:
            print("Database cursor is not initialized.")
            return
        table_name = self.get_table_name(doc_format)
        self.create_table_if_not_exists(doc_format)

        document_contents = [doc["content"] for doc in documents]
        embeddings = self.get_embeddings(document_contents)
        
        for document, embedding in zip(documents, embeddings):
            try:
                if doc_format == "text":
                    self.cur.execute(f"""
                        INSERT INTO {table_name} 
                        (book_title, chapter, section, subsection, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        document["metadata"].get("book"),
                        document["metadata"].get("chapter"),
                        document["metadata"].get("section"),
                        document["metadata"].get("subsection"),
                        document["content"],
                        json.dumps({"chunk_info": document["metadata"].get("chunk_info", {})}),
                        embedding
                    ))
                else:  # 图片表的新插入逻辑
                    image_path = document["metadata"].get("image_path", "")
                    parsed_info = self.parse_image_path(image_path)
                    
                    self.cur.execute(f"""
                        INSERT INTO {table_name} 
                        (book_title, file_path, page_number, figure_name, folder_path, 
                         content, metadata, embedding, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        document["metadata"].get("book"),
                        parsed_info.get("file_path"),
                        parsed_info.get("page_number"),
                        parsed_info.get("figure_name"),
                        parsed_info.get("folder_path"),
                        document["content"],
                        json.dumps({
                            "page_number": parsed_info.get("page_number"),
                            "figure_name": parsed_info.get("figure_name")
                        }),
                        embedding
                    ))
            except Exception as e:
                print(f"Error inserting document into table {table_name}: {e}")

        self.conn.commit()
        print("Transaction committed.")

        print(f"\nBatch Statistics:")
        print(f"Processed {len(documents)} documents in {time.time() - start_time:.2f} seconds")
        print(f"Average time per document: {(time.time() - start_time) / len(documents):.2f} seconds")

    
    def embedding_pipeline(self, file_dir: str, doc_format: str = "text", test: bool = False, batch_size: int = 500):
        print(f"Embedding pipeline started for {file_dir} with {doc_format} documents")
        loader = JSONLoader()
        count = 0

        for file in os.listdir(file_dir):
            count += 1
            if test and count >= 5:
                break
            doc_path = os.path.join(file_dir, file)
            if doc_path in self.processed_files:
                print(f"Skipping already processed file: {doc_path}")
                continue
            all_documents = []
            if doc_path.endswith(".json"):
                documents = loader.load_documents(doc_path, doc_format=doc_format, test=test)
                all_documents.extend(documents)

            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                self.load_and_embed_documents(batch, doc_format)
                
            # Mark this file as processed
            self.save_processed_file(doc_path)

    def check_database_records(self):
        """检查数据库中的记录数量"""
        if not self.cur:
            print("Database cursor is not initialized.")
            return
            
        tables = ["rag_text_embeddings", "rag_image_embeddings"]
        for table in tables:
            try:
                self.cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cur.fetchone()[0]
                print(f"\nTable {table} contains {count} records")
                
                # 检查一些示例记录的embedding维度
                self.cur.execute(f"SELECT embedding FROM {table} LIMIT 1")
                sample = self.cur.fetchone()
                if sample:
                    print(f"Sample embedding dimension: {len(sample[0])}")
                
            except Exception as e:
                print(f"Error checking table {table}: {e}")

if __name__ == "__main__":
    # Initialize the embedding pipeline
    pipeline = EmbeddingPipeline()
    
    # core text
    file_directory = "/root/lsj/data/Textbook_Rag_Resouces_rename/all_book/课本"
    pipeline.embedding_pipeline(file_dir=file_directory, doc_format="text", test=False, batch_size=500)

    # supplementary text
   # file_directory = "Textbook_Rag_Resouces/补充书库/课本"
   # pipeline.embedding_pipeline(file_dir=file_directory, doc_format="text", doc_type="supplementary", test=False, batch_size=500)


    # core image
   # file_directory = "Textbook_Rag_Resouces/核心书库/图片"
   # pipeline.embedding_pipeline(file_dir=file_directory, doc_format="image", doc_type="core", test=False, batch_size=500)

    # supplementary image
    file_directory = "/root/lsj/data/Textbook_Rag_Resouces_rename/all_book/图片"
    pipeline.embedding_pipeline(file_dir=file_directory, doc_format="image", test=False, batch_size=500)


    
    # Close database connection
    pipeline.cur.close()
    pipeline.conn.close()                  
    # Close database connection
    pipeline.cur.close()
    pipeline.conn.close()                  