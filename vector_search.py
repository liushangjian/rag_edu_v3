import asyncpg
from langchain_community.embeddings import JinaEmbeddings
import os
from dotenv import load_dotenv
import json
import asyncio
import aiohttp
import backoff
import numpy as np
from typing import List

load_dotenv()

class VectorSearcher:
    def __init__(self, db_config: dict = None):
        """初始化搜索器
        
        Args:
            db_config: 数据库配置，默认使用本地配置
        """
        self.db_config = db_config or {
            "database": "rag_db",
            "user": "edurag_user",
            "host": "localhost"
        }
        
        # 加载书籍列表
        with open('booklist.json', 'r', encoding='utf-8') as f:
            book_lists = json.load(f)
            self.cn_books = book_lists['core_book_list_cn'] + book_lists['supplementary_book_list_cn']
            self.en_books = book_lists['core_book_list_en'] + book_lists['supplementary_book_list_en']
        
        # Jina API配置
        self.jina_api_key = "jina_52307151035d4af595b0edf520b6c748KvnOEwI4JT6dQ1pqwmUlXbxfiHbK"
        self.jina_api_url = "https://api.jina.ai/v1/embeddings"
        
        # 添加所有有效书籍集合
        self.all_valid_books = set(self.cn_books + self.en_books)

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, TimeoutError),
        max_tries=3
    )
    async def _get_embedding(self, text: str) -> List[float]:
        """调用Jina API获取embedding"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }
        
        payload = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.query",
            "dimensions": 1024,
            "late_chunking": True,
            "input": [text]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.jina_api_url,
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status != 200:
                    raise Exception(f"Jina API调用失败: {response.status}")
                
                result = await response.json()
                embedding = result.get('data', [{}])[0].get('embedding', [])
                
                # 验证embedding维度
                embedding_array = np.array(embedding)
                if embedding_array.shape != (1024,):
                    raise ValueError(f"Embedding维度错误: {embedding_array.shape}")
                
                return embedding

    async def search_embeddings(self, pool, table_name: str, vector_str: str, book_titles: list = None, limit: int = 10):
        """异步执行向量搜索"""
        if book_titles:
            sql = """
            SELECT 
                book_title,
                metadata->>'chapter' as chapter,
                metadata->>'section' as section,
                metadata->>'subsection' as subsection,
                metadata->>'content' as content,
                metadata->>'image_path' as image_path,
                1 - (embedding <#> $1::vector) as similarity
            FROM {table_name}
            WHERE book_title = ANY($2)
            ORDER BY embedding <#> $1::vector
            LIMIT {limit};
            """
            params = [vector_str, book_titles]
        else:
            sql = """
            SELECT 
                book_title,
                metadata->>'chapter' as chapter,
                metadata->>'section' as section,
                metadata->>'subsection' as subsection,
                metadata->>'content' as content,
                metadata->>'image_path' as image_path,
                1 - (embedding <#> $1::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <#> $1::vector
            LIMIT {limit};
            """
            params = [vector_str]

        sql = sql.format(table_name=table_name, limit=limit)
        
        async with pool.acquire() as conn:
            results = await conn.fetch(sql, *params)
            return [dict(r) for r in results]

    async def search(
        self,
        query: str,
        language: str,
        custom_book_list: list = None,
        text_limit: int = 10,
        image_limit: int = 10,
    ):
        """执行向量搜索
        
        Args:
            query: 查询文本
            language: 语言代码 ('cn' 或 'en')
            custom_book_list: 可选的自定义书籍列表，覆盖默认语言书籍列表
            text_limit: 文本结果返回数量
            image_limit: 图片结果返回数量
        
        Returns:
            dict: 包含文本和图片搜索结果的字典
        """

        print(f"\nVectorSearcher开始搜索:")
        print(f"查询文本: {query}")
        print(f"语言: {language}")
        print(f"自定义书籍列表: {custom_book_list}")
        print(f"文本结果需求数量: {text_limit}")
        print(f"图片结果需求数量: {image_limit}")

        # 验证自定义书籍列表
        if custom_book_list:
            invalid_books = [book for book in custom_book_list if book not in self.all_valid_books]
            if invalid_books:
                print(f"\n警告: 以下书籍不在系统中，将被忽略:")
                for book in invalid_books:
                    print(f"- {book}")
            
            # 过滤出有效的书籍
            book_list = [book for book in custom_book_list if book in self.all_valid_books]
            if not book_list:
                print(f"\n警告: 没有有效的书籍可搜索，将使用默认书籍列表")
                book_list = self.cn_books if language == 'cn' else self.en_books
        else:
            book_list = self.cn_books if language == 'cn' else self.en_books

        print(f"最终使用的书籍列表: {book_list}")
            
        # 获取查询向量
        try:
            print("开始获取embedding...")
            query_embedding = await self._get_embedding(query)
            vector_str = f"[{','.join(map(str, query_embedding))}]"
        except Exception as e:
            print(f"获取embedding失败: {str(e)}")
            vector_str = f"[{','.join(['0.0'] * 1024)}]"  # fallback到零向量
        
        # 创建数据库连接池
        try:
            print("尝试连接数据库...")
            pool = await asyncpg.create_pool(**self.db_config)
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            raise
        
        try:
            # 并行执行文本和图片搜索
            print("开始执行文本向量搜索...")
            text_results, image_results = await asyncio.gather(
                self.search_embeddings(
                    pool, 
                    "rag_text_embeddings", 
                    vector_str, 
                    book_list, 
                    text_limit
                ),
                self.search_embeddings(
                    pool, 
                    "rag_image_embeddings", 
                    vector_str, 
                    book_list, 
                    image_limit
                )
            )
            
            print(f"文本搜索完成，结果数量: {len(text_results)}")
            print(f"图片搜索完成，结果数量: {len(image_results)}")
            
            return {
                "text_results": text_results,
                "image_results": image_results
            }
            
        finally:
            print("关闭数据库连接池")
            await pool.close()

async def test_search():
    """测试搜索功能"""
    print("初始化搜索器...")
    searcher = VectorSearcher()
    
    # 测试中文搜索
    print("\n测试中文搜索:")
    cn_query = "女性性别决定过程"
    print(f"查询: {cn_query}")
    
    cn_results = await searcher.search(
        query=cn_query,
        language="cn",
        text_limit=3,
        image_limit=2
    )
    
    print("\n文本结果:")
    for i, result in enumerate(cn_results["text_results"], 1):
        print(f"\n结果 {i}:")
        print(f"书名: {result['book_title']}")
        print(f"章节: {result['chapter']}")
        print(f"小节: {result['section']}")
        print(f"相似度: {result['similarity']:.4f}")
    
    print("\n图片结果:")
    for i, result in enumerate(cn_results["image_results"], 1):
        print(f"\n图片 {i}:")
        print(f"书名: {result['book_title']}")
        print(f"章节: {result['chapter']}")
        print(f"相似度: {result['similarity']:.4f}")
    
    # 测试英文搜索
    print("\n\n测试英文搜索:")
    en_query = "How does helicase function in DNA replication?"
    print(f"查询: {en_query}")
    
    en_results = await searcher.search(
        query=en_query,
        language="en",
        text_limit=3,
        image_limit=2
    )
    
    print("\n文本结果:")
    for i, result in enumerate(en_results["text_results"], 1):
        print(f"\n结果 {i}:")
        print(f"书名: {result['book_title']}")
        print(f"章节: {result['chapter']}")
        print(f"小节: {result['section']}")
        print(f"相似度: {result['similarity']:.4f}")
    
    print("\n图片结果:")
    for i, result in enumerate(en_results["image_results"], 1):
        print(f"\n图片 {i}:")
        print(f"书名: {result['book_title']}")
        print(f"章节: {result['chapter']}")
        print(f"相似度: {result['similarity']:.4f}")

if __name__ == "__main__":
    asyncio.run(test_search())