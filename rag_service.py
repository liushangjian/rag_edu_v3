from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path
import asyncio
from rag import RAGProcessor
from search_filter import SearchResultFilter, FilterConfig
import os
import time

@dataclass
class RAGServiceConfig:
    # 向量搜索配置
    text_limit: int = 15
    image_limit: int = 5
    
    # 结果过滤配置
    filtered_text_limit: int = 8
    filtered_image_limit: int = 1
    core_en_weight: float = 1.05
    
    # 日志配置
    log_dir: str = "logs"
    
    # 图片根目录配置
    image_root_dir: str = "/data/figures_download"
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RAGServiceConfig':
        return cls(
            text_limit=config_dict.get("text_limit", 15),
            image_limit=config_dict.get("image_limit", 5),
            filtered_text_limit=config_dict.get("filtered_text_limit", 8),
            filtered_image_limit=config_dict.get("filtered_image_limit", 1),
            core_en_weight=config_dict.get("core_en_weight", 1.05),
            log_dir=config_dict.get("log_dir", "logs"),
            image_root_dir=config_dict.get("image_root_dir", "/data/figures_download")
        )

@dataclass
class RAGResponse:
    final_answer: str
    formatted_references: str
    image_references: List[Dict]

class ImagePathIndex:
    """图片路径索引管理"""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.index = {}
        self.cache_path = os.path.join("cache", "image_path_index.json")
        
    def _build_index(self):
        """构建图片路径索引"""
        print("开始构建图片索引...")
        start_time = time.time()
        count = 0
        
        try:
            for root, _, files in os.walk(self.root_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.index[file] = os.path.join(root, file)
                        count += 1
                        if count % 1000 == 0:
                            print(f"已索引 {count} 个图片...")
            
            duration = time.time() - start_time
            print(f"索引构建完成，共 {count} 个图片，耗时 {duration:.2f} 秒")
            self._save_index()
            
        except Exception as e:
            print(f"构建索引时发生错误: {str(e)}")
            raise
    
    def _save_index(self):
        """保存索引到缓存文件"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
    
    def _load_from_cache(self) -> bool:
        """从缓存加载索引"""
        if not os.path.exists(self.cache_path):
            return False
            
        try:
            cache_mtime = os.path.getmtime(self.cache_path)
            newest_image_mtime = max(
                os.path.getmtime(os.path.join(root, f))
                for root, _, files in os.walk(self.root_dir)
                for f in files
                if f.endswith(('.png', '.jpg', '.jpeg'))
            )
            
            if newest_image_mtime > cache_mtime:
                return False
                
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
            return True
            
        except Exception as e:
            print(f"加载缓存失败: {str(e)}")
            return False
    
    @classmethod
    def initialize(cls, root_dir: str) -> 'ImagePathIndex':
        """初始化图片索引"""
        instance = cls(root_dir)
        if not instance._load_from_cache():
            instance._build_index()
        return instance
    
    def get_path(self, image_name: str) -> str:
        """获取图片完整路径"""
        return self.index.get(image_name, "")

class RAGService:
    def __init__(self, config: Optional[RAGServiceConfig] = None):
        self.config = config or RAGServiceConfig()
        self._processor = None
        self._image_index = None
    
    async def _init_processor(self):
        """延迟初始化RAG处理器和图片索引"""
        if self._processor is None:
            # 配置过滤器
            filter_config = FilterConfig(
                core_en_weight=self.config.core_en_weight,
                text_limit=self.config.filtered_text_limit,
                image_limit=self.config.filtered_image_limit
            )
            
            # 初始化处理器
            self._processor = RAGProcessor()
            self._processor.result_filter = SearchResultFilter(config=filter_config)
            
            # 更新日志目录
            self._processor.log_manager.base_dir = self.config.log_dir
            self._processor.log_manager.ensure_log_directory()
            
            # 初始化图片索引
            self._image_index = ImagePathIndex.initialize(self.config.image_root_dir)
    
    def _process_image_references(self, image_results: List[Dict]) -> List[Dict]:
        """处理图片引用，添加完整路径"""
        processed_refs = []
        for img in image_results:
            image_path = self._image_index.get_path(img["image_path"])
            if image_path:  # 只添加存在的图片
                processed_refs.append({
                    "book_title": img["book_title"],
                    "image_path": image_path,
                    "content": img["content"],
                    "similarity": img["similarity"]
                })
        return processed_refs

    async def process_query(
        self,
        question: str,
        selected_books: Optional[List[str]] = None,
        mode: str = "answer"
    ) -> RAGResponse:
        """处理查询并返回简化的响应"""
        # 确保处理器已初始化
        await self._init_processor()
        
        # 准备查询输入
        query_input = {
            "question": question,
            "selected_books": selected_books or [],
            "mode": mode
        }
        
        # 准备向量搜索配置
        vector_search_config = {
            "text_limit": self.config.text_limit,
            "image_limit": self.config.image_limit
        }
        
        try:
            # 处理查询
            result = await self._processor.process_query(
                json.dumps(query_input),
                vector_search_config=vector_search_config
            )
            
            # 处理参考文献，只包含实际引用的内容
            metadata, _ = self._processor.process_textbook_results(
                result.search_results["text_results"]
            )
            used_citations = self._processor._extract_citations(result.direct_answer)
            formatted_refs = self._processor.format_references(metadata, used_citations)
            
            # 处理图片引用，添加完整路径
            image_refs = self._process_image_references(
                result.search_results["image_results"]
            )
            
            return RAGResponse(
                final_answer=result.direct_answer,
                formatted_references=formatted_refs,
                image_references=image_refs
            )
            
        except Exception as e:
            print(f"处理查询时发生错误: {str(e)}")
            raise 

async def test():
    """测试RAG服务的基本功能"""
    print("开始测试RAG服务...")
    
    # 创建测试配置
    test_config = RAGServiceConfig(
        text_limit=15,
        image_limit=5,
        filtered_text_limit=8,
        filtered_image_limit=1,
        core_en_weight=1.05,
        log_dir="test_logs"
    )
    
    # 初始化服务
    service = RAGService(test_config)
    print("服务初始化完成")
    
    # 测试查询
    test_questions = [
        {
            "question": "7次跨膜蛋白是什么",
            "books": ["细胞生物学", "细胞生物学 王金发"]
        },
        {
            "question": "光合作用的基本过程是什么",
            "books": ["Campbell Biology",'植物生理学 陈春宇']
        },
        {
            "question": "细胞膜的结构是什么",
            "books": []
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n执行测试用例 {i}/{len(test_questions)}")
        print("-" * 50)
        print(f"问题: {test_case['question']}")
        print(f"选择的教材: {', '.join(test_case['books'])}")
        
        try:
            response = await service.process_query(
                question=test_case['question'],
                selected_books=test_case['books']
            )
            
            print("\n最终答案:")
            print("-" * 50)
            print(response.final_answer)
            
            print("\n参考文献:")
            print("-" * 50)
            print(response.formatted_references)
            
            if response.image_references:
                print("\n相关图片:")
                print("-" * 50)
                for img in response.image_references:
                    print(f"书籍: {img['book_title']}")
                    print(f"图片路径: {img['image_path']}")
                    print(f"说明: {img['content']}")
                    print(f"相似度: {img['similarity']:.4f}")
                    print()
                    
        except Exception as e:
            print(f"测试用例 {i} 执行失败: {str(e)}")
            continue
            
        print(f"\n测试用例 {i} 执行完成")
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test()) 