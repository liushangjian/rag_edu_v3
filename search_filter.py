from dataclasses import dataclass
from typing import List, Dict, Any
import json
from collections import defaultdict

@dataclass
class FilterConfig:
    """过滤器配置"""
    core_en_weight: float = 1.5  # 核心英文书籍的权重
    text_limit: int = 10  # 文本结果数量限制
    image_limit: int = 1  # 图片结果数量限制，默认改为1
    image_similarity_threshold: float = 0.6  # 图片相似度阈值
    
    def __post_init__(self):
        # 加载书籍列表
        with open('booklist.json', 'r', encoding='utf-8') as f:
            book_lists = json.load(f)
            self.core_en_books = set(book_lists['core_book_list_en'])

class SearchResultFilter:
    def __init__(self, config: FilterConfig = None):
        """初始化过滤器
        
        Args:
            config: 过滤器配置，如果为None则使用默认配置
        """
        self.config = config or FilterConfig()
        
    def _apply_weights(self, result: Dict[str, Any]) -> float:
        """应用权重到相似度分数
        
        Args:
            result: 单条搜索结果
            
        Returns:
            float: 加权后的相似度分数
        """
        similarity = result.get('similarity', 0)
        book_title = result.get('book_title', '')
        
        # 对核心英文书籍加权
        if book_title in self.config.core_en_books:
            similarity *= self.config.core_en_weight
            
        return similarity
        
    def filter_results(self, cn_results: Dict[str, List], en_results: Dict[str, List]) -> Dict[str, List]:
        """合并并过滤中英文搜索结果
        
        Args:
            cn_results: 中文搜索结果 {"text_results": [...], "image_results": [...]}
            en_results: 英文搜索结果 {"text_results": [...], "image_results": [...]}
            
        Returns:
            Dict[str, List]: 过滤后的结果 {"text_results": [...], "image_results": [...]}
        """
        print("\n开始过滤搜索结果...")
        
        # 合并文本结果
        all_text_results = []
        all_text_results.extend(cn_results.get('text_results', []))
        all_text_results.extend(en_results.get('text_results', []))
        
        # 合并图片结果
        all_image_results = []
        all_image_results.extend(cn_results.get('image_results', []))
        all_image_results.extend(en_results.get('image_results', []))
        
        # 应用权重并排序文本结果
        weighted_text_results = [
            (result, self._apply_weights(result))
            for result in all_text_results
        ]
        weighted_text_results.sort(key=lambda x: x[1], reverse=True)
        
        # 应用权重并排序图片结果，同时应用相似度阈值
        weighted_image_results = [
            (result, self._apply_weights(result))
            for result in all_image_results
            if result.get('similarity', 0) >= self.config.image_similarity_threshold  # 添加相似度阈值过滤
        ]
        weighted_image_results.sort(key=lambda x: x[1], reverse=True)
        
        # 取前N个结果
        filtered_text_results = [
            result for result, _ in weighted_text_results[:self.config.text_limit]
        ]
        filtered_image_results = [
            result for result, _ in weighted_image_results[:self.config.image_limit]
        ]
        
        print(f"文本结果: 从 {len(all_text_results)} 条过滤到 {len(filtered_text_results)} 条")
        print(f"图片结果: 从 {len(all_image_results)} 条过滤到 {len(filtered_image_results)} 条")
        
        # 打印权重信息
        print("\n应用的权重:")
        print(f"核心英文书籍权重: {self.config.core_en_weight}")
        print(f"应用的图片相似度阈值: {self.config.image_similarity_threshold}")
        
        return {
            "text_results": filtered_text_results,
            "image_results": filtered_image_results
        } 