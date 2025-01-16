from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import json
import asyncio
from llm import LLMClient
import logging
from vector_search import VectorSearcher
from collections import defaultdict
from search_filter import SearchResultFilter, FilterConfig
import os
from datetime import datetime
import time


@dataclass
class UserQuery:
    question: str
    selected_books: List[str]
    mode: str  # "answer" or "knowledge"

@dataclass
class ProcessedQuery:
    direct_answer: str
    selected_books: List[str]
    cn_query: str
    en_query: str
    search_results: dict
    task_times: dict

class LogManager:
    def __init__(self, base_dir="logs"):
        self.base_dir = base_dir
        self.ensure_log_directory()
        
    def ensure_log_directory(self):
        """确保日志目录存在"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
    def get_log_path(self):
        """获取当前日期的日志目录路径"""
        current_date = datetime.now().strftime("%Y%m%d")
        log_dir = os.path.join(self.base_dir, current_date)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir
        
    def save_query_log(self, query_data: dict):
        """保存查询日志"""
        log_dir = self.get_log_path()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(log_dir, f"query_{timestamp}.json")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=2)
            print(f"\n日志已保存: {log_file}")
        except Exception as e:
            print(f"保存日志失败: {str(e)}")

class RAGProcessor:
    def __init__(self):
        self.llm = LLMClient()
        self.vector_searcher = VectorSearcher()
        self.result_filter = SearchResultFilter()
        
        # 设置错误日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_error_logs.jsonl', mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 系统提示词
        self.ANSWER_PROMPT = """你是一位专业的生物学教师。请根据学生的问题，提供准确的生物学知识解答。

要求：
1. 回答长度控制在200字左右，并且在开头先回答答案。
2. 使用专业且易懂的语言
3. 只回答与生物学相关的问题
4. 如果问题不是生物相关，直接回复"抱歉，这不是生物学问题。请询问生物学相关的问题。"
5. 不要重复用户的非生物学问题
6. 不要建议用户问其他问题
7. 不要解释为什么不能回答
8. 保持回答简短专业

请回答以下问题："""

        self.KNOWLEDGE_PROMPT = """你是一位专业的生物学知识讲解专家。请对下面的生物学知识点进行详细解释。

要求：
1. 解释长度控制在200字左右
2. 使用专业且易懂的语言
3. 只解释生物学相关知识
4. 如果不是生物知识点，直接回复"抱歉，这不是生物学知识点。请询问生物学相关的知识。"
5. 不要重复用户的非生物学问题
6. 不要建议用户问其他问题
7. 不要解释为什么不能回答
8. 保持回答简短专业

请解释以下知识点："""

        self.QUERY_REWRITE_PROMPT = """请将用户的问题改写成更适合检索的形式。

要求：
1. 保留核心生物学概念
2. 去除无关词语
3. 使用专业术语
4. 长度精简，但包含所有必要信息
5. 只处理生物学相关内容
6. 直接输出改写后的查询语句，不要有任何解释、讨论或多个选项
7. 不要输出"或："、"注："等额外内容
8. 不要使用标点符号（除了必要的连字符）

请改写："""

        # 添加新的系统提示词
        self.FINAL_ANSWER_PROMPT = """你是一位专业的生物学教师。我会给你原始问题、初始答案和一些参考资料。请基于这些信息生成最终的回答。

要求：
1. 主要参考初始答案的内容和结构
2. 参考资料用来补充或验证信息
3. 回答长度控制在300-400字
4. 使用专业且易懂的语言
5. 当引用参考资料时，必须在引用内容后标注来源，格式为：[书名 | 章 | 节 | 小节]，如果没有节和小节，可以只显示到章
6. 只在实际使用了参考资料的内容时才添加引用标注
7. 保持回答的连贯性和逻辑性
8. 只回答与生物学相关的问题
9. 如果问题不是生物相关，直接回复"抱歉，这不是生物学问题。请询问生物学相关的问题。"
10. 不要重复用户的非生物学问题，不要建议用户问其他问题，不要解释为什么不能回答

原始问题：
{original_question}

初始答案：
{initial_answer}

参考资料：
{reference_texts}

请生成最终答案："""

        self.IMAGE_SELECTION_PROMPT = """作为生物学教学图片筛选专家，请评估以下图片与问题和答案的相关性。

原始问题：{question}

初始答案：{answer}

候选图片：
{image_descriptions}

要求：
1. 仔细分析每张图片的描述与问题和答案的关联度
2. 只选择真正相关且有助于理解的图片
3. 如果所有图片都不够相关或不合适，请明确指出"不选择任何图片"
4. 输出格式：
   - 如果选择图片：只输出图片编号，如 "2"
   - 如果不选择：输出 "0"
5. 不要解释原因，只输出数字结果

请评估并选择："""

        # 添加日志管理器
        self.log_manager = LogManager()

        # 加载有效书籍列表
        with open('booklist.json', 'r', encoding='utf-8') as f:
            book_lists = json.load(f)
            self.all_valid_books = set(
                book_lists['core_book_list_cn'] + 
                book_lists['supplementary_book_list_cn'] +
                book_lists['core_book_list_en'] + 
                book_lists['supplementary_book_list_en']
            )

    async def parse_input(self, input_json: str) -> Optional[UserQuery]:
        """解析输入的JSON"""
        try:
            data = json.loads(input_json)
            return UserQuery(
                question=data["question"],
                selected_books=data.get("selected_books", []),
                mode=data["mode"]
            )
        except Exception as e:
            print(f"输入JSON解析错误: {str(e)}")
            return None

    async def process_query(self, input_json: str, vector_search_config: dict = None) -> ProcessedQuery:
        """处理查询"""
        start_time = time.time()
        task_times = {}
        
        query = await self.parse_input(input_json)
        if not query:
            return None

        # 使用传入的向量搜索配置或默认值
        if vector_search_config is None:
            vector_search_config = {
                "text_limit": 10,
                "image_limit": 8
            }

        print(f"\n开始处理查询...")
        print(f"选中的书籍列表: {query.selected_books}")

        # 记录查询改写和检索阶段的实际开始时间
        search_phase_start = time.time()
        
        # 验证选中的书籍
        if query.selected_books:
            invalid_books = [book for book in query.selected_books if book not in self.all_valid_books]
            if invalid_books:
                print(f"\n警告: 以下选中的书籍不在系统中，将被忽略:")
                for book in invalid_books:
                    print(f"- {book}")
                
                # 更新查询中的书籍列表，只保留有效的书籍
                query.selected_books = [book for book in query.selected_books if book in self.all_valid_books]
                
                if not query.selected_books:
                    print(f"\n警告: 没有有效的选中书籍，将使用所有可用书籍")
        
        # 创建所有任务并记录开始时间
        search_task = asyncio.create_task(self._perform_vector_search(query, vector_search_config))
        direct_answer_start = time.time()
        direct_answer_task = asyncio.create_task(self._get_direct_answer(query))
        
        filtered_results = None
        direct_answer = None
        
        # 使用asyncio.as_completed()处理任务完成
        pending = {search_task, direct_answer_task}
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                try:
                    if task == search_task:
                        search_results = task.result()
                        cn_results, en_results = search_results[:2]
                        if len(search_results) > 2:
                            task_times.update(search_results[2])
                        
                        # 记录查询改写和检索阶段的实际总耗时
                        task_times['查询改写和检索总时间'] = time.time() - search_phase_start
                        print(f"\n查询改写和检索阶段完成，总耗时: {task_times['查询改写和检索总时间']:.2f}秒")
                        
                        print("\n开始过滤结果...")
                        filter_start = time.time()
                        filtered_results = self.result_filter.filter_results(
                            cn_results=cn_results,
                            en_results=en_results
                        )
                        task_times['结果过滤时间'] = time.time() - filter_start
                        print(f"结果过滤完成，耗时: {task_times['结果过滤时间']:.2f}秒")
                    
                    elif task == direct_answer_task:
                        direct_answer = task.result()
                        task_times['初始答案生成时间'] = time.time() - direct_answer_start
                        print(f"\n直接回答生成完成，耗时: {task_times['初始答案生成时间']:.2f}秒")
                    
                except Exception as e:
                    print(f"任务执行错误: {str(e)}")
                    self.logger.error(f"任务错误", exc_info=True)
                    if task == search_task:
                        filtered_results = {
                            "text_results": [],
                            "image_results": []
                        }
                    elif task == direct_answer_task:
                        direct_answer = "抱歉，处理过程中出现错误。"

        # 确保两个结果都已获得
        if filtered_results is None:
            filtered_results = {
                "text_results": [],
                "image_results": []
            }
        if direct_answer is None:
            direct_answer = "抱歉，处理过程中出现错误。"

        # 在生成最终答案之前添加图片筛选，这一部分已经能正常运行，但需要装进async任务池
        '''
        print("\n开始评估图片相关性...")
        image_selection_start = time.time()
        filtered_results["image_results"] = await self._select_relevant_images(
            question=query.question,
            initial_answer=direct_answer,
            image_results=filtered_results["image_results"]
        )
        task_times['图片筛选时间'] = time.time() - image_selection_start
        print(f"图片筛选完成，耗时: {task_times['图片筛选时间']:.2f}秒")
        print(f"筛选后保留的图片数量: {len(filtered_results['image_results'])}")
        '''
        
        # 生成最终答案
        final_answer_start = time.time()
        final_answer, citation_tracking = await self._generate_final_answer(
            initial_answer=direct_answer,
            search_results=filtered_results,
            original_question=query.question
        )
        task_times['最终答案生成时间'] = time.time() - final_answer_start
        print(f"\n最终答案生成完成，耗时: {task_times['最终答案生成时间']:.2f}秒")
        
        # 打印引用分析结果
        print("\n引用分析:")
        print("-" * 50)
        print(f"总引用数: {len(citation_tracking['extracted_citations'])}")
        print(f"成功匹配引用: {len(citation_tracking['matched_citations'])}")
        print(f"未匹配引用: {len(citation_tracking['unmatched_citations'])}")
        print(f"引用覆盖率: {citation_tracking['citation_coverage']:.2%}")
        
        # 添加参考文献打印
        print("\n" + "=" * 50)
        print("实际引用的参考文献:")
        print("-" * 50)
        metadata, _ = self.process_textbook_results(filtered_results["text_results"])
        used_citations = self._extract_citations(final_answer)
        references = self.format_references(metadata, used_citations)
        print(references)
        
        print("\n搜索完成，返回结果...")

        # 在处理完成后记录日志
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query_info": {
                "question": query.question,
                "mode": query.mode,
                "selected_books": query.selected_books,
                "vector_search": {
                    "text_recall_limit": vector_search_config["text_limit"],
                    "image_recall_limit": vector_search_config["image_limit"]
                }
            },
            "config": {
                "filter": {
                    "core_en_weight": self.result_filter.config.core_en_weight,
                    "filtered_text_limit": self.result_filter.config.text_limit,
                    "filtered_image_limit": self.result_filter.config.image_limit
                }
            },
            "processing_results": {
                "cn_query": cn_results.get("rewritten_query", ""),
                "en_query": en_results.get("rewritten_query", ""),
                "initial_answer": direct_answer,
                "final_answer": final_answer,
                "references": {
                    "all_references": self.format_references(metadata),
                    "used_references": self.format_references(metadata, used_citations)
                }
            },
            "performance_metrics": {
                "total_time": time.time() - start_time,
                "task_times": task_times
            },
            "citation_analysis": {
                "tracking": citation_tracking,
                "statistics": {
                    "total_citations": len(citation_tracking["extracted_citations"]),
                    "matched_citations": len(citation_tracking["matched_citations"]),
                    "unmatched_citations": len(citation_tracking["unmatched_citations"]),
                    "citation_coverage": citation_tracking["citation_coverage"]
                }
            },
            "filtered_results": {
                "text_references": [
                    {
                        "book": r["book_title"],
                        "chapter": r["chapter"],
                        "section": r["section"],
                        "subsection": r["subsection"],
                        "content": r["content"],
                        "similarity": r["similarity"]
                    } for r in filtered_results["text_results"]
                ],
                "image_references": [
                    {
                        "book": r["book_title"],
                        "image_path": r["image_path"],
                        "content": r["content"],
                        "similarity": r["similarity"]
                    } for r in filtered_results["image_results"]
                ]
            },
            "book_validation": {
                "original_books": query.selected_books,
                "invalid_books": invalid_books if query.selected_books else [],
                "valid_books": [book for book in query.selected_books if book in self.all_valid_books] if query.selected_books else [],
                "using_default_list": not query.selected_books or not [book for book in query.selected_books if book in self.all_valid_books]
            },
            "raw_search_results": {
                "cn": {
                    "text": [
                        {
                            "book": r["book_title"],
                            "chapter": r["chapter"],
                            "section": r["section"],
                            "subsection": r["subsection"],
                            "content": r["content"],
                            "similarity": r["similarity"]
                        } for r in cn_results["text_results"]
                    ],
                    "image": [
                        {
                            "book": r["book_title"],
                            "image_path": r["image_path"],
                            "content": r["content"],
                            "similarity": r["similarity"]
                        } for r in cn_results["image_results"]
                    ]
                },
                "en": {
                    "text": [
                        {
                            "book": r["book_title"],
                            "chapter": r["chapter"],
                            "section": r["section"],
                            "subsection": r["subsection"],
                            "content": r["content"],
                            "similarity": r["similarity"]
                        } for r in en_results["text_results"]
                    ],
                    "image": [
                        {
                            "book": r["book_title"],
                            "image_path": r["image_path"],
                            "content": r["content"],
                            "similarity": r["similarity"]
                        } for r in en_results["image_results"]
                    ]
                }
            }
        }
        
        # 异步保存日志
        asyncio.create_task(self._save_log(log_data))
        
        return ProcessedQuery(
            direct_answer=final_answer,
            selected_books=query.selected_books,
            cn_query=cn_results["text_results"][0]["content"] if cn_results["text_results"] else "",
            en_query=en_results["text_results"][0]["content"] if en_results["text_results"] else "",
            search_results=filtered_results,
            task_times=task_times
        )

    async def _get_direct_answer(self, query: UserQuery) -> str:
        """获取直接回答"""
        prompt = self.ANSWER_PROMPT if query.mode == "answer" else self.KNOWLEDGE_PROMPT
        response = await self.llm.chat(
            model="deepseek-chat",
            prompt=f"{prompt}\n\n{query.question}"
        )
        return response["content"]

    async def _rewrite_cn_query(self, query: UserQuery) -> str:
        """改写中文检索查询"""
        response = await self.llm.chat(
            model="deepseek-chat",
            prompt=f"{self.QUERY_REWRITE_PROMPT}\n\n{query.question}\n\n请用中文改写。"
        )
        return response["content"]

    async def _rewrite_en_query(self, query: UserQuery) -> str:
        """改写英文检索查询"""
        response = await self.llm.chat(
            model="deepseek-chat",
            prompt=f"{self.QUERY_REWRITE_PROMPT}\n\n{query.question}\n\n请用英文改写。"
        )
        return response["content"]

    def process_textbook_results(self, results: List[Dict]) -> Tuple[Dict, str]:
        """处理文本检索结果，返回元数据字典和页面内容字符串"""
        merged_metadata = defaultdict(lambda: defaultdict(lambda: {"sections": {}}))
        page_contents = []
        
        # 为每个结果分配唯一ID
        for idx, result in enumerate(results):
            ref_id = f"ref_{idx}"
            
            book = result.get("book_title", "Unknown Title") or "Unknown Title"
            chapter = result.get("chapter", "Unknown Chapter") or "Unknown Chapter"
            section = result.get("section", "") or ""
            subsection = result.get("subsection", "") or ""
            content = result.get("content", "") or ""
            similarity = result.get("similarity", 0)
            
            # 构建页面内容，添加引用ID
            page_content = [
                f"\n引用ID: {ref_id}",
                f"书名: {book}",
                f"章节: {chapter}"
            ]
            if section:
                page_content.append(f"节: {section}")
            if subsection:
                page_content.append(f"小节: {subsection}")
            page_content.extend([
                f"相似度: {similarity:.4f}",
                "",
                content,
                ""
            ])
            page_contents.append("\n".join(page_content))
            
            # 更新元数据结构，添加引用ID
            if section:
                if section not in merged_metadata[book][chapter]["sections"]:
                    merged_metadata[book][chapter]["sections"][section] = {
                        "subsections": set(),
                        "scores": [],
                        "contents": [],
                        "ref_ids": set()  # 添加引用ID集合
                    }
                
                section_data = merged_metadata[book][chapter]["sections"][section]
                if subsection:
                    section_data["subsections"].add(subsection)
                section_data["scores"].append(similarity)
                section_data["contents"].append(content)
                section_data["ref_ids"].add(ref_id)  # 记录引用ID
            else:
                # 如果没有section，直接存储在chapter级别
                if "scores" not in merged_metadata[book][chapter]:
                    merged_metadata[book][chapter].update({
                        "scores": [],
                        "contents": [],
                        "ref_ids": set()  # 添加引用ID集合
                    })
                merged_metadata[book][chapter]["scores"].append(similarity)
                merged_metadata[book][chapter]["contents"].append(content)
                merged_metadata[book][chapter]["ref_ids"].add(ref_id)  # 记录引用ID
        
        return dict(merged_metadata), "\n".join(page_contents)

    def _extract_citations(self, answer: str) -> set:
        """从答案中提取引用的书籍和章节信息
        
        Args:
            answer: 包含引用标记的答案文本
            
        Returns:
            set: 包含 (book_title, chapter, section, subsection) 元组的集合
        """
        citations = set()
        import re
        citation_pattern = r'\[(.*?)\]'
        matches = re.finditer(citation_pattern, answer)
        
        for match in matches:
            citation = match.group(1).strip()
            parts = [x.strip() for x in citation.split('|')]
            
            # 根据parts的长度构建不同级别的引用
            if len(parts) == 4:  # 书名|章|节|小节
                book, chapter, section, subsection = parts
                citations.add((book, chapter, section, subsection))
            elif len(parts) == 3:  # 书名|章|节
                book, chapter, section = parts
                citations.add((book, chapter, section, None))
            elif len(parts) == 2:  # 书名|章
                book, chapter = parts
                citations.add((book, chapter, None, None))
            else:  # 仅书名
                book = parts[0]
                citations.add((book, None, None, None))
                
        return citations

    def format_references(self, references: dict, used_citations: set = None) -> str:
        """格式化参考文献，采用灵活的引用匹配机制
        
        Args:
            references: 参考文献元数据
            used_citations: 实际被引用的 (book_title, chapter, section, subsection) 集合
        """
        formatted_refs = []
        
        for book, chapters in references.items():
            if not book or book == "Unknown Title":
                continue
            
            # 检查书籍是否被引用（模糊匹配）
            if used_citations is not None:
                book_cited = any(
                    citation[0].lower().replace(' ', '') in book.lower().replace(' ', '') or 
                    book.lower().replace(' ', '') in citation[0].lower().replace(' ', '')
                    for citation in used_citations
                )
            else:
                book_cited = True
                
            if not book_cited:
                continue
                
            formatted_refs.append(f"《{book}》")
            
            for chapter_name, chapter_data in chapters.items():
                if chapter_name == "Unknown Chapter":
                    continue
                
                # 检查章节是否被引用（模糊匹配）
                if used_citations is not None:
                    chapter_cited = any(
                        (citation[0].lower().replace(' ', '') in book.lower().replace(' ', '') or 
                         book.lower().replace(' ', '') in citation[0].lower().replace(' ', '')) and
                        (citation[1] is None or 
                         citation[1].lower().replace(' ', '') in chapter_name.lower().replace(' ', '') or
                         chapter_name.lower().replace(' ', '') in citation[1].lower().replace(' ', ''))
                        for citation in used_citations
                    )
                else:
                    chapter_cited = True
                
                if not chapter_cited:
                    continue
                
                # 处理章节级别
                chapter_scores = chapter_data.get("scores", [])
                if chapter_scores and not chapter_data.get("sections"):
                    avg_score = sum(chapter_scores) / len(chapter_scores)
                    formatted_refs.append(f"  {chapter_name} (相关度: {avg_score:.4f})")
                else:
                    formatted_refs.append(f"  {chapter_name}")
                
                # 处理节级别
                sections = chapter_data.get("sections", {})
                for section_name, section_data in sections.items():
                    if not section_name:
                        continue
                    
                    # 检查节是否被引用（模糊匹配）
                    if used_citations is not None:
                        section_cited = any(
                            (citation[0].lower().replace(' ', '') in book.lower().replace(' ', '') or 
                             book.lower().replace(' ', '') in citation[0].lower().replace(' ', '')) and
                            (citation[1] is None or 
                             citation[1].lower().replace(' ', '') in chapter_name.lower().replace(' ', '') or
                             chapter_name.lower().replace(' ', '') in citation[1].lower().replace(' ', '')) and
                            (citation[2] is None or 
                             citation[2].lower().replace(' ', '') in section_name.lower().replace(' ', '') or
                             section_name.lower().replace(' ', '') in citation[2].lower().replace(' ', ''))
                            for citation in used_citations
                        )
                    else:
                        section_cited = True
                        
                    if not section_cited:
                        continue
                    
                    formatted_refs.append(f"    {section_name}")
                    
                    # 处理小节级别
                    subsections = sorted(section_data.get("subsections", set()))
                    for subsection in subsections:
                        if not subsection:
                            continue
                            
                        # 检查小节是否被引用（模糊匹配）
                        if used_citations is not None:
                            subsection_cited = any(
                                len(citation) > 3 and citation[3] is not None and
                                (citation[3].lower().replace(' ', '') in subsection.lower().replace(' ', '') or
                                 subsection.lower().replace(' ', '') in citation[3].lower().replace(' ', ''))
                                for citation in used_citations
                            )
                        else:
                            subsection_cited = True
                            
                        if not subsection_cited:
                            continue
                            
                        if subsection == subsections[-1] and section_data.get("scores"):
                            avg_score = sum(section_data["scores"]) / len(section_data["scores"])
                            formatted_refs.append(f"      {subsection} (相关度: {avg_score:.4f})")
                        else:
                            formatted_refs.append(f"      {subsection}")
            
            formatted_refs.append("")  # 在每本书后添加空行
        
        return "\n".join(formatted_refs).rstrip()

    async def _generate_final_answer(self, initial_answer: str, search_results: dict, original_question: str) -> Tuple[str, Dict]:
        """生成最终答案，并返回引用追踪信息"""
        # 处理已过滤的搜索结果
        metadata, contents = self.process_textbook_results(
            search_results["text_results"]
        )
        
        # 格式化所有参考文献（用于提供给大模型）
        all_references = self.format_references(metadata)
        
        # 如果没有参考资料，直接返回初始答案
        if not contents.strip():
            return initial_answer, {
                "citation_tracking": {
                    "extracted_citations": [],
                    "matched_citations": [],
                    "unused_references": [],
                    "citation_coverage": 0.0
                }
            }
        
        # 构建提示词
        prompt = self.FINAL_ANSWER_PROMPT.format(
            original_question=original_question,
            initial_answer=initial_answer,
            reference_texts=all_references
        )
        
        try:
            response = await self.llm.chat(
                model="deepseek-chat",
                prompt=prompt
            )
            final_answer = response["content"]
            
            # 提取引用的内容
            used_citations = self._extract_citations(final_answer)
            
            # 构建引用追踪信息
            citation_tracking = {
                "extracted_citations": [
                    {
                        "book": citation[0],
                        "chapter": citation[1],
                        "ref_id": citation[2]
                    }
                    for citation in used_citations
                ],
                "matched_citations": [],
                "unmatched_citations": [],
                "unused_references": []
            }
            
            # 检查每个引用是否匹配到召回结果
            for book, chapters in metadata.items():
                for chapter_name, chapter_data in chapters.items():
                    chapter_refs = chapter_data.get("ref_ids", set())
                    
                    # 检查章节级别的引用
                    chapter_citations = [
                        citation for citation in used_citations
                        if citation[0] == book and citation[1] == chapter_name
                    ]
                    
                    for citation in chapter_citations:
                        # 放宽匹配条件：如果没有ref_id，只要书名和章节匹配即可
                        if citation[2] is None or citation[2] in chapter_refs:
                            citation_tracking["matched_citations"].append({
                                "book": book,
                                "chapter": chapter_name,
                                "ref_id": citation[2],
                                "level": "chapter"
                            })
                        else:
                            citation_tracking["unmatched_citations"].append({
                                "book": book,
                                "chapter": chapter_name,
                                "ref_id": citation[2],
                                "reason": "ref_id_not_found"
                            })
                    
                    # 检查节级别的引用
                    for section_name, section_data in chapter_data.get("sections", {}).items():
                        section_refs = section_data.get("ref_ids", set())
                        section_citations = [
                            citation for citation in used_citations
                            if citation[0] == book and citation[1] == chapter_name and citation[2] in section_refs
                        ]
                        
                        for citation in section_citations:
                            citation_tracking["matched_citations"].append({
                                "book": book,
                                "chapter": chapter_name,
                                "section": section_name,
                                "ref_id": citation[2],
                                "level": "section"
                            })
            
            # 记录未被引用的参考文献
            for book, chapters in metadata.items():
                for chapter_name, chapter_data in chapters.items():
                    all_refs = chapter_data.get("ref_ids", set())
                    for section_data in chapter_data.get("sections", {}).values():
                        all_refs.update(section_data.get("ref_ids", set()))
                    
                    unused_refs = all_refs - {citation[2] for citation in used_citations if citation[2]}
                    if unused_refs:
                        citation_tracking["unused_references"].append({
                            "book": book,
                            "chapter": chapter_name,
                            "unused_ref_ids": list(unused_refs)
                        })
            
            # 计算引用覆盖率
            total_refs = sum(len(chapter_data.get("ref_ids", set())) +
                            sum(len(section_data.get("ref_ids", set()))
                                for section_data in chapter_data.get("sections", {}).values())
                            for chapters in metadata.values()
                            for chapter_data in chapters.values())
            
            matched_refs = len(citation_tracking["matched_citations"])
            citation_tracking["citation_coverage"] = matched_refs / total_refs if total_refs > 0 else 0.0
            
            # 更新日志中的参考文献，只包含被引用的内容
            self.current_references = self.format_references(metadata, used_citations)
            
            return final_answer, citation_tracking
            
        except Exception as e:
            self.logger.error(f"生成最终答案时发生错误: {str(e)}", exc_info=True)
            return initial_answer, {
                "citation_tracking": {
                    "error": str(e),
                    "extracted_citations": [],
                    "matched_citations": [],
                    "unused_references": [],
                    "citation_coverage": 0.0
                }
            }

    async def _perform_vector_search(self, query: UserQuery, vector_search_config: dict) -> Tuple[dict, dict]:
        """执行向量搜索"""
        async def cn_search_flow():
            # 记录中文转写开始时间
            cn_rewrite_start = time.time()
            cn_query = await self._rewrite_cn_query(query)
            cn_rewrite_time = time.time() - cn_rewrite_start
            print(f"\n中文查询改写完成，耗时: {cn_rewrite_time:.2f}秒")
            print(f"中文改写结果: {cn_query}")
            
            # 记录中文搜索开始时间
            cn_search_start = time.time()
            results = await self.vector_searcher.search(
                query=cn_query,
                language="cn",
                custom_book_list=query.selected_books,
                text_limit=vector_search_config['text_limit'],
                image_limit=vector_search_config['image_limit']
            )
            cn_search_time = time.time() - cn_search_start
            print(f"中文向量搜索完成，耗时: {cn_search_time:.2f}秒")
            
            # 添加改写后的查询文本到结果中
            results["rewritten_query"] = cn_query
            
            return results, {"rewrite": cn_rewrite_time, "search": cn_search_time}

        async def en_search_flow():
            # 记录英文转写开始时间
            en_rewrite_start = time.time()
            en_query = await self._rewrite_en_query(query)
            en_rewrite_time = time.time() - en_rewrite_start
            print(f"\n英文查询改写完成，耗时: {en_rewrite_time:.2f}秒")
            print(f"英文改写结果: {en_query}")
            
            # 记录英文搜索开始时间
            en_search_start = time.time()
            results = await self.vector_searcher.search(
                query=en_query,
                language="en",
                custom_book_list=query.selected_books,
                text_limit=vector_search_config['text_limit'],
                image_limit=vector_search_config['image_limit']
            )
            en_search_time = time.time() - en_search_start
            print(f"英文向量搜索完成，耗时: {en_search_time:.2f}秒")
            
            # 添加改写后的查询文本到结果中
            results["rewritten_query"] = en_query
            
            return results, {"rewrite": en_rewrite_time, "search": en_search_time}

        (cn_results, cn_times), (en_results, en_times) = await asyncio.gather(
            cn_search_flow(),
            en_search_flow()
        )
        
        # 记录总的搜索时间统计
        search_times = {
            "cn_rewrite": cn_times["rewrite"],
            "cn_search": cn_times["search"],
            "en_rewrite": en_times["rewrite"],
            "en_search": en_times["search"]
        }
        
        return cn_results, en_results, search_times

    async def _save_log(self, log_data: dict):
        """异步保存日志"""
        try:
            self.log_manager.save_query_log(log_data)
        except Exception as e:
            print(f"保存日志时发生错误: {str(e)}")

    async def _select_relevant_images(self, question: str, initial_answer: str, image_results: List[Dict]) -> List[Dict]:
        """评估并筛选相关图片
        
        Args:
            question: 原始问题
            initial_answer: 初始答案
            image_results: 图片搜索结果列表
        
        Returns:
            List[Dict]: 筛选后的图片结果列表
        """
        if not image_results:
            return []
        
        # 获取top 5图片
        top_images = image_results[:5]
        
        # 构建图片描述列表
        image_descriptions = []
        for i, img in enumerate(top_images, 1):
            desc = (f"{i}. 来自《{img['book_title']}》\n"
                   f"图片描述：{img['content']}")
            image_descriptions.append(desc)
        
        # 构建评估提示词
        prompt = self.IMAGE_SELECTION_PROMPT.format(
            question=question,
            answer=initial_answer,
            image_descriptions="\n\n".join(image_descriptions)
        )
        
        try:
            # 调用LLM评估
            response = await self.llm.chat(
                model="deepseek-chat",
                prompt=prompt
            )
            
            # 解析响应
            selected_index = int(response["content"].strip())
            
            # 根据选择更新结果
            if selected_index > 0 and selected_index <= len(top_images):
                return [top_images[selected_index - 1]]
            return []
            
        except Exception as e:
            print(f"图片评估过程出错: {str(e)}")
            return []

if __name__ == "__main__":
    
    # 统一配置
    CONFIG = {
        # 基本查询配置
        "query": {
            "question": "面对喜欢的女生会有心动的感觉，这是怎么产生的？生理学上发生了什么？",
            "selected_books": [],  # 为空则使用所有书籍
            "mode": "answer"  # "answer" 或 "knowledge"
        },
        
        # 向量搜索配置，注意，中英文各一次的话，最后会召回双倍
        "vector_search": {
            "text_limit": 15,  # 文本搜索结果数量
            "image_limit": 5,  # 初始图片搜索结果数量，保持较大以便后续筛选
        },
        
        # 结果过滤配置
        "filter": {
            "text_limit": 8,  # 过滤后保留的文本结果数量
            "image_limit": 5,  # 保留五张图片用于筛选
            "core_en_weight": 1.05,  # 核心英文书籍权重
            "image_similarity_threshold": 0.6  # 图片相似度阈值
        },
        "logging": {
            "print_details": True  # 是否打印详细信息
        }
    }
    
    async def test():
        """测试搜索功能"""
        # 记录开始时间
        start_time = time.time()
        task_times = {}
        
        # 读取书单
        with open("booklist.json", "r", encoding="utf-8") as f:
            book_data = json.load(f)
            all_books = (
                book_data["core_book_list_cn"] + 
                book_data["core_book_list_en"] + 
                book_data["supplementary_book_list_cn"] + 
                book_data["supplementary_book_list_en"]
            )
        
        # 使用配置构建测试输入
        test_input = CONFIG["query"]
        
        # 初始化处理器，传入过滤配置
        filter_config = FilterConfig(
            core_en_weight=CONFIG["filter"]["core_en_weight"],
            text_limit=CONFIG["filter"]["text_limit"],
            image_limit=CONFIG["filter"]["image_limit"]
        )
        
        processor = RAGProcessor()
        processor.vector_searcher = VectorSearcher()
        processor.result_filter = SearchResultFilter(config=filter_config)
        
        print("开始处理查询...\n")
        print(f"原始问题: {test_input['question']}")
        print(f"模式: {test_input['mode']}")
        print(f"\n搜索配置:")
        print(f"文本结果数量: {CONFIG['vector_search']['text_limit']}")
        print(f"图片结果数量: {CONFIG['vector_search']['image_limit']}")
        print(f"过滤后文本数量: {CONFIG['filter']['text_limit']}")
        print(f"过滤后图片数量: {CONFIG['filter']['image_limit']}")
        print(f"核心英文书籍权重: {CONFIG['filter']['core_en_weight']}\n")
        
        # 处理查询并记录时间
        task_start = time.time()
        result = await processor.process_query(
            json.dumps(test_input),
            vector_search_config=CONFIG["vector_search"]
        )
        task_times["总处理时间"] = time.time() - task_start
        
        if CONFIG["logging"]["print_details"]:
            # 打印结果
            print("\n" + "=" * 50)
            print("初始直接回答:")
            print("-" * 50)
            print(result.direct_answer)
            
            print("\n" + "=" * 50)
            print("中文检索查询:")
            print("-" * 50)
            print(result.cn_query)
            
            print("\n" + "=" * 50)
            print("英文检索查询:")
            print("-" * 50)
            print(result.en_query)
            
            print("\n" + "=" * 50)
            print("最终生成答案:")
            print("-" * 50)
            print(result.direct_answer)
            
            # 添加参考文献打印
            print("\n" + "=" * 50)
            print("实际引用的参考文献:")
            print("-" * 50)
            metadata, _ = processor.process_textbook_results(result.search_results["text_results"])
            used_citations = processor._extract_citations(result.direct_answer)
            references = processor.format_references(metadata, used_citations)
            print(references)

        # 在最后添加详细的时间统计输出
        print("\n" + "=" * 50)
        print("执行时间统计:")
        print("-" * 50)
        total_time = time.time() - start_time
        print(f"总运行时间: {total_time:.2f}秒")
        print("\n各步骤详细耗时:")
        print(f"1. 查询改写和检索阶段: {result.task_times.get('查询改写和检索总时间', 0):.2f}秒")
        print(f"   详细分解:")
        print(f"   - 中文查询改写: {result.task_times.get('cn_rewrite', 0):.2f}秒")
        print(f"   - 中文向量检索: {result.task_times.get('cn_search', 0):.2f}秒")
        print(f"   - 英文查询改写: {result.task_times.get('en_rewrite', 0):.2f}秒")
        print(f"   - 英文向量检索: {result.task_times.get('en_search', 0):.2f}秒")
        print(f"2. 结果处理:")
        print(f"   - 初始答案生成: {result.task_times.get('初始答案生成时间', 0):.2f}秒")
        print(f"   - 结果过滤时间: {result.task_times.get('结果过滤时间', 0):.2f}秒")
        print(f"   - 最终答案生成: {result.task_times.get('最终答案生成时间', 0):.2f}秒")

    # 运行测试
    asyncio.run(test())


