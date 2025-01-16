# edu-rag-v3

## 核心组件

- **RAGProcessor**: 核心处理器，负责查询处理和答案生成
- **VectorSearcher**: 向量检索模块，支持文本和图片的相似度搜索
- **SearchResultFilter**: 搜索结果过滤器，支持智能权重和阈值过滤
- **RAGService**: 服务层封装，提供简化的API接口

## 主要流程
1. 查询输入示例
```python
# 用户输入示例
query_input = {
    "question": "问题内容",
    "selected_books": ["书籍列表"],
    "mode": "answer"  # 或 "knowledge"
}
```

2. 并行执行的主要任务
```python
# 两个主要异步任务
search_task = asyncio.create_task(self._perform_vector_search(...))
direct_answer_task = asyncio.create_task(self._get_direct_answer(...))
```

3. 向量检索流程
```python
# 1. 查询改写（中英双语）
cn_query = await self._rewrite_cn_query(query)
en_query = await self._rewrite_en_query(query)

# 2. 获取embedding并执行检索
embedding = await self._get_embedding(query)
results = await self.search_embeddings(...)
```

4. 结果过滤与整合
```python
# 过滤搜索结果
filtered_results = self.result_filter.filter_results(
    cn_results=cn_results,
    en_results=en_results
)
```

5. 最终答案生成
```python
final_answer = await self._generate_final_answer(
    initial_answer=direct_answer,
    search_results=filtered_results,
    original_question=query.question
)
```
# RAG 系统架构总结

## 1. 主要组件

1. **RAGProcessor**: 核心处理器
   - 负责协调整个处理流程
   - 管理 LLM、向量搜索和结果过滤等组件

2. **LogManager**: 日志管理器
   - 处理日志的保存和组织
   - 按日期创建目录结构

3. **VectorSearcher**: 向量搜索器
   - 处理中英文的向量检索
   - 支持文本和图片搜索

4. **SearchResultFilter**: 结果过滤器
   - 过滤和排序搜索结果
   - 支持配置权重和限制

## 2. 数据结构

1. **输入查询 (UserQuery)**:
```python
@dataclass
class UserQuery:
    question: str           # 用户问题
    selected_books: List[str]  # 选中的书籍列表
    mode: str              # "answer" 或 "knowledge"
```

2. **处理结果 (ProcessedQuery)**:
```python
@dataclass
class ProcessedQuery:
    direct_answer: str     # 初始直接回答
    final_answer: str      # 最终生成的答案
    selected_books: List[str]  # 使用的书籍
    cn_query: str         # 中文检索查询
    en_query: str         # 英文检索查询
    search_results: dict   # 搜索结果
    task_times: dict      # 任务耗时统计
```

## 3. 异步处理流程

1. **主流程 (process_query)**:
```python
async def process_query(input_json: str):
    # 1. 并行执行向量搜索和直接回答
    search_task = asyncio.create_task(_perform_vector_search())
    direct_answer_task = asyncio.create_task(_get_direct_answer())
    
    # 2. 等待两个任务完成
    search_results, direct_answer = await asyncio.gather(
        search_task, 
        direct_answer_task
    )
    
    # 3. 并行执行图片选择和最终答案生成
    image_task = asyncio.create_task(_select_relevant_images())
    final_answer_task = asyncio.create_task(_generate_final_answer())
    
    # 4. 等待最终结果
    image_results, final_answer = await asyncio.gather(
        image_task, 
        final_answer_task
    )
```

2. **向量搜索流程 (_perform_vector_search)**:
```python
async def _perform_vector_search():
    # 并行执行中英文搜索
    cn_results, en_results = await asyncio.gather(
        cn_search_flow(),
        en_search_flow()
    )
```

## 4. 关键异步任务

1. **查询改写和检索**:
   - 中文查询改写和检索
   - 英文查询改写和检索
   - 两个流程并行执行

2. **答案生成**:
   - 初始直接回答生成
   - 最终答案生成（基于检索结果）

3. **图片处理**:
   - 相关图片选择和评估
   - 与最终答案生成并行执行

4. **日志记录**:
   - 异步保存处理日志
   - 不阻塞主流程

## 5. 异步处理流程图

```graph TD
process_query
    [用户输入] --> parse_input
    parse_input --> |UserQuery| 并行任务1
    
    并行任务1
        _perform_vector_search
            cn_search_flow
                _rewrite_cn_query --> vector_search(cn)
            en_search_flow
                _rewrite_en_query --> vector_search(en)
        _get_direct_answer
    
    并行任务1 --> 结果过滤
        _perform_vector_search --> |search_results| 结果过滤
        _get_direct_answer --> |direct_answer| 结果过滤
    
    结果过滤 --> 并行任务2
    
    并行任务2
        _select_relevant_images
            [图片评估] --> [筛选结果]
        _generate_final_answer
            [参考资料整理] --> [生成答案]
    
    并行任务2 --> [最终结果]
    
    [最终结果] --> _save_log{异步}
```

这个系统通过异步处理和并行执行来优化性能，同时保持了良好的模块化结构和错误处理机制。主要通过 `asyncio` 实现并发，使用 `create_task` 和 `gather` 来管理异步任务。
