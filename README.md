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
