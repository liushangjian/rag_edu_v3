"""
RAG (检索增强生成) 日志记录器使用说明：

功能特点：
1. 支持按天自动归档日志文件
2. 自动记录每个处理步骤的时间和元数据
3. 支持检索结果记录
4. 支持LLM输入输出记录
5. 包含重试机制，提高可靠性
6. 支持实时进度打印

基本使用方法：
    # 初始化日志记录器
    logger = RAGLogger(
        log_dir="logs",              # 日志保存目录
        auto_save=True,              # 是否自动保存
        date_subfolder=True,         # 是否使用日期子文件夹
        max_retries=3                # 文件操作最大重试次数
    )
    
    # 记录用户查询
    logger.log_query("用户的问题")
    
    # 记录处理步骤
    logger.start_step("步骤名称", metadata={"额外信息": "值"})
    # ... 处理逻辑 ...
    logger.end_step("步骤名称", metadata={"结果": "成功"})
    
    # 记录检索结果
    logger.log_retrieval(
        source="知识库名称",
        total_docs=100,
        retrieved_docs=[{"doc1": "内容1"}, {"doc2": "内容2"}]
    )
    
    # 记录LLM交互
    logger.log_llm(
        llm_input="输入到LLM的内容",
        llm_output="LLM的响应内容"
    )
    
    # 手动保存日志（如果auto_save=False）
    logger.save(filename_prefix="custom_prefix")

日志文件结构：
- logs/                      # 主日志目录
  - 20240315/               # 日期子文件夹（如果启用）
    - rag_log_20240315_123456.json  # 日志文件

注意事项：
1. 建议在with语句中使用，确保资源正确释放
2. 可以使用 ~ 表示用户主目录
3. 所有文件操作都包含重试机制，提高可靠性
4. 日志文件使用JSON格式存储，方便后续分析
"""

import json
import time
from datetime import datetime
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class RAGStep:
    """Data class for recording RAG processing steps"""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    total_docs: int = 0
    retrieved_docs: List[Dict] = None
    metadata: Dict[str, Any] = None

@dataclass
class RAGLogData:
    """Data class for RAG log data"""
    timestamp: str
    query: str
    total_time: float = 0.0
    steps: Dict[str, RAGStep] = None
    retrieval_results: Dict[str, RetrievalResult] = None
    llm_input: str = ""
    llm_output: str = ""
    messages: List[Dict] = None

class RAGLogger:
    """Logger for RAG (Retrieval-Augmented Generation) scenarios"""
    
    def __init__(self, log_dir: str = "logs", auto_save: bool = True, 
                 date_subfolder: bool = True, max_retries: int = 3):
        """
        Initialize RAG logger
        
        Args:
            log_dir: Directory for storing logs
            auto_save: Whether to automatically save logs (when logging ends)
            date_subfolder: Whether to create date-based subfolders
            max_retries: Maximum number of retries for file operations
        """
        self.log_dir = os.path.expanduser(log_dir)  # 支持 ~ 路径
        self.auto_save = auto_save
        self.date_subfolder = date_subfolder
        self.max_retries = max_retries
        self.start_time = time.time()
        
        # Create log directory structure
        self.today = datetime.now().strftime("%Y%m%d")
        self.daily_log_dir = (os.path.join(self.log_dir, self.today) 
                            if date_subfolder else self.log_dir)
        
        # 创建目录时添加重试机制
        for attempt in range(self.max_retries):
            try:
                os.makedirs(self.daily_log_dir, exist_ok=True)
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to create log directory after {self.max_retries} attempts: {e}")
                time.sleep(1)  # 等待1秒后重试

        # Initialize log data
        self.step_times = {}
        self.log_data = RAGLogData(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query="",
            steps={},
            retrieval_results={},
            messages=[]
        )
        
        self.info("RAG Logger initialized successfully")

    def start_step(self, step_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Start recording a processing step
        
        Args:
            step_name: Name of the step
            metadata: Metadata related to the step
        """
        self.step_times[step_name] = time.time()
        self.log_data.steps[step_name] = RAGStep(
            name=step_name,
            start_time=self.step_times[step_name],
            metadata=metadata
        )
        print(f"[{step_name}] Started...")

    def end_step(self, step_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        End recording a processing step
        
        Args:
            step_name: Name of the step
            metadata: Metadata related to the step
        """
        if step_name in self.step_times:
            end_time = time.time()
            duration = end_time - self.step_times[step_name]
            
            step = self.log_data.steps.get(step_name)
            if step:
                step.end_time = end_time
                step.duration = duration
                if metadata:
                    step.metadata = metadata if not step.metadata else {**step.metadata, **metadata}
                    
            print(f"[{step_name}] Completed (Duration: {duration:.2f}s)")

    def log_retrieval(self, 
                     source: str,
                     total_docs: int,
                     retrieved_docs: List[Dict],
                     metadata: Dict[str, Any] = None) -> None:
        """
        Log retrieval results
        
        Args:
            source: Retrieval source (e.g., 'text', 'image')
            total_docs: Total number of documents
            retrieved_docs: List of retrieved documents
            metadata: Metadata related to retrieval
        """
        self.log_data.retrieval_results[source] = RetrievalResult(
            total_docs=total_docs,
            retrieved_docs=retrieved_docs,
            metadata=metadata
        )
        print(f"[{source} Retrieval] Retrieved {len(retrieved_docs)} results from {total_docs} documents")

    def log_llm(self, llm_input: str, llm_output: str) -> None:
        """
        Log LLM interaction
        
        Args:
            llm_input: Input content to LLM
            llm_output: Output content from LLM
        """
        self.log_data.llm_input = llm_input
        self.log_data.llm_output = llm_output
        print(f"[LLM] Generated response (Length: {len(llm_output)})")

    def log_query(self, query: str) -> None:
        """
        Log query content
        
        Args:
            query: User query content
        """
        self.log_data.query = query
        print(f"[Query] {query}")

    def info(self, message: str) -> None:
        """Log information level message"""
        self._log_message("INFO", message)

    def warning(self, message: str) -> None:
        """Log warning level message"""
        self._log_message("WARNING", message)

    def error(self, message: str) -> None:
        """Log error level message"""
        self._log_message("ERROR", message)

    def _log_message(self, level: str, message: str) -> None:
        """Internal method for logging messages"""
        print(f"[{level}] {message}")
        if not self.log_data.messages:
            self.log_data.messages = []
        self.log_data.messages.append({
            "level": level,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def save(self, filename_prefix: str = "rag_log") -> str:
        """
        Save log to file with retry mechanism
        
        Args:
            filename_prefix: Prefix for log filename
            
        Returns:
            str: Path to saved log file
        """
        self.log_data.total_time = time.time() - self.start_time
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(self.daily_log_dir, filename)
        
        # Convert log data to dictionary
        log_dict = asdict(self.log_data)
        
        # Save to file with retry mechanism
        for attempt in range(self.max_retries):
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(log_dict, f, ensure_ascii=False, indent=2)
                print(f"Log saved to: {filepath}")
                return filepath
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to save log file after {self.max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)  # 等待1秒后重试

    def __del__(self):
        """Destructor - save log if auto_save is enabled"""
        if self.auto_save:
            self.save()