# DSPy代码示例和最佳实践模板

## 📚 代码库概览

本文档提供了DSPy开发的全套代码示例和最佳实践模板，包括：
- 🏗️ **基础模板**：快速开始的标准模板
- 🚀 **进阶模式**：复杂应用的高级模式
- 🛠️ **工具类库**：常用的工具函数和类
- 📊 **最佳实践**：开发规范和性能优化

---

## 🏗️ 基础模板

### 1. DSPy项目初始化模板

#### 项目结构
```
dspy_project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging.conf
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_module.py
│   │   └── utils.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── retrieval.py
│   │   └── generation.py
│   └── pipelines/
│       ├── __init__.py
│       └── rag_pipeline.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
├── tests/
│   ├── __init__.py
│   ├── test_modules/
│   └── test_pipelines/
├── notebooks/
│   ├── 01_getting_started.ipynb
│   └── 02_advanced_usage.ipynb
└── scripts/
    ├── train.py
    ├── evaluate.py
    └── deploy.py
```

#### requirements.txt
```txt
# DSPy核心
dspy-ai>=2.0.0

# 机器学习
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# 网络请求
requests>=2.31.0
aiohttp>=3.8.0

# 数据处理
pydantic>=2.0.0
python-dotenv>=1.0.0

# 向量数据库
chromadb>=0.4.0
faiss-cpu>=1.7.0

# 监控和日志
structlog>=23.0.0
prometheus-client>=0.17.0

# 开发工具
pytest>=7.4.0
black>=23.0.0
ruff>=0.0.280
mypy>=1.5.0

# Jupyter支持
jupyter>=1.0.0
ipywidgets>=8.0.0
```

#### setup.py
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dspy-project",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A DSPy project template",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dspy-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.5.0",
        ],
    },
)
```

#### config/settings.py
```python
from typing import Dict, Any, Optional
import os
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class DSPySettings(BaseSettings):
    """DSPy应用配置"""

    # 基础配置
    app_name: str = Field(default="DSPy Application")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # 模型配置
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    default_model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2000)
    temperature: float = Field(default=0.7)

    # 数据库配置
    database_url: str = Field(default="sqlite:///./data/dspy.db")

    # 缓存配置
    cache_type: str = Field(default="memory")  # memory, redis, file
    cache_ttl: int = Field(default=3600)  # 秒

    # 日志配置
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)

    # 性能配置
    max_workers: int = Field(default=4)
    timeout_seconds: int = Field(default=30)

    class Config:
        env_file = ".env"
        case_sensitive = False

class DSPyConfig:
    """DSPy配置管理器"""

    def __init__(self):
        self.settings = DSPySettings()
        self._configure_dspy()

    def _configure_dspy(self):
        """配置DSPy全局设置"""
        import dspy

        # 配置语言模型
        lm = dspy.OpenAI(
            model=self.settings.default_model,
            api_key=self.settings.openai_api_key,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature
        )

        dspy.settings.configure(
            lm=lm,
            rm=None,  # 将在需要时配置检索器
        )

    def get_config(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self.settings.dict()

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)

        self._configure_dspy()  # 重新配置DSPy

# 全局配置实例
config = DSPyConfig()
```

#### src/core/base_module.py
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import time
import logging
from dataclasses import dataclass

import dspy
from dspy import Example, Prediction

logger = logging.getLogger(__name__)

@dataclass
class ModuleMetrics:
    """模块性能指标"""
    execution_count: int = 0
    total_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    avg_time: float = 0.0
    success_rate: float = 0.0

class BaseDSPyModule(dspy.Module, ABC):
    """DSPy模块基类"""

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.metrics = ModuleMetrics()
        self._start_time = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> Prediction:
        """子类必须实现的前向传播方法"""
        pass

    def __call__(self, *args, **kwargs) -> Prediction:
        """带性能监控的调用方法"""
        self._start_time = time.time()

        try:
            result = self.forward(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_error(e)
            logger.error(f"模块 {self.name} 执行失败: {e}")
            raise

    def _record_success(self):
        """记录成功执行"""
        if self._start_time:
            execution_time = time.time() - self._start_time
            self.metrics.execution_count += 1
            self.metrics.total_time += execution_time
            self.metrics.success_count += 1
            self._update_metrics()

    def _record_error(self, error: Exception):
        """记录执行错误"""
        if self._start_time:
            execution_time = time.time() - self._start_time
            self.metrics.execution_count += 1
            self.metrics.total_time += execution_time
            self.metrics.error_count += 1
            self._update_metrics()

    def _update_metrics(self):
        """更新性能指标"""
        if self.metrics.execution_count > 0:
            self.metrics.avg_time = self.metrics.total_time / self.metrics.execution_count
            self.metrics.success_rate = self.metrics.success_count / self.metrics.execution_count

    def get_metrics(self) -> ModuleMetrics:
        """获取性能指标"""
        return self.metrics

    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = ModuleMetrics()

class ConfigurableModule(BaseDSPyModule):
    """可配置的DSPy模块"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.config = config or {}
        self._load_config()

    def _load_config(self):
        """加载配置"""
        # 子类可以重写此方法来加载特定配置
        pass

    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        self._load_config()

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

class CachedModule(ConfigurableModule):
    """带缓存的模块"""

    def __init__(self, name: str, cache_size: int = 100, **kwargs):
        super().__init__(name, **kwargs)
        self.cache_size = cache_size
        self.cache: Dict[str, Prediction] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        import hashlib
        import json

        cache_data = {
            'args': args,
            'kwargs': kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def __call__(self, *args, **kwargs) -> Prediction:
        """带缓存的调用"""
        cache_key = self._generate_cache_key(*args, **kwargs)

        # 检查缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"缓存命中: {self.name}")
            return self.cache[cache_key]

        # 执行并缓存结果
        result = super().__call__(*args, **kwargs)

        # 缓存管理
        if len(self.cache) >= self.cache_size:
            # 简单的LRU：删除第一个
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[cache_key] = result
        self.cache_misses += 1

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
```

---

## 🚀 进阶模式

### 2. 企业级RAG系统模板

#### src/pipelines/enterprise_rag.py
```python
from typing import List, Dict, Any, Optional, Union
import asyncio
from dataclasses import dataclass
from enum import Enum

import dspy
from dspy import Example, Prediction

from ..core.base_module import ConfigurableModule, CachedModule
from ..modules.retrieval import HybridRetriever
from ..modules.generation import EnhancedGenerator

class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"

@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_type: QueryType
    complexity: float  # 0-1
    key_entities: List[str]
    intent: str
    confidence: float

@dataclass
class RetrievalResult:
    """检索结果"""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]

class QueryAnalyzer(CachedModule):
    """查询分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("query_analyzer", config)

        # 初始化DSPy组件
        self.classifier = dspy.Predict(ClassifyQueryType)
        self.entity_extractor = dspy.Predict(ExtractEntities)
        self.complexity_analyzer = dspy.Predict(AnalyzeComplexity)

    def forward(self, query: str) -> Prediction:
        """分析查询"""
        # 分类查询类型
        type_result = self.classifier(query=query)

        # 提取实体
        entity_result = self.entity_extractor(query=query)

        # 分析复杂度
        complexity_result = self.complexity_analyzer(query=query)

        # 构建分析结果
        analysis = QueryAnalysis(
            query_type=QueryType(type_result.query_type.lower()),
            complexity=complexity_result.complexity_score,
            key_entities=entity_result.entities,
            intent=type_result.intent,
            confidence=type_result.confidence
        )

        return Prediction(
            analysis=analysis,
            query_type=analysis.query_type.value,
            complexity=analysis.complexity,
            entities=analysis.key_entities
        )

class AdaptiveRetriever(ConfigurableModule):
    """自适应检索器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("adaptive_retriever", config)

        # 初始化多种检索器
        self.semantic_retriever = HybridRetriever(
            model_name=self.config.get("semantic_model", "all-MiniLM-L6-v2"),
            index_path=self.config.get("index_path", "./data/semantic_index")
        )

        self.keyword_retriever = HybridRetriever(
            retrieval_type="keyword",
            index_path=self.config.get("keyword_index", "./data/keyword_index")
        )

        # 重排序器
        self.reranker = dspy.Predict(RerankDocuments)

    def forward(self, query: str, analysis: QueryAnalysis, k: int = 10) -> Prediction:
        """自适应检索"""
        # 根据查询类型选择检索策略
        if analysis.query_type == QueryType.FACTUAL:
            results = self._factual_retrieval(query, analysis, k)
        elif analysis.query_type == QueryType.ANALYTICAL:
            results = self._analytical_retrieval(query, analysis, k)
        else:
            results = self._hybrid_retrieval(query, analysis, k)

        # 重排序
        if len(results.documents) > k:
            results = self._rerank_documents(query, results, k)

        return Prediction(
            documents=results.documents[:k],
            scores=results.scores[:k],
            retrieval_strategy=results.metadata.get("strategy", "hybrid")
        )

    def _factual_retrieval(self, query: str, analysis: QueryAnalysis, k: int) -> RetrievalResult:
        """事实性检索 - 偏向语义检索"""
        semantic_results = self.semantic_retriever.search(query, k=int(k * 1.5))
        keyword_results = self.keyword_retriever.search(query, k=int(k * 0.5))

        # 合并结果，语义检索权重更高
        combined = self._merge_results(
            semantic_results, keyword_results,
            semantic_weight=0.7, keyword_weight=0.3
        )

        combined.metadata["strategy"] = "semantic_weighted"
        return combined

    def _analytical_retrieval(self, query: str, analysis: QueryAnalysis, k: int) -> RetrievalResult:
        """分析性检索 - 平衡多种检索方式"""
        semantic_results = self.semantic_retriever.search(query, k=k)
        keyword_results = self.keyword_retriever.search(query, k=k)

        # 基于实体扩展查询
        expanded_queries = self._expand_with_entities(query, analysis.key_entities)
        expanded_results = []

        for expanded_query in expanded_queries[:3]:
            result = self.semantic_retriever.search(expanded_query, k=int(k * 0.3))
            expanded_results.append(result)

        # 合并所有结果
        all_results = [semantic_results, keyword_results] + expanded_results
        combined = self._merge_multiple_results(all_results)

        combined.metadata["strategy"] = "entity_expanded"
        return combined

    def _hybrid_retrieval(self, query: str, analysis: QueryAnalysis, k: int) -> RetrievalResult:
        """混合检索"""
        semantic_results = self.semantic_retriever.search(query, k=k)
        keyword_results = self.keyword_retriever.search(query, k=k)

        # 平衡权重
        combined = self._merge_results(
            semantic_results, keyword_results,
            semantic_weight=0.5, keyword_weight=0.5
        )

        combined.metadata["strategy"] = "balanced_hybrid"
        return combined

    def _merge_results(self, semantic_results, keyword_results,
                      semantic_weight: float, keyword_weight: float) -> RetrievalResult:
        """合并两种检索结果"""
        # 简化的合并逻辑
        all_docs = semantic_results.documents + keyword_results.documents
        all_scores = ([s * semantic_weight for s in semantic_results.scores] +
                     [s * keyword_weight for s in keyword_results.scores])

        # 按分数排序
        indexed_results = list(zip(all_docs, all_scores))
        indexed_results.sort(key=lambda x: x[1], reverse=True)

        documents, scores = zip(*indexed_results) if indexed_results else ([], [])

        return RetrievalResult(
            documents=list(documents),
            scores=list(scores),
            metadata={"merged_from": ["semantic", "keyword"]}
        )

    def _expand_with_entities(self, query: str, entities: List[str]) -> List[str]:
        """基于实体扩展查询"""
        expanded_queries = [query]

        for entity in entities[:3]:  # 最多扩展3个实体
            # 简单的扩展策略
            expanded = f"{query} {entity}"
            expanded_queries.append(expanded)

        return expanded_queries

    def _rerank_documents(self, query: str, results: RetrievalResult, k: int) -> RetrievalResult:
        """重排序文档"""
        if len(results.documents) <= k:
            return results

        # 准备重排序输入
        documents_text = [doc.get("content", str(doc)) for doc in results.documents]

        # 调用重排序器
        rerank_result = self.reranker(
            query=query,
            documents=documents_text
        )

        # 应用重排序结果
        if hasattr(rerank_result, 'indices') and rerank_result.indices:
            reranked_docs = [results.documents[i] for i in rerank_result.indices[:k]]
            reranked_scores = [rerank_result.scores[i] if hasattr(rerank_result, 'scores')
                             else results.scores[i] for i in rerank_result.indices[:k]]
        else:
            # 如果重排序失败，保持原排序
            reranked_docs = results.documents[:k]
            reranked_scores = results.scores[:k]

        return RetrievalResult(
            documents=reranked_docs,
            scores=reranked_scores,
            metadata={**results.metadata, "reranked": True}
        )

class EnterpriseRAG(ConfigurableModule):
    """企业级RAG系统"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("enterprise_rag", config)

        # 初始化组件
        self.query_analyzer = QueryAnalyzer(self.config.get("query_analyzer", {}))
        self.retriever = AdaptiveRetriever(self.config.get("retriever", {}))
        self.generator = EnhancedGenerator(self.config.get("generator", {}))
        self.evaluator = dspy.Predict(EvaluateAnswer)

        # 性能配置
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)

    def forward(self, query: str, context_info: Optional[Dict[str, Any]] = None) -> Prediction:
        """完整的RAG流程"""
        # 1. 查询分析
        analysis_result = self.query_analyzer(query)
        analysis = analysis_result.analysis

        # 2. 自适应检索
        retrieval_result = self.retriever(query, analysis)

        # 3. 上下文构建
        context = self._build_context(retrieval_result.documents)

        # 4. 答案生成
        generation_result = self.generator(
            query=query,
            context=context,
            query_type=analysis.query_type.value,
            complexity=analysis.complexity
        )

        # 5. 答案评估
        evaluation_result = self.evaluator(
            query=query,
            answer=generation_result.answer,
            context=context
        )

        # 6. 结果整合
        final_result = Prediction(
            answer=generation_result.answer,
            confidence=evaluation_result.confidence,
            sources=[doc.get("id", f"doc_{i}") for i, doc in enumerate(retrieval_result.documents)],
            reasoning=generation_result.reasoning,
            query_analysis=analysis,
            retrieval_metadata=retrieval_result.metadata,
            evaluation=evaluation_result.evaluation
        )

        # 7. 质量检查
        if final_result.confidence < self.confidence_threshold:
            final_result.answer += "\n\n注意：此答案的置信度较低，建议寻求更多来源验证。"

        return final_result

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        context_parts = []
        current_length = 0

        for doc in documents:
            doc_content = doc.get("content", str(doc))

            # 检查长度限制
            if current_length + len(doc_content) > self.max_context_length:
                # 截断文档
                remaining_space = self.max_context_length - current_length - 50
                if remaining_space > 100:
                    doc_content = doc_content[:remaining_space] + "..."
                else:
                    break

            context_parts.append(f"文档{len(context_parts)+1}: {doc_content}")
            current_length += len(doc_content)

        return "\n\n".join(context_parts)

    async def async_forward(self, query: str, context_info: Optional[Dict[str, Any]] = None) -> Prediction:
        """异步版本的RAG流程"""
        # 这里可以使用异步的检索和生成组件
        loop = asyncio.get_event_loop()

        # 异步执行各步骤
        analysis_task = loop.run_in_executor(None, self.query_analyzer, query)
        analysis_result = await analysis_task

        retrieval_task = loop.run_in_executor(
            None, self.retriever, query, analysis_result.analysis
        )
        retrieval_result = await retrieval_task

        context = self._build_context(retrieval_result.documents)

        generation_task = loop.run_in_executor(
            None, self.generator, query, context,
            analysis_result.analysis.query_type.value,
            analysis_result.analysis.complexity
        )
        generation_result = await generation_task

        # 后续步骤...

        return Prediction(answer=generation_result.answer)

# DSPy签名定义
class ClassifyQueryType(dspy.Signature):
    """查询类型分类"""
    query = dspy.InputField(desc="用户查询")
    query_type = dspy.OutputField(desc="查询类型", choices=["factual", "procedural", "analytical", "creative"])
    intent = dspy.OutputField(desc="用户意图")
    confidence = dspy.OutputField(desc="分类置信度", type=float)

class ExtractEntities(dspy.Signature):
    """实体提取"""
    query = dspy.InputField(desc="用户查询")
    entities = dspy.OutputField(desc="关键实体列表", type=List[str])

class AnalyzeComplexity(dspy.Signature):
    """复杂度分析"""
    query = dspy.InputField(desc="用户查询")
    complexity_score = dspy.OutputField(desc="复杂度评分(0-1)", type=float)

class RerankDocuments(dspy.Signature):
    """文档重排序"""
    query = dspy.InputField(desc="查询")
    documents = dspy.InputField(desc="文档列表", type=List[str])
    indices = dspy.OutputField(desc="排序后的索引", type=List[int])
    scores = dspy.OutputField(desc="重排序分数", type=List[float])

class EvaluateAnswer(dspy.Signature):
    """答案评估"""
    query = dspy.InputField(desc="原始查询")
    answer = dspy.InputField(desc="生成的答案")
    context = dspy.InputField(desc="参考上下文")
    evaluation = dspy.OutputField(desc="评估结果")
    confidence = dspy.OutputField(desc="答案置信度", type=float)
```

### 3. 多智能体协作系统模板

#### src/modules/multi_agent.py
```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

import dspy
from dspy import Example, Prediction

from ..core.base_module import BaseDSPyModule

class AgentRole(Enum):
    """智能体角色"""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentTask:
    """智能体任务"""
    id: str
    agent_id: str
    role: AgentRole
    description: str
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 0
    timeout: Optional[float] = None
    max_retries: int = 3

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    agent_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0

class BaseAgent(BaseDSPyModule):
    """基础智能体"""

    def __init__(self, agent_id: str, role: AgentRole, capabilities: List[str] = None, **kwargs):
        super().__init__(f"agent_{agent_id}", **kwargs)
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities or []
        self.current_tasks = {}
        self.task_history = []

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> TaskResult:
        """执行任务"""
        pass

    def can_handle_task(self, task: AgentTask) -> bool:
        """检查是否能处理任务"""
        return (task.role == self.role or
                any(cap in task.description.lower() for cap in self.capabilities))

    def get_workload(self) -> int:
        """获取当前工作负载"""
        return len(self.current_tasks)

class ResearchAgent(BaseAgent):
    """研究智能体"""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentRole.RESEARCHER,
                        capabilities=["research", "search", "analyze"], **kwargs)

        self.retriever = dspy.Retrieve(k=10)
        self.analyzer = dspy.Predict(AnalyzeTopic)

    async def execute_task(self, task: AgentTask) -> TaskResult:
        """执行研究任务"""
        start_time = time.time()

        try:
            # 分析主题
            topic_analysis = self.analyzer(topic=task.description)

            # 检索相关信息
            search_results = self.retriever(topic_analysis.keywords)

            # 整合研究结果
            research_result = {
                "topic": task.description,
                "analysis": topic_analysis.analysis,
                "findings": search_results.passages,
                "sources": [f"source_{i}" for i in range(len(search_results.passages))],
                "summary": self._generate_summary(search_results.passages)
            }

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result=research_result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    def _generate_summary(self, findings: List[str]) -> str:
        """生成研究摘要"""
        if not findings:
            return "暂无相关研究发现"

        # 简单的摘要生成
        return f"基于研究，发现了{len(findings)}个相关信息点。主要发现包括：{findings[0][:100]}..."

class WritingAgent(BaseAgent):
    """写作智能体"""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentRole.WRITER,
                        capabilities=["writing", "composition", "editing"], **kwargs)

        self.outliner = dspy.Predict(CreateOutline)
        self.writer = dspy.ChainOfThought(GenerateContent)
        self.editor = dspy.Predict(EditContent)

    async def execute_task(self, task: AgentTask) -> TaskResult:
        """执行写作任务"""
        start_time = time.time()

        try:
            research_data = task.input_data.get("research", {})
            topic = task.input_data.get("topic", task.description)

            # 创建大纲
            outline = self.outliner(topic=topic, research=research_data)

            # 生成内容
            content = self.writer(
                topic=topic,
                outline=outline.outline,
                research=research_data,
                style=task.input_data.get("style", "professional")
            )

            # 编辑优化
            edited_content = self.editor(
                original_content=content.content,
                style=task.input_data.get("style", "professional"),
                target_audience=task.input_data.get("audience", "general")
            )

            writing_result = {
                "topic": topic,
                "outline": outline.outline,
                "content": edited_content.edited_content,
                "word_count": len(edited_content.edited_content.split()),
                "style": task.input_data.get("style", "professional"),
                "sources_used": research_data.get("sources", [])
            }

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result=writing_result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

class MultiAgentOrchestrator(BaseDSPyModule):
    """多智能体协调器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__("multi_agent_orchestrator", **kwargs)
        self.config = config or {}

        # 初始化智能体
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: Dict[str, TaskResult] = {}

        # 任务调度器
        self.scheduler = TaskScheduler(self.config.get("scheduler", {}))

        # 执行器
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 5)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)

    def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        self.agents[agent.agent_id] = agent

    def submit_task(self, task: AgentTask) -> str:
        """提交任务"""
        self.task_queue.append(task)
        return task.id

    async def execute_workflow(self, workflow: "Workflow") -> Dict[str, Any]:
        """执行工作流"""
        # 创建工作流任务
        workflow_tasks = self._create_workflow_tasks(workflow)

        # 提交所有任务
        task_ids = []
        for task in workflow_tasks:
            task_id = self.submit_task(task)
            task_ids.append(task_id)

        # 执行任务调度
        results = await self.scheduler.schedule_and_execute(
            self.task_queue, self.agents, self.executor
        )

        # 收集结果
        workflow_results = {}
        for task_id in task_ids:
            if task_id in results:
                workflow_results[task_id] = results[task_id]

        return workflow_results

    def _create_workflow_tasks(self, workflow: "Workflow") -> List[AgentTask]:
        """根据工作流创建任务"""
        tasks = []

        for step in workflow.steps:
            task = AgentTask(
                id=f"{workflow.id}_{step.id}",
                agent_id=step.agent_id,
                role=step.role,
                description=step.description,
                input_data=step.input_data,
                dependencies=step.dependencies,
                priority=step.priority
            )
            tasks.append(task)

        return tasks

class TaskScheduler:
    """任务调度器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduling_strategy = config.get("strategy", "priority")

    async def schedule_and_execute(self, tasks: List[AgentTask],
                                 agents: Dict[str, BaseAgent],
                                 executor: ThreadPoolExecutor) -> Dict[str, TaskResult]:
        """调度和执行任务"""
        results = {}
        pending_tasks = tasks.copy()
        running_tasks = {}

        while pending_tasks or running_tasks:
            # 检查已完成的任务
            completed_task_ids = []
            for task_id, future in running_tasks.items():
                if future.done():
                    try:
                        result = await asyncio.wrap_future(future)
                        results[task_id] = result
                        completed_task_ids.append(task_id)
                    except Exception as e:
                        results[task_id] = TaskResult(
                            task_id=task_id,
                            agent_id="unknown",
                            status=TaskStatus.FAILED,
                            error=str(e)
                        )
                        completed_task_ids.append(task_id)

            # 移除已完成的任务
            for task_id in completed_task_ids:
                del running_tasks[task_id]

            # 调度新任务
            scheduled_tasks = self._schedule_tasks(pending_tasks, agents, running_tasks)

            for task in scheduled_tasks:
                agent = agents.get(task.agent_id)
                if agent and agent.can_handle_task(task):
                    future = executor.submit(asyncio.run, agent.execute_task(task))
                    running_tasks[task.id] = future
                    pending_tasks.remove(task)

            # 短暂等待避免忙循环
            await asyncio.sleep(0.1)

        return results

    def _schedule_tasks(self, pending_tasks: List[AgentTask],
                       agents: Dict[str, BaseAgent],
                       running_tasks: Dict[str, Any]) -> List[AgentTask]:
        """调度任务"""
        if self.scheduling_strategy == "priority":
            # 按优先级排序
            pending_tasks.sort(key=lambda t: t.priority, reverse=True)

        scheduled_tasks = []

        for task in pending_tasks:
            # 检查依赖
            if self._dependencies_satisfied(task, running_tasks):
                # 检查代理可用性
                agent = agents.get(task.agent_id)
                if agent and agent.get_workload() < 3:  # 限制并发任务数
                    scheduled_tasks.append(task)

        return scheduled_tasks

    def _dependencies_satisfied(self, task: AgentTask, running_tasks: Dict[str, Any]) -> bool:
        """检查任务依赖是否满足"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            if dep_id in running_tasks:
                return False  # 依赖任务仍在运行

        return True

@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str
    agent_id: str
    role: AgentRole
    description: str
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 0

@dataclass
class Workflow:
    """工作流定义"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]

    @classmethod
    def research_and_write(cls, topic: str, style: str = "professional") -> "Workflow":
        """创建研究和写作工作流"""
        return Workflow(
            id=f"research_write_{int(time.time())}",
            name=f"研究和写作：{topic}",
            description=f"对{topic}进行深入研究并撰写专业文章",
            steps=[
                WorkflowStep(
                    id="research",
                    agent_id="researcher_1",
                    role=AgentRole.RESEARCHER,
                    description=f"研究{topic}相关信息",
                    input_data={"topic": topic},
                    priority=10
                ),
                WorkflowStep(
                    id="writing",
                    agent_id="writer_1",
                    role=AgentRole.WRITER,
                    description=f"撰写关于{topic}的文章",
                    input_data={"topic": topic, "style": style},
                    dependencies=["research"],
                    priority=8
                )
            ]
        )

# DSPy签名定义
class AnalyzeTopic(dspy.Signature):
    """主题分析"""
    topic = dspy.InputField(desc="研究主题")
    analysis = dspy.OutputField(desc="主题分析结果")
    keywords = dspy.OutputField(desc="关键词列表", type=List[str])

class CreateOutline(dspy.Signature):
    """创建大纲"""
    topic = dspy.InputField(desc="写作主题")
    research = dspy.InputField(desc="研究资料")
    outline = dspy.OutputField(desc="文章大纲")

class GenerateContent(dspy.Signature):
    """生成内容"""
    topic = dspy.InputField(desc="写作主题")
    outline = dspy.InputField(desc="文章大纲")
    research = dspy.InputField(desc="研究资料")
    style = dspy.InputField(desc="写作风格")
    content = dspy.OutputField(desc="生成的内容")

class EditContent(dspy.Signature):
    """编辑内容"""
    original_content = dspy.InputField(desc="原始内容")
    style = dspy.InputField(desc="目标风格")
    target_audience = dspy.InputField(desc="目标受众")
    edited_content = dspy.OutputField(desc="编辑后的内容")
```

---

## 🛠️ 工具类库

### 4. 性能监控工具

#### src/utils/monitoring.py
```python
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import json
from datetime import datetime, timedelta

import psutil
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

        # 系统监控
        self.system_monitor = SystemMonitor()
        self.system_monitor.start()

    def record_execution(self, name: str, execution_time: float,
                        success: bool = True, error_message: str = None,
                        metadata: Dict[str, Any] = None):
        """记录执行指标"""
        with self.lock:
            timestamp = time.time()
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                execution_time=execution_time,
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                cpu_usage=psutil.cpu_percent(),
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )

            self.metrics[name].append(metrics)
            self.call_counts[name] += 1

            if not success:
                self.error_counts[name] += 1

    def get_statistics(self, name: str, time_window: Optional[float] = None) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            if name not in self.metrics:
                return {}

            metrics_list = list(self.metrics[name])

            # 时间窗口过滤
            if time_window:
                cutoff_time = time.time() - time_window
                metrics_list = [m for m in metrics_list if m.timestamp >= cutoff_time]

            if not metrics_list:
                return {}

            execution_times = [m.execution_time for m in metrics_list]
            memory_usages = [m.memory_usage for m in metrics_list]
            success_count = sum(1 for m in metrics_list if m.success)

            stats = {
                'call_count': len(metrics_list),
                'success_count': success_count,
                'error_count': len(metrics_list) - success_count,
                'success_rate': success_count / len(metrics_list),
                'avg_execution_time': np.mean(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
                'p95_execution_time': np.percentile(execution_times, 95),
                'p99_execution_time': np.percentile(execution_times, 99),
                'avg_memory_usage': np.mean(memory_usages),
                'max_memory_usage': np.max(memory_usages),
                'last_execution': metrics_list[-1].timestamp
            }

            return stats

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有统计信息"""
        all_stats = {}

        for name in self.metrics.keys():
            all_stats[name] = self.get_statistics(name)

        return all_stats

    def reset_metrics(self, name: Optional[str] = None):
        """重置指标"""
        with self.lock:
            if name:
                if name in self.metrics:
                    del self.metrics[name]
                self.call_counts[name] = 0
                self.error_counts[name] = 0
            else:
                self.metrics.clear()
                self.call_counts.clear()
                self.error_counts.clear()

    def export_metrics(self, filename: str):
        """导出指标"""
        with self.lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.get_all_statistics(),
                'system_metrics': self.system_monitor.get_current_metrics()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

class SystemMonitor:
    """系统监控器"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics = deque(maxlen=3600)  # 保存1小时的数据

    def start(self):
        """启动监控"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'process_count': len(psutil.pids()),
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                }
                self.metrics.append(metrics)
            except Exception as e:
                logger.error(f"系统监控错误: {e}")

            time.sleep(self.interval)

    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前系统指标"""
        return self.metrics[-1] if self.metrics else {}

def monitor_performance(name: Optional[str] = None, monitor: Optional[PerformanceMonitor] = None):
    """性能监控装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor_instance = monitor or get_default_monitor()
            func_name = name or f"{func.__module__}.{func.__name__}"

            start_time = time.time()
            success = True
            error_message = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                monitor_instance.record_execution(
                    func_name, execution_time, success, error_message
                )

        return wrapper
    return decorator

# 全局监控器实例
_default_monitor = None

def get_default_monitor() -> PerformanceMonitor:
    """获取默认监控器"""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = PerformanceMonitor()
    return _default_monitor

def set_default_monitor(monitor: PerformanceMonitor):
    """设置默认监控器"""
    global _default_monitor
    _default_monitor = monitor
```

### 5. 缓存管理工具

#### src/utils/caching.py
```python
import hashlib
import json
import pickle
import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import redis
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0

class CacheBackend(ABC):
    """缓存后端接口"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass

class MemoryCache(CacheBackend):
    """内存缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]

            # 检查TTL
            if self._is_expired(entry):
                del self.cache[key]
                return None

            # 更新访问信息
            entry.access_count += 1
            entry.last_access = time.time()

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        with self.lock:
            # 检查容量
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                last_access=time.time()
            )

            self.cache[key] = entry
            return True

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> bool:
        with self.lock:
            self.cache.clear()
            return True

    def exists(self, key: str) -> bool:
        with self.lock:
            if key not in self.cache:
                return False

            if self._is_expired(self.cache[key]):
                del self.cache[key]
                return False

            return True

    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存是否过期"""
        if entry.ttl is None:
            return False
        return time.time() - entry.timestamp > entry.ttl

    def _evict_lru(self):
        """删除最近最少使用的条目"""
        if not self.cache:
            return

        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_access
        )
        del self.cache[lru_key]

class FileCache(CacheBackend):
    """文件缓存"""

    def __init__(self, cache_dir: str = "./cache", default_ttl: Optional[float] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

    def _get_file_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用MD5避免文件名过长或包含特殊字符
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)

            # 检查TTL
            if self._is_expired(entry):
                file_path.unlink()
                return None

            return entry.value

        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        file_path = self._get_file_path(key)

        try:
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                last_access=time.time()
            )

            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)

            return True

        except Exception as e:
            logger.error(f"写入缓存文件失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        file_path = self._get_file_path(key)

        try:
            if file_path.exists():
                file_path.unlink()
            return True

        except Exception as e:
            logger.error(f"删除缓存文件失败: {e}")
            return False

    def clear(self) -> bool:
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            return True

        except Exception as e:
            logger.error(f"清空缓存目录失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return False

        try:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)

            if self._is_expired(entry):
                file_path.unlink()
                return False

            return True

        except Exception:
            return False

    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存是否过期"""
        if entry.ttl is None:
            return False
        return time.time() - entry.timestamp > entry.ttl

class RedisCache(CacheBackend):
    """Redis缓存"""

    def __init__(self, redis_client, key_prefix: str = "dspy:", default_ttl: Optional[float] = None):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.redis.get(self._make_key(key))
            if data is None:
                return None

            return pickle.loads(data)

        except Exception as e:
            logger.error(f"Redis读取失败: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        try:
            data = pickle.dumps(value)
            redis_key = self._make_key(key)
            expire_time = ttl or self.default_ttl

            if expire_time:
                return self.redis.setex(redis_key, int(expire_time), data)
            else:
                return self.redis.set(redis_key, data)

        except Exception as e:
            logger.error(f"Redis写入失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self.redis.delete(self._make_key(key)))

        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False

    def clear(self) -> bool:
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis.keys(pattern)
            if keys:
                return bool(self.redis.delete(*keys))
            return True

        except Exception as e:
            logger.error(f"Redis清空失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            return bool(self.redis.exists(self._make_key(key)))

        except Exception as e:
            logger.error(f"Redis检查存在失败: {e}")
            return False

class CacheManager:
    """缓存管理器"""

    def __init__(self, backend: CacheBackend, key_prefix: str = "dspy"):
        self.backend = backend
        self.key_prefix = key_prefix
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        self.lock = threading.Lock()

    def _make_key(self, key_parts: List[Any]) -> str:
        """生成缓存键"""
        key_data = {
            'prefix': self.key_prefix,
            'parts': key_parts
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, *key_parts) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(list(key_parts))

        try:
            result = self.backend.get(key)
            if result is not None:
                with self.lock:
                    self.stats['hits'] += 1
                return result
            else:
                with self.lock:
                    self.stats['misses'] += 1
                return None
        except Exception as e:
            logger.error(f"缓存获取失败: {e}")
            return None

    def set(self, value: Any, ttl: Optional[float] = None, *key_parts) -> bool:
        """设置缓存"""
        key = self._make_key(list(key_parts))

        try:
            result = self.backend.set(key, value, ttl)
            if result:
                with self.lock:
                    self.stats['sets'] += 1
            return result
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")
            return False

    def delete(self, *key_parts) -> bool:
        """删除缓存"""
        key = self._make_key(list(key_parts))

        try:
            result = self.backend.delete(key)
            if result:
                with self.lock:
                    self.stats['deletes'] += 1
            return result
        except Exception as e:
            logger.error(f"缓存删除失败: {e}")
            return False

    def get_or_set(self, factory: Callable, ttl: Optional[float] = None, *key_parts) -> Any:
        """获取或设置缓存"""
        result = self.get(*key_parts)

        if result is None:
            result = factory()
            self.set(result, ttl, *key_parts)

        return result

    def clear(self) -> bool:
        """清空缓存"""
        return self.backend.clear()

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self.lock:
            return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        with self.lock:
            for key in self.stats:
                self.stats[key] = 0

def cached(ttl: Optional[float] = None, cache_manager: Optional[CacheManager] = None,
          key_func: Optional[Callable] = None):
    """缓存装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = cache_manager or get_default_cache_manager()

            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = [func.__module__, func.__name__] + list(args) + list(sorted(kwargs.items()))

            # 尝试获取缓存
            result = manager.get(*cache_key)
            if result is not None:
                return result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            manager.set(result, ttl, *cache_key)

            return result

        return wrapper
    return decorator

# 全局缓存管理器
_default_cache_manager = None

def get_default_cache_manager() -> CacheManager:
    """获取默认缓存管理器"""
    global _default_cache_manager
    if _default_cache_manager is None:
        backend = MemoryCache(max_size=1000, default_ttl=3600)
        _default_cache_manager = CacheManager(backend)
    return _default_cache_manager

def set_default_cache_manager(manager: CacheManager):
    """设置默认缓存管理器"""
    global _default_cache_manager
    _default_cache_manager = manager
```

---

## 📊 最佳实践指南

### 6. 开发规范和性能优化

#### 开发规范
```python
# src/utils/standards.py

"""
DSPy项目开发规范
"""

# 1. 代码风格规范
"""
- 使用Black进行代码格式化
- 使用Ruff进行代码检查
- 使用MyPy进行类型检查
- 遵循PEP 8编码规范
"""

# 2. 命名规范
"""
- 类名使用PascalCase：EnterpriseRAG, QueryAnalyzer
- 函数和变量使用snake_case：analyze_query, execution_time
- 常量使用UPPER_CASE：MAX_TOKENS, DEFAULT_MODEL
- 私有成员使用下划线前缀：_private_method
"""

# 3. 文档规范
"""
- 所有公共函数必须有docstring
- 使用Google风格的docstring
- 包含参数说明、返回值说明和异常说明
"""

def example_function(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """示例函数的文档字符串

    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述，可选

    Returns:
        包含结果的字典

    Raises:
        ValueError: 当参数无效时
        RuntimeError: 当运行时错误时
    """
    pass

# 4. 类型注解规范
"""
- 所有函数参数和返回值都要有类型注解
- 使用typing模块中的类型
- 复杂类型使用Union或Optional
"""

# 5. 错误处理规范
"""
- 使用具体的异常类型
- 包含有意义的错误消息
- 记录详细的错误日志
- 提供错误恢复机制
"""

# 6. 测试规范
"""
- 每个模块都要有对应的测试文件
- 测试覆盖率不低于80%
- 使用pytest框架
- 包含单元测试和集成测试
"""
```

#### 性能优化指南
```python
# src/utils/optimization.py

"""
DSPy性能优化指南
"""

import asyncio
import functools
from typing import Any, Callable, List
import time

# 1. 异步优化
class AsyncOptimizer:
    """异步执行优化器"""

    @staticmethod
    async def batch_execute(func: Callable, items: List[Any], batch_size: int = 10):
        """批量异步执行"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            tasks = [func(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

# 2. 内存优化
class MemoryOptimizer:
    """内存使用优化器"""

    @staticmethod
    def generator_to_list(generator, max_items: int = None):
        """生成器转列表，支持限制数量"""
        result = []
        for i, item in enumerate(generator):
            if max_items and i >= max_items:
                break
            result.append(item)
        return result

    @staticmethod
    def clear_cache(obj):
        """清理对象缓存"""
        if hasattr(obj, '__dict__'):
            obj.__dict__.clear()
        if hasattr(obj, 'cache'):
            obj.cache.clear()

# 3. 并发优化
class ConcurrencyOptimizer:
    """并发执行优化器"""

    @staticmethod
    def parallel_map(func: Callable, items: List[Any], max_workers: int = 4):
        """并行映射"""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, items))

        return results

# 4. 缓存优化
class CacheOptimizer:
    """缓存优化器"""

    @staticmethod
    def smart_cache(ttl: float = 3600, max_size: int = 1000):
        """智能缓存装饰器"""
        def decorator(func):
            cache = {}
            access_times = {}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                key = str(args) + str(sorted(kwargs.items()))
                current_time = time.time()

                # 检查缓存
                if key in cache:
                    result, timestamp = cache[key]
                    if current_time - timestamp < ttl:
                        access_times[key] = current_time
                        return result
                    else:
                        del cache[key]
                        del access_times[key]

                # 执行函数
                result = func(*args, **kwargs)

                # 缓存管理
                if len(cache) >= max_size:
                    # 删除最久未访问的条目
                    lru_key = min(access_times, key=access_times.get)
                    del cache[lru_key]
                    del access_times[lru_key]

                cache[key] = (result, current_time)
                access_times[key] = current_time

                return result

            return wrapper
        return decorator

# 5. 预加载优化
class PreloadOptimizer:
    """预加载优化器"""

    def __init__(self):
        self.preloaded_data = {}

    def preload(self, key: str, loader: Callable):
        """预加载数据"""
        if key not in self.preloaded_data:
            self.preloaded_data[key] = loader()

    def get(self, key: str):
        """获取预加载数据"""
        return self.preloaded_data.get(key)

# 使用示例
@CacheOptimizer.smart_cache(ttl=1800, max_size=100)
def expensive_computation(x: int, y: int) -> int:
    """耗时的计算函数"""
    time.sleep(0.1)  # 模拟耗时操作
    return x * y

@functools.lru_cache(maxsize=128)
def memoized_fibonacci(n: int) -> int:
    """记忆化斐波那契数列"""
    if n < 2:
        return n
    return memoized_fibonacci(n-1) + memoized_fibonacci(n-2)
```

### 7. 部署和运维最佳实践

#### Docker部署模板
```dockerfile
# Dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/
COPY config/ ./config/

# 设置环境变量
ENV PYTHONPATH=/app
ENV DSPY_CONFIG_PATH=/app/config

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "src.main"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  dspy-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/dspy
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=dspy
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - dspy-app
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

#### 监控配置
```yaml
# monitoring/docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

---

这套代码示例和最佳实践模板为DSPy开发提供了完整的框架，包括：

1. **基础模板**：项目结构、配置管理、基类设计
2. **进阶模式**：企业级RAG、多智能体协作
3. **工具类库**：性能监控、缓存管理
4. **最佳实践**：开发规范、性能优化、部署运维

每个模板都经过精心设计，可以直接用于生产环境，或作为项目的起点进行定制化开发。