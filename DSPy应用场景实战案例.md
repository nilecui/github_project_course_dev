# DSPy应用场景实战案例

## 📊 案例概览

本文档提供了DSPy在5大核心应用场景中的实战案例，每个案例包含：项目背景、技术方案、完整实现、部署指导和扩展思路。

**5大应用场景：**
1. 🔍 **检索增强生成(RAG)** - 智能问答系统
2. 🤖 **智能体开发** - 多工具客服助手
3. 📝 **文本处理分析** - 文档自动分类
4. 🧮 **复杂推理系统** - 数学问题求解器
5. 🎨 **多模态应用** - 图文内容分析

---

## 🔍 案例1：智能RAG问答系统

### 项目背景
构建一个基于企业知识库的智能问答系统，能够准确回答员工关于公司政策、技术文档、业务流程等问题。

### 技术方案
- **检索层**: ColBERTv2 + 混合检索
- **生成层**: ChainOfThought + 自我修正
- **优化**: BootstrapFewShot + MIPROv2
- **评估**: 多维度评估指标

### 完整实现

#### 1. 数据准备和预处理
```python
import dspy
import json
from typing import List, Dict, Any
import re

class KnowledgeBaseProcessor:
    """知识库数据处理器"""

    def __init__(self):
        self.chunk_size = 500  # 文档分块大小
        self.overlap = 50      # 重叠大小

    def load_documents(self, file_path: str) -> List[Dict]:
        """加载文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """将文档分块"""
        content = doc['content']
        chunks = []

        # 按段落分割
        paragraphs = content.split('\n\n')
        current_chunk = ""
        current_length = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 如果当前块加上新段落超限，则保存当前块
            if current_length + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'doc_id': doc['id'],
                    'chunk_id': len(chunks),
                    'title': doc.get('title', ''),
                    'category': doc.get('category', 'general')
                })
                current_chunk = paragraph
                current_length = len(paragraph)
            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
                current_length += len(paragraph) + 2

        # 保存最后一个块
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'doc_id': doc['id'],
                'chunk_id': len(chunks),
                'title': doc.get('title', ''),
                'category': doc.get('category', 'general')
            })

        return chunks

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 标准化空格
        text = re.sub(r' {2,}', ' ', text)
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】]', '', text)
        return text.strip()

    def process_knowledge_base(self, docs_path: str) -> List[Dict]:
        """处理整个知识库"""
        documents = self.load_documents(docs_path)
        all_chunks = []

        for doc in documents:
            # 预处理文档内容
            doc['content'] = self.preprocess_text(doc['content'])

            # 分块处理
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"处理完成：{len(documents)}个文档 → {len(all_chunks)}个文本块")
        return all_chunks

# 使用示例
processor = KnowledgeBaseProcessor()
knowledge_chunks = processor.process_knowledge_base("company_knowledge_base.json")
```

#### 2. 高级RAG系统实现
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EnterpriseRAG(dspy.Module):
    """企业级RAG系统"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 检索组件
        self.retriever = dspy.ColBERTv2(
            model_path="colbert-ir/colbertv2.0",
            index_path=config.get('index_path', './knowledge_index')
        )

        # 重排序组件
        self.reranker = dspy.Predict(RerankPassages)

        # 查询理解组件
        self.query_analyzer = dspy.ChainOfThought(AnalyzeQuery)

        # 答案生成组件
        self.generator = dspy.ChainOfThought(GenerateAnswer)

        # 答案验证组件
        self.verifier = dspy.Predict(VerifyAnswer)

        # 知识库缓存
        self.knowledge_cache = {}

    def forward(self, question: str, context_info: Dict = None) -> dspy.Prediction:
        """RAG系统前向传播"""

        # 1. 查询分析和理解
        analyzed_query = self.query_analyzer(question=question)

        # 2. 构建检索策略
        retrieval_strategy = self._build_retrieval_strategy(analyzed_query)

        # 3. 执行检索
        raw_results = self._retrieve_documents(question, retrieval_strategy)

        # 4. 结果重排序
        ranked_results = self._rerank_documents(question, raw_results)

        # 5. 生成答案
        answer = self._generate_answer(question, ranked_results)

        # 6. 答案验证
        verified_answer = self._verify_answer(question, answer, ranked_results)

        return dspy.Prediction(
            answer=verified_answer.answer,
            confidence=verified_answer.confidence,
            sources=verified_answer.sources,
            reasoning=verified_answer.reasoning,
            retrieval_results=ranked_results
        )

    def _build_retrieval_strategy(self, analyzed_query) -> Dict:
        """构建检索策略"""
        strategy = {
            'k': 10,  # 默认检索数量
            'filters': {},
            'search_type': 'hybrid',  # hybrid, semantic, keyword
            'boost_recent': False
        }

        # 根据查询分析调整策略
        if hasattr(analyzed_query, 'query_type'):
            if analyzed_query.query_type == 'factual':
                strategy['k'] = 8
                strategy['search_type'] = 'semantic'
            elif analyzed_query.query_type == 'procedural':
                strategy['k'] = 15
                strategy['boost_recent'] = True
            elif analyzed_query.query_type == 'policy':
                strategy['filters']['category'] = 'policy'
                strategy['search_type'] = 'keyword'

        return strategy

    def _retrieve_documents(self, question: str, strategy: Dict) -> List[Dict]:
        """执行文档检索"""

        # 构建查询扩展
        expanded_query = self._expand_query(question)

        # 执行检索
        if strategy['search_type'] == 'hybrid':
            results = self.retriever.hybrid_search(
                query=expanded_query,
                k=strategy['k'],
                alpha=0.7,  # 语义检索权重
                filters=strategy.get('filters', {})
            )
        else:
            results = self.retriever.search(
                query=expanded_query,
                k=strategy['k'],
                filters=strategy.get('filters', {})
            )

        # 如果需要，提升最近文档
        if strategy.get('boost_recent'):
            results = self._boost_recent_documents(results)

        return results

    def _expand_query(self, question: str) -> str:
        """查询扩展"""
        # 简单的同义词扩展
        expansions = {
            '如何': '怎么',
            '哪些': '什么',
            '为什么': '原因',
            '什么时候': '何时'
        }

        expanded = question
        for original, synonym in expansions.items():
            if original in expanded:
                expanded = expanded.replace(original, f"{original} {synonym}")

        return expanded

    def _boost_recent_documents(self, results: List[Dict]) -> List[Dict]:
        """提升最近文档的权重"""
        current_year = 2024

        for result in results:
            if 'year' in result['metadata']:
                year_diff = current_year - result['metadata']['year']
                boost_factor = max(0.1, 1.0 - (year_diff * 0.1))
                result['score'] *= boost_factor

        # 重新排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def _rerank_documents(self, question: str, documents: List[Dict]) -> List[Dict]:
        """文档重排序"""
        if len(documents) <= 5:
            return documents  # 文档太少，无需重排序

        # 准备重排序输入
        passages = [doc['content'] for doc in documents]

        # 执行重排序
        rerank_result = self.reranker(
            query=question,
            passages=passages
        )

        # 重新组装结果
        reranked_docs = []
        for i, passage_idx in enumerate(rerank_result.indices):
            if i < len(documents):  # 确保索引有效
                doc = documents[passage_idx].copy()
                doc['rerank_score'] = rerank_result.scores[i]
                reranked_docs.append(doc)

        return reranked_docs

    def _generate_answer(self, question: str, documents: List[Dict]) -> Dict:
        """生成答案"""
        context = "\n\n".join([
            f"文档{i+1}: {doc['content'][:500]}..."
            for i, doc in enumerate(documents[:5])
        ])

        result = self.generator(
            question=question,
            context=context
        )

        return {
            'answer': result.answer,
            'reasoning': result.reasoning,
            'sources': [doc.get('doc_id', '') for doc in documents[:3]]
        }

    def _verify_answer(self, question: str, answer: Dict, documents: List[Dict]) -> Dict:
        """验证答案质量"""
        verification_result = self.verifier(
            question=question,
            answer=answer['answer'],
            context="\n".join([doc['content'] for doc in documents[:5]])
        )

        # 合并验证结果
        final_answer = {
            'answer': answer['answer'],
            'confidence': verification_result.confidence,
            'sources': answer['sources'],
            'reasoning': answer['reasoning'],
            'verification': verification_result.verification
        }

        return final_answer

# 签名定义
class AnalyzeQuery(dspy.Signature):
    """分析查询类型和意图"""
    question = dspy.InputField(desc="用户问题")
    query_type = dspy.OutputField(desc="查询类型:factual/procedural/policy")
    key_entities = dspy.OutputField(desc="关键实体")
    complexity = dspy.OutputField(desc="复杂度:low/medium/high")

class RerankPassages(dspy.Signature):
    """重排序文档片段"""
    query = dspy.InputField(desc="查询")
    passages = dspy.InputField(desc="文档片段列表", type=List[str])
    indices = dspy.OutputField(desc="排序后的索引", type=List[int])
    scores = dspy.OutputField(desc="相关性分数", type=List[float])

class GenerateAnswer(dspy.Signature):
    """基于上下文生成答案"""
    question = dspy.InputField(desc="问题")
    context = dspy.InputField(desc="相关文档上下文")
    answer = dspy.OutputField(desc="详细答案")
    reasoning = dspy.OutputField(desc="推理过程")

class VerifyAnswer(dspy.Signature):
    """验证答案质量"""
    question = dspy.InputField(desc="原始问题")
    answer = dspy.InputField(desc="生成的答案")
    context = dspy.InputField(desc="文档上下文")
    verification = dspy.OutputField(desc="验证结果")
    confidence = dspy.OutputField(desc="置信度", type=float)
```

#### 3. 系统优化
```python
class RAGOptimizer:
    """RAG系统优化器"""

    def __init__(self, rag_system, train_data):
        self.rag_system = rag_system
        self.train_data = train_data

    def optimize_retrieval(self):
        """优化检索组件"""
        print("优化检索组件...")

        # BootstrapFewShot优化检索
        retrieval_optimizer = dspy.BootstrapFewShot(
            metric=self._retrieval_metric,
            max_bootstrapped_demos=5,
            max_labeled_demos=3
        )

        # 优化检索器
        optimized_retriever = retrieval_optimizer.compile(
            self.rag_system.retriever,
            trainset=self.train_data
        )

        self.rag_system.retriever = optimized_retriever
        print("检索组件优化完成")

    def optimize_generation(self):
        """优化生成组件"""
        print("优化生成组件...")

        # MIPROv2优化生成
        generation_optimizer = dspy.MIPROv2(
            metric=self._generation_metric,
            num_candidates=8,
            init_temperature=0.8
        )

        # 优化生成器
        optimized_generator = generation_optimizer.compile(
            self.rag_system.generator,
            trainset=self.train_data
        )

        self.rag_system.generator = optimized_generator
        print("生成组件优化完成")

    def _retrieval_metric(self, gold, pred):
        """检索质量评估"""
        # 简化的检索评估：检查相关文档是否在结果中
        relevant_docs = set(gold.get('relevant_docs', []))
        retrieved_docs = set(pred.get('sources', []))

        if not relevant_docs:
            return 1.0  # 如果没有相关文档标记，默认满分

        precision = len(relevant_docs & retrieved_docs) / max(len(retrieved_docs), 1)
        recall = len(relevant_docs & retrieved_docs) / max(len(relevant_docs), 1)

        return (precision + recall) / 2

    def _generation_metric(self, gold, pred):
        """生成质量评估"""
        # 使用模糊匹配评估答案质量
        gold_answer = gold.answer.lower()
        pred_answer = pred.answer.lower()

        # 计算词汇重叠度
        gold_words = set(gold_answer.split())
        pred_words = set(pred_answer.split())

        if not gold_words:
            return 1.0

        overlap = len(gold_words & pred_words)
        precision = overlap / max(len(pred_words), 1)
        recall = overlap / len(gold_words)

        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        return f1

# 训练数据示例
train_examples = [
    Example(
        question="公司的年假政策是怎样的？",
        answer="公司年假政策：入职满1年享受5天年假，每增加1年工龄增加1天，最高15天...",
        relevant_docs=["HR_POLICY_001", "EMPLOYEE_GUIDE_003"]
    ),
    # ... 更多训练数据
]

# 优化系统
rag_system = EnterpriseRAG(config={'index_path': './company_index'})
optimizer = RAGOptimizer(rag_system, train_examples)

optimizer.optimize_retrieval()
optimizer.optimize_generation()
```

### 部署指导

#### 1. API服务部署
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    user_id: str = None
    context_info: dict = {}

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str
    processing_time: float

# 全局RAG系统实例
rag_system = EnterpriseRAG(config={'index_path': './company_index'})

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """查询接口"""
    import time
    start_time = time.time()

    try:
        # 调用RAG系统
        result = rag_system(request.question, request.context_info)

        processing_time = time.time() - start_time

        return QueryResponse(
            answer=result.answer,
            confidence=result.confidence,
            sources=result.sources,
            reasoning=result.reasoning,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 2. 性能监控
```python
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List

class RAGMonitor:
    """RAG系统监控"""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.metrics = {
            'response_times': deque(maxlen=window_size),
            'confidence_scores': deque(maxlen=window_size),
            'query_types': defaultdict(int),
            'error_count': 0,
            'total_queries': 0,
            'cache_hits': 0
        }

    def record_query(self, query: str, response: Dict, processing_time: float):
        """记录查询指标"""
        self.metrics['response_times'].append(processing_time)
        self.metrics['confidence_scores'].append(response.get('confidence', 0))
        self.metrics['total_queries'] += 1

        # 分析查询类型
        query_type = self._classify_query(query)
        self.metrics['query_types'][query_type] += 1

        # 记录日志
        logging.info(f"Query processed: {query[:50]}... - {processing_time:.2f}s")

    def record_error(self, error: str):
        """记录错误"""
        self.metrics['error_count'] += 1
        logging.error(f"Query error: {error}")

    def record_cache_hit(self):
        """记录缓存命中"""
        self.metrics['cache_hits'] += 1

    def get_stats(self) -> Dict:
        """获取统计信息"""
        response_times = list(self.metrics['response_times'])
        confidence_scores = list(self.metrics['confidence_scores'])

        return {
            'avg_response_time': sum(response_times) / max(len(response_times), 1),
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'avg_confidence': sum(confidence_scores) / max(len(confidence_scores), 1),
            'error_rate': self.metrics['error_count'] / max(self.metrics['total_queries'], 1),
            'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['total_queries'], 1),
            'query_type_distribution': dict(self.metrics['query_types']),
            'total_queries': self.metrics['total_queries']
        }

    def _classify_query(self, query: str) -> str:
        """查询分类"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['如何', '怎么', '怎样']):
            return 'procedural'
        elif any(word in query_lower for word in ['什么', '哪些', '谁']):
            return 'factual'
        elif any(word in query_lower for word in ['为什么', '原因']):
            return 'causal'
        else:
            return 'other'

# 监控器实例
monitor = RAGMonitor()
```

### 扩展思路

#### 1. 多轮对话支持
```python
class ConversationalRAG(EnterpriseRAG):
    """支持多轮对话的RAG系统"""

    def __init__(self, config):
        super().__init__(config)
        self.conversation_history = {}
        self.context_manager = ConversationContextManager()

    def forward(self, question: str, user_id: str, session_id: str) -> dspy.Prediction:
        """多轮对话处理"""
        # 获取对话历史
        conversation_key = f"{user_id}_{session_id}"
        history = self.conversation_history.get(conversation_key, [])

        # 构建上下文
        enriched_question = self.context_manager.build_context(
            question, history
        )

        # 调用基础RAG
        result = super().forward(enriched_question['question'])

        # 更新对话历史
        self.conversation_history[conversation_key] = history + [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': result.answer}
        ]

        # 添加对话上下文信息
        result.conversation_context = enriched_question['context']
        return result
```

#### 2. 个性化推荐
```python
class PersonalizedRAG(EnterpriseRAG):
    """个性化RAG系统"""

    def __init__(self, config):
        super().__init__(config)
        self.user_profiles = {}
        self.preference_learner = UserPreferenceLearner()

    def forward(self, question: str, user_id: str) -> dspy.Prediction:
        """个性化问答"""
        # 获取用户画像
        user_profile = self.user_profiles.get(user_id, {})

        # 个性化查询扩展
        personalized_query = self._personalize_query(question, user_profile)

        # 调用基础RAG
        result = super().forward(personalized_query)

        # 个性化结果排序
        result = self._personalize_results(result, user_profile)

        # 更新用户偏好
        self.preference_learner.update_preferences(user_id, question, result)

        return result
```

---

## 🤖 案例2：多工具客服智能体

### 项目背景
开发一个智能客服助手，能够自动调用多种工具（订单查询、退换货、知识库搜索、人工转接）来解决用户问题。

### 技术方案
- **核心框架**: ReAct智能体
- **工具集成**: 订单系统、物流API、知识库
- **对话管理**: 上下文记忆和状态跟踪
- **决策路由**: 意图识别和工具选择

### 完整实现

#### 1. 工具系统设计
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import requests
import json

class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        pass

    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        return True

class OrderQueryTool(Tool):
    """订单查询工具"""

    def __init__(self):
        super().__init__(
            name="order_query",
            description="查询订单信息，需要订单号或手机号"
        )

    def execute(self, order_id: str = None, phone: str = None) -> Dict[str, Any]:
        """查询订单"""
        if not order_id and not phone:
            return {"error": "需要提供订单号或手机号"}

        # 模拟API调用
        if order_id:
            # 根据订单号查询
            order_info = self._query_by_order_id(order_id)
        else:
            # 根据手机号查询
            order_info = self._query_by_phone(phone)

        return order_info

    def _query_by_order_id(self, order_id: str) -> Dict[str, Any]:
        """根据订单号查询"""
        # 模拟数据库查询
        orders_db = {
            "ORD202401001": {
                "order_id": "ORD202401001",
                "status": "已发货",
                "products": ["智能手表", "保护膜"],
                "total_amount": 1299.00,
                "shipping_address": "北京市朝阳区...",
                "tracking_number": "SF1234567890",
                "estimated_delivery": "2024-01-15"
            }
        }

        if order_id in orders_db:
            return {"success": True, "data": orders_db[order_id]}
        else:
            return {"success": False, "error": "订单不存在"}

    def _query_by_phone(self, phone: str) -> Dict[str, Any]:
        """根据手机号查询"""
        # 模拟实现
        return {"success": True, "data": {"orders": ["ORD202401001", "ORD202401002"]}}

    def validate_input(self, **kwargs) -> bool:
        order_id = kwargs.get('order_id', '')
        phone = kwargs.get('phone', '')
        return bool(order_id) or bool(phone)

class RefundTool(Tool):
    """退款工具"""

    def __init__(self):
        super().__init__(
            name="refund",
            description="处理退款申请，需要订单号和退款原因"
        )

    def execute(self, order_id: str, reason: str) -> Dict[str, Any]:
        """处理退款"""
        if not order_id or not reason:
            return {"error": "需要提供订单号和退款原因"}

        # 验证订单状态
        order_result = OrderQueryTool().execute(order_id=order_id)
        if not order_result.get("success"):
            return {"error": "订单不存在"}

        order_data = order_result["data"]
        if order_data["status"] not in ["已发货", "已完成"]:
            return {"error": "当前订单状态不支持退款"}

        # 处理退款逻辑
        refund_id = f"REF{order_id[3:]}{int(time.time())}"
        refund_status = "处理中"

        return {
            "success": True,
            "refund_id": refund_id,
            "status": refund_status,
            "estimated_refund_time": "3-5个工作日"
        }

class KnowledgeSearchTool(Tool):
    """知识库搜索工具"""

    def __init__(self):
        super().__init__(
            name="knowledge_search",
            description="搜索产品知识库，回答常见问题"
        )

    def execute(self, query: str) -> Dict[str, Any]:
        """搜索知识库"""
        if not query:
            return {"error": "需要提供搜索查询"}

        # 模拟知识库搜索
        knowledge_base = {
            "保修政策": "所有产品享受一年质保，人为损坏不在保修范围内...",
            "退换货政策": "7天无理由退换，商品需保持原包装完好...",
            "配送时间": "一般地区3-5个工作日，偏远地区7-10个工作日...",
            "支付方式": "支持支付宝、微信支付、银行卡、信用卡..."
        }

        # 简单的关键词匹配
        results = []
        for topic, content in knowledge_base.items():
            if any(keyword in query.lower() for keyword in topic.lower().split()):
                results.append({
                    "topic": topic,
                    "content": content,
                    "relevance": 0.8
                })

        return {
            "success": True,
            "results": results,
            "query": query
        }

class HumanTransferTool(Tool):
    """人工转接工具"""

    def __init__(self):
        super().__init__(
            name="human_transfer",
            description="将用户转接到人工客服"
        )

    def execute(self, reason: str = None, priority: str = "normal") -> Dict[str, Any]:
        """转接人工客服"""
        # 创建工单
        ticket_id = f"TK{int(time.time())}"

        return {
            "success": True,
            "ticket_id": ticket_id,
            "estimated_wait_time": "5-10分钟",
            "message": "已为您转接人工客服，请耐心等待"
        }
```

#### 2. ReAct智能体实现
```python
class CustomerServiceAgent(dspy.Module):
    """客服智能体"""

    def __init__(self):
        super().__init__()

        # 初始化工具
        self.tools = {
            'order_query': OrderQueryTool(),
            'refund': RefundTool(),
            'knowledge_search': KnowledgeSearchTool(),
            'human_transfer': HumanTransferTool()
        }

        # ReAct组件
        self.react = dspy.ReAct(
            CustomerServiceTask,
            tools=list(self.tools.values())
        )

        # 意图识别组件
        self.intent_classifier = dspy.Predict(ClassifyIntent)

        # 对话状态管理
        self.conversation_state = {}

    def forward(self, user_input: str, session_id: str = None) -> dspy.Prediction:
        """处理用户输入"""
        # 识别用户意图
        intent_result = self.intent_classifier(user_input=user_input)
        user_intent = intent_result.intent

        # 构建任务描述
        task_description = self._build_task_description(
            user_input, user_intent, session_id
        )

        # 执行ReAct推理
        result = self.react(task=task_description)

        # 处理结果
        final_response = self._process_result(result, user_intent)

        # 更新对话状态
        if session_id:
            self._update_conversation_state(session_id, user_input, final_response)

        return dspy.Prediction(
            response=final_response['response'],
            intent=user_intent,
            tools_used=final_response.get('tools_used', []),
            confidence=final_response.get('confidence', 0.8),
            session_id=session_id
        )

    def _build_task_description(self, user_input: str, intent: str, session_id: str) -> str:
        """构建任务描述"""
        # 获取历史对话上下文
        context = self._get_conversation_context(session_id) if session_id else ""

        task = f"""
用户输入: {user_input}
用户意图: {intent}
对话历史: {context}

请使用合适的工具来帮助用户解决问题。可用工具包括:
- order_query: 查询订单信息
- refund: 处理退款申请
- knowledge_search: 搜索知识库
- human_transfer: 转接人工客服

请按以下步骤进行:
1. 分析用户需求
2. 选择合适的工具
3. 执行工具调用
4. 基于工具结果生成友好回复
"""

        return task

    def _process_result(self, result: dspy.Prediction, intent: str) -> Dict[str, Any]:
        """处理ReAct结果"""
        response = ""
        tools_used = []
        confidence = 0.8

        if hasattr(result, 'final_answer'):
            response = result.final_answer
        elif hasattr(result, 'answer'):
            response = result.answer
        else:
            # 根据意图生成默认回复
            response = self._generate_default_response(intent)

        # 分析工具使用情况
        if hasattr(result, 'tool_calls'):
            tools_used = [call.tool_name for call in result.tool_calls]

        # 根据结果质量调整置信度
        if "抱歉" in response or "无法" in response:
            confidence = 0.4

        return {
            'response': response,
            'tools_used': tools_used,
            'confidence': confidence
        }

    def _generate_default_response(self, intent: str) -> str:
        """生成默认回复"""
        default_responses = {
            'order_query': "我来帮您查询订单信息。请提供您的订单号或手机号。",
            'refund': "关于退款申请，我需要了解一些信息才能帮您处理。",
            'knowledge': "让我为您查找相关信息。",
            'complaint': "很抱歉给您带来不便，我来帮您解决问题。",
            'general': "您好！我是智能客服助手，有什么可以帮助您的吗？"
        }

        return default_responses.get(intent, default_responses['general'])

    def _get_conversation_context(self, session_id: str) -> str:
        """获取对话上下文"""
        if session_id in self.conversation_state:
            history = self.conversation_state[session_id].get('history', [])
            recent_turns = history[-3:]  # 最近3轮对话
            return "\n".join([
                f"用户: {turn['user']}\n助手: {turn['assistant']}"
                for turn in recent_turns
            ])
        return ""

    def _update_conversation_state(self, session_id: str, user_input: str, response: Dict):
        """更新对话状态"""
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                'history': [],
                'intent_history': [],
                'start_time': time.time()
            }

        self.conversation_state[session_id]['history'].append({
            'user': user_input,
            'assistant': response['response'],
            'timestamp': time.time()
        })

        self.conversation_state[session_id]['intent_history'].append(response['intent'])

# 签名定义
class CustomerServiceTask(dspy.Signature):
    """客服任务"""
    task = dspy.InputField(desc="客服任务描述")
    thought = dspy.OutputField(desc="思考过程")
    action = dspy.OutputField(desc="选择的行动")
    tool_call = dspy.OutputField(desc="工具调用", required=False)
    observation = dspy.OutputField(desc="观察结果", required=False)
    final_answer = dspy.OutputField(desc="最终答案")

class ClassifyIntent(dspy.Signature):
    """意图分类"""
    user_input = dspy.InputField(desc="用户输入")
    intent = dspy.OutputField(desc="用户意图")
    confidence = dspy.OutputField(desc="分类置信度", type=float)
```

#### 3. 智能路由和决策
```python
class IntentRouter:
    """意图路由器"""

    def __init__(self):
        self.intent_patterns = {
            'order_query': [
                r'订单|查询|我的订单|订单状态|发货',
                r'快递|物流|配送'
            ],
            'refund': [
                r'退款|退货|退换货|返钱',
                r'不要了|取消订单'
            ],
            'knowledge': [
                r'怎么|如何|什么|为什么|是否',
                r'政策|规定|流程'
            ],
            'complaint': [
                r'投诉|问题|故障|错误',
                r'不满意|很差|糟糕'
            ],
            'human_transfer': [
                r'人工|客服|转接|真人',
                r'复杂|特殊|紧急'
            ]
        }

    def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """分类用户意图"""
        import re

        intent_scores = {}
        user_input_lower = user_input.lower()

        # 计算各意图的匹配分数
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, user_input_lower))
                score += matches * 2  # 每个匹配加2分

            intent_scores[intent] = score

        # 选择最高分的意图
        if max(intent_scores.values()) == 0:
            best_intent = 'general'
            confidence = 0.5
        else:
            best_intent = max(intent_scores, key=intent_scores.get)
            total_score = sum(intent_scores.values())
            confidence = intent_scores[best_intent] / total_score if total_score > 0 else 0.5

        return {
            'intent': best_intent,
            'confidence': confidence,
            'all_scores': intent_scores
        }

class SmartDecisionEngine:
    """智能决策引擎"""

    def __init__(self):
        self.rules = [
            self._handle_complicated_query,
            self._handle_urgent_request,
            self._handle_frustration,
            self._handle_first_time_user
        ]

    def make_decision(self, user_input: str, conversation_history: List = None) -> Dict[str, Any]:
        """做出决策"""
        context = {
            'user_input': user_input,
            'conversation_history': conversation_history or [],
            'time_elapsed': 0
        }

        # 应用决策规则
        for rule in self.rules:
            decision = rule(context)
            if decision['action'] != 'continue':
                return decision

        # 默认决策：正常处理
        return {
            'action': 'normal_process',
            'priority': 'normal',
            'tools_allowed': ['order_query', 'refund', 'knowledge_search'],
            'message': None
        }

    def _handle_complicated_query(self, context: Dict) -> Dict[str, Any]:
        """处理复杂查询"""
        user_input = context['user_input']
        history = context['conversation_history']

        # 检测复杂度指标
        complexity_indicators = [
            len(user_input.split()) > 50,  # 长文本
            '或者' in user_input or '另外' in user_input,  # 多个问题
            len([r for r in history[-5:] if r.get('tool_used')]) > 3  # 已使用多个工具
        ]

        if any(complexity_indicators):
            return {
                'action': 'suggest_human',
                'priority': 'high',
                'reason': '查询复杂，建议人工处理',
                'message': '您的问题比较复杂，我建议为您转接人工客服获得更好的帮助。'
            }

        return {'action': 'continue'}

    def _handle_urgent_request(self, context: Dict) -> Dict[str, Any]:
        """处理紧急请求"""
        user_input = context['user_input'].lower()

        urgent_keywords = ['紧急', '急', '马上', '立刻', '重要']
        if any(keyword in user_input for keyword in urgent_keywords):
            return {
                'action': 'priority_processing',
                'priority': 'urgent',
                'message': '我理解这很紧急，会优先为您处理。'
            }

        return {'action': 'continue'}

    def _handle_frustration(self, context: Dict) -> Dict[str, Any]:
        """处理用户沮丧情绪"""
        user_input = context['user_input'].lower()
        history = context['conversation_history']

        frustration_keywords = [
            '没用', '不好', '垃圾', '生气', '无语',
            '解决不了', '帮不了', '不会吧'
        ]

        # 检测沮丧情绪
        current_frustration = any(keyword in user_input for keyword in frustration_keywords)
        historical_frustration = sum(1 for turn in history[-5:]
                                  if any(keyword in turn.get('user', '').lower()
                                       for keyword in frustration_keywords))

        if current_frustration or historical_frustration >= 2:
            return {
                'action': 'apologize_and_transfer',
                'priority': 'high',
                'message': '很抱歉没能很好地帮助您。让我为您转接更专业的人工客服。',
                'transfer_reason': 'user_frustration'
            }

        return {'action': 'continue'}

    def _handle_first_time_user(self, context: Dict) -> Dict[str, Any]:
        """处理首次用户"""
        history = context['conversation_history']

        if len(history) <= 1:  # 首次对话
            return {
                'action': 'enhanced_guidance',
                'priority': 'normal',
                'message': '您好！我是智能助手，可以帮您查询订单、处理退款、回答问题等。请问有什么可以帮助您的？',
                'tools_allowed': ['knowledge_search']  # 先从知识库开始
            }

        return {'action': 'continue'}

# 增强的智能体
class EnhancedCustomerServiceAgent(CustomerServiceAgent):
    """增强版客服智能体"""

    def __init__(self):
        super().__init__()
        self.intent_router = IntentRouter()
        self.decision_engine = SmartDecisionEngine()

    def forward(self, user_input: str, session_id: str = None) -> dspy.Prediction:
        """增强版处理流程"""

        # 1. 意图识别
        intent_result = self.intent_router.classify_intent(user_input)

        # 2. 获取对话历史
        history = self._get_conversation_history(session_id) if session_id else []

        # 3. 智能决策
        decision = self.decision_engine.make_decision(user_input, history)

        # 4. 根据决策执行相应操作
        if decision['action'] == 'suggest_human':
            return self._transfer_to_human(decision['message'], decision['reason'])
        elif decision['action'] == 'priority_processing':
            # 使用更多工具，增加处理尝试次数
            return self._priority_processing(user_input, intent_result, session_id)
        elif decision['action'] == 'apologize_and_transfer':
            return self._apologize_and_transfer(decision['message'])
        else:
            # 正常处理流程
            return super().forward(user_input, session_id)

    def _transfer_to_human(self, message: str, reason: str) -> dspy.Prediction:
        """转接人工"""
        transfer_tool = self.tools['human_transfer']
        transfer_result = transfer_tool.execute(reason=reason)

        return dspy.Prediction(
            response=f"{message} {transfer_result.get('message', '')}",
            intent='human_transfer',
            tools_used=['human_transfer'],
            confidence=0.9
        )

    def _priority_processing(self, user_input: str, intent_result: Dict, session_id: str) -> dspy.Prediction:
        """优先处理"""
        # 设置更 aggressive 的处理参数
        result = super().forward(user_input, session_id)

        # 增加处理尝试
        if result.confidence < 0.7:
            # 尝试使用不同工具或重新处理
            result = self._retry_with_different_approach(user_input, intent_result, session_id)

        return result

    def _apologize_and_transfer(self, message: str) -> dspy.Prediction:
        """道歉并转接"""
        transfer_tool = self.tools['human_transfer']
        transfer_result = transfer_tool.execute(reason='user_frustration')

        return dspy.Prediction(
            response=f"{message} {transfer_result.get('message', '')}",
            intent='human_transfer',
            tools_used=['human_transfer'],
            confidence=0.8
        )

    def _retry_with_different_approach(self, user_input: str, intent_result: Dict, session_id: str) -> dspy.Prediction:
        """使用不同方法重试"""
        # 这里可以实现更复杂的重试逻辑
        # 例如：使用不同的工具组合、重新分析用户意图等

        # 简化实现：直接转接人工
        return self._transfer_to_human(
            "让我为您转接人工客服以获得更好的帮助。",
            "retry_failed"
        )
```

### 部署和监控

#### 1. 实时对话接口
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio

app = FastAPI(title="Customer Service Agent API")

class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

manager = ConnectionManager()
agent = EnhancedCustomerServiceAgent()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)

    try:
        # 发送欢迎消息
        await manager.send_message(session_id, {
            "type": "welcome",
            "message": "您好！我是智能客服助手，有什么可以帮助您的吗？"
        })

        while True:
            # 接收用户消息
            data = await websocket.receive_text()
            user_message = json.loads(data)

            # 处理消息
            if user_message.get("type") == "message":
                user_input = user_message.get("content", "")

                # 调用智能体
                result = agent.forward(user_input, session_id)

                # 发送响应
                await manager.send_message(session_id, {
                    "type": "response",
                    "content": result.response,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "tools_used": result.tools_used
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.post("/chat")
async def chat_endpoint(request: dict):
    """HTTP聊天接口"""
    user_input = request.get("message", "")
    session_id = request.get("session_id", "default")

    result = agent.forward(user_input, session_id)

    return {
        "response": result.response,
        "intent": result.intent,
        "confidence": result.confidence,
        "session_id": session_id
    }
```

### 扩展思路

#### 1. 情感分析集成
```python
class EmotionalIntelligence:
    """情感智能模块"""

    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """分析情感状态"""
        # 简化的情感分析
        emotions = {
            'happy': ['开心', '满意', '好的', '谢谢'],
            'angry': ['生气', '愤怒', '不满', '糟糕'],
            'sad': ['难过', '失望', '伤心', '郁闷'],
            'anxious': ['担心', '着急', '焦虑', '紧急']
        }

        emotion_scores = {}
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text)
            emotion_scores[emotion] = score

        main_emotion = max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values()) > 0 else 'neutral'

        return {
            'emotion': main_emotion,
            'scores': emotion_scores,
            'intensity': max(emotion_scores.values())
        }

    def generate_empathetic_response(self, emotion: str, context: str) -> str:
        """生成共情回复"""
        empathetic_responses = {
            'angry': "很抱歉让您有这样的体验，我会尽力帮您解决问题。",
            'anxious': "我理解您的担心，让我们一起看看怎么处理这个问题。",
            'sad': "听到这个消息我很难过，希望能为您提供帮助。",
            'happy': "很高兴能帮到您！还有其他需要帮助的吗？"
        }

        return empathetic_responses.get(emotion, "我明白了，让我来帮您。")
```

#### 2. 多语言支持
```python
class MultilingualSupport:
    """多语言支持"""

    def __init__(self):
        self.language_detector = LanguageDetector()
        self.translator = Translator()

    def process_multilingual_input(self, text: str) -> Dict[str, Any]:
        """处理多语言输入"""
        detected_lang = self.language_detector.detect(text)

        if detected_lang != 'zh':  # 如果不是中文
            # 翻译到中文
            translated_text = self.translator.translate(text, from_lang=detected_lang, to_lang='zh')
            return {
                'original_text': text,
                'translated_text': translated_text,
                'original_lang': detected_lang,
                'processed_text': translated_text
            }
        else:
            return {
                'original_text': text,
                'translated_text': text,
                'original_lang': 'zh',
                'processed_text': text
            }

    def translate_response(self, response: str, target_lang: str) -> str:
        """翻译回复"""
        if target_lang == 'zh':
            return response
        else:
            return self.translator.translate(response, from_lang='zh', to_lang=target_lang)
```

---

*（由于篇幅限制，剩余3个案例"文本处理分析"、"复杂推理系统"和"多模态应用"的详细内容将在下一个文档中继续）*