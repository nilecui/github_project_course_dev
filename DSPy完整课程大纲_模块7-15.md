# DSPy完整课程大纲 - 模块7-15补充

本文档是《DSPy完整课程大纲v2.0》的补充部分，包含模块7-15的详细内容。

---

## 模块7：智能体开发（第5-6周，15小时）

**学习目标**
- 掌握ReAct智能体的原理和实现
- 学会工具集成和使用
- 理解智能体的规划和决策机制
- 开发实用的智能体应用

**核心内容**

### 7.1 ReAct智能体基础（5小时）

**ReAct模式原理**
- Reasoning（推理）+ Acting（行动）
- 思考-行动-观察循环
- 工具选择和使用策略

```python
import dspy
from dspy import Tool

class ReActAgent(dspy.Module):
    """基础ReAct智能体"""

    def __init__(self, tools: List[Tool], max_iters=5):
        super().__init__()
        self.tools = tools
        self.max_iters = max_iters

        # ReAct模块
        class AgentSignature(dspy.Signature):
            """智能体任务执行"""
            task = dspy.InputField(desc="要完成的任务")
            tool_results = dspy.InputField(desc="工具执行结果历史")
            thought = dspy.OutputField(desc="当前思考")
            action = dspy.OutputField(desc="要执行的动作")
            action_input = dspy.OutputField(desc="动作的输入参数")

        self.agent = dspy.ChainOfThought(AgentSignature)

    def forward(self, task: str):
        history = []

        for iteration in range(self.max_iters):
            # 获取下一步行动
            result = self.agent(
                task=task,
                tool_results=str(history)
            )

            # 如果完成，返回结果
            if result.action.lower() == "finish":
                return dspy.Prediction(
                    answer=result.action_input,
                    steps=history
                )

            # 执行工具
            tool = self._get_tool(result.action)
            if tool:
                observation = tool.func(result.action_input)
                history.append({
                    "thought": result.thought,
                    "action": result.action,
                    "input": result.action_input,
                    "observation": observation
                })

        return dspy.Prediction(
            answer="未能在规定步骤内完成任务",
            steps=history
        )

    def _get_tool(self, tool_name: str):
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        return None
```

### 7.2 工具系统设计（5小时）

**工具定义和注册**
```python
from typing import Callable, Any
from pydantic import BaseModel

class ToolSchema(BaseModel):
    """工具模式定义"""
    name: str
    description: str
    parameters: dict

def tool(name: str, description: str):
    """工具装饰器"""
    def decorator(func: Callable) -> Tool:
        return Tool(
            func=func,
            name=name,
            desc=description
        )
    return decorator

# 定义工具
@tool("search_web", "在互联网上搜索信息")
def search_web(query: str) -> str:
    """网络搜索工具"""
    # 实现搜索逻辑
    return f"关于'{query}'的搜索结果..."

@tool("calculator", "执行数学计算")
def calculator(expression: str) -> float:
    """计算器工具"""
    try:
        return eval(expression)
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool("file_reader", "读取文件内容")
def read_file(filepath: str) -> str:
    """文件读取工具"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"读取错误: {str(e)}"

@tool("database_query", "查询数据库")
def query_database(sql: str) -> str:
    """数据库查询工具"""
    # 实现数据库查询逻辑
    return "查询结果..."

# 工具集合
tools = [search_web, calculator, read_file, query_database]
```

**复杂工具集成**
```python
class APITool(Tool):
    """API调用工具"""

    def __init__(self, api_config: dict):
        self.config = api_config
        super().__init__(
            func=self._call_api,
            name=api_config['name'],
            desc=api_config['description']
        )

    def _call_api(self, params: dict) -> Any:
        """调用API"""
        import requests

        response = requests.post(
            self.config['endpoint'],
            json=params,
            headers=self.config.get('headers', {})
        )

        return response.json()

# 使用API工具
weather_api = APITool({
    'name': 'get_weather',
    'description': '获取指定城市的天气信息',
    'endpoint': 'https://api.weather.com/v1/forecast',
    'headers': {'Authorization': 'Bearer TOKEN'}
})
```

### 7.3 高级智能体模式（5小时）

**多智能体协作**
```python
class MultiAgentSystem(dspy.Module):
    """多智能体协作系统"""

    def __init__(self):
        super().__init__()

        # 协调者智能体
        self.coordinator = CoordinatorAgent()

        # 专家智能体
        self.experts = {
            "researcher": ResearchAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent()
        }

    def forward(self, task: str):
        # 任务分解
        subtasks = self.coordinator.decompose_task(task)

        # 分配给专家
        results = {}
        for subtask in subtasks:
            agent_type = subtask['agent_type']
            if agent_type in self.experts:
                result = self.experts[agent_type](
                    task=subtask['description']
                )
                results[subtask['id']] = result

        # 整合结果
        final_result = self.coordinator.integrate_results(
            task=task,
            subtask_results=results
        )

        return final_result


class AgentWithMemory(dspy.Module):
    """带记忆的智能体"""

    def __init__(self, memory_size=100):
        super().__init__()

        # 短期记忆（对话历史）
        self.short_term_memory = []

        # 长期记忆（知识库）
        self.long_term_memory = VectorStore()

        # 工作记忆（当前任务状态）
        self.working_memory = {}

        self.agent = dspy.ReAct(tools=self.tools)

    def forward(self, task: str):
        # 检索相关记忆
        relevant_memories = self.long_term_memory.search(task, k=5)

        # 更新工作记忆
        self.working_memory['current_task'] = task
        self.working_memory['relevant_knowledge'] = relevant_memories

        # 执行任务
        result = self.agent(
            task=task,
            context=str(relevant_memories)
        )

        # 更新记忆
        self.short_term_memory.append({
            'task': task,
            'result': result
        })

        if len(self.short_term_memory) > self.memory_size:
            # 归档到长期记忆
            old_memory = self.short_term_memory.pop(0)
            self.long_term_memory.add(old_memory)

        return result
```

**智能体应用案例**
```python
class CustomerServiceAgent(dspy.Module):
    """客服智能体"""

    def __init__(self):
        super().__init__()

        # 工具
        tools = [
            Tool(self.search_kb, "search_kb", "搜索知识库"),
            Tool(self.check_order, "check_order", "查询订单状态"),
            Tool(self.submit_ticket, "submit_ticket", "提交工单"),
        ]

        self.agent = dspy.ReAct(tools=tools)

        # 对话管理
        self.conversation_state = {}

    def search_kb(self, query: str) -> str:
        """搜索知识库"""
        # 实现知识库搜索
        return "相关知识..."

    def check_order(self, order_id: str) -> str:
        """查询订单"""
        # 实现订单查询
        return f"订单{order_id}的状态..."

    def submit_ticket(self, issue: str) -> str:
        """提交工单"""
        # 实现工单提交
        return f"工单已创建: {issue}"

    def forward(self, user_input: str, session_id: str):
        # 获取对话历史
        history = self.conversation_state.get(session_id, [])

        # 执行智能体
        result = self.agent(
            task=user_input,
            context=str(history)
        )

        # 更新对话状态
        history.append({
            'user': user_input,
            'agent': result.answer
        })
        self.conversation_state[session_id] = history

        return result
```

**实践任务**
- [ ] 实现基础ReAct智能体
- [ ] 集成5种不同工具
- [ ] 开发带记忆的智能体
- [ ] 实现多智能体协作系统
- [ ] 构建客服智能体应用

**评估标准**
- 智能体功能完整性 ✓
- 工具集成正确性 ✓
- 任务完成质量 ✓
- 系统稳定性 ✓

---

## 模块8：复杂推理系统（第6周，12小时）

**学习目标**
- 掌握复杂推理任务的实现
- 理解不同推理策略的应用
- 优化推理质量和效率
- 处理多步骤推理问题

**核心内容**

### 8.1 推理模式对比（4小时）

**ChainOfThought vs ProgramOfThought**
```python
# ChainOfThought - 适合需要自然语言推理的任务
class MathWordProblem_CoT(dspy.Module):
    def __init__(self):
        super().__init__()

        class Solver(dspy.Signature):
            problem = dspy.InputField()
            solution = dspy.OutputField()

        self.solver = dspy.ChainOfThought(Solver)

    def forward(self, problem: str):
        return self.solver(problem=problem)

# ProgramOfThought - 适合需要精确计算的任务
class MathWordProblem_PoT(dspy.Module):
    def __init__(self):
        super().__init__()

        class Solver(dspy.Signature):
            problem = dspy.InputField()
            code = dspy.OutputField(desc="Python代码")
            result = dspy.OutputField()

        self.solver = dspy.ProgramOfThought(Solver)

    def forward(self, problem: str):
        return self.solver(problem=problem)

# 性能对比
problems = [
    "如果苹果每个3元，买7个苹果和5个橙子(每个2元)一共多少钱？",
    "一个数的平方根加上这个数的立方根等于10，这个数是多少？"
]

cot_solver = MathWordProblem_CoT()
pot_solver = MathWordProblem_PoT()

for problem in problems:
    print(f"\n问题: {problem}")
    print(f"CoT答案: {cot_solver(problem).solution}")
    print(f"PoT答案: {pot_solver(problem).result}")
```

### 8.2 复杂推理策略（4小时）

**分治策略**
```python
class DivideAndConquer(dspy.Module):
    """分治推理模块"""

    def __init__(self):
        super().__init__()

        # 问题分解器
        class DecomposeProblem(dspy.Signature):
            problem = dspy.InputField()
            subproblems = dspy.OutputField(desc="子问题列表")

        self.decomposer = dspy.ChainOfThought(DecomposeProblem)

        # 子问题求解器
        self.subsolver = dspy.ChainOfThought(SolveProblem)

        # 结果合并器
        class MergeSolutions(dspy.Signature):
            problem = dspy.InputField()
            subsolutions = dspy.InputField()
            final_solution = dspy.OutputField()

        self.merger = dspy.ChainOfThought(MergeSolutions)

    def forward(self, problem: str):
        # 1. 分解问题
        decomposition = self.decomposer(problem=problem)
        subproblems = eval(decomposition.subproblems)

        # 2. 解决子问题
        subsolutions = []
        for subproblem in subproblems:
            solution = self.subsolver(problem=subproblem)
            subsolutions.append(solution.answer)

        # 3. 合并结果
        final = self.merger(
            problem=problem,
            subsolutions=str(subsolutions)
        )

        return final
```

**启发式搜索**
```python
class HeuristicReasoning(dspy.Module):
    """启发式推理模块"""

    def __init__(self):
        super().__init__()

        # 候选生成器
        class GenerateCandidates(dspy.Signature):
            problem = dspy.InputField()
            candidates = dspy.OutputField(desc="候选解决方案列表")

        self.generator = dspy.ChainOfThought(GenerateCandidates)

        # 评估器
        class EvaluateCandidate(dspy.Signature):
            problem = dspy.InputField()
            candidate = dspy.InputField()
            score = dspy.OutputField(desc="评分0-10")
            rationale = dspy.OutputField()

        self.evaluator = dspy.Predict(EvaluateCandidate)

    def forward(self, problem: str, top_k=3):
        # 生成候选解
        candidates_result = self.generator(problem=problem)
        candidates = eval(candidates_result.candidates)

        # 评估候选解
        scored_candidates = []
        for candidate in candidates:
            evaluation = self.evaluator(
                problem=problem,
                candidate=candidate
            )
            scored_candidates.append({
                'candidate': candidate,
                'score': float(evaluation.score),
                'rationale': evaluation.rationale
            })

        # 排序并返回最佳方案
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return dspy.Prediction(
            best_solution=scored_candidates[0],
            alternatives=scored_candidates[1:top_k]
        )
```

### 8.3 推理质量保证（4小时）

**自我验证机制**
```python
class SelfVerifyingReasoner(dspy.Module):
    """带自我验证的推理器"""

    def __init__(self):
        super().__init__()

        # 推理器
        self.reasoner = dspy.ChainOfThought(SolveProblem)

        # 验证器
        class VerifySolution(dspy.Signature):
            problem = dspy.InputField()
            proposed_solution = dspy.InputField()
            reasoning = dspy.InputField()
            is_correct = dspy.OutputField(desc="true/false")
            issues = dspy.OutputField(desc="发现的问题")

        self.verifier = dspy.ChainOfThought(VerifySolution)

        # 修正器
        class CorrectSolution(dspy.Signature):
            problem = dspy.InputField()
            wrong_solution = dspy.InputField()
            issues = dspy.InputField()
            corrected_solution = dspy.OutputField()

        self.corrector = dspy.ChainOfThought(CorrectSolution)

    def forward(self, problem: str, max_attempts=3):
        for attempt in range(max_attempts):
            # 推理
            solution = self.reasoner(problem=problem)

            # 验证
            verification = self.verifier(
                problem=problem,
                proposed_solution=solution.answer,
                reasoning=solution.reasoning
            )

            if verification.is_correct.lower() == "true":
                return solution

            # 修正
            if attempt < max_attempts - 1:
                corrected = self.corrector(
                    problem=problem,
                    wrong_solution=solution.answer,
                    issues=verification.issues
                )
                solution = corrected

        return solution
```

**实践任务**
- [ ] 实现3种推理模式对比
- [ ] 开发分治推理系统
- [ ] 实现启发式搜索
- [ ] 添加自我验证机制
- [ ] 性能基准测试

**评估标准**
- 推理正确率 ✓
- 推理效率 ✓
- 代码质量 ✓

---

## 阶段三：优化与部署（第7-9周）

### 模块9：自动优化算法（第7周，18小时）

**学习目标**
- 掌握DSPy的各种优化算法
- 理解优化原理和应用场景
- 实现自动化优化流程
- 评估优化效果

**核心内容**

**9.1 BootstrapFewShot详解（6小时）**

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

class RAGSystem(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 准备训练数据
trainset = [
    dspy.Example(
        question="DSPy是什么？",
        answer="DSPy是一个用于编程语言模型的框架"
    ).with_inputs("question"),
    # 更多示例...
]

# 定义评估指标
def exact_match(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# 创建优化器
optimizer = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=8,  # 最多8个示例
    max_labeled_demos=4,  # 最多4个标注示例
    max_rounds=3  # 最多3轮自举
)

# 编译模块
rag_system = RAGSystem()
compiled_rag = optimizer.compile(
    rag_system,
    trainset=trainset
)

# 使用优化后的系统
result = compiled_rag(question="DSPy的优势是什么？")
```

**9.2 MIPROv2优化器（6小时）**

```python
from dspy.teleprompt import MIPROv2

# MIPRO优化器配置
optimizer = MIPROv2(
    metric=metric_function,
    prompt_model=dspy.OpenAI(model="gpt-4"),  # 用于生成prompt的模型
    task_model=dspy.OpenAI(model="gpt-3.5-turbo"),  # 任务执行模型
    num_candidates=10,  # 候选prompt数量
    init_temperature=1.0,  # 初始温度
    verbose=True
)

# 优化
compiled_program = optimizer.compile(
    program,
    trainset=train_data,
    num_trials=50,  # 尝试次数
    max_bootstrapped_demos=5,
    max_labeled_demos=3,
    eval_kwargs=dict(num_threads=8, display_progress=True),
    requires_permission_to_run=False
)
```

**9.3 COPRO和GEPA优化器（6小时）**

```python
from dspy.teleprompt import COPRO, GEPA

# COPRO - 协作提示优化
copro_optimizer = COPRO(
    metric=metric_function,
    breadth=10,  # 搜索广度
    depth=3,  # 搜索深度
    init_temperature=1.4
)

compiled_copro = copro_optimizer.compile(
    program,
    trainset=train_data,
    eval_kwargs=dict(num_threads=4)
)

# GEPA - 反思式提示进化
gepa_optimizer = GEPA(
    metric=metric_function,
    breadth=10,
    depth=3,
    num_optim_samples=50
)

compiled_gepa = gepa_optimizer.compile(
    program,
    trainset=train_data,
    valset=val_data
)
```

**实践任务**
- [ ] 使用BootstrapFewShot优化RAG系统
- [ ] 实现MIPROv2优化流程
- [ ] 对比不同优化器效果
- [ ] 建立优化效果评估体系

---

### 模块10：评估与调试（第7-8周，15小时）

**核心内容**

**10.1 评估框架（5小时）**

```python
from dspy.evaluate import Evaluate

# 定义多个评估指标
def accuracy_metric(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

def semantic_similarity(example, prediction, trace=None):
    # 使用嵌入计算语义相似度
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    emb1 = model.encode(example.answer)
    emb2 = model.encode(prediction.answer)

    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity > 0.8

# 创建评估器
evaluator = Evaluate(
    devset=test_data,
    num_threads=8,
    display_progress=True,
    display_table=True,
    return_outputs=True
)

# 执行评估
results = evaluator(
    compiled_program,
    metric=accuracy_metric
)

print(f"准确率: {results['accuracy']}")
print(f"平均分: {results['avg_metric']}")
```

**10.2 调试技术（5小时）**

```python
# 历史记录检查
dspy.inspect_history(n=3)  # 查看最近3次调用

# 详细日志
dspy.enable_logging()  # 启用详细日志
result = program(input_data)
dspy.disable_logging()

# 逐步调试
for step in program.trace():
    print(f"Step: {step.name}")
    print(f"Input: {step.input}")
    print(f"Output: {step.output}")
```

**10.3 性能分析（5小时）**

```python
import time
from functools import wraps

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {
            'latency': [],
            'token_usage': [],
            'cache_hits': 0,
            'cache_misses': 0
        }

    def monitor(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)

            latency = time.time() - start_time
            self.metrics['latency'].append(latency)

            return result
        return wrapper

    def report(self):
        return {
            'avg_latency': sum(self.metrics['latency']) / len(self.metrics['latency']),
            'p95_latency': np.percentile(self.metrics['latency'], 95),
            'p99_latency': np.percentile(self.metrics['latency'], 99)
        }
```

---

### 模块11：生产部署（第8-9周，15小时）

**11.1 容器化部署（5小时）**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  dspy-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: dspy_prod
      POSTGRES_USER: dspy
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

**11.2 API服务实现（5小时）**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dspy

app = FastAPI()

# 加载模型
compiled_model = dspy.load("models/compiled_rag.json")

class QueryRequest(BaseModel):
    question: str
    session_id: str = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = compiled_model(question=request.question)

        return QueryResponse(
            answer=result.answer,
            confidence=result.confidence,
            sources=result.sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**11.3 监控系统（5小时）**

```python
from prometheus_client import Counter, Histogram, generate_latest
import logging

# Prometheus指标
request_counter = Counter('dspy_requests_total', 'Total requests')
latency_histogram = Histogram('dspy_latency_seconds', 'Request latency')
error_counter = Counter('dspy_errors_total', 'Total errors')

@app.middleware("http")
async def monitor_requests(request, call_next):
    request_counter.inc()

    with latency_histogram.time():
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            error_counter.inc()
            raise

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## 阶段四：高级专题（第10-12周）

### 模块13-15：高级主题与毕业项目

由于篇幅限制，这些高级模块将包括：
- 自定义组件开发
- 多模态应用
- 前沿技术探索
- 毕业项目实战

详细内容请参考主课程大纲文档。

---

## 总结

本补充文档完善了DSPy完整课程大纲的模块7-12部分，包含：
- 智能体开发的完整实现
- 复杂推理系统的设计模式
- 自动优化算法的深入讲解
- 评估调试的最佳实践
- 生产部署的完整流程

**下一步学习建议**：
1. 按顺序完成每个模块的实践任务
2. 参与课程配套项目开发
3. 加入DSPy社区交流学习
4. 准备毕业项目设计

**课程更新**：本文档会持续更新，请关注最新版本。
