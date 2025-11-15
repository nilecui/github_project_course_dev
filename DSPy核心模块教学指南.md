# DSPy核心模块教学指南

## 📚 模块教学概览

本文档详细介绍了DSPy 8个核心功能模块的教学要点、API用法和最佳实践。每个模块包含：学习目标、核心概念、API详解、代码示例、常见问题和实践任务。

---

## 🏗️ 模块1：基础原语 (Primitives)

### 学习目标
- 理解DSPy的基础构建块
- 掌握核心类和接口的使用
- 学会数据结构的标准化表示

### 核心概念
- **Module类**：所有DSPy组件的基类
- **Example类**：数据样本的标准化表示
- **Prediction类**：预测结果的封装
- **Completions类**：多个预测结果的管理

### API详解

#### 1. Module基类
```python
import dspy
from typing import Any, Dict, Optional

class CustomModule(dspy.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化子模块
        self.submodule = dspy.Predict(dspy.Signature)

    def forward(self, *args, **kwargs) -> dspy.Prediction:
        """前向传播逻辑"""
        # 处理输入
        processed_input = self.process_input(*args, **kwargs)

        # 调用子模块
        result = self.submodule(**processed_input)

        # 返回预测结果
        return dspy.Prediction(**result)

    def process_input(self, *args, **kwargs) -> Dict[str, Any]:
        """输入预处理"""
        pass

# 使用示例
module = CustomModule()
result = module(input_text="Hello DSPy")
```

**关键特性：**
- `named_parameters()`: 获取所有可训练参数
- `save()`/`load()`: 模型保存和加载
- `copy()`: 模型复制
- `compile()`: 模型编译和优化

#### 2. Example类
```python
from dspy import Example

# 创建示例数据
example1 = Example(
    question="什么是机器学习？",
    answer="机器学习是人工智能的一个分支...",
    context="人工智能领域的相关知识",
    difficulty="简单"
)

# 字典式访问
print(example1.question)  # 输出问题
print(example1["answer"])  # 输出答案

# 批量创建示例
examples = [
    Example(question=q, answer=a)
    for q, a in zip(questions, answers)
]

# 示例数据操作
examples_with_metadata = [
    example.with_inputs(id=i, category="tech")
    for i, example in enumerate(examples)
]
```

**高级用法：**
```python
# 数据转换和增强
def augment_example(example: Example) -> Example:
    """示例数据增强"""
    augmented = example.copy()
    augmented.question = f"请详细回答：{augmented.question}"
    augmented.expected_length = "详细"
    return augmented

# 数据过滤
def filter_examples(examples: list[Example], min_length: int = 50) -> list[Example]:
    """过滤示例数据"""
    return [
        ex for ex in examples
        if len(ex.answer) >= min_length
    ]
```

#### 3. Prediction类
```python
from dspy import Prediction

# 创建预测结果
prediction = Prediction(
    answer="这是预测的答案",
    reasoning="推理过程：首先分析问题...",
    confidence=0.85,
    sources=["来源1", "来源2"]
)

# 访问预测结果
print(prediction.answer)
print(prediction.confidence)

# 预测结果比较
def compare_predictions(pred1: Prediction, pred2: Prediction) -> Prediction:
    """比较两个预测结果"""
    if pred1.confidence > pred2.confidence:
        return pred1
    return pred2

# 结果聚合
def aggregate_predictions(predictions: list[Prediction]) -> Prediction:
    """聚合多个预测结果"""
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions)

    # 选择置信度最高的答案
    best_prediction = max(predictions, key=lambda p: p.confidence)

    return Prediction(
        answer=best_prediction.answer,
        confidence=avg_confidence,
        reasoning="聚合了多个预测结果"
    )
```

#### 4. Completions类
```python
from dspy import Completions

# 创建多个预测结果
completions = Completions([
    Prediction(answer="答案1", confidence=0.9),
    Prediction(answer="答案2", confidence=0.7),
    Prediction(answer="答案3", confidence=0.8)
])

# 访问所有结果
for i, completion in enumerate(completions):
    print(f"选项{i+1}: {completion.answer} (置信度: {completion.confidence})")

# 选择最佳结果
best_completion = completions.best()
print(f"最佳答案: {best_completion.answer}")

# 按置信度排序
sorted_completions = completions.sorted_by_confidence()
```

### 常见问题

**Q1: Module和普通Python类有什么区别？**
A: Module提供了DSPy特有的功能：
- 自动参数跟踪和管理
- 支持编译和优化
- 标准化的前向传播接口
- 内存的保存和加载

**Q2: Example和普通字典有什么优势？**
A: Example提供了：
- 类型安全的字段访问
- 元数据管理功能
- 数据验证和转换
- 与DSPy生态系统的无缝集成

**Q3: 如何处理大型数据集？**
A: 建议使用：
- 分批处理
- 数据管道和缓存
- 内存优化技术
- 并行处理

### 实践任务
- [ ] 创建自定义Module类
- [ ] 实现Example数据的预处理
- [ ] 开发Prediction结果的后处理
- [ ] 构建Completions管理工具

---

## 📝 模块2：签名系统 (Signatures)

### 学习目标
- 掌握签名系统的设计理念
- 学会定义复杂任务规范
- 理解类型安全的数据流

### 核心概念
- **Signature类**：任务规范的声明式定义
- **InputField/OutputField**：字段描述和约束
- **类型系统**：复杂类型约束和验证

### API详解

#### 1. 基础签名定义
```python
import dspy
from typing import List, Optional

# 简单签名
class QuestionAnswering(dspy.Signature):
    """回答用户问题"""
    question = dspy.InputField(desc="用户的问题")
    answer = dspy.OutputField(desc="准确的答案")

# 复杂签名
class ComplexQA(dspy.Signature):
    """基于上下文回答复杂问题"""
    context = dspy.InputField(
        desc="相关文档片段",
        type=str,
        prefix="上下文："
    )
    question = dspy.InputField(
        desc="用户查询问题",
        type=str,
        prefix="问题："
    )
    answer = dspy.OutputField(
        desc="基于上下文的详细答案",
        type=str,
        prefix="答案："
    )
    confidence = dspy.OutputField(
        desc="答案置信度(0-1)",
        type=float,
        prefix="置信度："
    )
    sources = dspy.OutputField(
        desc="引用的具体来源",
        type=List[str],
        prefix="来源："
    )
```

#### 2. 高级签名技巧
```python
# 条件字段
class ConditionalQA(dspy.Signature):
    """条件性问答"""
    question = dspy.InputField(desc="用户问题")
    has_context = dspy.InputField(desc="是否有上下文", type=bool)
    context = dspy.InputField(desc="上下文信息", type=str, required=False)
    answer = dspy.OutputField(desc="答案")
    needs_clarification = dspy.OutputField(desc="是否需要澄清", type=bool)

# 枚举类型
class Classification(dspy.Signature):
    """文本分类"""
    text = dspy.InputField(desc="待分类文本")
    category = dspy.OutputField(
        desc="分类结果",
        type=str,
        choices=["技术", "商业", "教育", "娱乐"]
    )
    confidence = dspy.OutputField(desc="分类置信度", type=float)

# 嵌套结构
class StructuredAnalysis(dspy.Signature):
    """结构化分析"""
    content = dspy.InputField(desc="分析内容")
    analysis = dspy.OutputField(
        desc="结构化分析结果",
        type=Dict[str, Any]
    )
```

#### 3. 签名组合和继承
```python
# 基础签名
class BaseQA(dspy.Signature):
    question = dspy.InputField(desc="问题")
    answer = dspy.OutputField(desc="答案")

# 继承扩展
class EnhancedQA(BaseQA):
    context = dspy.InputField(desc="上下文", required=False)
    confidence = dspy.OutputField(desc="置信度", type=float)
    reasoning = dspy.OutputField(desc="推理过程")

# 签名组合
class MultiTaskQA(dspy.Signature):
    """多任务问答"""
    primary_task = BaseQA
    secondary_task = EnhancedQA
    task_priority = dspy.InputField(desc="任务优先级", type=str)
```

#### 4. 动态签名创建
```python
def create_dynamic_signature(task_name: str, inputs: list, outputs: list) -> type:
    """动态创建签名类"""

    class_attrs = {
        '__doc__': f"{task_name}任务签名",
        '__annotations__': {}
    }

    # 动态添加输入字段
    for field_name, field_desc in inputs:
        class_attrs[field_name] = dspy.InputField(desc=field_desc)
        class_attrs['__annotations__'][field_name] = str

    # 动态添加输出字段
    for field_name, field_desc in outputs:
        class_attrs[field_name] = dspy.OutputField(desc=field_desc)
        class_attrs['__annotations__'][field_name] = str

    return type(f"{task_name}Signature", (dspy.Signature,), class_attrs)

# 使用示例
QA_Signature = create_dynamic_signature(
    "自定义问答",
    [("question", "问题"), ("context", "上下文")],
    [("answer", "答案"), ("confidence", "置信度")]
)
```

### 字段验证和约束
```python
from pydantic import BaseModel, validator

class ValidatedQA(dspy.Signature):
    """带验证的问答签名"""
    question = dspy.InputField(desc="问题", min_length=5, max_length=500)
    answer = dspy.OutputField(desc="答案", min_length=10)

    # 自定义验证
    @validator('question')
    def validate_question(cls, v):
        if not v.endswith('?') and not v.endswith('？'):
            raise ValueError('问题应该以问号结尾')
        return v

    @validator('answer')
    def validate_answer(cls, v):
        if len(v.split()) < 3:
            raise ValueError('答案至少需要3个词')
        return v
```

### 常见问题

**Q1: 如何选择合适的字段类型？**
A: 根据数据特征选择：
- `str`: 文本内容
- `int/float`: 数值数据
- `bool`: 布尔值
- `List[str]`: 文本列表
- `Dict[str, Any]`: 结构化数据

**Q2: 字段描述如何编写最有效？**
A: 好的描述应该：
- 明确说明字段用途
- 提供格式示例
- 指定约束条件
- 使用简洁清晰的语言

**Q3: 如何处理可选字段？**
A: 使用`required=False`参数：
```python
class OptionalQA(dspy.Signature):
    question = dspy.InputField(desc="问题", required=True)
    context = dspy.InputField(desc="可选上下文", required=False)
    answer = dspy.OutputField(desc="答案", required=True)
```

### 实践任务
- [ ] 定义5种不同类型的任务签名
- [ ] 实现字段验证和约束
- [ ] 创建动态签名生成器
- [ ] 设计签名组合模式

---

## 🔮 模块3：预测模块 (Predict)

### 学习目标
- 掌握各种预测模块的使用方法
- 理解不同推理模式的适用场景
- 学会模块组合和优化

### 核心概念
- **Predict**: 基础预测功能
- **ChainOfThought**: 思维链推理
- **ReAct**: 推理+行动智能体
- **ProgramOfThought**: 程序化思维

### API详解

#### 1. 基础预测器
```python
import dspy

# 简单预测
class QA(dspy.Signature):
    question = dspy.InputField(desc="问题")
    answer = dspy.OutputField(desc="答案")

predict = dspy.Predict(QA)
result = predict(question="什么是DSPy？")
print(result.answer)

# 带配置的预测
predict_with_config = dspy.Predict(
    QA,
    temperature=0.7,
    max_tokens=500,
    n=1
)
```

#### 2. 思维链推理
```python
class MathProblem(dspy.Signature):
    problem = dspy.InputField(desc="数学问题")
    reasoning = dspy.OutputField(desc="解题步骤")
    answer = dspy.OutputField(desc="最终答案")

# 思维链预测器
cot = dspy.ChainOfThought(MathProblem)
result = cot(problem="如果2x + 5 = 13，求x的值")

# 输出结果
print("推理过程:", result.reasoning)
print("答案:", result.answer)

# 自定义思维链提示
class CustomCoT(dspy.ChainOfThought):
    def __init__(self, signature, **kwargs):
        super().__init__(signature, **kwargs)
        # 自定义思维链提示模板
        self.cot_prompt = """请按以下步骤思考：
1. 理解问题
2. 分析已知条件
3. 制定解决方案
4. 执行计算
5. 验证结果
"""
```

#### 3. ReAct智能体
```python
from dspy.predict.react import ReAct

class AgentTask(dspy.Signature):
    task = dspy.InputField(desc="任务描述")
    observation = dspy.InputField(desc="观察结果", required=False)
    thought = dspy.OutputField(desc="思考过程")
    action = dspy.OutputField(desc="下一步行动")
    final_answer = dspy.OutputField(desc="最终答案", required=False)

# 定义工具
class CalculatorTool:
    def __init__(self):
        self.name = "calculator"
        self.description = "执行数学计算"

    def __call__(self, expression):
        try:
            return eval(expression)
        except:
            return "计算错误"

class WebSearchTool:
    def __init__(self):
        self.name = "search"
        self.description = "网络搜索"

    def __call__(self, query):
        # 模拟搜索结果
        return f"搜索'{query}'的结果..."

# 创建ReAct智能体
tools = {
    'calculator': CalculatorTool(),
    'search': WebSearchTool()
}

agent = dspy.ReAct(AgentTask, tools=tools)

# 执行任务
result = agent(task="计算2024年到2025年有多少天")
print(result.final_answer)
```

#### 4. 程序化思维
```python
class CodeProblem(dspy.Signature):
    problem = dspy.InputField(desc="编程问题")
    code = dspy.OutputField(desc="解决方案代码")
    explanation = dspy.OutputField(desc="代码解释")

# 程序化思维预测器
pot = dspy.ProgramOfThought(CodeProblem)
result = pot(problem="写一个函数计算列表的平均值")

# 输出结果
print("解决方案代码:", result.code)
print("解释:", result.explanation)

# 自定义代码执行环境
class PythonInterpreter:
    def __init__(self):
        self.namespace = {}

    def execute(self, code):
        try:
            exec(code, self.namespace)
            return "执行成功"
        except Exception as e:
            return f"执行错误: {str(e)}"

# 带代码验证的PoT
class VerifiedPoT(dspy.ProgramOfThought):
    def __init__(self, signature, **kwargs):
        super().__init__(signature, **kwargs)
        self.interpreter = PythonInterpreter()

    def forward(self, **kwargs):
        result = super().forward(**kwargs)

        # 验证生成的代码
        if hasattr(result, 'code'):
            execution_result = self.interpreter.execute(result.code)
            result.execution_result = execution_result

        return result
```

#### 5. 高级预测模块
```python
# 多链比较
class ComparisonTask(dspy.Signature):
    question = dspy.InputField(desc="问题")
    answer = dspy.OutputField(desc="最终答案")
    reasoning = dspy.OutputField(desc="推理过程")

# 创建多个推理链
chains = [
    dspy.ChainOfThought(ComparisonTask),
    dspy.ReAct(ComparisonTask, tools=tools),
    dspy.ProgramOfThought(ComparisonTask)
]

# 多链比较
mcc = dspy.MultiChainComparison(
    chains=chains,
    compare_fn=lambda x, y: len(x.reasoning) > len(y.reasoning)
)

result = mcc(question="解释机器学习的基本原理")

# 最佳N选择
best_of_n = dspy.BestOfN(
    ComparisonTask,
    n=5,
    compare_fn=lambda x, y: x.confidence > y.confidence
)

# 迭代改进
refine = dspy.Refine(ComparisonTask, max_iterations=3)
result = refine(question="如何提高编程能力？")
```

### 性能优化技巧
```python
# 缓存优化
class CachedPredict(dspy.Predict):
    def __init__(self, signature, cache_size=1000, **kwargs):
        super().__init__(signature, **kwargs)
        self.cache = {}
        self.cache_size = cache_size

    def forward(self, **kwargs):
        # 生成缓存键
        cache_key = hash(frozenset(kwargs.items()))

        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 执行预测
        result = super().forward(**kwargs)

        # 更新缓存
        if len(self.cache) >= self.cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result
        return result

# 批处理优化
class BatchPredict(dspy.Module):
    def __init__(self, signature, batch_size=10):
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
        self.batch_size = batch_size

    def forward(self, **kwargs):
        # 批量处理逻辑
        results = []

        # 假设inputs是列表
        inputs = kwargs.get('inputs', [])

        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            batch_results = [self.predictor(input=item) for item in batch]
            results.extend(batch_results)

        return dspy.Prediction(results=results)
```

### 常见问题

**Q1: 如何选择合适的推理模式？**
A: 根据任务复杂度选择：
- 简单任务：使用Predict
- 需要推理：使用ChainOfThought
- 需要工具：使用ReAct
- 需要计算：使用ProgramOfThought

**Q2: 如何处理推理错误？**
A: 实现错误处理机制：
```python
class RobustPredict(dspy.Predict):
    def forward(self, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return super().forward(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                # 重试前调整参数
                self.temperature += 0.1
```

**Q3: 如何提高预测质量？**
A: 优化策略：
- 提供清晰的输入和上下文
- 使用合适的temperature和max_tokens
- 实现结果验证和过滤
- 使用多模型集成

### 实践任务
- [ ] 实现4种不同推理模式
- [ ] 对比分析各模式性能
- [ ] 开发带缓存的预测器
- [ ] 构建多模型集成系统

---

## ⚙️ 模块4：优化器 (Teleprompters)

### 学习目标
- 掌握各种优化算法的原理和使用
- 学会设计评估指标和优化目标
- 理解自动优化流程

### 核心概念
- **BootstrapFewShot**: 少样本自举优化
- **MIPROv2**: 多指令提示优化
- **COPRO**: 协作提示优化
- **BootstrapFinetune**: 模型微调优化

### API详解

#### 1. BootstrapFewShot优化
```python
import dspy

# 定义任务和评估指标
class QA(dspy.Signature):
    question = dspy.InputField(desc="问题")
    answer = dspy.OutputField(desc="答案")

def simple_metric(gold, pred):
    """简单评估指标"""
    return gold.answer.lower() == pred.answer.lower()

# 创建优化器
optimizer = dspy.BootstrapFewShot(
    metric=simple_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=4,
    max_rounds=2
)

# 准备训练数据
trainset = [
    Example(question="2+2等于几？", answer="4"),
    Example(question="北京是哪个国家的首都？", answer="中国"),
    # ... 更多训练数据
]

# 定义要优化的模块
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QA)

    def forward(self, question):
        return self.predict(question=question)

# 执行优化
qa_module = SimpleQA()
optimized_qa = optimizer.compile(qa_module, trainset=trainset)

# 测试优化结果
test_result = optimized_qa(question="3+3等于几？")
print(test_result.answer)
```

#### 2. MIPROv2优化
```python
# MIPROv2优化器
mipro_optimizer = dspy.MIPROv2(
    metric=simple_metric,
    num_candidates=10,
    init_temperature=1.0,
    compile_temperature=0.7
)

# 定义复杂模块
class ComplexQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(QA)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 执行MIPROv2优化
optimized_complex = mipro_optimizer.compile(
    ComplexQA(),
    trainset=trainset
)
```

#### 3. COPRO优化
```python
# COPRO协作优化器
copro_optimizer = dspy.COPRO(
    metric=simple_metric,
    breadth=10,
    depth=3,
    init_temperature=1.5
)

# 自定义模块用于COPRO
class AdaptiveQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = QA
        self.temperature = 0.7
        self.max_tokens = 200

    def forward(self, question):
        predictor = dspy.Predict(
            self.signature,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return predictor(question=question)

# 执行COPRO优化
optimized_adaptive = copro_optimizer.compile(
    AdaptiveQA(),
    trainset=trainset
)
```

#### 4. 自定义优化器
```python
class CustomOptimizer(dspy.teleprompter.Teleprompter):
    """自定义优化器实现"""

    def __init__(self, metric, search_space="default"):
        super().__init__()
        self.metric = metric
        self.search_space = search_space

    def compile(self, program, trainset):
        """编译和优化程序"""
        best_program = None
        best_score = float('-inf')

        # 搜索最优参数
        for params in self._search_params():
            # 编译程序
            compiled_program = self._compile_with_params(
                program, params
            )

            # 评估性能
            score = self._evaluate(compiled_program, trainset)

            # 更新最佳结果
            if score > best_score:
                best_score = score
                best_program = compiled_program

        return best_program

    def _search_params(self):
        """参数搜索空间"""
        if self.search_space == "temperature":
            return [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
        elif self.search_space == "examples":
            return [1, 3, 5, 8, 12, 16]
        else:
            return [{"temperature": t, "n": n}
                   for t in [0.1, 0.5, 0.9]
                   for n in [1, 3, 5]]

    def _compile_with_params(self, program, params):
        """使用参数编译程序"""
        # 复制程序
        compiled = program.copy()

        # 应用参数
        if isinstance(params, dict):
            for key, value in params.items():
                setattr(compiled, key, value)

        return compiled

    def _evaluate(self, program, trainset):
        """评估程序性能"""
        score = 0
        for example in trainset:
            try:
                prediction = program(**example.inputs())
                if self.metric(example, prediction):
                    score += 1
            except Exception:
                # 处理评估错误
                pass

        return score / len(trainset)

# 使用自定义优化器
custom_optimizer = CustomOptimizer(
    metric=simple_metric,
    search_space="temperature"
)

optimized_custom = custom_optimizer.compile(
    SimpleQA(),
    trainset=trainset
)
```

#### 5. 多阶段优化流程
```python
class MultiStageOptimizer:
    """多阶段优化流程"""

    def __init__(self):
        self.stage1 = dspy.BootstrapFewShot(
            metric=simple_metric,
            max_bootstrapped_demos=4
        )
        self.stage2 = dspy.MIPROv2(
            metric=simple_metric,
            num_candidates=5
        )
        self.stage3 = dspy.COPRO(
            metric=simple_metric,
            breadth=5,
            depth=2
        )

    def optimize(self, program, trainset):
        """执行多阶段优化"""
        print("阶段1: BootstrapFewShot优化...")
        stage1_result = self.stage1.compile(program, trainset)

        print("阶段2: MIPROv2优化...")
        stage2_result = self.stage2.compile(
            stage1_result,
            trainset
        )

        print("阶段3: COPRO优化...")
        stage3_result = self.stage3.compile(
            stage2_result,
            trainset
        )

        return stage3_result

# 使用多阶段优化
multi_optimizer = MultiStageOptimizer()
final_optimized = multi_optimizer.optimize(
    ComplexQA(),
    trainset=trainset
)
```

### 评估指标设计
```python
class AdvancedMetrics:
    """高级评估指标"""

    @staticmethod
    def exact_match(gold, pred):
        """精确匹配"""
        return gold.answer.lower().strip() == pred.answer.lower().strip()

    @staticmethod
    def fuzzy_match(gold, pred, threshold=0.8):
        """模糊匹配"""
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(
            None,
            gold.answer.lower(),
            pred.answer.lower()
        ).ratio()
        return similarity >= threshold

    @staticmethod
    semantic_similarity(gold, pred):
        """语义相似度"""
        # 这里可以集成sentence_transformers等
        # 简化实现
        return AdvancedMetrics.fuzzy_match(gold, pred, 0.7)

    @staticmethod
    def multi_metric(gold, pred):
        """多指标综合评估"""
        metrics = {
            'exact': AdvancedMetrics.exact_match(gold, pred),
            'fuzzy': AdvancedMetrics.fuzzy_match(gold, pred),
            'semantic': AdvancedMetrics.semantic_similarity(gold, pred)
        }

        # 加权平均
        weights = {'exact': 0.5, 'fuzzy': 0.3, 'semantic': 0.2}
        score = sum(metrics[k] * weights[k] for k in metrics)

        return score, metrics

# 使用高级指标
def advanced_metric(gold, pred):
    score, details = AdvancedMetrics.multi_metric(gold, pred)
    return score >= 0.6  # 返回是否达标
```

### 常见问题

**Q1: 如何选择合适的优化器？**
A: 根据数据规模和任务复杂度：
- 少量数据：BootstrapFewShot
- 中等数据：MIPROv2
- 大量数据：COPRO
- 需要微调：BootstrapFinetune

**Q2: 优化过程很慢怎么办？**
A: 优化策略：
- 减少训练数据或采样
- 降低搜索空间
- 使用并行处理
- 缓存中间结果

**Q3: 如何避免过拟合？**
A: 防止过拟合：
- 使用验证集监控
- 限制优化轮次
- 增加正则化
- 早停机制

### 实践任务
- [ ] 实现多种优化算法
- [ ] 设计自定义评估指标
- [ ] 构建多阶段优化流程
- [ ] 分析优化效果和成本

---

## 🔍 模块5：检索器 (Retrievers)

### 学习目标
- 掌握各种检索技术的原理
- 学会集成外部知识源
- 理解检索质量和优化

### 核心概念
- **Retrieve**: 通用检索接口
- **Embeddings**: 向量嵌入
- **ColBERTv2**: 稠密检索
- **WeaviateRM**: 向量数据库

### API详解

#### 1. 基础检索
```python
import dspy

# 配置检索器
retriever = dspy.Retrieve(k=5)  # 检索前5个最相关的文档

# 使用检索器
query = "什么是机器学习？"
results = retriever(query)

# 访问检索结果
for i, passage in enumerate(results.passages):
    print(f"文档{i+1}: {passage}")
    print(f"相似度: {results.scores[i]}")
    print("---")

# 带过滤的检索
filtered_results = retriever(
    query,
    filters={"category": "technology", "year": 2024}
)
```

#### 2. 向量嵌入检索
```python
from dspy.retrieve.embeddings import Embeddings

# 创建嵌入模型
embeddings = Embeddings(
    model="text-embedding-3-small",
    batch_size=100
)

# 嵌入文本
texts = ["机器学习是AI的分支", "深度学习使用神经网络"]
embeddings_vectors = embeddings.embed(texts)

# 相似度计算
def cosine_similarity(a, b):
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 查找最相似文本
def find_similar(query, corpus, embeddings_model):
    query_embedding = embeddings_model.embed([query])[0]
    corpus_embeddings = embeddings_model.embed(corpus)

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in corpus_embeddings
    ]

    # 返回最相似的文档
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return corpus[best_idx], similarities[best_idx]

# 使用示例
similar_text, similarity = find_similar(
    "AI和ML的关系",
    texts,
    embeddings
)
```

#### 3. ColBERTv2稠密检索
```python
# ColBERTv2检索器配置
colbert_retriever = dspy.ColBERTv2(
    model_path="colbert-ir/colbertv2.0",
    index_path="path/to/your/index"
)

# 构建索引（如果需要）
if not colbert_retriever.index_exists():
    documents = [
        "文档1的内容...",
        "文档2的内容...",
        # ... 更多文档
    ]
    colbert_retriever.build_index(documents)

# 执行检索
results = colbert_retriever("查询内容", k=10)

# 处理检索结果
retrieved_docs = []
for doc, score in zip(results.documents, results.scores):
    retrieved_docs.append({
        'content': doc,
        'score': score,
        'metadata': doc.metadata
    })
```

#### 4. Weaviate向量数据库
```python
# Weaviate配置
import weaviate

client = weaviate.Client("http://localhost:8080")

# Weaviate检索器
weaviate_rm = dspy.WeaviateRM(
    client=client,
    class_name="Document",
    text_field="content",
    vector_field="embedding"
)

# 混合检索（向量+关键词）
hybrid_results = weaviate_rm.hybrid_search(
    query="机器学习算法",
    alpha=0.7,  # 0=纯关键词，1=纯向量
    k=5
)

# 带过滤的检索
filtered_search = weaviate_rm.search(
    query="深度学习",
    filters=[
        {
            "path": ["category"],
            "operator": "Equal",
            "valueString": "技术文档"
        },
        {
            "path": ["publish_date"],
            "operator": "GreaterThan",
            "valueDate": "2024-01-01"
        }
    ],
    k=10
)
```

#### 5. 自定义检索器
```python
class CustomRetriever(dspy.Retrieve):
    """自定义检索器实现"""

    def __init__(self, knowledge_base, similarity_threshold=0.7):
        super().__init__()
        self.knowledge_base = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.embeddings = Embeddings()

        # 预计算嵌入
        self.doc_embeddings = self._precompute_embeddings()

    def _precompute_embeddings(self):
        """预计算文档嵌入"""
        return self.embeddings.embed(self.knowledge_base)

    def forward(self, query_or_queries, k=3):
        """检索相关文档"""
        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries

        all_results = []

        for query in queries:
            # 嵌入查询
            query_embedding = self.embeddings.embed([query])[0]

            # 计算相似度
            similarities = [
                cosine_similarity(query_embedding, doc_embedding)
                for doc_embedding in self.doc_embeddings
            ]

            # 筛选结果
            filtered_results = [
                (self.knowledge_base[i], sim)
                for i, sim in enumerate(similarities)
                if sim >= self.similarity_threshold
            ]

            # 排序并取top-k
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            top_results = filtered_results[:k]

            all_results.extend(top_results)

        return dspy.Prediction(
            passages=[r[0] for r in all_results],
            scores=[r[1] for r in all_results]
        )

# 使用自定义检索器
knowledge_base = [
    "DSPy是一个用于编程基础模型的框架...",
    "ChainOfThought是一种推理方法...",
    "ReAct结合了推理和行动..."
]

custom_retriever = CustomRetriever(knowledge_base)
results = custom_retriever("DSPy的推理方法")
```

#### 6. 多路检索和融合
```python
class MultiRetriever(dspy.Module):
    """多路检索和融合"""

    def __init__(self):
        super().__init__()
        self.dense_retriever = dspy.ColBERTv2()
        self.sparse_retriever = dspy.Retrieve()
        self.reranker = dspy.Predict(RerankPassages)

    def forward(self, query, k=10):
        # 稠密检索
        dense_results = self.dense_retriever(query, k=k*2)

        # 稀疏检索
        sparse_results = self.sparse_retriever(query, k=k*2)

        # 合并结果
        all_passages = list(set(
            dense_results.passages + sparse_results.passages
        ))

        # 重排序
        reranked = self.reranker(
            query=query,
            passages=all_passages
        )

        return reranked[:k]

# RerankPassages签名
class RerankPassages(dspy.Signature):
    query = dspy.InputField(desc="查询")
    passages = dspy.InputField(desc="文档列表", type=List[str])
    ranked_passages = dspy.OutputField(desc="排序后的文档", type=List[str])
    scores = dspy.OutputField(desc="相关性分数", type=List[float])
```

### 检索质量评估
```python
class RetrievalEvaluator:
    """检索质量评估"""

    def __init__(self, retriever):
        self.retriever = retriever

    def precision_at_k(self, query, relevant_docs, k=5):
        """Precision@K"""
        results = self.retriever(query, k=k)
        retrieved_docs = set(results.passages)
        relevant_docs_set = set(relevant_docs)

        intersection = retrieved_docs & relevant_docs_set
        return len(intersection) / k

    def recall_at_k(self, query, relevant_docs, k=5):
        """Recall@K"""
        results = self.retriever(query, k=k)
        retrieved_docs = set(results.passages)
        relevant_docs_set = set(relevant_docs)

        intersection = retrieved_docs & relevant_docs_set
        return len(intersection) / len(relevant_docs_set)

    def mean_reciprocal_rank(self, query, relevant_docs, k=10):
        """平均倒数排名"""
        results = self.retriever(query, k=k)

        for i, passage in enumerate(results.passages):
            if passage in relevant_docs:
                return 1 / (i + 1)

        return 0

    def evaluate_dataset(self, queries_relevant_pairs):
        """评估整个数据集"""
        metrics = {
            'precision@5': [],
            'precision@10': [],
            'recall@5': [],
            'recall@10': [],
            'mrr': []
        }

        for query, relevant_docs in queries_relevant_pairs:
            metrics['precision@5'].append(
                self.precision_at_k(query, relevant_docs, 5)
            )
            metrics['precision@10'].append(
                self.precision_at_k(query, relevant_docs, 10)
            )
            metrics['recall@5'].append(
                self.recall_at_k(query, relevant_docs, 5)
            )
            metrics['recall@10'].append(
                self.recall_at_k(query, relevant_docs, 10)
            )
            metrics['mrr'].append(
                self.mean_reciprocal_rank(query, relevant_docs)
            )

        # 计算平均值
        return {k: sum(v)/len(v) for k, v in metrics.items()}

# 使用评估器
evaluator = RetrievalEvaluator(custom_retriever)

# 测试数据
test_data = [
    ("DSPy的特点", ["文档1", "文档3"]),
    ("RAG系统", ["文档2", "文档4", "文档5"]),
    # ... 更多测试数据
]

evaluation_results = evaluator.evaluate_dataset(test_data)
print("检索评估结果:", evaluation_results)
```

### 常见问题

**Q1: 如何选择合适的检索器？**
A: 根据数据特点选择：
- 小规模数据：简单的向量检索
- 大规模数据：ColBERTv2或Weaviate
- 多模态数据：专用检索器
- 实时需求：缓存优化检索

**Q2: 如何提高检索质量？**
A: 优化策略：
- 改进文档预处理和分块
- 优化嵌入模型选择
- 使用重排序技术
- 融合多种检索策略

**Q3: 如何处理检索延迟？**
A: 性能优化：
- 预计算和缓存
- 批量处理
- 并行检索
- 索引优化

### 实践任务
- [ ] 实现多种检索器
- [ ] 构建混合检索系统
- [ ] 开发检索质量评估工具
- [ ] 优化检索性能和延迟

---

## 📞 模块6：客户端适配 (Clients)

### 学习目标
- 掌握多种LLM提供商的集成
- 学会统一客户端接口使用
- 理解缓存和错误处理

### 核心概念
- **LM**: 语言模型基础抽象
- **OpenAI**: OpenAI API集成
- **缓存系统**: 智能缓存机制
- **适配器**: 格式转换适配

### API详解

#### 1. 基础语言模型
```python
import dspy

# 配置OpenAI模型
lm = dspy.OpenAI(
    model="gpt-3.5-turbo",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000
)

# 设置全局模型
dspy.settings.configure(lm=lm)

# 使用模型
response = lm("你好，请介绍一下DSPy")
print(response)

# 带元数据的请求
response_with_metadata = lm(
    "解释机器学习",
    temperature=0.3,
    max_tokens=500,
    top_p=0.9
)
```

#### 2. 多模型提供商
```python
# OpenAI客户端
openai_client = dspy.OpenAI(
    model="gpt-4",
    api_key="openai-key"
)

# Anthropic客户端
anthropic_client = dspy.Anthropic(
    model="claude-3-sonnet-20240229",
    api_key="anthropic-key"
)

# 本地模型客户端
local_client = dspy.HFClientVLLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000"
)

# 模型路由器
class ModelRouter:
    def __init__(self):
        self.models = {
            'fast': dspy.OpenAI(model="gpt-3.5-turbo"),
            'quality': dspy.OpenAI(model="gpt-4"),
            'local': dspy.HFClientVLLM(model="llama2-7b")
        }

    def get_model(self, task_type, complexity):
        """根据任务类型和复杂度选择模型"""
        if task_type == "generation" and complexity == "high":
            return self.models['quality']
        elif task_type == "generation":
            return self.models['fast']
        else:
            return self.models['local']

# 使用路由器
router = ModelRouter()
model = router.get_model("generation", "high")
response = model("写一个关于AI的故事")
```

#### 3. 智能缓存系统
```python
from typing import Optional, Dict, Any
import hashlib
import json
import time

class SmartCache:
    """智能缓存系统"""

    def __init__(self, max_size=1000, ttl=3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）

    def _generate_key(self, prompt: str, **kwargs) -> str:
        """生成缓存键"""
        cache_data = {
            'prompt': prompt,
            'kwargs': kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """获取缓存结果"""
        key = self._generate_key(prompt, **kwargs)

        if key in self.cache:
            entry = self.cache[key]

            # 检查是否过期
            if time.time() - entry['timestamp'] < self.ttl:
                entry['access_count'] += 1
                return entry['response']
            else:
                # 删除过期条目
                del self.cache[key]

        return None

    def set(self, prompt: str, response: str, **kwargs):
        """设置缓存"""
        key = self._generate_key(prompt, **kwargs)

        # 检查缓存大小
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # 添加新条目
        self.cache[key] = {
            'response': response,
            'timestamp': time.time(),
            'access_count': 1
        }

    def _evict_lru(self):
        """删除最近最少使用的条目"""
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['access_count']
        )
        del self.cache[lru_key]

# 带缓存的模型客户端
class CachedLM:
    def __init__(self, base_lm, cache_size=500):
        self.base_lm = base_lm
        self.cache = SmartCache(max_size=cache_size)

    def __call__(self, prompt, **kwargs):
        # 检查缓存
        cached_response = self.cache.get(prompt, **kwargs)
        if cached_response:
            print("使用缓存结果")
            return cached_response

        # 调用基础模型
        response = self.base_lm(prompt, **kwargs)

        # 缓存结果
        self.cache.set(prompt, response, **kwargs)

        return response

# 使用缓存模型
base_lm = dspy.OpenAI(model="gpt-3.5-turbo")
cached_lm = CachedLM(base_lm)

response1 = cached_lm("什么是DSPy？")
response2 = cached_lm("什么是DSPy？")  # 从缓存获取
```

#### 4. 错误处理和重试
```python
import random
import time
from typing import Callable, Any

class RobustLM:
    """带错误处理的模型客户端"""

    def __init__(self, base_lm, max_retries=3, backoff_factor=2):
        self.base_lm = base_lm
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def __call__(self, prompt: str, **kwargs) -> str:
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return self.base_lm(prompt, **kwargs)

            except Exception as e:
                last_exception = e

                # 不同错误类型的处理策略
                if "rate limit" in str(e).lower():
                    # 速率限制：指数退避
                    wait_time = self.backoff_factor ** attempt
                    print(f"遇到速率限制，等待{wait_time}秒...")
                    time.sleep(wait_time)

                elif "connection" in str(e).lower():
                    # 连接错误：短暂等待后重试
                    wait_time = random.uniform(1, 3)
                    print(f"连接错误，{wait_time}秒后重试...")
                    time.sleep(wait_time)

                elif "quota" in str(e).lower():
                    # 配额用尽：不重试
                    raise e

                else:
                    # 其他错误：短暂等待
                    if attempt < self.max_retries - 1:
                        time.sleep(1)

        # 所有重试都失败了
        raise Exception(
            f"模型调用失败，已重试{self.max_retries}次。最后错误：{last_exception}"
        )

# 使用健壮的模型
robust_lm = RobustLM(dspy.OpenAI(model="gpt-3.5-turbo"))
try:
    response = robust_lm("测试消息")
except Exception as e:
    print(f"模型调用失败: {e}")
```

#### 5. 模型适配器
```python
class ModelAdapter:
    """模型适配器，统一不同模型的接口"""

    def __init__(self, model_config):
        self.model_config = model_config
        self.models = self._initialize_models()

    def _initialize_models(self):
        """初始化模型"""
        models = {}

        for name, config in self.model_config.items():
            if config['provider'] == 'openai':
                models[name] = dspy.OpenAI(
                    model=config['model'],
                    api_key=config['api_key']
                )
            elif config['provider'] == 'anthropic':
                models[name] = dspy.Anthropic(
                    model=config['model'],
                    api_key=config['api_key']
                )
            # 添加更多提供商...

        return models

    def generate(self, prompt: str, model_name: str = None, **kwargs):
        """生成文本"""
        if model_name is None:
            model_name = list(self.models.keys())[0]  # 使用默认模型

        if model_name not in self.models:
            raise ValueError(f"未找到模型: {model_name}")

        # 统一参数处理
        unified_kwargs = self._unify_parameters(kwargs, model_name)

        return self.models[model_name](prompt, **unified_kwargs)

    def _unify_parameters(self, kwargs, model_name):
        """统一不同模型的参数"""
        unified = kwargs.copy()

        # 参数映射
        if model_name == 'anthropic-claude':
            if 'max_tokens' in unified:
                unified['max_tokens'] = min(unified['max_tokens'], 4096)

        elif model_name == 'openai-gpt3':
            if 'temperature' not in unified:
                unified['temperature'] = 0.7

        return unified

# 配置多个模型
model_configs = {
    'gpt4': {
        'provider': 'openai',
        'model': 'gpt-4',
        'api_key': 'your-openai-key'
    },
    'claude': {
        'provider': 'anthropic',
        'model': 'claude-3-sonnet-20240229',
        'api_key': 'your-anthropic-key'
    }
}

adapter = ModelAdapter(model_configs)

# 统一接口调用
gpt4_response = adapter.generate(
    "写一个Python函数",
    model_name='gpt4',
    max_tokens=500,
    temperature=0.3
)

claude_response = adapter.generate(
    "写一个Python函数",
    model_name='claude',
    max_tokens=500
)
```

#### 6. 批处理和并行处理
```python
import concurrent.futures
from typing import List, Callable

class BatchProcessor:
    """批处理和并行处理"""

    def __init__(self, model, max_workers=4):
        self.model = model
        self.max_workers = max_workers

    def process_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """批量处理"""
        if len(prompts) == 1:
            # 单个提示直接处理
            return [self.model(prompts[0], **kwargs)]

        # 并行处理
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self.model, prompt, **kwargs)
                for prompt in prompts
            ]

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"处理失败: {e}")
                    results.append("")  # 或其他错误处理

        return results

    def process_with_retry(self, prompts: List[str], **kwargs) -> List[str]:
        """带重试的批处理"""
        results = []
        failed_indices = []

        # 第一次尝试
        for i, prompt in enumerate(prompts):
            try:
                result = self.model(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"提示{i}处理失败: {e}")
                results.append("")
                failed_indices.append(i)

        # 重试失败的提示
        if failed_indices:
            print(f"重试{len(failed_indices)}个失败的提示...")
            retry_results = self.process_batch(
                [prompts[i] for i in failed_indices],
                **kwargs
            )

            for i, result in zip(failed_indices, retry_results):
                results[i] = result

        return results

# 使用批处理器
processor = BatchProcessor(dspy.OpenAI(model="gpt-3.5-turbo"))

prompts = [
    "什么是人工智能？",
    "解释机器学习",
    "深度学习有什么应用？",
    "自然语言处理的发展历程"
]

# 批量处理
results = processor.process_batch(
    prompts,
    temperature=0.7,
    max_tokens=200
)

for prompt, result in zip(prompts, results):
    print(f"问题: {prompt}")
    print(f"答案: {result[:100]}...")
    print("---")
```

### 常见问题

**Q1: 如何处理不同模型的API限制？**
A: 策略包括：
- 实现速率限制检测
- 使用指数退避重试
- 配置多模型轮换
- 设置合理的超时时间

**Q2: 如何降低API调用成本？**
A: 成本优化：
- 使用智能缓存
- 优化prompt长度
- 选择合适的模型
- 批量处理请求

**Q3: 如何确保API调用的可靠性？**
A: 可靠性保证：
- 实现完善的错误处理
- 设置合理的重试机制
- 监控API状态和性能
- 准备备用模型和方案

### 实践任务
- [ ] 集成多种LLM提供商
- [ ] 实现智能缓存系统
- [ ] 开发错误处理机制
- [ ] 构建批处理工具

---

*（由于篇幅限制，剩余2个模块"评估模块"和"适配器模块"的详细内容将在下一个文档中继续）*