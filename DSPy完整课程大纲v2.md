# DSPy完整课程大纲 v2.0
## 编程式AI系统开发实战课程

---

## 📖 课程简介

### 课程概述
**课程名称：** DSPy - 声明式自改进AI系统开发实战
**课程版本：** 2.0
**技术版本：** DSPy 3.0.4
**更新日期：** 2024年11月

### 课程特色
- 🎯 **编程思维**：用编程而非提示的方式构建AI系统
- 🔧 **模块化设计**：学习构建可复用、可组合的AI组件
- ⚡ **自动优化**：掌握最先进的prompt和权重优化算法
- 🚀 **生产就绪**：从原型到生产部署的完整流程
- 💡 **实战导向**：每个模块配套真实项目案例

### 适用人群
- ✅ Python开发者（有1年以上开发经验）
- ✅ AI/ML工程师（希望提升AI系统开发能力）
- ✅ 产品经理（需要理解AI系统架构）
- ✅ 技术负责人（规划AI产品技术路线）
- ✅ 研究人员（探索前沿AI技术）

### 前置要求
**必备技能：**
- Python 3.10+ 编程基础
- 基本的命令行操作
- Git版本控制
- 基础的机器学习概念

**推荐技能：**
- 面向对象编程经验
- Web API开发经验
- 数据处理经验（pandas等）
- 了解Transformer和LLM基础

### 学习成果
完成课程后，您将能够：
1. ✅ 设计和实现复杂的AI系统架构
2. ✅ 构建生产级RAG（检索增强生成）系统
3. ✅ 开发功能完整的AI智能体
4. ✅ 使用自动优化算法提升系统性能
5. ✅ 部署和维护大规模AI应用
6. ✅ 贡献开源项目并参与社区

### 课程时长
- **总学习时长：** 12周（84天）
- **每周学习时间：** 10-15小时
- **总学时：** 120-180小时
- **理论与实践比例：** 3:7

---

## 🗺️ 课程结构

### 四大学习阶段

```
阶段一：基础入门（第1-3周）
    ├── 模块1：DSPy入门与环境搭建
    ├── 模块2：签名系统与数据流
    ├── 模块3：基础预测模块
    └── 模块4：简单任务实战

阶段二：进阶开发（第4-6周）
    ├── 模块5：Module类与组合设计
    ├── 模块6：检索增强生成(RAG)
    ├── 模块7：智能体开发
    └── 模块8：复杂推理系统

阶段三：优化部署（第7-9周）
    ├── 模块9：自动优化算法
    ├── 模块10：评估与调试
    ├── 模块11：生产部署最佳实践
    └── 模块12：性能监控与维护

阶段四：高级专题（第10-12周）
    ├── 模块13：自定义组件开发
    ├── 模块14：多模态与前沿应用
    ├── 模块15：毕业项目与答辩
    └── 职业发展与社区贡献
```

---

## 📚 详细课程内容

### 🌱 阶段一：基础入门（第1-3周）

#### 模块1：DSPy入门与环境搭建（第1周，10小时）

**学习目标**
- 理解DSPy的设计哲学和核心优势
- 掌握完整的开发环境搭建
- 了解DSPy生态系统和应用场景
- 运行第一个DSPy程序

**核心内容**

**1.1 DSPy简介（2小时）**
- DSPy是什么：Declarative Self-improving Python
- 为什么选择编程而非提示
  - 传统prompt engineering的局限性
  - 编程式开发的优势
  - 可维护性和可扩展性
- DSPy vs LangChain vs Semantic Kernel对比
- 成功案例展示
  - Stanford NLP的研究应用
  - 企业级生产案例
  - 开源社区项目

**1.2 核心概念概览（2小时）**
- **Signatures（签名）**：任务的输入输出规范
- **Modules（模块）**：可组合的AI组件
- **Teleprompters（优化器）**：自动优化算法
- **Predictors（预测器）**：推理执行器
- **Retrievers（检索器）**：知识获取组件
- 数据流和执行流程

**1.3 环境搭建实战（3小时）**
```bash
# 创建Python虚拟环境
python -m venv dspy_env
source dspy_env/bin/activate  # Windows: dspy_env\Scripts\activate

# 安装DSPy
pip install dspy

# 安装开发依赖
pip install jupyter notebook ipython
pip install python-dotenv  # 用于管理API密钥

# 验证安装
python -c "import dspy; print(dspy.__version__)"
```

**API密钥配置**
```python
# .env 文件配置
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
```

**开发工具配置**
- VSCode + Python扩展
- Jupyter Notebook配置
- Git仓库初始化

**1.4 第一个DSPy程序（3小时）**
```python
import dspy
from dspy import OpenAI

# 1. 配置语言模型
lm = OpenAI(model="gpt-3.5-turbo", max_tokens=250)
dspy.settings.configure(lm=lm)

# 2. 定义任务签名
class QuestionAnswering(dspy.Signature):
    """根据问题生成准确的答案"""
    question = dspy.InputField(desc="用户提出的问题")
    answer = dspy.OutputField(desc="详细的答案")

# 3. 创建预测器
qa_predictor = dspy.Predict(QuestionAnswering)

# 4. 执行预测
question = "什么是DSPy？它有什么优势？"
result = qa_predictor(question=question)

print(f"问题：{question}")
print(f"答案：{result.answer}")
```

**对比实验：传统Prompt vs DSPy**
```python
# 传统方式
prompt = f"请回答以下问题：{question}\n答案："
# 需要手动处理prompt格式、解析响应等

# DSPy方式
result = qa_predictor(question=question)
# 自动处理格式化、类型验证、错误处理
```

**实践任务**
- [ ] 完成开发环境搭建
- [ ] 配置至少2种LLM（OpenAI, Anthropic等）
- [ ] 运行并理解第一个程序
- [ ] 修改签名，添加新的输入/输出字段
- [ ] 对比3种不同模型的输出质量
- [ ] 撰写学习笔记（至少500字）

**评估标准**
- 环境搭建成功 ✓
- 程序运行无误 ✓
- 理解核心概念 ✓
- 完成对比实验 ✓

---

#### 模块2：签名系统与数据流（第1-2周，12小时）

**学习目标**
- 深入理解Signature系统的设计理念
- 掌握复杂任务规范的定义方法
- 学会数据流管理和验证
- 实现类型安全的AI程序

**核心内容**

**2.1 签名系统深度解析（4小时）**

**基础签名定义**
```python
import dspy
from pydantic import BaseModel, Field
from typing import List, Optional

# 简单签名
class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

# 带描述的签名
class DetailedQA(dspy.Signature):
    """专业问答系统签名"""
    context = dspy.InputField(desc="相关背景信息")
    question = dspy.InputField(desc="用户问题")
    answer = dspy.OutputField(desc="基于context的准确答案")
    confidence = dspy.OutputField(desc="答案置信度，0-1之间的浮点数")
```

**复杂签名设计**
```python
class MultiStepReasoning(dspy.Signature):
    """多步推理任务签名"""
    # 输入字段
    problem = dspy.InputField(desc="需要解决的问题")
    constraints = dspy.InputField(desc="约束条件列表")

    # 输出字段
    reasoning_steps = dspy.OutputField(desc="推理步骤的列表")
    final_answer = dspy.OutputField(desc="最终答案")
    alternative_solutions = dspy.OutputField(desc="备选方案")
    confidence_score = dspy.OutputField(desc="置信度评分")

class CodeGeneration(dspy.Signature):
    """代码生成签名"""
    requirements = dspy.InputField(desc="功能需求描述")
    language = dspy.InputField(desc="编程语言")
    style_guide = dspy.InputField(desc="代码风格指南", default="PEP8")

    code = dspy.OutputField(desc="生成的代码")
    explanation = dspy.OutputField(desc="代码说明")
    test_cases = dspy.OutputField(desc="测试用例")
```

**2.2 数据流管理（4小时）**

**Example类使用**
```python
from dspy import Example

# 创建训练样本
train_examples = [
    Example(
        question="Python中如何创建列表？",
        answer="使用方括号[]或list()函数",
        category="基础",
        difficulty="简单"
    ).with_inputs("question"),  # 标记输入字段

    Example(
        question="解释装饰器的工作原理",
        answer="装饰器是一个接受函数并返回新函数的高阶函数...",
        category="进阶",
        difficulty="中等"
    ).with_inputs("question")
]

# 数据集分割
from sklearn.model_selection import train_test_split
trainset, devset = train_test_split(
    train_examples,
    test_size=0.2,
    random_state=42
)
```

**数据验证和清洗**
```python
def validate_example(example: Example) -> bool:
    """验证示例数据的完整性"""
    required_fields = ["question", "answer"]

    # 检查必需字段
    for field in required_fields:
        if not hasattr(example, field) or not getattr(example, field):
            return False

    # 检查数据质量
    if len(example.question) < 5:
        return False
    if len(example.answer) < 10:
        return False

    return True

# 清洗数据集
clean_examples = [ex for ex in train_examples if validate_example(ex)]
```

**2.3 类型系统与Pydantic集成（4小时）**

**使用Pydantic定义复杂类型**
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class DifficultyLevel(str, Enum):
    EASY = "简单"
    MEDIUM = "中等"
    HARD = "困难"

class QuestionMetadata(BaseModel):
    difficulty: DifficultyLevel
    topics: List[str]
    estimated_time: int = Field(..., ge=1, le=120, description="预计解答时间(分钟)")

    @validator('topics')
    def validate_topics(cls, v):
        if len(v) == 0:
            raise ValueError("至少需要一个主题标签")
        return v

class StructuredQA(dspy.Signature):
    """结构化问答签名"""
    question = dspy.InputField()
    metadata = dspy.InputField()  # QuestionMetadata类型

    answer = dspy.OutputField()
    confidence = dspy.OutputField()
    sources = dspy.OutputField()  # List[str]类型
```

**实践任务**
- [ ] 设计5种不同复杂度的任务签名
- [ ] 实现完整的数据验证管道
- [ ] 使用Pydantic定义复杂数据类型
- [ ] 创建可复用的签名模板库
- [ ] 处理边界情况和错误

**评估标准**
- 签名设计合理性 ✓
- 数据流处理正确性 ✓
- 类型安全实现 ✓
- 错误处理完善性 ✓

---

#### 模块3：基础预测模块（第2周，12小时）

**学习目标**
- 掌握所有基础预测模块的使用
- 理解不同推理模式的适用场景
- 学会选择和组合预测模块
- 优化推理质量和效率

**核心内容**

**3.1 Predict - 基础预测（2小时）**
```python
import dspy

# 最简单的预测
class Sentiment(dspy.Signature):
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="正面、负面或中性")

predictor = dspy.Predict(Sentiment)
result = predictor(text="这个产品太棒了！")
print(result.sentiment)  # 输出：正面
```

**3.2 ChainOfThought - 思维链推理（3小时）**
```python
class MathProblem(dspy.Signature):
    """数学问题求解"""
    problem = dspy.InputField(desc="数学问题描述")
    answer = dspy.OutputField(desc="最终答案")

# 使用思维链
cot = dspy.ChainOfThought(MathProblem)
result = cot(problem="如果一个苹果3元，买5个需要多少钱？")

print(f"推理过程：{result.reasoning}")
print(f"答案：{result.answer}")

# 输出示例：
# 推理过程：首先，一个苹果3元。买5个苹果，需要计算3 × 5 = 15元
# 答案：15元
```

**ChainOfThought的高级用法**
```python
class ComplexReasoning(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

# 配置推理深度
cot_deep = dspy.ChainOfThought(
    ComplexReasoning,
    reasoning_depth=3  # 更深层次的推理
)

# 多步骤推理
result = cot_deep(
    context="量子计算利用量子叠加和纠缠原理...",
    question="量子计算相比经典计算的优势是什么？"
)
```

**3.3 ReAct - 推理+行动智能体（3小时）**
```python
from dspy import Tool

# 定义工具
def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    # 实现搜索逻辑
    return f"关于{query}的信息..."

def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 创建工具列表
tools = [
    Tool(func=search_wikipedia, name="搜索", desc="搜索维基百科获取信息"),
    Tool(func=calculate, name="计算器", desc="计算数学表达式")
]

# ReAct智能体
class ReactAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(tools=tools)

    def forward(self, task):
        return self.react(task=task)

# 使用智能体
agent = ReactAgent()
result = agent(task="查找北京的人口数量并计算其10%是多少")
```

**3.4 ProgramOfThought - 程序化思维（2小时）**
```python
class MathSolver(dspy.Signature):
    problem = dspy.InputField()
    solution = dspy.OutputField()

# 程序化思维会生成Python代码来解决问题
pot = dspy.ProgramOfThought(MathSolver)
result = pot(problem="计算斐波那契数列的第20项")

print(f"生成的代码：\n{result.code}")
print(f"执行结果：{result.solution}")
```

**3.5 高级预测模块（2小时）**

**BestOfN - 多次采样选最优**
```python
best_of_n = dspy.BestOfN(
    predictor=dspy.ChainOfThought(MathProblem),
    n=5,  # 生成5个候选答案
    metric=lambda x, y: x.answer == y.answer  # 选择标准
)
```

**MultiChainComparison - 多链比较**
```python
mcc = dspy.MultiChainComparison(
    signature=ComplexReasoning,
    num_chains=3  # 生成3条推理链并比较
)
```

**KNN - 基于相似样本的推理**
```python
knn = dspy.KNN(
    k=3,  # 使用3个最相似的样本
    trainset=train_examples
)
```

**Refine - 迭代改进**
```python
refine = dspy.Refine(
    signature=QuestionAnswering,
    max_iterations=3  # 最多迭代3次改进答案
)
```

**实践任务**
- [ ] 实现并对比所有预测模块
- [ ] 创建性能对比报告
- [ ] 设计模块组合方案
- [ ] 优化推理质量和速度
- [ ] 实现错误重试机制

**评估标准**
- 正确使用各种模块 ✓
- 理解适用场景 ✓
- 性能对比分析 ✓
- 组合设计合理 ✓

---

#### 模块4：简单任务实战（第3周，12小时）

**学习目标**
- 综合运用前三个模块的知识
- 完成3个实战项目
- 建立完整的开发流程
- 掌握评估和优化方法

**项目1：智能问答系统（4小时）**

**需求分析**
- 支持多轮对话
- 基于上下文的问答
- 答案质量评估
- 支持多种问题类型

**系统设计**
```python
import dspy
from typing import List, Optional

class ContextualQA(dspy.Module):
    """上下文感知的问答系统"""

    def __init__(self):
        super().__init__()

        # 定义签名
        class QASignature(dspy.Signature):
            """基于上下文的问答"""
            history = dspy.InputField(desc="对话历史")
            context = dspy.InputField(desc="相关文档")
            question = dspy.InputField(desc="当前问题")
            answer = dspy.OutputField(desc="准确答案")
            confidence = dspy.OutputField(desc="置信度")

        # 使用思维链推理
        self.qa_engine = dspy.ChainOfThought(QASignature)

        # 对话历史管理
        self.conversation_history = []

    def forward(self, context: str, question: str) -> dspy.Prediction:
        # 构建历史记录
        history_text = "\n".join([
            f"Q: {h['question']}\nA: {h['answer']}"
            for h in self.conversation_history[-3:]  # 保留最近3轮
        ])

        # 执行问答
        result = self.qa_engine(
            history=history_text,
            context=context,
            question=question
        )

        # 更新历史
        self.conversation_history.append({
            "question": question,
            "answer": result.answer
        })

        return result

# 使用示例
qa_system = ContextualQA()

context = """
DSPy是一个用于编程语言模型的框架。它允许开发者通过编写Python代码
而不是手工编写prompts来构建AI系统。DSPy提供了自动优化算法来改进系统性能。
"""

result1 = qa_system(context=context, question="DSPy是什么？")
print(f"答案1：{result1.answer}")

result2 = qa_system(context=context, question="它的主要优势是什么？")
print(f"答案2：{result2.answer}")
```

**项目2：文本分类器（4小时）**

**多类别分类实现**
```python
from typing import List
from enum import Enum

class Category(str, Enum):
    TECHNOLOGY = "科技"
    BUSINESS = "商业"
    SPORTS = "体育"
    ENTERTAINMENT = "娱乐"
    POLITICS = "政治"

class TextClassifier(dspy.Module):
    """多类别文本分类器"""

    def __init__(self, categories: List[str]):
        super().__init__()

        class ClassificationSignature(dspy.Signature):
            text = dspy.InputField(desc="待分类文本")
            category = dspy.OutputField(desc=f"类别，可选值：{', '.join(categories)}")
            confidence = dspy.OutputField(desc="分类置信度")
            keywords = dspy.OutputField(desc="关键词列表")

        self.classifier = dspy.ChainOfThought(ClassificationSignature)
        self.categories = categories

    def forward(self, text: str):
        result = self.classifier(text=text)
        return result

# 创建分类器
categories = [c.value for c in Category]
classifier = TextClassifier(categories=categories)

# 测试
text = "苹果公司发布了最新的iPhone，搭载革命性的A17芯片"
result = classifier(text=text)
print(f"类别：{result.category}")
print(f"置信度：{result.confidence}")
print(f"关键词：{result.keywords}")
```

**情感分析扩展**
```python
class SentimentAnalyzer(dspy.Module):
    """情感分析器"""

    def __init__(self):
        super().__init__()

        class SentimentSignature(dspy.Signature):
            text = dspy.InputField()
            sentiment = dspy.OutputField(desc="正面/负面/中性")
            intensity = dspy.OutputField(desc="情感强度0-1")
            aspects = dspy.OutputField(desc="具体方面的情感")

        self.analyzer = dspy.ChainOfThought(SentimentSignature)

    def forward(self, text: str):
        return self.analyzer(text=text)
```

**项目3：内容生成器（4小时）**

**多场景内容生成**
```python
class ContentGenerator(dspy.Module):
    """智能内容生成器"""

    def __init__(self):
        super().__init__()

        # 不同类型的生成器
        self.generators = {
            "creative_writing": self._setup_creative_writer(),
            "technical_doc": self._setup_tech_writer(),
            "summary": self._setup_summarizer(),
        }

    def _setup_creative_writer(self):
        class CreativeWriting(dspy.Signature):
            topic = dspy.InputField()
            style = dspy.InputField(desc="写作风格")
            length = dspy.InputField(desc="目标字数")
            content = dspy.OutputField(desc="创作内容")

        return dspy.ChainOfThought(CreativeWriting)

    def _setup_tech_writer(self):
        class TechnicalDoc(dspy.Signature):
            feature = dspy.InputField(desc="功能描述")
            audience = dspy.InputField(desc="目标读者")
            documentation = dspy.OutputField(desc="技术文档")
            examples = dspy.OutputField(desc="代码示例")

        return dspy.ChainOfThought(TechnicalDoc)

    def _setup_summarizer(self):
        class Summarize(dspy.Signature):
            text = dspy.InputField()
            max_length = dspy.InputField()
            summary = dspy.OutputField()
            key_points = dspy.OutputField()

        return dspy.Predict(Summarize)

    def forward(self, content_type: str, **kwargs):
        generator = self.generators.get(content_type)
        if not generator:
            raise ValueError(f"不支持的内容类型：{content_type}")

        return generator(**kwargs)

# 使用示例
generator = ContentGenerator()

# 创意写作
creative = generator(
    content_type="creative_writing",
    topic="未来城市",
    style="科幻",
    length="500字"
)

# 技术文档
tech_doc = generator(
    content_type="technical_doc",
    feature="用户认证系统",
    audience="后端开发者"
)

# 文本摘要
summary = generator(
    content_type="summary",
    text="长文本...",
    max_length="200字"
)
```

**实践任务**
- [ ] 完成3个项目的开发
- [ ] 编写完整的测试用例
- [ ] 建立性能评估体系
- [ ] 撰写项目文档
- [ ] 进行代码审查和优化

**评估标准**
- 功能完整性（40分）
- 代码质量（30分）
- 文档完善性（20分）
- 创新性（10分）

---

### 🚀 阶段二：进阶开发（第4-6周）

#### 模块5：Module类与组合设计（第4周，15小时）

**学习目标**
- 深入理解Module基类的设计理念
- 掌握复杂模块的开发方法
- 学会系统架构设计
- 实现高度可复用的组件

**核心内容**

**5.1 Module基类深度解析（5小时）**

**Module生命周期**
```python
import dspy
from typing import Any, Dict, List

class CustomModule(dspy.Module):
    """自定义模块完整示例"""

    def __init__(self, config: Dict[str, Any]):
        """初始化阶段"""
        super().__init__()

        # 1. 配置管理
        self.config = config

        # 2. 子模块初始化
        self._setup_submodules()

        # 3. 状态初始化
        self._setup_state()

        # 4. 缓存初始化
        self._setup_cache()

    def _setup_submodules(self):
        """设置子模块"""
        # 预测模块
        self.predictor = dspy.ChainOfThought(MySignature)

        # 检索模块（如果需要）
        if self.config.get('use_retrieval'):
            self.retriever = dspy.Retrieve(k=5)

        # 其他子模块
        self.validator = ValidationModule()

    def _setup_state(self):
        """初始化状态"""
        self.call_count = 0
        self.error_count = 0
        self.cache_hits = 0

    def _setup_cache(self):
        """设置缓存"""
        from functools import lru_cache
        self.cache = {}

    def forward(self, **kwargs) -> dspy.Prediction:
        """前向传播 - 主要逻辑"""
        # 1. 输入验证
        validated_input = self._validate_input(**kwargs)

        # 2. 缓存检查
        cache_key = self._get_cache_key(**validated_input)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        # 3. 核心处理
        try:
            result = self._process(**validated_input)
            self.call_count += 1
        except Exception as e:
            self.error_count += 1
            result = self._handle_error(e, **validated_input)

        # 4. 结果验证和缓存
        validated_result = self._validate_output(result)
        self.cache[cache_key] = validated_result

        return validated_result

    def _validate_input(self, **kwargs):
        """输入验证"""
        # 实现输入验证逻辑
        return kwargs

    def _process(self, **kwargs):
        """核心处理逻辑"""
        # 调用子模块
        result = self.predictor(**kwargs)
        return result

    def _validate_output(self, result):
        """输出验证"""
        # 实现输出验证逻辑
        return result

    def _handle_error(self, error, **kwargs):
        """错误处理"""
        # 实现错误恢复逻辑
        return dspy.Prediction(error=str(error))

    def _get_cache_key(self, **kwargs):
        """生成缓存键"""
        import hashlib
        import json
        key_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def reset_stats(self):
        """重置统计信息"""
        self.call_count = 0
        self.error_count = 0
        self.cache_hits = 0

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache)
        }
```

**5.2 高级组合模式（5小时）**

**串联组合（Pipeline）**
```python
class Pipeline(dspy.Module):
    """模块串联组合"""

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, **kwargs):
        result = kwargs
        for module in self.modules:
            result = module(**result)
        return result

# 使用示例
pipeline = Pipeline(
    QueryExpander(),      # 步骤1：查询扩展
    DocumentRetriever(),  # 步骤2：文档检索
    AnswerGenerator()     # 步骤3：答案生成
)
```

**并联组合（Ensemble）**
```python
class Ensemble(dspy.Module):
    """模块并联组合"""

    def __init__(self, *modules, aggregation="vote"):
        super().__init__()
        self.modules = modules
        self.aggregation = aggregation

    def forward(self, **kwargs):
        # 并行执行所有模块
        results = [module(**kwargs) for module in self.modules]

        # 聚合结果
        if self.aggregation == "vote":
            return self._majority_vote(results)
        elif self.aggregation == "average":
            return self._average(results)
        else:
            return results

# 使用示例
ensemble = Ensemble(
    dspy.ChainOfThought(QA),
    dspy.ReAct(QA),
    dspy.ProgramOfThought(QA),
    aggregation="vote"
)
```

**条件组合（Router）**
```python
class Router(dspy.Module):
    """智能路由模块"""

    def __init__(self, routes: Dict[str, dspy.Module]):
        super().__init__()

        # 路由判断器
        class RouteDecision(dspy.Signature):
            query = dspy.InputField()
            available_routes = dspy.InputField()
            best_route = dspy.OutputField()
            confidence = dspy.OutputField()

        self.router = dspy.Predict(RouteDecision)
        self.routes = routes

    def forward(self, query: str, **kwargs):
        # 决定路由
        route_names = list(self.routes.keys())
        decision = self.router(
            query=query,
            available_routes=", ".join(route_names)
        )

        # 执行对应模块
        selected_module = self.routes[decision.best_route]
        return selected_module(query=query, **kwargs)

# 使用示例
router = Router({
    "factual": FactualQA(),
    "creative": CreativeQA(),
    "technical": TechnicalQA()
})
```

**递归组合（Iterative）**
```python
class IterativeRefiner(dspy.Module):
    """迭代改进模块"""

    def __init__(self, base_module, max_iterations=3):
        super().__init__()
        self.base_module = base_module
        self.max_iterations = max_iterations

        # 质量评估器
        class QualityCheck(dspy.Signature):
            output = dspy.InputField()
            is_satisfactory = dspy.OutputField(desc="是或否")
            improvement_suggestions = dspy.OutputField()

        self.evaluator = dspy.Predict(QualityCheck)

    def forward(self, **kwargs):
        result = self.base_module(**kwargs)

        for iteration in range(self.max_iterations):
            # 评估质量
            quality = self.evaluator(output=str(result))

            if quality.is_satisfactory.lower() == "是":
                break

            # 改进
            kwargs['previous_output'] = str(result)
            kwargs['suggestions'] = quality.improvement_suggestions
            result = self.base_module(**kwargs)

        return result
```

**5.3 生产级RAG模块实现（5小时）**

完整的生产级RAG系统设计将在模块6详细讲解。

**实践任务**
- [ ] 实现完整的Module生命周期管理
- [ ] 开发4种组合模式
- [ ] 创建可配置的模块工厂
- [ ] 实现模块性能监控
- [ ] 编写模块单元测试

**评估标准**
- 模块设计合理性 ✓
- 代码质量和可维护性 ✓
- 性能和效率 ✓
- 文档完整性 ✓

---

#### 模块6：检索增强生成(RAG)（第4-5周，15小时）

**学习目标**
- 掌握RAG系统的完整架构
- 理解各种检索策略
- 实现高性能RAG系统
- 优化检索和生成质量

**核心内容**

**6.1 RAG系统架构（3小时）**

**基础RAG实现**
```python
import dspy
from typing import List

class BasicRAG(dspy.Module):
    """基础RAG系统"""

    def __init__(self, k=3):
        super().__init__()

        # 检索器
        self.retrieve = dspy.Retrieve(k=k)

        # 生成器签名
        class GenerateAnswer(dspy.Signature):
            context = dspy.InputField(desc="检索到的相关文档")
            question = dspy.InputField(desc="用户问题")
            answer = dspy.OutputField(desc="基于context的准确答案")

        # 使用思维链生成答案
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        # 检索相关文档
        context = self.retrieve(question).passages

        # 生成答案
        prediction = self.generate(context=context, question=question)

        return dspy.Prediction(
            context=context,
            answer=prediction.answer,
            reasoning=prediction.reasoning
        )

# 配置检索器
import dspy
from dspy import ColBERTv2

# 使用ColBERT检索器
colbert = ColBERTv2(url='http://localhost:8080/api/search')
dspy.settings.configure(rm=colbert)

# 使用RAG
rag = BasicRAG(k=5)
result = rag(question="DSPy的主要优势是什么？")
print(f"答案：{result.answer}")
print(f"来源：{result.context}")
```

**6.2 高级检索策略（6小时）**

**多跳检索（Multi-hop Retrieval）**
```python
class MultiHopRAG(dspy.Module):
    """多跳检索RAG系统"""

    def __init__(self, max_hops=2):
        super().__init__()
        self.max_hops = max_hops

        # 检索器
        self.retrieve = dspy.Retrieve(k=3)

        # 查询生成器
        class GenerateSearchQuery(dspy.Signature):
            context = dspy.InputField()
            question = dspy.InputField()
            next_query = dspy.OutputField(desc="下一个搜索查询")

        self.query_generator = dspy.ChainOfThought(GenerateSearchQuery)

        # 答案生成器
        class GenerateAnswer(dspy.Signature):
            contexts = dspy.InputField(desc="所有检索到的文档")
            question = dspy.InputField()
            answer = dspy.OutputField()

        self.answer_generator = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        all_contexts = []
        current_query = question

        # 多跳检索
        for hop in range(self.max_hops):
            # 检索当前查询
            passages = self.retrieve(current_query).passages
            all_contexts.extend(passages)

            # 生成下一个查询（除了最后一跳）
            if hop < self.max_hops - 1:
                next_q = self.query_generator(
                    context="\n".join(passages),
                    question=question
                )
                current_query = next_q.next_query

        # 基于所有上下文生成答案
        answer = self.answer_generator(
            contexts="\n\n".join(all_contexts),
            question=question
        )

        return dspy.Prediction(
            contexts=all_contexts,
            answer=answer.answer,
            hops=self.max_hops
        )
```

**混合检索（Hybrid Retrieval）**
```python
class HybridRAG(dspy.Module):
    """混合检索策略RAG"""

    def __init__(self):
        super().__init__()

        # 稠密检索（语义搜索）
        self.dense_retriever = dspy.Retrieve(k=10, mode='dense')

        # 稀疏检索（关键词搜索）
        self.sparse_retriever = dspy.Retrieve(k=10, mode='sparse')

        # 重排序器
        class RerankPassages(dspy.Signature):
            query = dspy.InputField()
            passages = dspy.InputField()
            top_k = dspy.InputField()
            ranked_passages = dspy.OutputField(desc="重排序后的文档列表")

        self.reranker = dspy.Predict(RerankPassages)

        # 生成器
        self.generator = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str, top_k=5):
        # 并行检索
        dense_results = self.dense_retriever(question).passages
        sparse_results = self.sparse_retriever(question).passages

        # 合并结果（去重）
        all_passages = list(set(dense_results + sparse_results))

        # 重排序
        reranked = self.reranker(
            query=question,
            passages="\n\n".join(all_passages),
            top_k=str(top_k)
        )

        # 生成答案
        answer = self.generator(
            context=reranked.ranked_passages,
            question=question
        )

        return answer
```

**查询路由（Query Routing）**
```python
class RoutedRAG(dspy.Module):
    """带查询路由的RAG系统"""

    def __init__(self):
        super().__init__()

        # 查询分类器
        class ClassifyQuery(dspy.Signature):
            query = dspy.InputField()
            category = dspy.OutputField(desc="factual/analytical/creative")
            complexity = dspy.OutputField(desc="simple/medium/complex")

        self.classifier = dspy.Predict(ClassifyQuery)

        # 不同类型的RAG模块
        self.rag_modules = {
            "factual_simple": BasicRAG(k=3),
            "factual_complex": MultiHopRAG(max_hops=2),
            "analytical": AnalyticalRAG(),
            "creative": CreativeRAG()
        }

    def forward(self, query: str):
        # 分类查询
        classification = self.classifier(query=query)

        # 选择合适的RAG模块
        module_key = f"{classification.category}_{classification.complexity}"
        selected_rag = self.rag_modules.get(
            module_key,
            self.rag_modules["factual_simple"]  # 默认
        )

        # 执行检索和生成
        return selected_rag(query=query)
```

**6.3 RAG系统优化（6小时）**

**检索质量优化**
```python
class OptimizedRAG(dspy.Module):
    """优化的RAG系统"""

    def __init__(self):
        super().__init__()

        # 查询增强
        class EnhanceQuery(dspy.Signature):
            original_query = dspy.InputField()
            enhanced_query = dspy.OutputField()
            keywords = dspy.OutputField()

        self.query_enhancer = dspy.ChainOfThought(EnhanceQuery)

        # 检索
        self.retriever = dspy.Retrieve(k=10)

        # 文档过滤
        class FilterRelevance(dspy.Signature):
            query = dspy.InputField()
            document = dspy.InputField()
            is_relevant = dspy.OutputField(desc="yes/no")
            relevance_score = dspy.OutputField(desc="0-1")

        self.filter = dspy.Predict(FilterRelevance)

        # 上下文压缩
        class CompressContext(dspy.Signature):
            documents = dspy.InputField()
            query = dspy.InputField()
            compressed = dspy.OutputField(desc="最相关的信息摘要")

        self.compressor = dspy.ChainOfThought(CompressContext)

        # 生成
        self.generator = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        # 1. 查询增强
        enhanced = self.query_enhancer(original_query=question)

        # 2. 检索文档
        passages = self.retriever(enhanced.enhanced_query).passages

        # 3. 过滤相关文档
        relevant_docs = []
        for doc in passages:
            relevance = self.filter(query=question, document=doc)
            if relevance.is_relevant.lower() == "yes":
                relevant_docs.append((doc, float(relevance.relevance_score)))

        # 按相关性排序
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in relevant_docs[:5]]

        # 4. 上下文压缩
        compressed = self.compressor(
            documents="\n\n".join(top_docs),
            query=question
        )

        # 5. 生成答案
        answer = self.generator(
            context=compressed.compressed,
            question=question
        )

        return answer
```

**性能优化**
```python
class HighPerformanceRAG(dspy.Module):
    """高性能RAG系统"""

    def __init__(self):
        super().__init__()

        # 缓存
        from functools import lru_cache
        self.cache = {}

        # 异步检索
        import asyncio
        self.async_enabled = True

        # 批处理
        self.batch_size = 32

        # 模块
        self.retriever = dspy.Retrieve(k=5)
        self.generator = dspy.ChainOfThought(GenerateAnswer)

    @dspy.asyncify  # DSPy的异步装饰器
    async def forward_async(self, question: str):
        # 异步检索
        passages = await self.retriever(question)

        # 异步生成
        answer = await self.generator(
            context=passages.passages,
            question=question
        )

        return answer

    def forward(self, question: str):
        # 检查缓存
        cache_key = hash(question)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 执行
        if self.async_enabled:
            import asyncio
            result = asyncio.run(self.forward_async(question))
        else:
            passages = self.retriever(question).passages
            result = self.generator(context=passages, question=question)

        # 缓存结果
        self.cache[cache_key] = result
        return result
```

**实践任务**
- [ ] 实现基础RAG系统
- [ ] 开发多跳检索功能
- [ ] 实现混合检索策略
- [ ] 添加查询路由机制
- [ ] 优化检索和生成质量
- [ ] 性能基准测试

**评估标准**
- 检索准确率 ✓
- 答案质量 ✓
- 系统性能 ✓
- 代码质量 ✓

---

*[由于内容过长，后续模块7-15的详细内容将继续...]*

## 📊 评估体系

### 学习评估方式
- **平时作业：** 35%（每周练习题）
- **阶段项目：** 40%（4个阶段项目）
- **期末项目：** 25%（毕业设计）

### 认证证书
- **基础证书：** 完成阶段一和二
- **进阶证书：** 完成阶段一、二、三
- **专家证书：** 完成全部四个阶段
- **优秀证书：** 期末项目评级A+

---

## 🎓 学习资源

### 官方资源
- [DSPy官方文档](https://dspy.ai/)
- [GitHub仓库](https://github.com/stanfordnlp/dspy)
- [Discord社区](https://discord.gg/dspy)

### 扩展阅读
- DSPy论文系列
- LLM应用开发最佳实践
- Prompt Engineering指南

---

**课程版权：** 本课程基于开源DSPy框架，遵循MIT许可证
**联系方式：** 课程社区 + 在线答疑
