# DSPy练习题库

## 📚 题库概览

本题库包含15个模块的分层练习题，按难度和类型分类：
- **理论题**：概念理解和原理分析
- **编程题**：代码实现和API使用
- **项目题**：综合应用和实战项目

**题目难度等级：**
- ⭐ 初级：基础概念和简单实现
- ⭐⭐ 中级：综合运用和复杂逻辑
- ⭐⭐⭐ 高级：系统设计和优化创新

---

## 🏗️ 模块1：基础原语 (Primitives)

### 理论题

#### ⭐ 初级理论题
**T1.1** DSPy的核心理念是什么？与传统的prompt engineering有什么区别？

> **参考答案**：DSPy的核心理念是"编程而非提示"(Programming over Prompting)。区别：
> - 传统prompt engineering通过精心设计文本提示来引导模型
> - DSPy通过编程方式组合模块化的AI组件
> - DSPy提供自动优化算法来提升系统性能
> - DSPy强调声明式和可组合的系统架构

**T1.2** 解释DSPy中Module类的作用，为什么所有组件都要继承Module？

> **参考答案**：Module类的作用：
> - 提供统一的接口规范
> - 支持参数跟踪和管理
> - 实现编译和优化功能
> - 提供保存/加载机制
> - 支持梯度计算和反向传播

**T1.3** Example类和普通的Python字典有什么优势？

> **参考答案**：优势包括：
> - 类型安全的字段访问
> - 内置的数据验证机制
> - 支持元数据管理
> - 与DSPy生态系统无缝集成
> - 提供便捷的数据操作方法

#### ⭐⭐ 中级理论题
**T1.4** 分析Prediction类和Completions类的设计模式，说明为什么需要这样的封装？

> **参考答案**：设计模式分析：
> - Prediction类：封装单次预测结果，包含置信度和元数据
> - Completions类：管理多个预测结果，支持排序和选择
> - 这样的封装实现了：
>   * 结果的标准化表示
>   * 便于结果比较和评估
>   * 支持不确定性量化的结果
>   * 为后续处理提供统一接口

**T1.5** 在什么情况下需要使用PythonInterpreter？如何确保安全性？

> **参考答案**：使用场景：
> - ProgramOfThought模块需要执行生成的代码
> - 数学计算和数据处理任务
> - 需要动态代码执行的场景
>
> 安全措施：
> - 使用沙箱环境隔离执行
> - 限制可用的库和函数
> - 设置执行时间限制
> - 验证生成的代码安全性
> - 记录和审计执行日志

#### ⭐⭐⭐ 高级理论题
**T1.6** 设计一个DSPy组件的生命周期管理系统，考虑内存管理、缓存策略和性能优化。

> **参考答案**：生命周期管理系统设计：
>
> **初始化阶段**：
> - 组件注册和依赖注入
> - 参数初始化和验证
> - 资源分配（GPU、内存）
>
> **运行阶段**：
> - 智能缓存策略（LRU、TTL）
> - 内存池管理和对象复用
> - 异步执行和批处理优化
> - 性能监控和自适应调整
>
> **清理阶段**：
> - 资源释放和内存回收
> - 状态持久化和恢复
> - 清理临时文件和缓存
>
> **性能优化**：
> - 延迟初始化和懒加载
> - 预计算和结果缓存
> - 并行处理和负载均衡

### 编程题

#### ⭐ 初级编程题
**P1.1** 实现一个简单的自定义Module类，用于文本长度统计：

```python
import dspy
from typing import Dict, Any

class TextLengthAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # 在这里初始化必要的组件

    def forward(self, text: str) -> dspy.Prediction:
        # 实现文本分析逻辑
        # 返回包含字符数、词数、句子数的Prediction
        pass

# 测试代码
analyzer = TextLengthAnalyzer()
result = analyzer("Hello DSPy! This is a test sentence.")
print(result.char_count)   # 应该输出字符数
print(result.word_count)   # 应该输出词数
print(result.sentence_count)  # 应该输出句子数
```

> **参考答案**：
```python
class TextLengthAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text: str) -> dspy.Prediction:
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') + text.split('!') + text.split('?') if s.strip()])

        return dspy.Prediction(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            original_text=text
        )
```

**P1.2** 创建一个Example数据集处理器，实现数据验证和清洗：

```python
from dspy import Example
from typing import List, Dict

class DatasetProcessor:
    def __init__(self):
        self.processed_count = 0
        self.errors = []

    def validate_example(self, example: Example) -> bool:
        # 验证Example是否包含必要字段
        # 检查数据类型和格式
        pass

    def clean_example(self, example: Example) -> Example:
        # 清洗数据：去除多余空格、标准化格式等
        pass

    def process_dataset(self, raw_examples: List[Dict]) -> List[Example]:
        # 批量处理数据集
        pass

# 使用示例
raw_data = [
    {"question": "  What is AI?  ", "answer": "Artificial Intelligence"},
    {"question": "How does ML work?", "answer": 123},  # 错误数据类型
    {"question": "", "answer": "Deep learning is..."}  # 空问题
]

processor = DatasetProcessor()
clean_dataset = processor.process_dataset(raw_data)
print(f"处理了{processor.processed_count}条数据，发现{len(processor.errors)}个错误")
```

> **参考答案**：
```python
class DatasetProcessor:
    def __init__(self):
        self.processed_count = 0
        self.errors = []

    def validate_example(self, example: Example) -> bool:
        if not hasattr(example, 'question') or not hasattr(example, 'answer'):
            return False

        if not isinstance(example.question, str) or not isinstance(example.answer, str):
            return False

        if len(example.question.strip()) == 0 or len(example.answer.strip()) == 0:
            return False

        return True

    def clean_example(self, example: Example) -> Example:
        cleaned = example.copy()
        cleaned.question = cleaned.question.strip()
        cleaned.answer = cleaned.answer.strip()

        # 标准化标点符号
        cleaned.question = cleaned.question.replace("  ", " ")
        cleaned.answer = cleaned.answer.replace("  ", " ")

        return cleaned

    def process_dataset(self, raw_examples: List[Dict]) -> List[Example]:
        processed_examples = []

        for i, raw_data in enumerate(raw_examples):
            try:
                example = Example(**raw_data)

                if self.validate_example(example):
                    cleaned_example = self.clean_example(example)
                    processed_examples.append(cleaned_example)
                    self.processed_count += 1
                else:
                    self.errors.append(f"第{i}条数据验证失败: {raw_data}")

            except Exception as e:
                self.errors.append(f"第{i}条数据处理异常: {str(e)}")

        return processed_examples
```

#### ⭐⭐ 中级编程题
**P1.3** 实现一个带缓存的Prediction管理器，支持相似结果的合并：

```python
import hashlib
from typing import Dict, List, Optional
import time

class CachedPredictionManager:
    def __init__(self, max_cache_size: int = 1000, similarity_threshold: float = 0.8):
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, dspy.Prediction] = {}
        self.access_times: Dict[str, float] = {}

    def _generate_cache_key(self, prediction: dspy.Prediction) -> str:
        # 基于Prediction内容生成缓存键
        pass

    def _calculate_similarity(self, pred1: dspy.Prediction, pred2: dspy.Prediction) -> float:
        # 计算两个Prediction的相似度
        pass

    def get(self, prediction: dspy.Prediction) -> Optional[dspy.Prediction]:
        # 获取缓存的相似预测
        pass

    def put(self, prediction: dspy.Prediction):
        # 存储预测到缓存
        pass

    def clear_expired(self, ttl_seconds: int = 3600):
        # 清理过期缓存
        pass

# 测试代码
manager = CachedPredictionManager()

# 创建一些测试预测
pred1 = dspy.Prediction(answer="这是一个答案", confidence=0.9)
pred2 = dspy.Prediction(answer="这是另一个答案", confidence=0.8)
pred3 = dspy.Prediction(answer="这是一个答案", confidence=0.85)  # 相似于pred1

manager.put(pred1)
manager.put(pred2)

# 测试缓存查找
similar_pred = manager.get(pred3)
print(similar_pred.answer if similar_pred else "未找到相似预测")
```

> **参考答案**：
```python
class CachedPredictionManager:
    def __init__(self, max_cache_size: int = 1000, similarity_threshold: float = 0.8):
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, dspy.Prediction] = {}
        self.access_times: Dict[str, float] = {}

    def _generate_cache_key(self, prediction: dspy.Prediction) -> str:
        # 基于Prediction主要字段生成键
        key_data = {
            'answer': getattr(prediction, 'answer', ''),
            'main_fields': []
        }

        # 获取所有字符串字段
        for attr in dir(prediction):
            if not attr.startswith('_') and isinstance(getattr(prediction, attr), str):
                key_data['main_fields'].append(f"{attr}:{getattr(prediction, attr)}")

        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _calculate_similarity(self, pred1: dspy.Prediction, pred2: dspy.Prediction) -> float:
        # 简单的文本相似度计算
        answer1 = getattr(pred1, 'answer', '').lower()
        answer2 = getattr(pred2, 'answer', '').lower()

        if not answer1 or not answer2:
            return 0.0

        # 计算Jaccard相似度
        set1 = set(answer1.split())
        set2 = set(answer2.split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def get(self, prediction: dspy.Prediction) -> Optional[dspy.Prediction]:
        # 检查直接匹配
        cache_key = self._generate_cache_key(prediction)
        if cache_key in self.cache:
            self.access_times[cache_key] = time.time()
            return self.cache[cache_key]

        # 检查相似预测
        for cached_key, cached_pred in self.cache.items():
            if self._calculate_similarity(prediction, cached_pred) >= self.similarity_threshold:
                self.access_times[cached_key] = time.time()
                return cached_pred

        return None

    def put(self, prediction: dspy.Prediction):
        # 检查缓存大小
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()

        cache_key = self._generate_cache_key(prediction)
        self.cache[cache_key] = prediction
        self.access_times[cache_key] = time.time()

    def _evict_lru(self):
        if not self.access_times:
            return

        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]

    def clear_expired(self, ttl_seconds: int = 3600):
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > ttl_seconds
        ]

        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
```

**P1.4** 实现一个PythonInterpreter的安全包装器，支持代码执行限制和错误处理：

```python
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import signal

class SafePythonInterpreter:
    def __init__(self, timeout_seconds: int = 30, memory_limit_mb: int = 100):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.allowed_modules = ['math', 'random', 'datetime', 'json', 're']

    def _validate_code(self, code: str) -> bool:
        # 验证代码安全性
        # 检查危险函数和模块导入
        pass

    def _create_sandbox_environment(self) -> Dict[str, Any]:
        # 创建安全的执行环境
        pass

    def execute(self, code: str) -> Dict[str, Any]:
        # 安全执行Python代码
        pass

    def execute_with_timeout(self, code: str) -> Dict[str, Any]:
        # 带超时的代码执行
        pass

# 测试代码
interpreter = SafePythonInterpreter(timeout_seconds=5)

# 测试安全代码
safe_code = """
import math
result = math.sqrt(16)
print(f"The square root is: {result}")
"""

# 测试危险代码
dangerous_code = """
import os
os.system("echo 'This could be dangerous!'")
"""

print("执行安全代码:")
result1 = interpreter.execute(safe_code)
print(result1)

print("\n执行危险代码:")
result2 = interpreter.execute(dangerous_code)
print(result2)
```

> **参考答案**：
```python
class SafePythonInterpreter:
    def __init__(self, timeout_seconds: int = 30, memory_limit_mb: int = 100):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.allowed_modules = ['math', 'random', 'datetime', 'json', 're']
        self.dangerous_functions = [
            'eval', 'exec', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
            'dir', 'help', 'exit', 'quit'
        ]

    def _validate_code(self, code: str) -> bool:
        # 检查危险函数调用
        for func in self.dangerous_functions:
            if func in code:
                return False

        # 检查模块导入
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_modules:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.allowed_modules:
                        return False
        except SyntaxError:
            return False

        return True

    def _create_sandbox_environment(self) -> Dict[str, Any]:
        # 创建受限的执行环境
        safe_globals = {}

        # 只导入允许的模块
        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        # 添加安全的内置函数
        safe_builtins = {
            'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'abs': abs, 'min': min, 'max': max, 'sum': sum
        }

        safe_globals['__builtins__'] = safe_builtins
        return safe_globals

    def execute(self, code: str) -> Dict[str, Any]:
        if not self._validate_code(code):
            return {
                'success': False,
                'error': 'Code validation failed - potentially dangerous content',
                'output': None
            }

        return self.execute_with_timeout(code)

    def execute_with_timeout(self, code: str) -> Dict[str, Any]:
        import sys
        from io import StringIO

        # 捕获输出
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # 创建安全环境
            safe_globals = self._create_sandbox_environment()
            safe_locals = {}

            # 设置超时
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            # 执行代码
            exec(code, safe_globals, safe_locals)

            # 取消超时
            signal.alarm(0)

            # 获取结果和输出
            result = safe_locals.get('result', None)
            output = captured_output.getvalue()

            return {
                'success': True,
                'result': result,
                'output': output,
                'error': None
            }

        except TimeoutError as e:
            return {
                'success': False,
                'error': f'Execution timeout: {str(e)}',
                'output': captured_output.getvalue()
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution error: {str(e)}',
                'output': captured_output.getvalue()
            }
        finally:
            # 恢复stdout
            sys.stdout = old_stdout
            signal.alarm(0)
```

#### ⭐⭐⭐ 高级编程题
**P1.5** 实现一个高性能的批处理系统，支持异步执行和资源管理：

```python
import asyncio
import concurrent.futures
from typing import List, Callable, Any, Dict, Optional
import time
import threading
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0

class HighPerformanceBatchProcessor:
    def __init__(self, max_workers: int = 4, max_concurrent_tasks: int = 10):
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks = set()
        self.completed_tasks = {}
        self.executor = None
        self.loop = None
        self.shutdown_event = asyncio.Event()

    async def start(self):
        # 启动批处理器
        pass

    async def submit_task(self, task: Task) -> str:
        # 提交任务到队列
        pass

    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        # 获取任务结果
        pass

    async def process_tasks(self):
        # 处理任务队列
        pass

    async def execute_task(self, task: Task) -> TaskResult:
        # 执行单个任务
        pass

    async def shutdown(self):
        # 优雅关闭处理器
        pass

    def get_statistics(self) -> Dict[str, Any]:
        # 获取统计信息
        pass

# 使用示例
async def example_task(data: str, multiplier: int = 2) -> str:
    await asyncio.sleep(0.1)  # 模拟异步工作
    return f"{data}_{multiplier}"

async def main():
    processor = HighPerformanceBatchProcessor(max_workers=4)
    await processor.start()

    # 提交多个任务
    task_ids = []
    for i in range(20):
        task = Task(
            id=f"task_{i}",
            function=example_task,
            args=(f"data_{i}",),
            kwargs={"multiplier": i + 1},
            priority=i % 3  # 0-2优先级
        )
        task_id = await processor.submit_task(task)
        task_ids.append(task_id)

    # 等待所有任务完成
    results = []
    for task_id in task_ids:
        result = await processor.get_task_result(task_id, timeout=5.0)
        results.append(result)

    print(f"完成{len(results)}个任务")
    print(processor.get_statistics())

    await processor.shutdown()

# 运行示例
# asyncio.run(main())
```

> **参考答案**：
```python
class HighPerformanceBatchProcessor:
    def __init__(self, max_workers: int = 4, max_concurrent_tasks: int = 10):
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks = set()
        self.completed_tasks = {}
        self.executor = None
        self.loop = None
        self.shutdown_event = asyncio.Event()
        self.statistics = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'total_execution_time': 0.0
        }

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        # 启动任务处理协程
        for _ in range(self.max_concurrent_tasks):
            asyncio.create_task(self.process_tasks())

    async def submit_task(self, task: Task) -> str:
        # 使用负优先级，因为PriorityQueue是最小堆
        await self.task_queue.put((-task.priority, time.time(), task))
        self.statistics['submitted'] += 1
        return task.id

    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        start_time = time.time()

        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} not completed within timeout")

            if self.shutdown_event.is_set():
                break

            await asyncio.sleep(0.01)

        return self.completed_tasks.get(task_id, TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error="Task not found or processor shutdown"
        ))

    async def process_tasks(self):
        while not self.shutdown_event.is_set():
            try:
                # 获取任务，设置超时避免阻塞
                priority, timestamp, task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )

                # 检查并发限制
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    # 重新放回队列
                    await self.task_queue.put((priority, timestamp, task))
                    await asyncio.sleep(0.1)
                    continue

                # 执行任务
                self.running_tasks.add(task.id)
                asyncio.create_task(self.execute_task(task))

            except asyncio.TimeoutError:
                # 超时继续循环
                continue
            except Exception as e:
                print(f"Task processing error: {e}")

    async def execute_task(self, task: Task) -> TaskResult:
        start_time = time.time()

        try:
            # 检查任务是否超时
            if task.timeout:
                result = await asyncio.wait_for(
                    self._run_task_function(task),
                    timeout=task.timeout
                )
            else:
                result = await self._run_task_function(task)

            execution_time = time.time() - start_time

            task_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                retry_count=task.retry_count
            )

            self.statistics['completed'] += 1
            self.statistics['total_execution_time'] += execution_time

        except Exception as e:
            execution_time = time.time() - start_time

            if task.retry_count < task.max_retries:
                # 重试任务
                task.retry_count += 1
                await self.task_queue.put((-task.priority, time.time(), task))

                task_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=f"Failed, retry {task.retry_count}/{task.max_retries}: {str(e)}",
                    execution_time=execution_time,
                    retry_count=task.retry_count
                )
            else:
                # 最终失败
                task_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=f"Failed after {task.max_retries} retries: {str(e)}",
                    execution_time=execution_time,
                    retry_count=task.retry_count
                )

                self.statistics['failed'] += 1

        finally:
            self.running_tasks.discard(task.id)
            self.completed_tasks[task.id] = task_result

    async def _run_task_function(self, task: Task) -> Any:
        # 在线程池中执行同步函数，直接运行异步函数
        if asyncio.iscoroutinefunction(task.function):
            return await task.function(*task.args, **task.kwargs)
        else:
            # 在线程池中运行同步函数
            return await self.loop.run_in_executor(
                self.executor,
                lambda: task.function(*task.args, **task.kwargs)
            )

    async def shutdown(self):
        self.shutdown_event.set()

        # 等待正在运行的任务完成
        while self.running_tasks:
            await asyncio.sleep(0.1)

        # 关闭线程池
        if self.executor:
            self.executor.shutdown(wait=True)

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.statistics.copy()
        stats['running'] = len(self.running_tasks)
        stats['pending'] = self.task_queue.qsize()
        stats['success_rate'] = (
            stats['completed'] / max(stats['submitted'], 1) * 100
        )
        stats['avg_execution_time'] = (
            stats['total_execution_time'] / max(stats['completed'], 1)
        )
        return stats
```

### 项目题

#### ⭐ 项目1：DSPy组件监控面板
**项目描述**：构建一个实时监控系统，跟踪DSPy组件的性能指标、使用情况和健康状态。

**功能要求**：
- 实时显示组件调用次数、响应时间、错误率
- 支持组件性能图表和历史数据查询
- 提供异常检测和告警功能
- 支持组件配置的动态调整

**技术栈**：
- FastAPI + WebSocket 后端
- React + Chart.js 前端
- Redis 缓存数据
- SQLite 存储历史数据

**评估标准**：
- 功能完整性（40%）
- 实时性能（30%）
- 代码质量（20%）
- 用户体验（10%）

#### ⭐⭐ 项目2：智能DSPy组件推荐系统
**项目描述**：基于任务描述和使用历史，推荐最适合的DSPy组件组合。

**功能要求**：
- 分析任务特征（复杂度、领域、数据类型）
- 维护组件性能数据库和使用模式
- 提供组件推荐和优化建议
- 支持A/B测试验证推荐效果

**技术栈**：
- DSPy 核心框架
- scikit-learn 机器学习
- MongoDB 存储使用数据
- Jupyter Notebook 数据分析

**评估标准**：
- 推荐准确性（35%）
- 系统性能（25%）
- 数据分析深度（25%）
- 创新性（15%）

#### ⭐⭐⭐ 项目3：DSPy分布式执行引擎
**项目描述**：构建一个支持大规模分布式执行的DSPy组件运行引擎。

**功能要求**：
- 支持跨多台机器的组件调度
- 实现负载均衡和故障恢复
- 提供组件间的数据管道优化
- 支持动态资源分配和扩展

**技术栈**：
- Kubernetes 容器编排
- gRPC 组件通信
- Prometheus 监控
- Docker 容器化

**评估标准**：
- 分布式性能（40%）
- 可靠性（30%）
- 扩展性（20%）
- 技术复杂度（10%）

---

## 📝 模块2：签名系统 (Signatures)

### 理论题

#### ⭐ 初级理论题
**T2.1** DSPy Signature系统的设计哲学是什么？为什么需要这样的系统？

> **参考答案**：设计哲学：
> - 声明式编程范式：通过声明输入输出规范来定义任务
> - 类型安全：确保数据流的正确性和一致性
> - 可组合性：支持复杂系统的模块化构建
> - 自动优化：为编译器提供明确的优化目标

**T2.2** InputField和OutputField有哪些常用的参数？各自的作用是什么？

> **参考答案**：常用参数：
> - `desc`：字段描述，指导模型理解
> - `type`：字段类型，支持类型验证
> - `prefix`：前缀文本，格式化显示
> - `required`：是否必需，控制必填项
> - `choices`：候选值列表，枚举类型
> - `default`：默认值，可选参数

#### ⭐⭐ 中级理论题
**T2.3** 分析DSPy签名系统的类型安全机制，说明其与Python类型提示的关系。

> **参考答案**：类型安全机制：
> - 运行时类型验证：基于pydantic的动态类型检查
> - 编译时类型推断：支持静态分析和优化
> - 类型转换和强制：自动处理类型兼容性
> - 错误处理和调试：提供详细的类型错误信息

与Python类型提示的关系：
> - 兼容Python类型提示语法
> - 扩展了类型系统的表达能力
> - 提供了更强的运行时保证
> - 支持自定义类型和验证器

**T2.4** 在什么情况下需要动态创建签名？如何保证动态签名的质量和安全性？

> **参考答案**：使用场景：
> - 需要根据用户输入或配置生成任务规范
> - 构建通用的任务处理框架
> - 实现插件化的组件系统
> - 支持多变的业务场景

质量保证措施：
> - 输入验证和清理
> - 字段完整性检查
> - 类型安全验证
> - 默认值和约束设置

安全措施：
> - 限制可用的字段类型
> - 防止代码注入攻击
> - 验证字段描述的合法性
> - 实现权限控制和审计

#### ⭐⭐⭐ 高级理论题
**T2.5** 设计一个签名系统的版本管理和兼容性机制，考虑向前兼容和向后兼容。

> **参考答案**：版本管理机制：
>
> **版本标识**：
> - 语义化版本号（major.minor.patch）
> - 签名哈希值用于快速比较
> - 元数据记录版本变更历史
>
> **向前兼容**：
> - 新增字段设置默认值
> - 保持现有字段的语义不变
> - 使用可选字段扩展功能
> - 提供迁移工具和指南
>
> **向后兼容**：
> - 废弃字段的优雅处理
> - 类型转换和适配器
> - 兼容性测试套件
> - 降级处理机制

### 编程题

#### ⭐ 初级编程题
**P2.1** 创建一个文本分类任务的签名，支持多种分类类别：

```python
import dspy
from typing import List

# 定义文本分类签名
class TextClassification(dspy.Signature):
    # 在这里定义输入输出字段
    # 支持以下功能：
    # 1. 文本输入
    # 2. 多类别分类（技术、娱乐、体育、新闻等）
    # 3. 置信度输出
    # 4. 分类理由说明
    pass

# 测试代码
classifier = dspy.Predict(TextClassification)

test_texts = [
    "最新的iPhone 15发布了，搭载A17芯片",
    "昨天NBA总决赛精彩纷呈",
    "人工智能技术正在快速发展"
]

for text in test_texts:
    result = classifier(text=text)
    print(f"文本: {text}")
    print(f"分类: {result.category}")
    print(f"置信度: {result.confidence}")
    print(f"理由: {result.reasoning}")
    print("---")
```

> **参考答案**：
```python
class TextClassification(dspy.Signature):
    """对文本进行分类"""
    text = dspy.InputField(desc="待分类的文本内容")
    category = dspy.OutputField(
        desc="文本分类结果",
        type=str,
        choices=["技术", "娱乐", "体育", "新闻", "商业", "教育", "其他"]
    )
    confidence = dspy.OutputField(desc="分类置信度(0-1)", type=float)
    reasoning = dspy.OutputField(desc="分类理由和依据")
```

**P2.2** 实现一个签名验证器，检查签名的完整性和合理性：

```python
from typing import List, Dict, Any
import inspect

class SignatureValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_signature(self, signature_class) -> Dict[str, Any]:
        # 验证签名类
        pass

    def check_field_completeness(self, signature_class) -> bool:
        # 检查字段完整性
        pass

    def validate_field_descriptions(self, signature_class) -> List[str]:
        # 验证字段描述质量
        pass

    def check_type_consistency(self, signature_class) -> bool:
        # 检查类型一致性
        pass

    def generate_report(self) -> Dict[str, Any]:
        # 生成验证报告
        pass

# 测试签名
class GoodSignature(dspy.Signature):
    """高质量的签名示例"""
    question = dspy.InputField(desc="用户提出的问题")
    answer = dspy.OutputField(desc="详细准确的答案")

class BadSignature(dspy.Signature):
    """有问题的签名示例"""
    input_data = dspy.InputField()  # 缺少描述
    output = dspy.OutputField()     # 缺少描述和类型

validator = SignatureValidator()

print("验证GoodSignature:")
result1 = validator.validate_signature(GoodSignature)
print(validator.generate_report())

print("\n验证BadSignature:")
result2 = validator.validate_signature(BadSignature)
print(validator.generate_report())
```

> **参考答案**：
```python
class SignatureValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_signature(self, signature_class) -> Dict[str, Any]:
        self.errors.clear()
        self.warnings.clear()

        # 基本检查
        if not inspect.isclass(signature_class):
            self.errors.append("签名必须是一个类")

        if not issubclass(signature_class, dspy.Signature):
            self.errors.append("签名必须继承自dspy.Signature")

        # 字段检查
        self.check_field_completeness(signature_class)
        self.validate_field_descriptions(signature_class)
        self.check_type_consistency(signature_class)

        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy()
        }

    def check_field_completeness(self, signature_class):
        fields = self._get_signature_fields(signature_class)

        if not fields:
            self.errors.append("签名没有定义任何字段")
            return

        input_fields = [f for f in fields if isinstance(f, dspy.InputField)]
        output_fields = [f for f in fields if isinstance(f, dspy.OutputField)]

        if not input_fields:
            self.errors.append("签名缺少输入字段")

        if not output_fields:
            self.errors.append("签名缺少输出字段")

        # 检查必需字段
        for field_name, field in fields.items():
            if hasattr(field, 'required') and not field.required:
                self.warnings.append(f"字段{field_name}是可选的，可能影响使用")

    def validate_field_descriptions(self, signature_class) -> List[str]:
        fields = self._get_signature_fields(signature_class)

        for field_name, field in fields.items():
            if not hasattr(field, 'desc') or not field.desc:
                self.errors.append(f"字段{field_name}缺少描述")
            elif len(field.desc.strip()) < 5:
                self.warnings.append(f"字段{field_name}的描述过于简短")
            elif not any(c in field.desc for c in ['。', '.', '？', '?']):
                self.warnings.append(f"字段{field_name}的描述建议使用完整句子")

    def check_type_consistency(self, signature_class) -> bool:
        fields = self._get_signature_fields(signature_class)

        for field_name, field in fields.items():
            if hasattr(field, 'type'):
                # 检查类型是否合法
                valid_types = [str, int, float, bool, list, dict, List, Dict]
                if field.type not in valid_types and not hasattr(field.type, '__origin__'):
                    self.warnings.append(f"字段{field_name}使用了不常见的类型")

        return len(self.errors) == 0

    def _get_signature_fields(self, signature_class):
        # 获取签名的所有字段
        fields = {}

        for name in dir(signature_class):
            if not name.startswith('_'):
                attr = getattr(signature_class, name)
                if isinstance(attr, (dspy.InputField, dspy.OutputField)):
                    fields[name] = attr

        return fields

    def generate_report(self) -> Dict[str, Any]:
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        recommendations = []

        if self.errors:
            recommendations.append("修复所有错误才能正常使用签名")

        if self.warnings:
            recommendations.append("考虑修复警告以提高签名质量")

        if not self.errors and not self.warnings:
            recommendations.append("签名质量良好，可以正常使用")

        return recommendations
```

#### ⭐⭐ 中级编程题
**P2.3** 实现一个签名模板系统，支持参数化签名生成：

```python
from typing import Dict, List, Any, Type
import json

class SignatureTemplate:
    def __init__(self, template_name: str, template_def: Dict[str, Any]):
        self.template_name = template_name
        self.template_def = template_def

    def validate_template(self) -> bool:
        # 验证模板定义的合法性
        pass

    def generate_signature(self, parameters: Dict[str, Any] = None) -> Type[dspy.Signature]:
        # 根据模板和参数生成具体的签名类
        pass

    def get_required_parameters(self) -> List[str]:
        # 获取模板所需的参数
        pass

class SignatureTemplateEngine:
    def __init__(self):
        self.templates = {}

    def load_template(self, template_path: str) -> bool:
        # 从文件加载模板
        pass

    def register_template(self, template: SignatureTemplate):
        # 注册模板
        pass

    def create_signature(self, template_name: str, parameters: Dict[str, Any] = None) -> Type[dspy.Signature]:
        # 创建签名
        pass

    def list_templates(self) -> List[str]:
        # 列出所有可用模板
        pass

# 模板定义示例
QA_TEMPLATE = {
    "name": "question_answering",
    "description": "通用问答模板",
    "parameters": {
        "domain": {
            "type": "string",
            "default": "general",
            "description": "专业领域"
        },
        "style": {
            "type": "string",
            "choices": ["formal", "casual", "technical"],
            "default": "formal"
        }
    },
    "fields": {
        "input_fields": [
            {
                "name": "question",
                "description": "关于{domain}的问题",
                "type": "string"
            }
        ],
        "output_fields": [
            {
                "name": "answer",
                "description": "{style}风格的答案",
                "type": "string"
            },
            {
                "name": "confidence",
                "description": "答案置信度",
                "type": "float",
                "required": False
            }
        ]
    }
}

# 使用示例
engine = SignatureTemplateEngine()

# 注册模板
template = SignatureTemplate("qa_template", QA_TEMPLATE)
engine.register_template(template)

# 生成签名
qa_signature = engine.create_signature("qa_template", {
    "domain": "科技",
    "style": "technical"
})

print(f"生成的签名类: {qa_signature.__name__}")

# 测试使用
predictor = dspy.Predict(qa_signature)
result = predictor(question="什么是机器学习？")
print(f"答案: {result.answer}")
```

> **参考答案**：
```python
class SignatureTemplate:
    def __init__(self, template_name: str, template_def: Dict[str, Any]):
        self.template_name = template_name
        self.template_def = template_def

    def validate_template(self) -> bool:
        required_keys = ['name', 'parameters', 'fields']

        for key in required_keys:
            if key not in self.template_def:
                raise ValueError(f"模板缺少必需的键: {key}")

        fields = self.template_def['fields']
        if 'input_fields' not in fields or 'output_fields' not in fields:
            raise ValueError("模板必须包含输入和输出字段定义")

        return True

    def generate_signature(self, parameters: Dict[str, Any] = None) -> Type[dspy.Signature]:
        if not self.validate_template():
            raise ValueError("模板验证失败")

        parameters = parameters or {}

        # 合并默认参数
        merged_params = {}
        for param_name, param_def in self.template_def['parameters'].items():
            merged_params[param_name] = parameters.get(param_name, param_def.get('default'))

        # 生成类定义
        class_attrs = {
            '__doc__': self.template_def.get('description', ''),
            '__annotations__': {}
        }

        # 处理字段
        self._process_fields(class_attrs, merged_params)

        # 创建签名类
        signature_name = f"{self.template_name}_{hash(str(parameters))}"
        signature_class = type(signature_name, (dspy.Signature,), class_attrs)

        return signature_class

    def _process_fields(self, class_attrs: Dict, parameters: Dict[str, Any]):
        fields = self.template_def['fields']

        # 处理输入字段
        for field_def in fields['input_fields']:
            field_name = field_def['name']
            field_desc = self._substitute_parameters(field_def['description'], parameters)
            field_type = self._get_field_type(field_def.get('type', 'string'))
            required = field_def.get('required', True)

            class_attrs[field_name] = dspy.InputField(
                desc=field_desc,
                type=field_type,
                required=required
            )
            class_attrs['__annotations__'][field_name] = field_type

        # 处理输出字段
        for field_def in fields['output_fields']:
            field_name = field_def['name']
            field_desc = self._substitute_parameters(field_def['description'], parameters)
            field_type = self._get_field_type(field_def.get('type', 'string'))
            required = field_def.get('required', True)

            class_attrs[field_name] = dspy.OutputField(
                desc=field_desc,
                type=field_type,
                required=required
            )
            class_attrs['__annotations__'][field_name] = field_type

    def _substitute_parameters(self, text: str, parameters: Dict[str, Any]) -> str:
        """替换模板参数"""
        for param_name, param_value in parameters.items():
            placeholder = "{" + param_name + "}"
            text = text.replace(placeholder, str(param_value))
        return text

    def _get_field_type(self, type_str: str):
        """获取字段类型"""
        type_map = {
            'string': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        return type_map.get(type_str.lower(), str)

    def get_required_parameters(self) -> List[str]:
        """获取必需参数"""
        required_params = []

        for param_name, param_def in self.template_def['parameters'].items():
            if param_def.get('required', False) and 'default' not in param_def:
                required_params.append(param_name)

        return required_params

class SignatureTemplateEngine:
    def __init__(self):
        self.templates = {}

    def load_template(self, template_path: str) -> bool:
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_def = json.load(f)

            template = SignatureTemplate(template_def['name'], template_def)
            self.register_template(template)
            return True

        except Exception as e:
            print(f"加载模板失败: {e}")
            return False

    def register_template(self, template: SignatureTemplate):
        self.templates[template.template_name] = template

    def create_signature(self, template_name: str, parameters: Dict[str, Any] = None) -> Type[dspy.Signature]:
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 不存在")

        template = self.templates[template_name]
        return template.generate_signature(parameters)

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())
```

#### ⭐⭐⭐ 高级编程题
**P2.4** 实现一个签名优化器，基于使用数据自动优化签名定义：

```python
from typing import Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class OptimizationType(Enum):
    FIELD_REORDERING = "field_reordering"
    DESCRIPTION_IMPROVEMENT = "description_improvement"
    TYPE_OPTIMIZATION = "type_optimization"
    FIELD_ADDITION = "field_addition"
    FIELD_REMOVAL = "field_removal"

@dataclass
class SignatureUsage:
    signature_name: str
    usage_count: int
    success_rate: float
    avg_response_time: float
    error_patterns: List[str]
    field_usage_stats: Dict[str, Dict[str, Any]]

@dataclass
class OptimizationSuggestion:
    optimization_type: OptimizationType
    description: str
    expected_improvement: float
    confidence: float
    changes: Dict[str, Any]

class SignatureOptimizer:
    def __init__(self):
        self.usage_data = {}
        self.optimization_history = []
        self.performance_benchmarks = {}

    def record_usage(self, signature_class, execution_time: float, success: bool,
                    error_msg: str = None, field_access: Dict[str, bool] = None):
        # 记录签名使用数据
        pass

    def analyze_performance(self, signature_class: Type[dspy.Signature]) -> Dict[str, Any]:
        # 分析签名性能
        pass

    def generate_optimization_suggestions(self, signature_class: Type[dspy.Signature]) -> List[OptimizationSuggestion]:
        # 生成优化建议
        pass

    def apply_optimization(self, signature_class: Type[dspy.Signature],
                          suggestion: OptimizationSuggestion) -> Type[dspy.Signature]:
        # 应用优化建议
        pass

    def benchmark_optimization(self, original_signature: Type[dspy.Signature],
                              optimized_signature: Type[dspy.Signature],
                              test_cases: List[Dict]) -> Dict[str, Any]:
        # 对比优化效果
        pass

# 使用示例
class TestSignature(dspy.Signature):
    """测试签名 - 需要优化"""
    input_text = dspy.InputField(desc="输入文本")
    result = dspy.OutputField(desc="处理结果")

optimizer = SignatureOptimizer()

# 模拟使用数据
for i in range(100):
    execution_time = np.random.normal(1.5, 0.3)  # 平均1.5秒
    success = np.random.random() > 0.2  # 80%成功率
    error_msg = "timeout" if not success and np.random.random() > 0.5 else "invalid_input"

    field_access = {
        'input_text': True,
        'result': success  # 失败时可能无法访问result字段
    }

    optimizer.record_usage(TestSignature, execution_time, success, error_msg, field_access)

# 分析和优化
performance = optimizer.analyze_performance(TestSignature)
suggestions = optimizer.generate_optimization_suggestions(TestSignature)

print("性能分析:", performance)
print(f"生成{len(suggestions)}个优化建议:")

for i, suggestion in enumerate(suggestions):
    print(f"{i+1}. {suggestion.description}")
    print(f"   预期改进: {suggestion.expected_improvement:.2f}")
    print(f"   置信度: {suggestion.confidence:.2f}")

    # 应用第一个优化建议
    if i == 0:
        optimized_signature = optimizer.apply_optimization(TestSignature, suggestion)
        print(f"   优化后的签名: {optimized_signature.__name__}")
```

> **参考答案**：
```python
class SignatureOptimizer:
    def __init__(self):
        self.usage_data = {}
        self.optimization_history = []
        self.performance_benchmarks = {}

    def record_usage(self, signature_class, execution_time: float, success: bool,
                    error_msg: str = None, field_access: Dict[str, bool] = None):
        signature_name = signature_class.__name__

        if signature_name not in self.usage_data:
            self.usage_data[signature_name] = {
                'usage_count': 0,
                'success_count': 0,
                'total_time': 0.0,
                'errors': [],
                'field_usage': {}
            }

        data = self.usage_data[signature_name]
        data['usage_count'] += 1
        data['total_time'] += execution_time

        if success:
            data['success_count'] += 1
        else:
            data['errors'].append(error_msg)

        # 记录字段访问情况
        if field_access:
            for field_name, accessed in field_access.items():
                if field_name not in data['field_usage']:
                    data['field_usage'][field_name] = {'access_count': 0, 'access_rate': 0.0}

                if accessed:
                    data['field_usage'][field_name]['access_count'] += 1

                data['field_usage'][field_name]['access_rate'] = (
                    data['field_usage'][field_name]['access_count'] / data['usage_count']
                )

    def analyze_performance(self, signature_class: Type[dspy.Signature]) -> Dict[str, Any]:
        signature_name = signature_class.__name__

        if signature_name not in self.usage_data:
            return {'error': '没有使用数据'}

        data = self.usage_data[signature_name]

        return {
            'signature_name': signature_name,
            'usage_count': data['usage_count'],
            'success_rate': data['success_count'] / data['usage_count'],
            'avg_response_time': data['total_time'] / data['usage_count'],
            'error_patterns': self._analyze_error_patterns(data['errors']),
            'field_analysis': data['field_usage'],
            'performance_score': self._calculate_performance_score(data)
        }

    def generate_optimization_suggestions(self, signature_class: Type[dspy.Signature]) -> List[OptimizationSuggestion]:
        performance = self.analyze_performance(signature_class)
        suggestions = []

        # 响应时间优化
        if performance['avg_response_time'] > 2.0:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.FIELD_REORDERING,
                description="响应时间过长，建议优化字段顺序和类型",
                expected_improvement=0.3,
                confidence=0.7,
                changes={'reorder_fields': True}
            ))

        # 成功率优化
        if performance['success_rate'] < 0.8:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.DESCRIPTION_IMPROVEMENT,
                description="成功率偏低，建议改进字段描述",
                expected_improvement=0.2,
                confidence=0.8,
                changes={'improve_descriptions': True}
            ))

        # 字段使用率优化
        low_usage_fields = [
            field for field, stats in performance['field_analysis'].items()
            if stats['access_rate'] < 0.3
        ]

        if low_usage_fields:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.FIELD_REMOVAL,
                description=f"移除使用率低的字段: {', '.join(low_usage_fields)}",
                expected_improvement=0.15,
                confidence=0.6,
                changes={'remove_fields': low_usage_fields}
            ))

        # 类型优化
        if 'invalid_input' in performance['error_patterns']:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.TYPE_OPTIMIZATION,
                description="输入类型错误较多，建议改进类型定义",
                expected_improvement=0.25,
                confidence=0.8,
                changes={'add_type_validation': True}
            ))

        return suggestions

    def apply_optimization(self, signature_class: Type[dspy.Signature],
                          suggestion: OptimizationSuggestion) -> Type[dspy.Signature]:
        if suggestion.optimization_type == OptimizationType.DESCRIPTION_IMPROVEMENT:
            return self._improve_descriptions(signature_class)
        elif suggestion.optimization_type == OptimizationType.FIELD_REORDERING:
            return self._reorder_fields(signature_class)
        elif suggestion.optimization_type == OptimizationType.FIELD_REMOVAL:
            return self._remove_fields(signature_class, suggestion.changes['remove_fields'])
        elif suggestion.optimization_type == OptimizationType.TYPE_OPTIMIZATION:
            return self._optimize_types(signature_class)
        else:
            return signature_class

    def _improve_descriptions(self, signature_class: Type[dspy.Signature]) -> Type[dspy.Signature]:
        class_attrs = {}

        for attr_name in dir(signature_class):
            if not attr_name.startswith('_'):
                attr = getattr(signature_class, attr_name)
                if isinstance(attr, (dspy.InputField, dspy.OutputField)):
                    # 改进字段描述
                    improved_attr = attr.copy()
                    if hasattr(attr, 'desc'):
                        improved_desc = attr.desc + " (请提供详细和准确的描述)"
                        improved_attr.desc = improved_desc

                    class_attrs[attr_name] = improved_attr

        new_class_name = f"Improved{signature_class.__name__}"
        return type(new_class_name, (dspy.Signature,), class_attrs)

    def _reorder_fields(self, signature_class: Type[dspy.Signature]) -> Type[dspy.Signature]:
        # 简化实现：按字母顺序重新排列字段
        class_attrs = {}
        fields = {}

        # 收集所有字段
        for attr_name in dir(signature_class):
            if not attr_name.startswith('_'):
                attr = getattr(signature_class, attr_name)
                if isinstance(attr, (dspy.InputField, dspy.OutputField)):
                    fields[attr_name] = attr

        # 按字母排序
        sorted_fields = dict(sorted(fields.items()))
        class_attrs.update(sorted_fields)

        new_class_name = f"Reordered{signature_class.__name__}"
        return type(new_class_name, (dspy.Signature,), class_attrs)

    def _remove_fields(self, signature_class: Type[dspy.Signature],
                      fields_to_remove: List[str]) -> Type[dspy.Signature]:
        class_attrs = {}

        for attr_name in dir(signature_class):
            if not attr_name.startswith('_') and attr_name not in fields_to_remove:
                attr = getattr(signature_class, attr_name)
                if isinstance(attr, (dspy.InputField, dspy.OutputField)):
                    class_attrs[attr_name] = attr

        new_class_name = f"Simplified{signature_class.__name__}"
        return type(new_class_name, (dspy.Signature,), class_attrs)

    def _optimize_types(self, signature_class: Type[dspy.Signature]) -> Type[dspy.Signature]:
        class_attrs = {}

        for attr_name in dir(signature_class):
            if not attr_name.startswith('_'):
                attr = getattr(signature_class, attr_name)
                if isinstance(attr, (dspy.InputField, dspy.OutputField)):
                    optimized_attr = attr.copy()

                    # 添加类型验证
                    if not hasattr(optimized_attr, 'type'):
                        optimized_attr.type = str

                    class_attrs[attr_name] = optimized_attr

        new_class_name = f"Typed{signature_class.__name__}"
        return type(new_class_name, (dspy.Signature,), class_attrs)

    def _analyze_error_patterns(self, errors: List[str]) -> Dict[str, int]:
        pattern_counts = {}
        for error in errors:
            if error in pattern_counts:
                pattern_counts[error] += 1
            else:
                pattern_counts[error] = 1
        return pattern_counts

    def _calculate_performance_score(self, data: Dict) -> float:
        success_rate = data['success_count'] / data['usage_count']
        avg_time = data['total_time'] / data['usage_count']

        # 综合评分：成功率权重70%，响应时间权重30%
        success_score = success_rate * 0.7
        time_score = max(0, 1 - (avg_time / 5.0)) * 0.3  # 假设5秒为基准

        return success_score + time_score
```

### 项目题

#### ⭐ 项目1：DSL签名设计器
**项目描述**：开发一个可视化工具，帮助用户设计和配置DSPy签名。

**功能要求**：
- 拖拽式字段编辑器
- 实时签名预览
- 模板库和示例
- 代码导出功能

**技术栈**：
- React + TypeScript前端
- Node.js后端
- Monaco Editor代码编辑
- D3.js可视化

**评估标准**：
- 易用性（40%）
- 功能完整性（30%）
- 代码质量（20%）
- 创新性（10%）

#### ⭐⭐ 项目2：智能签名推荐系统
**项目描述**：基于任务描述和上下文，自动推荐最适合的签名配置。

**功能要求**：
- 自然语言任务解析
- 相似签名匹配
- 个性化推荐算法
- 效果评估和反馈

**技术栈**：
- DSPy核心框架
- NLP文本处理
- 机器学习推荐算法
- 用户行为分析

**评估标准**：
- 推荐准确性（35%）
- 算法性能（25%）
- 用户体验（25%）
- 技术创新（15%）

---

*（由于篇幅限制，剩余13个模块的练习题将在后续文档中继续）*