# DSPy教学评估体系

## 📊 评估体系概览

本文档提供了完整的DSPy课程教学评估体系，包括学习目标检测、知识掌握度评估、项目质量评定和综合能力认证。

**评估维度：**
- 🎯 **理论知识**：概念理解和原理掌握
- 💻 **实践技能**：代码实现和问题解决
- 🚀 **项目能力**：综合应用和创新开发
- 📈 **学习态度**：参与度和持续改进

---

## 🎯 学习目标检测

### 阶段性学习目标

#### 第一阶段：DSPy基础（第1-3周）
**核心学习目标：**
- ✅ 理解DSPy的设计哲学和核心概念
- ✅ 掌握开发环境搭建和基础配置
- ✅ 能够定义复杂任务规范
- ✅ 熟练使用基础预测模块
- ✅ 完成第一个DSPy应用

**检测指标：**
```python
# 阶段1检测标准
class Stage1Assessment:
    """第一阶段评估标准"""

    ASSESSMENT_CRITERIA = {
        'concept_understanding': {  # 概念理解 (30%)
            'dspy_philosophy': {
                'weight': 0.4,
                'indicators': [
                    '能清晰解释DSPy与传统prompt engineering的区别',
                    '理解编程式AI开发的优势',
                    '了解DSPy生态系统和应用场景'
                ]
            },
            'core_concepts': {
                'weight': 0.6,
                'indicators': [
                    '掌握Module、Signature、Prediction等核心概念',
                    '理解数据流和类型系统',
                    '了解组件间的关系和交互'
                ]
            }
        },

        'practical_skills': {  # 实践技能 (40%)
            'environment_setup': {
                'weight': 0.3,
                'indicators': [
                    '独立完成开发环境搭建',
                    '正确配置API密钥和依赖',
                    '解决常见的配置问题'
                ]
            },
            'basic_usage': {
                'weight': 0.4,
                'indicators': [
                    '能够定义和使用签名',
                    '熟练使用基础预测模块',
                    '掌握数据预处理和验证'
                ]
            },
            'debugging': {
                'weight': 0.3,
                'indicators': [
                    '能够识别和解决常见错误',
                    '使用调试工具分析问题',
                    '理解错误信息的含义'
                ]
            }
        },

        'project_completion': {  # 项目完成 (30%)
            'functionality': {
                'weight': 0.5,
                'indicators': [
                    '项目功能完整且可运行',
                    '实现了所有要求的功能点',
                    '代码结构清晰合理'
                ]
            },
            'quality': {
                'weight': 0.5,
                'indicators': [
                    '代码质量高，符合规范',
                    '有适当的注释和文档',
                    '错误处理完善'
                ]
            }
        }
    }

    def assess_student(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估学生表现"""
        assessment_result = {
            'total_score': 0,
            'category_scores': {},
            'detailed_feedback': {},
            'recommendations': []
        }

        total_weighted_score = 0

        for category, criteria in self.ASSESSMENT_CRITERIA.items():
            category_score = 0
            category_weight = sum(c['weight'] for c in criteria.values())
            detailed_feedback = {}

            for subcategory, subcriteria in criteria.items():
                subcategory_score = self._assess_subcategory(
                    subcriteria, student_data.get(subcategory, {})
                )

                category_score += subcategory_score * subcriteria['weight']
                detailed_feedback[subcategory] = {
                    'score': subcategory_score,
                    'feedback': self._generate_feedback(subcategory_score, subcriteria['indicators'])
                }

            # 归一化分数
            normalized_score = category_score / category_weight if category_weight > 0 else 0
            assessment_result['category_scores'][category] = {
                'score': normalized_score,
                'weight': category_weight / 100,  # 转换为0-1范围
                'details': detailed_feedback
            }

            total_weighted_score += normalized_score * (category_weight / 100)

        assessment_result['total_score'] = total_weighted_score
        assessment_result['recommendations'] = self._generate_recommendations(
            assessment_result['category_scores']
        )

        return assessment_result

    def _assess_subcategory(self, subcriteria: Dict, student_data: Dict) -> float:
        """评估子类别"""
        total_score = 0
        indicator_count = len(subcriteria['indicators'])

        for indicator in subcriteria['indicators']:
            # 简化的评分逻辑，实际应用中会更复杂
            score = student_data.get(indicator, 0)
            total_score += score

        return total_score / indicator_count if indicator_count > 0 else 0

    def _generate_feedback(self, score: float, indicators: List[str]) -> str:
        """生成反馈意见"""
        if score >= 0.8:
            return "优秀！完全掌握了相关技能"
        elif score >= 0.6:
            return "良好！基本掌握，还有提升空间"
        elif score >= 0.4:
            return "合格！需要加强练习"
        else:
            return "需要改进！建议重新学习和练习"

    def _generate_recommendations(self, category_scores: Dict) -> List[str]:
        """生成学习建议"""
        recommendations = []

        for category, data in category_scores.items():
            if data['score'] < 0.6:
                if category == 'concept_understanding':
                    recommendations.append("建议重新学习基础概念，观看相关视频教程")
                elif category == 'practical_skills':
                    recommendations.append("建议多做编程练习，重点提升实践技能")
                elif category == 'project_completion':
                    recommendations.append("建议分析优秀项目案例，学习最佳实践")

        if not recommendations:
            recommendations.append("表现优秀！可以开始下一阶段的学习")

        return recommendations
```

#### 第二阶段：模块化系统构建（第4-6周）
**核心学习目标：**
- ✅ 掌握Module基类的高级用法
- ✅ 构建生产级RAG系统
- ✅ 开发功能完整的智能体
- ✅ 理解系统架构设计原则

**检测标准：**
```python
# 阶段2检测标准
class Stage2Assessment:
    """第二阶段评估标准"""

    COMPETENCY_MATRIX = {
        'advanced_module_development': {
            'beginner': {
                'description': '能够使用现有的DSPy模块',
                'indicators': ['正确使用Module基类', '调用API完成基本功能']
            },
            'intermediate': {
                'description': '能够自定义简单的DSPy模块',
                'indicators': ['继承Module类', '实现forward方法', '处理基本错误']
            },
            'advanced': {
                'description': '能够开发复杂的自定义模块',
                'indicators': ['设计模块架构', '实现状态管理', '添加缓存机制', '性能优化']
            }
        },

        'rag_system_implementation': {
            'beginner': {
                'description': '理解RAG基本概念',
                'indicators': ['解释RAG原理', '使用基础检索器']
            },
            'intermediate': {
                'description': '能够构建基础RAG系统',
                'indicators': ['实现检索和生成', '处理查询路由', '优化检索质量']
            },
            'advanced': {
                'description': '能够构建企业级RAG系统',
                'indicators': ['混合检索策略', '多级重排序', '性能优化', '监控和调试']
            }
        },

        'agent_development': {
            'beginner': {
                'description': '理解智能体基本概念',
                'indicators': ['使用ReAct组件', '集成简单工具']
            },
            'intermediate': {
                'description': '能够开发功能完整的智能体',
                'indicators': ['设计对话管理', '集成多个工具', '处理复杂任务']
            },
            'advanced': {
                'description': '能够构建多智能体协作系统',
                'indicators': ['智能体协调', '任务调度', '负载均衡', '容错机制']
            }
        }
    }

    def evaluate_competency(self, skill_area: str, student_work: Dict) -> Dict[str, Any]:
        """评估学生能力水平"""
        if skill_area not in self.COMPETENCY_MATRIX:
            raise ValueError(f"未知的技能领域: {skill_area}")

        competency_levels = self.COMPETENCY_MATRIX[skill_area]
        student_level = self._determine_level(student_work, competency_levels)

        return {
            'skill_area': skill_area,
            'current_level': student_level,
            'level_description': competency_levels[student_level]['description'],
            'mastered_indicators': self._check_indicators(
                student_work, competency_levels[student_level]['indicators']
            ),
            'next_level_suggestions': self._get_next_level_suggestions(
                student_level, competency_levels
            )
        }

    def _determine_level(self, student_work: Dict, levels: Dict) -> str:
        """确定学生能力水平"""
        # 从高级到低级检查
        for level in ['advanced', 'intermediate', 'beginner']:
            indicators = levels[level]['indicators']
            if all(student_work.get(indicator, False) for indicator in indicators):
                return level

        return 'beginner'  # 默认为初级

    def _check_indicators(self, student_work: Dict, indicators: List[str]) -> List[str]:
        """检查已掌握的指标"""
        mastered = []
        for indicator in indicators:
            if student_work.get(indicator, False):
                mastered.append(indicator)
        return mastered

    def _get_next_level_suggestions(self, current_level: str, levels: Dict) -> List[str]:
        """获取下一阶段的学习建议"""
        level_order = ['beginner', 'intermediate', 'advanced']
        current_index = level_order.index(current_level)

        if current_index >= len(level_order) - 1:
            return ["已经达到最高水平，可以学习高级专题"]

        next_level = level_order[current_index + 1]
        next_indicators = levels[next_level]['indicators']

        return [f"学习并掌握: {indicator}" for indicator in next_indicators]
```

---

## 💻 实时评估系统

### 在线评估平台
```python
# src/assessment/evaluation_platform.py

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """问题类型"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"
    PRACTICAL_EXERCISE = "practical_exercise"
    PROJECT_EVALUATION = "project_evaluation"

@dataclass
class Question:
    """评估题目"""
    id: str
    type: QuestionType
    title: str
    description: str
    options: Optional[List[str]] = None
    correct_answer: Optional[Any] = None
    points: int = 10
    difficulty: str = "medium"  # easy, medium, hard
    module: str = ""
    tags: List[str] = None
    time_limit: Optional[int] = None  # 秒

@dataclass
class AssessmentResult:
    """评估结果"""
    student_id: str
    assessment_id: str
    score: float
    total_points: float
    answers: Dict[str, Any]
    question_scores: Dict[str, float]
    time_spent: float
    timestamp: float
    feedback: Dict[str, str]

class AutoGrader:
    """自动评分器"""

    def __init__(self):
        self.graders = {
            QuestionType.MULTIPLE_CHOICE: self._grade_multiple_choice,
            QuestionType.TRUE_FALSE: self._grade_true_false,
            QuestionType.SHORT_ANSWER: self._grade_short_answer,
            QuestionType.CODE_COMPLETION: self._grade_code_completion,
            QuestionType.PRACTICAL_EXERCISE: self._grade_practical_exercise,
            QuestionType.PROJECT_EVALUATION: self._grade_project
        }

    def grade_answer(self, question: Question, student_answer: Any) -> Dict[str, Any]:
        """评分单个答案"""
        grader = self.graders.get(question.type)
        if not grader:
            raise ValueError(f"不支持的问题类型: {question.type}")

        return grader(question, student_answer)

    def _grade_multiple_choice(self, question: Question, student_answer: str) -> Dict[str, Any]:
        """评分选择题"""
        correct = student_answer == question.correct_answer
        score = question.points if correct else 0

        return {
            'score': score,
            'correct': correct,
            'feedback': "正确！" if correct else f"正确答案是: {question.correct_answer}",
            'points_earned': score,
            'points_possible': question.points
        }

    def _grade_true_false(self, question: Question, student_answer: bool) -> Dict[str, Any]:
        """评分判断题"""
        correct = student_answer == question.correct_answer
        score = question.points if correct else 0

        return {
            'score': score,
            'correct': correct,
            'feedback': "正确！" if correct else f"正确答案是: {question.correct_answer}",
            'points_earned': score,
            'points_possible': question.points
        }

    def _grade_short_answer(self, question: Question, student_answer: str) -> Dict[str, Any]:
        """评分简答题"""
        # 简化的关键词匹配评分
        correct_answer = question.correct_answer.lower()
        student_answer_lower = student_answer.lower()

        correct_keywords = correct_answer.split()
        student_keywords = student_answer_lower.split()

        match_count = sum(1 for word in correct_keywords if word in student_keywords)
        match_ratio = match_count / len(correct_keywords) if correct_keywords else 0

        score = int(question.points * match_ratio)

        return {
            'score': score,
            'correct': match_ratio >= 0.7,  # 70%以上算正确
            'feedback': f"得分: {score}/{question.points}。关键词匹配率: {match_ratio:.1%}",
            'points_earned': score,
            'points_possible': question.points,
            'match_ratio': match_ratio
        }

    def _grade_code_completion(self, question: Question, student_answer: str) -> Dict[str, Any]:
        """评分代码补全题"""
        try:
            # 执行学生代码
            exec_globals = {}
            exec(student_answer, exec_globals)

            # 检查是否包含预期的函数或变量
            expected_elements = question.correct_answer.get('expected_elements', [])
            found_elements = []

            for element in expected_elements:
                if element in exec_globals:
                    found_elements.append(element)

            score_ratio = len(found_elements) / len(expected_elements) if expected_elements else 0
            score = int(question.points * score_ratio)

            return {
                'score': score,
                'correct': score_ratio >= 0.8,
                'feedback': f"代码执行成功。找到{len(found_elements)}/{len(expected_elements)}个预期元素",
                'points_earned': score,
                'points_possible': question.points,
                'found_elements': found_elements
            }

        except Exception as e:
            return {
                'score': 0,
                'correct': False,
                'feedback': f"代码执行失败: {str(e)}",
                'points_earned': 0,
                'points_possible': question.points,
                'error': str(e)
            }

    def _grade_practical_exercise(self, question: Question, student_answer: Dict) -> Dict[str, Any]:
        """评分实践练习"""
        # 这里可以实现更复杂的评分逻辑
        # 例如：运行测试用例、检查代码质量等

        test_cases = question.correct_answer.get('test_cases', [])
        passed_tests = 0

        for test_case in test_cases:
            try:
                # 模拟测试用例执行
                result = self._run_test_case(student_answer, test_case)
                if result.get('success', False):
                    passed_tests += 1
            except Exception:
                pass

        score_ratio = passed_tests / len(test_cases) if test_cases else 0
        score = int(question.points * score_ratio)

        return {
            'score': score,
            'correct': score_ratio >= 0.8,
            'feedback': f"通过{passed_tests}/{len(test_cases)}个测试用例",
            'points_earned': score,
            'points_possible': question.points,
            'passed_tests': passed_tests,
            'total_tests': len(test_cases)
        }

    def _grade_project(self, question: Question, student_answer: Dict) -> Dict[str, Any]:
        """评分项目"""
        # 项目评分通常需要人工评审
        # 这里提供基础的自动化检查

        evaluation_criteria = {
            'functionality': self._check_functionality(student_answer),
            'code_quality': self._check_code_quality(student_answer),
            'documentation': self._check_documentation(student_answer),
            'testing': self._check_testing(student_answer)
        }

        total_score = sum(evaluation_criteria.values())
        max_score = len(evaluation_criteria) * question.points
        final_score = int(question.points * (total_score / max_score))

        return {
            'score': final_score,
            'correct': final_score >= question.points * 0.6,
            'feedback': self._generate_project_feedback(evaluation_criteria),
            'points_earned': final_score,
            'points_possible': question.points,
            'detailed_scores': evaluation_criteria
        }

    def _run_test_case(self, student_answer: Dict, test_case: Dict) -> Dict:
        """运行测试用例"""
        # 简化实现
        return {'success': True, 'output': 'test passed'}

    def _check_functionality(self, project: Dict) -> float:
        """检查功能完整性"""
        return 0.8  # 简化实现

    def _check_code_quality(self, project: Dict) -> float:
        """检查代码质量"""
        return 0.7  # 简化实现

    def _check_documentation(self, project: Dict) -> float:
        """检查文档完整性"""
        return 0.6  # 简化实现

    def _check_testing(self, project: Dict) -> float:
        """检查测试覆盖"""
        return 0.5  # 简化实现

    def _generate_project_feedback(self, scores: Dict[str, float]) -> str:
        """生成项目反馈"""
        feedback_parts = []
        for criterion, score in scores.items():
            status = "优秀" if score >= 0.8 else "良好" if score >= 0.6 else "需改进"
            feedback_parts.append(f"{criterion}: {status}")

        return " | ".join(feedback_parts)

class AdaptiveAssessment:
    """自适应评估系统"""

    def __init__(self, question_bank: Dict[str, List[Question]], grader: AutoGrader):
        self.question_bank = question_bank
        self.grader = grader
        self.student_profiles: Dict[str, Dict] = {}

    def create_assessment(self, student_id: str, module: str, difficulty: str = "adaptive") -> List[Question]:
        """创建个性化评估"""
        student_profile = self.student_profiles.get(student_id, {})

        if difficulty == "adaptive":
            # 根据学生表现调整难度
            performance = student_profile.get('performance', {})
            avg_score = performance.get(module, {}).get('avg_score', 0.5)

            if avg_score >= 0.8:
                difficulty = "hard"
            elif avg_score >= 0.6:
                difficulty = "medium"
            else:
                difficulty = "easy"

        # 选择题目
        available_questions = self.question_bank.get(module, [])
        selected_questions = [
            q for q in available_questions
            if q.difficulty == difficulty
        ]

        # 随机选择指定数量的题目
        import random
        random.shuffle(selected_questions)
        return selected_questions[:10]  # 10道题

    def evaluate_student(self, student_id: str, assessment_id: str, answers: Dict[str, Any]) -> AssessmentResult:
        """评估学生答案"""
        start_time = time.time()

        total_score = 0
        total_points = 0
        question_scores = {}
        feedback = {}

        # 获取评估题目（这里简化处理）
        # 实际实现中需要从数据库或缓存中获取

        for question_id, student_answer in answers.items():
            # 这里需要获取对应的题目
            # question = get_question_by_id(question_id)

            # 模拟评分
            grade_result = {
                'score': 8,
                'points_earned': 8,
                'points_possible': 10,
                'feedback': 'Good job!'
            }

            total_score += grade_result['score']
            total_points += grade_result['points_possible']
            question_scores[question_id] = grade_result['score']
            feedback[question_id] = grade_result['feedback']

        final_score = (total_score / total_points * 100) if total_points > 0 else 0

        # 更新学生档案
        self._update_student_profile(student_id, assessment_id, final_score)

        assessment_result = AssessmentResult(
            student_id=student_id,
            assessment_id=assessment_id,
            score=final_score,
            total_points=total_points,
            answers=answers,
            question_scores=question_scores,
            time_spent=time.time() - start_time,
            timestamp=time.time(),
            feedback=feedback
        )

        return assessment_result

    def _update_student_profile(self, student_id: str, assessment_id: str, score: float):
        """更新学生档案"""
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = {
                'assessments': [],
                'performance': {},
                'learning_path': []
            }

        profile = self.student_profiles[student_id]
        profile['assessments'].append({
            'assessment_id': assessment_id,
            'score': score,
            'timestamp': time.time()
        })

        # 更新性能数据
        module = self._extract_module_from_assessment(assessment_id)
        if module not in profile['performance']:
            profile['performance'][module] = {
                'scores': [],
                'avg_score': 0,
                'trend': 'stable'
            }

        module_performance = profile['performance'][module]
        module_performance['scores'].append(score)
        module_performance['avg_score'] = sum(module_performance['scores']) / len(module_performance['scores'])

        # 分析趋势
        if len(module_performance['scores']) >= 3:
            recent_scores = module_performance['scores'][-3:]
            if recent_scores[-1] > recent_scores[0]:
                module_performance['trend'] = 'improving'
            elif recent_scores[-1] < recent_scores[0]:
                module_performance['trend'] = 'declining'

    def _extract_module_from_assessment(self, assessment_id: str) -> str:
        """从评估ID提取模块信息"""
        # 简化实现
        return assessment_id.split('_')[0]

    def generate_learning_recommendations(self, student_id: str) -> List[str]:
        """生成学习建议"""
        if student_id not in self.student_profiles:
            return ["请先完成一些评估测试"]

        profile = self.student_profiles[student_id]
        recommendations = []

        for module, performance in profile['performance'].items():
            avg_score = performance['avg_score']
            trend = performance['trend']

            if avg_score < 0.6:
                recommendations.append(f"建议重点复习{module}模块的基础知识")
            elif avg_score < 0.8:
                recommendations.append(f"建议加强{module}模块的练习")
            elif trend == 'declining':
                recommendations.append(f"注意{module}模块的技能保持，建议复习")

        if not recommendations:
            recommendations.append("学习表现优秀！可以尝试挑战更高难度的内容")

        return recommendations
```

---

## 📈 知识掌握度追踪

### 学习进度可视化
```python
# src/assessment/progress_tracking.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta

class LearningProgressTracker:
    """学习进度跟踪器"""

    def __init__(self):
        self.progress_data = {}
        self.mastery_levels = {
            'beginner': {'range': (0, 0.4), 'color': 'red', 'label': '初学'},
            'developing': {'range': (0.4, 0.6), 'color': 'orange', 'label': '发展中'},
            'proficient': {'range': (0.6, 0.8), 'color': 'yellow', 'label': '熟练'},
            'advanced': {'range': (0.8, 1.0), 'color': 'green', 'label': '精通'}
        }

    def record_progress(self, student_id: str, module: str, score: float,
                       timestamp: float = None):
        """记录学习进度"""
        if timestamp is None:
            timestamp = time.time()

        if student_id not in self.progress_data:
            self.progress_data[student_id] = {
                'modules': {},
                'timeline': []
            }

        student_data = self.progress_data[student_id]

        if module not in student_data['modules']:
            student_data['modules'][module] = []

        student_data['modules'][module].append({
            'score': score,
            'timestamp': timestamp
        })

        student_data['timeline'].append({
            'module': module,
            'score': score,
            'timestamp': timestamp
        })

    def get_mastery_level(self, student_id: str, module: str) -> str:
        """获取掌握水平"""
        if student_id not in self.progress_data:
            return 'beginner'

        module_scores = self.progress_data[student_id]['modules'].get(module, [])
        if not module_scores:
            return 'beginner'

        # 使用最近几次的平均分
        recent_scores = [s['score'] for s in module_scores[-3:]]
        avg_score = sum(recent_scores) / len(recent_scores)

        for level, config in self.mastery_levels.items():
            if config['range'][0] <= avg_score < config['range'][1]:
                return level

        return 'advanced'  # 如果分数 >= 0.8

    def generate_progress_report(self, student_id: str) -> Dict[str, Any]:
        """生成进度报告"""
        if student_id not in self.progress_data:
            return {'error': 'Student not found'}

        student_data = self.progress_data[student_id]
        modules = student_data['modules']

        report = {
            'student_id': student_id,
            'overall_progress': self._calculate_overall_progress(modules),
            'module_mastery': {},
            'learning_velocity': self._calculate_learning_velocity(student_data['timeline']),
            'recommendations': self._generate_progress_recommendations(modules),
            'streaks': self._calculate_learning_streaks(student_data['timeline'])
        }

        for module, scores in modules.items():
            report['module_mastery'][module] = {
                'mastery_level': self.get_mastery_level(student_id, module),
                'average_score': sum(s['score'] for s in scores) / len(scores),
                'improvement': self._calculate_improvement(scores),
                'consistency': self._calculate_consistency(scores)
            }

        return report

    def visualize_progress(self, student_id: str, save_path: str = None):
        """可视化学习进度"""
        if student_id not in self.progress_data:
            print("Student not found")
            return

        student_data = self.progress_data[student_id]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'学习进度报告 - {student_id}', fontsize=16)

        # 1. 模块掌握情况雷达图
        self._plot_mastery_radar(student_data, axes[0, 0])

        # 2. 时间线进度图
        self._plot_timeline_progress(student_data, axes[0, 1])

        # 3. 分数分布直方图
        self._plot_score_distribution(student_data, axes[1, 0])

        # 4. 学习热力图
        self._plot_learning_heatmap(student_data, axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _calculate_overall_progress(self, modules: Dict[str, List]) -> float:
        """计算整体进度"""
        if not modules:
            return 0

        all_scores = []
        for module_scores in modules.values():
            if module_scores:
                all_scores.extend([s['score'] for s in module_scores])

        return sum(all_scores) / len(all_scores) if all_scores else 0

    def _calculate_learning_velocity(self, timeline: List[Dict]) -> float:
        """计算学习速度"""
        if len(timeline) < 2:
            return 0

        # 计算单位时间内的分数提升
        sorted_timeline = sorted(timeline, key=lambda x: x['timestamp'])
        time_span = sorted_timeline[-1]['timestamp'] - sorted_timeline[0]['timestamp']
        score_improvement = sorted_timeline[-1]['score'] - sorted_timeline[0]['score']

        return score_improvement / (time_span / (24 * 3600)) if time_span > 0 else 0  # 每天提升

    def _calculate_improvement(self, scores: List[Dict]) -> float:
        """计算改进幅度"""
        if len(scores) < 2:
            return 0

        first_score = scores[0]['score']
        last_score = scores[-1]['score']
        return last_score - first_score

    def _calculate_consistency(self, scores: List[Dict]) -> float:
        """计算学习一致性"""
        if len(scores) < 3:
            return 1.0

        score_values = [s['score'] for s in scores]
        avg_score = sum(score_values) / len(score_values)
        variance = sum((s - avg_score) ** 2 for s in score_values) / len(score_values)

        # 一致性越高，方差越小
        consistency = 1 - min(variance, 1.0)
        return consistency

    def _generate_progress_recommendations(self, modules: Dict) -> List[str]:
        """生成进度建议"""
        recommendations = []

        for module, scores in modules.items():
            if len(scores) >= 3:
                recent_trend = scores[-1]['score'] - scores[-3]['score']
                if recent_trend < -0.1:
                    recommendations.append(f"{module}模块近期表现下降，建议复习")

        overall_progress = self._calculate_overall_progress(modules)
        if overall_progress < 0.6:
            recommendations.append("整体进度较慢，建议增加学习时间")

        return recommendations if recommendations else ["学习进度良好！继续保持"]

    def _calculate_learning_streaks(self, timeline: List[Dict]) -> Dict[str, int]:
        """计算学习连续天数"""
        if not timeline:
            return {'current_streak': 0, 'longest_streak': 0}

        # 转换为日期
        dates = [datetime.fromtimestamp(t['timestamp']).date() for t in timeline]
        unique_dates = sorted(list(set(dates)))

        current_streak = 0
        longest_streak = 0
        temp_streak = 0
        last_date = None

        today = datetime.now().date()

        for date in unique_dates:
            if last_date is None or (date - last_date).days == 1:
                temp_streak += 1
            else:
                temp_streak = 1

            if date == today or (today - date).days == 1:
                current_streak = temp_streak

            longest_streak = max(longest_streak, temp_streak)
            last_date = date

        return {
            'current_streak': current_streak,
            'longest_streak': longest_streak
        }

    def _plot_mastery_radar(self, student_data: Dict, ax):
        """绘制掌握情况雷达图"""
        modules = list(student_data['modules'].keys())
        scores = []

        for module in modules:
            mastery_level = self.get_mastery_level(list(self.progress_data.keys())[0], module)
            # 将掌握水平转换为数值
            level_scores = {'beginner': 0.2, 'developing': 0.5, 'proficient': 0.7, 'advanced': 0.9}
            scores.append(level_scores.get(mastery_level, 0))

        # 简化的雷达图实现
        angles = [i / len(modules) * 2 * 3.14159 for i in range(len(modules))]
        angles += angles[:1]  # 闭合图形
        scores += scores[:1]

        ax.plot(angles, scores, 'o-', linewidth=2, label='掌握水平')
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(modules)
        ax.set_ylim(0, 1)
        ax.set_title('模块掌握情况')
        ax.grid(True)

    def _plot_timeline_progress(self, student_data: Dict, ax):
        """绘制时间线进度"""
        timeline = student_data['timeline']
        if not timeline:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('学习时间线')
            return

        # 转换时间戳为日期
        dates = [datetime.fromtimestamp(t['timestamp']) for t in timeline]
        scores = [t['score'] for t in timeline]

        ax.plot(dates, scores, marker='o')
        ax.set_xlabel('日期')
        ax.set_ylabel('分数')
        ax.set_title('学习时间线')
        ax.grid(True)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_score_distribution(self, student_data: Dict, ax):
        """绘制分数分布"""
        all_scores = []
        for module_scores in student_data['modules'].values():
            all_scores.extend([s['score'] for s in module_scores])

        if not all_scores:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('分数分布')
            return

        ax.hist(all_scores, bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('分数')
        ax.set_ylabel('频次')
        ax.set_title('分数分布')
        ax.grid(True)

    def _plot_learning_heatmap(self, student_data: Dict, ax):
        """绘制学习热力图"""
        # 创建热力图数据
        modules = list(student_data['modules'].keys())
        dates = []
        data = []

        timeline = student_data['timeline']
        if not timeline:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('学习热力图')
            return

        # 按日期分组
        date_groups = {}
        for entry in timeline:
            date = datetime.fromtimestamp(entry['timestamp']).date()
            if date not in date_groups:
                date_groups[date] = {}
            date_groups[date][entry['module']] = entry['score']

        dates = sorted(date_groups.keys())

        # 构建矩阵
        matrix = []
        for date in dates:
            row = []
            for module in modules:
                score = date_groups[date].get(module, 0)
                row.append(score)
            matrix.append(row)

        # 创建热力图
        df = pd.DataFrame(matrix,
                        index=[d.strftime('%m-%d') for d in dates],
                        columns=modules)

        sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title('学习热力图')
        ax.set_xlabel('模块')
        ax.set_ylabel('日期')
```

---

## 🏆 综合能力认证

### 认证标准和流程
```python
# src/assessment/certification.py

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta

class CertificationLevel(Enum):
    """认证等级"""
    FOUNDATION = "foundation"      # 基础认证
    INTERMEDIATE = "intermediate" # 进阶认证
    ADVANCED = "advanced"         # 高级认证
    EXPERT = "expert"            # 专家认证

@dataclass
class CertificationRequirement:
    """认证要求"""
    level: CertificationLevel
    min_modules_completed: int
    min_overall_score: float
    required_projects: List[str]
    special_requirements: Dict[str, Any]
    validity_period: int  # 天数

class CertificationSystem:
    """认证系统"""

    def __init__(self):
        self.requirements = self._initialize_requirements()
        self.certifications = {}
        self.issuing_authority = "DSPy认证中心"

    def _initialize_requirements(self) -> Dict[CertificationLevel, CertificationRequirement]:
        """初始化认证要求"""
        return {
            CertificationLevel.FOUNDATION: CertificationRequirement(
                level=CertificationLevel.FOUNDATION,
                min_modules_completed=4,
                min_overall_score=0.7,
                required_projects=["基础问答系统"],
                special_requirements={
                    'coding_exercises': 20,
                    'theory_tests': 10
                },
                validity_period=365
            ),

            CertificationLevel.INTERMEDIATE: CertificationRequirement(
                level=CertificationLevel.INTERMEDIATE,
                min_modules_completed=8,
                min_overall_score=0.8,
                required_projects=["企业级RAG系统", "智能客服助手"],
                special_requirements={
                    'coding_exercises': 40,
                    'theory_tests': 20,
                    'project_evaluations': 2
                },
                validity_period=730
            ),

            CertificationLevel.ADVANCED: CertificationRequirement(
                level=CertificationLevel.ADVANCED,
                min_modules_completed=12,
                min_overall_score=0.85,
                required_projects=["复杂推理系统", "多智能体协作", "性能优化项目"],
                special_requirements={
                    'coding_exercises': 60,
                    'theory_tests': 30,
                    'project_evaluations': 4,
                    'code_review_contributions': 5
                },
                validity_period=1095
            ),

            CertificationLevel.EXPERT: CertificationRequirement(
                level=CertificationLevel.EXPERT,
                min_modules_completed=15,
                min_overall_score=0.9,
                required_projects=["创新应用项目", "开源贡献项目"],
                special_requirements={
                    'coding_exercises': 80,
                    'theory_tests': 40,
                    'project_evaluations': 6,
                    'code_review_contributions': 10,
                    'community_contributions': 3,
                    'technical_articles': 2
                },
                validity_period=1825
            )
        }

    def evaluate_certification_eligibility(self, student_id: str,
                                         level: CertificationLevel,
                                         student_data: Dict) -> Dict[str, Any]:
        """评估认证资格"""
        requirement = self.requirements[level]
        evaluation_result = {
            'eligible': False,
            'level': level.value,
            'requirement_scores': {},
            'missing_requirements': [],
            'recommendations': [],
            'estimated_completion_time': None
        }

        total_score = 0
        max_score = 0

        # 评估模块完成情况
        modules_completed = len(student_data.get('completed_modules', []))
        module_score = min(modules_completed / requirement.min_modules_completed, 1.0)
        evaluation_result['requirement_scores']['modules'] = module_score
        total_score += module_score * 0.3
        max_score += 0.3

        if modules_completed < requirement.min_modules_completed:
            evaluation_result['missing_requirements'].append(
                f"还需完成 {requirement.min_modules_completed - modules_completed} 个模块"
            )

        # 评估整体分数
        overall_score = student_data.get('overall_score', 0)
        score_score = min(overall_score / requirement.min_overall_score, 1.0)
        evaluation_result['requirement_scores']['overall_score'] = score_score
        total_score += score_score * 0.3
        max_score += 0.3

        if overall_score < requirement.min_overall_score:
            evaluation_result['missing_requirements'].append(
                f"整体分数需达到 {requirement.min_overall_score * 100:.0f} 分以上"
            )

        # 评估项目完成情况
        completed_projects = student_data.get('completed_projects', [])
        required_projects = requirement.required_projects
        project_score = len([p for p in required_projects if p in completed_projects]) / len(required_projects)
        evaluation_result['requirement_scores']['projects'] = project_score
        total_score += project_score * 0.4
        max_score += 0.4

        missing_projects = [p for p in required_projects if p not in completed_projects]
        if missing_projects:
            evaluation_result['missing_requirements'].append(
                f"还需完成项目: {', '.join(missing_projects)}"
            )

        # 评估特殊要求
        special_score = 0
        special_max = 0

        for requirement_name, required_count in requirement.special_requirements.items():
            completed_count = student_data.get(requirement_name, 0)
            item_score = min(completed_count / required_count, 1.0)
            evaluation_result['requirement_scores'][requirement_name] = item_score

            # 权重分配
            weight = 0.4 / len(requirement.special_requirements)
            special_score += item_score * weight
            special_max += weight

            if completed_count < required_count:
                evaluation_result['missing_requirements'].append(
                    f"{requirement_name}: 还需 {required_count - completed_count} 项"
                )

        total_score += special_score
        max_score += special_max

        # 最终评估
        final_score = total_score / max_score if max_score > 0 else 0
        evaluation_result['eligible'] = final_score >= 0.9  # 90%完成度
        evaluation_result['completion_percentage'] = final_score

        # 生成建议
        if not evaluation_result['eligible']:
            evaluation_result['recommendations'] = self._generate_certification_recommendations(
                evaluation_result, student_data
            )

        return evaluation_result

    def issue_certification(self, student_id: str, level: CertificationLevel,
                          student_data: Dict) -> Dict[str, Any]:
        """颁发认证"""
        eligibility = self.evaluate_certification_eligibility(student_id, level, student_data)

        if not eligibility['eligible']:
            raise ValueError("学生不符合认证要求")

        certification = {
            'certificate_id': f"DSPY-{level.value.upper()}-{student_id}-{int(time.time())}",
            'student_id': student_id,
            'level': level.value,
            'issuing_authority': self.issuing_authority,
            'issue_date': datetime.now().isoformat(),
            'expiry_date': (datetime.now() + timedelta(days=self.requirements[level].validity_period)).isoformat(),
            'verification_code': self._generate_verification_code(student_id, level),
            'skills_validated': self._get_validated_skills(level),
            'achievement_badges': self._generate_achievement_badges(student_data),
            'blockchain_hash': self._generate_blockchain_hash(student_id, level)  # 可选的区块链验证
        }

        self.certifications[student_id] = certification
        return certification

    def verify_certification(self, certificate_id: str, verification_code: str) -> Dict[str, Any]:
        """验证认证"""
        # 在实际应用中，这里会查询数据库或区块链
        for student_id, cert in self.certifications.items():
            if cert['certificate_id'] == certificate_id and cert['verification_code'] == verification_code:
                return {
                    'valid': True,
                    'certificate': cert,
                    'status': 'active' if datetime.fromisoformat(cert['expiry_date']) > datetime.now() else 'expired'
                }

        return {'valid': False, 'error': '认证信息未找到或验证码错误'}

    def _generate_verification_code(self, student_id: str, level: CertificationLevel) -> str:
        """生成验证码"""
        import hashlib
        import random

        data = f"{student_id}-{level.value}-{datetime.now().isoformat()}-{random.random()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    def _get_validated_skills(self, level: CertificationLevel) -> List[str]:
        """获取验证的技能"""
        skill_map = {
            CertificationLevel.FOUNDATION: [
                "DSPy基础概念理解", "环境配置", "基础模块使用", "简单任务实现"
            ],
            CertificationLevel.INTERMEDIATE: [
                "模块化系统设计", "RAG系统构建", "智能体开发", "性能优化"
            ],
            CertificationLevel.ADVANCED: [
                "高级架构设计", "系统优化", "多模态应用", "创新开发"
            ],
            CertificationLevel.EXPERT: [
                "技术领导力", "创新研究", "开源贡献", "社区影响"
            ]
        }
        return skill_map.get(level, [])

    def _generate_achievement_badges(self, student_data: Dict) -> List[Dict[str, str]]:
        """生成成就徽章"""
        badges = []

        # 基于不同成就生成徽章
        if student_data.get('perfect_scores', 0) > 0:
            badges.append({
                'name': '完美主义者',
                'description': '多次获得满分',
                'icon': '🏆'
            })

        if student_data.get('helpful_contributions', 0) > 10:
            badges.append({
                'name': '社区贡献者',
                'description': '积极参与社区讨论',
                'icon': '🤝'
            })

        if student_data.get('fast_learner', False):
            badges.append({
                'name': '快速学习者',
                'description': '学习进度领先',
                'icon': '🚀'
            })

        return badges

    def _generate_blockchain_hash(self, student_id: str, level: CertificationLevel) -> str:
        """生成区块链哈希（可选实现）"""
        # 简化实现，实际应用中会与区块链平台集成
        import hashlib
        data = f"{student_id}-{level.value}-{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _generate_certification_recommendations(self, eligibility: Dict,
                                              student_data: Dict) -> List[str]:
        """生成认证建议"""
        recommendations = []

        # 基于缺失要求生成建议
        missing = eligibility.get('missing_requirements', [])
        if missing:
            recommendations.append("优先完成以下要求:")
            recommendations.extend([f"  • {req}" for req in missing[:3]])

        # 基于完成度生成时间估算
        completion_percentage = eligibility.get('completion_percentage', 0)
        if completion_percentage < 0.5:
            recommendations.append("预计需要2-3个月达到认证要求")
        elif completion_percentage < 0.8:
            recommendations.append("预计需要1-2个月达到认证要求")
        else:
            recommendations.append("预计需要2-4周达到认证要求")

        # 个性化建议
        overall_score = student_data.get('overall_score', 0)
        if overall_score < 0.7:
            recommendations.append("建议重点提升理论知识和基础技能")

        if student_data.get('completed_projects', 0) < 2:
            recommendations.append("建议增加项目实践经验")

        return recommendations

class LearningPathRecommender:
    """学习路径推荐器"""

    def __init__(self, certification_system: CertificationSystem):
        self.cert_system = certification_system
        self.learning_paths = self._initialize_learning_paths()

    def _initialize_learning_paths(self) -> Dict[str, List[Dict]]:
        """初始化学习路径"""
        return {
            'beginner': [
                {'step': 1, 'module': 'DSPy基础', 'duration': 1, 'resources': ['官方文档', '基础教程']},
                {'step': 2, 'module': '签名系统', 'duration': 1, 'resources': ['实践练习', '示例代码']},
                {'step': 3, 'module': '预测模块', 'duration': 1, 'resources': ['视频教程', '编程练习']},
                {'step': 4, 'module': '简单项目', 'duration': 1, 'resources': ['项目模板', '指导文档']}
            ],
            'intermediate': [
                {'step': 1, 'module': '高级模块', 'duration': 2, 'resources': ['进阶教程', '案例研究']},
                {'step': 2, 'module': 'RAG系统', 'duration': 2, 'resources': ['实战项目', '最佳实践']},
                {'step': 3, 'module': '智能体开发', 'duration': 2, 'resources': ['开发指南', '工具文档']},
                {'step': 4, 'module': '系统优化', 'duration': 1, 'resources': ['性能调优', '监控工具']}
            ]
        }

    def recommend_learning_path(self, student_id: str, target_level: CertificationLevel,
                              student_data: Dict) -> Dict[str, Any]:
        """推荐学习路径"""
        current_level = self._assess_current_level(student_data)

        recommendation = {
            'student_id': student_id,
            'target_level': target_level.value,
            'current_level': current_level,
            'estimated_duration': self._estimate_duration(current_level, target_level),
            'learning_path': [],
            'milestones': [],
            'resources': self._recommend_resources(current_level, target_level),
            'success_probability': self._calculate_success_probability(student_data, target_level)
        }

        # 生成具体学习路径
        path = self._generate_path(current_level, target_level, student_data)
        recommendation['learning_path'] = path

        # 生成里程碑
        recommendation['milestones'] = self._generate_milestones(path)

        return recommendation

    def _assess_current_level(self, student_data: Dict) -> str:
        """评估当前水平"""
        modules_completed = len(student_data.get('completed_modules', []))
        overall_score = student_data.get('overall_score', 0)

        if modules_completed < 4 or overall_score < 0.7:
            return 'beginner'
        elif modules_completed < 8 or overall_score < 0.8:
            return 'intermediate'
        elif modules_completed < 12 or overall_score < 0.85:
            return 'advanced'
        else:
            return 'expert'

    def _estimate_duration(self, current_level: str, target_level: CertificationLevel) -> Dict[str, int]:
        """估算学习时长"""
        level_order = ['beginner', 'intermediate', 'advanced', 'expert']
        current_index = level_order.index(current_level)
        target_index = list(CertificationLevel).index(target_level)

        weeks_needed = (target_index - current_index) * 4  # 每个级别4周

        return {
            'minimum_weeks': weeks_needed // 2,
            'expected_weeks': weeks_needed,
            'maximum_weeks': weeks_needed * 2
        }

    def _generate_path(self, current_level: str, target_level: CertificationLevel,
                      student_data: Dict) -> List[Dict]:
        """生成学习路径"""
        # 这里简化实现，实际应用中会更复杂
        base_path = self.learning_paths.get(current_level, [])

        # 根据学生数据调整路径
        adjusted_path = []
        for step in base_path:
            # 检查是否已完成
            if step['module'] in student_data.get('completed_modules', []):
                step['status'] = 'completed'
            else:
                step['status'] = 'pending'

            adjusted_path.append(step)

        return adjusted_path

    def _generate_milestones(self, path: List[Dict]) -> List[Dict]:
        """生成学习里程碑"""
        milestones = []
        cumulative_duration = 0

        for i, step in enumerate(path):
            cumulative_duration += step['duration']
            if i % 2 == 1:  # 每两个步骤一个里程碑
                milestones.append({
                    'milestone': f"完成第{i+1}个学习阶段",
                    'estimated_week': cumulative_duration,
                    'goal': f"掌握{step['module']}相关技能"
                })

        return milestones

    def _recommend_resources(self, current_level: str, target_level: CertificationLevel) -> List[Dict]:
        """推荐学习资源"""
        resources = []

        # 基础资源
        resources.extend([
            {'type': 'documentation', 'title': 'DSPy官方文档', 'priority': 'high'},
            {'type': 'tutorial', 'title': '入门视频教程', 'priority': 'high'},
            {'type': 'practice', 'title': '编程练习平台', 'priority': 'medium'}
        ])

        # 根据目标级别添加资源
        if target_level in [CertificationLevel.ADVANCED, CertificationLevel.EXPERT]:
            resources.extend([
                {'type': 'research', 'title': '最新研究论文', 'priority': 'medium'},
                {'type': 'community', 'title': '开源项目贡献', 'priority': 'low'}
            ])

        return resources

    def _calculate_success_probability(self, student_data: Dict,
                                    target_level: CertificationLevel) -> float:
        """计算成功概率"""
        factors = {
            'current_score': student_data.get('overall_score', 0) / 100,
            'learning_consistency': student_data.get('consistency_score', 0.5),
            'time_availability': student_data.get('weekly_study_hours', 10) / 20,
            'previous_success_rate': student_data.get('completion_rate', 0.8)
        }

        # 加权计算
        weights = [0.3, 0.2, 0.3, 0.2]
        probability = sum(f * w for f, w in zip(factors.values(), weights))

        return min(probability, 1.0)
```

这套完整的教学评估体系为DSPy课程提供了：

1. **阶段性评估**：明确的学习目标和检测标准
2. **实时评估系统**：自动评分和自适应测试
3. **进度追踪**：可视化学习进度和掌握度
4. **能力认证**：多级认证体系和学习路径推荐

所有组件都可以直接集成到在线学习平台中，为学习者提供全面的评估和指导。