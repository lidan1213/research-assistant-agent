"""app/evaluation 包：评测任务持久化与历史对比。"""
from app.evaluation.store import EvaluationStore, get_evaluation_store  # noqa: F401
from app.evaluation.answer_eval import evaluate_answer  # noqa: F401
