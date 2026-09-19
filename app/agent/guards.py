"""Agent 护栏（Guards）：纯函数集合，Agent 主循环之外的决策规则。

- SimpleTaskDetector：判断任务是否简单（≤60 字、无图片、无复杂意图）→ 可用便宜模型直答
- BudgetGuard：token / 迭代预算检查，防止上下文膨胀与死循环

设计：纯函数、无 I/O、无状态，便于单测与复用（多 Agent worker 共用同一套规则）。
"""
from __future__ import annotations

from typing import Any

# 复杂意图关键词：命中即需要主模型 + 工具链
HEAVY_KEYWORDS = (
    "检索", "搜索", "知识库", "总结", "分析", "比较", "对比", "写", "生成",
    "规划", "翻译", "代码", "论文", "公式", "计算", "多少", "为什么", "如何",
)


def recommended_tool(message: str) -> str:
    """根据明确意图给出工具路由建议；空字符表示交由模型判断。

    这不直接执行工具，只用于给 ReAct 一个可解释的路由约束。
    """
    text = (message or "").strip().lower()
    if not text:
        return ""
    if "arxiv" in text or any(k in text for k in ("搜索论文", "查找论文", "检索论文")):
        return "arxiv_search"
    if "python" in text or any(k in text for k in ("用代码", "运行代码", "编程计算")):
        return "code_executor"
    if any(k in text for k in ("计算", "等于多少", "乘以", "除以")):
        return "calculator"
    if any(k in text for k in (
        "知识库", "rag", "react", "self-rag", "graphrag", "llama",
        "bm25", "rrf", "reranker", "recall", "precision", "mrr", "ndcg",
        "embedding", "分块", "overlap", "supervisor", "黑板", "多 agent",
        "大语言模型", "智能体", "agent",
    )):
        return "knowledge_search"
    return ""


def tool_matches_recommendation(recommended: str, used_tools: list[str]) -> bool:
    if not recommended:
        return True
    acceptable = {recommended}
    if recommended == "knowledge_search":
        acceptable.update({"kg_query", "arxiv_search"})
    return any(name in acceptable for name in used_tools)


def answer_needs_revision(answer: str) -> bool:
    """仅拦截明显无效的最终答案，避免对简短数值答案误判。"""
    text = (answer or "").strip()
    if not text:
        return True
    return any(marker in text for marker in (
        "未能生成有效回答", "因达到运行限制而中断", "请换一种问法重试",
    ))


def is_simple_task(message: str, images: list[str] | None = None) -> bool:
    """简单任务预判：≤60 字、无图片、不含复杂意图关键词 → 可用便宜模型直答。

    关键词覆盖需要检索/分析/写作/计算的场景——这些必须走主模型 + 工具链。
    """
    if images:
        return False
    m = (message or "").strip()
    if not m or len(m) > 60:
        return False
    # 确定性工具路由必须先于“简单问题直答”。否则像“RAG 的原理是什么”
    # 这类短问题会被辅助模型直接回答，绕过 knowledge_search，导致答案虽对
    # 却没有知识库依据。只要已有明确工具建议，就进入完整 Agent 工具链。
    if recommended_tool(m):
        return False
    return not any(k in m for k in HEAVY_KEYWORDS)


def estimate_tokens(messages: list[Any], chars_per_token: int = 3) -> int:
    """近似 token 估算：按字符数 / chars_per_token（中文约 1.5-2 字符/token，取 3 保守）。"""
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total += len(str(content)) // chars_per_token + 4  # 每消息固定开销
        tc = getattr(m, "tool_calls", None)
        if tc:
            total += sum(len(str(t.arguments or "")) // chars_per_token for t in tc)
    return total


def over_budget(messages: list[Any], token_budget: int) -> bool:
    """是否超出 token 预算。"""
    if token_budget <= 0:
        return False
    return estimate_tokens(messages) > token_budget
