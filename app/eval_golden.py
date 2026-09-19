"""Agent 回归评测集（Golden Set）：固定问题验证回答质量，防止行为退化。

共享模块：scripts/eval_golden.py（CLI）与管理端 /api/admin/evaluate 都用它。
运行需要真实 LLM（ruoli/DeepSeek），联网用例会调 web_search，耗时较长。
"""
from __future__ import annotations

from app.memory.stores.sqlite import SQLiteStore
from app.api.deps import get_agent

# 固定问题集：问题 -> (类别, [关键断言])
GOLDEN_CASES: list[dict] = [
    # --- 纯推理/计算（calculator） ---
    {"category": "calculator", "question": "请计算 23 乘以 17 等于多少？", "must_contain": ["391"]},
    {"category": "calculator", "question": "125 除以 5 的结果是多少？", "must_contain": ["25"]},
    # --- 知识库检索（kb） ---
    {"category": "kb", "question": "量子点是什么？它的尺寸范围是多少？", "must_contain": ["量子点", "纳米"]},
    # --- 联网搜索（search，可选） ---
    {"category": "search", "question": "2024 年诺贝尔物理学奖颁给了谁？", "must_contain": ["霍普菲尔德", "辛顿"], "use_web": True},
    # --- 普通问答（plain） ---
    {"category": "plain", "question": "简要解释什么是检索增强生成（RAG）？", "must_contain": ["检索", "生成"]},
]


async def run_case(case: dict, session_id: str) -> tuple[bool, str]:
    """运行单个用例，返回 (通过?, 回答摘要)。"""
    agent = get_agent(use_web=case.get("use_web", False))
    try:
        answer = await agent.run(session_id, case["question"], use_plan=False)
    except Exception as e:  # noqa: BLE001
        return False, f"执行异常: {e}"
    missing = [k for k in case["must_contain"] if k not in answer]
    if missing:
        return False, f"缺少关键内容: {missing} | 回答: {answer[:120]}"
    return True, answer[:120]


async def run_all(db_path: str, category: str | None = None) -> dict:
    """跑全量（或指定类别）Golden Set，返回报告 dict。"""
    cases = [c for c in GOLDEN_CASES if not category or c["category"] == category]
    store = SQLiteStore(db_path)
    results: list[dict] = []
    try:
        for i, case in enumerate(cases, 1):
            sid = f"golden__{case['category']}_{i}"
            try:
                await store.clear(sid)
            except Exception:  # noqa: BLE001
                pass
            ok, detail = await run_case(case, sid)
            results.append(
                {
                    "category": case["category"],
                    "question": case["question"],
                    "must_contain": case["must_contain"],
                    "ok": ok,
                    "detail": detail,
                }
            )
    finally:
        store.close()
    passed = sum(1 for r in results if r["ok"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "results": results,
    }
