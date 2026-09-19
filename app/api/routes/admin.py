"""管理员路由：查看学生列表、各学生的对话会话与对话记录（仅 admin）。"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.memory.stores.sqlite import SQLiteStore
from app.config import get_settings
from app.core.auth import ROLE_STUDENT, User, UserStore, require_admin
from app.core.exceptions import ForbiddenError
from app.services.admin_metrics import aggregate_cost_trend, aggregate_retrieval_stats
from app.services.admin_students import admin_student_service

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class AnswerEvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    use_llm_judge: bool = False


@router.post("/answer-evaluate")
async def answer_evaluate(req: AnswerEvaluationRequest) -> dict:
    """答案正确性与忠实度评测；可选 LLM Judge，失败时自动保留确定性结果。"""
    from app.evaluation.answer_eval import evaluate_answer

    judge = None
    if req.use_llm_judge:
        from app.llm.factory import get_llm

        judge = get_llm()
    return await evaluate_answer(
        req.question,
        req.answer,
        req.contexts,
        req.reference_answer,
        judge=judge,
    )


@router.get("/cost-trend")
async def cost_trend(days: int = Query(14, ge=1, le=90)) -> dict:
    """成本趋势：从 traces(kind=llm) 按天聚合成本/请求/token，并按学生拆分。

    成本按 usage.py 内置单价表估算（detail 中携带 prompt=/completion= token 数）。
    """
    from app.llm.trace import get_trace_ledger

    rows = get_trace_ledger().query(limit=5000, kind="llm")
    return aggregate_cost_trend(rows, days)


@router.get("/retrieval-stats")
async def retrieval_stats(days: int = Query(30, ge=1, le=90)) -> dict:
    """检索质量看板：从 traces(kind=rag) 聚合检索次数/失败率/零命中率/文档命中排行/最近检索。

    detail 格式（retriever.py 写入）：query='...' top=<score> hits=[...]
    """
    from app.llm.trace import get_trace_ledger

    rows = get_trace_ledger().query(limit=5000, kind="rag")
    return aggregate_retrieval_stats(rows, days)


# ---------- RAG qrels 检索准确度评测（后台任务 + 轮询 + 持久化） ----------
_retrieval_eval_tasks: dict[str, dict] = {}
_retrieval_eval_lock = threading.Lock()


def _start_persisted_task(run_id: str, kind: str, dataset: str, config: dict | None = None) -> None:
    """落盘任务记录：服务重启后仍可通过 /evaluation-runs 查询历史。"""
    try:
        from app.evaluation import get_evaluation_store

        get_evaluation_store().start(run_id, kind, dataset, config)
    except Exception:  # noqa: BLE001
        pass  # 存储失败不影响评测本身


def _finish_persisted_task(run_id: str, report: dict, status: str = "done") -> None:
    try:
        from app.evaluation import get_evaluation_store

        get_evaluation_store().finish(run_id, report, status)
    except Exception:  # noqa: BLE001
        pass


@router.post("/retrieval-evaluate")
async def start_retrieval_evaluate(dataset: str = Query("extended", pattern="^(baseline|extended|chinese)$")) -> dict:
    """启动 qrels 检索评测，计算 Recall/Precision/HitRate/MRR/nDCG。

    dataset=baseline 使用 5 条基准查询；dataset=extended 使用扩展语料的 11 条查询；
    dataset=chinese 使用中文评测集（存在时）。
    """
    import asyncio as _asyncio
    import uuid as _uuid

    task_id = _uuid.uuid4().hex[:12]
    _start_persisted_task(task_id, "retrieval", dataset, {"k_list": [1, 3, 5]})
    with _retrieval_eval_lock:
        _retrieval_eval_tasks[task_id] = {"status": "running", "report": None, "dataset": dataset}

    async def _run() -> None:
        try:
            from app.api.deps import get_retriever
            from app.eval_retrieval import evaluate_from_file

            qrels_name = {
                "baseline": "retrieval_qrels.json",
                "extended": "extended_retrieval_qrels.json",
                "chinese": "chinese_retrieval_qrels.json",
            }[dataset]
            qrels = Path(get_settings().memory.sqlite_path).parent / "knowledge" / qrels_name
            report = await evaluate_from_file(get_retriever(), qrels, [1, 3, 5])
            report["dataset_name"] = dataset
            _finish_persisted_task(task_id, report, "done")
        except Exception as e:  # noqa: BLE001
            report = {"error": str(e), "query_count": 0, "k_list": [1, 3, 5], "aggregate": {}, "per_query": []}
            _finish_persisted_task(task_id, report, "failed")
        with _retrieval_eval_lock:
            _retrieval_eval_tasks[task_id] = {"status": "done", "report": report}

    _asyncio.create_task(_run())
    return {"task_id": task_id, "dataset": dataset}


@router.get("/retrieval-evaluate/{task_id}")
async def get_retrieval_evaluate(task_id: str) -> dict:
    """查询 qrels 检索评测任务（内存优先，重启后从持久化存储读取）。"""
    with _retrieval_eval_lock:
        task = _retrieval_eval_tasks.get(task_id)
    if task is None:
        try:
            from app.evaluation import get_evaluation_store

            saved = get_evaluation_store().get(task_id)
            if saved is not None:
                return {"status": "done" if saved["status"] == "done" else saved["status"],
                        "report": saved["report"] or {"error": saved["error"] or "任务中断"}}
        except Exception:  # noqa: BLE001
            pass
        return {"status": "not_found"}
    return {"status": task["status"], **( {"report": task["report"]} if task["status"] == "done" else {})}


@router.get("/evaluation-runs")
async def evaluation_runs(kind: str | None = Query(None, pattern="^(retrieval|golden|agent|answer)$"),
                          limit: int = Query(20, ge=1, le=100)) -> dict:
    """评测历史（持久化）：按类型列最近运行，供管理端对比。"""
    try:
        from app.evaluation import get_evaluation_store

        runs = get_evaluation_store().list(kind=kind, limit=limit)
        return {"total": len(runs), "runs": runs}
    except Exception as e:  # noqa: BLE001
        return {"total": 0, "runs": [], "error": str(e)}


@router.get("/audit")
async def audit_log(days: int = Query(30, ge=1, le=90)) -> dict:
    """内容审计：敏感词命中 / 越权尝试日志（traces kind=audit）。"""
    import json as _json
    import time as _time
    from collections import Counter
    from datetime import datetime

    from app.llm.trace import get_trace_ledger

    rows = get_trace_ledger().query(limit=5000, kind="audit")
    cutoff = _time.time() - days * 86400
    items: list[dict] = []
    word_counter: Counter = Counter()
    for r in rows:
        if r["created_at"] < cutoff:
            continue
        try:
            detail = _json.loads(r["detail"] or "{}")
        except Exception:  # noqa: BLE001
            detail = {}
        label = r["name"] or ""
        if label.startswith("sensitive:"):
            word = detail.get("word") or ""
            if word:
                word_counter[word] += 1
        items.append(
            {
                "time": datetime.fromtimestamp(r["created_at"]).strftime("%m-%d %H:%M"),
                "label": label,
                "detail": detail,
            }
        )
    items.sort(key=lambda x: x["time"], reverse=True)
    return {
        "days": days,
        "total": len(items),
        "sensitive_total": sum(word_counter.values()),
        "top_words": [{"word": k, "count": v} for k, v in word_counter.most_common(20)],
        "items": items[:50],
    }


# ---------- Golden Set 一键评测（后台任务 + 轮询） ----------
_eval_tasks: dict[str, dict] = {}
_eval_lock = threading.Lock()


@router.post("/evaluate")
async def start_evaluate() -> dict:
    """启动 Golden Set 评测（后台异步执行，耗时 1-5 分钟）。"""
    import uuid as _uuid

    task_id = _uuid.uuid4().hex[:12]
    _start_persisted_task(task_id, "golden", "golden_set", {})
    with _eval_lock:
        _eval_tasks[task_id] = {"status": "running", "report": None}

    async def _run() -> None:
        try:
            from app.eval_golden import run_all

            report = await run_all(
                str(Path(get_settings().memory.sqlite_path).parent / "golden_eval.db")
            )
            _finish_persisted_task(task_id, report, "done")
        except Exception as e:  # noqa: BLE001
            report = {"error": str(e), "total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "results": []}
            _finish_persisted_task(task_id, report, "failed")
        with _eval_lock:
            _eval_tasks[task_id] = {"status": "done", "report": report}

    import asyncio

    asyncio.create_task(_run())
    return {"task_id": task_id}


@router.get("/evaluate/{task_id}")
async def get_evaluate(task_id: str) -> dict:
    """查询评测任务状态/结果（内存优先，重启后从持久化存储读取）。"""
    with _eval_lock:
        task = _eval_tasks.get(task_id)
    if task is None:
        try:
            from app.evaluation import get_evaluation_store

            saved = get_evaluation_store().get(task_id)
            if saved is not None:
                return {"status": "done" if saved["status"] == "done" else saved["status"],
                        "report": saved["report"] or {"error": saved["error"] or "任务中断"}}
        except Exception:  # noqa: BLE001
            pass
        return {"status": "not_found"}
    if task["status"] == "running":
        return {"status": "running"}
    return {"status": "done", "report": task["report"]}


@router.get("/usage")
async def usage() -> dict:
    """LLM 用量与成本总览（token / 请求数 / 估算成本，按模型分账）。"""
    from app.llm.usage import get_usage_ledger

    return get_usage_ledger().summary()


@router.get("/traces")
async def traces(
    session_id: str | None = None,
    limit: int = 200,
) -> dict:
    """会话级链路追踪：按会话查 LLM/工具调用明细（耗时/成败），session_id 空=最近全部。"""
    from app.llm.trace import get_trace_ledger

    ledger = get_trace_ledger()
    rows = ledger.query(session_id, limit=min(limit, 1000))
    return {"total": len(rows), "traces": rows}


@router.get("/traces/by-trace/{trace_id}")
async def trace_chain(trace_id: str) -> dict:
    """按 trace_id 查询一次请求的完整调用链（API → 模型路由 → LLM → 工具 → 检索）。

    回答「为什么慢 / 调了哪些模型 / 用了哪些工具 / 失败在哪一步」。
    """
    from app.llm.trace import get_trace_ledger

    ledger = get_trace_ledger()
    chain = ledger.query_by_trace(trace_id)
    if not chain:
        return {"trace_id": trace_id, "total": 0, "chain": []}
    chain.sort(key=lambda r: r["id"])
    # 聚合摘要：模型调用次数/工具列表/总耗时/失败点
    models = {}
    tools = []
    total_ms = 0
    failures = []
    for r in chain:
        total_ms += r.get("duration_ms") or 0
        if r["kind"] == "llm":
            models[r["name"]] = models.get(r["name"], 0) + 1
        elif r["kind"] == "tool" and r["name"] not in tools:
            tools.append(r["name"])
        if not r.get("success"):
            failures.append({"name": r["name"], "kind": r["kind"], "detail": (r.get("detail") or "")[:120]})
    return {
        "trace_id": trace_id,
        "total": len(chain),
        "total_ms": total_ms,
        "models": models,
        "tools": tools,
        "failures": failures,
        "chain": chain,
    }


@router.get("/citation-coverage")
async def citation_coverage_stats(limit: int = Query(100, ge=1, le=1000)) -> dict:
    """从 answer Trace 的 detail 中聚合引用覆盖率。"""
    import json as _json
    from app.eval_citations import citation_coverage
    from app.llm.trace import get_trace_ledger

    rows = get_trace_ledger().query(limit=limit, kind="answer")
    reports = []
    for row in rows:
        try:
            data = _json.loads(row["detail"] or "{}")
            reports.append({"trace_id": row.get("trace_id", ""), **citation_coverage(data.get("answer", ""), int(data.get("source_count", 0)))})
        except Exception:
            continue
    avg = round(sum(r["citation_coverage"] for r in reports) / len(reports), 4) if reports else 0.0
    return {"total": len(reports), "avg_coverage": avg, "reports": reports[:50]}


@router.get("/traces/{session_id}")
async def trace_summary(session_id: str) -> dict:
    """某会话的调用统计（LLM/工具次数、失败数、总耗时）。"""
    from app.llm.trace import get_trace_ledger

    return get_trace_ledger().session_summary(session_id)


@router.get("/dashboard")
async def dashboard() -> dict:
    """运营总览：用户数 / 会话数 / 消息数 / 笔记数 / 知识库文档数。"""
    from pathlib import Path

    # 用户数
    users = UserStore().list_by_role(ROLE_STUDENT)
    from app.core.auth import ROLE_ADMIN

    admins = UserStore().list_by_role(ROLE_ADMIN)
    total_users = len(admins) + len(users)
    # 会话 / 消息数（memory.db）
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        memory_stats = store.statistics()
        session_count = memory_stats["sessions"]
        message_count = memory_stats["messages"]
    finally:
        store.close()
    # 笔记数
    notes_dir = Path("./data/notes")
    note_count = len(list(notes_dir.glob("*.md"))) if notes_dir.exists() else 0
    # 知识库文档数（ChromaDB，尽力而为）
    kb_docs = -1  # -1 表示查询失败/未配置
    try:
        from app.api.deps import get_retriever

        retriever = get_retriever()
        kb_docs = retriever.count()
    except Exception:  # noqa: BLE001
        kb_docs = -1
    return {
        "total_users": total_users,
        "student_count": len(users),
        "session_count": session_count,
        "message_count": message_count,
        "note_count": note_count,
        "kb_documents": kb_docs,
    }


class StudentSummary(BaseModel):
    username: str
    session_count: int


class SessionSummary(BaseModel):
    session_id: str
    message_count: int
    first_message: str
    last_message: str


class SessionDetail(BaseModel):
    session_id: str
    username: str
    messages: list[dict]


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=4, max_length=64, description="新密码（至少 4 位）")


@router.get("/students", response_model=list[StudentSummary])
async def list_students(_: User = Depends(require_admin)) -> list[StudentSummary]:
    """列出全部学生账号及各自的会话数。"""
    return [StudentSummary(**row) for row in admin_student_service.list_students()]


@router.post("/students/{username}/reset-password")
async def reset_student_password(
    username: str, req: ResetPasswordRequest, _: User = Depends(require_admin)
) -> dict:
    """管理员重置学生密码（学生忘记密码时用）。"""
    admin_student_service.reset_password(username, req.new_password)
    return {"username": username, "message": "密码已重置"}


@router.get("/students/{username}/sessions", response_model=list[SessionSummary])
async def list_student_sessions(username: str, _: User = Depends(require_admin)) -> list[SessionSummary]:
    """某学生的会话列表（按最后活动倒序）。"""
    sessions = admin_student_service.list_sessions(username)
    return [
        SessionSummary(
            session_id=s["session_id"],
            message_count=s["message_count"],
            first_message=s["first_message"],
            last_message=s["last_message"],
        )
        for s in sessions
    ]


@router.get("/students/{username}/sessions/{session_id}", response_model=SessionDetail)
async def get_session_detail(
    username: str, session_id: str, _: User = Depends(require_admin)
) -> SessionDetail:
    """某学生某个会话的完整对话记录。"""
    return SessionDetail(
        session_id=session_id,
        username=username,
        messages=await admin_student_service.session_detail(username, session_id),
    )


# ---------- Redis 面板（L2 热数据层） ----------
@router.get("/redis")
async def redis_status() -> dict:
    """Redis 状态：连接可用性 / 记忆 key 数 / 检索缓存 / 排行榜。"""
    from app.cache.redis_client import get_redis, redis_available
    from app.cache.ranking import rank_top
    from app.cache.ratelimit import fixed_window

    available = await redis_available()
    info: dict = {
        "available": available,
        "memory_keys": 0,
        "cache_keys": 0,
        "rank_keys": 0,
    }
    if available:
        try:
            r = get_redis()
            keys = await r.keys("*")
            info["memory_keys"] = sum(1 for k in keys if k.startswith("agent:"))
            info["cache_keys"] = sum(1 for k in keys if k.startswith("rag:"))
            info["rank_keys"] = sum(1 for k in keys if k.startswith("rank:"))
            info["total_keys"] = len(keys)
            info["db_size"] = await r.dbsize()
        except Exception:  # noqa: BLE001
            pass
    info["hot_queries"] = await rank_top("hot_query", 10)
    info["top_docs"] = await rank_top("doc_hit", 10)
    # 自检限流链路（不真正限流：limit 极大）
    ok, _ = await fixed_window("selfcheck", "admin", limit=10**9, window=60)
    info["ratelimit_ok"] = ok
    return info


@router.get("/redis/rankings")
async def redis_rankings(
    kind: str = Query("hot_query", pattern="^(hot_query|doc_hit)$"),
    n: int = Query(20, ge=1, le=100),
) -> dict:
    """排行榜明细：hot_query=检索热词 / doc_hit=文档命中排行。"""
    from app.cache.ranking import rank_count, rank_top

    return {
        "kind": kind,
        "total_members": await rank_count(kind),
        "items": await rank_top(kind, n),
    }
