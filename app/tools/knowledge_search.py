"""knowledge_search 工具：Agent 对话时检索「当前用户」的个人知识库。

- 通过 ContextVar 感知当前登录用户（WS 层在收到消息时设置）；
- 未设置用户（如直接脚本调用）时检索全局知识库（collection=knowledge）；
- 检索结果带来源文件名，便于 Agent 引用与回答溯源。
"""
from __future__ import annotations

from contextvars import ContextVar

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.tools.base import ToolResult, tool

logger = get_logger("tools.knowledge_search")

# 当前登录用户（WS 层设置）。未设置时回退全局知识库。
current_username: ContextVar[str | None] = ContextVar("current_username", default=None)

# 当前对话选择的知识库名（WS 层设置）。None=默认库；"none"=不使用知识库。
current_kb_name: ContextVar[str | None] = ContextVar("current_kb_name", default=None)

# 全局知识库检索器（惰性创建，供未登录场景）
_global_retriever = None

# 查询改写缓存：query -> rewritten（进程内，避免同一问题重复调用 LLM）
_query_cache: dict[str, str] = {}


async def _rewrite_query(query: str) -> str:
    """检索前用 LLM 改写查询：学术场景扩展术语/中英互补，提升召回。

    - 缓存相同 query 的改写结果（进程内）；
    - LLM 失败/超时回退原查询，绝不影响检索主流程。
    """
    cached = _query_cache.get(query)
    if cached is not None:
        return cached
    normalized = (query or "").strip()
    # 短且包含明确学术术语的查询直接检索：额外 LLM 改写往往
    # 不改变意图，却会增加一次网络往返和长尾失败点。
    explicit_terms = (
        "rag", "react", "self-rag", "graphrag", "llama", "bm25", "rrf",
        "reranker", "recall", "precision", "mrr", "ndcg", "embedding",
        "supervisor", "overlap",
    )
    if len(normalized) <= 48 and any(term in normalized.lower() for term in explicit_terms):
        _query_cache[query] = normalized
        return normalized
    try:
        from app.llm.base import ChatMessage, MessageRole
        from app.llm.gateway import get_llm_gateway
        from app.llm.router import TaskType  # 查询改写属内部低价值调用，路由到辅助模型

        resp = await get_llm_gateway().chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "你是学术检索助手。把下面的查询改写成更适合知识库检索的关键词组合："
                        "中英文术语互补、补充同义词/上位概念，最多 20 个字，"
                        "只输出改写结果，不要解释、不要引号。\n\n"
                        f"原始查询：{query}"
                    ),
                )
            ],
            tools=None,
            temperature=0.0,
            task_type=TaskType.QUERY_REWRITE,
        )
        rewritten = (resp.content or "").strip().strip('"\'')
        if rewritten and len(rewritten) <= 60 and rewritten != query:
            _query_cache[query] = rewritten
            logger.info("查询改写: %s -> %s", query, rewritten)
            return rewritten
    except Exception as e:  # noqa: BLE001
        logger.debug("查询改写失败（回退原查询）: %s", e)
    _query_cache[query] = query  # 失败也缓存，避免反复尝试
    return query


def _get_global_retriever():
    global _global_retriever
    if _global_retriever is None:
        from app.knowledge.retriever import Retriever

        _global_retriever = Retriever()
    return _global_retriever


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(..., description="要检索的问题或关键词")
    top_k: int = Field(3, ge=1, le=10, description="返回的文档块数量")


@tool("knowledge_search", "检索用户自己的知识库（上传过的资料），返回相关文档片段；找不到或库为空时返回提示", KnowledgeSearchArgs)
async def knowledge_search(query: str, top_k: int = 3) -> ToolResult:
    username = current_username.get()
    kb_name = current_kb_name.get()
    # 查询改写：扩展学术术语提升召回（失败回退原查询）
    search_query = await _rewrite_query(query)
    try:
        if not username:
            hits = await _get_global_retriever().retrieve(search_query, top_k=top_k)
            source_label = "全局知识库"
        elif kb_name == "none":
            # 用户选择"不使用知识库"：直接返回提示，不检索任何库
            return ToolResult(
                success=True,
                output="[知识库未启用] 本次对话选择了「不使用知识库」，未检索任何资料。"
                "如需检索，请在上方选择具体知识库。",
            )
        else:
            from app.knowledge.user_kb import UserKnowledgeBase

            kb = UserKnowledgeBase(username, kb_name)
            hits = await kb.search(search_query, top_k=top_k)  # 内置 0.35 相似度阈值
            source_label = f"用户 {username} 的知识库「{kb_name or '默认'}」"
        if not hits:
            return ToolResult(
                success=True,
                output=f"[{source_label}] 没有找到与「{query}」相关的内容。"
                "可以提示用户上传相关文件到个人知识库，或尝试换一个查询词。",
            )
        # 组装成 Agent 易读的文本：来源 + 相似度 + 片段
        lines = [f"检索来源：{source_label}，命中 {len(hits)} 条："]
        if any(h.get("low_confidence") for h in hits):
            lines.append("⚠️ 检索置信度较低：以下内容仅作参考，当前知识库可能缺少与问题直接匹配的资料。")
        from app.rag.evidence import analyze_evidence

        evidence_report = analyze_evidence(hits, query)
        if evidence_report["has_conflict"]:
            lines.append(
                "⚠️ [EVIDENCE_CONFLICT] 检索资料存在同条件证据冲突。最终回答必须并列说明各方结论及来源；"
                "只能依据下列可审计的来源质量信息作暂定裁决，不得按相似度静默选边。"
            )
            for group in evidence_report["conflicts"]:
                decision = group["reason"]
                if group.get("preferred_member"):
                    decision += f"；暂时倾向来源 [{group['preferred_member']}]（不是事实定论）"
                lines.append(
                    f"- {group['id']}：来源编号 {group['members']}；"
                    + "；".join(group["reasons"])
                    + f"；裁决状态={group['status']}；{decision}"
                )
        if evidence_report["has_contextual_difference"]:
            lines.append(
                "ℹ️ [EVIDENCE_CONTEXT_DIFFERENCE] 文档结论表面不同，但实验条件不一致。"
                "回答时应解释适用边界，不得把它表述为同条件下的直接矛盾。"
            )
            for group in evidence_report["contextual_differences"]:
                lines.append(
                    f"- {group['id']}：来源编号 {group['members']}；"
                    + "；".join(group["condition_differences"])
                    + f"；{group['reason']}"
                )
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata") or {}
            src = meta.get("source") or meta.get("doc_id") or "未知来源"
            text = (h.get("text") or "").strip()
            # 保留合理长度，避免 observation 超长被截断
            if len(text) > 800:
                text = text[:800] + "…"
            claim = evidence_report["claims"][i - 1]
            conditions = ", ".join(
                f"{key}={value}" for key, value in claim["conditions"].items() if value is not None
            ) or "未报告"
            quality = claim["quality"]
            version_label = (
                f"v{meta.get('version_number')} / {meta.get('version_id')}"
                if meta.get("version_number") else "legacy"
            )
            lines.append(
                f"[{i}] 来源:{src} | 相似度:{h.get('score', 0):.3f} | "
                f"文档版本:{version_label} | 证据质量:{quality['score']:.2f}（仅基于可审计元数据） | "
                f"条件:{conditions}\n{text}"
            )
        return ToolResult(success=True, output="\n\n".join(lines))
    except Exception as e:  # noqa: BLE001
        logger.warning("knowledge_search 失败: %s", e)
        return ToolResult(success=False, error=f"知识库检索失败: {e}")
