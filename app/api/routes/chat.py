"""对话路由：同步返回 + SSE 流式返回 + 会话历史查询 + 会话导出 + 会话分享。"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from app.core.auth import ROLE_ADMIN, User, get_current_user
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_agent
from app.config import get_settings
from app.core.exceptions import ForbiddenError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.sessions import session_service
from app.services.sharing import SHARE_TTL, sharing_service

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(get_current_user)])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, user: User = Depends(get_current_user)) -> ChatResponse:
    # 用户级限流：固定窗口 30 次/分钟（Redis 不可用自动放行）
    from app.cache.ratelimit import fixed_window

    ok, retry_after = await fixed_window("chat", user.username, limit=30, window=60)
    if not ok:
        from fastapi import HTTPException

        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {retry_after} 秒后再试")
    agent = get_agent(use_web=req.use_web)
    session_id = req.session_id or str(uuid.uuid4())
    answer = await agent.run(session_id, req.message, use_plan=req.use_plan)
    return ChatResponse(
        session_id=session_id, answer=answer, model=get_settings().llm.model
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    agent = get_agent(use_web=req.use_web)
    session_id = req.session_id or str(uuid.uuid4())

    async def event_gen():
        async for ev in agent.stream(session_id, req.message, use_plan=req.use_plan):
            yield {"data": json.dumps(ev.to_dict(), ensure_ascii=False)}

    return EventSourceResponse(event_gen())


@router.get("/sessions")
async def chat_sessions(
    user: User = Depends(get_current_user),
):
    """列出当前用户的所有会话（按 "{username}__" 前缀隔离）。"""
    return {
        "username": user.username,
        "sessions": session_service.list(user.username),
    }


class RenameTitleRequest(BaseModel):
    title: str


@router.post("/sessions/{session_id}/rename")
async def rename_session(
    session_id: str,
    req: RenameTitleRequest,
    user: User = Depends(get_current_user),
):
    """重命名会话标题（仅限当前用户自己的会话）。"""
    title = session_service.rename(user.username, session_id, req.title)
    return {"session_id": session_id, "title": title}


@router.get("/history")
async def chat_history(
    session_id: str = Query(...),
    user: User = Depends(get_current_user),
):
    """返回指定会话的完整消息记录（仅限当前用户自己的会话）。

    会话归属校验：session_id 必须以 "{username}__" 前缀开头（WS 层写入时强制加前缀）。
    """
    _, msgs = await session_service.history(
        user.username, session_id, require_prefix=True
    )
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": m.role.value,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
                "tool_calls": m.tool_calls,
            }
            for m in msgs
        ],
    }


@router.get("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """导出会话为 Markdown（含标题/摘要/完整对话，工具调用折叠为引用块）。"""
    export = await session_service.export_data(user.username, session_id)
    msgs = export["messages"]
    title = export["title"]
    summary = export["summary"]

    lines = [f"# {title}", ""]
    if summary:
        lines += [f"> 📝 摘要：{summary}", ""]
    lines += [f"- 会话 ID：{session_id}", f"- 导出时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

    for m in msgs:
        role = m.role.value
        content = (m.content or "").strip()
        if role == "system":
            continue
        if role == "user":
            lines += ["## 🧑 用户", "", content, ""]
        elif role == "assistant":
            # 带工具调用的 assistant 消息：展示意图
            if m.tool_calls:
                for tc in m.tool_calls:
                    lines += [f"**调用工具** `{tc.name}`：`{tc.arguments}`", ""]
            elif content:
                lines += ["## 🤖 助手", "", content, ""]
        elif role == "tool":
            # 工具结果：折叠为引用块，避免撑爆导出
            text = content[:500] + ("…" if len(content) > 500 else "")
            lines += [f"> 🔧 `{m.name or 'tool'}` 返回：", f"> {text.replace(chr(10), chr(10) + '> ')}", ""]
    md = "\n".join(lines)
    filename = f"{user.username}_{session_id[-12:]}.md"
    return PlainTextResponse(
        md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------- 会话分享（只读链接） ----------

class ShareResponse(BaseModel):
    share_id: str
    url: str
    expires_in: int


@router.post("/sessions/{session_id}/share", response_model=ShareResponse)
async def share_session(
    session_id: str,
    user: User = Depends(get_current_user),
) -> ShareResponse:
    """生成会话只读分享链接（HMAC 签名 + 7 天过期，无任何写权限）。"""
    share_id = sharing_service.create(user.username, session_id)
    return ShareResponse(
        share_id=share_id,
        url=f"/api/chat/share/{share_id}",  # 完整 API 路径（前端拼 origin）
        expires_in=SHARE_TTL,
    )


# 分享视图独立 router：无登录依赖（公开只读，仅签名校验）
public_router = APIRouter(tags=["share"])


@public_router.get("/chat/share/{share_id}")
async def view_shared_session(share_id: str):
    """公开只读视图：任何人持链接可查看（无鉴权，仅签名 + 过期校验）。"""
    from fastapi.responses import HTMLResponse

    shared = await sharing_service.read(share_id)
    msgs = shared["messages"]
    title = shared["title"]
    summary = shared["summary"]

    # 渲染只读 HTML（无任何操作按钮/表单，仅展示）
    lines = [f"<h1>{title}</h1>"]
    if summary:
        lines.append(f"<blockquote>📝 {summary}</blockquote>")
    for m in msgs:
        if m.role.value == "system":
            continue
        role = m.role.value
        content = (m.content or "").strip()
        if role == "user":
            lines.append(f'<div class="msg user"><b>🧑 用户</b><p>{content}</p></div>')
        elif role == "assistant" and not m.tool_calls and content:
            lines.append(f'<div class="msg assistant"><b>🤖 助手</b><p>{content}</p></div>')
        elif role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f'<div class="msg tool"><b>🔧 调用 {tc.name}</b><pre>{tc.arguments}</pre></div>')
        elif role == "tool":
            text = content[:300] + ("…" if len(content) > 300 else "")
            lines.append(f'<div class="msg tool"><b>🔧 {m.name or "tool"} 返回</b><pre>{text}</pre></div>')
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>{title} · 分享</title><style>
body{{max-width:760px;margin:24px auto;padding:0 16px;font-family:-apple-system,'Segoe UI',sans-serif;color:#111827;background:#f8fafc}}
h1{{font-size:20px}} blockquote{{background:#fef3c7;border-left:4px solid #f59e0b;padding:8px 12px;border-radius:6px;margin:0 0 16px}}
.msg{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px;margin:10px 0}}
.msg.user{{border-left:4px solid #6366f1}} .msg.assistant{{border-left:4px solid #10b981}}
.msg.tool{{background:#f1f5f9;font-size:13px;border-left:4px solid #94a3b8}}
pre{{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font-size:13px;font-family:ui-monospace,Consolas,monospace}}
p{{margin:6px 0 0;line-height:1.7}} .foot{{color:#64748b;font-size:12px;text-align:center;margin-top:24px}}
</style></head><body>
{''.join(lines)}
<div class="foot">来自科研助手 Agent 的只读分享 · 链接 7 天内有效</div>
</body></html>"""
    return HTMLResponse(html)


# ---------- 会话重试 / 全文搜索 ----------


class RetryResponse(BaseModel):
    removed: int


@router.post("/sessions/{session_id}/restore", response_model=RetryResponse)
async def restore_session(
    session_id: str,
    user: User = Depends(get_current_user),
) -> RetryResponse:
    """取消会话归档（从归档列表恢复到普通列表）。"""
    return RetryResponse(removed=session_service.restore(user.username, session_id))


@router.post("/sessions/{session_id}/retry", response_model=RetryResponse)
async def retry_session(
    session_id: str,
    user: User = Depends(get_current_user),
) -> RetryResponse:
    """重试准备：删除最后一次 user 提问之后的所有消息（清掉半成品回答），前端随后重发原提问。"""
    return RetryResponse(removed=session_service.retry(user.username, session_id))


class SessionSearchHit(BaseModel):
    session_id: str
    title: str
    hits: int
    snippet: str


class BatchDeleteRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=200)


@router.post("/sessions/batch-delete")
async def batch_delete_sessions(
    req: BatchDeleteRequest, user: User = Depends(get_current_user)
) -> dict:
    """批量删除会话（仅限自己的；不存在的静默跳过）。"""
    return {"deleted": session_service.batch_delete(user.username, req.ids)}


@router.post("/sessions/export-all")
async def export_all_sessions(user: User = Depends(get_current_user)):
    """导出当前用户全部会话为合并 Markdown。"""
    sessions = await session_service.export_all_data(user.username)
    lines: list[str] = [f"# {user.username} 的科研助手会话导出\n"]
    for s in sessions:
        msgs = s["messages"]
        lines.append(f"\n## {s.get('title') or s['session_id']}\n")
        if s.get("summary"):
            lines.append(f"> 📝 {s['summary']}\n")
        for m in msgs:
            if m.role.value == "system":
                continue
            if m.role.value == "user":
                lines.append(f"**用户**：{m.content}\n")
            elif m.role.value == "assistant" and not m.tool_calls and m.content:
                lines.append(f"**助手**：{m.content}\n")
    md = "\n".join(lines)
    from urllib.parse import quote

    safe_name = quote(f"{user.username}_sessions.md")
    return PlainTextResponse(
        md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=\"sessions.md\"; filename*=UTF-8''{safe_name}"},
    )


class SessionSearchResponse(BaseModel):
    query: str
    hits: list[SessionSearchHit]


@router.get("/sessions/search", response_model=SessionSearchResponse)
async def search_sessions(
    q: str = Query(..., min_length=1, max_length=100),
    user: User = Depends(get_current_user),
) -> SessionSearchResponse:
    """会话全文搜索：学生搜自己的会话；admin 可搜全部学生。按最近命中倒序。"""
    hits = session_service.search(
        q, user.username, search_all=user.role == ROLE_ADMIN
    )
    return SessionSearchResponse(query=q, hits=hits)
