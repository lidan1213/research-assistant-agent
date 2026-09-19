"""WebSocket 路由：三种模式（ReAct / 先规划后执行 / 多 Agent）统一走 WS 双向流式对话。

消息协议（客户端 -> 服务端）：
    {"type": "chat", "mode": "react"|"plan"|"multi", "session_id": "...",
     "message": "...", "use_plan": true, "use_web": false}
    {"type": "stop"}                        # 随时叫停当前任务

服务端 -> 客户端：
    Agent 事件: {"type": "plan"|"step_start"|"action"|"observation"|"answer"|"done"|...}
    {"type": "stopped"}                     # 任务已被用户取消
    {"type": "error", "message": "..."}     # 出错

停止机制：每个 chat 任务包装为 asyncio.Task，收到 stop 即 task.cancel()，
agent 流（async generator）收到 CancelledError 后中断，不再写入后续记忆。
"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import get_agent, get_multiagent, get_plan_agent
from app.core.auth import User, verify_token
from app.core.logging import get_logger

logger = get_logger("ws")

router = APIRouter(tags=["ws"])


async def _prune_user_sessions(username: str, keep: int = 50) -> None:
    """按用户前缀清理历史会话，只保留最近 keep 个（防 memory.db 无限增长）。"""
    try:
        from app.memory.conversation import ConversationMemory

        mem = ConversationMemory()
        store = getattr(mem, "_store", None)
        if store is None or not hasattr(store, "prune_sessions"):
            return
        deleted = store.prune_sessions(prefix=f"{username}__", keep=keep)
        if deleted:
            logger.info(f"会话清理 [{username}] 删除 {deleted} 个旧会话")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"会话清理失败: {e}")


async def _ensure_session_title(session_id: str, message: str, user: User | None = None) -> None:
    """会话首条消息时用 LLM 生成简短标题（失败则回退首句截断，不阻塞对话）。"""
    try:
        from app.memory.conversation import ConversationMemory

        mem = ConversationMemory()
        if await mem.get_title(session_id):
            return  # 已有标题，跳过
        # 标题只取前 80 字，控制成本
        sample = message.strip()[:80]
        from app.llm.base import ChatMessage, MessageRole
        from app.llm.gateway import get_llm_gateway
        from app.llm.router import TaskType

        resp = await get_llm_gateway().chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "给下面这句用户提问生成一个 10 字以内的对话标题，"
                        "只输出标题本身，不要引号、不要解释：\n\n" + sample
                    ),
                )
            ],
            tools=None,
            task_type=TaskType.TITLE_GEN,
        )
        title = (resp.content or "").strip().strip('"\'，。！？')
        if not title:
            raise ValueError("空标题")
        await mem.set_title(session_id, title[:30])
        logger.info(f"会话标题已生成 [{session_id}] -> {title}")
    except Exception as e:  # noqa: BLE001
        # 回退：首句截断，保证列表可读性
        try:
            from app.memory.conversation import ConversationMemory

            fallback = message.strip().replace("\n", " ")[:20] or "新对话"
            await ConversationMemory().set_title(session_id, fallback)
        except Exception:  # noqa: BLE001
            logger.debug(f"会话标题生成失败: {e}")


async def _stream_agent(
    ws: WebSocket,
    mode: str,
    session_id: str,
    message: str,
    use_plan: bool,
    use_web: bool,
    user: User | None = None,
    kb_name: str | None = None,
    model: str | None = None,
    images: list[str] | None = None,
    resume_token: str | None = None,
    requested_skill: str | None = None,
) -> None:
    """按模式分发到对应 Agent，并逐事件推送。取消（CancelledError）向上传播。"""
    # 设置当前用户 + 知识库选择 ContextVar，让 knowledge_search 工具感知：
    # current_username 决定检索哪个用户，current_kb_name 决定检索哪个库（None=默认库，"none"=不检索）
    from app.llm.trace import current_session_id, current_trace_id
    from app.skills.context import active_skill_prompt
    from app.skills.registry import get_skill_registry
    from app.tools.knowledge_search import current_kb_name, current_username

    tok_u = current_username.set(user.username if user else None)
    tok_k = current_kb_name.set(kb_name)
    tok_s = current_session_id.set(session_id)  # 链路追踪：本次调用归属该会话
    tok_t = current_trace_id.set(uuid.uuid4().hex[:16])
    skill = get_skill_registry().match(message, requested_skill)
    tok_skill = active_skill_prompt.set(skill.prompt() if skill else "")
    try:
        from app.agent.router import get_agent_router

        decision = get_agent_router().route(
            message, requested_mode=mode, images=images, use_web=use_web
        )
        await ws.send_json({"type": "route", **decision.to_dict()})
        await ws.send_json({
            "type": "skill",
            "matched": skill.public_dict() if skill else None,
            "selection": requested_skill or "auto",
        })
        from app.llm.channel import configured_channel

        await ws.send_json({
            "type": "provider",
            **configured_channel(use_aux=decision.model_tier == "fast"),
        })
        execution_mode = {
            "direct": "react",
            "single_agent": "react",
            "plan_execute": "plan",
            "multi_agent": "multi",
        }[decision.route]
        if skill and mode == "auto" and skill.preferred_mode:
            execution_mode = skill.preferred_mode
        if decision.route == "direct" and not skill:
            from app.llm.base import ChatMessage, MessageRole
            from app.llm.gateway import get_llm_gateway
            from app.llm.router import TaskType
            from app.memory.conversation import ConversationMemory

            memory = ConversationMemory()
            await memory.add_user(session_id, message)
            response = await get_llm_gateway().chat(
                [ChatMessage(role=MessageRole.USER, content=message)],
                tools=None, task_type=TaskType.SIMPLE_CHAT,
            )
            answer = (response.content or "").strip()
            await memory.add_assistant(session_id, answer)
            await ws.send_json({"type": "answer", "content": answer, "reason": "direct_route"})
            await ws.send_json({"type": "done", "route": "direct"})
            return

        allowed_tools = set(decision.tools or ())
        if skill:
            allowed_tools.update(skill.tools)
        allowed = tuple(allowed_tools) or None
        if execution_mode == "multi":
            ma = get_multiagent()
            async for ev in ma.stream(message, session_id=session_id):
                await ws.send_json(ev if isinstance(ev, dict) else ev.to_dict())
        elif execution_mode == "plan":
            agent = get_plan_agent(use_web=use_web, allowed_tools=allowed)
            source = agent.resume(resume_token, message) if resume_token else agent.stream(message)
            async for ev in source:
                await ws.send_json(ev.to_dict())
        else:  # react（默认）
            agent = get_agent(
                use_web=use_web, model=model, allowed_tools=allowed
            )
            # 注入答案自评回调：eval 结果以事件推送前端（异步发送，失败静默）
            async def _eval_cb(_sid: str, data: dict) -> None:
                try:
                    await ws.send_json({"type": "eval", **data})
                except Exception:  # noqa: BLE001
                    pass

            agent._eval_callback = _eval_cb  # noqa: SLF001
            async for ev in agent.stream(session_id, message, use_plan=use_plan, images=images):
                await ws.send_json(ev.to_dict())
    except asyncio.CancelledError:
        raise  # 用户叫停：取消向上传播，不发送 error
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent 流执行异常")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        current_username.reset(tok_u)
        current_kb_name.reset(tok_k)
        current_session_id.reset(tok_s)
        current_trace_id.reset(tok_t)
        active_skill_prompt.reset(tok_skill)


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    # WebSocket 无法携带 Authorization header（浏览器限制），token 通过 URL query 传入。
    # 注意：ws.query_params.get("token") 已是解析后的值，直接校验，勿再二次解析。
    token = ws.query_params.get("token") or ""
    user = verify_token(token)
    if user is None:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "未登录或登录已过期"})
        await ws.close()
        return
    await ws.accept()
    # 用户级限流：WS 消息 60 条/分钟（Redis 不可用自动放行；超限发错误事件不断连）
    from app.cache.ratelimit import fixed_window

    # MCP 工具懒注册：放独立后台任务（幂等），与当前 WS 连接生命周期解耦，
    # 避免 stdio_client 的 cancel scope 绑定在单个连接 task 上
    try:
        from app.tools.base import registry
        from app.tools.mcp import ensure_mcp_tools

        asyncio.create_task(ensure_mcp_tools(registry))
    except Exception:  # noqa: BLE001
        pass
    current: asyncio.Task | None = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "消息必须是 JSON"})
                continue

            # 停止信号：取消当前正在执行的任务
            if data.get("type") == "stop":
                if current is not None and not current.done():
                    current.cancel()
                await ws.send_json({"type": "stopped"})
                continue

            message = data.get("message", "")
            # 用户级限流：WS 消息 60 条/分钟（Redis 不可用自动放行）
            ok_rl, retry_after = await fixed_window("ws_chat", user.username, limit=60, window=60)
            if not ok_rl:
                await ws.send_json(
                    {"type": "error", "message": f"请求过于频繁，请 {retry_after} 秒后再试"}
                )
                continue
            # 内容审计：用户消息敏感词命中即记录（不拦截，供管理员复核）
            if message:
                from app.core.audit import check_sensitive, record_sensitive_hit

                for w in check_sensitive(message):
                    record_sensitive_hit(user.username, w, message, data.get("session_id") or "")
            if not message:
                continue
            if len(message) > 8000:
                await ws.send_json(
                    {"type": "error", "message": "输入过长（上限 8000 字），请精简后重试"}
                )
                continue

            # 新消息顶掉仍在跑的旧任务（防止多任务交错）
            if current is not None and not current.done():
                current.cancel()

            session_id = data.get("session_id") or str(uuid.uuid4())
            # 会话归属当前登录用户：session_id 统一加 "{username}__" 前缀，
            # 管理员可按前缀查询某学生的全部会话
            if not session_id.startswith(f"{user.username}__"):
                session_id = f"{user.username}__{session_id}"
            mode = data.get("mode", "auto")
            use_plan = bool(data.get("use_plan", True))
            use_web = bool(data.get("use_web", False))
            # 用户自选模型：非空时覆盖默认（None/缺省=配置默认模型）
            model = data.get("model")
            if model is not None and not isinstance(model, str):
                model = str(model)
            # 知识库选择：None/缺省=默认库；"none"=不使用知识库；其他=指定库名
            kb_name = data.get("kb_name")
            if kb_name is not None and not isinstance(kb_name, str):
                kb_name = str(kb_name)
            skill_name = data.get("skill", "auto")
            if not isinstance(skill_name, str) or len(skill_name) > 64:
                skill_name = "auto"
            # 多模态：可选携带一张图片（前端已压缩的 data URL；限 8MB 防撑爆 WS）
            images = None
            img = data.get("image")
            if isinstance(img, str) and img.startswith("data:image/") and len(img) < 8 * 1024 * 1024:
                images = [img]
            current = asyncio.create_task(
                _stream_agent(
                    ws, mode, session_id, message, use_plan, use_web,
                    user, kb_name, model, images, data.get("resume_token"), skill_name,
                )
            )
            # 会话标题：首条消息时异步生成（不阻塞对话；已有标题则跳过）
            asyncio.create_task(_ensure_session_title(session_id, message, user))
            # 会话清理：每用户保留最近 50 个会话，超出自动删最旧的（防无限增长）
            asyncio.create_task(_prune_user_sessions(user.username))
    except WebSocketDisconnect:
        if current is not None and not current.done():
            current.cancel()
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("WS 主循环异常")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass
