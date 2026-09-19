"""ResearchAgent：基于 ReAct（推理-行动）范式的科研助手核心。

执行流程（可流式）：
1. 把用户问题写入记忆；
2. （可选）先用 Planner 拆解任务为步骤；
3. 进入循环：调 LLM -> 若返回 tool_calls 则执行工具并把结果回写 -> 否则得到最终答案；
4. 把答案写入记忆并产出 done 事件。

护栏（防止长链路失控 / 上下文爆炸）：
- token 预算守门：每轮推理前估算上下文 token，超出预算则立即强制收尾；
- observation 截断：单个工具输出过长时被截断，避免撑爆上下文；
- 工具失败自修正：工具抛异常时按 `tool_max_retries` 重试，并把错误作为 observation 回写，
  让 LLM 有机会改用其他工具或基于已有信息回答。

每个阶段都以 `AgentEvent` 形式流式吐出，便于前端做 SSE / WebSocket 渲染。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from typing import AsyncIterator

from app.memory.conversation import ConversationMemory
from app.agent.events import AgentEvent
from app.agent.finalizer import SessionFinalizer
from app.agent.planner import Planner
from app.agent.prompts import CITATION_PROMPT, SYSTEM_PROMPT
from app.config import get_settings
from app.core.logging import get_logger
from app.llm.base import ChatMessage, MessageRole, ToolCall
from app.tools.base import ToolRegistry
from app.tools.executor import ToolExecutor

logger = get_logger("agent")

# 近似换算：1 token ≈ 4 字符（中英文混合的经验值），仅用于预算守门，非精确计量。
_CHARS_PER_TOKEN = 4

# 需要强制要求 [n] 引用的检索类工具
RETRIEVAL_TOOLS = {"knowledge_search", "web_search", "arxiv_search", "pdf_reader"}


def _has_citation(text: str) -> bool:
    """检查文本是否包含 [n] / [n,m] / [n-m] 形式的引用标记。"""
    return bool(re.search(r"\[\d+(?:[-,]\d+)*\]", text))


def _acknowledges_conflict(text: str) -> bool:
    """Require explicit disagreement language and at least two cited sources."""
    value = (text or "").lower()
    markers = ("冲突", "分歧", "不一致", "相反", "尚无定论", "未解决", "conflict", "disagree")
    refs = {int(n) for group in re.findall(r"\[(\d+(?:[-,]\d+)*)\]", value)
            for n in re.split(r"[-,]", group) if n.isdigit()}
    return any(marker in value for marker in markers) and len(refs) >= 2


class ResearchAgent:
    def __init__(
        self,
        llm,
        memory: ConversationMemory,
        tools: ToolRegistry,
        planner: Planner | None = None,
        longterm=None,  # LongTermMemory | None：跨会话记忆（L2），默认关闭
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
        token_budget: int | None = None,
        max_obs_chars: int | None = None,
        tool_max_retries: int | None = None,
        history_compress_ratio: float | None = None,
        history_keep_msgs: int | None = None,
        llm_recovery_timeout: int | None = None,
        no_progress_limit: int | None = None,
    ) -> None:
        s = get_settings().agent
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.longterm = longterm  # 跨会话记忆：None 时不注入/不提取
        self._recalled = False  # 每轮只注入一次跨会话记忆
        self._pending_images: list[str] = []  # 多模态：本轮回合待携带图片（首轮消费）
        # 模型智能路由：配置了 aux_model（便宜模型）才启用——短问答直接走 aux 单轮回答，
        # 复杂任务（检索/分析/写作/工具链）走主模型完整流程。未配置时行为与原来完全一致。
        self._routing_enabled = bool(get_settings().llm.aux_model)
        # 答案自评回调（WS 层注入）：收到 eval 结果时推送给前端；None=不推送
        self._eval_callback = None
        self._background_tasks: set[asyncio.Task] = set()
        self.finalizer = SessionFinalizer(self.memory, self._aux_llm, self.longterm)
        self.max_iterations = max_iterations or s.max_iterations
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else s.max_tool_calls
        # token 预算：显式配置 >0 用配置值；否则按模型上下文窗口 × 0.6 自动计算
        configured_budget = token_budget if token_budget is not None else s.token_budget
        if configured_budget and configured_budget > 0:
            self.token_budget = configured_budget
        else:
            ctx = get_settings().llm.max_context_tokens or 128000
            self.token_budget = int(ctx * 0.6)
        self.max_obs_chars = max_obs_chars if max_obs_chars is not None else s.max_obs_chars
        self.tool_max_retries = (
            tool_max_retries if tool_max_retries is not None else s.tool_max_retries
        )
        # 单个工具调用的总超时（含重试累计）：防网络挂起拖死 worker
        self.tool_timeout = getattr(s, "tool_timeout", 90) or 90
        self.llm_recovery_timeout = (
            llm_recovery_timeout if llm_recovery_timeout is not None
            else getattr(s, "llm_recovery_timeout", 30)
        )
        self.no_progress_limit = (
            no_progress_limit if no_progress_limit is not None
            else getattr(s, "no_progress_limit", 2)
        )
        # 统一工具执行运行时（Tool Runtime）：超时/重试/连续失败/重复调用/截断/追踪
        self.tool_executor = ToolExecutor(
            self.tools,
            timeout=self.tool_timeout,
            max_retries=self.tool_max_retries,
            max_chars=self.max_obs_chars or 8000,
        )
        # 历史压缩参数（ContextManager）：显式传入 > 配置默认
        self.history_compress_ratio = (
            history_compress_ratio if history_compress_ratio is not None
            else getattr(s, "history_compress_ratio", 0.55) or 0.55
        )
        self.history_keep_msgs = (
            history_keep_msgs if history_keep_msgs is not None
            else getattr(s, "history_keep_msgs", 16) or 16
        )
        # 上下文构建器（Context Manager）：消息组装/长期记忆注入/历史压缩/多模态挂载
        from app.agent.context import ContextBuilder

        self._context_builder = ContextBuilder(
            self.memory,
            self.longterm,
            compress_ratio=self.history_compress_ratio,
            keep_msgs=self.history_keep_msgs,
            token_budget=self.token_budget,
            chars_per_token=_CHARS_PER_TOKEN,
        )
        # 历史滑窗按 token 预算截断（与 agent 预算联动；非 ConversationMemory 则忽略）
        if hasattr(memory, "token_budget") and hasattr(memory, "window"):
            memory.token_budget = self.token_budget

    # ---------- 公共入口 ----------

    async def stream(
        self,
        session_id: str,
        user_message: str,
        *,
        use_plan: bool = True,
        images: list[str] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行，逐事件产出 AgentEvent。"""
        await self.memory.add_user(session_id, user_message)
        # 多模态：图片仅在首轮 LLM 调用携带（不持久化）
        self._pending_images = list(images or [])
        # 模型智能路由：短问答（≤60 字、无复杂意图、无图片）→ aux 便宜模型单轮直答，
        # 不进入工具循环（省钱）；复杂任务照常走 ReAct 主模型。仅配置 aux_model 时启用。
        from app.skills.context import active_skill_prompt

        if (
            self._routing_enabled
            and not active_skill_prompt.get()
            and self._is_simple_task(user_message, images)
        ):
            try:
                from app.llm.gateway import get_llm_gateway
                from app.llm.router import TaskType

                resp = await get_llm_gateway().chat(
                    [ChatMessage(role=MessageRole.USER, content=user_message)],
                    task_type=TaskType.SIMPLE_CHAT,
                )
                answer = (resp.content or "").strip()
                await self.memory.add_assistant(session_id, answer)
                yield AgentEvent("answer", {"content": answer, "reason": "simple_route"})
                yield AgentEvent("done", {"iterations": 1})
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"简单问答路由失败，回退主流程: {e}")

        plan_steps: list[str] = []
        if use_plan:
            # 流式规划：规划文本逐 token 产出 plan_stream 事件，结束后发 plan 事件
            async for kind, val in self.planner.plan_stream(user_message):
                if kind == "delta":
                    yield AgentEvent("plan_stream", {"delta": val})
                else:
                    plan_steps = val
                    if plan_steps:
                        yield AgentEvent("plan", {"steps": plan_steps})

        messages = await self._build_messages(session_id, plan_steps)
        tool_schemas = self.tools.schemas()
        tool_call_count = 0
        used_tools: list[str] = []
        route_retry_used = False
        answer_retry_used = False
        citation_retry_used = False
        conflict_evidence_seen = False
        observation_signatures: set[str] = set()
        no_progress_count = 0
        from app.agent.guards import (
            answer_needs_revision,
            recommended_tool,
            tool_matches_recommendation,
        )

        route_tool = recommended_tool(user_message)
        # 对参数可由原问题直接确定的检索工具做预取：
        # 将「LLM 选工具 -> 检索 -> LLM 总结」缩短为「检索 -> LLM 总结」。
        if route_tool in {"knowledge_search", "arxiv_search"}:
            arguments = {"query": user_message}
            if route_tool == "knowledge_search":
                arguments["top_k"] = 3
            else:
                arguments["max_results"] = 5
                arguments["download"] = False
            tc = ToolCall(
                id=f"route_{uuid.uuid4().hex[:12]}",
                name=route_tool,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
            await self.memory.add_assistant(session_id, "", tool_calls=[tc])
            yield AgentEvent("action", {"tool": tc.name, "arguments": tc.arguments})
            started = time.monotonic()
            observation = await self._run_tool(tc)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            truncated = self._truncate_observation(observation)
            await self.memory.add_tool(session_id, tc.id, tc.name, truncated)
            yield AgentEvent("observation", {
                "tool": tc.name,
                "output": truncated,
                "truncated": truncated != observation,
                "duration_ms": elapsed_ms,
            })
            used_tools.append(route_tool)
            conflict_evidence_seen = conflict_evidence_seen or "[EVIDENCE_CONFLICT]" in observation
            tool_call_count += 1
            messages = await self._build_messages(session_id, plan_steps)
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content="已完成必要检索。请优先基于工具观察直接给出最终答案；只在证据明显不足时再调用其他工具。",
            ))
        elif route_tool:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    f"路由器判定本任务应优先调用 `{route_tool}`。"
                    "请先用工具取得可验证结果，再给出最终答案。"
                ),
            ))

        for iteration in range(self.max_iterations):
            # 护栏①：token 预算守门
            if self._over_budget(messages):
                logger.warning("会话 %s 上下文接近 token 预算，强制收尾", session_id)
                answer = await self._force_finish(session_id, messages)
                yield AgentEvent("answer", {"content": answer, "reason": "token_budget"})
                yield AgentEvent("done", {"iterations": iteration + 1})
                return

            # 流式调用 LLM：逐 token 产出 stream 事件，流结束时拿到完整 content 与 tool_calls
            try:
                stream_chunks: list[str] = []
                tool_calls: list[ToolCall] = []
                finish_reason: str | None = None
                async for chunk in self.llm.chat_stream(
                    messages, tools=tool_schemas
                ):
                    if chunk.content_delta:
                        stream_chunks.append(chunk.content_delta)
                        yield AgentEvent("stream", {"delta": chunk.content_delta})
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 流式调用中断，尝试有界降级: %s", e)
                partial = "".join(stream_chunks).strip()
                if partial:
                    # 已收到的内容优先返回，避免流末尾超时把整段有效回答作废。
                    answer = partial + "\n\n（模型连接中断，已保留中断前生成的内容。）"
                    await self.memory.add_assistant(session_id, answer)
                    yield AgentEvent("answer", {"content": answer, "reason": "partial_stream_recovery"})
                    yield AgentEvent("done", {"iterations": iteration + 1, "degraded": True})
                    self._schedule_finalize(session_id, user_message, answer)
                    await asyncio.sleep(0.01)
                    return
                try:
                    # 没有任何流式内容时只恢复一次，并设置外层硬超时；即使 provider
                    # 内部配置了重试，也会被 wait_for 的总时限截断。
                    recovered = await asyncio.wait_for(
                        self.llm.chat(messages, tools=None),
                        timeout=self.llm_recovery_timeout,
                    )
                    answer = (recovered.content or "").strip()
                    if answer:
                        await self.memory.add_assistant(session_id, answer)
                        yield AgentEvent("answer", {"content": answer, "reason": "llm_timeout_recovery"})
                        yield AgentEvent("done", {"iterations": iteration + 1, "degraded": True})
                        self._schedule_finalize(session_id, user_message, answer)
                        await asyncio.sleep(0.01)
                        return
                except Exception as recovery_error:  # noqa: BLE001
                    logger.warning("LLM 有界降级失败: %s", recovery_error)
                yield AgentEvent("error", {"message": str(e), "reason": "llm_unavailable"})
                return

            content = "".join(stream_chunks).strip()

            # 空答案自愈：LLM 返回空内容且无工具调用（偶发），非流式重试一次；
            # 仍为空则给出兜底提示，不静默返回空白回答
            if not tool_calls and not content:
                logger.warning("会话 %s LLM 返回空内容，重试一次", session_id)
                try:
                    retry_resp = await self.llm.chat(messages, tools=tool_schemas)
                    content = (retry_resp.content or "").strip()
                    if not retry_resp.tool_calls and content:
                        answer = content
                        await self.memory.add_assistant(session_id, answer)
                        yield AgentEvent("answer", {"content": answer})
                        self._schedule_finalize(session_id, user_message, answer)
                        yield AgentEvent("done", {"iterations": iteration + 1})
                        await asyncio.sleep(0.01)  # 让后台收尾任务获得首次调度，不等待网络完成
                        return
                except Exception as e:  # noqa: BLE001
                    logger.warning("空答案重试失败: %s", e)

            # 无工具调用 -> 最终回答
            if not tool_calls:
                answer = content or "（本次未能生成有效回答，请换一种问法重试）"

                # 强制引用：若本轮回合用过检索类工具，但答案缺少 [n] 引用标记，
                # 追加引用规范并让 LLM 重写一次（仅一次，避免死循环）。
                if (
                    any(t in RETRIEVAL_TOOLS for t in used_tools)
                    and (
                        not _has_citation(answer)
                        or (conflict_evidence_seen and not _acknowledges_conflict(answer))
                    )
                    and not citation_retry_used
                ):
                    citation_retry_used = True
                    messages.append(ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            CITATION_PROMPT
                            + "\n\n当前答案未满足引用或冲突披露要求。请基于已获得的检索资料"
                              "重新给出最终答案，并在每条关键结论后标注 [n]。若资料存在冲突，"
                              "必须引用至少两个冲突来源并明确披露分歧；没有可核验依据时不得选边。"
                        ),
                    ))
                    continue
                if (
                    route_tool
                    and not tool_matches_recommendation(route_tool, used_tools)
                    and not route_retry_used
                ):
                    route_retry_used = True
                    messages.append(ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            f"你还没有执行必要的 `{route_tool}`。"
                            "不要直接回答；请立即调用该工具。"
                        ),
                    ))
                    continue
                if answer_needs_revision(answer) and not answer_retry_used:
                    answer_retry_used = True
                    messages.append(ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            "当前答案无效。请使用已获得的工具观察，"
                            "直接给出包含关键结论的最终答案，不要再重复相同调用。"
                        ),
                    ))
                    continue
                await self.memory.add_assistant(session_id, answer)
                yield AgentEvent("answer", {"content": answer})
                # 会话收尾（后台任务，不阻塞 done）：
                # 1) 跨会话记忆：提取持久事实；2) 会话摘要；3) 答案质量自评
                self._schedule_finalize(session_id, user_message, answer)
                yield AgentEvent("done", {"iterations": iteration + 1})
                await asyncio.sleep(0.01)
                return

            # 有工具调用 -> 执行并回写
            if content:
                yield AgentEvent("thought", {"content": content})

            await self.memory.add_assistant(
                session_id, content, tool_calls=tool_calls
            )

            # 工具并行执行（Tool Parallelism）：先全部广播 action，再 asyncio.gather
            # 并发跑所有工具（多工具场景提速），完成后按原始顺序回写 observation——
            # LLM 依赖 tool_call_id 配对，顺序必须与 tool_calls 一致。
            for tc in tool_calls:
                used_tools.append(tc.name)
                yield AgentEvent("action", {"tool": tc.name, "arguments": tc.arguments})
            _tool_t0 = time.monotonic()
            results = await asyncio.gather(
                *(self._run_tool(tc) for tc in tool_calls),
                return_exceptions=True,
            )
            # 并行批总耗时（=最慢工具耗时；每个 observation 展示同一批耗时）
            _tool_elapsed_ms = int((time.monotonic() - _tool_t0) * 1000)
            for tc, obs in zip(tool_calls, results):
                if isinstance(obs, BaseException):
                    obs = f"工具 {tc.name} 执行异常：{obs}"
                conflict_evidence_seen = conflict_evidence_seen or "[EVIDENCE_CONFLICT]" in str(obs)
                truncated = self._truncate_observation(obs)
                await self.memory.add_tool(session_id, tc.id, tc.name, truncated)
                yield AgentEvent(
                    "observation",
                    {
                        "tool": tc.name,
                        "output": truncated,
                        "truncated": truncated != obs,
                        "duration_ms": _tool_elapsed_ms,
                    },
                )

            # 护栏④：无进展检测。模型即使不断微调参数，只要同一工具反复给出
            # 相同 observation，也视为没有获得新信息，提前收尾而不是继续空转。
            batch_material = "\n".join(
                f"{tc.name}:{str(obs)[:2000]}" for tc, obs in zip(tool_calls, results)
            )
            batch_signature = hashlib.sha256(batch_material.encode("utf-8", errors="ignore")).hexdigest()
            if batch_signature in observation_signatures:
                no_progress_count += 1
            else:
                observation_signatures.add(batch_signature)
                no_progress_count = 0

            if self.no_progress_limit and no_progress_count >= self.no_progress_limit:
                logger.warning("会话 %s 连续无新工具信息，提前强制收尾", session_id)
                messages = await self._build_messages(session_id, plan_steps)
                answer = await self._force_finish(session_id, messages)
                yield AgentEvent("answer", {"content": answer, "reason": "no_progress"})
                yield AgentEvent("done", {"iterations": iteration + 1})
                return

            # 护栏⑤：工具调用总次数上限（防联网搜索等死循环）
            tool_call_count += len(tool_calls)
            if self.max_tool_calls and tool_call_count >= self.max_tool_calls:
                logger.warning(
                    "会话 %s 工具调用达上限(%d)，强制收尾",
                    session_id,
                    self.max_tool_calls,
                )
                # 重新读取消息，确保刚执行的 observation 不会在强制收尾时丢失。
                messages = await self._build_messages(session_id, plan_steps)
                answer = await self._force_finish(session_id, messages)
                yield AgentEvent("answer", {"content": answer, "reason": "max_tool_calls"})
                yield AgentEvent("done", {"iterations": iteration + 1})
                return

            # 刷新消息（含新的 observation），进入下一轮
            messages = await self._build_messages(session_id, plan_steps)

        # 超过迭代上限：强制收尾
        answer = await self._force_finish(session_id, messages)
        yield AgentEvent("answer", {"content": answer, "reason": "max_iterations"})
        yield AgentEvent("done", {"iterations": self.max_iterations})

    async def run(self, session_id: str, user_message: str, *, use_plan: bool = True, images: list[str] | None = None) -> str:
        """非流式执行，返回最终回答文本。"""
        answer = ""
        async for event in self.stream(session_id, user_message, use_plan=use_plan, images=images):
            if event.type == "answer":
                answer = event.data.get("content", "")
        return answer

    # ---------- 护栏与辅助 ----------

    async def _force_finish(self, session_id: str, messages: list[ChatMessage]) -> str:
        """在达到限制时强制产出最终回答。"""
        try:
            capped = messages + [
                ChatMessage(
                    role=MessageRole.USER,
                    content="请基于已有信息直接给出最终回答，无需再调用工具。",
                )
            ]
            final = await asyncio.wait_for(
                self.llm.chat(capped, tools=None),
                timeout=self.llm_recovery_timeout,
            )
            answer = final.content.strip()
            await self.memory.add_assistant(session_id, answer)
            return answer
        except Exception as e:  # noqa: BLE001
            logger.exception("强制收尾时 LLM 调用失败")
            return "（因达到运行限制而中断，未能生成最终回答）"

    async def _run_tool(self, tc: ToolCall) -> str:
        """执行单个工具，委托 ToolExecutor（统一超时/重试/护栏/追踪）。

        重试策略（ToolGuard，见 app/tools/executor.py）：
        - 参数类错误（retryable=False）：不重试，直接把错误回写让 LLM 修正参数或换工具；
        - 执行/网络类错误（retryable=True）：按 tool_max_retries 退避重试；
        - 连续失败检测：同一工具连续失败 N 次自动跳过；
        - 重复调用检测：同工具同参数重复调用提示循环。
        """
        return await self.tool_executor.execute(tc.name, tc.arguments)

    def _truncate_observation(self, obs: str) -> str:
        """护栏②：单个 observation 超长截断。"""
        if self.max_obs_chars and len(obs) > self.max_obs_chars:
            return (
                obs[: self.max_obs_chars]
                + f"\n…[内容过长已截断，原文 {len(obs)} 字符]"
            )
        return obs

    def _estimate_tokens(self, messages: list[ChatMessage]) -> int:
        """近似 token 估算（委托 app.agent.guards）。"""
        from app.agent.guards import estimate_tokens

        return estimate_tokens(messages, chars_per_token=_CHARS_PER_TOKEN)

    def _over_budget(self, messages: list[ChatMessage]) -> bool:
        """护栏③：上下文 token 是否超过预算（预留少量给收尾回答）。"""
        from app.agent.guards import over_budget

        return over_budget(messages, self.token_budget)

    # ---------- 内部辅助 ----------

    async def _build_messages(
        self, session_id: str, plan_steps: list[str]
    ) -> list[ChatMessage]:
        """构建消息列表（委托 ContextBuilder：系统提示/长期记忆/压缩/多模态）。"""
        msgs, consumed = await self._context_builder.build(
            session_id,
            plan_steps,
            self._pending_images,
            recall_longterm=not self._recalled,
        )
        self._recalled = True
        if consumed:
            self._pending_images = []
        return msgs

    def _aux_llm(self):
        """ModelRouter：内部低价值调用用的 LLM。

        配置了 aux_model 时返回辅助（便宜）模型；否则回退主模型 self.llm
        （未启用路由时行为与之前完全一致，测试 stub 也走这里）。
        """
        from app.config import get_settings as _gs

        if _gs().llm.aux_model:
            from app.llm.factory import get_aux_llm

            return get_aux_llm()
        return self.llm

    @staticmethod
    def _is_simple_task(message: str, images: list[str] | None) -> bool:
        """简单任务预判（委托 app.agent.guards 纯函数）。"""
        from app.agent.guards import is_simple_task

        return is_simple_task(message, images)

    async def _maybe_compress_history(
        self, session_id: str, history: list[ChatMessage]
    ) -> list[ChatMessage]:
        """上下文压缩（ContextManager）：历史估算 token 超过预算 × 比例时，
        把最早的对话轮次压缩为 LLM 摘要，保留最近 history_keep_msgs 条原文。

        - 压缩边界永远落在 USER 消息上（不会从 tool 消息切开，避免孤立 tool 400）；
        - 摘要作为一条 system 消息注入，Agent 仍能感知早期上下文（主题/结论/偏好）；
        - 压缩失败静默回退原文，不影响主流程。
        """
        ratio = self.history_compress_ratio
        keep = self.history_keep_msgs
        if ratio <= 0 or self.token_budget <= 0:
            return history
        if len(history) <= keep + 2:
            return history
        hist_tokens = sum(self._estimate_tokens([m]) for m in history)
        if hist_tokens <= self.token_budget * ratio:
            return history

        # 压缩边界：从尾部保留 keep 条，向前回退到最近的 USER 消息
        boundary = len(history) - keep
        while boundary > 0 and history[boundary].role != MessageRole.USER:
            boundary -= 1
        if boundary <= 0:
            return history  # 找不到合适的 USER 边界，放弃压缩

        early = history[:boundary]
        recent = history[boundary:]
        try:
            transcript = "\n".join(
                f"{m.role.value}: {(m.content or '')[:400]}"
                for m in early
                if m.content
            )[:6000]
            # 路由到辅助模型（ModelRouter）：摘要属内部低价值调用，用便宜模型省钱
            resp = await self._aux_llm().chat(
                [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            "你是对话摘要器。把下面的早期对话压缩为一段 150 字以内的摘要，"
                            "必须保留：用户的研究主题、已确认的关键结论、重要数据/发现、"
                            "用户偏好。只输出摘要正文，不要任何前缀。"
                        ),
                    ),
                    ChatMessage(role=MessageRole.USER, content=transcript or "（无文本内容）"),
                ],
                tools=None,
                temperature=0.2,
            )
            summary = (resp.content or "").strip()
            if summary:
                logger.info(
                    "会话 %s 历史压缩：%d 条 -> 摘要（保留最近 %d 条原文）",
                    session_id, boundary, len(recent),
                )
                return [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            "## 早期对话摘要（已压缩，仅作背景参考）\n"
                            f"{summary}\n"
                            "以下为最近对话原文："
                        ),
                    )
                ] + recent
        except Exception as e:  # noqa: BLE001
            logger.debug("历史压缩失败（回退原文）: %s", e)
        return history

    async def _finalize_session(
        self, session_id: str, user_message: str, answer: str
    ) -> None:
        """会话收尾（全部失败静默，不阻塞主流程）：

        1. 跨会话记忆：提取持久事实（L2）；
        2. 会话自动摘要：整场对话压成摘要存会话元数据（供列表/恢复展示）；
        3. 答案质量自评：评估是否回答了问题，产出 eval 事件。
        """
        self.finalizer.eval_callback = self._eval_callback
        await self.finalizer.finalize(session_id, user_message, answer)

    def _schedule_finalize(self, session_id: str, user_message: str, answer: str) -> None:
        """将摘要、事实提取和答案自评真正放到响应后执行。"""
        task = asyncio.create_task(self._finalize_session(session_id, user_message, answer))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _summarize_session(self, session_id: str) -> None:
        """会话自动摘要：把整场对话（最近的）压成 120 字以内摘要，存会话元数据。

        供会话列表展示"这轮聊了什么"；失败静默（不影响 done）。
        """
        await self.finalizer.summarize(session_id)

    async def _self_eval_answer(
        self, session_id: str, user_message: str, answer: str
    ) -> None:
        """答案质量自评：用轻量模型评估回答质量，产出 eval 事件。

        eval 事件字段：{answered, confidence, missing, tooltip}。
        前端据此展示"置信度徽章/建议补充"；失败静默（无 eval 事件不影响主流程）。
        """
        self.finalizer.eval_callback = self._eval_callback
        await self.finalizer.evaluate(session_id, user_message, answer)

    async def _eval_hook(self, session_id: str, data: dict) -> None:
        """自评结果回调（默认为空；WS 层注入真正的发送逻辑）。"""
        if self._eval_callback is not None:
            try:
                await self._eval_callback(session_id, data)
            except Exception:  # noqa: BLE001
                pass

    async def _extract_and_store_facts(
        self, session_id: str, user_message: str, answer: str
    ) -> None:
        """对话结束后提取持久事实存入长期记忆（L2）。失败静默，不影响主流程。"""
        await self.finalizer.extract_facts(session_id, user_message, answer)
