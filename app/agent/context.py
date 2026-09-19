"""Context Builder：Agent 上下文构建（消息组装、长期记忆注入、历史压缩、多模态挂载）。

从 ResearchAgent._build_messages 抽取，依赖注入（memory / longterm / 压缩参数），
与 Agent 主循环解耦，便于单测与多 Agent worker 复用。

职责：
1. System Prompt 组装（基础 prompt + 计划步骤 + 跨会话记忆）
2. 历史读取 + 上下文压缩（压缩边界落在 USER 消息，避免孤立 tool 400）
3. 多模态图片挂载（首轮 user 消息）
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm.base import ChatMessage, MessageRole

logger = logging.getLogger("agent.context")


class ContextBuilder:
    """上下文构建器（无状态设计：每次 build 独立，recalled 由外部管理）。"""

    def __init__(
        self,
        memory: Any,
        longterm: Any = None,
        *,
        compress_ratio: float = 0.6,
        keep_msgs: int = 10,
        token_budget: int = 0,
        chars_per_token: int = 3,
        system_prompt: str | None = None,
    ) -> None:
        if system_prompt is None:
            from app.agent.prompts import SYSTEM_PROMPT

            system_prompt = SYSTEM_PROMPT
        self.memory = memory
        self.longterm = longterm
        self.compress_ratio = compress_ratio
        self.keep_msgs = keep_msgs
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token
        self.system_prompt = system_prompt

    async def build(
        self,
        session_id: str,
        plan_steps: list[str] | None = None,
        pending_images: list[str] | None = None,
        *,
        recall_longterm: bool = True,
    ) -> tuple[list[ChatMessage], list[str]]:
        """构建完整消息列表。

        返回 (messages, consumed_images)：consumed_images 是已挂载到首轮
        user 消息的图片（调用方应据此清空 pending 队列）。
        """
        system_text = self.system_prompt
        from app.skills.context import append_active_skill

        system_text = append_active_skill(system_text)
        if plan_steps:
            steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan_steps))
            system_text += f"\n\n本次任务的建议执行计划：\n{steps}"

        # 跨会话记忆（L2）注入：用最近用户消息语义召回历史事实
        if recall_longterm and self.longterm is not None:
            try:
                history_msgs = await self.memory.history(session_id)
                recent_user = next(
                    (m.content for m in reversed(history_msgs) if m.role == MessageRole.USER),
                    "",
                )
                if recent_user:
                    username = session_id.split("__", 1)[0] if "__" in session_id else None
                    # Unowned/legacy sessions do not receive cross-session facts.
                    if username:
                        try:
                            facts = self.longterm.search_facts(recent_user, k=3, username=username)
                        except TypeError:  # compatibility with injected legacy/test stores
                            facts = self.longterm.search_facts(recent_user, k=3)
                    else:
                        facts = []
                    if facts:
                        lines = "\n".join(
                            f"- [{f.get('status', 'active')}] {f['content']}"
                            for f in facts[:3]
                        )
                        system_text += (
                            "\n\n## 跨会话记忆（来自你之前的研究，仅供参考，"
                            "如有冲突以当前对话为准）\n" + lines
                        )
            except Exception as e:  # noqa: BLE001
                logger.debug("跨会话记忆召回失败（忽略）: %s", e)

        history = await self.memory.history(session_id)
        history = await self._maybe_compress_history(session_id, history)
        msgs = [ChatMessage(role=MessageRole.SYSTEM, content=system_text)] + history

        # 多模态：待携带图片挂到最新一条 user 消息（消费后返回，由调用方清空）
        consumed: list[str] = []
        if pending_images and msgs and msgs[-1].role == MessageRole.USER:
            msgs[-1].images = list(pending_images)
            consumed = list(pending_images)
        return msgs, consumed

    async def _maybe_compress_history(
        self, session_id: str, history: list[ChatMessage]
    ) -> list[ChatMessage]:
        """上下文压缩：历史 token 超预算 × 比例时，把最早轮次压成摘要。

        - 压缩边界永远落在 USER 消息上（不会从 tool 消息切开，避免孤立 tool 400）；
        - 摘要作为一条 system 消息注入；
        - 压缩失败静默回退原文。
        """
        ratio = self.compress_ratio
        keep = self.keep_msgs
        if ratio <= 0 or self.token_budget <= 0:
            return history
        if len(history) <= keep + 2:
            return history
        hist_tokens = self._estimate_tokens(history)
        if hist_tokens <= self.token_budget * ratio:
            return history

        boundary = len(history) - keep
        while boundary > 0 and history[boundary].role != MessageRole.USER:
            boundary -= 1
        if boundary <= 0:
            return history

        early = history[:boundary]
        recent = history[boundary:]
        transcript = "\n".join(
            f"{'用户' if m.role == MessageRole.USER else 'AI'}: {(m.content or '')[:200]}"
            for m in early
            if m.content
        )[:3000]
        summary = await self._summarize(transcript)
        if not summary:
            return history
        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"（以下是本会话早前内容的摘要，供参考）\n{summary}",
            )
        ] + recent

    async def _summarize(self, transcript: str) -> str:
        """调用 LLM 压缩历史（失败返回空串，调用方回退原文）。"""
        try:
            from app.llm.base import ChatMessage, MessageRole
            from app.llm.gateway import get_llm_gateway
            from app.llm.router import TaskType

            resp = await get_llm_gateway().chat(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            "把下面的对话压缩成 120 字以内的中文摘要，"
                            "保留主题、关键结论与用户偏好，只输出摘要：\n\n" + transcript
                        ),
                    )
                ],
                tools=None,
                task_type=TaskType.SUMMARIZATION,
            )
            return (resp.content or "").strip()[:400]
        except Exception as e:  # noqa: BLE001
            logger.debug("历史压缩失败（回退原文）: %s", e)
            return ""

    def _estimate_tokens(self, messages: list[ChatMessage]) -> int:
        total_chars = sum(len(m.content or "") for m in messages)
        return total_chars // self.chars_per_token
