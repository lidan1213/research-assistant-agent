"""Post-answer session processing."""
from __future__ import annotations

import asyncio
import json

from app.core.logging import get_logger
from app.llm.base import ChatMessage, MessageRole

logger = get_logger("agent.finalizer")


class SessionFinalizer:
    """Generate session summaries, quality evaluations, and durable facts."""

    def __init__(self, memory, llm_provider, longterm=None) -> None:
        self.memory = memory
        self.llm_provider = llm_provider
        self.longterm = longterm
        self.eval_callback = None

    async def finalize(self, session_id: str, user_message: str, answer: str) -> None:
        await asyncio.gather(
            self.extract_facts(session_id, user_message, answer),
            self.summarize(session_id),
            self.evaluate(session_id, user_message, answer),
            return_exceptions=True,
        )

    async def summarize(self, session_id: str) -> None:
        try:
            history = await self.memory.history(session_id)
            transcript = "\n".join(
                f"{message.role.value}: {(message.content or '')[:200]}"
                for message in history[-20:]
                if message.content
            )[:5000]
            if not transcript:
                return
            response = await self.llm_provider().chat(
                [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            "你是对话摘要器。把下面的科研对话压缩为一段 120 字以内的摘要，"
                            "保留：用户研究主题、关键结论、使用的工具/方法、未决问题。"
                            "只输出摘要正文，不要前缀。"
                        ),
                    ),
                    ChatMessage(role=MessageRole.USER, content=transcript),
                ],
                tools=None,
                temperature=0.2,
            )
            summary = (response.content or "").strip()
            if summary:
                await self.memory.set_summary(session_id, summary)
        except Exception as exc:  # noqa: BLE001
            logger.debug("会话摘要生成失败（忽略）: %s", exc)

    async def evaluate(self, session_id: str, question: str, answer: str) -> None:
        if not answer or len(answer) < 10:
            return
        try:
            prompt = (
                "你是答案质检员。评估下面的回答是否充分回答了用户问题。\n"
                '输出 JSON：{"answered": true/false, "confidence": "high|medium|low", '
                '"missing": "缺失的关键信息（无则空串）"}\n\n'
                f"用户问题：{question[:200]}\n\n助手回答：{answer[:800]}"
            )
            response = await self.llm_provider().chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)],
                tools=None,
                json_mode=True,
                temperature=0.0,
            )
            data = self._parse_json(response.content)
            answered = bool(data.get("answered", True))
            confidence = data.get("confidence") or "medium"
            missing = (data.get("missing") or "").strip()
            tooltip = ""
            if not answered:
                tooltip = "⚠️ 回答可能未完全覆盖问题"
            elif confidence == "low":
                tooltip = "⚠️ 回答置信度较低，建议结合知识库/联网进一步核实"
            elif missing:
                tooltip = f"💡 可能遗漏：{missing}"
            if tooltip and self.eval_callback is not None:
                await self.eval_callback(
                    session_id,
                    {
                        "answered": answered,
                        "confidence": confidence,
                        "missing": missing,
                        "tooltip": tooltip,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("答案自评失败（忽略）: %s", exc)

    async def extract_facts(
        self, session_id: str, question: str, answer: str
    ) -> None:
        if self.longterm is None:
            return
        try:
            prompt = (
                "从下面这段对话中提取对后续研究有用的持久事实"
                "（用户的研究主题、关键结论、使用过的工具/方法、用户偏好），"
                '用 JSON 数组输出，每项 {"content":"事实","category":"user_fact"|"finding",'
                '"memory_key":"稳定主题键，如 user.research_topic；无稳定键则空串",'
                '"confidence":0到1,"ttl_days":null或正整数}。同一属性应始终使用相同 memory_key，'
                '临时状态应设置 ttl_days，最多 3 条；没有则输出 []：\n\n'
                f"用户：{question[:200]}\n\n助手：{answer[:600]}"
            )
            response = await self.llm_provider().chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)],
                tools=None,
                json_mode=True,
            )
            items = self._parse_json(response.content)
            if not isinstance(items, list):
                return
            for item in items[:3]:
                content = (item.get("content") or "").strip()
                if content:
                    self.longterm.append_fact(
                        session_id,
                        content=content,
                        category=item.get("category") or "finding",
                        tags=["auto"],
                        memory_key=(item.get("memory_key") or "").strip()[:100],
                        confidence=item.get("confidence", 0.8),
                        ttl_days=item.get("ttl_days"),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("长期事实提取失败（忽略）: %s", exc)

    @staticmethod
    def _parse_json(content: str):
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return json.loads(text)
