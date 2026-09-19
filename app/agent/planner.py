"""规划器：把用户目标拆解为有序步骤，供 Agent 顺序推进。"""
from __future__ import annotations

import re

from app.agent.prompts import PLANNER_PROMPT
from app.llm.base import ChatMessage, MessageRole


class Planner:
    def __init__(self, llm) -> None:
        self.llm = llm

    async def plan(self, goal: str, max_steps: int = 5) -> list[str]:
        """返回步骤列表；失败时回退为单步目标。"""
        steps: list[str] = []
        async for kind, val in self.plan_stream(goal, max_steps):
            if kind == "steps":
                steps = val
        return steps or [goal]

    async def plan_stream(self, goal: str, max_steps: int = 5):
        """流式规划：逐 token 产出 ("delta", 文本)，结束后产出 ("steps", 步骤列表)。

        调用方：`async for kind, val in planner.plan_stream(goal): ...`
        """
        try:
            chunks: list[str] = []
            async for chunk in self.llm.chat_stream(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=f"{PLANNER_PROMPT}\n\n用户目标：{goal}",
                    )
                ],
                tools=None,
            ):
                if chunk.content_delta:
                    chunks.append(chunk.content_delta)
                    yield ("delta", chunk.content_delta)
            content = "".join(chunks)
            steps = [
                re.sub(r"^\s*\d+[.、)]\s*", "", line).strip()
                for line in content.splitlines()
                if line.strip()
            ]
            steps = [s for s in steps if s][:max_steps]
            yield ("steps", steps or [goal])
        except Exception:  # noqa: BLE001
            yield ("steps", [goal])
