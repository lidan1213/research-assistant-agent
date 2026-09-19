"""统一记忆管理器：把「会话记忆(L1)」与「长期记忆(L2/L3)」串成一条链路。

链路（写入，运行时 → 长期）：
    工具观察 / 专家结论 → 会话记忆(L1) 落盘
    多 Agent 黑板+链路成品 → 汇总为「本次研究结论」→ 长期记忆(L2) + 会话索引(L3)

链路（读取，新会话 → 召回）：
    用户问题 → 检索 L2(语义/关键字召回相关事实) + 拉取 L1(当前会话最近历史)
              → 拼成上下文注入 LLM，使助手具备「跨会话记忆」
"""
from __future__ import annotations

from typing import Optional

from app.memory.conversation import ConversationMemory
from app.config import get_settings
from app.core.logging import get_logger
from app.memory.longterm import LongTermMemory

logger = get_logger("memory.manager")


class MemoryManager:
    def __init__(
        self,
        session: Optional[ConversationMemory] = None,
        longterm: Optional[LongTermMemory] = None,
    ) -> None:
        self.session = session or ConversationMemory()
        self.longterm = longterm or LongTermMemory(get_settings().memory.longterm_path)

    async def record_run(
        self,
        session_id: str,
        query: str,
        answer: str,
        *,
        facts: Optional[list] = None,
        title: str = "",
    ) -> None:
        """一次研究/对话结束后，沉淀到会话记忆与长期记忆。"""
        await self.session.add_user(session_id, query)
        await self.session.add_assistant(session_id, answer)
        self.longterm.save_session(
            session_id, title=title or query[:40], summary=answer[:200]
        )
        for f in facts or []:
            if isinstance(f, dict):
                self.longterm.append_fact(
                    session_id,
                    content=f.get("content", ""),
                    category=f.get("category", "finding"),
                    tags=f.get("tags"),
                    memory_key=f.get("memory_key", ""),
                    confidence=f.get("confidence", 1.0),
                    ttl_days=f.get("ttl_days"),
                )
            else:
                self.longterm.append_fact(session_id, str(f))

    async def recall(self, session_id: str, query: str, k: int = 5) -> dict:
        """召回与新问题相关的长期事实 + 当前会话历史，供注入 prompt。"""
        username = session_id.split("__", 1)[0] if "__" in session_id else None
        facts = self.longterm.search_facts(query, k=k, username=username) if username else []
        history = await self.session.history(session_id)
        return {"facts": facts, "history": history}


_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """进程内单例（首次访问时按配置惰性构建）。"""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
