"""记忆子系统：长期记忆（跨会话）与统一记忆管理器。

分层：
- L0 工作记忆：单次运行内的推理缓冲（ReAct step / 多 Agent 黑板+链路），运行结束即释放。
- L1 会话记忆：ConversationMemory（见 app.agent.memory），可内存 / SQLite / Redis。
- L2 长期记忆：LongTermMemory（本模块），跨会话沉淀的事实、结论、文献要点。
- L3 元索引：会话登记表（sessions），用于跨会话定位与召回。
"""
from app.memory.longterm import LongTermMemory
from app.memory.manager import MemoryManager, get_memory_manager

__all__ = ["LongTermMemory", "MemoryManager", "get_memory_manager"]
