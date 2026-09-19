"""科研助手 Agent 框架。

一个基于 FastAPI 的、可扩展的科研助手 Agent 骨架，覆盖：
- LLM 抽象与多供应商接入
- 工具系统（Tool Calling / Function Calling）
- 短期 + 长期记忆
- 规划 + ReAct 推理循环
- 知识库检索（RAG）
- 同步 / SSE / WebSocket 多种交互通道
"""

__version__ = "0.1.0"
