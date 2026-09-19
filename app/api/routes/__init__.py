"""路由聚合。"""
from app.api.routes import agent, chat, graph, knowledge, tools, ws

__all__ = ["chat", "agent", "tools", "knowledge", "ws", "graph"]
