"""LLM 接入层：抽象接口 + OpenAI 兼容实现 + 工厂。"""
from app.llm.base import BaseLLM, ChatMessage, MessageRole
from app.llm.factory import get_llm
from app.llm.openai_provider import OpenAICompatibleLLM

__all__ = ["BaseLLM", "ChatMessage", "MessageRole", "OpenAICompatibleLLM", "get_llm"]
