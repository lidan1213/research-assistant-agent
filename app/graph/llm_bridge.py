"""LangChain / LangGraph 与现有配置的桥接。

复用 app.config 中的 LLM 配置，构造一个 OpenAI 兼容的
`langchain_openai.ChatOpenAI` 实例，使 LangGraph 工作流能直接使用
与本项目一致的模型、Base URL、Key（支持 openai / deepseek / ollama 等）。

设计说明（架构重构）：
- 本项目存在「自研 LLM 层（app/llm/）」与「LangGraph 编排层（app/graph/）」两套模型入口，
  本桥接是两者唯一的适配点；
- LangGraph 需要 langchain_core.ChatModel 接口（bind_tools / astream），无法直接复用
  自研 BaseLLM，因此这里保持 ChatOpenAI 适配，但模型选择（主/辅、任务类型路由）统一
  通过 app.llm.gateway 的决策，避免业务代码各自选模型；
- usage/trace 由 ChatOpenAI 内部调用 openai SDK 记录（与自研层一致走 UsageLedger）。
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache
def get_langchain_llm():
    """返回（并缓存）一个 LangChain ChatOpenAI 实例。"""
    from langchain_openai import ChatOpenAI

    s = get_settings().llm
    return ChatOpenAI(
        model=s.model,
        base_url=s.base_url,
        api_key=s.api_key or "EMPTY",
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        max_retries=2,
    )


def get_configured_model(temperature: float | None = None):
    """在需要时基于配置克隆一个指定温度的模型。"""
    llm = get_langchain_llm()
    return llm.with_config({"temperature": temperature}) if temperature is not None else llm
