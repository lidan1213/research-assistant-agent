"""LLM 工厂：按配置选择并缓存供应商实例（含多模型路由）。

- `get_llm()`：主模型（对话主链路）；
- `get_aux_llm()`：辅助模型（内部低价值调用：摘要/事实提取/查询改写/强制收尾），
  通过 `LLMSettings.aux_*` 配置；未配置时回退主模型。
- 缓存键为 (provider, base_url, api_key, model)，不同模型各自实例。
"""
from __future__ import annotations

from app.config import get_settings
from app.core.exceptions import AgentError
from app.llm.base import BaseLLM
from app.llm.openai_provider import OpenAICompatibleLLM

_REGISTRY: dict[str, type[BaseLLM]] = {
    "openai": OpenAICompatibleLLM,
    "deepseek": OpenAICompatibleLLM,  # 同为 OpenAI 兼容协议
    "ollama": OpenAICompatibleLLM,
    "vllm": OpenAICompatibleLLM,
}

_cache: dict[tuple, BaseLLM] = {}


def register_llm(provider: str, cls: type[BaseLLM]) -> None:
    """运行时注册新的 LLM 供应商。"""
    _REGISTRY[provider.lower()] = cls


def _build_instance(provider: str, base_url: str, api_key: str, model: str) -> BaseLLM:
    if provider.lower() == "langchain":
        # LangChain 桥：适配成本项目 BaseLLM 接口（ChatOpenAI 内部实现）
        from app.llm.langchain_adapter import LangChainChatModel

        return LangChainChatModel(
            base_url=base_url or None,
            api_key=api_key or None,
            model=model or None,
        )
    cls = _REGISTRY.get(provider.lower())
    if cls is None:
        raise AgentError(f"未注册的 LLM 供应商: {provider}", code="LLM_NOT_FOUND")
    # OpenAI 兼容供应商：显式传参可覆盖默认配置；其余供应商只按 provider 实例化
    if cls is OpenAICompatibleLLM:
        return cls(
            base_url=base_url or None,
            api_key=api_key or None,
            model=model or None,
        )
    return cls()


def get_llm(
    provider: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseLLM:
    """获取（并缓存）LLM 实例。

    参数缺省时使用主配置（LLMSettings.llm.*）；传入非空参数可覆盖（多模型路由用）。
    """
    settings = get_settings().llm
    key = (
        provider or settings.provider,
        base_url or settings.base_url,
        api_key or settings.api_key,
        model or settings.model,
    )
    if key in _cache:
        return _cache[key]
    instance = _build_instance(*key)
    _cache[key] = instance
    return instance


def get_aux_llm() -> BaseLLM:
    """辅助模型实例（ModelRouter）：内部低价值调用用便宜模型，降低成本。

    未配置 aux_model 时回退主模型（行为与未启用路由前完全一致）。
    """
    s = get_settings().llm
    if not s.aux_model:
        return get_llm()
    return get_llm(
        provider=s.aux_provider or None,
        model=s.aux_model,
        base_url=s.aux_base_url or None,
        api_key=s.aux_api_key or None,
    )
