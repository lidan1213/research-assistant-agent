"""LLM Gateway / Model Router / Structured Output 测试。"""
from __future__ import annotations

import asyncio

import pytest

from app.llm.router import ModelRouter, TaskType
from app.llm.structured_output import (
    OutputParseError,
    extract_json_text,
    parse,
    parse_and_validate,
    require_fields,
)


# ---------- Model Router ----------
def test_router_default_strategy():
    r = ModelRouter()
    assert r.uses_aux(TaskType.QUERY_REWRITE) is True
    assert r.uses_aux(TaskType.ENTITY_EXTRACTION) is True
    assert r.uses_aux(TaskType.SUMMARIZATION) is True
    assert r.uses_aux(TaskType.REASONING) is False
    assert r.uses_aux(TaskType.FINAL_ANSWER) is False
    assert r.uses_aux(None) is False  # 默认走主模型


def test_router_override_and_force_main():
    r = ModelRouter(overrides={"reasoning": True})
    assert r.uses_aux("reasoning") is True
    d = r.route(TaskType.REASONING, force_main=True)
    assert d["use_aux"] is False


def test_router_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_OVERRIDES", '{"reasoning": true}')
    r = ModelRouter()
    assert r.uses_aux("reasoning") is True


# ---------- Structured Output ----------
def test_parse_plain_json():
    assert parse('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    assert parse('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_noisy_prefix():
    assert parse('以下是结果：\n{"a": 1}\n完毕') == {"a": 1}


def test_parse_array():
    assert parse('[{"a": 1}]') == [{"a": 1}]


def test_parse_trailing_comma_repaired():
    assert parse('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_parse_garbage_raises():
    with pytest.raises(OutputParseError):
        parse("抱歉，我无法生成结构化输出")


def test_parse_and_validate_fields():
    data, err = parse_and_validate(
        '{"name": "x"}', require_fields("name", "type")
    )
    assert data == {"name": "x"}
    assert err == "缺少必填字段: type"

    data, err = parse_and_validate('{"name": "x", "type": "y"}', require_fields("name"))
    assert err is None and data["type"] == "y"


def test_parse_and_validate_garbage():
    data, err = parse_and_validate("不是 JSON", require_fields("name"))
    assert data is None and err is not None


def test_extract_json_text_fence():
    assert extract_json_text('xx ```json\n{"k": 1}\n``` yy') == '{"k": 1}'


# ---------- Gateway（stub LLM 验证路由与调用链） ----------
class StubMainLLM:
    model = "main-model"

    async def chat(self, messages, **kwargs):
        from app.llm.base import LLMResponse

        return LLMResponse(content='{"ok": true}')


class StubAuxLLM:
    model = "aux-model"

    async def chat(self, messages, **kwargs):
        from app.llm.base import LLMResponse

        return LLMResponse(content="aux response")


def test_gateway_routes_to_aux_for_simple_task(monkeypatch):
    """查询改写类任务应走辅助模型。"""
    from app.llm import gateway as gw

    calls = {"aux": 0, "main": 0}

    def fake_get_aux_llm():
        calls["aux"] += 1
        return StubAuxLLM()

    def fake_get_llm():
        calls["main"] += 1
        return StubMainLLM()

    monkeypatch.setattr("app.llm.factory.get_aux_llm", fake_get_aux_llm)
    monkeypatch.setattr("app.llm.factory.get_llm", fake_get_llm)

    from app.llm.base import ChatMessage, MessageRole

    async def run():
        g = gw.LLMGateway()
        await g.chat(
            [ChatMessage(role=MessageRole.USER, content="hi")],
            task_type=TaskType.QUERY_REWRITE,
        )
        await g.chat(
            [ChatMessage(role=MessageRole.USER, content="complex")],
            task_type=TaskType.REASONING,
        )
        # 强制主模型
        await g.chat(
            [ChatMessage(role=MessageRole.USER, content="x")],
            task_type=TaskType.QUERY_REWRITE,
            force_main=True,
        )

    asyncio.run(run())
    assert calls["aux"] == 1
    assert calls["main"] == 2


def test_gateway_aux_failure_falls_back_and_opens_circuit(monkeypatch):
    """辅助模型失败时当次回退主模型，熔断期后续请求不再触发 aux。"""
    from app.llm import gateway as gw

    calls = {"aux_chat": 0, "main_chat": 0}

    class FailingAux:
        model = "aux"

        async def chat(self, messages, **kwargs):
            calls["aux_chat"] += 1
            raise RuntimeError("429 cooldown")

    class WorkingMain:
        model = "main"

        async def chat(self, messages, **kwargs):
            from app.llm.base import LLMResponse

            calls["main_chat"] += 1
            return LLMResponse(content="fallback")

    monkeypatch.setattr("app.llm.factory.get_aux_llm", lambda: FailingAux())
    monkeypatch.setattr("app.llm.factory.get_llm", lambda **kwargs: WorkingMain())

    async def run():
        from app.llm.base import ChatMessage, MessageRole

        gateway = gw.LLMGateway()
        messages = [ChatMessage(role=MessageRole.USER, content="hi")]
        first = await gateway.chat(messages, task_type=TaskType.QUERY_REWRITE)
        second = await gateway.chat(messages, task_type=TaskType.QUERY_REWRITE)
        assert first.content == second.content == "fallback"

    asyncio.run(run())
    assert calls == {"aux_chat": 1, "main_chat": 2}


def test_gateway_chat_json_retry_on_garbage(monkeypatch):
    """chat_json 解析失败按 max_retries 重试。"""
    from app.llm import gateway as gw

    responses = ["不是 JSON", '{"name": "ok", "type": "t"}']

    class StubLLM:
        model = "m"

        async def chat(self, messages, **kwargs):
            from app.llm.base import LLMResponse

            return LLMResponse(content=responses.pop(0))

    monkeypatch.setattr("app.llm.factory.get_llm", lambda **k: StubLLM())
    monkeypatch.setattr("app.llm.factory.get_aux_llm", lambda **k: StubLLM())

    async def run():
        g = gw.LLMGateway()
        data, err = await g.chat_json(
            "prompt", validator=require_fields("name", "type"), max_retries=1
        )
        assert err is None
        assert data["name"] == "ok"

    asyncio.run(run())


def test_gateway_chat_json_exhausts_retries(monkeypatch):
    from app.llm import gateway as gw

    class StubLLM:
        model = "m"

        async def chat(self, messages, **kwargs):
            from app.llm.base import LLMResponse

            return LLMResponse(content="永远不是 JSON")

    monkeypatch.setattr("app.llm.factory.get_llm", lambda **k: StubLLM())
    monkeypatch.setattr("app.llm.factory.get_aux_llm", lambda **k: StubLLM())

    async def run():
        g = gw.LLMGateway()
        data, err = await g.chat_json(
            "prompt", validator=require_fields("name"), max_retries=1
        )
        assert data is None
        assert err is not None

    asyncio.run(run())
