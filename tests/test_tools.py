"""工具逻辑单元测试（无需联网/LLM）。"""
import asyncio

import pytest

from app.tools.registry import registry


def test_calculator_safe():
    res = asyncio.run(registry.call("calculator", '{"expression":"(1+2)*3"}'))
    assert res.success
    assert "9" in res.output


def test_calculator_rejects_code():
    # 不应执行危险的属性访问 / 调用
    res = asyncio.run(registry.call("calculator", '{"expression":"__import__(\'os\').system(\'echo hi\')"}'))
    assert not res.success


def test_citation_dedup():
    payload = {
        "papers": [
            {"title": "A Survey on RAG", "authors": ["X"], "year": 2024},
            {"title": "A Survey on RAG", "authors": ["X"], "year": 2024},
            {"title": "LLM Agents", "authors": ["Y"], "year": 2025},
        ],
        "style": "apa",
    }
    res = asyncio.run(registry.call("citation", __import__("json").dumps(payload)))
    assert res.success
    assert res.output["count"] == 2
    assert res.output["duplicates_removed"] == 1


def test_unknown_tool():
    res = asyncio.run(registry.call("nonexistent", "{}"))
    assert not res.success
