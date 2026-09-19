"""MCP 工具接入测试（不依赖真实 MCP 服务器）。"""
from __future__ import annotations

import asyncio

from app.tools.base import ToolRegistry, ToolResult


def test_parse_config_valid(monkeypatch):
    """合法 JSON 配置解析出服务器列表。"""
    from app.tools.mcp import _parse_config

    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"name":"fetch","command":"npx","args":["-y","@modelcontextprotocol/server-fetch"]}]',
    )
    servers = _parse_config()
    assert len(servers) == 1
    assert servers[0]["name"] == "fetch"
    assert servers[0]["command"] == "npx"


def test_parse_config_invalid(monkeypatch):
    """非法 JSON 配置返回空列表。"""
    from app.tools.mcp import _parse_config

    monkeypatch.setenv("MCP_SERVERS", "not-json{{{")
    assert _parse_config() == []


def test_parse_config_empty(monkeypatch):
    """未配置返回空列表。"""
    from app.tools.mcp import _parse_config

    # 显式空数组覆盖项目 .env，保证测试与开发机配置隔离。
    monkeypatch.setenv("MCP_SERVERS", "[]")
    assert _parse_config() == []


def test_mcp_tool_wraps_schema():
    """MCPTool 包装：name 前缀 + schema 透传 + run 转发到未连接服务器报错。"""
    from app.tools.mcp import MCPTool

    t = MCPTool(
        "fetch",
        {"name": "fetch_url", "description": "抓取网页", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    )
    assert t.name == "mcp_fetch_fetch_url"
    assert t.description.startswith("[MCP fetch]")
    schema = t.parameters_schema()
    assert schema["properties"]["url"]["type"] == "string"

    # 服务器未连接（测试环境不启动真实服务器）→ 返回失败 ToolResult
    result = asyncio.run(t.run(url="https://example.com"))
    assert isinstance(result, ToolResult)
    assert not result.success


def test_ensure_no_config_registers_nothing(monkeypatch):
    """未配置 MCP_SERVERS 时 ensure_mcp_tools 返回 0 且不注册任何工具。"""
    monkeypatch.setenv("MCP_SERVERS", "[]")
    from app.tools.mcp import MCPManager, ensure_mcp_tools

    # 重置单例，避免受其他测试配置影响
    MCPManager._instance = None  # noqa: SLF001
    reg = ToolRegistry()
    n = asyncio.run(ensure_mcp_tools(reg))
    assert n == 0
    assert reg.list() == []
