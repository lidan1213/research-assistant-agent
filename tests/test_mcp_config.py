"""MCP configuration parsing and per-server environment isolation."""
from __future__ import annotations

import json


def test_parse_mcp_server_config(monkeypatch):
    from app.tools.mcp import _parse_config

    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps([
            {
                "name": "filesystem",
                "command": "cmd",
                "args": ["/c", "npx", "server", "D:/data"],
                "env": {"MODE": "readonly"},
            }
        ]),
    )
    servers = _parse_config()
    assert servers == [{
        "name": "filesystem",
        "command": "cmd",
        "args": ["/c", "npx", "server", "D:/data"],
        "env": {"MODE": "readonly"},
    }]


def test_parse_mcp_rejects_invalid_shapes(monkeypatch):
    from app.tools.mcp import _parse_config

    monkeypatch.setenv("MCP_SERVERS", '{"name":"not-a-list"}')
    assert _parse_config() == []
    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"name":"bad","command":"x","args":"not-a-list"}]',
    )
    assert _parse_config() == []
