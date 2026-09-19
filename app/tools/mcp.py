"""MCP（Model Context Protocol）工具接入。

把外部 MCP 服务器（stdio 子进程，如 npx 启动的官方服务器）暴露的工具注册进
Agent 的 ToolRegistry，Agent 即可像调用内置工具一样调用它们。

配置（.env 或环境变量 MCP_SERVERS，JSON 数组）：
    MCP_SERVERS=[{"name":"fetch","command":"npx","args":["-y","@modelcontextprotocol/server-fetch"]}]

接入方式：WS 连接建立后调用 `await ensure_mcp_tools(registry)`（幂等）。
- 未配置 MCP_SERVERS / mcp 未安装 / 服务器启动失败：静默跳过，不影响主流程；
- 工具名为 mcp_{server}_{tool}，schema 直接透传 MCP 服务器声明；
- 会话进程级持有（stdio 子进程 + ClientSession），首个工具调用时惰性启动。
"""
from __future__ import annotations

import asyncio
import json
import os

from app.core.logging import get_logger
from app.tools.base import BaseTool, ToolResult, ToolRegistry

logger = get_logger("tools.mcp")

try:  # mcp 未安装时降级：本模块空转
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # noqa: BLE001
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]

_REGISTERED = False  # 进程内只注册一次


def _parse_config() -> list[dict]:
    raw = os.environ.get("MCP_SERVERS", "").strip()
    if not raw:
        try:
            from app.config import get_settings

            raw = get_settings().mcp_servers.strip()
        except Exception:  # noqa: BLE001
            raw = ""
    if not raw:
        return []
    try:
        servers = json.loads(raw)
        if not isinstance(servers, list):
            return []
        out = []
        for server in servers:
            if not isinstance(server, dict) or not server.get("name") or not server.get("command"):
                continue
            args = server.get("args") or []
            env = server.get("env") or {}
            if not isinstance(args, list) or not isinstance(env, dict):
                logger.warning(f"MCP 服务器参数无效，已忽略: {server.get('name')}")
                continue
            out.append({
                "name": str(server["name"]),
                "command": str(server["command"]),
                "args": [str(arg) for arg in args],
                "env": {str(key): str(value) for key, value in env.items()},
            })
        return out
    except json.JSONDecodeError as e:
        logger.warning(f"MCP_SERVERS 配置不是合法 JSON，已忽略: {e}")
        return []


class MCPManager:
    """进程级单例：启动并持有各 MCP 服务器会话（stdio）。"""

    _instance: "MCPManager | None" = None

    def __init__(self) -> None:
        self._servers = _parse_config()
        self._sessions: dict[str, ClientSession] = {}
        self._cm_handles: dict[str, tuple] = {}  # 持有 stdio 上下文防关闭
        self._lock = asyncio.Lock()
        self._started = False

    @classmethod
    def get(cls) -> "MCPManager":
        if cls._instance is None:
            cls._instance = MCPManager()
        return cls._instance

    def configured(self) -> bool:
        return bool(self._servers)

    async def ensure(self) -> None:
        """启动所有已配置的 MCP 服务器（幂等）。"""
        if self._started or not self._servers or ClientSession is None:
            return
        async with self._lock:
            if self._started:
                return
            for srv in self._servers:
                name = srv["name"]
                try:
                    params = StdioServerParameters(
                        command=srv["command"],
                        args=srv.get("args") or [],
                        # 继承主进程环境，并允许每个 MCP Server 单独覆盖变量。
                        env={**os.environ, **(srv.get("env") or {})},
                    )
                    cm = stdio_client(params)
                    read, write = await cm.__aenter__()
                    session = ClientSession(read, write)
                    await session.__aenter__()
                    await session.initialize()
                    self._sessions[name] = session
                    self._cm_handles[name] = (cm, read, write)
                    logger.info(f"MCP 服务器已连接: {name} ({srv['command']})")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"MCP 服务器启动失败（跳过）: {name} -> {e}")
            self._started = True

    async def list_tools(self) -> list[dict]:
        """返回 [{server, name, description, input_schema}]。"""
        await self.ensure()
        out: list[dict] = []
        for name, session in self._sessions.items():
            try:
                tools = await session.list_tools()
                for t in tools.tools:
                    out.append(
                        {
                            "server": name,
                            "name": t.name,
                            "description": t.description or "",
                            "input_schema": t.inputSchema or {},
                        }
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"MCP 工具列表获取失败（{name}）: {e}")
        return out

    async def call_tool(self, server: str, tool: str, arguments: dict) -> ToolResult:
        await self.ensure()
        session = self._sessions.get(server)
        if session is None:
            return ToolResult(success=False, error=f"MCP 服务器未连接: {server}", retryable=False)
        try:
            result = await session.call_tool(tool, arguments)
            if result.isError:
                return ToolResult(success=False, error=str(result.content), retryable=False)
            texts = [
                c.text for c in (result.content or []) if hasattr(c, "text") and c.text
            ]
            return ToolResult(success=True, output="\n".join(texts) if texts else str(result.content))
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"MCP 调用失败: {e}")


class MCPTool(BaseTool):
    """把一个 MCP 工具包装成 Agent 可调用的 BaseTool。"""

    def __init__(self, server: str, tool: dict) -> None:
        self._server = server
        self._tool_name = tool["name"]
        self.name = f"mcp_{server}_{tool['name']}"
        self.description = f"[MCP {server}] {tool['description'] or tool['name']}"
        self._schema = tool.get("input_schema") or {"type": "object", "properties": {}}

    def parameters_schema(self) -> dict:
        return self._schema

    async def run(self, **kwargs) -> ToolResult:
        return await MCPManager.get().call_tool(self._server, self._tool_name, kwargs)


async def ensure_mcp_tools(registry: ToolRegistry) -> int:
    """把配置的 MCP 工具注册进 registry（幂等；失败静默）。返回注册数。"""
    global _REGISTERED
    if _REGISTERED or ClientSession is None:
        return 0
    mgr = MCPManager.get()
    if not mgr.configured():
        _REGISTERED = True
        return 0
    try:
        tools = await mgr.list_tools()
        for t in tools:
            if registry.get(f"mcp_{t['server']}_{t['name']}"):
                continue
            registry.register(MCPTool(t["server"], t))
        _REGISTERED = True
        if tools:
            names = ", ".join(f"mcp_{t['server']}_{t['name']}" for t in tools[:5])
            logger.info(f"MCP 工具已注册: {len(tools)} 个（{names}…）")
        return len(tools)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"MCP 工具注册失败（跳过）: {e}")
        _REGISTERED = True
        return 0
