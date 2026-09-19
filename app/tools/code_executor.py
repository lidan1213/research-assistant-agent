"""代码执行工具（沙箱）。

⚠️ 安全说明：
- `subprocess` 模式：本机子进程执行用户代码，**仅适合本地/受信任环境**（默认）；
- `docker` 模式：容器隔离（--network none + 内存/CPU 限制），适合生产部署，
  需本机安装 Docker 且已拉取 python 镜像（首次运行自动尝试拉取）；
- `auto` 模式：检测到 Docker 可用则用容器，否则回退子进程并记录警告。
配置项见 ToolSettings.code_exec_*。
"""
from __future__ import annotations

import asyncio
import shutil
from typing import Any

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.logging import get_logger
from app.tools.base import ToolResult, tool

logger = get_logger("tools.code_executor")


class CodeArgs(BaseModel):
    code: str = Field(..., description="要执行的 Python 代码")
    timeout: int | None = Field(None, ge=1, le=60, description="超时秒数（覆盖默认）")


@tool(name="code_executor", description="执行 Python 代码片段并返回 stdout/stderr", params=CodeArgs)
async def code_executor(code: str, timeout: int | None = None) -> dict[str, Any]:
    limit = timeout or get_settings().tools.code_exec_timeout
    sandbox = get_settings().tools.code_exec_sandbox

    # 容器模式：优先使用（auto 模式下 docker 不可用时回退子进程）
    if sandbox == "docker" or (sandbox == "auto" and await _docker_available()):
        return await _run_in_docker(code, limit)

    if sandbox == "docker":
        return {"success": False, "error": "已配置 docker 沙箱但 Docker 不可用，请检查 Docker 是否已安装并启动"}

    if sandbox == "auto":
        logger.warning("Docker 不可用，code_executor 回退到本机子进程执行（非隔离环境）")

    return await _run_subprocess(code, limit)


async def _docker_available() -> bool:
    """检测 Docker CLI 是否可用（不实际拉镜像）。"""
    if shutil.which("docker") is None:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "version", "--format", "{{.Server.Version}}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


async def _run_in_docker(code: str, timeout: int) -> dict[str, Any]:
    """在 Docker 容器中隔离执行：无网络 + 内存/CPU 限制 + 超时杀掉。"""
    s = get_settings().tools
    image = s.code_exec_docker_image
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", f"{s.code_exec_memory_mb}m",
        "--cpus", str(s.code_exec_cpus),
        "--pids-limit", "64",          # 限制进程数，防 fork 炸弹
        "--cap-drop", "ALL",           # 丢弃全部 Linux capabilities
        "--security-opt", "no-new-privileges",
        "--read-only",                 # 只读根文件系统
        "-i",                          # 从 stdin 读代码
        image, "python", "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(code.encode("utf-8")), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "error": f"执行超时（>{timeout}s）已在容器外杀掉"}
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", "replace"),
            "stderr": stderr.decode("utf-8", "replace"),
            "returncode": proc.returncode,
        }
    except FileNotFoundError:
        return ToolResult(success=False, error="未找到 docker 可执行文件")
    except Exception as e:  # noqa: BLE001
        return ToolResult(success=False, error=f"Docker 执行异常: {e}")


async def _run_subprocess(code: str, timeout: int) -> dict[str, Any]:
    """本机子进程执行（非隔离，仅限受信任环境）。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "error": f"执行超时（>{timeout}s）被杀掉"}
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", "replace"),
            "stderr": stderr.decode("utf-8", "replace"),
            "returncode": proc.returncode,
        }
    except FileNotFoundError:
        return ToolResult(success=False, error="未找到 python 可执行文件")
    except Exception as e:  # noqa: BLE001
        return ToolResult(success=False, error=f"执行异常: {e}")
