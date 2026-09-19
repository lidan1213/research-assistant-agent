"""启动脚本：python scripts/run.py。

等价于：uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

if __name__ == "__main__":
    import platform

    s = get_settings()
    dev = s.app_env == "development"
    # Windows 不支持 uvicorn 多 worker（WinError 10022），强制单 worker；
    # 防卡死靠 agent._run_tool 的 asyncio.wait_for 工具超时（见 agent.py）。
    # Linux/macOS 生产环境可用 2 worker 提升并发容错。
    workers = 1 if (dev or platform.system() == "Windows") else 2
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=s.host,
        port=s.port,
        reload=dev,
        workers=workers,
        log_level=s.log_level.lower(),
    )
