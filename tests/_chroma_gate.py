"""ChromaDB 测试前置门禁（独立模块，避免 conftest 导入机制差异）。"""
from __future__ import annotations

import os
import urllib.request

CHROMA_URL = os.environ.get("CHROMA_URL", "http://127.0.0.1:8001")


def chroma_available() -> bool:
    """探测 ChromaDB REST 服务是否存活（测试前门禁，避免挂掉后出现误导性断言）。"""
    try:
        with urllib.request.urlopen(f"{CHROMA_URL}/api/v1/heartbeat", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False
