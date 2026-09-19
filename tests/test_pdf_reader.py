"""pdf_reader 真实功能测试（离线）。

覆盖：
- 本地路径读取多页 PDF，返回结构化结果（页数/正文/是否截断）。
- URL 分支：在本机起一个临时 HTTP 服务，验证从链接读取与临时文件清理。
- 错误处理：路径不存在返回失败。
"""
import asyncio
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.tools.pdf_reader import pdf_reader
from app.tools.registry import registry

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_paper.pdf"


@pytest.fixture
def local_server():
    """在 fixtures 目录起一个仅本机、测试用的 HTTP 服务。"""
    handler = lambda *a, **k: SimpleHTTPRequestHandler(*a, directory=str(FIXTURE.parent), **k)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/sample_paper.pdf"
    server.shutdown()


@pytest.mark.asyncio
async def test_read_local_pdf_returns_structure():
    res = await registry.call("pdf_reader", json.dumps({"path": str(FIXTURE)}))
    assert res.success, res.error
    out = res.output
    assert isinstance(out, dict)
    assert out["num_pages"] >= 1
    assert "Research Assistant Agent" in out["text"]
    assert "Retrieval-augmented generation" in out["text"]
    # 默认 max_chars 不会把整本塞进来（截断标志按需出现，但文本不超上限）
    assert len(out["text"]) <= 6000


@pytest.mark.asyncio
async def test_read_from_url(local_server):
    res = await registry.call("pdf_reader", json.dumps({"url": local_server}))
    assert res.success, res.error
    out = res.output
    assert "Multi-Agent Coordination" in out["text"]
    assert out["source"].startswith("http://")


@pytest.mark.asyncio
async def test_max_chars_truncation():
    res = await registry.call(
        "pdf_reader", json.dumps({"path": str(FIXTURE), "max_chars": 120})
    )
    assert res.success
    out = res.output
    assert out["truncated"] is True
    assert len(out["text"]) <= 120


@pytest.mark.asyncio
async def test_missing_file_returns_failure():
    res = await registry.call("pdf_reader", json.dumps({"path": "C:/no/such/file.pdf"}))
    assert res.success is False
    assert "文件" in res.error or "获取" in res.error
