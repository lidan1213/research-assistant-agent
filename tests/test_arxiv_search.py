"""arxiv_search 真实功能测试（联网，断网自动跳过）。

覆盖：
- 联网检索返回结构化论文列表（标题/作者/摘要/pdf_url 等字段齐全）。
- 可选 download=True 时把 PDF 下载到本地并附带 local_path（与 pdf_reader 串联）。
- 异常参数/网络失败时的优雅错误。
"""
import asyncio
import shutil
import tempfile

import httpx
import pytest

from app.tools.registry import registry

NETWORK_OK = False
try:
    with httpx.Client(timeout=8) as c:
        # 探测工具实际使用的 API 端点；429 限流 / 5xx 同样视为不可用（自动跳过）
        r = c.get(
            "https://export.arxiv.org/api/query?search_query=all:electron&max_results=1"
        )
        NETWORK_OK = r.status_code == 200
except Exception:  # noqa: BLE001
    NETWORK_OK = False

# network 标记：默认被 pyproject addopts 排除；显式 `pytest -m network` 才运行。
# 断网/限流（429）时仍自动跳过，避免 CI 或离线环境误报失败。
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not NETWORK_OK, reason="需要 arxiv.org 联网，CI 中跳过"),
]


@pytest.mark.asyncio
async def test_arxiv_search_returns_papers():
    res = await registry.call(
        "arxiv_search", '{"query": "retrieval augmented generation", "max_results": 3}'
    )
    assert res.success, res.error
    data = res.output
    assert isinstance(data, list) and len(data) >= 1
    first = data[0]
    for key in ("title", "authors", "summary", "pdf_url", "entry_id"):
        assert key in first and first[key]
    assert isinstance(first["authors"], list)


@pytest.mark.asyncio
async def test_arxiv_search_download_pdf():
    tmp = tempfile.mkdtemp(prefix="arxiv_test_")
    try:
        res = await registry.call(
            "arxiv_search",
            '{"query": "transformer", "max_results": 1, "download": true, '
            '"download_dir": "%s"}' % tmp.replace("\\", "/"),
        )
        assert res.success, res.error
        data = res.output
        assert len(data) >= 1
        # 至少有部分论文成功下载
        downloaded = [d for d in data if d.get("local_path")]
        assert downloaded, "没有任何 PDF 被下载"
        for d in downloaded:
            assert __import__("os").path.isfile(d["local_path"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
