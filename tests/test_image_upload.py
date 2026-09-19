"""图片上传 OCR 入库测试（mock OCR 避免真实 LLM 调用）。"""
from __future__ import annotations

import io

import pytest

from app.knowledge.user_kb import is_image_file, ocr_image_text
from tests._chroma_gate import chroma_available

pytestmark = pytest.mark.skipif(
    not chroma_available(),
    reason="ChromaDB REST (127.0.0.1:8001) 未运行",
)


def test_is_image_file():
    assert is_image_file("figure.png")
    assert is_image_file("scan.JPG")
    assert is_image_file("chart.webp")
    assert not is_image_file("paper.txt")
    assert not is_image_file("doc.pdf")


def test_ocr_image_text_failure(monkeypatch):
    """OCR 异常时返回空串（不抛错）。"""

    def _boom(*a, **k):
        raise RuntimeError("LLM 不可用")

    monkeypatch.setattr("app.llm.factory.get_llm", _boom)  # ocr_image_text 内部从 factory import
    import asyncio

    text = asyncio.run(ocr_image_text("x.png", b"fake-image-bytes"))
    assert text == ""


def test_upload_image_goes_through_ocr(client, monkeypatch):
    """上传图片：走 OCR 分支，提取文字入库。"""
    import asyncio

    async def _fake_ocr(filename, data):
        return "量子点太阳能电池效率 30% 瓶颈是稳定性"

    monkeypatch.setattr("app.knowledge.user_kb.ocr_image_text", _fake_ocr)  # 路由函数内 import 的源模块

    token = client.post("/api/auth/login", json={"username": "student2", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    client.delete("/api/userkb/clear", headers=h)
    r = client.post(
        "/api/userkb/upload",
        headers=h,
        data={"save_to_kb": "true"},
        files={"file": ("figure.png", io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "image/png")},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["saved_to_kb"] is True
    assert j["chunks"] > 0
    assert "量子点" in j["message"] or j["chunks"] >= 1

    # 检索命中 OCR 文字
    docs = client.get("/api/userkb/documents", headers=h).json()["documents"]
    assert any("figure.png" in d["source"] for d in docs)
    client.delete("/api/userkb/clear", headers=h)
