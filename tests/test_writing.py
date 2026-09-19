"""论文写作接口测试（mock LLM 返回固定内容）。"""
from __future__ import annotations

from app.llm.base import LLMResponse


def _install_stub_llm(monkeypatch, content: str):
    """把 writing 路由（经 LLMGateway）使用的 get_llm 换成固定返回的 stub。

    Gateway 内部从 app.llm.factory import get_llm，因此 stub 工厂层。
    """

    class _Stub:
        async def chat(self, *a, **k):
            return LLMResponse(content=content)

        async def stream(self, *a, **k):
            yield LLMResponse(content=content)

    monkeypatch.setattr("app.llm.factory.get_llm", lambda *a, **k: _Stub())
    monkeypatch.setattr("app.llm.factory.get_aux_llm", lambda *a, **k: _Stub())


def test_writing_outline_parses_json(client, monkeypatch):
    """大纲：LLM 返回 JSON → 解析为章节列表。"""
    _install_stub_llm(
        monkeypatch,
        '[{"id":"s1","title":"引言","description":"研究背景"},{"id":"s2","title":"方法","description":"实验设计"}]',
    )
    token = client.post(
        "/api/auth/login", json={"username": "student1", "password": "123456"}
    ).json()["token"]
    r = client.post(
        "/api/writing/outline",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"topic": "量子点电池", "paper_type": "综述"},
    )
    assert r.status_code == 200
    j = r.json()
    assert len(j["sections"]) == 2
    assert j["sections"][0]["title"] == "引言"


def test_writing_outline_tolerates_codeblock(client, monkeypatch):
    """大纲：LLM 输出带 markdown 代码块也能解析。"""
    _install_stub_llm(monkeypatch, '```json\n[{"id":"s1","title":"结论","description":"总结"}]\n```')
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    r = client.post(
        "/api/writing/outline",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"topic": "测试", "paper_type": "研究论文"},
    )
    assert r.status_code == 200
    assert r.json()["sections"][0]["title"] == "结论"


def test_writing_draft_and_cite(client, monkeypatch):
    """草稿 + 引用格式化返回内容。"""
    _install_stub_llm(monkeypatch, "这是草稿内容。")
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    r1 = client.post("/api/writing/draft", headers=h, json={
        "topic": "量子点电池", "paper_type": "综述",
        "section": {"id": "s1", "title": "引言", "description": "背景"},
        "outline": [{"id": "s1", "title": "引言", "description": "背景"}],
    })
    assert r1.status_code == 200
    assert r1.json()["content"] == "这是草稿内容。"

    r2 = client.post("/api/writing/cite", headers=h, json={
        "refs": ["Quantum dots for solar cells, 2023"], "style": "apa",
    })
    assert r2.status_code == 200
    assert r2.json()["formatted"]


def test_writing_export_assembles_md(client):
    """导出：拼装完整 Markdown 并下载。"""
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    r = client.post(
        "/api/writing/export",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "topic": "量子点电池研究", "paper_type": "综述",
            "sections": [{"title": "引言", "content": "背景内容"}, {"title": "结论", "content": "总结内容"}],
        },
    )
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")
    assert "# 量子点电池研究" in r.text
    assert "## 引言" in r.text
    assert "背景内容" in r.text


def test_writing_export_docx_and_pdf(client):
    """Word/PDF 导出具有正确格式、文件名和可解析内容。"""
    from io import BytesIO

    from docx import Document
    from pypdf import PdfReader

    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "topic": "量子点电池研究", "paper_type": "综述",
        "sections": [{"title": "引言", "content": "背景内容\n\n- 要点一\n- 要点二"}],
    }

    docx_response = client.post(
        "/api/writing/export", headers=headers, json={**payload, "format": "docx"}
    )
    assert docx_response.status_code == 200
    assert "wordprocessingml.document" in docx_response.headers["content-type"]
    doc = Document(BytesIO(docx_response.content))
    assert "量子点电池研究" in "\n".join(p.text for p in doc.paragraphs)
    assert "filename*=UTF-8''" in docx_response.headers["content-disposition"]

    pdf_response = client.post(
        "/api/writing/export", headers=headers, json={**payload, "format": "pdf"}
    )
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"
    assert pdf_response.content.startswith(b"%PDF")
    assert len(PdfReader(BytesIO(pdf_response.content)).pages) >= 1
