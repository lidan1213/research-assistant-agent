"""文档版本更新测试：同名文件重新上传 → 替换旧版本。"""
from __future__ import annotations

import io

import pytest

from tests._chroma_gate import chroma_available

pytestmark = pytest.mark.skipif(
    not chroma_available(),
    reason="ChromaDB REST (127.0.0.1:8001) 未运行",
)


def _upload(client, token, filename, content):
    return client.post(
        "/api/userkb/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"save_to_kb": "true"},
        files={"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")},
    )


def test_upload_version_replace(client):
    """同名文件不同内容 → 版本更新（删旧入新），列表只有一个新版本。"""
    token = client.post("/api/auth/login", json={"username": "student3", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    client.delete("/api/userkb/clear", headers=h)

    r1 = _upload(client, token, "paper_v2.txt", "量子点电池效率 30%。")
    assert r1.status_code == 200
    assert r1.json()["version_updated"] is False
    j1 = r1.json()
    assert j1["chunks"] >= 1

    # 同内容再次上传 → 去重拒绝（duplicate）
    rdup = _upload(client, token, "paper_v2.txt", "量子点电池效率 30%。")
    assert rdup.json()["duplicate"] is True

    # 同名不同内容 → 版本更新
    r2 = _upload(client, token, "paper_v2.txt", "量子点电池效率已提升至 35%，稳定性改善。")
    assert r2.status_code == 200
    assert r2.json()["version_updated"] is True
    assert r2.json()["chunks"] >= 1

    # 列表只有一份 paper_v2.txt（旧版本已替换）
    docs = client.get("/api/userkb/documents", headers=h).json()["documents"]
    same = [d for d in docs if d["source"] == "paper_v2.txt"]
    assert len(same) == 1
    assert "35%" in same[0]["text"] or "35%" in str(same[0])

    # 检索能命中新内容（search 接口用表单提交）
    hits = client.post("/api/userkb/search", headers=h, data={"query": "稳定性改善", "top_k": "3"}).json()
    assert any(
        "paper_v2.txt" in ((x.get("metadata") or {}).get("source", "") or "")
        for x in hits.get("hits", [])
    )

    client.delete("/api/userkb/clear", headers=h)
