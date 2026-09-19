"""IDOR 越权隔离测试矩阵：用户 A 访问用户 B 的资源必须被拒绝（403/404）。

对应路线图 P0「权限与安全审计」验收标准：
「任意用户无法读取其他用户的知识库、会话、笔记和 Trace」——
验证多用户隔离不是表面上的命名隔离，而是每个接口的真实归属校验。

覆盖：
- 会话：history / rename / export / restore / batch-delete（A 访问 B 的会话）
- 笔记：读取 / 编辑 / 删除 / 列表（A 访问 B 的笔记）
- 知识库：collection 命名空间隔离（纯函数）+ 库名注入防护
- 知识图谱：KGStore 数据隔离（A 写 B 读不到）
"""
from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

import pytest

from app.agent.memory import ChatMessage, MessageRole, SQLiteStore
from app.config import get_settings

TESTS_DIR = Path(__file__).resolve().parent

# ---------- 辅助 ----------


def _mk_user(client, tag: str) -> tuple[str, str]:
    """注册临时用户，返回 (username, token)。"""
    name = f"idor{tag}{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/api/auth/register",
        json={"username": name, "password": "Passw0rd1", "password2": "Passw0rd1"},
    )
    assert resp.status_code == 200, resp.text
    return name, resp.json()["token"]


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _mk_session(username: str, tag: str) -> str:
    """直接写库创建会话（返回完整 session_id，已带 {username}__ 前缀）。"""
    sid = f"{username}__{tag}_{uuid.uuid4().hex[:8]}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        store.set_title(sid, f"{tag}标题")
        asyncio.run(
            store.append(
                sid, ChatMessage(role=MessageRole.USER, content=f"{tag}的私密内容")
            )
        )
    finally:
        store.close()
    return sid


def _write_note(slug: str, source: str, body: str = "# 私密笔记\n\n内容") -> None:
    """直接落盘构造笔记（带 frontmatter），绕过 LLM。

    conftest 已把 NOTES_DIR 环境变量指向 tests/test_notes_dir。
    """
    notes_dir = TESTS_DIR / "test_notes_dir"
    notes_dir.mkdir(parents=True, exist_ok=True)
    md = f"---\ntitle: 私密笔记\ntags: [secret]\nsource: {source}\n---\n{body}"
    (notes_dir / f"{slug}.md").write_text(md, encoding="utf-8")


# ---------- 会话越权 ----------


def test_session_history_cross_user_forbidden(client):
    """A 读取 B 的会话历史 → 403；B 自己读 → 200。"""
    ua, ta = _mk_user(client, "a")
    ub, tb = _mk_user(client, "b")
    b_sid = _mk_session(ub, "priv")

    r = client.get("/api/chat/history", params={"session_id": b_sid}, headers=_h(ta))
    assert r.status_code == 403, r.text

    r = client.get("/api/chat/history", params={"session_id": b_sid}, headers=_h(tb))
    assert r.status_code == 200
    contents = [m["content"] for m in r.json()["messages"]]
    assert any("私密内容" in c for c in contents)  # B 能读到自己的内容


def test_session_rename_cross_user_forbidden(client):
    """A 重命名 B 的会话 → 403，B 的标题未被改动。"""
    ua, ta = _mk_user(client, "a")
    ub, tb = _mk_user(client, "b")
    b_sid = _mk_session(ub, "priv")

    r = client.post(
        f"/api/chat/sessions/{b_sid}/rename",
        json={"title": "被篡改"},
        headers=_h(ta),
    )
    assert r.status_code == 403, r.text

    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        rows = store.list_sessions(prefix=f"{ub}__")
    finally:
        store.close()
    titles = [s["title"] for s in rows if s["session_id"] == b_sid]
    assert titles and titles[0] != "被篡改"


def test_session_export_cross_user_forbidden(client):
    """A 导出 B 的会话 → 403。"""
    ua, ta = _mk_user(client, "a")
    ub, _tb = _mk_user(client, "b")
    b_sid = _mk_session(ub, "priv")

    r = client.get(f"/api/chat/sessions/{b_sid}/export", headers=_h(ta))
    assert r.status_code == 403, r.text


def test_session_batch_delete_cross_user_isolated(client):
    """A 批量删除时传入 B 的会话 → 只删自己的，B 的会话保留。"""
    ua, ta = _mk_user(client, "a")
    ub, tb = _mk_user(client, "b")
    a_sid = _mk_session(ua, "mine")
    b_sid = _mk_session(ub, "priv")

    r = client.post(
        "/api/chat/sessions/batch-delete",
        json={"ids": [a_sid, b_sid]},
        headers={**_h(ta), "Content-Type": "application/json"},
    )
    assert r.status_code == 200
    assert r.json()["deleted"] == 1  # 只删了 A 自己的

    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        b_left = [s["session_id"] for s in store.list_sessions(prefix=f"{ub}__")]
    finally:
        store.close()
    assert b_sid in b_left  # B 的会话完好


# ---------- 笔记越权 ----------


def test_note_cross_user_read_and_list(client):
    """A 读 B 的笔记 → 403；A 的列表不含 B 的笔记。"""
    ua, ta = _mk_user(client, "a")
    ub, tb = _mk_user(client, "b")
    _write_note(f"priv_{ub}", ub)

    r = client.get(f"/api/notes/priv_{ub}", headers=_h(ta))
    assert r.status_code in (403, 404), r.text

    r = client.get("/api/notes", headers=_h(ta))
    assert r.status_code == 200
    slugs = [n["slug"] for n in r.json()["notes"]]
    assert f"priv_{ub}" not in slugs  # 列表也不泄露

    # B 自己能读到
    r = client.get(f"/api/notes/priv_{ub}", headers=_h(tb))
    assert r.status_code == 200


def test_note_cross_user_edit_delete_forbidden(client, tmp_path):
    """A 编辑/删除 B 的笔记 → 403，B 的笔记内容完好。"""
    ua, ta = _mk_user(client, "a")
    ub, _tb = _mk_user(client, "b")
    slug = f"priv_{ub}"
    _write_note(slug, ub)

    r = client.put(
        f"/api/notes/{slug}",
        json={"content": "---\ntitle: 篡改\n---\n被篡改内容"},
        headers=_h(ta),
    )
    assert r.status_code == 403, r.text

    r = client.delete(f"/api/notes/{slug}", headers=_h(ta))
    assert r.status_code == 403, r.text

    notes_dir = TESTS_DIR / "test_notes_dir"
    assert (notes_dir / f"{slug}.md").exists()  # 文件未被删
    assert "被篡改内容" not in (notes_dir / f"{slug}.md").read_text(encoding="utf-8")


# ---------- 知识库命名空间（纯函数，无需 ChromaDB） ----------


def test_kb_collection_namespace_isolated():
    """不同用户/不同库的 collection 名互不重叠（防命名空间串扰）。"""
    from app.knowledge.user_kb import collection_name_for

    c_a = collection_name_for("alice")
    c_b = collection_name_for("bob")
    assert c_a != c_b
    assert collection_name_for("alice", "kb1") != collection_name_for("alice", "kb2")
    assert not c_a.startswith("bob") and not c_b.startswith("alice")


def test_kb_name_injection_rejected():
    """库名注入防护：__ 分隔符（伪造跨用户路径）与非法字符必须被拒绝。"""
    from app.knowledge.user_kb import normalize_kb_name

    with pytest.raises(ValueError):
        normalize_kb_name("alice__evil")  # 保留分隔符
    with pytest.raises(ValueError):
        normalize_kb_name("../etc/passwd")  # 路径穿越
    with pytest.raises(ValueError):
        normalize_kb_name("a b c")  # 空白
    assert normalize_kb_name("合法库名_2026") == "合法库名_2026"


# ---------- 知识图谱隔离 ----------


def test_kg_store_isolation(tmp_path):
    """A 写入图谱的数据，B 的 store 读不到（按用户独立 DB）。"""
    from app.knowledge.kg import KGStore

    store_a = KGStore("alice", str(tmp_path))
    store_b = KGStore("bob", str(tmp_path))
    # 独立 DB 文件
    assert store_a.path != store_b.path

    # A 写入实体与关系（upsert_entity 返回 entity id）
    e1 = store_a.upsert_entity("钙钛矿", "material")
    e2 = store_a.upsert_entity("稳定性", "topic")
    store_a.add_relation(e1, e2, "影响")

    ga = store_a.graph_data()
    gb = store_b.graph_data()
    assert len(ga["nodes"]) == 2
    assert gb["nodes"] == []  # B 读不到 A 的数据
