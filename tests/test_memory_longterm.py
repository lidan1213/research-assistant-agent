"""记忆链路测试：长期记忆(LongTermMemory) + SQLite 会话后端 + MemoryManager。

覆盖：落盘持久化、跨实例恢复、关键字召回（FTS5/LIKE 自适应）、会话索引、链路 record/recall。
"""
import pytest

from app.agent.memory import ConversationMemory, SQLiteStore
from app.memory.longterm import LongTermMemory
from app.memory.manager import MemoryManager


def test_longterm_persist_and_search(tmp_path):
    db = tmp_path / "lt.db"
    # chroma=False：纯 SQLite 单测（隔离、离线可跑），ChromaDB 主路径见 test_longterm_chroma_integration.py
    lt = LongTermMemory(str(db), chroma=False)
    lt.save_session("s1", title="q1", summary="ans1")
    lt.append_fact("s1", "The attention mechanism powers Transformers.", category="finding")
    lt.append_fact("s1", "BERT is a pretrained language model.", category="finding")
    lt.close()

    # 重新打开，验证落盘与召回
    lt2 = LongTermMemory(str(db), chroma=False)
    assert lt2.list_sessions()[0]["id"] == "s1"
    hits = lt2.search_facts("attention", k=5)
    assert any("attention mechanism" in h["content"] for h in hits)
    facts = lt2.get_session_facts("s1")
    assert len(facts) == 2
    lt2.close()


def test_longterm_search_by_category(tmp_path):
    db = tmp_path / "lt.db"
    lt = LongTermMemory(str(db), chroma=False)
    lt.append_fact("s1", "Alpha result 1", category="literature")
    lt.append_fact("s1", "Beta observation", category="finding")
    by_cat = lt.search_facts("result", category="literature", k=5)
    assert len(by_cat) == 1 and by_cat[0]["category"] == "literature"
    lt.close()


def test_longterm_facts_are_isolated_by_username(tmp_path):
    lt = LongTermMemory(str(tmp_path / "isolated.db"), chroma=False)
    lt.append_fact("alice__s1", "alice private quantum result", category="finding")
    lt.append_fact("bob__s1", "bob private quantum result", category="finding")
    alice = lt.search_facts("quantum", username="alice", k=10)
    bob = lt.search_facts("quantum", username="bob", k=10)
    assert {x["username"] for x in alice} == {"alice"}
    assert {x["username"] for x in bob} == {"bob"}
    assert all("bob private" not in x["content"] for x in alice)


def test_longterm_deduplicates_and_supersedes_versions(tmp_path):
    lt = LongTermMemory(str(tmp_path / "versions.db"), chroma=False)
    first = lt.append_fact("alice__s1", "模型版本为 1.0", memory_key="model-version")
    duplicate = lt.append_fact("alice__s2", "模型版本为 1.0", memory_key="model-version")
    second = lt.append_fact("alice__s3", "模型版本为 2.0", memory_key="model-version")
    assert duplicate == first and second != first
    assert lt._conn.execute("SELECT status FROM facts WHERE id=?", (first,)).fetchone()[0] == "superseded"
    assert [x["content"] for x in lt.search_facts("模型版本", username="alice", k=10)] == ["模型版本为 2.0"]


def test_longterm_conflicts_are_retained_and_expiry_is_pruned(tmp_path):
    lt = LongTermMemory(str(tmp_path / "lifecycle.db"), chroma=False)
    a = lt.append_fact("alice__s1", "材料效率达到 25.7%", category="finding")
    b = lt.append_fact("alice__s2", "材料效率达到 23.1%", category="finding")
    assert lt._conn.execute("SELECT id, status FROM facts ORDER BY id").fetchall() == [
        (a, "conflicted"), (b, "conflicted")
    ]
    lt._conn.execute("UPDATE facts SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?", (a,))
    lt._conn.commit()
    assert lt.prune_expired("alice") == 1
    assert [x["id"] for x in lt.search_facts("材料效率", username="alice", k=10)] == [b]


def test_sqlite_session_store_survives_reopen(tmp_path):
    db = tmp_path / "mem.db"
    s1 = SQLiteStore(str(db))
    import asyncio

    asyncio.run(s1.append("sess", __import__("app.llm.base", fromlist=["ChatMessage"]).ChatMessage(role=__import__("app.llm.base", fromlist=["MessageRole"]).MessageRole.USER, content="hi")))
    s1.close()

    s2 = SQLiteStore(str(db))
    msgs = asyncio.run(s2.get("sess"))
    assert len(msgs) == 1 and msgs[0].content == "hi"
    s2.close()


def test_conversation_memory_sqlite_backend(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    mem = ConversationMemory(window=10, backend=store)
    import asyncio

    asyncio.run(mem.add_user("sess", "hello"))
    asyncio.run(mem.add_assistant("sess", "hi there"))
    hist = asyncio.run(mem.history("sess"))
    assert [m.content for m in hist] == ["hello", "hi there"]


@pytest.mark.asyncio
async def test_memory_manager_record_and_recall(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    session = ConversationMemory(window=10, backend=store)
    lt = LongTermMemory(str(tmp_path / "lt.db"), chroma=False)
    mm = MemoryManager(session=session, longterm=lt)

    await mm.record_run(
        "alice__sess1",
        "什么是注意力机制？",
        "注意力机制是一种加权聚合方法。",
        facts=[{"category": "blackboard", "content": "Researcher: found attention paper", "tags": ["researcher"]}],
    )

    recalled = await mm.recall("alice__sess1", "attention", k=3)
    assert any("attention paper" in f["content"] for f in recalled["facts"])
    # 当前会话历史也被召回
    assert any("注意力机制" in m.content for m in recalled["history"])
