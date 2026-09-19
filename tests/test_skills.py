"""Local Skill discovery, matching, state, API, and prompt injection tests."""
from __future__ import annotations

from app.llm.base import ChatMessage, MessageRole
from app.skills.context import active_skill_prompt
from app.skills.registry import SkillRegistry


SKILL = """---
name: test-review
description: 测试用综述流程
triggers:
  - 文献综述
  - literature review
tools: [knowledge_search, arxiv_search]
preferred_mode: plan
---

先检索证据，再进行对比，最后输出带引用的结论。
"""


def _registry(tmp_path):
    root = tmp_path / "skills"
    folder = root / "test-review"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
    return SkillRegistry(root, tmp_path / "state.json")


def test_skill_discovery_matching_and_manual_selection(tmp_path):
    registry = _registry(tmp_path)
    skill = registry.get("test-review")
    assert skill is not None
    assert skill.enabled and not skill.errors
    assert skill.tools == ("knowledge_search", "arxiv_search")
    assert registry.match("请完成一篇 RAG 文献综述") == skill
    assert registry.match("普通问答") is None
    assert registry.match("普通问答", "test-review") == skill


def test_skill_enable_state_persists(tmp_path):
    registry = _registry(tmp_path)
    disabled = registry.set_enabled("test-review", False)
    assert not disabled.enabled
    assert registry.match("文献综述") is None
    restored = SkillRegistry(registry.root, registry.state_path)
    assert not restored.get("test-review").enabled


async def test_context_builder_injects_request_skill():
    from app.agent.context import ContextBuilder

    class Memory:
        async def history(self, _session_id):
            return [ChatMessage(role=MessageRole.USER, content="问题")]

    token = active_skill_prompt.set("## 当前 Skill\n必须先构建证据表。")
    try:
        messages, _ = await ContextBuilder(Memory(), system_prompt="基础规则").build("s1")
    finally:
        active_skill_prompt.reset(token)
    assert "基础规则" in messages[0].content
    assert "必须先构建证据表" in messages[0].content


def test_skills_api_list_and_match(client, monkeypatch, tmp_path):
    registry = _registry(tmp_path)
    monkeypatch.setattr("app.api.routes.skills.get_skill_registry", lambda: registry)
    token = client.post(
        "/api/auth/login", json={"username": "student1", "password": "123456"}
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    listed = client.get("/api/skills", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["skills"][0]["name"] == "test-review"
    matched = client.post(
        "/api/skills/match",
        headers=headers,
        json={"query": "请写文献综述", "skill": "auto"},
    )
    assert matched.status_code == 200
    assert matched.json()["matched"]["name"] == "test-review"


def test_skill_admin_controls(client, monkeypatch, tmp_path):
    registry = _registry(tmp_path)
    monkeypatch.setattr("app.api.routes.skills.get_skill_registry", lambda: registry)
    student = client.post(
        "/api/auth/login", json={"username": "student1", "password": "123456"}
    ).json()["token"]
    denied = client.put(
        "/api/skills/test-review/enabled",
        headers={"Authorization": f"Bearer {student}"},
        json={"enabled": False},
    )
    assert denied.status_code == 403
    admin = client.post(
        "/api/auth/login", json={"username": "admin", "password": "123456"}
    ).json()["token"]
    changed = client.put(
        "/api/skills/test-review/enabled",
        headers={"Authorization": f"Bearer {admin}"},
        json={"enabled": False},
    )
    assert changed.status_code == 200
    assert changed.json()["enabled"] is False
