"""Plan-Execute 在缺少用户资料时暂停，并从持久化步骤恢复。"""
from __future__ import annotations

import asyncio

from app.agent.pending_tasks import PendingTaskStore
from app.agent.plan_execute import PlanExecuteAgent
from app.llm.base import LLMResponse, MessageRole, ToolCall
from app.tools.base import ToolResult


class OneStepPlanner:
    async def plan(self, goal):
        return ["读取并总结这篇 PDF"]


class MissingDocumentRegistry:
    def __init__(self):
        self.calls = 0

    def schemas(self):
        return [{"name": "knowledge_search", "description": "kb", "parameters": {}}]

    async def call(self, name, args):
        self.calls += 1
        return ToolResult(success=True, output="没有找到用户指定的 PDF 文档")


class PauseResumeLLM:
    def __init__(self):
        self.tool_round = 0

    async def chat(self, messages, **kwargs):
        system = messages[0].content or ""
        if kwargs.get("tools") is None:
            return LLMResponse(content="已根据用户补充的论文内容完成总结")
        if "用户补充信息" in system:
            return LLMResponse(content="已根据用户补充的论文内容完成总结")
        if messages[-1].role == MessageRole.USER:
            return LLMResponse(content="资料不足，等待用户补充")
        self.tool_round += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(
                id=f"t{self.tool_round}",
                name="knowledge_search",
                arguments=f'{{"query":"PDF 变体 {self.tool_round}"}}',
            )],
        )


def test_missing_document_pauses_and_resumes_from_same_step(tmp_path, monkeypatch):
    import app.agent.pending_tasks as pending_mod

    monkeypatch.setattr(
        pending_mod,
        "_store",
        PendingTaskStore(str(tmp_path / "pending.db"), ttl_seconds=3600),
    )
    agent = PlanExecuteAgent(
        llm=PauseResumeLLM(),
        planner=OneStepPlanner(),
        tools=MissingDocumentRegistry(),
        max_step_iters=3,
        max_tool_calls=10,
    )

    async def first_run():
        return [event async for event in agent.stream("请总结我上传的这篇 PDF")]

    first = asyncio.run(first_run())
    request = next(event for event in first if event.type == "needs_user_input")
    assert request.data["reason"] == "required_document_missing"
    assert request.data["resume_point"] == "step_1"
    assert request.data["resume_token"]
    assert not any(event.type == "answer" for event in first)

    async def resume():
        return [event async for event in agent.resume(
            request.data["resume_token"],
            "论文标题为 Example Paper，补充正文：核心贡献是……",
        )]

    resumed = asyncio.run(resume())
    assert any(event.type == "resumed" for event in resumed)
    answer = next(event for event in resumed if event.type == "answer")
    assert "完成总结" in answer.data["content"]

    # token 是一次性的，避免重复恢复同一暂停任务。
    reused = asyncio.run(resume())
    assert reused[0].type == "error"
    assert reused[0].data["reason"] == "invalid_resume_token"


def test_pending_task_survives_store_recreation(tmp_path):
    """SQLite 暂停点可由新进程/新 Store 实例读取。"""
    path = str(tmp_path / "persistent_pending.db")
    token = PendingTaskStore(path).save({"goal": "测试", "step_index": 1})
    restored = PendingTaskStore(path).pop(token)
    assert restored == {"goal": "测试", "step_index": 1}


def test_user_input_request_distinguishes_document_and_web_permission():
    from app.agent.user_input import build_user_input_request

    doc = build_user_input_request(
        goal="总结我上传的论文", step="读取 PDF", failed_tool="knowledge_search",
        available_tools={"knowledge_search"},
    )
    assert doc["required_input"] == "document_or_reference"

    web = build_user_input_request(
        goal="查询冷门概念", step="搜索资料", failed_tool="arxiv_search",
        available_tools={"knowledge_search", "arxiv_search"},
    )
    assert web["required_input"] == "enable_web_or_reference"
