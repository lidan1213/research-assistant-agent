"""PlanExecuteAgent 工具委托 ToolExecutor 测试。"""
from __future__ import annotations

import asyncio

from app.agent.plan_execute import PlanExecuteAgent
from app.llm.base import LLMResponse
from app.tools.base import ToolResult, ToolRegistry


class FakeTool:
    name = "stub_tool"
    description = "stub"

    def to_function_schema(self):
        return {"name": self.name, "description": self.description, "parameters": {}}

    async def execute(self, arguments_json: str) -> ToolResult:
        return ToolResult(success=True, output="工具结果内容")


class FakeRegistry:
    """模拟 registry（无 get 方法 → 走 _execute_via_call 兼容路径）。"""

    def __init__(self):
        self.calls = 0

    def subset(self, exclude=None, include=None):
        return self

    def schemas(self):
        return [{"name": "stub_tool", "description": "stub", "parameters": {}}]

    async def call(self, name, args):
        self.calls += 1
        return ToolResult(success=True, output="工具结果内容")


class FakeLLM:
    """Planner + 执行循环共用：返回工具调用 → 再返回最终结论。"""

    def __init__(self):
        self.chat_count = 0

    async def chat(self, messages, **kwargs):
        self.chat_count += 1
        if self.chat_count <= 2:  # planner 调用 1 次 + 执行循环第 1 次
            from app.llm.base import ToolCall

            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="stub_tool", arguments="{}")],
            )
        return LLMResponse(content="最终结论")


def test_plan_agent_uses_executor():
    """PlanExecuteAgent 工具执行委托 ToolExecutor（registry 兼容路径）。"""
    reg = FakeRegistry()
    llm = FakeLLM()
    agent = PlanExecuteAgent(llm=llm, tools=reg, max_step_iters=1, max_tool_calls=5)

    events = []
    async def run():
        async for ev in agent.stream("测试目标"):
            events.append(ev.type)
        return events

    asyncio.run(run())
    assert "observation" in events
    assert reg.calls >= 1  # 工具确实被调用
    assert "answer" in events


def test_plan_agent_keeps_loop_guard_across_steps():
    """第二步确认重复结果后，第三步不得再次真正调用相同工具。"""
    from app.llm.base import ToolCall

    class StepLLM:
        def __init__(self):
            self.tool_round = 0

        async def chat(self, messages, **kwargs):
            from app.llm.base import MessageRole

            if messages and messages[-1].role == MessageRole.USER:
                return LLMResponse(content="基于已有信息收尾")
            self.tool_round += 1
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(
                    id=f"t{self.tool_round}",
                    name="stub_tool",
                    arguments=f'{{"query":"变体{self.tool_round}"}}',
                )],
            )

    reg = FakeRegistry()
    agent = PlanExecuteAgent(
        llm=StepLLM(), tools=reg, max_step_iters=1, max_tool_calls=10,
    )

    async def run_step(index: int):
        return [item async for item in agent._execute_step(
            goal="同一个总任务",
            plan_text="步骤1；步骤2；步骤3",
            steps=["步骤1", "步骤2", "步骤3"],
            idx=index,
            step=f"步骤{index}",
            prev_results=[],
        )]

    asyncio.run(run_step(1))
    second = asyncio.run(run_step(2))
    third = asyncio.run(run_step(3))

    assert reg.calls == 2  # 第三阶段被任务级黑板拦截，没有落到真实工具
    assert any(payload.get("blocked") for kind, payload in second if kind == "observation")
    assert any(payload.get("blocked") for kind, payload in third if kind == "observation")


def test_blocked_knowledge_search_executes_arxiv_recovery():
    """知识库路径被封禁后，应真正执行 arXiv 替代路径并回写证据。"""
    from app.llm.base import MessageRole, ToolCall

    class RecoveryRegistry:
        def __init__(self):
            self.calls = []

        def schemas(self):
            return [
                {"name": "knowledge_search", "description": "kb", "parameters": {}},
                {"name": "arxiv_search", "description": "papers", "parameters": {}},
            ]

        async def call(self, name, args):
            self.calls.append(name)
            return ToolResult(success=True, output="arXiv 找到了 GraphRAG 论文证据")

    class RecoveryLLM:
        async def chat(self, messages, **kwargs):
            if messages[-1].role == MessageRole.USER:
                return LLMResponse(content="根据论文证据完成回答")
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(
                    id="kb_again", name="knowledge_search", arguments='{"query":"GraphRAG"}',
                )],
            )

    reg = RecoveryRegistry()
    agent = PlanExecuteAgent(
        llm=RecoveryLLM(), tools=reg, max_step_iters=1, max_tool_calls=10,
    )
    agent._task_loop_guard.record("knowledge_search", "知识库没有结果")
    agent._task_loop_guard.record("knowledge_search", "知识库没有结果")

    async def run():
        return [item async for item in agent._execute_step(
            goal="解释 GraphRAG",
            plan_text="查找资料并回答",
            steps=["查找 GraphRAG 资料"],
            idx=1,
            step="查找 GraphRAG 资料",
            prev_results=[],
        )]

    events = asyncio.run(run())
    assert reg.calls == ["arxiv_search"]
    assert any(kind == "recovery" and data["tool"] == "arxiv_search" for kind, data in events)
    assert any(
        kind == "observation" and data.get("recovery") and "论文证据" in data["output"]
        for kind, data in events
    )
