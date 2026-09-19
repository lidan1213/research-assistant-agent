"""多 Agent 共享黑板测试（不触网，使用可脚本化的 LangChain mock）。

验证：
1. 共享黑板存在于图状态；
2. stream 能产出 delegate（委派）、blackboard（黑板更新）、answer（最终答案）三类事件；
3. 专家结论被写入黑板，且随事件对外暴露。
"""
from app.graph.multi_agent import MultiAgentSupervisor
from langchain_core.messages import AIMessage


class _FakeLC:
    """最小 LangChain chat model mock：按调用顺序吐出脚本步骤，支持 bind_tools/ainvoke。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, state, config=None):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        tcs = step.get("tool_calls")
        if tcs:
            return AIMessage(
                content=step.get("content", ""),
                tool_calls=[
                    {"name": tc["name"], "args": tc.get("args", {}), "id": tc.get("id", "c"), "type": "tool"}
                    for tc in tcs
                ],
            )
        return AIMessage(content=step.get("content", ""))


# 流程：supervisor 委派 Writer -> Writer 给出结论 -> supervisor 给出最终答案
SCRIPT = [
    {"tool_calls": [{"name": "transfer_to_Writer", "args": {}, "id": "t1"}]},
    {"content": "Writer 结论：已汇总各方成果，形成结构化综述要点。"},
    {"content": "最终答案：依据专家写入共享黑板的成果，给出完整综述。"},
]


def test_multiagent_builds_with_blackboard():
    ma = MultiAgentSupervisor(llm=_FakeLC(SCRIPT), tools=[])
    assert ma._graph is not None
    assert set(ma.members) == {"Researcher", "Analyst", "Writer"}


async def test_multiagent_stream_delegate_blackboard_answer():
    ma = MultiAgentSupervisor(llm=_FakeLC(SCRIPT), tools=[])
    events = [e async for e in ma.stream("测试问题")]
    types = [e["type"] for e in events]

    assert "delegate" in types, f"缺少委派事件: {types}"
    assert "blackboard" in types, f"缺少黑板事件: {types}"
    assert "answer" in types, f"缺少答案事件: {types}"

    delegate = next(e for e in events if e["type"] == "delegate")
    assert delegate["to"] == "Writer", delegate

    bb = next(e for e in events if e["type"] == "blackboard")
    assert "writer" in bb["full"], bb["full"]
    assert "Writer 结论" in bb["full"]["writer"], bb["full"]


async def test_multiagent_run_returns_final_answer():
    ma = MultiAgentSupervisor(llm=_FakeLC(SCRIPT), tools=[])
    answer = await ma.run("测试问题")
    assert "最终答案" in answer, answer
