"""多 Agent handoff 直传测试：验证主管委派专家时，把上游专家在黑板上的成果
作为已交付中间产物（inbox）直接注入下游专家的 prompt，而非仅间接共享。

不触网，使用可脚本化的 LangChain mock（`_FakeLC`）。
"""
from app.graph.multi_agent import MultiAgentSupervisor
from langchain_core.messages import AIMessage


class _FakeLC:
    """最小 LangChain chat model mock：按调用顺序吐出脚本步骤，记录每次 ainvoke 的 system。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.sys_log = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, state, config=None):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if state and getattr(state[0], "content", None):
            self.sys_log.append(state[0].content)
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


# 流程：supervisor 委派 Researcher -> Researcher 产出[FINDING] -> supervisor 委派 Writer
#       -> Writer 产出[WRITTEN] -> supervisor 给出最终答案
SCRIPT = [
    {"tool_calls": [{"name": "transfer_to_Researcher", "args": {}, "id": "t1"}]},
    {"content": "[FINDING] Researcher 检索到 RAG 原理资料"},
    {"tool_calls": [{"name": "transfer_to_Writer", "args": {}, "id": "t2"}]},
    {"content": "[WRITTEN] Writer 完成结构化综述"},
    {"content": "最终答案：依据专家写入黑板的成果，给出完整综述。"},
]


async def test_handoff_delegate_incoming_carries_upstream():
    ma = MultiAgentSupervisor(llm=_FakeLC(SCRIPT), tools=[])
    events = [e async for e in ma.stream("请综述 RAG 原理")]
    delegates = [e for e in events if e["type"] == "delegate"]
    assert [d["to"] for d in delegates] == ["Researcher", "Writer"], delegates

    # 委派给 Writer 时，应携带 Researcher 在黑板上的成果作为直传
    writer_del = delegates[1]
    assert "researcher" in writer_del["incoming"], writer_del
    assert writer_del["payload"] is None


async def test_handoff_injects_upstream_into_worker_prompt():
    ma = MultiAgentSupervisor(llm=_FakeLC(SCRIPT), tools=[])
    _ = [e async for e in ma.stream("请综述 RAG 原理")]
    lc = ma._llm
    # 调用顺序：sup, researcher, sup, writer, sup -> Writer 是第 4 次(index 3)
    writer_sys = lc.sys_log[3]
    assert "Writer 专家" in writer_sys, "应为 Writer 的 system prompt"
    assert "[FINDING]" in writer_sys, "Writer 未收到上游 Researcher 的直传产物"


async def test_handoff_blackboard_final_has_both():
    ma = MultiAgentSupervisor(llm=_FakeLC(SCRIPT), tools=[])
    events = [e async for e in ma.stream("请综述 RAG 原理")]
    bb_events = [e for e in events if e["type"] == "blackboard"]
    final_bb = bb_events[-1]["full"]
    assert "researcher" in final_bb and "writer" in final_bb, final_bb


async def test_handoff_explicit_payload_passed_through():
    """主管在 handoff 时显式携带 payload，应出现在 delegate 事件中。"""
    script = [
        {"tool_calls": [{"name": "transfer_to_Writer", "args": {"payload": "请重点写引用格式"}, "id": "t1"}]},
        {"content": "[W] Writer 完成"},
        {"content": "最终答案"},
    ]
    ma = MultiAgentSupervisor(llm=_FakeLC(script), tools=[])
    events = [e async for e in ma.stream("写综述")]
    delegates = [e for e in events if e["type"] == "delegate"]
    assert delegates[0]["payload"] == "请重点写引用格式", delegates
