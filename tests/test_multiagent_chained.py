"""多 Agent 链式直传测试：验证 A→B→C 的成品一路沿链路带下去（主链路 _prev_member_）。

不触网，使用可脚本化的 LangChain mock（_FakeLC）。
"""
from app.graph.multi_agent import MultiAgentSupervisor
from langchain_core.messages import AIMessage


class _FakeLC:
    """最小 LangChain chat model mock：按脚本顺序吐步骤，记录每次 ainvoke 的 system。"""

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


# A(Researcher) -> B(Analyst) -> C(Writer) 的链式委派
CHAIN_SCRIPT = [
    {"tool_calls": [{"name": "transfer_to_Researcher", "args": {}, "id": "t1"}]},
    {"content": "[FINDING] Researcher 检索到 RAG 原理资料"},
    {"tool_calls": [{"name": "transfer_to_Analyst", "args": {}, "id": "t2"}]},
    {"content": "[ANALYSIS] Analyst 给出 RAG 的复杂度分析"},
    {"tool_calls": [{"name": "transfer_to_Writer", "args": {}, "id": "t3"}]},
    {"content": "[WRITTEN] Writer 完成结构化综述"},
    {"content": "最终答案：依据专家链路上交付的成果，给出完整综述。"},
]


async def test_handoff_chain_order():
    ma = MultiAgentSupervisor(llm=_FakeLC(CHAIN_SCRIPT), tools=[])
    events = [e async for e in ma.stream("请综述 RAG 原理")]
    delegates = [e for e in events if e["type"] == "delegate"]
    assert [d["to"] for d in delegates] == ["Researcher", "Analyst", "Writer"], delegates

    # 每一跳的「上一专家」应正确：首跳无 prev，后续紧跟前置专家
    assert delegates[0].get("prev") is None
    assert delegates[1]["prev"] == "researcher"
    assert delegates[2]["prev"] == "analyst"


async def test_handoff_chain_writer_receives_full_upstream():
    ma = MultiAgentSupervisor(llm=_FakeLC(CHAIN_SCRIPT), tools=[])
    events = [e async for e in ma.stream("请综述 RAG 原理")]
    writer_del = [e for e in events if e["type"] == "delegate" and e["to"] == "Writer"][0]
    # Writer 应同时收到 Researcher 与 Analyst 的直传
    assert "researcher" in writer_del["incoming"], writer_del
    assert "analyst" in writer_del["incoming"], writer_del


async def test_handoff_chain_prev_injected_into_worker_prompt():
    ma = MultiAgentSupervisor(llm=_FakeLC(CHAIN_SCRIPT), tools=[])
    _ = [e async for e in ma.stream("请综述 RAG 原理")]
    lc = ma._llm
    # 调用顺序：sup, researcher, sup, analyst, sup, writer, sup
    # Writer 的 system 在第 6 次（index 5）
    writer_sys = lc.sys_log[5]
    assert "Writer 专家" in writer_sys, "应为 Writer 的 system prompt"
    assert "[ANALYSIS]" in writer_sys, "Writer 未收到上一专家(Analyst)的链式直传产物"
    assert "analyst" in writer_sys.lower(), "Worker prompt 应点名上一专家"


async def test_handoff_chain_event_sequence():
    ma = MultiAgentSupervisor(llm=_FakeLC(CHAIN_SCRIPT), tools=[])
    events = [e async for e in ma.stream("请综述 RAG 原理")]
    chain_events = [e for e in events if e["type"] == "chain"]
    # 三次委派 -> 三个 chain 步骤，且逐步累积
    assert len(chain_events) == 3, chain_events
    steps = chain_events[-1]["steps"]
    # 每步含 from/to/payload；存在上游产物时还携带 product（用于可视化成品链路）
    assert steps[0]["from"] is None and steps[0]["to"] == "researcher"
    assert steps[1]["from"] == "researcher" and steps[1]["to"] == "analyst"
    assert "product" in steps[1] and "[FINDING]" in steps[1]["product"]
    assert steps[2]["from"] == "analyst" and steps[2]["to"] == "writer"
    assert "product" in steps[2] and "[ANALYSIS]" in steps[2]["product"]
