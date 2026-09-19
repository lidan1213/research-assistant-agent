"""手写 ReAct 循环 vs LangGraph 图编排 的行为一致性测试。

用「脚本化 Mock LLM」驱动两套实现跑同一场景（先调 calculator，再给终态），
断言两者都：
1. 调用了同一个工具（calculator）；
2. 产出相同的最终回答；
3. 事件时序一致（action 早于 answer）。

目的：证明手写循环与 LangGraph 图在等价输入下行为一致，
后续改动任一实现都能被这套确定性测试守住。
"""
import asyncio
import json

from app.agent.agent import ResearchAgent
from app.agent.memory import ConversationMemory
from app.graph.langchain_tools import build_langchain_tools
from app.graph.single_agent import GraphResearchAgent
from app.llm.base import BaseLLM, ChatMessage, LLMResponse, MessageRole, ToolCall
from app.tools.registry import registry

# 同一份「脚本」：第 1 次调用请求 calculator，第 2 次给出终态。
SCRIPT = [
    {"tool_calls": [{"name": "calculator", "arguments": json.dumps({"expression": "2+2"})}]},
    {"content": "2+2 的结果是 4。"},
]

EXPECTED_ANSWER = "2+2 的结果是 4。"
TARGET_TOOL = "calculator"


class _StubPlanner:
    async def plan(self, goal: str, max_steps: int = 5):
        return []  # 关闭规划，让脚本索引对齐

    async def plan_stream(self, goal: str, max_steps: int = 5):
        yield ("steps", [])  # 关闭规划，让脚本索引对齐


class MockReActLLM(BaseLLM):
    """实现 BaseLLM 的脚本化 mock：按调用次数吐出预设响应。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def chat(self, messages, *, tools=None, temperature=None, max_tokens=None, json_mode=False):
        idx = min(self.calls, len(self.script) - 1)
        step = self.script[idx]
        self.calls += 1
        tcs = [
            ToolCall(id=f"call_{i}", name=tc["name"], arguments=tc["arguments"])
            for i, tc in enumerate(step.get("tool_calls", []))
        ]
        return LLMResponse(
            content=step.get("content", ""),
            tool_calls=tcs,
            finish_reason="tool_calls" if tcs else "stop",
        )

    def stream(self, messages, *, tools=None, temperature=None, max_tokens=None, json_mode=False):
        async def gen():
            yield ""

        return gen()

    async def chat_stream(self, messages, *, tools=None, temperature=None, max_tokens=None):
        """流式版：复用 chat 的一次性结果，拆成多个增量块（含 tool_calls 汇总）。"""
        from app.llm.base import StreamChunk

        resp = await self.chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
        content = resp.content or ""
        # 按 4 字符切块模拟逐 token；最后一块携带完整 tool_calls 与 finish_reason
        step = max(1, len(content) // 4)
        for i in range(0, len(content), step):
            yield StreamChunk(content_delta=content[i:i + step])
        yield StreamChunk(
            content_delta="",
            tool_calls=resp.tool_calls or None,
            finish_reason=resp.finish_reason or ("tool_calls" if resp.tool_calls else "stop"),
        )


class MockLangchainLLM:
    """最小化的 LangChain chat model mock：支持 bind_tools / invoke / ainvoke。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self._tools = None

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def _next(self):
        idx = min(self.calls, len(self.script) - 1)
        step = self.script[idx]
        self.calls += 1
        return step

    def _to_ai(self, step):
        from langchain_core.messages import AIMessage

        tcs = step.get("tool_calls")
        if tcs:
            return AIMessage(
                content=step.get("content", ""),
                tool_calls=[
                    {
                        "name": tc["name"],
                        "args": json.loads(tc["arguments"]),
                        "id": f"c{i}",
                        "type": "tool",
                    }
                    for i, tc in enumerate(tcs)
                ],
            )
        return AIMessage(content=step.get("content", ""))

    def invoke(self, messages):
        return self._to_ai(self._next())

    async def ainvoke(self, state, config=None):
        return self._to_ai(self._next())


def _run_handwritten():
    agent = ResearchAgent(
        llm=MockReActLLM(SCRIPT),
        memory=ConversationMemory(window=10),
        tools=registry,
        planner=_StubPlanner(),
        max_iterations=5,
    )
    events = []

    async def _go():
        async for ev in agent.stream("sess-hw", "计算 2+2"):
            events.append(ev.to_dict())

    asyncio.run(_go())
    return events


def _run_graph():
    agent = GraphResearchAgent(
        llm=MockLangchainLLM(SCRIPT),
        tools=build_langchain_tools(registry),
        max_iterations=5,
    )
    events = []

    async def _go():
        async for ev in agent.stream("计算 2+2"):
            events.append(ev)

    asyncio.run(_go())
    return events


def _extract_tools_and_answer(events):
    tools = []
    answer = None
    action_idx = None
    for i, ev in enumerate(events):
        if ev.get("type") == "action":
            # 手写：data["tool"]；LangGraph：data["tool_calls"][*]["name"]
            if ev.get("tool"):
                tools.append(ev["tool"])
            for tc in ev.get("tool_calls", []) or []:
                tools.append(tc.get("name"))
            action_idx = i
        if ev.get("type") == "answer" and answer is None:
            answer = ev.get("content", "")
            if action_idx is not None:
                assert i > action_idx, "answer 必须出现在 action 之后"
    return tools, answer


def test_handwritten_react_behavior():
    events = _run_handwritten()
    tools, answer = _extract_tools_and_answer(events)
    assert TARGET_TOOL in tools, f"手写 Agent 未调用 {TARGET_TOOL}: {tools}"
    assert answer is not None and EXPECTED_ANSWER in answer


def test_graph_react_behavior():
    events = _run_graph()
    tools, answer = _extract_tools_and_answer(events)
    assert TARGET_TOOL in tools, f"LangGraph Agent 未调用 {TARGET_TOOL}: {tools}"
    assert answer is not None and EXPECTED_ANSWER in answer


def test_react_implementations_consistent():
    """两套实现在等价输入下行为一致。"""
    hw_tools, hw_answer = _extract_tools_and_answer(_run_handwritten())
    g_tools, g_answer = _extract_tools_and_answer(_run_graph())

    assert hw_tools == g_tools, f"工具调用不一致: 手写={hw_tools} 图={g_tools}"
    assert hw_answer == g_answer, f"终态不一致: 手写={hw_answer!r} 图={g_answer!r}"
