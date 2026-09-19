"""基于 LangGraph 的科研 Agent（单 Agent 图编排，等价于 ReAct）。

相比 app/agent/agent.py 的手动循环，这里用 StateGraph 显式声明：
reason（推理/决策）-> tools（工具执行）的有环图，由条件边驱动，
更易于扩展节点（检索、反思、审核等）。

图结构：
    START -> reason --有 tool_calls--> tools -> reason
                      --无 tool_calls--> END
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, MessagesState, START, StateGraph

from app.graph.langchain_tools import build_langchain_tools, dispatch_tool
from app.graph.llm_bridge import get_langchain_llm
from app.tools.registry import registry as default_registry

DEFAULT_SYSTEM = (
    "你是一位严谨的科研助手，基于提供的工具完成检索、计算与引用整理，"
    "给出有依据、结构化的回答。"
)


class GraphResearchAgent:
    def __init__(self, llm=None, tools=None, max_iterations: int = 8) -> None:
        self.max_iterations = max_iterations
        self.tools = tools if tools is not None else build_langchain_tools(default_registry)
        self.llm = (llm or get_langchain_llm()).bind_tools(self.tools)
        self._graph = self._build()

    def _reason(self, state: MessagesState):
        resp = self.llm.invoke(state["messages"])
        return {"messages": [resp]}

    async def _tools(self, state: MessagesState):
        last = state["messages"][-1]
        outs = [await dispatch_tool(tc, self.tools) for tc in getattr(last, "tool_calls", [])]
        return {"messages": outs}

    @staticmethod
    def _should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def _build(self):
        g = StateGraph(MessagesState)
        g.add_node("reason", self._reason)
        g.add_node("tools", self._tools)
        g.add_edge(START, "reason")
        g.add_conditional_edges(
            "reason", self._should_continue, {"tools": "tools", END: END}
        )
        g.add_edge("tools", "reason")
        return g.compile()

    async def run(self, query: str, system_prompt: str | None = None) -> str:
        msgs = [SystemMessage(system_prompt or DEFAULT_SYSTEM), HumanMessage(query)]
        result = await self._graph.ainvoke(
            {"messages": msgs}, {"recursion_limit": self.max_iterations + 2}
        )
        return result["messages"][-1].content

    async def stream(self, query: str, system_prompt: str | None = None):
        """异步流式：逐节点产出事件（reason / tools / answer）。"""
        msgs = [SystemMessage(system_prompt or DEFAULT_SYSTEM), HumanMessage(query)]
        async for chunk in self._graph.astream({"messages": msgs}):
            for node, update in chunk.items():
                if node == "reason":
                    msg = update["messages"][-1]
                    if getattr(msg, "tool_calls", None):
                        # tool_calls 在 LangChain 中为 dict 列表，统一转成可序列化 dict
                        tcs = [dict(tc) if not hasattr(tc, "model_dump") else tc.model_dump() for tc in msg.tool_calls]
                        yield {"type": "action", "tool_calls": tcs}
                    else:
                        yield {"type": "answer", "content": msg.content}
                elif node == "tools":
                    for m in update["messages"]:
                        yield {"type": "observation", "tool": getattr(m, "name", ""), "output": m.content}
