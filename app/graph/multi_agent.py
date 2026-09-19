"""多 Agent 协同：主管（Supervisor）+ 专家（Researcher / Analyst / Writer）。

协作模式：主管读取对话，决定把任务交给哪个专家（通过 handoff 工具），
各专家各自带工具完成子任务后把结果回传主管；主管在信息充分时停止委派并产出最终答案。

共享黑板（Blackboard）：
    各专家在完成子任务后，会把关键结论写入一个跨节点共享的 `blackboard`（图状态字段），
    主管在委派决策时会读取黑板上的进展，避免重复委派、并让 Writer 能直接汇总已有成果，
    而不必把所有中间产物都塞进对话消息历史。黑板同时随流式事件对外暴露，便于前端可视化。

图结构：
    START -> supervisor
    supervisor --transfer_to_X--> X 专家 --> supervisor
    supervisor --无 tool_calls--> END

状态：
    TeamState(MessagesState): messages + blackboard（带合并 reducer）
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from app.graph.langchain_tools import build_langchain_tools, dispatch_tool, make_handoff
from app.graph.llm_bridge import get_langchain_llm
from app.core.logging import get_logger
from app.tools.registry import registry as default_registry

logger = get_logger("graph.multi_agent")

SUPERVISOR_PROMPT = """你是科研团队的主管，负责把用户问题分配给最合适的专家。
可用专家：
- Researcher：负责检索（网络/arXiv/PDF）与运行代码验证；
- Analyst：负责数值计算与文献引用（APA/BibTeX）整理；
- Writer：负责整合各方结果，产出结构化、带引用的最终回答。

规则：
1. 根据当前进展选择下一步要委派的专家，或调用对应 handoff 工具移交；
2. 当信息已经足够给出完整回答时，直接给出最终总结（不调用任何工具即表示结束）；
3. 各专家会把关键结论写入「共享黑板」，你做委派决策时应参考黑板上的进展，
   避免重复委派同一专家去做已完成的工作。
"""

RESEARCHER_PROMPT = """你是 Researcher 专家。利用检索与代码执行工具收集资料、验证想法，
只产出事实与中间结论，并把关键结论写入共享黑板（结构化要点），不要写最终综述。"""

ANALYST_PROMPT = """你是 Analyst 专家。利用计算与引用工具做定量分析和文献规范化，
给出可复用的数值结果与规范引用，并把结果写入共享黑板。"""

WRITER_PROMPT = """你是 Writer 专家。整合主管与各位专家在共享黑板上的成果，产出最终答案：
结构化、条理清晰、关键结论带引用来源。不要重复检索，只做综合撰写。"""

MEMBERS = ["Researcher", "Analyst", "Writer"]
_ROLE_TOOLS = {
    "Researcher": {"web_search", "arxiv_search", "pdf_reader", "code_executor", "knowledge_search"},
    "Analyst": {"calculator", "citation", "code_executor"},
    "Writer": set(),
}
_ROLE_PROMPTS = {
    "Researcher": RESEARCHER_PROMPT,
    "Analyst": ANALYST_PROMPT,
    "Writer": WRITER_PROMPT,
}


def _merge_blackboard(a: dict, b: dict) -> dict:
    """黑板 reducer：节点返回的局部更新与全局黑板合并（后者覆盖前者）。"""
    return {**a, **b}


def _merge_inbox(a: dict, b: dict) -> dict:
    """inbox reducer：主管委派专家时写入「目标专家 -> 上游已交付的中间产物」。

    后者覆盖前者，即每个专家记住「最近一次被委派时」上游交付给它的内容。
    """
    return {**a, **b}


def _extend_chain(a: list, b: list) -> list:
    """chain reducer：把每次委派产生的「链式传递步骤」追加到有序链上。"""
    return list(a) + list(b)


def _first_text(content) -> str:
    """LangChain 的 message.content 可能是 str 或 content parts 列表，统一成字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(c if isinstance(c, str) else str(c) for c in content)
    return str(content)


class TeamState(MessagesState):
    """自定义图状态：消息历史 + 共享黑板 + 专家间直传的 inbox（带合并 reducer）。

    - blackboard：各专家把关键结论写入，主管做委派决策时参考；
    - inbox：主管委派某专家时，把「其他专家在黑板上的成果」作为已交付中间产物
      直传给该专家（比黑板更显式、可定向），worker 执行时会注入 prompt。
    """

    blackboard: Annotated[dict, _merge_blackboard] = {}
    inbox: Annotated[dict, _merge_inbox] = {}
    chain: Annotated[list, _extend_chain] = []  # 有序链路：A->B->C 的逐步直传记录


class MultiAgentSupervisor:
    def __init__(self, llm=None, tools=None, max_iterations: int = 12) -> None:
        self.max_iterations = max_iterations
        self.members = MEMBERS
        self._llm = llm or get_langchain_llm()
        all_tools = tools if tools is not None else build_langchain_tools(default_registry)

        self.workers = {
            name: self._make_worker(name, _ROLE_PROMPTS[name], [t for t in all_tools if t.name in _ROLE_TOOLS[name]])
            for name in MEMBERS
        }

        handoffs = [make_handoff(m) for m in MEMBERS]
        self.supervisor_llm = self._llm.bind_tools(handoffs)
        self._graph = self._build()

    def _base_llm(self):
        # 与主管共用同一底层 LLM，保证可注入 mock 进行确定性测试
        return self._llm

    async def _traced_llm_call(self, span: str, llm, messages):
        """包装 LangChain LLM 调用并记录 Trace（component=llm，name=span:model）。

        LangGraph 侧使用 langchain ChatModel（不经自研 BaseLLM），这里手动补记
        trace，使 supervisor/worker 的调用纳入统一 trace_id 链路。
        """
        import time as _time

        t0 = _time.monotonic()
        try:
            resp = await llm.ainvoke(messages)
            from app.llm.trace import get_trace_ledger

            get_trace_ledger().record(
                "llm",
                f"{span}:{getattr(llm, 'model_name', getattr(llm, 'model', 'langchain'))}",
                success=True,
                duration_ms=int((_time.monotonic() - t0) * 1000),
                detail=f"graph_span={span} messages={len(messages)}",
            )
            return resp
        except Exception as e:  # noqa: BLE001
            from app.llm.trace import get_trace_ledger

            get_trace_ledger().record(
                "llm",
                f"{span}:{getattr(llm, 'model_name', getattr(llm, 'model', 'langchain'))}",
                success=False,
                duration_ms=int((_time.monotonic() - t0) * 1000),
                detail=f"graph_span={span} error={str(e)[:150]}",
            )
            raise

    # ---------- 节点 ----------
    async def _supervisor(self, state: TeamState):
        from app.skills.context import append_active_skill

        bb = state.get("blackboard") or {}
        chain = state.get("chain") or []
        # 上一个实际产出的专家：优先取链式上一步的目标；否则取黑板上最近写入的键
        prev_member = (chain[-1]["to"] if chain else None) or (
            next(reversed(list(bb.keys())), None) if bb else None
        )
        extra = ""
        if bb:
            lines = "\n".join(f"- {k}: {v}" for k, v in bb.items())
            extra = (
                f"\n\n【共享黑板】各专家已写入的进展：\n{lines}\n"
                f"请基于黑板进展决定下一步委派（多个专家可分别补充不同方面）。"
            )
        resp = await self._traced_llm_call(
            "supervisor",
            self.supervisor_llm,
            [SystemMessage(append_active_skill(SUPERVISOR_PROMPT + extra))] + state["messages"],
        )

        # 委派时，把「其他专家在黑板上的成果」作为已交付中间产物直传给目标专家，
        # 让下游专家无需从黑板间接读取，而是直接拿到上游产物（inbox 注入）。
        # 其中与「上一专家」的直传是主链路（prev），其余上游为补充上下文。
        inbox_update: dict = {}
        chain_update: list = []
        tcs = getattr(resp, "tool_calls", None)
        if tcs:
            tgt = tcs[0]["name"].replace("transfer_to_", "")
            if tgt in self.members:
                # 黑板 key 统一为小写（worker 写入时也用小写），故比较与写入都用 lower
                tgt_key = tgt.lower()
                incoming = {k: v for k, v in bb.items() if k != tgt_key}
                # 主管也可以在 handoff 时显式携带一段指令/数据（payload）
                payload = (tcs[0].get("args") or {}).get("payload")
                if payload:
                    incoming = {**incoming, "_payload_": payload}
                # 主链路直传：上一专家的核心结论作为直接交付物
                if prev_member and prev_member in bb and prev_member != tgt_key:
                    incoming["_prev_member_"] = prev_member
                    incoming["_prev_product_"] = bb[prev_member]
                if incoming:
                    inbox_update = {tgt_key: incoming}
                # 链式步骤记录（用于可视化 A -> B -> C 的成品传递）
                step = {"from": prev_member, "to": tgt_key, "payload": payload}
                if prev_member and prev_member in bb and prev_member != tgt_key:
                    step["product"] = bb[prev_member]
                chain_update = [step]

        result: dict = {"messages": [resp]}
        if inbox_update:
            result["inbox"] = inbox_update
        if chain_update:
            result["chain"] = chain_update
        return result

    def _make_worker(self, name: str, prompt: str, tools: list):
        wllm = self._base_llm().bind_tools(tools) if tools else self._base_llm()
        key = name.lower()

        async def worker(state: TeamState):
            sys_text = prompt
            # 主管直传过来的上游中间产物：显式注入 prompt，避免重复检索/计算
            incoming = (state.get("inbox") or {}).get(key) or {}
            if incoming:
                parts = []
                prev = incoming.get("_prev_member_")
                prev_product = incoming.get("_prev_product_")
                if prev and prev_product is not None:
                    parts.append(
                        f"【上一专家 {prev} 直接交付给你的核心结论（主链路直传）】\n{prev_product}"
                    )
                for k, v in incoming.items():
                    if k in ("_payload_", "_prev_member_", "_prev_product_"):
                        continue
                    label = "主管指令" if k == "_payload_" else f"{k} 专家已交付"
                    parts.append(f"【{label}】\n{v}")
                ctx = "\n\n".join(parts)
                sys_text = (
                    prompt
                    + "\n\n以下是由主管/其他专家直接交付给你的中间产物（成品一路沿链路带下来），"
                    + "请直接使用、不要重复检索或计算：\n"
                    + ctx
                )
            from app.skills.context import append_active_skill

            sys = SystemMessage(append_active_skill(sys_text))
            history = state["messages"]
            if not tools:
                final = await self._traced_llm_call(name, wllm, [sys] + history)
                return {"messages": [final], "blackboard": {key: _first_text(final.content)}}
            # 单轮工具调用（保持有界，避免无限循环），把中间步骤一并写回消息历史
            resp = await self._traced_llm_call(name, wllm, [sys] + history)
            out = [resp]
            for tc in getattr(resp, "tool_calls", []) or []:
                out.append(await dispatch_tool(tc, tools))
            final = await self._traced_llm_call(name, wllm, [sys] + history + out)
            out.append(final)
            return {"messages": out, "blackboard": {key: _first_text(final.content)}}

        return worker

    def _route(self, state: TeamState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return last.tool_calls[0]["name"]  # transfer_to_XXX
        return END

    # ---------- 图构建 ----------
    def _build(self):
        g = StateGraph(TeamState)
        g.add_node("supervisor", self._supervisor)
        for name in self.members:
            g.add_node(name, self.workers[name])
        g.add_edge(START, "supervisor")
        mapping = {f"transfer_to_{m}": m for m in self.members}
        mapping[END] = END
        g.add_conditional_edges("supervisor", self._route, mapping)
        for name in self.members:
            g.add_edge(name, "supervisor")
        return g.compile()

    async def run(self, query: str, session_id: str | None = None) -> str:
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(query)]},
            {"recursion_limit": self.max_iterations + 2},
        )
        answer = None
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                answer = m.content
                break
        if answer is None:
            answer = result["messages"][-1].content

        # 记忆链路：运行结束后把「黑板结论 + 最终答案」沉淀到长期记忆（跨会话复用）。
        # 仅在显式传入 session_id 时触发，未传入则保持原行为（不影响既有测试/调用）。
        if session_id:
            try:
                from app.memory.manager import get_memory_manager

                bb = result.get("blackboard") or {}
                facts = [
                    {"category": "blackboard", "content": f"{k}: {v}", "tags": [k]}
                    for k, v in bb.items()
                ]
                mm = get_memory_manager()
                await mm.record_run(
                    session_id, query, answer, facts=facts, title=query[:40]
                )
            except Exception as e:  # 记忆持久化失败不应中断主流程
                logger.warning("记忆持久化失败（已忽略）: %s", e)

        return answer

    async def stream(self, query: str, session_id: str | None = None):
        """异步流式：逐节点/逐工具产出细粒度事件，并随专家工作同步暴露共享黑板。

        事件类型：
        - delegate    : 主管委派给某专家（to）
        - action      : 某专家调用工具（agent, tool_calls）
        - observation : 工具返回结果（agent, tool, output）
        - worker      : 某专家产出的结论（agent, content）
        - blackboard  : 黑板更新（agent, updates 增量, full 全局）
        - answer      : 主管给出的最终回答（content）
        """
        full_bb: dict = {}
        full_chain: list = []
        final_answer = ""
        async for chunk in self._graph.astream(
            {"messages": [HumanMessage(query)]},
            {"recursion_limit": self.max_iterations + 2},
        ):
            for node, update in chunk.items():
                if node == "supervisor":
                    msg = update["messages"][-1]
                    if getattr(msg, "tool_calls", None):
                        tc = msg.tool_calls[0]
                        tgt = tc["name"].replace("transfer_to_", "")
                        payload = (tc.get("args") or {}).get("payload")
                        # 主管在委派时写入的 inbox[tgt.lower()] 即本次直传给该专家的上游产物
                        incoming = (update.get("inbox") or {}).get(tgt.lower()) or {}
                        inc_keys = [
                            k for k in incoming if k not in ("_payload_", "_prev_member_", "_prev_product_")
                        ]
                        # 主链路直传：上一专家
                        prev = incoming.get("_prev_member_")
                        yield {
                            "type": "delegate",
                            "to": tgt,
                            "incoming": inc_keys,
                            "payload": payload,
                            "prev": prev,
                        }
                        # 有序链式步骤：A -> B -> C 的成品一路带下来
                        step = (update.get("chain") or [])[-1] if update.get("chain") else None
                        if step:
                            full_chain.append(step)
                            yield {
                                "type": "chain",
                                "steps": list(full_chain),
                                "from": step.get("from"),
                                "to": step.get("to"),
                                "prev": prev,
                            }
                    else:
                        final_answer = _first_text(msg.content)
                        yield {"type": "answer", "content": final_answer}
                else:
                    for m in update.get("messages", []):
                        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            tcs = [dict(tc) for tc in m.tool_calls]
                            yield {"type": "action", "agent": node, "tool_calls": tcs}
                        elif isinstance(m, ToolMessage):
                            yield {
                                "type": "observation",
                                "agent": node,
                                "tool": getattr(m, "name", ""),
                                "output": _first_text(m.content),
                            }
                        elif isinstance(m, AIMessage):
                            yield {"type": "worker", "agent": node, "content": _first_text(m.content)}
                    bb_upd = update.get("blackboard") or {}
                    if bb_upd:
                        full_bb.update(bb_upd)
                        yield {
                            "type": "blackboard",
                            "agent": node,
                            "updates": bb_upd,
                            "full": dict(full_bb),
                        }
        if session_id and final_answer:
            try:
                from app.memory.manager import get_memory_manager
                facts = [
                    {"category": "blackboard", "content": f"{k}: {v}", "tags": [k],
                     "memory_key": f"blackboard.{k}"}
                    for k, v in full_bb.items()
                ]
                await get_memory_manager().record_run(
                    session_id, query, final_answer, facts=facts, title=query[:40]
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("流式多 Agent 记忆持久化失败（已忽略）: %s", e)
        yield {"type": "done", "route": "multi_agent"}
