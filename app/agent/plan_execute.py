"""先规划后执行（Plan-and-Execute）Agent。

流程：
    Planner 生成计划 → 按步骤逐条执行（每步一个独立小循环，可调工具）→ 汇总各步结果生成最终答案

与 ReAct 单 Agent 的区别：
    ReAct 是「自由推理循环」：思考→行动→观察 直到给出答案；
    Plan-and-Execute 先由 Planner 显式拆解步骤，再按步骤推进，每步独立上下文，
    适合多步骤、长流程任务（文献综述、方案对比、实验设计），不易跑偏、上下文可控。

流式事件：
    plan → (step_start → thought? → action/observation* → step_done)* → answer → done
"""
from __future__ import annotations

from typing import AsyncIterator

from app.agent.events import AgentEvent
from app.agent.planner import Planner
from app.config import get_settings
from app.core.logging import get_logger
from app.llm.base import ChatMessage, MessageRole, ToolCall
from app.skills.context import append_active_skill
from app.tools.base import ToolRegistry

logger = get_logger("plan_execute")

# 每步执行的系统提示：专注当前步骤，信息足够即给出该步结论
STEP_SYSTEM = """你是一位严谨的科研助手 Agent，当前处于「按计划分步执行」阶段。

整体研究目标：{goal}
执行计划：
{plan}

当前正在执行第 {index} 步（共 {total} 步）：{step}
{prev_results}

请只专注于完成当前这一步：
- 需要查资料/计算/读文档时，调用对应工具获取事实；
- 当这一步的信息已经足够时，直接给出这一步的结论（不要再调用工具），结论精炼、标注依据；
- 不要提前处理后面的步骤，也不要重复已完成步骤的工作。
"""

# 汇总系统提示：把所有步骤结果整合为最终回答
SUMMARIZE_SYSTEM = """你是一位严谨的科研助手 Agent。以下是按计划分步执行收集到的全部材料，
请综合整理成一份完整、结构化的最终回答，直接回应用户的目标：{goal}

执行计划：
{plan}

各步骤结果：
{steps_output}

要求：结构清晰、要点化，结论标注来源（论文/网页），不要遗漏关键信息。
"""


class PlanExecuteAgent:
    def __init__(
        self,
        llm,
        planner: Planner | None = None,
        tools: ToolRegistry | None = None,
        max_step_iters: int = 3,
        max_tool_calls: int | None = None,
    ) -> None:
        s = get_settings().agent
        self.llm = llm
        self.planner = planner or Planner(llm)
        self.tools = tools
        self.max_step_iters = max_step_iters
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else s.max_tool_calls
        self.max_obs_chars = s.max_obs_chars
        self._tool_count = 0
        self.recovery_budget = getattr(s, "recovery_budget", 2)
        # 统一工具执行运行时（Tool Runtime）：与 ResearchAgent 共用同一套护栏
        from app.tools.executor import ToolExecutor

        self.tool_executor = ToolExecutor(
            self.tools,
            timeout=getattr(s, "tool_timeout", 90) or 90,
            max_retries=getattr(s, "tool_max_retries", 1),
            max_chars=self.max_obs_chars or 8000,
        )
        from app.agent.loop_guard import TaskLoopGuard
        from app.agent.recovery import RecoveryPolicy

        self._task_loop_guard = TaskLoopGuard()
        self._recovery_policy = RecoveryPolicy(budget=self.recovery_budget)

    async def stream(self, goal: str) -> AsyncIterator[AgentEvent]:
        """执行「规划 → 逐步执行 → 汇总」全流程，逐事件产出。"""
        # ---------- 1. 规划 ----------
        steps = await self.planner.plan(goal)
        yield AgentEvent("plan", {"steps": steps})

        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        step_results: list[str] = []
        self._tool_count = 0  # 全局工具调用预算（跨步骤共享）
        # 任务级循环状态：所有步骤共享，步骤切换时不会清空。
        from app.agent.loop_guard import TaskLoopGuard
        from app.agent.recovery import RecoveryPolicy

        self._task_loop_guard = TaskLoopGuard()
        self._recovery_policy = RecoveryPolicy(budget=self.recovery_budget)

        async for event in self._stream_state(goal, steps, step_results, start_index=0):
            yield event

    async def resume(self, resume_token: str, user_input: str) -> AsyncIterator[AgentEvent]:
        """读取持久化暂停点，用用户补充信息从原步骤继续。"""
        from app.agent.loop_guard import TaskLoopGuard
        from app.agent.pending_tasks import get_pending_task_store
        from app.agent.recovery import RecoveryPolicy

        state = get_pending_task_store().pop(resume_token)
        if state is None:
            yield AgentEvent("error", {
                "reason": "invalid_resume_token",
                "message": "恢复令牌不存在或已过期，请重新发起任务。",
            })
            return

        self._tool_count = int(state.get("tool_count", 0))
        self._task_loop_guard = TaskLoopGuard(
            seen_results=set(state.get("seen_results", [])),
            blocked_tools=dict(state.get("blocked_tools", {})),
        )
        failed_tool = state.get("failed_tool", "")
        # 用户提供了新信息，允许原失败工具基于新上下文重新尝试一次。
        if failed_tool:
            self._task_loop_guard.blocked_tools.pop(failed_tool, None)
        self._recovery_policy = RecoveryPolicy(
            budget=int(state.get("recovery_budget", self.recovery_budget)),
            attempted=set(state.get("recovery_attempted", [])),
        )
        retained = state.get("evidence", [])
        supplement = user_input.strip()
        if retained:
            supplement += "\n\n暂停前已获得的证据：\n" + "\n".join(str(x) for x in retained[-4:])
        yield AgentEvent("resumed", {
            "resume_token": resume_token,
            "step_index": int(state["step_index"]) + 1,
        })
        async for event in self._stream_state(
            state["goal"],
            list(state["steps"]),
            list(state.get("step_results", [])),
            start_index=int(state["step_index"]),
            supplemental_input=supplement,
        ):
            yield event

    async def _stream_state(
        self,
        goal: str,
        steps: list[str],
        step_results: list[str],
        *,
        start_index: int,
        supplemental_input: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """执行或恢复既有计划；start_index 之前的结果不会重新计算。"""
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))

        # ---------- 2. 逐步执行 ----------
        for zero_idx in range(start_index, len(steps)):
            idx = zero_idx + 1
            step = steps[zero_idx]
            execution_step = step
            if zero_idx == start_index and supplemental_input:
                execution_step += f"\n\n用户补充信息：\n{supplemental_input}"
            yield AgentEvent(
                "step_start",
                {"index": idx, "step": step, "total": len(steps)},
            )
            result = ""
            async for item in self._execute_step(
                goal=goal,
                plan_text=plan_text,
                steps=steps,
                idx=idx,
                step=execution_step,
                prev_results=step_results,
            ):
                kind, payload = item
                if kind == "result":
                    result = payload
                elif kind == "needs_user_input":
                    from app.agent.pending_tasks import get_pending_task_store

                    pending_state = {
                        "goal": goal,
                        "steps": steps,
                        "step_results": step_results,
                        "step_index": zero_idx,
                        "tool_count": self._tool_count,
                        "seen_results": list(self._task_loop_guard.seen_results),
                        "blocked_tools": self._task_loop_guard.blocked_tools,
                        "recovery_budget": self._recovery_policy.budget,
                        "recovery_attempted": list(self._recovery_policy.attempted),
                        "failed_tool": payload.get("failed_tool", ""),
                        "evidence": payload.get("evidence", []),
                    }
                    token = get_pending_task_store().save(pending_state)
                    data = {**payload, "resume_token": token, "resume_point": f"step_{idx}"}
                    yield AgentEvent("needs_user_input", data)
                    yield AgentEvent("paused", {"resume_token": token, "step_index": idx})
                    return
                else:
                    yield AgentEvent(kind, payload)

            step_results.append(result)
            yield AgentEvent("step_done", {"index": idx, "step": step, "result": result})

        # ---------- 3. 汇总 ----------
        answer = await self._summarize(goal, plan_text, steps, step_results)
        yield AgentEvent("answer", {"content": answer})
        yield AgentEvent("done", {"steps": len(step_results), "tool_calls": self._tool_count})

    async def _execute_step(
        self,
        goal: str,
        plan_text: str,
        steps: list[str],
        idx: int,
        step: str,
        prev_results: list[str],
    ) -> AsyncIterator[tuple[str, dict]]:
        """执行单个步骤：小循环（LLM 带工具 → 执行 → 回写），直到 LLM 给出该步结论。

        产出事件元组 ("thought"|"action"|"observation", payload)，
        最后产出 ("result", 该步结论字符串)。
        """
        prev = (
            "\n".join(f"步骤 {i + 1} 结果：{r}" for i, r in enumerate(prev_results))
            or "（尚无已完成步骤）"
        )
        system = STEP_SYSTEM.format(
            goal=goal, plan=plan_text, index=idx, total=len(steps), step=step, prev_results=prev
        )
        system = append_active_skill(system)
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system)
        ]
        tool_schemas = self.tools.schemas() if self.tools else None
        done = False
        result = ""
        evidence: list[str] = []

        for _ in range(self.max_step_iters):
            if self.max_tool_calls and self._tool_count >= self.max_tool_calls:
                break  # 全局工具预算耗尽

            resp = await self.llm.chat(messages, tools=tool_schemas)

            # 无工具调用 → 该步结论
            if not resp.tool_calls:
                result = resp.content.strip()
                done = True
                break

            if resp.content:
                yield ("thought", {"content": resp.content})

            # 执行本批工具调用（受全局预算约束）
            remaining = (
                self.max_tool_calls - self._tool_count if self.max_tool_calls else len(resp.tool_calls)
            )
            batch_calls = resp.tool_calls[: max(0, remaining)]
            for tc in batch_calls:
                blocked, reason = self._task_loop_guard.is_blocked(tc.name)
                if blocked:
                    yield (
                        "observation",
                        {
                            "tool": tc.name,
                            "output": f"任务级循环保护已阻止重复调用：{reason}",
                            "truncated": False,
                            "blocked": True,
                        },
                    )
                    recovery = await self._recover_search(step, tc.name)
                    if recovery:
                        yield ("recovery", {
                            k: recovery[k] for k in (
                                "from_tool", "tool", "arguments", "reason", "remaining_budget"
                            )
                        })
                        yield ("action", {
                            "tool": recovery["tool"],
                            "arguments": recovery["arguments"],
                            "recovery": True,
                        })
                        yield ("observation", {
                            "tool": recovery["tool"],
                            "output": recovery["output"],
                            "truncated": False,
                            "recovery": True,
                        })
                        messages.extend(recovery["messages"])
                        evidence.append(recovery["output"])
                    else:
                        from app.agent.user_input import build_user_input_request

                        request = build_user_input_request(
                            goal=goal,
                            step=step,
                            failed_tool=tc.name,
                            available_tools={s.get("name", "") for s in (tool_schemas or [])},
                        )
                        yield ("needs_user_input", {
                            **request,
                            "failed_tool": tc.name,
                            "evidence": evidence,
                        })
                        return
                    # 已切换新证据源，或所有恢复路径耗尽；都进入有界收尾。
                    break
                self._tool_count += 1
                yield ("action", {"tool": tc.name, "arguments": tc.arguments})
                # 统一 ToolExecutor：超时/重试/连续失败/重复调用/截断/追踪
                content = await self.tool_executor.execute(tc.name, tc.arguments)
                truncated = (
                    content[: self.max_obs_chars]
                    + f"\n…[内容过长已截断，原文 {len(content)} 字符]"
                    if self.max_obs_chars and len(content) > self.max_obs_chars
                    else content
                )
                yield (
                    "observation",
                    {"tool": tc.name, "output": truncated, "truncated": truncated != content},
                )
                has_progress = self._task_loop_guard.record(tc.name, truncated)
                evidence.append(truncated)
                # 回写历史（assistant tool_calls 与 tool 消息配对，保证 API 校验通过）
                messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=resp.content or "",
                        tool_calls=[tc],
                    )
                )
                if not has_progress:
                    yield (
                        "observation",
                        {
                            "tool": tc.name,
                            "output": "检测到该工具跨步骤返回了相同结果，已加入本任务阻塞列表。",
                            "truncated": False,
                            "blocked": True,
                        },
                    )
                    recovery = await self._recover_search(step, tc.name)
                    if recovery:
                        yield ("recovery", {
                            k: recovery[k] for k in (
                                "from_tool", "tool", "arguments", "reason", "remaining_budget"
                            )
                        })
                        yield ("action", {
                            "tool": recovery["tool"],
                            "arguments": recovery["arguments"],
                            "recovery": True,
                        })
                        yield ("observation", {
                            "tool": recovery["tool"],
                            "output": recovery["output"],
                            "truncated": False,
                            "recovery": True,
                        })
                        messages.extend(recovery["messages"])
                        evidence.append(recovery["output"])
                    else:
                        from app.agent.user_input import build_user_input_request

                        request = build_user_input_request(
                            goal=goal,
                            step=step,
                            failed_tool=tc.name,
                            available_tools={s.get("name", "") for s in (tool_schemas or [])},
                        )
                        yield ("needs_user_input", {
                            **request,
                            "failed_tool": tc.name,
                            "evidence": evidence,
                        })
                        return
                    break

            # 一旦当前批次触发任务级阻塞，不再让模型在本步骤继续换参数空转。
            if any(self._task_loop_guard.is_blocked(tc.name)[0] for tc in batch_calls):
                break
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=truncated,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )

        # 该步未收敛（迭代上限 / 预算耗尽）：强制让 LLM 基于已有信息给出该步结论
        if not done:
            try:
                forced = await self.llm.chat(
                    messages
                    + [
                        ChatMessage(
                            role=MessageRole.USER,
                            content="请基于当前已有信息，直接给出这一步的结论，不要再调用工具。",
                        )
                    ],
                    tools=None,
                )
                result = forced.content.strip() or "（该步骤未能产出结论）"
            except Exception as e:  # noqa: BLE001
                logger.warning("步骤 %d 强制收尾失败: %s", idx, e)
                result = "（该步骤执行异常，已跳过）"

        yield ("result", result)

    async def _recover_search(self, query: str, failed_tool: str) -> dict | None:
        """切换到尚未尝试的检索源，并把新证据回写给当前步骤。"""
        schemas = self.tools.schemas() if self.tools else []
        available = {s.get("name", "") for s in schemas}
        blocked = set(self._task_loop_guard.blocked_tools)
        decision = self._recovery_policy.next_search(
            failed_tool=failed_tool,
            query=query,
            available_tools=available,
            blocked_tools=blocked,
        )
        if decision is None:
            return None
        if self.max_tool_calls and self._tool_count >= self.max_tool_calls:
            return None

        self._tool_count += 1
        output = await self.tool_executor.execute(decision.tool, decision.arguments)
        self._task_loop_guard.record(decision.tool, output)
        tc = ToolCall(
            id=f"recovery_{self._tool_count}",
            name=decision.tool,
            arguments=decision.arguments,
        )
        return {
            "from_tool": failed_tool,
            "tool": decision.tool,
            "arguments": decision.arguments,
            "reason": decision.reason,
            "output": output,
            "remaining_budget": self._recovery_policy.budget,
            "messages": [
                ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[tc]),
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=output,
                    tool_call_id=tc.id,
                    name=tc.name,
                ),
            ],
        }

    async def _summarize(
        self, goal: str, plan_text: str, steps: list[str], step_results: list[str]
    ) -> str:
        """汇总各步骤结果，生成最终回答。"""
        steps_output = "\n\n".join(
            f"【步骤 {i + 1}】{s}\n结果：{r}" for i, (s, r) in enumerate(zip(steps, step_results))
        )
        try:
            resp = await self.llm.chat(
                [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=append_active_skill(
                            SUMMARIZE_SYSTEM.format(
                                goal=goal, plan=plan_text, steps_output=steps_output
                            )
                        ),
                    )
                ],
                tools=None,
            )
            return resp.content.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("汇总失败: %s", e)
            return "（汇总失败）各步骤结果如下：\n" + steps_output[:800]
