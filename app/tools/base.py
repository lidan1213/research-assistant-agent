"""工具基类、装饰器与注册表。

核心概念：
- `ToolResult`：工具统一返回结构（成功/失败、内容、错误信息）。
- `BaseTool`：所有工具的抽象基类；子类实现 `run`。
- `FunctionTool`：用普通 async 函数 + Pydantic 参数模型快速构建工具，
  自动从模型生成 OpenAI function-calling JSON Schema。
- `tool`：装饰器，把一个函数注册为工具。
- `ToolRegistry`：全局注册表，供 Agent 查询与调度。
"""
from __future__ import annotations

import abc
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    success: bool
    output: Any = ""
    error: str | None = None
    # 失败是否值得重试：参数类错误（JSON 解析/参数校验/类型不匹配）为 False——
    # 重试只会得到同样结果；网络/执行类异常为 True（重试可能成功）
    retryable: bool = True

    def to_message_content(self) -> str:
        """把结果序列化为可回写给 LLM 的文本。

        PromptGuard：工具输出属于未经信任的外部数据，统一包裹 <tool_data> 标记，
        与 system prompt 的安全准则呼应，让 LLM 把其中的任何指令视为数据而非命令。
        """
        if not self.success:
            return f"[工具执行失败] {self.error}"
        if isinstance(self.output, (dict, list)):
            body = json.dumps(self.output, ensure_ascii=False, indent=2)
        else:
            body = str(self.output)
        return f"<tool_data>\n{body}\n</tool_data>"


class BaseTool(abc.ABC):
    name: str = ""
    description: str = ""

    @abc.abstractmethod
    async def run(self, **kwargs) -> ToolResult | Any:
        """执行工具逻辑。返回 ToolResult 或任意可被序列化的对象。"""

    # ---- 以下方法一般无需重写 ----

    def parameters_schema(self) -> dict:
        """返回参数的 JSON Schema（空对象表示无参）。子类可重写。"""
        return {"type": "object", "properties": {}, "required": []}

    def to_function_schema(self) -> dict:
        """转换为 OpenAI / 通用 function-calling 结构。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema(),
        }

    async def execute(self, arguments_json: str) -> ToolResult:
        """Agent 调用入口：解析 JSON 参数 -> 调用 run -> 归一化为 ToolResult。

        参数类错误（JSON 解析失败 / 参数校验失败 / 类型不匹配）标记 retryable=False，
        因为重试只会得到相同结果，避免 Agent 层白费重试次数。
        """
        try:
            args = json.loads(arguments_json or "{}")
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"参数不是合法 JSON: {e}", retryable=False)

        try:
            result = await self.run(**args)
        except TypeError as e:
            return ToolResult(success=False, error=f"参数不匹配: {e}", retryable=False)
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"执行异常: {e}")

        if isinstance(result, ToolResult):
            return result
        return ToolResult(success=True, output=result)


def _model_to_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    # 去掉 pydantic 注入的附加字段，保留 OpenAI 兼容结构
    schema.pop("$defs", None)
    schema.pop("title", None)
    return schema


class FunctionTool(BaseTool):
    """基于函数 + Pydantic 参数模型构建的工具。"""

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        name: str,
        description: str,
        params_model: type[BaseModel] | None = None,
    ):
        self._func = func
        self.name = name
        self.description = description
        self._params_model = params_model
        # 用函数签名补全描述
        if not description:
            self.description = (func.__doc__ or name).strip().split("\n")[0]

    def parameters_schema(self) -> dict:
        if self._params_model is None:
            return {"type": "object", "properties": {}, "required": []}
        return _model_to_schema(self._params_model)

    async def run(self, **kwargs) -> Any:
        if self._params_model is not None:
            try:
                validated = self._params_model(**kwargs)
            except ValidationError as e:
                return ToolResult(success=False, error=f"参数校验失败: {e}", retryable=False)
            return await self._func(**validated.model_dump())
        return await self._func(**kwargs)


def tool(
    name: str | None = None,
    description: str = "",
    params: type[BaseModel] | None = None,
    register: bool = True,
) -> Callable[[Callable[..., Awaitable[Any]]], FunctionTool]:
    """装饰器：将 async 函数注册为工具。

    用法：
        @tool(name="web_search", description="网络检索", params=SearchArgs)
        async def search(args: SearchArgs) -> str: ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> FunctionTool:
        tool_name = name or func.__name__
        ft = FunctionTool(func, tool_name, description, params)
        if register:
            registry.register(ft)
        return ft

    return decorator


class ToolRegistry:
    """全局工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, t: BaseTool) -> None:
        if not t.name:
            raise ValueError("工具必须设置 name")
        self._tools[t.name] = t

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.to_function_schema() for t in self._tools.values()]

    def subset(
        self, exclude: set[str] | None = None, include: set[str] | None = None
    ) -> "ToolRegistry":
        """返回工具子集的新注册表（浅拷贝工具引用，不影响全局注册表）。

        用途：按请求动态裁剪工具集，例如关闭「联网检索」时排除 web_search。
        """
        out = ToolRegistry()
        for name, t in self._tools.items():
            if exclude and name in exclude:
                continue
            if include and name not in include:
                continue
            out._tools[name] = t
        return out

    async def call(self, name: str, arguments_json: str) -> ToolResult:
        t = self.get(name)
        if t is None:
            return ToolResult(success=False, error=f"未找到工具: {name}")
        return await t.execute(arguments_json)


# 全局唯一注册表
registry = ToolRegistry()
