"""Base classes for agent tools."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

TParams = TypeVar("TParams")
TResult = TypeVar("TResult")

ToolSchema = dict[str, Any]


def make_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> ToolSchema:
    schema: ToolSchema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }
    return schema


class ToolResult:
    def __init__(
        self,
        output: Any = None,
        error: str | None = None,
        tool_call_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
    ):
        self.output = output
        self.error = error
        self.tool_call_id = tool_call_id
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self.content_blocks = content_blocks

    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.error:
            result["error"] = self.error
        else:
            result["result"] = self.output
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseToolInvocation(ABC, Generic[TParams, TResult]):
    def __init__(self, params: TParams):
        self.params = params

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    async def execute(self) -> ToolResult:
        pass


class BaseDeclarativeTool(ABC):
    def __init__(self, name: str, schema: ToolSchema):
        self.name = name
        self.schema = schema

    @abstractmethod
    async def build(self, params: dict[str, Any]) -> BaseToolInvocation:
        pass


class ToolRegistry:
    def __init__(self, tools: list[BaseDeclarativeTool]):
        self.tools = {tool.name: tool for tool in tools}

    def get_tool(self, name: str) -> BaseDeclarativeTool | None:
        return self.tools.get(name)

    def get_tool_schemas(self) -> list[ToolSchema]:
        return [tool.schema for tool in self.tools.values()]

    async def build_invocation(
        self, name: str, params: dict[str, Any]
    ) -> BaseToolInvocation | None:
        tool = self.get_tool(name)
        if not tool:
            return None
        return await tool.build(params)
