"""Base class for LLM services with streaming and non-streaming support."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

ThinkingLevel = Literal["none", "low", "med", "high"]

Role = Literal["user", "assistant", "tool", "system"]


class ToolCallFunction(TypedDict):
    """The function payload inside a tool call."""

    name: str
    arguments: str


class _ToolCallRequired(TypedDict):
    """Required fields on a tool call."""

    id: str
    function: ToolCallFunction


class ToolCall(_ToolCallRequired, total=False):
    """A tool call emitted by the assistant."""

    type: str  # typically "function"
    thought_signature: str | None
    extra_content: dict[str, Any]


class _MessageRequired(TypedDict):
    """Required fields present on every chat message."""

    role: Role
    content: str | list[Any]


class Message(_MessageRequired, total=False):
    """OpenAI-style chat message. All providers translate to/from this format.

    Required keys: ``role``, ``content``.
    Optional keys vary by role (e.g. ``tool_calls`` on assistant messages,
    ``tool_call_id`` on tool messages).
    """

    # Assistant message fields
    tool_calls: list[ToolCall]
    thought_summary: str
    thinking_signature: str
    reasoning_items: list[Any]
    # Tool message fields
    tool_call_id: str
    name: str


@dataclass
class StreamChunk:
    text_delta: str | None = None
    thought_delta: str | None = None
    thought_signature: str | None = None
    tool_calls_delta: list[dict] | None = None
    finish_reason: str | None = None
    reasoning_items: list[dict] | None = None
    usage_raw: dict[str, Any] | None = None


class BaseLLMService(ABC):
    def __init__(self, model_id: str, thinking_level: ThinkingLevel = "med"):
        self.model_id = model_id
        self.thinking_level = thinking_level

    @abstractmethod
    async def generate_with_tools_streaming(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        ...
        yield StreamChunk()  # pragma: no cover

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """Non-streaming generate. Optionally pass response_format for structured output.

        response_format example for JSON schema:
            {"type": "json_schema", "json_schema": {"name": "rooms", "strict": True, "schema": {...}}}
        """
        text = ""
        thought = ""
        tool_calls_raw: list[dict] = []
        usage_raw: dict[str, Any] | None = None

        async for chunk in self.generate_with_tools_streaming(
            system_prompt,
            messages,
            tools or [],
            response_format=response_format,
        ):
            if chunk.text_delta:
                text += chunk.text_delta
            if chunk.thought_delta:
                thought += chunk.thought_delta
            if chunk.tool_calls_delta:
                tool_calls_raw.extend(chunk.tool_calls_delta)
            if chunk.usage_raw:
                usage_raw = chunk.usage_raw

        return {
            "text": text,
            "thought": thought,
            "tool_calls": tool_calls_raw,
            "usage_raw": usage_raw,
        }

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000,
    ) -> str:
        messages: list[Message] = [{"role": "user", "content": prompt}]
        result = await self.generate(system, messages)
        return result["text"]
