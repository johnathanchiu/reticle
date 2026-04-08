"""AgentLoop — base agentic loop with inbox and hooks.

Provides the core loop: drain inbox -> LLM call -> tool exec -> repeat.
If a message arrives during tool execution, the turn restarts so the
agent sees the new context before its next LLM call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from reticle.agent.events import (
    AgentEvent,
    CompleteEvent,
    EventType,
    LLMUsageEvent,
    TerminateReason,
    TextDeltaEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnStartEvent,
)
from reticle.llm.base import BaseLLMService, Message, ToolCall, ToolCallFunction
from reticle.agent.tools.base import BaseDeclarativeTool, ToolRegistry, ToolResult


class AgentSnapshot(TypedDict):
    """JSON-serializable snapshot of agent state for checkpointing."""

    conversation: list[dict[str, Any]]
    turn_counter: int


__all__ = ["AgentEvent", "AgentLoop", "EventType", "Message", "TerminateReason"]

logger = logging.getLogger(__name__)


@dataclass
class StreamResponse:
    """Accumulated response from LLM streaming."""

    text: str = ""
    thought: str = ""
    thinking_signature: str | None = None
    reasoning_items: list | None = None
    usage_raw: dict | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ToolCallAccumulator:
    """Accumulates streaming tool call deltas into complete tool calls."""

    def __init__(self):
        self._calls: dict[int, dict[str, Any]] = {}

    def add_delta(self, delta: dict) -> None:
        idx = delta.get("index", 0)
        if idx not in self._calls:
            self._calls[idx] = {
                "id": delta.get("id"),
                "function": {"name": "", "arguments": ""},
            }

        call = self._calls[idx]
        if delta.get("function", {}).get("name"):
            call["function"]["name"] = delta["function"]["name"]
        if delta.get("function", {}).get("arguments"):
            call["function"]["arguments"] += delta["function"]["arguments"]
        if delta.get("id"):
            call["id"] = delta["id"]
        # Preserve thought_signature and extra_content (needed by Gemini)
        if delta.get("thought_signature"):
            call["thought_signature"] = delta["thought_signature"]
        if delta.get("extra_content"):
            call["extra_content"] = delta["extra_content"]

    def build(self) -> list[ToolCall]:
        result: list[ToolCall] = []
        for tc_data in self._calls.values():
            tool_call_id = tc_data.get("id") or f"call_{uuid.uuid4().hex}"
            tc = ToolCall(
                id=tool_call_id,
                function=ToolCallFunction(
                    name=tc_data.get("function", {}).get("name", ""),
                    arguments=tc_data.get("function", {}).get("arguments", ""),
                ),
                type="function",
            )
            if tc_data.get("thought_signature"):
                tc["thought_signature"] = tc_data["thought_signature"]
            if tc_data.get("extra_content"):
                tc["extra_content"] = tc_data["extra_content"]
            result.append(tc)
        return result


class AgentLoop:
    """Base agentic loop with inbox and hooks.

    Subclasses set ``self.conversation`` before calling ``_run_loop()``.
    The ``run_single_turn()`` method can also be called externally (e.g. by
    a tick scheduler) to drive the agent one LLM turn at a time.
    """

    def __init__(
        self,
        llm: BaseLLMService,
        tools: list[BaseDeclarativeTool],
        system_prompt: str,
        max_turns: int = 50,
        trace: bool = False,
    ):
        self.llm = llm
        self.tool_registry = ToolRegistry(tools)
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.conversation: list[Message] = []
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._turn_counter: int = 0
        self._last_response: StreamResponse | None = None
        self._trace_enabled = trace
        self._trace: list[dict] = []
        self._trace_start: float = 0.0
        self._current_turn: dict[str, Any] = {}

    async def inject(self, message: Message) -> None:
        """Push a message into this agent's inbox."""
        await self._inbox.put(message)

    async def _execute_tool(
        self, tool_call: ToolCall, parsed_args: dict[str, Any] | None = None
    ) -> tuple[ToolResult, Message]:
        """Execute a single tool call.

        If *parsed_args* is provided, skip re-parsing the JSON arguments.
        """
        func_name = tool_call["function"]["name"]
        tool_call_id = tool_call["id"]

        if parsed_args is not None:
            func_args = parsed_args
        else:
            try:
                func_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                result = ToolResult(
                    error=f"Invalid JSON in arguments: {e}",
                    tool_call_id=tool_call_id,
                )
                return result, self._tool_response_msg(tool_call_id, func_name, result)

        # Let subclasses modify params before execution
        func_args = await self.pre_tool_execute(tool_call, func_args)

        try:
            invocation = await self.tool_registry.build_invocation(func_name, func_args)
            if not invocation:
                result = ToolResult(
                    error=f"Tool {func_name} not found",
                    tool_call_id=tool_call_id,
                )
            else:
                result = await invocation.execute()
                result.tool_call_id = tool_call_id
        except Exception as e:
            result = ToolResult(
                error=f"Tool execution error: {e}",
                tool_call_id=tool_call_id,
            )

        return result, self._tool_response_msg(tool_call_id, func_name, result)

    @staticmethod
    def _tool_response_msg(tool_call_id: str, name: str, result: ToolResult) -> Message:
        if result.content_blocks is not None:
            # Rich content (e.g. images + text) — store as list for LLM services
            content: str | list[dict] = result.content_blocks
        else:
            content = json.dumps(
                {k: v for k, v in result.to_dict().items() if k != "tool_call_id"}
            )
        return Message(
            role="tool",
            content=content,  # type: ignore[typeddict-item]
            tool_call_id=tool_call_id,
            name=name,
        )

    # --- Single-turn entry point ---

    async def run_single_turn(self) -> AsyncGenerator[AgentEvent, None]:
        """Run one LLM turn: drain inbox -> stream -> exec tools -> yield events.

        This is the primary entry point for external callers (e.g. TickScheduler)
        that drive the agent one turn at a time.
        """
        self._drain_inbox()
        self._on_turn_start()
        self._turn_counter += 1

        # Initialize trace start on first turn
        if self._trace_enabled and self._trace_start == 0.0:
            self._trace_start = time.time()

        turn_start = TurnStartEvent(turn=self._turn_counter)
        self._record_event(turn_start)
        yield turn_start

        response = StreamResponse()
        async for event in self._stream_llm(response):
            self._record_event(event)
            yield event

        self.conversation.append(self._build_assistant_msg(response))
        self._last_response = response

        # Execute tool calls, breaking if inbox gets a new message
        executed = 0
        for tool_call in response.tool_calls:
            async for event in self._run_tool(tool_call):
                self._record_event(event)
                yield event
            executed += 1
            if not self._inbox.empty():
                break

        # Dummy results for skipped tool calls (OpenAI requires every
        # tool_call in the assistant message to have a matching result).
        self._add_skipped_tool_results(response.tool_calls[executed:])

        # Check termination
        should_stop, reason = self.should_terminate(response)
        if should_stop:
            complete = self._complete_event(reason or TerminateReason.GOAL, response.text)
            self._record_event(complete)
            yield complete

    # --- Core loop ---

    async def run(self) -> AsyncGenerator[AgentEvent, None]:
        """Run the full agentic loop until termination or max turns.

        Subclasses may override to add setup (e.g. seeding the conversation).
        The core iteration lives in ``_run_loop`` so subclasses that override
        ``run()`` can still delegate to the loop without recursion.
        """
        async for event in self._run_loop():
            yield event

    async def _run_loop(self) -> AsyncGenerator[AgentEvent, None]:
        """Core loop. Iterates ``run_single_turn`` up to ``max_turns``."""
        for _ in range(self.max_turns):
            async for event in self.run_single_turn():
                yield event
                if isinstance(event, CompleteEvent):
                    return

        yield self._complete_event(TerminateReason.MAX_TURNS, "Max turns reached")

    # --- Loop helpers ---

    def _drain_inbox(self) -> None:
        """Move all pending inbox messages into the conversation."""
        while not self._inbox.empty():
            self.conversation.append(self._inbox.get_nowait())

    async def _stream_llm(self, response: StreamResponse) -> AsyncGenerator[AgentEvent, None]:
        """Stream an LLM call, accumulating into *response* and yielding events."""
        accumulator = ToolCallAccumulator()
        tools = self.tool_registry.get_tool_schemas()

        async for chunk in self.llm.generate_with_tools_streaming(
            self.system_prompt, self.conversation, tools
        ):
            if chunk.thought_delta:
                response.thought += chunk.thought_delta
                yield ThoughtEvent(delta=chunk.thought_delta)
            if chunk.text_delta:
                response.text += chunk.text_delta
                yield TextDeltaEvent(delta=chunk.text_delta)
            if chunk.tool_calls_delta:
                for delta in chunk.tool_calls_delta:
                    accumulator.add_delta(delta)
            if chunk.thought_signature:
                response.thinking_signature = chunk.thought_signature
            if chunk.reasoning_items:
                response.reasoning_items = chunk.reasoning_items
            if chunk.usage_raw:
                response.usage_raw = chunk.usage_raw

        response.tool_calls = accumulator.build()

        # Emit usage event so the simulation can track costs
        if response.usage_raw:
            raw = response.usage_raw
            yield LLMUsageEvent(
                model=self.llm.model_id,
                input_tokens=raw.get("input_tokens", 0),
                output_tokens=raw.get("output_tokens", 0),
            )

    @staticmethod
    def _build_assistant_msg(response: StreamResponse) -> Message:
        """Build an OpenAI-format assistant message from a streamed response."""
        msg = Message(role="assistant", content=response.text)
        if response.thought:
            msg["thought_summary"] = response.thought
        if response.thinking_signature:
            msg["thinking_signature"] = response.thinking_signature
        if response.tool_calls:
            msg["tool_calls"] = response.tool_calls
        if response.reasoning_items:
            msg["reasoning_items"] = response.reasoning_items
        return msg

    async def _run_tool(self, tool_call: ToolCall) -> AsyncGenerator[AgentEvent, None]:
        """Execute a tool call, yielding TOOL_CALL and TOOL_RESULT events."""
        func = tool_call["function"]
        name = func["name"]
        try:
            args = json.loads(func["arguments"])
        except json.JSONDecodeError:
            args = {}

        tc_id = tool_call["id"]
        yield ToolCallEvent(id=tc_id, name=name, args=args)

        result, tool_response = await self._execute_tool(tool_call, parsed_args=args)
        self.conversation.append(tool_response)

        yield ToolResultEvent(
            tool_call_id=result.tool_call_id or "",
            output=result.output or "",
            error=result.error or "",
            metadata={
                k: float(v) for k, v in result.metadata.items() if isinstance(v, (int, float))
            },
            content_blocks=tuple(result.content_blocks) if result.content_blocks else (),
        )

    def _add_skipped_tool_results(self, skipped_calls: list[ToolCall]) -> None:
        for skipped in skipped_calls:
            tc_id = skipped["id"]
            name = skipped["function"]["name"]
            self.conversation.append(
                Message(
                    role="tool",
                    content=json.dumps({"error": "Skipped — interrupted by new message."}),
                    tool_call_id=tc_id,
                    name=name,
                )
            )

    @staticmethod
    def _complete_event(reason: TerminateReason, message: str = "Agent stopped") -> CompleteEvent:
        return CompleteEvent(
            success=reason == TerminateReason.GOAL,
            reason=reason,
            message=message,
        )

    # --- Trace ---

    def _record_event(self, event: AgentEvent) -> None:
        """Record an agent event into the structured trace. No-op when trace disabled."""
        if not self._trace_enabled:
            return
        ts = round(time.time() - self._trace_start, 2)

        if isinstance(event, TurnStartEvent):
            if self._current_turn:
                self._trace.append(self._current_turn)
            self._current_turn = {
                "turn": event.turn,
                "timestamp": ts,
                "thought": "",
                "text": "",
                "tool_calls": [],
                "tool_results": [],
                "usage": None,
            }
        elif isinstance(event, ThoughtEvent):
            self._current_turn["thought"] += event.delta
        elif isinstance(event, TextDeltaEvent):
            self._current_turn["text"] += event.delta
        elif isinstance(event, ToolCallEvent):
            self._current_turn["tool_calls"].append(
                {"id": event.id, "name": event.name, "args": event.args, "timestamp": ts}
            )
        elif isinstance(event, ToolResultEvent):
            entry: dict[str, Any] = {"tool_call_id": event.tool_call_id, "timestamp": ts}
            if event.error:
                entry["error"] = event.error
            else:
                output = str(event.output)
                entry["output"] = output
            if event.metadata:
                entry["metadata"] = event.metadata
            # Collect images and text from content_blocks for trace persistence
            if event.content_blocks:
                images = []
                texts = []
                for block in event.content_blocks:
                    if isinstance(block, dict) and block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            images.append(
                                {
                                    "media_type": source.get("media_type", "image/png"),
                                    "data": source["data"],
                                }
                            )
                    elif isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block["text"])
                if images:
                    entry["images"] = images
                if texts:
                    entry["feedback"] = "\n".join(texts)
            self._current_turn["tool_results"].append(entry)
        elif isinstance(event, LLMUsageEvent):
            self._current_turn["usage"] = {
                "model": event.model,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
            }
        elif isinstance(event, CompleteEvent):
            if self._current_turn:
                self._trace.append(self._current_turn)
                self._current_turn = {}
            self._trace.append(
                {
                    "event": "complete",
                    "success": event.success,
                    "reason": str(event.reason),
                    "message": event.message,
                    "timestamp": ts,
                }
            )

    # --- Hooks ---

    def _on_turn_start(self) -> None:
        """Hook called at the start of each turn. Override to refresh state."""

    def should_terminate(self, response: StreamResponse) -> tuple[bool, TerminateReason | None]:
        """Check whether the loop should stop. Override for domain-specific logic."""
        return False, None

    @property
    def trace(self) -> list[dict]:
        """Per-turn structured trace records. Only populated when trace=True."""
        return self._trace

    async def pre_tool_execute(
        self, tool_call: ToolCall, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Hook to modify params before execution."""
        return params

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            conversation=list(self.conversation),  # type: ignore[arg-type]
            turn_counter=self._turn_counter,
        )

    def restore(self, data: AgentSnapshot) -> None:
        self.conversation = list(data["conversation"])  # type: ignore[arg-type]
        self._turn_counter = data["turn_counter"]
